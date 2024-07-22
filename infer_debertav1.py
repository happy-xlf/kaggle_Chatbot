import os
import gc
import time
import random
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
DATA_DIR = Path('./data/')
test = pd.read_csv(DATA_DIR / 'test.csv')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()
class CFG:
    exp_no = 3
    wandb = False
    debug = True
    apex = True
    print_freq = 20
    num_workers = 4
    model_path = './save/'
    model = "/root/autodl-tmp/Code/kaggle_Chatbot/models/debeartav3_base"
    data_path = ''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = "microsoft/deberta-v3-base"
    gradient_checkpointing = True
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 3
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 16
    max_len = 1536
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    target_cols = ['labels']
    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
tokenizer = AutoTokenizer.from_pretrained(CFG.model, use_fast=False)
CFG.tokenizer = tokenizer

# prompt
print(f'== prompt ==')
lengths = []
prompt_new = []
tk0 = tqdm(test['prompt'].fillna("").values, total=len(test))
for text in tk0:
    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    if(length > 256): # 512
#         text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(text)[:256] + tokenizer.tokenize(text)[-256:]) # 最初と最後の256トークンを文章として扱う
        text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(text)[:128] + tokenizer.tokenize(text)[-128:]) # 最初と最後の128トークンを文章として扱う
    prompt_new.append(text)

# response_a
print(f'== response_a ==')
response_a_new = []
tk0 = tqdm(test['response_a'].fillna("").values, total=len(test))
for text in tk0:
    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    if(length > 768): # 512
#         text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(text)[:256] + tokenizer.tokenize(text)[-256:]) # 最初と最後の256トークンを文章として扱う
        text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(text)[:384] + tokenizer.tokenize(text)[-384:]) # 最初と最後の384トークンを文章として扱う
    response_a_new.append(text)

# response_b
print(f'== response_b ==')
response_b_new = []
tk0 = tqdm(test['response_b'].fillna("").values, total=len(test))
for text in tk0:
    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    if(length > 768): # 512
#         text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(text)[:256] + tokenizer.tokenize(text)[-256:]) # 最初と最後の256トークンを文章として扱う
        text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(text)[:384] + tokenizer.tokenize(text)[-384:]) # 最初と最後の256トークンを文章として扱う
    response_b_new.append(text)


test['prompt_new'] = prompt_new
test['response_a_new'] = response_a_new
test['response_b_new'] = response_b_new


def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=CFG.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.prompts = df['prompt'].values
        self.responses_a = df['response_a'].values
        self.responses_b = df['response_b'].values

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        prompt = self.prompts[item]
        response_a = self.responses_a[item]
        response_b = self.responses_b[item]
        text = prompt + self.cfg.tokenizer.sep_token + response_a + self.cfg.tokenizer.sep_token + response_b
        inputs = prepare_input(self.cfg, text)
        return inputs


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 3)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output


def inference_fn(valid_loader, model, device):
    model.eval()
    preds = []
    model.to(device)
    for step, inputs in tqdm(enumerate(valid_loader)):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
            # [batchsize,3]
            y_preds = F.softmax(y_preds, dim=1)  
            # 
            preds.append(y_preds.detach().cpu().numpy())
    predictions = np.concatenate(preds)

    return predictions

test_dataset = TestDataset(CFG, test)
# test_loader = DataLoader(test_dataset,
#                          batch_size=CFG.batch_size,
#                          shuffle=False,
#                          num_workers=0, pin_memory=True, drop_last=False) # 学習済みモデルが「num_worker=0(localでモデリング)」でやってるから、値を増やしてもデッドロックする
predictions = []
# n_gpus = torch.cuda.device_count()
# print(n_gpus)

for fold in [0,1]:
    try:
        test_loader = DataLoader(test_dataset,
                                 batch_size=CFG.batch_size,
                                 shuffle=False,
                                 num_workers=0, pin_memory=True, drop_last=False) # 学習済みモデルが「num_worker=0(localでモデリング)」でやってるから、値を増やしてもデッドロックする
    except :
        test_loader = DataLoader(test_dataset,
                                 batch_size=CFG.batch_size,
                                 shuffle=False,
                                 num_workers=4, pin_memory=True, drop_last=False) # 学習済みモデルが「num_worker=0(localでモデリング)」でやってるから、値を増やしてもデッドロックする
    model = CustomModel(CFG, config_path=CFG.model_path + 'config.pth', pretrained=False)
#     if n_gpus > 1:
#         model = torch.nn.DataParallel(model)
#         model.to(device)
    state = torch.load(CFG.model_path + f"fold{fold}_best_exp003.pth",
                       map_location=CFG.device)
    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, CFG.device)
    predictions.append(prediction)
    del model, state, prediction; gc.collect()
    torch.cuda.empty_cache()
    # [batch,3] 0 [bathc,3]1 ->[batch,3]
predictions = np.mean(predictions, axis=0)
test[['winner_model_a', 'winner_model_b', 'winner_tie']] = predictions

sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
sample_sub = sample_sub[['id']].merge(test[['id', 'winner_model_a', 'winner_model_b', 'winner_tie']], on='id', how='left')
sample_sub.to_csv('submission.csv', index=False)
sample_sub.head()

