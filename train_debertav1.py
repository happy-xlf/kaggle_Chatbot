#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :train_debertav1.py
# @Time      :2024/07/21 11:50:27
# @Author    :Lifeng
# @Description :
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.metrics import log_loss
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import os
import gc
import time
import random
import warnings

warnings.filterwarnings("ignore")

# ====================================================
# CFG
# ====================================================


class CFG:
    exp_no = 3
    wandb = False
    debug = True
    apex = True
    print_freq = 20
    num_workers = 4
    model_path = "/root/autodl-tmp/Code/kaggle_Chatbot/models/debeartav3_base"
    data_path = "/root/autodl-tmp/Code/kaggle_Chatbot/data/train_clean.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = "microsoft/deberta-v3-base"
    gradient_checkpointing = True
    scheduler = "cosine"  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 3
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 8
    max_len = 1536
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    target_cols = ["label"]
    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()
train = pd.read_csv(CFG.data_path)
tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
CFG.tokenizer = tokenizer

print("== prompt ==")
lengths = []
prompt_new = []
tk0 = tqdm(train["prompt"].fillna("").values, total=len(train))
for text in tk0:
    length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if length > 512:
        text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(text)[:256] + tokenizer.tokenize(text)[-256:])
    prompt_new.append(text)
    tk0.update(1)
tk0.close()

print("== response_a ==")
response_a_new = []
tk0 = tqdm(train["response_a"].fillna("").values, total=len(train))
for text in tk0:
    length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if length > 512:
        text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(text)[:256] + tokenizer.tokenize(text)[-256:])
    response_a_new.append(text)
    tk0.update(1)
tk0.close()

print("== response_b ==")
response_b_new = []
tk0 = tqdm(train["response_b"].fillna("").values, total=len(train))
for text in tk0:
    length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if length > 512:
        text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(text)[:256] + tokenizer.tokenize(text)[-256:])
    response_b_new.append(text)
    tk0.update(1)
tk0.close()

train["prompt_new"] = prompt_new
train["response_a_new"] = response_a_new
train["response_b_new"] = response_b_new

def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=cfg.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs

class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.prompts = df["prompt_new"].values
        self.response_a = df["response_a_new"].values
        self.response_b = df["response_b_new"].values
        self.labels = df[cfg.target_cols].values

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, item):
        prompt = self.prompts[item]
        response_a = self.response_a[item]
        response_b = self.response_b[item]
        text = prompt + self.cfg.tokenizer.sep_token + response_a + self.cfg.tokenizer.sep_token + response_b
        inputs = prepare_input(self.cfg, text)
        label = torch.LongTensor(self.labels[item])
        return inputs, label
    

def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = v[:, :mask_len]
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
            self.config = AutoConfig.from_pretrained(cfg.model_path, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model_path, config=self.config)
        else:
            self.model = AutoModel(self.config)
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

def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    step_sum = 0
    losses = []
    for step, (inputs, labels) in tqdm(enumerate(train_loader)):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
            #             print(y_preds, labels, F.softmax(y_preds, dim=1))
            loss = criterion(y_preds, labels.squeeze(1))
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        scaler.scale(loss).backward()
        losses.append(loss.item() * batch_size)
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        step_sum += batch_size
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if CFG.batch_scheduler:
                scheduler.step()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Loss: {3} '
                  'LR: {4}  '
                  .format(epoch + 1, step, len(train_loader),
                          sum(losses) / step_sum,
                          scheduler.get_lr()[0]))

    return sum(losses) / step_sum

def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    preds = []
    step_sum = 0
    losses = []

    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds,
                             labels.squeeze(1))  # 
            if CFG.gradient_accumulation_steps > 1:
                loss = loss / CFG.gradient_accumulation_steps
            y_preds = F.softmax(y_preds, dim=1)  
            preds.append(y_preds.detach().cpu().numpy())
            losses.append(loss.item() * batch_size)
            step_sum += batch_size
            if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
                print('EVAL: [{0}/{1}] '
                      'Loss: {2}'
                      .format(step, len(valid_loader),
                              sum(losses) / step_sum))
    predictions = np.concatenate(preds)

    return sum(losses) / step_sum, predictions

# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold):
    print(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values

    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)  #
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=4, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, './save/config.pth')
    model.to(CFG.device)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        # param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr,
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps,
                num_cycles=cfg.num_cycles
            )
        return scheduler

    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    #     criterion = nn.CrossEntropyLoss(reduction='mean') # RMSELoss(reduction="mean")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # RMSELoss(reduction="mean")

    best_score = np.inf

    for epoch in range(CFG.epochs):

        # start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, CFG.device)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, CFG.device)

        # scoring
        score = log_loss(valid_labels, predictions)

        print(f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}')
        print(f'Epoch {epoch + 1} - Score: {score:.4f}')

        if best_score > score:
            best_score = score
            print(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        f"./save/fold{fold}_best_exp{str(CFG.exp_no).zfill(3)}.pth")

    predictions = \
    torch.load( f"./save/fold{fold}_best_exp{str(CFG.exp_no).zfill(3)}.pth",
               map_location=torch.device('cpu'))['predictions']
    valid_folds[['pred_winner_model_a', 'pred_winner_model_b', 'pred_winner_tie']] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds


if __name__ == '__main__':
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            _oof_df = train_loop(train, fold)
            oof_df = pd.concat([oof_df, _oof_df])