from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download(
    "LLM-Research/Meta-Llama-3-8B-Instruct", cache_dir="./Models/"
)
print("Download finished")
