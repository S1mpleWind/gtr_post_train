import json
import numpy as np
from transformers import AutoTokenizer

data_path = "/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/data_prepare/processed_data.jsonl"
model_path = "/share/public/public_models/Qwen3-8B"

tok = AutoTokenizer.from_pretrained(model_path)

lens = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        lens.append(len(x["input_ids"]))

arr = np.array(lens)
print("count:", len(arr))
print("p50:", int(np.percentile(arr, 50)))
print("p90:", int(np.percentile(arr, 90)))
print("p95:", int(np.percentile(arr, 95)))
print("p99:", int(np.percentile(arr, 99)))
print("max:", int(arr.max()))
print(">2048:", int((arr > 2048).sum()))
print(">4096:", int((arr > 4096).sum()))