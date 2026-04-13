import json
import argparse
import numpy as np
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="评估处理后数据的长度分布")
    parser.add_argument("--data_path", type=str, 
                        default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/data_prepare/processed_data.jsonl",
                        help="JSONL 数据文件路径")  
    args = parser.parse_args()

    # tok = AutoTokenizer.from_pretrained(args.model_path)

    lens = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            lens.append(len(x["input_ids"]))

    if not lens:
        print("未找到数据。")
        return

    arr = np.array(lens)
    print(f"--- 统计报告: {args.data_path} ---")
    print("总条数:", len(arr))
    print("p50:", int(np.percentile(arr, 50)))
    print("p90:", int(np.percentile(arr, 90)))
    print("p95:", int(np.percentile(arr, 95)))
    print("p99:", int(np.percentile(arr, 99)))
    print("最大值:", int(arr.max()))
    print(">2048:", int((arr > 2048).sum()))
    print(">4096:", int((arr > 4096).sum()))

if __name__ == "__main__":
    main()