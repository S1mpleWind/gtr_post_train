import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

class EvalDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                if "input_ids" in x and "token_positions" in x and "target_token_ids" in x:
                    self.data.append(x)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, pad_token_id: int, max_length: int):
    proc = []
    max_len = 0
    for item in batch:
        ids = item["input_ids"]
        pos = item["token_positions"]
        tgt = item["target_token_ids"]
        
        # 裁剪超长 input
        if len(ids) > max_length:
            cut = len(ids) - max_length
            ids = ids[cut:]
            new_pos, new_tgt = [], []
            for p, t in zip(pos, tgt):
                np = p - cut
                if 0 <= np < len(ids):
                    new_pos.append(np)
                    new_tgt.append(t)
            pos, tgt = new_pos, new_tgt
            
        max_len = max(max_len, len(ids))
        proc.append({
            "input_ids": ids,
            "token_positions": pos,
            "target_token_ids": tgt,
        })
        
    input_ids_out = []
    attention_mask_out = []
    for x in proc:
        ids = x["input_ids"]
        pad_len = max_len - len(ids)
        input_ids_out.append(ids + [pad_token_id] * pad_len)
        attention_mask_out.append([1] * len(ids) + [0] * pad_len)
        
    return {
        "input_ids": torch.tensor(input_ids_out, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_out, dtype=torch.long),
        "token_positions": [x["token_positions"] for x in proc],
        "target_token_ids": [x["target_token_ids"] for x in proc],
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/share/public/public_models/Qwen3-8B")
    parser.add_argument("--data_path", type=str, default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/data_prepare/processed_data_long.jsonl")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=10000, help="测试的最大样本数，-1 表示全部")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备 Tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    # 2. 准备数据
    print(f"Loading dataset from {args.data_path}...")
    dataset = EvalDataset(args.data_path)
    if args.max_samples > 0:
        n = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, list(range(n)))
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: collate_fn(b, pad_token_id, args.max_length)
    )

    # 3. 准备模型 (以 bf16 并在单卡上运行基础推理即可)
    print(f"Loading Base Model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    model.eval()

    # 4. 测试循环
    total_ce_loss = 0.0
    total_valid_tokens = 0

    progress_bar = tqdm(loader, desc="Evaluating Baseline CE")
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            positions_batch = batch["token_positions"]
            targets_batch = batch["target_token_ids"]
            
            B = input_ids.size(0)
            
            # Forward pass 获取基础模型的 logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()  # [B, Seq_Len, Vocab_Size]
            
            batch_tokens = 0
            batch_loss_sum = 0.0

            for b in range(B):
                valid_len = int(attention_mask[b].sum().item())
                pos_list = positions_batch[b]
                tgt_list = targets_batch[b]
                
                for p, tgt_id in zip(pos_list, tgt_list):
                    # 只有位于有效范围内的 token 参与计算 (且减1因为预测的是 next token)
                    if 0 < p < valid_len:
                        # 模型输入的 p-1 个 token 输出的 logit，用来预测第 p 个位置的 token
                        token_logit = logits[b, p - 1].unsqueeze(0) 
                        
                        # 【新增】和训练代码完全一致的 Clamp 操作
                        # token_logit = torch.nan_to_num(token_logit, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
                        
                        target = torch.tensor([tgt_id], device=device, dtype=torch.long)
                        
                        ce = F.cross_entropy(token_logit, target, reduction="sum")
                        
                        if torch.isfinite(ce):
                            batch_loss_sum += ce.item()
                            batch_tokens += 1
            
            total_ce_loss += batch_loss_sum
            total_valid_tokens += batch_tokens
            
            current_avg_ce = total_ce_loss / max(total_valid_tokens, 1)
            progress_bar.set_postfix({"Current Avg CE": f"{current_avg_ce:.4f}"})

    final_avg_ce = total_ce_loss / max(total_valid_tokens, 1)
    print("\n" + "="*50)
    print(f"Evaluation Completed!")
    print(f"Total Valid Target Tokens: {total_valid_tokens}")
    print(f"Baseline Average Validation CE Loss: {final_avg_ce:.6f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()