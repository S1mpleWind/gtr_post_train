"""
- turns of conversation
- Teacher Forcing
- focusing on the position of assistant-repleyed token, teacher logits(top-k)
"""

import os
import json
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


class ShareGPTDataset(Dataset):
    def __init__(self, data_dir, num_samples=None, seed=42):
        self.data = []
        jsonl_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jsonl")]

        for p in tqdm(jsonl_files, desc="加载数据文件"):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        x = json.loads(line)
                        if "conversation" in x and len(x["conversation"]) > 0:
                            self.data.append(x)
                    except Exception:
                        continue

        random.seed(seed)
        random.shuffle(self.data)
        if num_samples is not None and num_samples < len(self.data):
            self.data = self.data[:num_samples]

        print(f"可用对话数: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def chat_ids(tokenizer, messages, add_generation_prompt=False):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt
    )


# TODO: pair the conversation
def extract_turn_pairs(turns):
    """
    1) [{"human":"...", "assistant":"..."}]
    2) [{"human":"..."}, {"assistant":"..."}, ...]
    """
    pairs = []
    if not isinstance(turns, list):
        return pairs

    # already pairs
    has_both = any(isinstance(t, dict) and ("human" in t and "assistant" in t) for t in turns)
    if has_both:
        for t in turns:
            if not isinstance(t, dict):
                continue
            u = (t.get("human") or "").strip()
            a = (t.get("assistant") or "").strip()
            if u and a:
                pairs.append((u, a))
        return pairs

    #make pairs
    pending_user = None
    for t in turns:
        if not isinstance(t, dict):
            continue
        u = (t.get("human") or "").strip()
        a = (t.get("assistant") or "").strip()

        if u and not a:
            pending_user = u
        elif a and pending_user:
            pairs.append((pending_user, a))
            pending_user = None

    return pairs


# TODO    
def chat_ids(tokenizer, messages, add_generation_prompt=False):
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,      #  BatchEncoding
        return_tensors=None     #? Key: python list
    )

    # 兼容兜底
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]

    return ids


def process_conversations(model, tokenizer, dataset, args):
    """"""
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    saved_turns = 0
    saved_tokens = 0

    dbg_no_pairs = 0
    dbg_template_fail = 0
    dbg_empty_span = 0
    dbg_forward_fail = 0
    dbg_zero_token_positions = 0

    with open(args.output_path, "w", encoding="utf-8") as fout:
        pbar = tqdm(dataset, desc="处理对话", unit="dialog")

        for sample in pbar:
            conv_id = sample.get("conversation_id", "unknown")
            turns = sample.get("conversation", [])

            turn_pairs = extract_turn_pairs(turns)

            if len(turn_pairs) == 0:
                dbg_no_pairs += 1
                continue

            history_msgs = [{"role": "system", "content": args.system_prompt}] if args.system_prompt else []

            # global_cut_offset = 0
            # first_turn_computed_cut = False

            for turn_idx, (user_text, assistant_text) in enumerate(turn_pairs):
                context_msgs = history_msgs + [{"role": "user", "content": user_text}]
                full_msgs = context_msgs + [{"role": "assistant", "content": assistant_text}]

                try:
                    prompt_ids = chat_ids(tokenizer, 
                                          context_msgs, 
                                          add_generation_prompt=True)
                    full_ids = chat_ids(tokenizer, 
                                        full_msgs, 
                                        add_generation_prompt=False)
                except Exception:
                    dbg_template_fail += 1
                    continue

                ans_start = len(prompt_ids)
                if len(full_ids) <= ans_start:
                    history_msgs.extend([
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ])
                    raise ValueError(f"答案起点超出输入长度: conv={conv_id} turn={turn_idx+1} ans_start={ans_start} full_len={len(full_ids)}")
                    # continue


                # 局部计算窗口
                current_len = len(full_ids)
                if current_len > args.max_length:
                    cut_num = current_len - args.max_length
                    full_ids = full_ids[cut_num:]
                    
                    # 同步修正 assistant 的起始索引
                    ans_start = max(0, ans_start - cut_num)
                    
                    #如果 ans_start 被裁到了 0，说明这一轮已经没有完整的 Prompt
                    if ans_start == 0:
                        # print(f"[DEBUG] Skip turn because prompt is fully truncated.")
                        continue


                input_ids = torch.tensor(full_ids, dtype=torch.long, device=model.device).unsqueeze(0)
                attn = torch.ones_like(input_ids, device=model.device)

                try:
                    with torch.no_grad():
                        out = model(input_ids=input_ids, 
                                    attention_mask=attn, 
                                    use_cache=False, 
                                    return_dict=True)
                        logits = out.logits[0]  # [L, V]

                except Exception as e:
                    dbg_forward_fail += 1
                    print(f"[DEBUG][forward_fail] conv={conv_id} turn={turn_idx+1} err={repr(e)}")
                    continue

                teacher_topk_ids = []
                teacher_topk_vals = []
                target_token_ids = []
                token_positions = []

                for token_pos in range(ans_start, input_ids.shape[1]):
                    pred_pos = token_pos - 1
                    if pred_pos < 0:
                        continue

                    step_logits = logits[pred_pos]
                    topv, topi = torch.topk(step_logits, k=args.top_k, dim=-1)

                    teacher_topk_ids.append(topi.detach().cpu().tolist())
                    teacher_topk_vals.append(topv.detach().cpu().float().tolist())
                    target_token_ids.append(int(input_ids[0, token_pos].item()))
                    token_positions.append(int(token_pos))

                if len(token_positions) == 0:
                    history_msgs.extend([
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ])
                    continue
                
                rec = {
                    "conversation_id": conv_id,
                    "turn": turn_idx + 1,
                    "input_ids": full_ids,
                    "assistant_start": ans_start,
                    "token_positions": token_positions,
                    "target_token_ids": target_token_ids,
                    "teacher_topk_ids": teacher_topk_ids,
                    "teacher_topk_logits": teacher_topk_vals
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                saved_turns += 1
                saved_tokens += len(token_positions)

                history_msgs.extend([
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ])

            pbar.set_postfix({"turns": saved_turns, "tokens": saved_tokens})

    print(f"\n完成: turns={saved_turns}, assistant_tokens={saved_tokens}")
    print(
        f"[DEBUG_SUMMARY] no_pairs={dbg_no_pairs}, template_fail={dbg_template_fail}, "
        f"empty_span={dbg_empty_span}, forward_fail={dbg_forward_fail}, zero_token_positions={dbg_zero_token_positions}"
    )
    print(f"输出: {args.output_path}")

# ...existing code...

def main():
    import argparse

    parser = argparse.ArgumentParser("Prepare token-level KD data from local ShareGPT")
    parser.add_argument("--data_dir", 
                        type=str, 
                        default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/data_prepare/raw_data")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_path", type=str, default="/share/public/public_models/Llama-3.1-8B-Instruct")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--top_k", type=int, default=20)

    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--output_path",
                        type=str,
                        default = "/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/data_prepare/processed_data_llama.jsonl")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = ShareGPTDataset(args.data_dir, args.num_samples, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    ).eval()

    process_conversations(model, tokenizer, dataset, args)


if __name__ == "__main__":
    main()