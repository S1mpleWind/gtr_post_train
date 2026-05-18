import os
import json
import random
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_messages(question, answer, system_prompt=""):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": question.strip()})
    msgs.append({"role": "assistant", "content": answer.strip()})
    return msgs


def chat_ids(tokenizer, messages, add_generation_prompt=False):
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors=None,
    )
    if isinstance(ids, dict):
        ids = ids.get("input_ids", ids)
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]
    return ids


def process_jsonl(model, tokenizer, args):
    saved_samples = 0
    saved_tokens = 0

    num_layers = model.config.num_hidden_layers
    l_idx = 2
    m_idx = num_layers // 2
    u_idx = num_layers - 3

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.input_path, "r", encoding="utf-8") as fin, open(args.output_path, "w", encoding="utf-8") as fout:
        pbar = tqdm(fin, desc="processing jsonl", unit="sample")
        for i, line in enumerate(pbar):
            if args.max_samples > 0 and i >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue

            question = str(ex.get("question", "")).strip()
            answer = str(ex.get("generated_solution", "")).strip()
            if not question or not answer:
                continue

            # prompt-only 用于确定 assistant 起始位置
            prompt_msgs = []
            if args.system_prompt:
                prompt_msgs.append({"role": "system", "content": args.system_prompt})
            prompt_msgs.append({"role": "user", "content": question})

            full_msgs = build_messages(question, answer, args.system_prompt)

            try:
                prompt_ids = chat_ids(tokenizer, prompt_msgs, add_generation_prompt=True)
                full_ids = chat_ids(tokenizer, full_msgs, add_generation_prompt=False)
            except Exception:
                continue

            ans_start = len(prompt_ids)
            if len(full_ids) <= ans_start:
                continue

            # 左截断到 max_length，并同步位置
            if len(full_ids) > args.max_length:
                cut = len(full_ids) - args.max_length
                full_ids = full_ids[cut:]
                ans_start = max(0, ans_start - cut)
                if ans_start == 0:
                    continue

            input_ids = torch.tensor(full_ids, dtype=torch.long, device=model.device).unsqueeze(0)
            attn = torch.ones_like(input_ids, device=model.device)

            try:
                with torch.no_grad():
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        use_cache=False,
                        return_dict=True,
                        output_hidden_states=True,
                    )
                logits = out.logits[0]            # [L, V]
                hidden_states = out.hidden_states # tuple
            except Exception:
                continue

            teacher_topk_ids = []
            teacher_topk_vals = []
            target_token_ids = []
            token_positions = []
            token_hiddens = []
            token_3hiddens = []

            # 对 assistant token 做监督，预测位是 token_pos-1
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

                last_hidden = hidden_states[-1][0, pred_pos, :].detach().cpu().tolist()
                h_l = hidden_states[l_idx][0, pred_pos, :].detach().cpu()
                h_m = hidden_states[m_idx][0, pred_pos, :].detach().cpu()
                h_u = hidden_states[u_idx][0, pred_pos, :].detach().cpu()
                h_3 = torch.cat([h_l, h_m, h_u], dim=-1)

                token_hiddens.append(last_hidden)
                token_3hiddens.append(h_3.tolist())

            if len(token_positions) == 0:
                continue

            # 保留原始部分字段以便溯源（如存在）
            rec = {
                "conversation_id": ex.get("conversation_id", f"sample_{i}"),
                "turn": ex.get("turn", 1),
                "input_ids": full_ids,
                "assistant_start": ans_start,
                "token_positions": token_positions,
                "target_token_ids": target_token_ids,
                "teacher_topk_ids": teacher_topk_ids,
                "teacher_topk_logits": teacher_topk_vals,
                "token_hiddens": token_hiddens,
                "token_3hiddens": token_3hiddens,
                "question": question,
                "generated_solution": answer,
                "expected_answer": ex.get("expected_answer"),
                "predicted_answer": ex.get("predicted_answer"),
                "error_message": ex.get("error_message"),
                "is_correct": ex.get("is_correct"),
                "generation_type": ex.get("generation_type"),
                "dataset": ex.get("dataset"),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            saved_samples += 1
            saved_tokens += len(token_positions)
            pbar.set_postfix(samples=saved_samples, tokens=saved_tokens)

    print("done")
    print(f"saved samples: {saved_samples}")
    print(f"saved assistant tokens: {saved_tokens}")
    print(f"output: {args.output_path}")


def main():
    parser = argparse.ArgumentParser("JSONL -> SpecMoD 4hidden token-level JSONL (question + generated_solution)")
    parser.add_argument("--input_path", type=str, default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/data_prepare/raw_data/openmathinstruct/train.jsonl", help="输入 JSONL，含字段 question, generated_solution 等")
    parser.add_argument("--output_path", type=str, default="./processed_data_openmath_4hidden.jsonl", help="输出 JSONL 路径")
    parser.add_argument("--model_path", type=str, default="/share/public/public_models/Llama-3.1-8B-Instruct")
    parser.add_argument("--max_samples", type=int, default=12000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system_prompt", type=str, default="")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    ).eval()

    process_jsonl(model, tokenizer, args)


if __name__ == "__main__":
    main()