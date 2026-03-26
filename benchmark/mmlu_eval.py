#!/usr/bin/env python3
import os
import re
import sys
import time
import argparse
from collections import defaultdict

import torch
import jsonlines
import numpy as np
import datasets
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# 回答筛选
INVALID_ANS = "[invalid]"
CHOICES = ["A", "B", "C", "D"]

LETTER_PATTERNS = [
    re.compile(r"final\s*answer\s*(is|:)?\s*([ABCD])", re.IGNORECASE),
    re.compile(r"answer\s*(is|:)?\s*([ABCD])", re.IGNORECASE),
    re.compile(r"\b([ABCD])\b"),
    re.compile(r"答案\s*(是|:)?\s*([ABCD])"),
    re.compile(r"选项\s*([ABCD])"),
]

def dedupe_consecutive_lines(text):
    lines = [ln.rstrip() for ln in text.splitlines()]
    out = []
    prev = None
    for ln in lines:
        if ln == prev:
            continue
        out.append(ln)
        prev = ln
    return "\n".join(out).strip()

# 防止模型继续输出，进行截段
def clean_completion(raw_text, eos_token=None):
    text = raw_text
    if eos_token and eos_token in text:
        text = text.split(eos_token)[0]

    stop_markers = [
        "\n\nQuestion:",
        "\nQuestion:",
        "\n\nQ:",
        "\nQ:",
        "\n\n题目：",
        "\n题目：",
    ]
    cut_pos = len(text)
    for marker in stop_markers:
        p = text.find(marker)
        if p != -1:
            cut_pos = min(cut_pos, p)

    text = text[:cut_pos]
    text = dedupe_consecutive_lines(text)
    return text.strip()

def safe_decode_gen(tokens, tokenizer, eos_token):
    raw = tokenizer.decode(tokens, skip_special_tokens=False)
    return clean_completion(raw, eos_token=eos_token)

def normalize_gold_answer(ans):
    if ans is None:
        return INVALID_ANS

    if isinstance(ans, int):
        if 0 <= ans < 4:
            return CHOICES[ans]
        return INVALID_ANS

    s = str(ans).strip().upper()
    if s in CHOICES:
        return s
    if s.isdigit():
        idx = int(s)
        if 0 <= idx < 4:
            return CHOICES[idx]
    return INVALID_ANS

def extract_choice_letter(completion):
    text = completion.strip()
    if not text:
        return INVALID_ANS

    for pat in LETTER_PATTERNS:
        m = pat.search(text)
        if m:
            g = m.group(m.lastindex)
            if g:
                g = g.strip().upper()
                if g in CHOICES:
                    return g

    all_letters = re.findall(r"\b([ABCD])\b", text.upper())
    if all_letters:
        return all_letters[-1]

    return INVALID_ANS

def is_correct(pred, gold):
    return pred != INVALID_ANS and gold != INVALID_ANS and pred == gold

def build_messages(doc):
    q = doc.get("question", "")
    choices = doc.get("choices", [])
    if not isinstance(choices, list):
        choices = []
    if len(choices) < 4:
        choices = (choices + ["", "", "", ""])[:4]

    user_prompt = (
        "Please answer the following multiple-choice question.\n"
        "Output only one letter from A, B, C, D as the final answer.\n\n"
        f"Question: {q}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Answer:"
    )

    messages = [
        {
            "role": "system",
            "content": "You are a careful assistant for multiple-choice reasoning."
        },
        {"role": "user", "content": user_prompt},
    ]
    return messages

def load_eval_split(sample_input_file, dataset_name, subset, split):
    if sample_input_file is None:
        cfg = datasets.DownloadConfig(resume_download=True, max_retries=100)
        ds = load_dataset(dataset_name, subset, split=split, download_config=cfg)
        return ds

    sample_input_file = os.path.expanduser(sample_input_file)
    if os.path.isdir(sample_input_file):
        ds = load_from_disk(sample_input_file)
        if isinstance(ds, dict):
            if split in ds:
                return ds[split]
            if "test" in ds:
                return ds["test"]
        return ds

    if sample_input_file.endswith(".jsonl") or sample_input_file.endswith(".json"):
        with jsonlines.open(sample_input_file) as reader:
            return list(reader)

    ds = load_dataset("json", data_files={split: sample_input_file})
    return ds[split]

def main():
    parser = argparse.ArgumentParser(description="MMLU eval with chat template.")
    parser.add_argument("-c", "--checkpoint-path", type=str, default="/share/public/public_models/Qwen3-8B")
    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument("-o", "--sample-output-file", type=str, default="mmlu_res.jsonl")
    parser.add_argument("--dataset-name", type=str, default="cais/mmlu")
    parser.add_argument("--subset", type=str, default="all")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    test = load_eval_split(args.sample_input_file, args.dataset_name, args.subset, args.split)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)

    print("Loading model ...")
    torch_dtype = torch.float16 if device == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).eval()

    try:
        model.generation_config = GenerationConfig.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    except Exception:
        model.generation_config = GenerationConfig()

    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50

    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and eos_token_id is not None:
        pad_token_id = eos_token_id

    out_path = args.sample_output_file
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    f = open(out_path, "w", encoding="utf-8")
    writer = jsonlines.Writer(f)

    total = len(test) if isinstance(test, list) else getattr(test, "num_rows", None)
    acc_res = []
    sub_acc = defaultdict(list)

    iterator = test if isinstance(test, list) else test

    for i, doc in enumerate(iterator, start=1):
        try:
            messages = build_messages(doc)

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")

            gen_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "temperature": 1e-6,
                "top_p": 1.0,
                "top_k": 50,
                "eos_token_id": eos_token_id,
                "pad_token_id": pad_token_id,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask

            outputs = model.generate(**gen_kwargs)
            out_ids = outputs[0] if outputs.ndim == 2 else outputs
            gen_ids = out_ids[input_ids.shape[1]:]
            completion = safe_decode_gen(gen_ids, tokenizer, eos_token)

            pred = extract_choice_letter(completion)
            gold = normalize_gold_answer(doc.get("answer"))
            acc = is_correct(pred, gold)

            doc_out = dict(doc)
            doc_out["completion"] = completion
            doc_out["pred"] = pred
            doc_out["gold"] = gold
            doc_out["acc"] = acc
            writer.write(doc_out)
            f.flush()

            acc_res.append(acc)
            subject = doc.get("subject", "unknown")
            sub_acc[subject].append(acc)

            if args.debug:
                print("=== SAMPLE", i, "===")
                print("pred:", pred, "gold:", gold, "acc:", acc)
                print(completion)

            if i % 20 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] processed {i}/{total}")

            if i % 1200 == 0:
                break #? 1.4w 太多了

        except Exception as e:
            err = {
                "idx": i,
                "question": doc.get("question") if isinstance(doc, dict) else None,
                "error": repr(e),
            }
            writer.write(err)
            f.flush()
            print(f"Error at sample {i}: {e}", file=sys.stderr)

    writer.close()
    f.close()

    if acc_res:
        print("Overall Acc:", float(np.mean(acc_res)))
    else:
        print("Overall Acc: N/A (no valid samples)")

    if sub_acc:
        print("Per-subject Acc:")
        for k in sorted(sub_acc.keys()):
            print(k, float(np.mean(sub_acc[k])))

if __name__ == "__main__":
    main()
