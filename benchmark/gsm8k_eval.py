#!/usr/bin/env python3
import os
import re
import sys
import time
import argparse
import torch
import jsonlines
import numpy as np
import datasets
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ANS_RE = re.compile(r"####\s*(-?[0-9][0-9,\.]*)")
FINAL_ANS_PATTERNS = [
    re.compile(r"the answer is\s*\$?\s*(-?[0-9][0-9,\.]*)", re.IGNORECASE),
    re.compile(r"answer:\s*\$?\s*(-?[0-9][0-9,\.]*)", re.IGNORECASE),
]
INVALID_ANS = "[invalid]"


def doc_to_text(doc, fewshot_prompt):
    # 加上 Answer: 作为收束提示，减少模型续写下一题
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\nAnswer: "
    )


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

# fixed:回答会多生成其他的问题，截取好像写的不对？
def clean_completion(raw_text, eos_token=None):
    text = raw_text

    # 1) 先按 eos 截断
    if eos_token and eos_token in text:
        text = text.split(eos_token)[0]

    # 2) 遇到下一题标记就截断
    stop_markers = [
        "\n\nQuestion:",
        "\nQuestion:",
        "\n\nQ:",
        "\nQ:",
        "\nLet's think step by step\nQuestion:",
    ]
    cut_pos = len(text)
    for marker in stop_markers:
        p = text.find(marker)
        if p != -1:
            cut_pos = min(cut_pos, p)
    text = text[:cut_pos]

    # 3) 去掉连续重复行
    text = dedupe_consecutive_lines(text)
    return text.strip()


def safe_decode_gen(tokens, tokenizer, eos_token):
    raw = tokenizer.decode(tokens, skip_special_tokens=False)
    # print(raw)
    return clean_completion(raw, eos_token=eos_token)


def parse_num(s):
    s = s.strip().replace(",", "")
    try:
        # 兼容整数/小数
        return float(s) if "." in s else int(s)
    except Exception:
        return INVALID_ANS


def extract_answer_hf(answer_text):
    m = ANS_RE.search(answer_text)
    if not m:
        return INVALID_ANS
    return parse_num(m.group(1))


def extract_answer(completion):
    # 优先从 "The answer is ..." 提取
    for pat in FINAL_ANS_PATTERNS:
        m = pat.search(completion)
        if m:
            val = parse_num(m.group(1))
            if val != INVALID_ANS:
                return val

    # 回退：取最后一个数字
    nums = re.findall(r"-?[0-9][0-9,\.]*", completion)
    if not nums:
        return INVALID_ANS
    return parse_num(nums[-1])


def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    pred = extract_answer(completion)
    return pred == gold


def load_test_split(sample_input_file):
    if sample_input_file is None:
        cfg = datasets.DownloadConfig(resume_download=True, max_retries=100)
        ds = load_dataset("gsm8k", "main", download_config=cfg)
        return ds["test"]

    sample_input_file = os.path.expanduser(sample_input_file)
    if os.path.isdir(sample_input_file):
        ds = load_from_disk(sample_input_file)
        return ds["test"]

    if sample_input_file.endswith(".jsonl") or sample_input_file.endswith(".json"):
        with jsonlines.open(sample_input_file) as reader:
            return list(reader)

    ds = load_dataset("json", data_files={"test": sample_input_file})
    return ds["test"]


def main():
    parser = argparse.ArgumentParser(description="GSM8K eval with robust decoding for Qwen.")
    parser.add_argument("-c", "--checkpoint-path", type=str, default="/share/public/public_models/Qwen3-8B")
    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument("-o", "--sample-output-file", type=str, default="gsm8k_res.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    fewshot_prompt = open("gsm8k_prompt.txt", "r", encoding="utf-8").read()
    test = load_test_split(args.sample_input_file)

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

    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    out_path = args.sample_output_file
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    f = open(out_path, "w", encoding="utf-8")
    writer = jsonlines.Writer(f)

    total = len(test) if isinstance(test, list) else getattr(test, "num_rows", None)
    acc_res = []

    for i, doc in enumerate(test, start=1):
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
                },
                {"role": "user", "content": doc["question"]},
            ]

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
                "eos_token_id": eos_token_id,
                "pad_token_id": pad_token_id,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask

            outputs = model.generate(**gen_kwargs)

            out_ids = outputs[0] if outputs.ndim == 2 else outputs
            gen_ids = out_ids[input_ids.shape[1]:]
            completion = safe_decode_gen(gen_ids, tokenizer, eos_token)

            print("completed")

            if args.debug:
                print("=== SAMPLE", i, "===")
                print(completion)

            answer = doc.get("answer")
            try:
                acc = is_correct(completion, answer) if answer is not None else False
            except AssertionError:
                acc = False

            doc_out = dict(doc)
            doc_out["completion"] = completion
            doc_out["acc"] = acc
            writer.write(doc_out)
            f.flush()

            acc_res.append(acc)

            if i % 10 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] processed {i}/{total}")
            
            # if i % 100 == 0:
            #     break

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
        print("Acc:", float(np.mean(acc_res)))
    else:
        print("Acc: N/A (no valid samples)")
    


if __name__ == "__main__":
    main()