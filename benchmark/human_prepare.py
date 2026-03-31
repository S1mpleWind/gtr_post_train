#!/usr/bin/env python3
import os
import re
import sys
import json
import time
import argparse
from typing import Dict, Any, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_markdown_fence(text: str) -> str:
    if "```" not in text:
        return text
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[0].strip("\n")
    return text.replace("```python", "").replace("```", "").strip()


def dedupe_consecutive_lines(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: List[str] = []
    prev = None
    for ln in lines:
        if ln == prev:
            continue
        out.append(ln)
        prev = ln
    return "\n".join(out).rstrip()


def cut_at_stop_markers(text: str) -> str:
    stop_markers = [
        "\n\nclass ",
        "\nclass ",
        "\n\ndef ",
        "\ndef ",
        "\nif __name__",
        "\n# Example",
        "\n# Test",
        "\nassert ",
        "\n```",
        "\n###",
        "</think>",
        "\nThe code",
        "\nThis function",
        "\nreturn the ",
    ]
    cut_pos = len(text)
    for marker in stop_markers:
        p = text.find(marker)
        if p != -1:
            cut_pos = min(cut_pos, p)
    return text[:cut_pos]


def strip_pollution_lines(text: str) -> str:
    bad_patterns = [
        r"^\s*```",
        r"^\s*###",
        r"^\s*<[/]?think>\s*$",
        r"^\s*This function\b",
        r"^\s*The code\b",
        r"^\s*return the\b",
        r"^\s*def the function continuation\b",
        r".*\*\*ARGS\*\*.*",
        r".*\*\*int\*\*.*",
        r".*\*\*str\*\*.*",
        r".*\*\*float\*\*.*",
    ]
    bad_re = re.compile("|".join(bad_patterns), re.IGNORECASE)
    kept: List[str] = []
    for ln in text.splitlines():
        if bad_re.search(ln):
            continue
        kept.append(ln)
    return "\n".join(kept).rstrip()


def clean_completion(raw_text: str, prompt: str, eos_token: str = None) -> str:
    text = raw_text

    if eos_token and eos_token in text:
        text = text.split(eos_token)[0]

    text = strip_markdown_fence(text)

    if text.startswith(prompt):
        text = text[len(prompt):]

    text = cut_at_stop_markers(text)
    text = strip_pollution_lines(text)
    text = dedupe_consecutive_lines(text)
    text = text.rstrip()

    if text and not text.startswith("\n"):
        text = "\n" + text

    return text


def build_chat_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert Python code completion engine for HumanEval.\n"
                "Return ONLY executable Python code continuation.\n"
                "Hard constraints:\n"
                "1) Output code only. No explanation, no markdown, no XML tags.\n"
                "2) Do NOT repeat the prompt.\n"
                "3) Do NOT add unrelated functions/classes.\n"
                "4) Preserve indentation and style.\n"
                "5) Stop after the target function completion.\n"
                "6) Never output placeholders such as ARGS/TODO/'the code would be'.\n"
                "7) Output must be syntactically valid Python."
            ),
        },
        {
            "role": "user",
            "content": (
                "Complete the following HumanEval prompt.\n"
                "Return only the missing continuation so it can be appended directly.\n"
                "Do not include explanations or tests.\n\n"
                "PROMPT:\n"
                f"{prompt}\n\n"
                "Now output ONLY the continuation code:"
            ),
        },
    ]


def generate_with_prompt(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
    use_chat_template: bool,
    debug: bool = False,
) -> str:
    if use_chat_template:
        messages = build_chat_messages(prompt)
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
        except TypeError:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    out_ids = outputs[0] if outputs.ndim == 2 else outputs
    gen_ids = out_ids[input_ids.shape[1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=False)

    if debug:
        print("===== RAW GENERATION =====")
        print(raw)

    eos_token = getattr(tokenizer, "eos_token", None)
    completion = clean_completion(raw, prompt=prompt, eos_token=eos_token)
    return completion


def main():
    parser = argparse.ArgumentParser(description="Prepare HumanEval samples in official format.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default="/share/public/public_models/Llama-3.1-8B-Instruct",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "-f",
        "--sample-input-file",
        type=str,
        default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/human-eval/data/HumanEval.jsonl",
        help="HumanEval problems jsonl",
    )
    parser.add_argument(
        "-o",
        "--sample-output-file",
        type=str,
        default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/human-eval/data/llama_samples.jsonl",
        help="Output jsonl with fields: task_id, completion",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Only run first N samples")
    parser.add_argument(
        "--use-chat-template",
        dest="use_chat_template",
        action="store_true",
        help="Use tokenizer chat template",
    )
    parser.add_argument(
        "--no-chat-template",
        dest="use_chat_template",
        action="store_false",
        help="Disable chat template and use raw prompt",
    )
    parser.set_defaults(use_chat_template=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    problems = read_jsonl(os.path.expanduser(args.sample_input_file))
    if args.limit is not None:
        problems = problems[: args.limit]

    total = len(problems)
    print(f"Loaded {total} problems")

    out_rows: List[Dict[str, Any]] = []
    for i, doc in enumerate(problems, start=1):
        try:
            task_id = doc["task_id"]
            prompt = doc["prompt"]

            completion = generate_with_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                use_chat_template=args.use_chat_template,
                debug=args.debug,
            )

            out_rows.append(
                {
                    "task_id": task_id,
                    "completion": completion,
                }
            )

            if i % 10 == 0 or i == total:
                print(f"[{time.strftime('%H:%M:%S')}] processed {i}/{total}")

            if args.debug:
                print("===== CLEAN COMPLETION =====")
                print(completion)
                print("=" * 60)

        except Exception as e:
            print(f"Error at sample {i}: {repr(e)}", file=sys.stderr)
            out_rows.append(
                {
                    "task_id": doc.get("task_id", f"unknown_{i}"),
                    "completion": "",
                }
            )

    write_jsonl(args.sample_output_file, out_rows)
    print(f"Saved: {args.sample_output_file}")


if __name__ == "__main__":
    main()
