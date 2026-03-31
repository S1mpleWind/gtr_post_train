#!/usr/bin/env python3
import os
import re
import sys
import json
import time
import argparse
import inspect
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationMixin
from transformers.masking_utils import create_causal_mask as hf_create_causal_mask

sys.path.insert(0, os.path.abspath("/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD"))

import model.llama_model_adaptor_global_router as llama_router_mod
from model.utils import Spec_update_model_kwargs_for_generation, Global_router, ShadowAdapter3
from model.EAGLE_model import Model as SpecModel


GenerationMixin._update_model_kwargs_for_generation = Spec_update_model_kwargs_for_generation

# transformers 版本兼容：某些版本 create_causal_mask 不支持 position_ids
if "position_ids" not in inspect.signature(hf_create_causal_mask).parameters:
    def _create_causal_mask_compat(*args, **kwargs):
        kwargs.pop("position_ids", None)
        return hf_create_causal_mask(*args, **kwargs)

    # 覆盖模型模块内部引用
    llama_router_mod.create_causal_mask = _create_causal_mask_compat

Spec_LlamaForCausalLM = llama_router_mod.Spec_LlamaForCausalLM


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
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
    out = []
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
    ]
    cut_pos = len(text)
    for marker in stop_markers:
        p = text.find(marker)
        if p != -1:
            cut_pos = min(cut_pos, p)
    return text[:cut_pos]


def clean_completion(raw_text: str, prompt: str, eos_token: str = None) -> str:
    text = raw_text

    if eos_token and eos_token in text:
        text = text.split(eos_token)[0]

    text = strip_markdown_fence(text)

    if text.startswith(prompt):
        text = text[len(prompt):]

    text = cut_at_stop_markers(text)
    text = dedupe_consecutive_lines(text)
    text = text.rstrip()

    if text and not text.startswith("\n"):
        text = "\n" + text

    return text


def build_inputs(tokenizer, prompt: str, device: str):
    messages = [
        {
            "role": "system",
            "content": "You are a precise Python coding assistant. Output code only."
        },
        {
            "role": "user",
            "content": (
                "Complete the following Python code task from HumanEval.\n"
                "Only output the function continuation without explanation.\n\n"
                + prompt
            )
        },
    ]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
    except TypeError:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

    return inputs


def main():
    parser = argparse.ArgumentParser(description="HumanEval prepare with SpecMoD infer pipeline.")
    parser.add_argument("-c", "--checkpoint-path", type=str, default="/share/public/public_models/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "-f",
        "--sample-input-file",
        type=str,
        default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/human-eval/data/HumanEval.jsonl",
    )
    parser.add_argument(
        "-o",
        "--sample-output-file",
        type=str,
        default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/human-eval/data/spec_llama_samples.jsonl",
    )

    parser.add_argument("--spec-model-path", type=str, default="/home/xujiaming/xujiaming/models/EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument(
        "--router-path",
        type=str,
        default="/home/xujiaming/xujiaming/models/Llama3.1_8B_global_router_1024_Model1_non_thinking.pt",
    )
    parser.add_argument(
        "--backbone-dir",
        type=str,
        default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/checkpoint/llama_backbone/backbone_final_2.pt",
    )
    parser.add_argument(
        "--adaptor-dir",
        type=str,
        default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/checkpoint/llama_adaptor_2",
    )

    parser.add_argument("--use-backbone", action="store_true")
    parser.add_argument("--skip-adaptor-layer", type=int, default=31)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1e-6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)

    print("Loading Spec Llama model ...")
    ori_model = Spec_LlamaForCausalLM.from_pretrained(args.checkpoint_path)
    if device == "cuda":
        ori_model = ori_model.half().to(device)
    else:
        ori_model = ori_model.to(device)
    ori_model.eval()

    if args.use_backbone:
        print(f"Loading finetuned backbone weights from {args.backbone_dir}")
        state_dict = torch.load(args.backbone_dir, map_location=device)
        if device == "cuda":
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    state_dict[k] = v.to(torch.float16)
        ori_model.load_state_dict(state_dict, strict=False)

    print("Loading SpecModel (EAGLE) ...")
    spec_model = SpecModel.from_pretrained(
        Spec_model_path=args.spec_model_path,
        Ori_model_path=args.checkpoint_path,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    spec_model.eval()

    layers = ori_model.config.num_hidden_layers
    router = Global_router(
        input_dim=ori_model.config.hidden_size * 2,
        hidden_dim=1024,
        output_dim=layers,
    ).to(device)

    print(f"Loading router weights from {args.router_path}")
    router_weight = torch.load(args.router_path, map_location=device)
    router.load_state_dict(router_weight)
    if device == "cuda":
        router = router.half()
    router.eval()

    print("Loading adaptors ...")
    adaptor = [None]
    for i in range(1, layers):
        if i == args.skip_adaptor_layer:
            adaptor.append(None)
            continue

        layer_adaptor = ShadowAdapter3(ori_model.config.hidden_size, 1024)
        layer_adaptor_path = os.path.join(args.adaptor_dir, f"adapter_layer_{i}_1024_final.pt")
        layer_adaptor_weight = torch.load(layer_adaptor_path, map_location=device)
        layer_adaptor.load_state_dict(layer_adaptor_weight)
        if device == "cuda":
            layer_adaptor = layer_adaptor.half()
        layer_adaptor = layer_adaptor.to(ori_model.device).eval()
        adaptor.append(layer_adaptor)

    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and eos_token_id is not None:
        pad_token_id = eos_token_id
        tokenizer.pad_token_id = pad_token_id

    problems = read_jsonl(os.path.expanduser(args.sample_input_file))
    if args.limit is not None:
        problems = problems[: args.limit]

    total = len(problems)
    print(f"Loaded {total} HumanEval problems")

    out_rows: List[Dict[str, Any]] = []

    for idx, doc in enumerate(problems, start=1):
        task_id = doc.get("task_id", f"unknown_{idx}")
        prompt = doc.get("prompt", "")

        try:
            inputs = build_inputs(tokenizer, prompt=prompt, device=device)

            # 每个样本都重置内部输入状态，避免跨样本残留
            if hasattr(ori_model.model, "input_id_init"):
                ori_model.model.input_id_init()
            elif hasattr(ori_model.model, "input_ids"):
                ori_model.model.input_ids = None

            try:
                spec_model.reset_kv()
            except Exception:
                pass

            with torch.no_grad():
                outputs = ori_model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    router=router,
                    spec_model=spec_model,
                    adaptor=adaptor,
                    last_hidden_state=None,
                )

            gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
            raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
            completion = clean_completion(raw_text, prompt=prompt, eos_token=eos_token)

            if args.debug:
                print("=" * 80)
                print(f"[{task_id}] RAW")
                print(raw_text)
                print(f"[{task_id}] CLEAN")
                print(completion)

            out_rows.append(
                {
                    "task_id": task_id,
                    "completion": completion,
                }
            )

        except Exception as e:
            print(f"Error at {task_id}: {repr(e)}", file=sys.stderr)
            out_rows.append(
                {
                    "task_id": task_id,
                    "completion": "",
                }
            )

        if idx % 10 == 0 or idx == total:
            print(f"[{time.strftime('%H:%M:%S')}] processed {idx}/{total}")

    write_jsonl(args.sample_output_file, out_rows)
    print(f"Saved: {args.sample_output_file}")


if __name__ == "__main__":
    main()