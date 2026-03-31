
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
from transformers.generation.utils import GenerationMixin
from transformers import AutoTokenizer

# 让 benchmark 目录下运行时可导入 SpecMoD
sys.path.insert(0, os.path.abspath("/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD"))

from model.llama_model_adaptor_global_router import Spec_LlamaForCausalLM
from model.utils import Spec_update_model_kwargs_for_generation, Global_router, ShadowAdapter3
from model.EAGLE_model import Model as SpecModel


# 使用自定义 generation kwargs 更新逻辑
GenerationMixin._update_model_kwargs_for_generation = Spec_update_model_kwargs_for_generation

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
            "content": "You are a careful assistant for multiple-choice reasoning.",
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
    parser = argparse.ArgumentParser(description="MMLU infer with Spec Llama + router/adaptor/spec_model.")
    parser.add_argument("-c", "--checkpoint-path", type=str, default="/share/public/public_models/Llama-3.1-8B-Instruct")
    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument("-o", "--sample-output-file", type=str, default="mmlu_res_infer.jsonl")
    parser.add_argument("--dataset-name", type=str, default="cais/mmlu")
    parser.add_argument("--subset", type=str, default="all")
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--spec-model-path", type=str, default="/home/xujiaming/xujiaming/models/EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--router-path", type=str, default="/home/xujiaming/xujiaming/models/Llama3.1_8B_global_router_1024_Model1_non_thinking.pt")
    parser.add_argument("--backbone-dir", type=str, default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/checkpoint/llama_backbone_v1/backbone_final.pt")
    parser.add_argument("--use-backbone", action="store_true")
    parser.add_argument("--adaptor-dir", type=str, default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/checkpoint/llama_adaptor_v1")

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    test = load_eval_split(args.sample_input_file, args.dataset_name, args.subset, args.split)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and device == "cuda":
                state_dict[k] = v.to(torch.float16)
        ori_model.load_state_dict(state_dict, strict=False)

    print("Loading SpecModel (EAGLE) ...")
    spec_model = SpecModel.from_pretrained(
        Spec_model_path=args.spec_model_path,
        Ori_model_path=args.checkpoint_path,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    LAYERS = ori_model.config.num_hidden_layers
    router = Global_router(
        input_dim=ori_model.config.hidden_size * 2,
        hidden_dim=1024,
        output_dim=LAYERS,
    ).to(device)

    router_weight = torch.load(args.router_path, map_location=device)
    router.load_state_dict(router_weight)
    if device == "cuda":
        router = router.half()
    router.eval()

    print("Loading adaptors ...")
    adaptor = [None]
    for i in range(1, LAYERS):
        # 你之前 gsm8k 脚本里的保留逻辑
        if i == 31:
            adaptor.append(None)
        else:
            layer_adaptor = ShadowAdapter3(ori_model.config.hidden_size, 1024)
            layer_adaptor_weight = torch.load(
                f"{args.adaptor_dir}/adapter_layer_{i}_1024_final.pt",
                map_location=device,
            )
            layer_adaptor.load_state_dict(layer_adaptor_weight)
            if device == "cuda":
                layer_adaptor = layer_adaptor.half()
            layer_adaptor = layer_adaptor.to(ori_model.device).eval()
            adaptor.append(layer_adaptor)

    eos_token = getattr(tokenizer, "eos_token", None)

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
            ).to(device)

            # reset 内部状态
            try:
                ori_model.model.input_ids = None
            except Exception:
                pass

            try:
                spec_model.reset_kv()
            except Exception:
                pass

            outputs = ori_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1e-6,
                top_p=1.0,
                top_k=50,
                router=router,
                spec_model=spec_model,
                adaptor=adaptor,
                last_hidden_state=None,
            )

            out_ids = outputs[0] if outputs.ndim == 2 else outputs
            gen_ids = out_ids[inputs["input_ids"].shape[1]:]
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

            if i%2000 ==0:
                break

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
