import re
import torch
import argparse
import jsonlines
import numpy as np
import time
import datasets
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.generation.utils import GenerationMixin

# 从 SpecMoD 文件中导入自定义类
import sys, os
sys.path.insert(0, os.path.abspath("/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD"))

from model.qwen3_model_global_soft_router_pipeline import Spec_Qwen3ForCausalLM
from model.utils import Spec_update_model_kwargs_for_generation, Global_router
from model.EAGLE_model import Model as SpecModel

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
FINAL_ANS_PATTERNS = [
    re.compile(r"the answer is\s*\$?\s*(-?[0-9][0-9,\.]*)", re.IGNORECASE),
    re.compile(r"answer:\s*\$?\s*(-?[0-9][0-9,\.]*)", re.IGNORECASE),
]
INVALID_ANS = "[invalid]"

# 将 datasets 的 generation mixin 指向自定义函数（跟 inference 文件一致）
GenerationMixin._update_model_kwargs_for_generation = Spec_update_model_kwargs_for_generation

def doc_to_text(doc):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\nAnswer: "
    )

def parse_num(s):
    s = s.strip().replace(",", "")
    try:
        return float(s) if "." in s else int(s)
    except Exception:
        return INVALID_ANS

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
        "\nLet's think step by step\nQuestion:",
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

def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS

def extract_answer(completion):
    # 优先从 "The answer is ..." 提取
    for pat in FINAL_ANS_PATTERNS:
        m = pat.search(completion)
        if m:
            val = parse_num(m.group(1))
            if val != INVALID_ANS:
                return val

    # 回退：取最后一个数字（支持负数和小数）
    nums = re.findall(r"-?[0-9][0-9,\.]*", completion)
    if not nums:
        return INVALID_ANS
    return parse_num(nums[-1])

def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Qwen3 Spec pipeline on GSM8K.")
    parser.add_argument("-c", "--checkpoint-path", type=str, default="/share/public/public_models/Qwen3-8B")
    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument("-o", "--sample-output-file", type=str, default="gsm8k_res_infer.jsonl")
    parser.add_argument("--spec-model-path", type=str, default="/home/xujiaming/xujiaming/models/Qwen3-8B_eagle3")
    parser.add_argument("--router-path", type=str, default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/final_model/global_router/global_router_1024_Model1_non_thinking_first.pt")
    parser.add_argument("--backbone-dir", type=str, default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/checkpoint/backbone/backbone_1.pt")
    parser.add_argument("--use-backbone", action="store_true")
    parser.add_argument("--max-gen", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1e-6)
    args = parser.parse_args()

    fewshot_prompt = open("gsm8k_prompt.txt").read()

    # 加载数据集：优先支持 jsonl 文件或 HF datasets 缓存目录
    if args.sample_input_file is not None:
        # 如果传入的是 jsonl 文件
        if args.sample_input_file.endswith(".jsonl") or args.sample_input_file.endswith(".json"):
            with jsonlines.open(args.sample_input_file) as reader:
                test = list(reader)
        else:
            # 尝试 load_from_disk（datasets.save_to_disk 格式）
            dataset = load_from_disk(args.sample_input_file)
            test = dataset["test"]
    else:
        config = datasets.DownloadConfig(resume_download=True, max_retries=100)
        dataset = load_dataset("gsm8k", "main", download_config=config)
        test = dataset["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)

    print("Loading Spec Qwen3 model ...")
    ori_model = Spec_Qwen3ForCausalLM.from_pretrained(
        args.checkpoint_path,
        #attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
    )
    if device == "cuda":
        ori_model = ori_model.half().to(device)
    else:
        ori_model = ori_model.to(device)

    # 加载微调 backbone 权重
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
        dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    LAYERS = ori_model.config.num_hidden_layers
    router = Global_router(input_dim=ori_model.config.hidden_size * 2, hidden_dim=1024, output_dim=LAYERS).to(device)
    router_weight = torch.load(args.router_path, map_location=device)
    router.load_state_dict(router_weight)
    if device == "cuda":
        router = router.half()

    eos_token = getattr(tokenizer, "eos_token", None)

    f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))
    acc_res = []

    # 如果 test 是 Dataset 对象，则按迭代器；若为 list(jsons) 也支持
    iterator = test if isinstance(test, list) else test

    for i, doc in enumerate(iterator):
        try:
            # 构造 chat messages（与 inference 文件一致）
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."}
            ]
            messages.append({"role": "user", "content": doc["question"]})

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            # 确保模型内部状态初始化
            try:
                ori_model.model.input_id_init()
            except Exception:
                # 如果没有该方法，忽略
                pass

            try:
                spec_model.reset_kv()
            except Exception:
                pass

            outputs = ori_model.generate(
                **inputs,
                max_new_tokens=args.max_gen,
                temperature=args.temperature,
                router=router,
                do_sample = False,
                spec_model=spec_model,
                last_hidden_state=None
            )

            # 提取生成 tokens 并解码为文本，剪掉prompt
            gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
            completion = safe_decode_gen(gen_tokens, tokenizer, eos_token)

            answer = doc.get("answer")
            try:
                acc = is_correct(completion, answer) if answer is not None else False
            except AssertionError:
                acc = False

            try:
                doc["completion"] = completion
                doc["acc"] = acc
                f_output.write(doc)
            except Exception:
                f_output.write({"question": doc.get("question"), "answer": answer, "completion": completion, "acc": acc})

            acc_res.append(acc)

            if (i + 1) % 10 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] processed {i + 1}")

            # if (i + 1) % 100 == 0:
            #     break

        except Exception as e:
            print(f"Error at sample {i}: {e}", file=sys.stderr)
            err = {
                "idx": i,
                "question": doc.get("question") if isinstance(doc, dict) else None,
                "error": repr(e),
            }
            f_output.write(err)

    f_output.close()
    print("Acc: ", np.mean(acc_res))