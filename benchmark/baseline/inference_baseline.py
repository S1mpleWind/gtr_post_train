'''
Baseline inference script using vanilla Qwen3-8B without any modifications.
'''

from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
from tqdm import tqdm

def load_questions(question_file: str, begin=None, end=None):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions

def main(args):
    model_path = "/share/public/public_models/Llama-3.1-8B-Instruct"
    dataset_path = '/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/dataset/' + args.dataset + '/question.jsonl'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        device_map="cuda"
        )
    
    prompts_list = []
    outputs_list = []
    
    questions = load_questions(dataset_path, args.begin, args.end)
    
    for question in tqdm(questions):
        messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
        prompt = question["turns"][0]
        prompts_list.append({"prompt": prompt})
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # change into the templete of Qwen
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Vanilla generation
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_gen,
            temperature=0.000001,
        )
        
        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        gen_txt = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        outputs_list.append(gen_txt)
        
        print(gen_txt)
    
    # Save results
    with open("/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/baseline/prompts_baseline.jsonl", "w", encoding="utf-8") as f:
        for prompt_obj in prompts_list:
            f.write(json.dumps(prompt_obj, ensure_ascii=False) + "\n")
    
    with open("/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/baseline/outputs_baseline.json", "w", encoding="utf-8") as f:
        json.dump(outputs_list, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(prompts_list)} prompts to prompts_baseline.jsonl")
    print(f"Saved {len(outputs_list)} outputs to outputs_baseline.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="mt_bench")
    parser.add_argument("--begin", "-b", type=int, default=None)
    parser.add_argument("--end", "-e", type=int, default=None)
    parser.add_argument("--max_gen", type=int, default=500)
    args = parser.parse_args()
    main(args)