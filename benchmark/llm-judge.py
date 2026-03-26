import json
import random
import requests
import multiprocessing
from tqdm import tqdm, trange
import re
import argparse

def chat(prompt):
    url = "https://cloud.infini-ai.com/maas/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-spwdayuga7tf4fvt"
    }
    payload = {
        "model": "kimi-k2.5",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
    }
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()
    
    if "choices" not in data:
        print(f"api error, status = {response.status_code}")
        print(f"response body = {data}")  # 关键：打印实际返回内容
        raise RuntimeError(f"api error, status = {response.status_code}, body = {data}")
    
    return data['choices'][0]['message']['content']

def main(args):
    dims = ["Relevance", "Accuracy", "Coherence", "Clarity", "Breadth and Depth", "Reading Experience"]
    scores = [0, 0, 0, 0, 0, 0]
    count = 0
    
    # 读取 prompt 和 answer
    prompts = open(args.prompts_file, "r", encoding="utf-8")
    prompts = [json.loads(line) for line in prompts]
    prompts = [pro["prompt"] for pro in prompts]
    
    with open(args.answers_file, 'r', encoding='utf-8') as file:
        answers = json.load(file)
    
    prompt_template = open(args.judge_template, "r", encoding="utf-8").read()
    
    for i in trange(len(prompts)):
        prompt = prompts[i]
        res = answers[i]
        input_prompt = prompt_template.replace('$INST$', prompt).replace('$RESPONSE$', res)
        output = chat(input_prompt)
        output = output[:output.find('}')+1]
        
        try:
            json_output = json.loads(output)
            tag = 0
            this_score = []
            for x in range(len(dims)):
                try:
                    this_score.append(int(json_output[dims[x]]))
                except:
                    tag = 1
                    break
            if tag == 0:
                print("success")
                count += 1
                for p in range(len(this_score)):
                    scores[p] += this_score[p]
        except:
            continue
    
    # 输出结果
    print("\n" + "="*50)
    print(f"Results for {args.answers_file}")
    print("="*50)
    for ppp in range(len(dims)):
        avg_score = scores[ppp]/count if count > 0 else 0
        print(f"{dims[ppp]:20s} : {avg_score:.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", "-p", type=str, default="prompts.jsonl",
                        help="Path to prompts JSONL file")
    parser.add_argument("--answers", "-a", type=str, default="outputs.json",
                        help="Path to answers JSON file")
    parser.add_argument("--template", "-t", type=str, default="judge.txt",
                        help="Path to judge template file")
    args = parser.parse_args()
    args.prompts_file = args.prompts
    args.answers_file = args.answers
    args.judge_template = args.template
    main(args)