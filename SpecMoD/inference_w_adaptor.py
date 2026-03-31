'''
This file is mainly designed for collecting training data for the adaptor in each layer.

The details are in Feishu "Adaptor 训练数据"

'''


from transformers import AutoTokenizer

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from typing import Optional
import json, tqdm
import torch
import torch.nn as nn
from model.utils import storage, ShadowAdapter2, ShadowAdapter3,record


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
    from model.qwen3_model_adaptor_pipeline import  Spec_Qwen3ForCausalLM
    
    if args.device == "infini":
        model_path = f"/share/others/public_models/{args.model}/"
        dataset_path = '/home/xujiaming/xujiaming/Paper/dataset/'+args.dataset+'/question.jsonl'
    elif args.device == "qz":
        model_path = f"/inspire/hdd/global_public/public_models/Qwen/{args.model}/"
        dataset_path = '/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/benchmark/'+args.dataset+'/question.jsonl'
    else:
        assert False, "device error"
        
    

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Spec_Qwen3ForCausalLM.from_pretrained(model_path).half().to('cuda')
    LAYERS = model.config.num_hidden_layers
    adaptor = [None, ]
    
    # adaptor = nn.ModuleList([
    #         ShadowAdapter3(model.config.hidden_size, 2048) for _ in range(LAYERS)
    #     ])
    # adaptor.load_state_dict(torch.load("/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/checkpoint/adaptor/2048_finetune/final_adapters_2048_20251217_0700.pt"))
    # adaptor.half().to(model.device)
    
    
    for i in range(1, LAYERS):
        if i == 34:
            adaptor.append(None)
        else:
            layer_adaptor = ShadowAdapter3(model.config.hidden_size, 2048)
            layer_adaptor_weight = torch.load(f"./checkpoint/adaptor/2048/adapter_layer_{i}_2048_Model3_0.95.pt")
            layer_adaptor.load_state_dict(layer_adaptor_weight)
            layer_adaptor = layer_adaptor.half().to(model.device)
            adaptor.append(layer_adaptor)
            

        
    

    questions = load_questions(dataset_path,args.begin,args.end)
    for question in tqdm.tqdm(questions):
        messages = [
            {"role": "system",
                "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
        prompt = question["turns"][0]
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking = True, 
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=args.max_gen, temperature=0.000001, adaptor=adaptor)
        print(tokenizer.decode(outputs[0]))
    print(record.get_average_len())
  

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="infini")
    parser.add_argument("--dataset", "-d", type=str, default="mt-bench")
    # dataset: mt-bench, gsm8k, alpaca, sum, vicuna-bench
    parser.add_argument("--model", "-m", type=str, default="Qwen3-8B")
    parser.add_argument("--begin", "-b", type=int, default=None)
    parser.add_argument("--end", "-e", type=int, default=None)
    parser.add_argument("--setting", type=str, default='dev')
    parser.add_argument("--max_gen", type=int, default=100)

    
    args = parser.parse_args()
    main(args)