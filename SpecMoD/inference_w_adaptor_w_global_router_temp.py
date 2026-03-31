'''
This file is mainly designed for collecting training data for the adaptor in each layer.

The details are in Feishu "Adaptor 训练数据"

改进适配微调

'''


from transformers import AutoTokenizer

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from typing import Optional
import json, tqdm
import torch
import torch.nn as nn
from model.utils import storage, ShadowAdapter2, ShadowAdapter3, PathPredictorMLP, record, Global_router
import time
from model.EAGLE_model import Model as SpecModel


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
    from model.qwen3_model_adaptor_global_router_pipeline import  Spec_Qwen3ForCausalLM
    from model.utils import Spec_update_model_kwargs_for_generation
    from transformers.generation.utils import GenerationMixin
    GenerationMixin._update_model_kwargs_for_generation = Spec_update_model_kwargs_for_generation

    #model_path = "/share/public/public_models/Qwen3-14B"
    model_path = "/share/public/public_models/Qwen3-8B"
    dataset_path = '/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/dataset/' + args.dataset + '/question.jsonl'

    

    tokenizer = AutoTokenizer.from_pretrained(model_path)


    # original model: from dist load weight and convert to 16fp
    ori_model = Spec_Qwen3ForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" 
        ).half().to('cuda')
    
    
    # 加载微调后的backbone权重
    if args.use_backbone:
        print(f"Loading finetuned backbone weights from {args.backbone_dir}")
        state_dict = torch.load(args.backbone_dir, map_location='cuda')
        ori_model.load_state_dict(state_dict, strict=False)
        
    #Spec_model_path = "/home/xujiaming/xujiaming/models/Qwen3-14B_eagle3"
    Spec_model_path = "/home/xujiaming/xujiaming/models/Qwen3-8B_eagle3"
    spec_model = SpecModel.from_pretrained(
        Spec_model_path=Spec_model_path, 
        Ori_model_path=model_path, 
        dtype=torch.float16).to(ori_model.device)
    LAYERS = ori_model.config.num_hidden_layers
    adaptor = [None, ]
    router = Global_router(input_dim=ori_model.config.hidden_size*2, hidden_dim=1024, output_dim=LAYERS).to(ori_model.device)
    #router_weight = torch.load("/home/xujiaming/xujiaming/jiaoyifan/SpecMoD/final_model/global_router/Qwen3_14B_global_router_1024_Model1_non_thinking.pt")
    #router_weight = torch.load("router_weight = torch.load("/home/xujiaming/xujiaming/jiaoyifan/SpecMoD/final_model/global_router/global_router_1024_Model1_non_thinking_first.pt")
    
    #router_weight = torch.load("/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/final_model/global_router/Qwen3_14B_global_router_1024_Model1_non_thinking.pt")
    router_weight = torch.load("/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/final_model/global_router/global_router_1024_Model1_non_thinking_first.pt")
    # router = PathPredictorMLP(n_layers=LAYERS, mlp_internal_dim=2048, llm_hidden_dim=ori_model.config.hidden_size*2).to(ori_model.device)
    # router_weight = torch.load("/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/checkpoint/global_router/global_router_2048_Model2.pt")
    router.load_state_dict(router_weight)
    router = router.half()
    
    
    prompts_list = []
    outputs_list = []


    
    for i in range(1, LAYERS):
        # TODO: change the layer
        if i == 34:
            adaptor.append(None)
        else:
            layer_adaptor = ShadowAdapter3(ori_model.config.hidden_size, 1024)
            layer_adaptor_weight = torch.load(f"{args.adaptor_dir}/adapter_layer_{i}_1024_final.pt")
            layer_adaptor.load_state_dict(layer_adaptor_weight)
            layer_adaptor = layer_adaptor.half().to(ori_model.device)
            adaptor.append(layer_adaptor)
    


    questions = load_questions(dataset_path,args.begin,args.end)
    for question in tqdm.tqdm(questions):
        messages = [
            {"role": "system",
                "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
        prompt = question["turns"][0]

        # record the prompts
        #system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
  
        prompts_list.append({"prompt": prompt})

        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # 处理成qwen要求的chat模板的规范
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking = args.enable_thinking, 
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(ori_model.device)

        ori_model.model.input_id_init()

        spec_model.reset_kv()
        # 索引第一批
        outputs = ori_model.generate(
            **inputs, 
            max_new_tokens=args.max_gen, 
            temperature=args.temperature, 
            adaptor=adaptor, 
            router=router,
            spec_model=spec_model, 
            last_hidden_state = None)
        
        #use slide to get the pure output
        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        gen_txt = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        print (gen_txt)

        outputs_list.append(gen_txt)

        print(tokenizer.decode(outputs[0]))
        
        print(record.get_average_len())

    if args.write_record :
        with open(os.path.join(args.out_dir, "record_stats.txt"), "a", encoding="utf-8") as f:
            f.write("record id: {}\n".format(args.dataset))
            # f.write("exec_layer_list: {}\n".format(record.exec_layer_list))
            try:
                f.write("average: {}\n".format(record.get_average_len()))
            except Exception as e:
                f.write("average: error {}\n".format(e))
        


    with open(os.path.join(args.out_dir, "prompts.jsonl"), "w", encoding="utf-8") as f:
        for prompt_obj in prompts_list:
            f.write(json.dumps(prompt_obj, ensure_ascii=False) + "\n")


    with open(os.path.join(args.out_dir, "outputs.jsonl"), "w", encoding="utf-8") as f:
        json.dump(outputs_list, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(prompts_list)} prompts to prompts.jsonl")
    print(f"Saved {len(outputs_list)} outputs to outputs.json")

    # print(record.get_average_len())
  

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="humaneval")
    # dataset: mt-bench, gsm8k, alpaca, sum, vicuna-bench
    parser.add_argument("--model", "-m", type=str, default="Qwen3-8B")
    parser.add_argument("--begin", "-b", type=int, default=None)
    parser.add_argument("--end", "-e", type=int, default=None)
    parser.add_argument("--max_gen", type=int, default=500)
    parser.add_argument("--temperature", "-t", type=float, default=0.000001)
    parser.add_argument("--out_dir", type=str, default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/tmp")

    parser.add_argument("--adaptor_dir",type=str,default = "/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/checkpoint/adaptor_with_full_backbone_v1")

    parser.add_argument("--use_backbone",default = False)
    parser.add_argument("--backbone_dir", type=str, default ="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/checkpoint/full_backbone/backbone_final_v1.pt")

    parser.add_argument("--write_record", default = False)

    parser.add_argument("--enable_thinking",default = False)
    args = parser.parse_args()
    main(args)