import json

def load_prompts(prompt_file):
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def load_outputs(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        outputs = json.load(f)  # 你的 outputs.json 是 list
    return outputs


def merge(prompt_file, output_file, out_file):
    prompts = load_prompts(prompt_file)
    outputs = load_outputs(output_file)

    assert len(prompts) == len(outputs), \
        f"Length mismatch: {len(prompts)} vs {len(outputs)}"

    with open(out_file, "w", encoding="utf-8") as f:
        for i in range(len(outputs)):
            # 你之前存的是 {"prompt": "..."}
            question = prompts[i].get("prompt", "")

            obj = {
                "question": question,
                "completion": outputs[i]
            }

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved {len(outputs)} samples to {out_file}")


if __name__ == "__main__":
    merge(
        prompt_file="prompts.jsonl",
        output_file="outputs.json",
        out_file="evalscope_input.jsonl"
    )