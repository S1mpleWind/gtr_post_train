"""Generate responses from an RL training checkpoint without GenerationMixin patches."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from RL4SpecMoD.model import DecodePolicy
from RL4SpecMoD.train import eos_ids, prompt_ids, read_data


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sample", action="store_true", help="Sample tokens and routes; default uses argmax/threshold")
    parser.add_argument("--seed", type=int, default=42)
    cli = parser.parse_args()
    if cli.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    torch.manual_seed(cli.seed)
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    config = argparse.Namespace(**checkpoint["args"])
    config.device = cli.device
    policy = DecodePolicy(config)
    policy.router.load_state_dict(checkpoint["router"])
    policy.adaptors.load_state_dict(checkpoint["adaptors"])
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    eos = eos_ids(policy, tokenizer)
    destination = Path(cli.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        for record in read_data(cli.data):
            prompt = prompt_ids(tokenizer, record, config, policy.target.device)
            state, logits = policy.prefill(prompt)
            tokens, executed, transitions = [], 0, 0
            for _ in range(cli.max_new_tokens):
                token = policy.token_dist(logits).sample()[0] if cli.sample else logits.argmax(-1)[0]
                tokens.append(token.item())
                if token.item() in eos or len(tokens) == cli.max_new_tokens:
                    break
                logits, _, _, _, _, execute = policy.step(state, token, greedy=not cli.sample)
                transitions += 1
                executed += execute.sum().item()
            result = dict(record)
            result.update(response=tokenizer.decode(tokens, skip_special_tokens=True),
                          generated_ids=tokens, routed_transitions=transitions,
                          execute_fraction=executed / (transitions * policy.depth) if transitions else 1.)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
