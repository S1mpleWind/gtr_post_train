"""Online grouped PPO over actual SpecMoD decode routes and response tokens."""
import argparse
import copy
import importlib
import json
import math
import random
import sys
from pathlib import Path

# No assumption about checkout directory name or the launch working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.distributions import Bernoulli, kl_divergence
from transformers import AutoTokenizer

from RL4SpecMoD.model import DecodePolicy


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--eagle", required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--adaptor-pattern", required=True,
                   help="Checkpoint path with {layer} and optionally {dim}")
    p.add_argument("--backbone", help="Optional complete fine-tuned target state_dict")
    p.add_argument("--data", required=True, help="JSONL: prompt, messages, or turns")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", help="Path to training checkpoint.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--router-dim", type=int, default=1024)
    p.add_argument("--adaptor-dim", type=int, default=1024)
    p.add_argument("--forced-layers", type=int, nargs="+")
    p.add_argument("--train-adaptors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=1000, help="Total rollout groups, including resumed ones")
    p.add_argument("--ppo-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=.2)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--token-temperature", type=float, default=1.)
    p.add_argument("--route-temperature", type=float, default=1.)
    p.add_argument("--quality-weight", type=float, default=1.)
    p.add_argument("--compute-weight", type=float, default=.2)
    p.add_argument("--kl-coef", type=float, default=.01)
    p.add_argument("--entropy-coef", type=float, default=.001)
    p.add_argument("--reward-fn", help="module:function(tokenizer, prompt_record, response_text) -> float")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=50)
    a = p.parse_args()
    for name in ("group_size", "steps", "ppo_epochs", "max_prompt_tokens", "save_every"):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if a.group_size < 2 or a.max_new_tokens < 2:
        p.error("group-size and max-new-tokens must be at least 2")
    for name in ("lr", "max_grad_norm", "token_temperature", "route_temperature"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            p.error(f"{name} must be finite and positive")
    if not 0 < a.clip < 1:
        p.error("clip must be between 0 and 1")
    return a


def read_data(path):
    with open(path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    if not data:
        raise ValueError("Empty prompt dataset")
    return data


def prompt_ids(tokenizer, record, args, device):
    if "messages" in record:
        messages = record["messages"]
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("messages must end with a user turn; remove reference answers")
    else:
        prompt = record.get("prompt")
        if prompt is None:
            prompt = record["turns"][0]
        messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=args.enable_thinking, return_tensors="pt",
    )
    # Fail instead of silently dropping the system message or corrupting chat delimiters.
    if ids.shape[1] > args.max_prompt_tokens:
        raise ValueError("Prompt exceeds --max-prompt-tokens; filter data or increase the limit")
    if ids.shape[1] == 0:
        raise ValueError("Empty tokenized prompt")
    return ids.to(device)


def eos_ids(policy, tokenizer):
    value = policy.target.generation_config.eos_token_id
    if value is None:
        value = tokenizer.eos_token_id
    return set(value if isinstance(value, list) else [value])


@torch.no_grad()
def rollout(policy, prompt, eos, args):
    state, logits = policy.prefill(prompt)
    first = policy.token_dist(logits).sample()[0]
    tokens = [first]
    actions, old_log_probs = [], []
    executed = 0
    if first.item() not in eos:
        for _ in range(args.max_new_tokens - 1):
            logits, action, route_lp, _, _, execute = policy.step(state, tokens[-1])
            token_distribution = policy.token_dist(logits)
            token = token_distribution.sample()[0]
            # Joint probability of sampled path and next token. Forced layers have no action term.
            old_log_probs.append((route_lp + token_distribution.log_prob(token)[0]).cpu())
            actions.append(action.cpu())
            executed += execute.sum().item()
            tokens.append(token)
            if token.item() in eos:
                break
    count = len(actions)
    del state
    return dict(tokens=torch.stack(tokens).cpu(), actions=actions,
                old_log_probs=old_log_probs,
                execute_fraction=executed / (count * policy.depth) if count else 1.)


def replay(policy, reference_router, prompt, trajectory, advantage, scale, args):
    """Rebuild trajectory caches with recorded actions; no stale rollout graphs."""
    state, _ = policy.prefill(prompt)
    tokens = trajectory["tokens"].to(prompt.device)
    totals = dict(loss=0., kl=0., entropy=0., clip_fraction=0.)
    for i, (action, old_lp) in enumerate(zip(trajectory["actions"], trajectory["old_log_probs"])):
        logits, _, route_lp, dist, observation, _ = policy.step(state, tokens[i], action)
        token_lp = policy.token_dist(logits).log_prob(tokens[i + 1])[0]
        new_lp = route_lp + token_lp
        ratio = (new_lp - old_lp.to(new_lp.device)).clamp(-20, 20).exp()
        surrogate = torch.minimum(ratio * advantage,
                                  ratio.clamp(1 - args.clip, 1 + args.clip) * advantage)
        # Recompute the reference on the SAME observation (not a separate decode).
        with torch.no_grad():
            ref_logits = reference_router(observation)[0, 0]
            ref_dist = Bernoulli(logits=ref_logits[policy.free] / args.route_temperature)
        kl = kl_divergence(dist, ref_dist).sum()
        entropy = dist.entropy().sum()
        loss = (-surrogate + args.kl_coef * kl - args.entropy_coef * entropy) * scale
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite PPO loss; optimizer update aborted")
        loss.backward()
        # Bound memory: one decode transition graph, while keeping numeric KV history.
        policy.detach_cache(state)
        totals["loss"] += loss.item()
        totals["kl"] += kl.item() * scale
        totals["entropy"] += entropy.item() * scale
        totals["clip_fraction"] += float(abs(ratio.item() - 1) > args.clip) * scale
    return totals


def save(policy, reference_router, optimizer, step, order, cursor, args):
    directory = Path(args.output) / f"step-{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    state = dict(router=policy.router.state_dict(), adaptors=policy.adaptors.state_dict(),
                 reference_router=reference_router.state_dict(), optimizer=optimizer.state_dict(),
                 step=step, order=order, cursor=cursor, args=vars(args),
                 random_state=random.getstate(), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    temporary = directory / "checkpoint.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(directory / "checkpoint.pt")
    torch.save(policy.router.state_dict(), directory / "router.pt")
    for layer, module in policy.adaptors.items():
        torch.save(module.state_dict(), directory / f"adapter_layer_{layer}_{args.adaptor_dim}_rl.pt")
    with open(directory / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    return str(directory)


def main():
    args = arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = DecodePolicy(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = read_data(args.data)
    reference = copy.deepcopy(policy.router).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    order = list(range(len(data)))
    random.shuffle(order)
    cursor, start = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        # These options define the behavior policy, reference, rewards, and data cursor.
        keys = ("model", "eagle", "backbone", "router_dim", "adaptor_dim", "forced_layers",
                "train_adaptors", "data", "dtype", "enable_thinking", "max_prompt_tokens",
                "max_new_tokens", "group_size", "token_temperature", "route_temperature",
                "quality_weight", "compute_weight", "reward_fn", "clip", "kl_coef",
                "entropy_coef", "ppo_epochs", "lr", "max_grad_norm")
        for key in keys:
            if state["args"][key] != getattr(args, key):
                raise ValueError(f"Resume config mismatch: {key}")
        policy.router.load_state_dict(state["router"])
        policy.adaptors.load_state_dict(state["adaptors"])
        reference.load_state_dict(state["reference_router"])
        optimizer.load_state_dict(state["optimizer"])
        start, order, cursor = state["step"], state["order"], state["cursor"]
        if sorted(order) != list(range(len(data))):
            raise ValueError("Resume dataset length changed")
        random.setstate(state["random_state"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    reward_fn = None
    if args.reward_fn:
        module, name = args.reward_fn.rsplit(":", 1)
        reward_fn = getattr(importlib.import_module(module), name)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    eos = eos_ids(policy, tokenizer)
    for step in range(start + 1, args.steps + 1):
        if cursor == len(order):
            random.shuffle(order)
            cursor = 0
        record = data[order[cursor]]
        cursor += 1
        prompt = prompt_ids(tokenizer, record, args, policy.target.device)
        group = []
        for _ in range(args.group_size):
            trajectory = rollout(policy, prompt, eos, args)
            tokens = trajectory["tokens"].to(prompt.device)
            nll = policy.teacher_nll(prompt, tokens)
            text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
            quality = float(reward_fn(tokenizer, record, text)) if reward_fn else -nll
            reward = args.quality_weight * quality + args.compute_weight * (1 - trajectory["execute_fraction"])
            if not math.isfinite(reward):
                raise ValueError("Reward must be finite")
            trajectory.update(reward=reward, quality=quality, teacher_nll=nll)
            group.append(trajectory)
        rewards = torch.tensor([t["reward"] for t in group])
        # Leave-one-out baseline depends only on other independently sampled trajectories.
        advantages = (rewards - (rewards.sum() - rewards) / (args.group_size - 1)).tolist()
        metrics = dict(loss=0., kl=0., entropy=0., clip_fraction=0.)
        for _ in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            active = False
            for trajectory, advantage in zip(group, advantages):
                count = len(trajectory["actions"])
                if not count:
                    continue
                active = True
                # Sum transition policy gradients, average independent trajectories.
                # Constant horizon normalization avoids weighting a trajectory by its sampled length.
                scale = 1 / (args.group_size * (args.max_new_tokens - 1))
                result = replay(policy, reference, prompt, trajectory, advantage, scale, args)
                for key in metrics:
                    metrics[key] += result[key] / args.ppo_epochs
            if active:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm,
                                               error_if_nonfinite=True)
                optimizer.step()
        metrics.update(step=step, reward=rewards.mean().item(),
                       teacher_nll=sum(t["teacher_nll"] for t in group) / args.group_size,
                       execute_fraction=sum(t["execute_fraction"] for t in group) / args.group_size,
                       response_tokens=sum(len(t["tokens"]) for t in group) / args.group_size,
                       routed_transitions=sum(len(t["actions"]) for t in group))
        print(json.dumps(metrics), flush=True)
        with open(output / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        if step % args.save_every == 0 or step == args.steps:
            print("Saved " + save(policy, reference, optimizer, step, order, cursor, args), flush=True)


if __name__ == "__main__":
    main()
