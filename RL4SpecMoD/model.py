"""Explicit decode driver for the repository's hard-router Qwen3 pipeline."""
import importlib
import importlib.util
from dataclasses import dataclass

import torch
from torch.distributions import Bernoulli, Categorical
from transformers.cache_utils import DynamicCache

from SpecMoD.model.EAGLE_model import Model as Eagle
from SpecMoD.model.utils import Global_router, ShadowAdapter3


def model_class():
    # Support both the current working-tree reorganization and committed layout.
    root = "SpecMoD.model."
    nested = root + "qwen3.qwen3_model_adaptor_global_router_pipeline"
    if importlib.util.find_spec(root + "qwen3") is not None:
        return importlib.import_module(nested).Spec_Qwen3ForCausalLM
    return importlib.import_module(
        root + "qwen3_model_adaptor_global_router_pipeline"
    ).Spec_Qwen3ForCausalLM


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


@dataclass
class State:
    ids: torch.Tensor
    cache: DynamicCache
    features: torch.Tensor


class DecodePolicy:
    def __init__(self, args):
        dtype = getattr(torch, args.dtype)
        self.target = model_class().from_pretrained(
            args.model, torch_dtype=dtype, attn_implementation="sdpa"
        ).to(args.device).eval().requires_grad_(False)
        if args.backbone:
            # Backbone snapshots must match exactly; do not silently ignore weights.
            self.target.load_state_dict(load_state(args.backbone), strict=True)
        config = self.target.config
        self.depth = config.num_hidden_layers
        self.forced = sorted(set(args.forced_layers or
                                 [0, 1, 2, self.depth - 2, self.depth - 1]))
        if any(i < 0 or i >= self.depth for i in self.forced):
            raise ValueError("forced layer index out of range")
        self.free = [i for i in range(self.depth) if i not in self.forced]
        if not self.free:
            raise ValueError("At least one layer must be routable")
        self.router = Global_router(config.hidden_size * 2, args.router_dim, self.depth)
        self.router.load_state_dict(load_state(args.router))
        # Keep optimizer/master parameters fp32, including for fp16 base models.
        self.router = self.router.to(args.device).float().eval()
        self.adaptors = torch.nn.ModuleDict()
        for i in self.free:
            module = ShadowAdapter3(config.hidden_size, args.adaptor_dim)
            path = args.adaptor_pattern.format(layer=i, dim=args.adaptor_dim)
            module.load_state_dict(load_state(path))
            self.adaptors[str(i)] = module.to(args.device).float().eval()
        self.adaptors.requires_grad_(args.train_adaptors)
        self.eagle = Eagle.from_pretrained(
            Spec_model_path=args.eagle, Ori_model_path=args.model,
            dtype=dtype, device=torch.device(args.device),
        ).eval().requires_grad_(False)
        self.token_temperature = args.token_temperature
        self.route_temperature = args.route_temperature
        self.train_adaptors = args.train_adaptors

    def parameters(self):
        return [p for module in (self.router, self.adaptors)
                for p in module.parameters() if p.requires_grad]

    def layers(self, ids, cache, execute):
        """Use original decoder layers; flag=False still projects and appends K/V."""
        model = self.target.model
        hidden = model.embed_tokens(ids)
        position = torch.arange(cache.get_seq_length(),
                                cache.get_seq_length() + ids.shape[1], device=ids.device)
        position_ids = position.unsqueeze(0)
        mask = model._update_causal_mask(None, hidden, position, cache, False)
        rotary = model.rotary_emb(hidden, position_ids)
        taps = []
        # These are pre-layer states, matching the three EAGLE taps in Qwen3Model.forward.
        indices = {2, self.depth // 2, self.depth - 3}
        for i, layer in enumerate(model.layers):
            if i in indices:
                taps.append(hidden)
            run = bool(execute[i])
            if run:
                hidden = layer(
                    hidden, attention_mask=mask, position_ids=position_ids,
                    past_key_value=cache, use_cache=True, output_attentions=False,
                    cache_position=position, position_embeddings=rotary, flag=True,
                )[0]
            else:
                # Match the original skip branch while keeping adaptor master weights fp32.
                layer.self_attn(
                    hidden_states=layer.input_layernorm(hidden),
                    attention_mask=mask, position_ids=position_ids,
                    past_key_value=cache, cache_position=position,
                    position_embeddings=rotary, flag=False,
                )
                hidden = hidden + self.adaptors[str(i)](hidden.float()).to(hidden.dtype)
        features = torch.cat(taps, dim=-1)
        logits = self.target.lm_head(model.norm(hidden[:, -1:]))[:, -1].float()
        return logits, features

    @torch.no_grad()
    def prefill(self, prompt):
        self.eagle.reset_kv()
        cache = DynamicCache()
        logits, features = self.layers(prompt, cache, [True] * self.depth)
        return State(prompt, cache, features), logits

    def token_dist(self, logits):
        return Categorical(logits=logits / self.token_temperature)

    def step(self, state, previous_token, action=None, greedy=False):
        """One routed decode transition: observe -> route -> layers -> next-token logits."""
        token = previous_token.reshape(1, 1)
        state.ids = torch.cat((state.ids, token), dim=1)
        with torch.no_grad():
            eagle_hidden = self.eagle.topK_generate(state.features, state.ids)
            embedding = self.target.model.embed_tokens(token)
            observation = torch.cat((embedding, eagle_hidden), dim=-1).detach().float()
        route_logits = self.router(observation)[0, 0] / self.route_temperature
        distribution = Bernoulli(logits=route_logits[self.free])
        if action is None:
            action = (distribution.probs >= .5).float() if greedy else distribution.sample()
        else:
            action = action.to(route_logits.device)
        log_prob = distribution.log_prob(action).sum()
        execute = torch.ones(self.depth, dtype=torch.bool, device=route_logits.device)
        execute[self.free] = action.bool()
        logits, features = self.layers(token, state.cache, execute)
        state.features = features.detach()
        return logits, action, log_prob, distribution, observation, execute

    @staticmethod
    def detach_cache(state):
        # transformers==4.51.1 DynamicCache, as pinned by this repository.
        state.cache.key_cache = [x.detach() for x in state.cache.key_cache]
        state.cache.value_cache = [x.detach() for x in state.cache.value_cache]
        state.features = state.features.detach()

    @torch.no_grad()
    def teacher_nll(self, prompt, tokens):
        """Frozen full-depth teacher on the *sampled* response, with incremental KV."""
        cache = DynamicCache()
        logits, _ = self.layers(prompt, cache, [True] * self.depth)
        losses = []
        for i, token in enumerate(tokens):
            losses.append(-logits.log_softmax(-1)[0, token].item())
            if i + 1 < len(tokens):
                logits, _ = self.layers(token.reshape(1, 1), cache, [True] * self.depth)
        return sum(losses) / len(losses)
