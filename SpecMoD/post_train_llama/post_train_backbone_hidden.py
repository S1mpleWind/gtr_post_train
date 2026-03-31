"""
使用 Ray + DeepSpeed ZeRO Stage 2 多卡训练
微调 adaptor 以及 backbone 所有层的参数
"""

import sys
import os
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.llama_model_adaptor_global_soft_router import Spec_LlamaForCausalLM
from model.utils import ShadowAdapter3, Global_router
from model.EAGLE_model import Model as SpecModel

from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

import deepspeed


class CombinedModel(nn.Module):
    """
    把 backbone 和所有 adaptor 包装成一个 nn.Module，
    使 DeepSpeed 能建立完整的参数映射表，避免 ZeRO 初始化时 KeyError。
    """
    def __init__(self, backbone, adaptors):
        super().__init__()
        self.backbone = backbone
        for i, a in enumerate(adaptors):
            if a is not None:
                self.add_module(f'adaptor_{i}', a)

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)


class AdaptorTrainDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                if "input_ids" in x and "token_positions" in x:
                    self.data.append(x)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, pad_token_id: int, max_length: int):
    proc = []
    max_len = 0
    for item in batch:
        ids = item["input_ids"]
        pos = item["token_positions"]
        tgt = item["target_token_ids"]
        tk_ids = item["teacher_topk_ids"]
        tk_logits = item["teacher_topk_logits"]
        hiddens = item.get("token_hiddens", [])

        if len(ids) > max_length:
            cut = len(ids) - max_length
            ids = ids[cut:]
            new_pos, new_tgt, new_tk_ids, new_tk_logits, new_hiddens = [], [], [], [], []
            for p, t, ki, kv, h in zip(pos, tgt, tk_ids, tk_logits, hiddens):
                np = p - cut
                if 0 <= np < len(ids):
                    new_pos.append(np)
                    new_tgt.append(t)
                    new_tk_ids.append(ki)
                    new_tk_logits.append(kv)
                    new_hiddens.append(h)
            pos, tgt, tk_ids, tk_logits, hiddens = new_pos, new_tgt, new_tk_ids, new_tk_logits, new_hiddens

        max_len = max(max_len, len(ids))
        proc.append({
            "input_ids": ids,
            "token_positions": pos,
            "target_token_ids": tgt,
            "teacher_topk_ids": tk_ids,
            "teacher_topk_logits": tk_logits,
            "token_hiddens": hiddens,
        })

    input_ids_out = []
    attention_mask_out = []
    for x in proc:
        ids = x["input_ids"]
        pad_len = max_len - len(ids)
        input_ids_out.append(ids + [pad_token_id] * pad_len)
        attention_mask_out.append([1] * len(ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids_out, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_out, dtype=torch.long),
        "token_positions": [x["token_positions"] for x in proc],
        "target_token_ids": [x["target_token_ids"] for x in proc],
        "teacher_topk_ids": [x["teacher_topk_ids"] for x in proc],
        "teacher_topk_logits": [x["teacher_topk_logits"] for x in proc],
        "token_hiddens": [x["token_hiddens"] for x in proc],
    }

def _collect_decode_logits_and_hidden_for_positions_batch(
    model, spec_model, adaptors, router,
    input_ids_2d: torch.Tensor, attention_mask_2d: torch.Tensor, positions_need_batch: list,
):
    B, L = input_ids_2d.shape
    out_logits_map_batch = [{} for _ in range(B)]
    out_hidden_map_batch = [{} for _ in range(B)]
    
    if hasattr(model.model, "input_id_init"):
        model.model.input_id_init()
    elif hasattr(model.model, "input_ids"):
        model.model.input_ids = None
    spec_model.reset_kv()
    
    with torch.no_grad():
        prefill_outputs = model(
            input_ids=input_ids_2d,
            attention_mask=attention_mask_2d,
            use_cache=False,
            adaptor=adaptors,
            router=router,
            spec_model=spec_model,
            last_hidden_state=None,
            output_hidden_states=False,
        )
        prefill_last_hidden = prefill_outputs.last_hidden_state.detach()
    
    if input_ids_2d.size(1) <= 1:
        gate_scores = torch.ones(
            (B, model.config.num_hidden_layers, 1),
            device=input_ids_2d.device,
            dtype=prefill_last_hidden.dtype,
        )
    else:
        hs_for_spec = prefill_last_hidden[:, :-1, :]
        spec_hidden_states = spec_model.topK_generate(
            hidden_states=hs_for_spec,
            input_ids=input_ids_2d,
        )
        emb_for_router = model.get_input_embeddings()(input_ids_2d[:, -1:])
        if spec_hidden_states.size(1) != emb_for_router.size(1):
            emb_for_router = emb_for_router[:, -spec_hidden_states.size(1):, :]
        input_feature = torch.cat([emb_for_router, spec_hidden_states], dim=-1).to(emb_for_router.device)
        raw_gate = router(input_feature)
        if raw_gate.dim() == 2:
            gate_logits = raw_gate
        elif raw_gate.dim() == 3:
            if raw_gate.size(1) == 1:
                gate_logits = raw_gate[:, 0, :]
            elif raw_gate.size(2) == 1:
                gate_logits = raw_gate[:, :, 0]
            else:
                raise ValueError(f"Unexpected router output shape: {tuple(raw_gate.shape)}")
        else:
            raise ValueError(f"Unexpected router output dim: {raw_gate.dim()}, shape={tuple(raw_gate.shape)}")
        gate_scores = (gate_logits > 0.0).to(dtype=emb_for_router.dtype).unsqueeze(-1).detach()
    
    if hasattr(model.model, "input_id_init"):
        model.model.input_id_init()
    elif hasattr(model.model, "input_ids"):
        model.model.input_ids = None
    spec_model.reset_kv()
    
    outputs = model(
        input_ids=input_ids_2d,
        attention_mask=attention_mask_2d,
        use_cache=False,
        adaptor=adaptors,
        router=router,
        spec_model=spec_model,
        last_hidden_state=None,
        output_hidden_states=True,  # 使能隐藏状态输出
        gate_scores_override=gate_scores,
    )
    logits = outputs.logits.float()
    hidden_states = outputs.hidden_states  # tuple of tensors, last one is [B, L, hidden_size]
    
    if hidden_states is not None and len(hidden_states) > 0:
        last_hidden = hidden_states[-1]  # [B, L, hidden_size]
    else:
        last_hidden = None
    
    for b in range(B):
        valid_len = int(attention_mask_2d[b].sum().item())
        for p in positions_need_batch[b]:
            if 0 < p < valid_len:
                out_logits_map_batch[b][p] = logits[b, p - 1]
                if last_hidden is not None:
                    out_hidden_map_batch[b][p] = last_hidden[b, p - 1]  # [hidden_size]
    
    return out_logits_map_batch, out_hidden_map_batch



def _sample_distill_targets_for_batch(batch, attention_mask, max_distill_tokens_per_sample):
    B = attention_mask.size(0)
    sampled_pos_batch, sampled_tgt_batch, sampled_tk_ids_batch, sampled_tk_logits_batch = [], [], [], []
    for b in range(B):
        valid_len = int(attention_mask[b].sum().item())
        pos_list = batch["token_positions"][b]
        tgt_list = batch["target_token_ids"][b]
        tk_ids_list = batch["teacher_topk_ids"][b]
        tk_logits_list = batch["teacher_topk_logits"][b]
        zipped = []
        for p, tgt, tk_ids, tk_logits in zip(pos_list, tgt_list, tk_ids_list, tk_logits_list):
            p = int(p)
            if 0 < p < valid_len:
                zipped.append((p, tgt, tk_ids, tk_logits))
        if len(zipped) > max_distill_tokens_per_sample:
            idx = sorted(random.sample(range(len(zipped)), max_distill_tokens_per_sample))
            zipped = [zipped[i] for i in idx]
        if len(zipped) == 0:
            sampled_pos_batch.append([])
            sampled_tgt_batch.append([])
            sampled_tk_ids_batch.append([])
            sampled_tk_logits_batch.append([])
            continue
        sampled_pos_batch.append([x[0] for x in zipped])
        sampled_tgt_batch.append([x[1] for x in zipped])
        sampled_tk_ids_batch.append([x[2] for x in zipped])
        sampled_tk_logits_batch.append([x[3] for x in zipped])
    return sampled_pos_batch, sampled_tgt_batch, sampled_tk_ids_batch, sampled_tk_logits_batch


def _collect_decode_logits_for_positions_batch(
    model, spec_model, adaptors, router,
    input_ids_2d: torch.Tensor, attention_mask_2d: torch.Tensor, positions_need_batch: list,
):
    B, L = input_ids_2d.shape
    out_map_batch = [{} for _ in range(B)]
    if hasattr(model.model, "input_id_init"):
        model.model.input_id_init()
    elif hasattr(model.model, "input_ids"):
        model.model.input_ids = None
    spec_model.reset_kv()
    with torch.no_grad():
        prefill_outputs = model(
            input_ids=input_ids_2d,
            attention_mask=attention_mask_2d,
            use_cache=False,
            adaptor=adaptors,
            router=router,
            spec_model=spec_model,
            last_hidden_state=None,
            output_hidden_states=False,
        )
        prefill_last_hidden = prefill_outputs.last_hidden_state.detach()
    if input_ids_2d.size(1) <= 1:
        gate_scores = torch.ones(
            (B, model.config.num_hidden_layers, 1),
            device=input_ids_2d.device,
            dtype=prefill_last_hidden.dtype,
        )
    else:
        hs_for_spec = prefill_last_hidden[:, :-1, :]
        spec_hidden_states = spec_model.topK_generate(
            hidden_states=hs_for_spec,
            input_ids=input_ids_2d,
        )
        emb_for_router = model.get_input_embeddings()(input_ids_2d[:, -1:])
        if spec_hidden_states.size(1) != emb_for_router.size(1):
            emb_for_router = emb_for_router[:, -spec_hidden_states.size(1):, :]
        input_feature = torch.cat([emb_for_router, spec_hidden_states], dim=-1).to(emb_for_router.device)
        raw_gate = router(input_feature)
        if raw_gate.dim() == 2:
            gate_logits = raw_gate
        elif raw_gate.dim() == 3:
            if raw_gate.size(1) == 1:
                gate_logits = raw_gate[:, 0, :]
            elif raw_gate.size(2) == 1:
                gate_logits = raw_gate[:, :, 0]
            else:
                raise ValueError(f"Unexpected router output shape: {tuple(raw_gate.shape)}")
        else:
            raise ValueError(f"Unexpected router output dim: {raw_gate.dim()}, shape={tuple(raw_gate.shape)}")
        gate_scores = (gate_logits > 0.0).to(dtype=emb_for_router.dtype).unsqueeze(-1).detach()
    if hasattr(model.model, "input_id_init"):
        model.model.input_id_init()
    elif hasattr(model.model, "input_ids"):
        model.model.input_ids = None
    spec_model.reset_kv()
    outputs = model(
        input_ids=input_ids_2d,
        attention_mask=attention_mask_2d,
        use_cache=False,
        adaptor=adaptors,
        router=router,
        spec_model=spec_model,
        last_hidden_state=None,
        output_hidden_states=False,
        gate_scores_override=gate_scores,
    )
    logits = outputs.logits.float()
    for b in range(B):
        valid_len = int(attention_mask_2d[b].sum().item())
        for p in positions_need_batch[b]:
            if 0 < p < valid_len:
                out_map_batch[b][p] = logits[b, p - 1]
    return out_map_batch

def _compute_hidden_loss(student_hidden, teacher_hidden_tensor, loss_type="norm_mse", eps=1e-6):
    # 统一用 float32 计算，避免 bf16 精度损失影响 loss 稳定性
    s = student_hidden.float()
    t = teacher_hidden_tensor.float()

    if loss_type == "norm_mse":
        s = F.normalize(s, p=2, dim=-1, eps=eps)
        t = F.normalize(t, p=2, dim=-1, eps=eps)
        return F.mse_loss(s, t)

    if loss_type == "cosine":
        s = F.normalize(s, p=2, dim=-1, eps=eps)
        t = F.normalize(t, p=2, dim=-1, eps=eps)
        return 1.0 - F.cosine_similarity(s, t, dim=-1)

    raise ValueError(f"Unsupported hidden loss type: {loss_type}")

def _grad_l2_norm(params):
    total = 0.0
    cnt = 0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            continue
        n = g.norm(2).item()
        total += n * n
        cnt += 1
    return total ** 0.5, cnt

def train_func(config):
    import torch
    import os
    import deepspeed
    from torch.utils.data import DataLoader, Subset
    from torch.utils.tensorboard import SummaryWriter
    from ray import train

    # dis
    import torch.distributed as dist

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    is_main_process = (rank == 0)

    def _sync_any_true(local_flag: bool) -> bool:
        # 任意 rank 为 True，则全局 True
        flag = torch.tensor([1 if local_flag else 0], device=device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    # dataset
    train_dataset = AdaptorTrainDataset(config["train_data_path"])
    if config["max_train_samples"] > 0:
        n = min(config["max_train_samples"], len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(n)))
        if is_main_process:
            print(f"[INFO] 使用数据集: {n} 条样本")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(
            batch=batch,
            pad_token_id=config["pad_token_id"],
            max_length=config["max_length"],
        ),
    )

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    model_path = config["model_path"]
    spec_model_path = config["eagle_path"]

    #ori_model = Spec_Qwen3ForCausalLM.from_pretrained(model_path).float().to(device)
    ori_model = Spec_LlamaForCausalLM.from_pretrained(model_path).bfloat16().to(device)
    
    ori_model.enable_input_require_grads()

    # 冻结全部参数，再解冻 attention / mlp 
    for param in ori_model.parameters():
        param.requires_grad = False
    for name, param in ori_model.named_parameters():
        if "norm" not in name and "lm_head" not in name and "embedding" not in name:
            param.requires_grad = True

    ori_model.train()
    for m in ori_model.modules():
        if isinstance(m, (nn.Dropout, nn.LayerNorm)):
            m.eval()

    spec_model = SpecModel.from_pretrained(
        Spec_model_path=spec_model_path,
        Ori_model_path=model_path,
        dtype=torch.bfloat16,
        #dtype=torch.float
    ).to(device)

    LAYERS = ori_model.config.num_hidden_layers

    router = Global_router(
        input_dim=ori_model.config.hidden_size * 2,
        hidden_dim=1024,
        output_dim=LAYERS,
    ).to(device)
    if os.path.isfile(config["router_path"]):
        router_weight = torch.load(config["router_path"], map_location=device)
        router.load_state_dict(router_weight)
    router = router.bfloat16().eval()
    #router = router.float().eval()

    for p in router.parameters():
        p.requires_grad = False

    # ── 构建 adaptors 列表 ────────────────────────────────────────────────────
    adaptors = [None]   # index 0 占位
    for i in range(1, LAYERS):
        if i == 31:
            adaptors.append(None)
        else:
            layer_adaptor = ShadowAdapter3(ori_model.config.hidden_size, config["adaptor_hidden_dim"])
            adaptor_weight_path = (
                f"/home/xujiaming/xujiaming/models/llama_adaptor/"
                f"adapter_layer_{i}_1024_Model3.pt"
            )
            if os.path.exists(adaptor_weight_path):
                layer_adaptor_weight = torch.load(adaptor_weight_path, map_location=device)
                layer_adaptor.load_state_dict(layer_adaptor_weight)
            #layer_adaptor = layer_adaptor.float().to(device).train()
            layer_adaptor = layer_adaptor.bfloat16().to(device).train()
            adaptors.append(layer_adaptor)

    # 汇总需要优化的参数
    adaptor_params = []
    for a in adaptors:
        if a is not None:
            adaptor_params += list(a.parameters())

    backbone_params = [p for n, p in ori_model.named_parameters() if p.requires_grad]

    # all_params 仅用于梯度统计，不传给 optimizer
    all_params = adaptor_params + backbone_params

    # ── 用 CombinedModel 包装，让 DeepSpeed 能感知所有参数 ────────────────────
    combined_model = CombinedModel(ori_model, adaptors)

    optimizer = optim.AdamW(
        [
            {"params": adaptor_params, "lr": config["adaptor_lr"]},
            {"params": backbone_params, "lr": config["backbone_lr"]},
        ],
        weight_decay=config["weight_decay"],
        eps=1e-4,
    )

    # ── DeepSpeed ZeRO Stage 2 配置 ──────────────────────────────────────────
    ds_config = {
        "train_batch_size": config["batch_size"] * world_size,
        "train_micro_batch_size_per_gpu": config["batch_size"],
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 2,
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1,
    }

    ds_engine, optimizer, _, _ = deepspeed.initialize(
        model=combined_model,
        optimizer=optimizer,
        config=ds_config,
    )

    # 通过 ds_engine.module.backbone 访问原始 backbone
    backbone = ds_engine.module.backbone

    writer = SummaryWriter(log_dir=config["log_dir"]) if (is_main_process and config["use_tensorboard"]) else None

    # ── 训练主循环 
    # TODO：异步退出的问题
    global_step = 0
    for epoch in range(config["num_epochs"]):
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_hidden_mse = 0.0
        step_count = 0

        progress_bar = (
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
            if is_main_process else train_loader
        )

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            T = config["kd_temperature"]
            B = input_ids.size(0)

            sampled_pos_batch, sampled_tgt_batch, sampled_tk_ids_batch, sampled_tk_logits_batch = (
                _sample_distill_targets_for_batch(
                    batch=batch,
                    attention_mask=attention_mask,
                    max_distill_tokens_per_sample=config["max_distill_tokens_per_sample"],
                )
            )

            # if sum(len(x) for x in sampled_pos_batch) == 0:
            #     continue

            #TODO 全局同步
            local_no_targets = (sum(len(x) for x in sampled_pos_batch) == 0)
            if _sync_any_true(local_no_targets):
                continue

            # 获取logits和隐藏状态
            logits_map_batch, hidden_map_batch = _collect_decode_logits_and_hidden_for_positions_batch(
                model=backbone, 
                spec_model=spec_model,
                adaptors=adaptors,
                router=router,
                input_ids_2d=input_ids,
                attention_mask_2d=attention_mask,
                positions_need_batch=sampled_pos_batch,
            )

            # 建立原始位置到隐藏状态的映射
            teacher_hidden_maps = []
            for b in range(B):
                orig_positions = batch["token_positions"][b]
                orig_hiddens = batch["token_hiddens"][b]
                hidden_by_pos = {pos: hiddens for pos, hiddens in zip(orig_positions, orig_hiddens)}
                teacher_hidden_maps.append(hidden_by_pos)

            all_ce_losses = []
            all_kd_losses = []
            all_hidden_losses = []

            for b in range(B):
                logits_map = logits_map_batch[b]
                hidden_map = hidden_map_batch[b]  # key=position, value=hidden_state [hidden_size]
                teacher_hidden_by_pos = teacher_hidden_maps[b]  # key=position, value=hidden_state list
                
                for p, tgt_id, topk_ids, topk_logits in zip(
                    sampled_pos_batch[b], sampled_tgt_batch[b],
                    sampled_tk_ids_batch[b], sampled_tk_logits_batch[b],
                ):
                    if p not in logits_map:
                        continue
                    s = logits_map[p].float()
                    if not torch.isfinite(s).all():
                        if is_main_process:
                            print("infinite s")
                        continue
                    s = torch.nan_to_num(s, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)

                    ce = F.cross_entropy(
                        s.unsqueeze(0),
                        torch.tensor([tgt_id], device=device, dtype=torch.long),
                        reduction="mean",
                    )
                    if torch.isfinite(ce):
                        all_ce_losses.append(ce)

                    if len(topk_ids) > 0 and len(topk_logits) > 0:
                        ids_t = torch.tensor(topk_ids, device=device, dtype=torch.long)
                        tlog_t = torch.tensor(topk_logits, device=device, dtype=torch.float32)
                        tlog_t = torch.nan_to_num(tlog_t, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-30, 30)
                        s_topk = s.index_select(0, ids_t).float()
                        s_topk = torch.nan_to_num(s_topk, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30, 30)

                        s_topk = s_topk - s_topk.max(dim=-1, keepdim=True)[0].detach()

                        log_p = F.log_softmax((s_topk / T).unsqueeze(0), dim=-1)
                        with torch.no_grad():
                            q = F.softmax((tlog_t / T).unsqueeze(0), dim=-1)

                        kd = F.kl_div(log_p, q, reduction="batchmean") * (T * T)
                        kd = torch.clamp(kd, max=15.0)

                        eps = 1e-8
                        token_entropy = -(q * torch.log(q.clamp_min(eps))).sum(dim=-1)
                        batch_entropy = token_entropy.mean()
                        max_entropy = torch.log(torch.tensor(float(q.size(-1)), device=q.device))
                        kd_weight = torch.clamp(
                            batch_entropy / (max_entropy + 1e-8), min=0.05, max=1.0
                        ).detach()

                        if torch.isfinite(kd * kd_weight):
                            all_kd_losses.append(kd * kd_weight)

                    # hidden loss（支持归一化MSE或Cosine）
                    if p in teacher_hidden_by_pos and len(teacher_hidden_by_pos[p]) > 0 and p in hidden_map:
                        student_hidden = hidden_map[p]  # [hidden_size]
                        teacher_hidden_tensor = torch.tensor(
                            teacher_hidden_by_pos[p],
                            device=device,
                            dtype=torch.float32,
                        )

                        if student_hidden.shape[0] == teacher_hidden_tensor.shape[0]:
                            hidden_loss = _compute_hidden_loss(
                                student_hidden=student_hidden,
                                teacher_hidden_tensor=teacher_hidden_tensor,
                                loss_type=config["hidden_loss_type"],
                                eps=config["hidden_norm_eps"],
                            )
                            if torch.isfinite(hidden_loss).all():
                                all_hidden_losses.append(hidden_loss)

            # TODO
            local_no_loss = (len(all_ce_losses) == 0 and len(all_kd_losses) == 0 and len(all_hidden_losses) == 0)
            if _sync_any_true(local_no_loss):
                continue

            loss_parts = []
            batch_ce_sum = 0.0
            batch_kd_sum = 0.0
            batch_hidden_sum = 0.0

            if len(all_ce_losses) > 0:
                ce_loss = torch.stack(all_ce_losses).mean()
                loss_parts.append(config["ce_weight"] * ce_loss)
                batch_ce_sum = ce_loss.item()

            if len(all_kd_losses) > 0:
                kd_loss = torch.stack(all_kd_losses).mean()
                loss_parts.append(config["kd_weight"] * kd_loss)
                batch_kd_sum = kd_loss.item()

            if len(all_hidden_losses) > 0:
                hidden_loss = torch.stack(all_hidden_losses).mean()
                loss_parts.append(config["hidden_mse_weight"] * hidden_loss)
                batch_hidden_sum = hidden_loss.item()

            batch_loss = sum(loss_parts)

            local_bad_loss = (not torch.isfinite(batch_loss))
            if _sync_any_true(local_bad_loss):
                continue

            ds_engine.backward(batch_loss)

            ce_token_cnt = len(all_ce_losses)
            kd_token_cnt = len(all_kd_losses)
            hidden_token_cnt = len(all_hidden_losses)
            distill_token_cnt = max(ce_token_cnt, kd_token_cnt)
            grad_pre, grad_param_cnt = _grad_l2_norm(all_params)

            if is_main_process and config["debug_grad_stats"] and (global_step % config["debug_grad_interval"] == 0):
                print(
                    f"[GRAD-STAT pre] step={global_step} "
                    f"distill_tokens={distill_token_cnt} ce_tokens={ce_token_cnt} kd_tokens={kd_token_cnt} "
                    f"hidden_tokens={hidden_token_cnt} "
                    f"batch_loss={batch_loss.item():.6f} ce_mean={batch_ce_sum:.6f} kd_mean={batch_kd_sum:.6f} "
                    f"hidden_mean={batch_hidden_sum:.6f} "
                    f"grad_pre={grad_pre:.6f} grad_param_cnt={grad_param_cnt}"
                )

            ds_engine.step()

            total_loss = batch_loss.item()
            total_ce = batch_ce_sum
            total_kd = batch_kd_sum
            total_hidden_mse = batch_hidden_sum
            step_count += 1
            global_step += 1

            if is_main_process and writer is not None and (global_step % config["log_interval"] == 0):
                writer.add_scalar("train/loss_step", total_loss, global_step)
                writer.add_scalar("train/ce_step", total_ce, global_step)
                writer.add_scalar("train/kd_step", total_kd, global_step)
                writer.add_scalar("train/hidden_mse_step", total_hidden_mse, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            if is_main_process and hasattr(progress_bar, "set_postfix"):
                progress_bar.set_postfix(
                    {
                        "step": global_step,
                        "loss": f"{total_loss:.4f}",
                        "ce": f"{total_ce:.4f}",
                        "kd": f"{total_kd:.4f}",
                        "hidden": f"{total_hidden_mse:.3e}",
                    }
                )

        if is_main_process:
            print(f"\nEpoch {epoch+1} 完成")

    # ── 保存（只在主进程） ────────────────────────────────────────────────────
    if is_main_process:
        os.makedirs(config["save_dir"], exist_ok=True)
        os.makedirs(config["save_backbone_dir"], exist_ok=True)
        for i, adaptor in enumerate(adaptors):
            if adaptor is not None:
                save_path = f"{config['save_dir']}/adapter_layer_{i}_{config['adaptor_hidden_dim']}_final.pt"
                torch.save(adaptor.state_dict(), save_path)
        torch.save(
            backbone.state_dict(),
            os.path.join(config["save_backbone_dir"], "backbone_final.pt"),
        )
        if writer is not None:
            writer.close()
        print("\n训练完成！")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str,
                        default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/data_prepare/processed_data_llama3.jsonl")
    parser.add_argument("--router_path", type=str,
                        default="/home/xujiaming/xujiaming/models/Llama3.1_8B_global_router_1024_Model1_non_thinking.pt")
    parser.add_argument("--eagle_path", type=str,
                        default="/home/xujiaming/xujiaming/models/EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--adaptor_hidden_dim", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--backbone_lr", type=float, default=4e-5)
    parser.add_argument("--adaptor_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.02)

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--kd_temperature", type=float, default=1)
    parser.add_argument("--ce_weight", type=float, default=0.03)
    parser.add_argument("--kd_weight", type=float, default=0.3)
    parser.add_argument("--hidden_mse_weight", type=float, default=2)

    parser.add_argument("--hidden_loss_type", type=str, default="cosine",
                    choices=["norm_mse", "cosine"])
    parser.add_argument("--hidden_norm_eps", type=float, default=1e-6)

    parser.add_argument("--max_train_samples", type=int, default=1200)
    parser.add_argument("--max_distill_tokens_per_sample", type=int, default=64)
    parser.add_argument("--use_tensorboard", type=bool, default=True)
    parser.add_argument("--log_dir", type=str, default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/runs/llama_post_train_h1")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_dir", type=str,
                        default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/checkpoint/llama_adaptor_h1")
    parser.add_argument("--save_backbone_dir", type=str,
                        default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/checkpoint/llama_backbone_h1")
    parser.add_argument("--debug_grad_stats", type=bool, default=False)
    parser.add_argument("--debug_grad_interval", type=int, default=1)
    parser.add_argument("--printWarn", default=False)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("/share/public/public_models/Llama-3.1-8B-Instruct")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    scaling_config = ScalingConfig(
        num_workers=6,
        use_gpu=True
    )

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "model_path": "/share/public/public_models/Llama-3.1-8B-Instruct",
            "eagle_path": args.eagle_path,
            "router_path": args.router_path,
            "adaptor_hidden_dim": args.adaptor_hidden_dim,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "backbone_lr": args.backbone_lr,
            "adaptor_lr": args.adaptor_lr,
            "weight_decay": args.weight_decay,
            "hidden_loss_type": args.hidden_loss_type,
            "hidden_norm_eps": args.hidden_norm_eps,
            "num_epochs": args.num_epochs,
            "kd_temperature": args.kd_temperature,
            "ce_weight": args.ce_weight,
            "kd_weight": args.kd_weight,
            "hidden_mse_weight": args.hidden_mse_weight,
            "max_train_samples": args.max_train_samples,
            "max_distill_tokens_per_sample": args.max_distill_tokens_per_sample,
            "use_tensorboard": args.use_tensorboard,
            "log_dir": args.log_dir,
            "log_interval": args.log_interval,
            "save_dir": args.save_dir,
            "save_backbone_dir": args.save_backbone_dir,
            "debug_grad_stats": args.debug_grad_stats,
            "debug_grad_interval": args.debug_grad_interval,
            "printWarn": args.printWarn,
            "pad_token_id": pad_token_id,
            "train_data_path": args.train_data_path,
        },
        scaling_config=scaling_config,
    )
    result = trainer.fit()