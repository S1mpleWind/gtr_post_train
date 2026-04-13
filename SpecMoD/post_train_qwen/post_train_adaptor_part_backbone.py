"""
端到端训练 Adaptor（一次性前向 batch input_ids）
- 使用 soft router pipeline
- 每个 batch 只做一次模型前向
- 从 logits[b, p-1] 取监督位置 p 的预测分布
- 单卡版
- 同时微调 backbone 和 adaptor
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

# 添加父目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.qwen3_model_adaptor_global_soft_router_pipeline import Spec_Qwen3ForCausalLM
from model.utils import ShadowAdapter3, Global_router
from model.EAGLE_model import Model as SpecModel

from transformers import get_linear_schedule_with_warmup

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
        if len(ids) > max_length:
            cut = len(ids) - max_length
            ids = ids[cut:]
            new_pos, new_tgt, new_tk_ids, new_tk_logits = [], [], [], []
            for p, t, ki, kv in zip(pos, tgt, tk_ids, tk_logits):
                np = p - cut
                if 0 <= np < len(ids):
                    new_pos.append(np)
                    new_tgt.append(t)
                    new_tk_ids.append(ki)
                    new_tk_logits.append(kv)
            pos, tgt, tk_ids, tk_logits = new_pos, new_tgt, new_tk_ids, new_tk_logits
        max_len = max(max_len, len(ids))
        proc.append({
            "input_ids": ids,
            "token_positions": pos,
            "target_token_ids": tgt,
            "teacher_topk_ids": tk_ids,
            "teacher_topk_logits": tk_logits,
        })
    input_ids = []
    attention_mask = []
    for x in proc:
        ids = x["input_ids"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "token_positions": [x["token_positions"] for x in proc],
        "target_token_ids": [x["target_token_ids"] for x in proc],
        "teacher_topk_ids": [x["teacher_topk_ids"] for x in proc],
        "teacher_topk_logits": [x["teacher_topk_logits"] for x in proc],
    }

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
    model.model.input_id_init()
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
    model.model.input_id_init()
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

def train_adaptors_end_to_end(args):
    total_success_backprop = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir) if args.use_tensorboard else None

    model_path = "/share/public/public_models/Qwen3-8B"
    spec_model_path = args.eagle_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    ori_model = Spec_Qwen3ForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
    ).bfloat16().to(device)

    ori_model.enable_input_require_grads()

    # 1. 把所有主干参数全部冻结
    for param in ori_model.parameters():
        param.requires_grad = False
    
    # 2. 解冻：只解冻forced_layer
    # num_layers = ori_model.config.num_hidden_layers
    # n_tune = 4
    for name, param in ori_model.named_parameters():
        if any([f"layers.{j}." in name for j in [0,1,34,35]]):
            if "norm" not in name and "emb" not in name and "lm" not in name: 
                param.requires_grad = True

    ori_model.train()
    for m in ori_model.modules():
        if isinstance(m, (nn.Dropout, nn.LayerNorm)):
            m.eval()

    spec_model = SpecModel.from_pretrained(
        Spec_model_path=spec_model_path,
        Ori_model_path=model_path,
        dtype=torch.bfloat16,
    ).to(device)

    LAYERS = ori_model.config.num_hidden_layers
    router = Global_router(
        input_dim=ori_model.config.hidden_size * 2,
        hidden_dim=1024,
        output_dim=LAYERS,
    ).to(device)

    if os.path.isfile(args.router_path):
        print(f"using {args.router_path}")
        router_weight = torch.load(args.router_path, map_location=device)
        router.load_state_dict(router_weight)

    router = router.bfloat16()
    router.eval()
    for p in router.parameters():
        p.requires_grad = False

    adaptors = [None]
    all_params = []

    for i in range(1, LAYERS):
        if i == 34:
            adaptors.append(None)
        else:
            layer_adaptor = ShadowAdapter3(ori_model.config.hidden_size, args.adaptor_hidden_dim)
            adaptor_weight_path = f"/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/final_model/adaptor/8B/adapter_layer_{i}_1024_Model3_0.95.pt"
            if os.path.exists(adaptor_weight_path):
                layer_adaptor_weight = torch.load(adaptor_weight_path, map_location=device)
                layer_adaptor.load_state_dict(layer_adaptor_weight)
            layer_adaptor = layer_adaptor.bfloat16().to(device)
            layer_adaptor.train()
            adaptors.append(layer_adaptor)
            all_params += list(layer_adaptor.parameters())
    all_params += list(ori_model.parameters())

    adaptor_params = []
    for i, adaptor in enumerate(adaptors):
        if adaptor is not None:
            adaptor_params += list(adaptor.parameters())
    backbone_params = [p for n, p in ori_model.named_parameters() if p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": adaptor_params, "lr": args.adaptor_lr},
            {"params": backbone_params, "lr": args.backbone_lr},
        ],
        weight_decay=args.weight_decay,
        eps=1e-4,
    )


    train_dataset = AdaptorTrainDataset(args.train_data_path)
    if args.max_train_samples is not None and args.max_train_samples > 0:
        n = min(args.max_train_samples, len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(n)))
        print(f"[INFO] 使用小数据集: {n} 条样本")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(
            batch=batch,
            pad_token_id=pad_token_id,
            max_length=args.max_length,
        ),
    )


    global_step = 0
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        step_count = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            T = args.kd_temperature
            B = input_ids.size(0)
            sampled_pos_batch, sampled_tgt_batch, sampled_tk_ids_batch, sampled_tk_logits_batch = _sample_distill_targets_for_batch(
                batch=batch,
                attention_mask=attention_mask,
                max_distill_tokens_per_sample=args.max_distill_tokens_per_sample,
            )
            if sum(len(x) for x in sampled_pos_batch) == 0:
                continue
            logits_map_batch = _collect_decode_logits_for_positions_batch(
                model=ori_model,
                spec_model=spec_model,
                adaptors=adaptors,
                router=router,
                input_ids_2d=input_ids,
                attention_mask_2d=attention_mask,
                positions_need_batch=sampled_pos_batch,
            )
            all_ce_losses = []
            all_kd_losses = []
            for b in range(B):
                logits_map = logits_map_batch[b]
                pos_list = sampled_pos_batch[b]
                tgt_list = sampled_tgt_batch[b]
                tk_ids_list = sampled_tk_ids_batch[b]
                tk_logits_list = sampled_tk_logits_batch[b]
                for p, tgt_id, topk_ids, topk_logits in zip(pos_list, tgt_list, tk_ids_list, tk_logits_list):
                    if p not in logits_map:
                        continue
                    s = logits_map[p].float()
                    if not torch.isfinite(s).all():
                        print("infinite s")
                        continue
                    
                    # s = torch.nan_to_num(s, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
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
                        # tlog_t = torch.nan_to_num(tlog_t, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-30, 30)
                        s_topk = s.index_select(0, ids_t).float()
                        # s_topk = torch.nan_to_num(s_topk, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30, 30)

                        #TODO fixing explosion by this
                        s_topk = s_topk - s_topk.max(dim=-1, keepdim=True)[0].detach()

                        log_p = F.log_softmax((s_topk / T).unsqueeze(0), dim=-1)

                        with torch.no_grad():
                            q = F.softmax((tlog_t / T).unsqueeze(0), dim=-1)
                        kd = F.kl_div(log_p, q, reduction="batchmean") * (T * T)
                        kd = torch.clamp(kd, max=17.0)

                        eps = 1e-8
                        token_entropy = -(q * torch.log(q.clamp_min(eps))).sum(dim=-1)
                        batch_entropy = token_entropy.mean()
                        max_entropy = torch.log(torch.tensor(float(q.size(-1)), device=q.device))
                        kd_weight = torch.clamp(batch_entropy / (max_entropy + 1e-8), min=0.05, max=1.0).detach()

                        if torch.isfinite(kd*kd_weight):
                            all_kd_losses.append(kd*kd_weight)
            if len(all_ce_losses) == 0 and len(all_kd_losses) == 0:
                optimizer.zero_grad(set_to_none=True)
                continue
            loss_parts = []
            batch_ce_sum = 0.0
            batch_kd_sum = 0.0

            if len(all_ce_losses) > 0:
                ce_loss = torch.stack(all_ce_losses).mean()
                loss_parts.append(args.ce_weight * ce_loss)
                batch_ce_sum = ce_loss.item()

            if len(all_kd_losses) > 0:
                kd_loss = torch.stack(all_kd_losses).mean()
                loss_parts.append(args.kd_weight * kd_loss)
                batch_kd_sum = kd_loss.item()
            batch_loss = sum(loss_parts)

            if not torch.isfinite(batch_loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            
            batch_loss.backward()
            ce_token_cnt = len(all_ce_losses)
            kd_token_cnt = len(all_kd_losses)
            distill_token_cnt = max(ce_token_cnt, kd_token_cnt)
            grad_pre, grad_param_cnt = _grad_l2_norm(all_params)

            if args.debug_grad_stats and (global_step % args.debug_grad_interval == 0):
                print(
                    f"[GRAD-STAT pre] step={global_step} "
                    f"distill_tokens={distill_token_cnt} ce_tokens={ce_token_cnt} kd_tokens={kd_token_cnt} "
                    f"batch_loss={batch_loss.item():.6f} ce_mean={batch_ce_sum:.6f} kd_mean={batch_kd_sum:.6f} "
                    f"grad_pre={grad_pre:.6f} grad_param_cnt={grad_param_cnt}"
                )

            if grad_pre > 1000 or not torch.isfinite(torch.tensor(grad_pre)):
                if args.printWarn:
                    print(f"[CRITICAL] Grad too high ({grad_pre:.2f}), rejecting this batch to protect weights.")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.9)

            bad_grad = any((p.grad is not None) and (not torch.isfinite(p.grad).all()) for p in all_params)
            if bad_grad:
                if args.printWarn:
                    print("[WARN] non-finite grad, skip step")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            optimizer.step()
            # scheduler.step()
            total_success_backprop += 1
            with torch.no_grad():
                for i, adaptor in enumerate(adaptors):
                    if adaptor is None:
                        continue
                    for name, param in adaptor.named_parameters():
                        if param.abs().max() > 10.0:
                            print(f"[FATAL] 第 {i} 层参数 {name} 已越界，最大值: {param.abs().max().item()}")
                            param.data.clamp_(-5.0, 5.0)
            valid_sample_cnt = B
            batch_loss_sum = batch_loss.item()
            loss_for_log = batch_loss_sum / max(1, valid_sample_cnt)
            ce_for_log = batch_ce_sum / max(1, valid_sample_cnt)
            kd_for_log = batch_kd_sum / max(1, valid_sample_cnt)
            total_loss += loss_for_log
            total_ce += ce_for_log
            total_kd += kd_for_log
            step_count += 1
            global_step += 1
            if writer is not None and (global_step % args.log_interval == 0):
                writer.add_scalar("train/loss_step", loss_for_log, global_step)
                writer.add_scalar("train/ce_step", ce_for_log, global_step)
                writer.add_scalar("train/kd_step", kd_for_log, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            progress_bar.set_postfix(
                {
                    "step": global_step,
                    "loss": f"{loss_for_log:.4f}",
                    "ce": f"{ce_for_log:.4f}",
                    "kd": f"{kd_for_log:.4f}",
                    "avg_loss": f"{(total_loss / max(1, step_count)):.4f}",
                }
            )
        avg_loss = total_loss / max(1, step_count)
        avg_ce = total_ce / max(1, step_count)
        avg_kd = total_kd / max(1, step_count)
        print(f"\nEpoch {epoch+1} 平均损失: {avg_loss:.4f}")
        if writer is not None:
            writer.add_scalar("train/loss_epoch", avg_loss, epoch + 1)
            writer.add_scalar("train/ce_epoch", avg_ce, epoch + 1)
            writer.add_scalar("train/kd_epoch", avg_kd, epoch + 1)
        if (epoch + 1) % args.save_interval == 0:
            for i, adaptor in enumerate(adaptors):
                if adaptor is not None:
                    save_path = f"{args.save_dir}/adapter_layer_{i}_{args.adaptor_hidden_dim}_epoch{epoch+1}.pt"
                    torch.save(adaptor.state_dict(), save_path)
            print(f"✓ 保存到: {args.save_dir}")
            backbone_save_path = os.path.join(args.save_backbone_dir, f"backbone_final.pt")
            torch.save(ori_model.state_dict(), backbone_save_path)
            print(f"Backbone saved to: {backbone_save_path}")
    for i, adaptor in enumerate(adaptors):
        if adaptor is not None:
            save_path = f"{args.save_dir}/adapter_layer_{i}_{args.adaptor_hidden_dim}_final.pt"
            torch.save(adaptor.state_dict(), save_path)
    if writer is not None:
        writer.close()
    print("\n训练完成！")
    print(f"成功的反向{total_success_backprop}")

def main(args):
    train_adaptors_end_to_end(args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/data_prepare/processed_data_long.jsonl")
    parser.add_argument("--router_path", type=str, default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/final_model/global_router/global_router_1024_Model1_non_thinking_first.pt")
    parser.add_argument("--eagle_path", type=str, default="/home/xujiaming/xujiaming/models/Qwen3-8B_eagle3")

    parser.add_argument("--adaptor_hidden_dim", type=int, default=1024)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--adaptor_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--ce_weight", type=float, default=0.05)
    parser.add_argument("--kd_weight", type=float, default=0.8)

    parser.add_argument("--max_train_samples", type=int, default=4000)
    parser.add_argument("--max_distill_tokens_per_sample", type=int, default=256)

    parser.add_argument("--use_tensorboard", type=bool, default=True)
    parser.add_argument("--log_dir", type=str, default="./runs/adaptor_backbone_forced_post_train")
    parser.add_argument("--log_interval", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/checkpoint/adaptor_with_backbone_forced")
    parser.add_argument("--save_backbone_dir", type=str, default="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD/checkpoint/backbone")
    parser.add_argument("--save_interval", type=int, default=1)

    parser.add_argument("--debug_compare", default=False)
    parser.add_argument("--debug_compare_tokens", type=int, default=3)
    
    parser.add_argument("--debug_grad_stats", type=bool, default=True)
    parser.add_argument("--debug_grad_interval", type=int, default=1)
    parser.add_argument("--printWarn", default=False)
    args = parser.parse_args()
    main(args)