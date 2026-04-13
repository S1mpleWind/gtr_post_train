# SpecMoD

本 README 主要说明两件事：

1. 如何准备 post-train 需要的 token-level 蒸馏数据（data_prepare）
2. 如何选择并运行不同的 post-train 训练脚本（post_train）

---

## 1. 项目中与本文相关的目录

- `../data_prepare/`
	- `sequenced_tokens_prepare.py`：生成 token-level top-k teacher logits 数据
	- `sequenced_tokens_hidden_prepare.py`：在上面基础上，额外保存 teacher hidden states
	- `evaluate_data.py`：统计生成数据的长度分布

- `post_train/`（Qwen 方向）
	- adaptor-only、adaptor+backbone、backbone-only、单卡/DDP/Ray+DeepSpeed 脚本

- `post_train_llama/`（Llama 方向）
	- 与 Qwen 类似的训练入口，包含 hidden loss 版本

---

## 2. Data Prepare

### 2.1 输入数据格式

`sequenced_tokens_prepare.py` 和 `sequenced_tokens_hidden_prepare.py` 会遍历一个目录下所有 `.jsonl` 文件。

每条样本至少应包含：

- `conversation` 字段（list）

兼容两种 turn 格式：

1. 配对格式：`[{"human": "...", "assistant": "..."}, ...]`
2. 交替格式：`[{"human": "..."}, {"assistant": "..."}, ...]`

脚本会自动提取有效的 `(human, assistant)` 对话对。

### 2.2 输出数据格式（核心）

两份脚本都会输出 `.jsonl`，每行一个训练样本，主要字段：

- `conversation_id`
- `turn`
- `input_ids`
- `assistant_start`
- `token_positions`
- `target_token_ids`
- `teacher_topk_ids`
- `teacher_topk_logits`

其中 `sequenced_tokens_hidden_prepare.py` 还会额外输出：

- `token_hiddens`（teacher 最后一层 hidden，对应监督 token）

### 2.3 运行示例

先进入仓库根目录后执行。

#### A. 生成 top-k logits 蒸馏数据

```bash
python data_prepare/sequenced_tokens_prepare.py \
	--data_dir /path/to/raw_data \
	--model_path /share/public/public_models/Llama-3.1-8B-Instruct \
	--num_samples 5000 \
	--max_length 2048 \
	--top_k 20 \
	--output_path /path/to/processed_data_llama.jsonl
```

#### B. 生成带 hidden 的蒸馏数据

```bash
python data_prepare/sequenced_tokens_hidden_prepare.py \
	--data_dir /path/to/raw_data \
	--model_path /share/public/public_models/Llama-3.1-8B-Instruct \
	--num_samples 500 \
	--max_length 2048 \
	--top_k 20 \
	--output_path /path/to/processed_data_llama3.jsonl
```

#### C. 检查数据长度分布

```bash
python data_prepare/evaluate_data.py
```

---

## 3. Post-Train 脚本怎么选

### 3.1 Qwen 路线（`post_train/`）

#### 1) 只训 adaptor（推荐起步）

- 单卡：`post_train/post_train_adaptor_para.py`
- DDP：`post_train/post_train_adaptor_para_ddp.py`

适用：

- 快速验证数据与 loss 是否正常
- 显存和训练稳定性优先

#### 2) adaptor + backbone 一起训

- 单卡：`post_train/post_training_backbone_adaptor.py`
- DDP：`post_train/post_training_backbone_adaptor_ddp.py`

适用：

- 需要更高上限
- 愿意承担更高算力和调参成本

#### 3) backbone-only

- `post_train/post_train_backbone_only.py`
- `post_train/post_train_backbone_only copy.py`（实验分支版本）

#### 4) Ray + DeepSpeed 全量/大规模

- `post_train/post_train_backbone_full.py`

#### 5) baseline 评估

- `post_train/baseline_ce.py`

用于计算基础 CE 参考值，便于对比训练收益。

### 3.2 Llama 路线（`post_train_llama/`）

- `post_train_llama/post_train_backbone_full.py`
- `post_train_llama/post_train_backbone_only.py`
- `post_train_llama/post_train_backbone_hidden.py`（包含 hidden loss）

---

## 4. 常见训练参数说明

这些参数在多数脚本中含义一致：

- `--train_data_path`：data_prepare 产出的训练 jsonl
- `--router_path`：router 权重路径
- `--eagle_path`：spec model / eagle 模型路径
- `--batch_size`：单卡 batch
- `--max_length`：样本最大长度（会裁剪）
- `--kd_temperature`：KD 温度
- `--ce_weight` / `--kd_weight`：loss 权重
- `--max_distill_tokens_per_sample`：每样本最多参与蒸馏的 token 数
- `--save_dir`：adaptor 保存目录
- `--save_backbone_dir`：backbone 保存目录（如脚本支持）

---

## 5. 最小可跑流程（建议）

1. 先用 `sequenced_tokens_prepare.py` 生成小规模数据（例如 `--num_samples 500`）
2. 跑 `post_train/post_train_adaptor_para.py`（单卡 adaptor-only）
3. 确认 loss 正常下降后，再切到 DDP 或 Ray+DeepSpeed 版本
4. 若需要 hidden 对齐，再改用 `sequenced_tokens_hidden_prepare.py` + 对应 hidden 训练脚本

---

## 6. 注意事项

- 建议先在小数据集上检查：
	- 数据字段完整性
	- loss 是否为有限值
	- 梯度是否异常爆炸
- DDP / DeepSpeed 脚本的 `batch_size` 通常是单卡 batch，实际总 batch 会乘以卡数。
- 某些脚本是实验版本，参数默认值偏研究用途，建议按你当前机器资源重设。
