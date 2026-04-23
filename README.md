# Readme

本 README 主要说明两部分代码

1. 准备 post-train 需要的 token-level 蒸馏数据（data_prepare）
2. post-train 训练脚本

---

## 1. 项目中与本文相关的目录

- `../data_prepare/`
	- `sequenced_tokens_prepare.py`：生成 token-level top-k teacher logits 数据
	- `sequenced_tokens_hidden_prepare.py`：在上面基础上，额外保存 teacher hidden states
	- `evaluate_data.py`：统计生成数据的长度分布

- `post_train_qwen/`（Qwen 方向）
	- adaptor-only、adaptor+part backbone、backbone-only的单卡/DDP脚本
	- adaptor+full backbone Ray+DeepSpeed 脚本

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
python data_prepare/evaluate_data.py --data_path
```

---

## 3. Post-Train 

### 3.1 以 Llama 为例（`post_train_llama/`）

- `post_train_llama/post_train_backbone_full.py`
	- 训练adaptor + full—backbone
- `post_train_llama/post_train_backbone_only.py`
	- 不使用adaptot只训练backbone
- `post_train_llama/post_train_backbone_hidden.py`
	- adaptor + backbone ， 同时包含 hidden loss


### 3.2 对应改动的pipeline
- 修改的pipeline
 - `SpecMoD/model/llama_model_adaptor_global_soft_router.py`
 - `SpecMoD/model/qwen3_model_adaptor_global_soft_router_pipeline.py`

- 新增的
 - `SpecMoD/model/llama_model_global_router.py`
 - `SpecMoD/model/llama_model_global_soft_router.py`
 - `SpecMoD/model/qwen3_model_global_router_pipeline.py`
 - `SpecMoD/model/qwen3_model_global_soft_router_pipeline.py`

### 3.3 代码的一些逻辑
采用**两次 prefill**来进行训练
1. 第一次prefill获取所有的 hidden
2. 清空 self.input_ids，准备第二次prefill； 用得到的 hidden 去计算 router 的结果 gates
3. 通过改进的pipeline的后门，传入router的结果并用 gates 实现 prefill 的跳层
4. 得到每个位置经过router以及adaptor干涉之后的结果，反向传播
