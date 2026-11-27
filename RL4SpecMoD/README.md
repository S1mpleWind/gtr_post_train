# SpecMoD Qwen3 decode RL

这部分直接沿用本仓库 **Qwen3 + EAGLE3 + global hard router + ShadowAdapter3** 的推理路径，对实际自回归生成进行在线 RL。默认更新 router 和 adaptor，冻结 Qwen3 backbone、EAGLE、初始 reference router。支持 `--no-train-adaptors` 仅训练 router。

## 与现有 Qwen3 流程的对应关系

阅读入口为 `SpecMoD/inference/qwen/inference_w_adaptor_w_global_router_temp.py` 和 `SpecMoD/model/qwen3/qwen3_model_adaptor_global_router_pipeline.py`。兼容 Git 中旧的 `SpecMoD/model/qwen3_model_adaptor_global_router_pipeline.py` 布局，无需提交现有重构。

1. Prefill 全层执行，建立各层 KV，产生第一个 token；不对 prefill 路由。
2. 收集进入第 2、`L//2`、`L-3` 层之前的 hidden state，按层序拼接，作为 EAGLE3 特征。首轮包含整个 prompt，之后为当前 decode token。
3. EAGLE `topK_generate` 接收上述特征和累计 token IDs；内部丢弃第一个 ID，以其独立 `stable_kv` 对齐增量输入。
4. 拼接当前 token embedding 与 EAGLE 输出，送入 `Global_router`。现有推理以 sigmoid > 0.5 选层；RL 中改为独立 Bernoulli 采样，以获得路径 log probability。
5. 执行层照常计算；跳层仍进行 layernorm、Q/K/V 投影、RoPE、KV append，省去 attention 核心与 MLP，再应用 `x + adaptor(x)`。
6. 默认强制执行 `[0,1,2,L-2,L-1]`，8B 的 36 层即原代码 `[0,1,2,34,35]`。可用 `--forced-layers` 显式覆盖；强制层不参与采样、路径 log probability 或 router KL。

`model.py` 直接调用原 decoder layer、attention、rotary 和 causal mask，没有修改现有模型，也无需全局 monkey-patch `GenerationMixin`。所有序列 batch size 为 1，适配原实现的动态路径；group 按序生成。

## 训练目标

每次从同一 prompt 独立生成 `group-size` 条完整 response，支持 EOS 和长度上限。每个 decode transition 的联合动作是「执行层 mask + 下一个 token」，其 log probability 为自由层 Bernoulli log probabilities 之和加 token log probability。第一个 token 来自冻结的完整 prefill，不计入优化动作。

默认整段奖励为：

```
quality = -mean(full_depth_teacher_NLL(sampled_response))
reward = quality_weight * quality + compute_weight * (1 - execute_fraction)
```

Teacher 是同一个冻结 backbone 的完整层路径，独立 KV，在 **rollout 实际生成的前缀** 上逐 token 计分；不需要额外加载一个 8B teacher。NLL 使用原始 logits，不用采样温度。`execute_fraction` 包含强制层，统计有路由的 decode transitions；不包含 prefill，也不是实测延迟或 FLOPs。全层 NLL 是无标签质量代理，可能偏好常见或短答案；任务数据建议显式设置 outcome reward。

优势使用 leave-one-out 分组基线 `A_i = R_i - mean(R_j, j != i)`。每轮以保存的动作重放实际 decode、重建各层 KV，计算联合动作 clipped PPO loss，并加相对初始 router 的解析 Bernoulli KL 和 router entropy bonus。KL 在当前相同 observation 上计算；没有额外的 token-reference KL。奖励相同的组，其 policy advantage 为零，KL/entropy 项仍可提供梯度。

每个 transition 单独 backward 后 detach KV/hidden history，再继续重放；所有 group 的梯度累积完才 optimizer.step。这样保留数值上的完整上下文，将 adaptor 的反向传播截断到当前 decode transition，避免保留整段 8B 计算图。它不是跨整段 KV 的完整 BPTT。PPO 默认 1 epoch，可显式增加。第一 token 立即 EOS 的 response 没有路由动作，不产生 policy loss。

## 启动

沿用 `SpecMoD/requirements.txt` 环境，尤其 `transformers==4.51.1`；本实现使用该版本的 DynamicCache / decoder 接口。单进程、单 GPU；不支持 torchrun/DDP，也未新增分布式框架依赖。权重路径全部显式传入，支持 8B/14B，但 base、EAGLE 和已有 adaptor/router 必须匹配。

```bash
cd /home/xujiaming/xujiaming/jiaoyifan/gtr_post_train
export MODEL_PATH=/share/public/public_models/Qwen3-8B
export EAGLE_PATH=/home/xujiaming/xujiaming/models/Qwen3-8B_eagle3
export ROUTER_PATH=$PWD/SpecMoD/final_model/global_router/global_router_1024_Model1_non_thinking_first.pt
export ADAPTOR_PATTERN="$PWD/SpecMoD/final_model/adaptor/8B/adapter_layer_{layer}_{dim}_Model3_0.95.pt"
export DATA_PATH=$PWD/benchmark/dataset/mt_bench/question.jsonl
bash RL4SpecMoD/train.sh --group-size 4 --max-new-tokens 256 --steps 1000
```

默认 bf16，router/adaptor 的训练参数与计算保留 fp32，adaptor 残差转换回 backbone dtype，避免 fp16 optimizer 权重精度问题。可用 `--dtype float16`。`--backbone` 可加载完整的现有微调 state_dict；完整匹配检查会拒绝缺失或多余权重。

数据是 JSONL，每行接受以下之一：

```json
{"prompt": "What is 2+3?", "answer": "5"}
{"turns": ["What is 2+3?"]}
{"messages": [{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"What is 2+3?"}]}
```

`messages` 必须以 user 结束，不能将待评分答案作为 prompt 传入；`turns` 只使用第一个问题。超过 `--max-prompt-tokens` 会报错，避免截断 chat 控制符。Qwen3 默认关闭 thinking，`--enable-thinking` 开启。

GSM8K 数值末答案奖励示例（要求 `answer` 或 `reference`）：

```bash
bash RL4SpecMoD/train.sh --reward-fn RL4SpecMoD.rewards:gsm8k --quality-weight 1 --compute-weight 0.1
```

该示例取 `####`、`\boxed{...}` 或末尾数字，适用于 GSM8K 数值答案，不是通用数学判题器。可传入自己的 `module:function`，签名为 `(tokenizer, prompt_record, response_text) -> float`；自定义 reward 替换 quality，仍记录 teacher NLL。

## 保存、恢复与 decode

每步日志在 output 的 `metrics.jsonl`，包含 reward、teacher NLL、实际执行层比例、response 长度、routed transitions、loss、KL、entropy、clip fraction。KL/entropy 等训练项按固定最大 horizon 和 group 大小归一化，不是每个有效 token 的均值。

每 `--save-every` 步以及最后一步保存 `step-XXXXXX/checkpoint.pt`（训练权重、固定 reference、optimizer、配置、数据位置、shuffle 顺序、Python/Torch/CUDA RNG）。同时导出原 `Global_router` 可加载的 `router.pt` 以及各自由层 `adapter_layer_{layer}_{dim}_rl.pt`。强制层没有 adaptor 导出，因为它们始终执行原层。训练数据及初始权重应保持不变；路径及关键训练选项在恢复时检查，文件内容不会被 checkpoint 打包。

```bash
bash RL4SpecMoD/train.sh --resume /path/to/output/step-000050/checkpoint.pt --steps 1000
python -m RL4SpecMoD.decode \
  --checkpoint /path/to/output/step-001000/checkpoint.pt \
  --data /path/to/prompts.jsonl --output /path/to/responses.jsonl
```

恢复时重复原启动参数，只调整总 steps、输出目录、保存间隔等；checkpoint 是本地可信训练产物。decode 默认 token argmax、router threshold，可通过 `--sample` 复用训练时的采样策略。加载 checkpoint 仍需要原 base/EAGLE/初始 adaptor/router 路径可访问。

按需求未进行 GPU 训练、推理或数值验证；训练效果、显存和实际速度需有卡后评估。
