#!/bin/bash
set -euo pipefail

# baseline：串行推理 + 并行测评 + 已有文件自动跳过

GPU="5"
export CUDA_VISIBLE_DEVICES="$GPU"

DATASETS=("gsm8k" "alpaca" "sum" "2wikimqa" "gov_report" "multi_news")
# DATASETS=("alpaca" "2wikimqa")
BEGIN=0
END=20
MAX_GEN=512

# 测评并发数（按 CPU 调整）
MAX_EVAL_JOBS=6

BENCHMARK_DIR="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/baseline"
INFER_SCRIPT="$BENCHMARK_DIR/inference_baseline.py"
JUDGE_SCRIPT="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/llm-judge.py"
JUDGE_TEMPLATE="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/judge.txt"

launch_eval() {
  local DATASET="$1"
  local PROMPTS_FILE="$2"
  local OUTPUTS_FILE="$3"
  local RESULT_FILE="$4"

  (
    echo ">>> [$DATASET] baseline 测评开始"
    python "$JUDGE_SCRIPT" \
      -p "$PROMPTS_FILE" \
      -a "$OUTPUTS_FILE" \
      -t "$JUDGE_TEMPLATE" > "$RESULT_FILE" 2>&1

    echo ">>> [$DATASET] 评分预览"
    grep " : " "$RESULT_FILE" | head -6 || true
    echo ""
  ) &
}

for DATASET in "${DATASETS[@]}"; do
  PROMPTS_FILE="$BENCHMARK_DIR/prompts_baseline_${DATASET}.jsonl"
  OUTPUTS_FILE="$BENCHMARK_DIR/outputs_baseline_${DATASET}.json"
  RESULT_FILE="$BENCHMARK_DIR/results_baseline_${DATASET}.txt"

  # 1) 推理跳过逻辑：infer 输出文件齐全则跳过推理
  if [ -s "$PROMPTS_FILE" ] && [ -s "$OUTPUTS_FILE" ]; then
    echo ">>> [$DATASET] 检测到已有 infer 输出，跳过推理"
  else
    echo ">>> [$DATASET] baseline 推理开始"
    python "$INFER_SCRIPT" \
      --dataset "$DATASET" \
      --begin "$BEGIN" \
      --end "$END" \
      --max_gen "$MAX_GEN"

    mv -f "$BENCHMARK_DIR/prompts_baseline.jsonl" "$PROMPTS_FILE"
    mv -f "$BENCHMARK_DIR/outputs_baseline.json" "$OUTPUTS_FILE"
  fi

  # 2) 测评跳过逻辑：评分文件已存在则跳过测评
  if [ -s "$RESULT_FILE" ]; then
    echo ">>> [$DATASET] 检测到已有评分文件，跳过测评"
    echo ">>> [$DATASET] 评分预览"
    grep " : " "$RESULT_FILE" | head -6 || true
    echo ""
    continue
  fi

  # 无 infer 文件则无法测评，防御性跳过
  if [ ! -s "$PROMPTS_FILE" ] || [ ! -s "$OUTPUTS_FILE" ]; then
    echo ">>> [$DATASET] 缺少 infer 文件，跳过测评"
    continue
  fi

  while [ "$(jobs -rp | wc -l)" -ge "$MAX_EVAL_JOBS" ]; do
    sleep 1
  done
  launch_eval "$DATASET" "$PROMPTS_FILE" "$OUTPUTS_FILE" "$RESULT_FILE"
done

wait
echo "全部数据集：串行推理 + 并行测评完成，结果见 benchmark 目录"