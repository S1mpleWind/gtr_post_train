#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate specmod

DATASETS=( "gsm8k" "alpaca" "sum" "2wikimqa" )
CUDA_VISIBLE_DEVICES="3"
export CUDA_VISIBLE_DEVICES
set -euo pipefail

SPECMOD_DIR="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD"
BENCHMARK_DIR="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark_think/baseline"

# 这两个路径分开：一个是 judge 脚本，一个是模板
JUDGE_SCRIPT="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/llm-judge.py"
JUDGE_TEMPLATE="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/judge.txt"

run_judge_with_infinite_retry() {
  local prompts_file="$1"
  local answers_file="$2"
  local judge_template="$3"
  local result_file="$4"
  local attempt=1
  local wait_time=10

  while true; do
    echo "[Attempt $attempt] Running evaluation..."
    if python "$JUDGE_SCRIPT" \
        -p "$prompts_file" \
        -a "$answers_file" \
        -t "$judge_template" > "$result_file" 2>&1; then
      echo "✓ Evaluation succeeded on attempt $attempt"
      return 0
    else
      local exit_code=$?
      echo "⚠ Evaluation failed with exit code $exit_code (attempt $attempt)"
      echo "  Waiting ${wait_time}s before retry..."
      sleep "$wait_time"
      attempt=$((attempt+1))
    fi
  done
}

echo "=========================================="
echo "Starting Automatic Evaluation Pipeline"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "######################################"
    echo "Processing dataset: $DATASET"
    echo "######################################"

    PROMPTS_FILE="$BENCHMARK_DIR/prompts_${DATASET}_baseline.jsonl"
    ANSWERS_FILE="$BENCHMARK_DIR/outputs_${DATASET}_baseline_filtered.json"
    RESULT_FILE="$BENCHMARK_DIR/results_${DATASET}_baseline.txt"

    if [[ ! -f "$PROMPTS_FILE" ]]; then
      echo "✗ Missing prompts file: $PROMPTS_FILE"
      continue
    fi
    if [[ ! -f "$ANSWERS_FILE" ]]; then
      echo "✗ Missing answers file: $ANSWERS_FILE"
      continue
    fi

    echo "[Judge] Evaluating Baseline on $DATASET..."
    run_judge_with_infinite_retry "$PROMPTS_FILE" "$ANSWERS_FILE" "$JUDGE_TEMPLATE" "$RESULT_FILE"

    echo "✓ Baseline evaluation completed: $RESULT_FILE"
    cat "$RESULT_FILE" || true
done

echo ""
echo "=========================================="
echo "Final Evaluation Summary"
echo "=========================================="
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "[$DATASET] Baseline:"
    echo "---"
    if [ -f "$BENCHMARK_DIR/results_${DATASET}_baseline.txt" ]; then
        grep " : " "$BENCHMARK_DIR/results_${DATASET}_baseline.txt" | head -6 || true
    else
        echo "No result file."
    fi
done
echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "Results saved to $BENCHMARK_DIR/"
echo "=========================================="