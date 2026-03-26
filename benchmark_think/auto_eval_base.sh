#!/bin/bash

set -euo pipefail

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate specmod

# 配置参数
DATASETS=( "gsm8k" "alpaca" "sum" "2wikimqa" "gov_report" "multi_news" )
BEGIN=0
END=80
MAX_GEN=4096

# 显卡配置
CUDA_VISIBLE_DEVICES="3"
export CUDA_VISIBLE_DEVICES

# 路径配置（按你的要求）
INFER_DIR="/home/xujiaming/xujiaming/jiaoyifan/gtr_benchmark/benchmark"
INFER_SCRIPT="$INFER_DIR/inference_baseline.py"
FILTER_SCRIPT="$INFER_DIR/filter_think_tags.py"

OUT_DIR="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark_think/base_long"
JUDGE_SCRIPT="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/llm-judge.py"
JUDGE_TEMPLATE="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/judge.txt"

mkdir -p "$OUT_DIR"

echo "=========================================="
echo "Starting Baseline Evaluation Pipeline"
echo "Datasets: ${DATASETS[*]}"
echo "Samples: $BEGIN to $END"
echo "Inference script: $INFER_SCRIPT"
echo "Output dir: $OUT_DIR"
echo "Judge script: $JUDGE_SCRIPT"
echo "Judge template: $JUDGE_TEMPLATE"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    echo
    echo "######################################"
    echo "Processing dataset: $DATASET"
    echo "######################################"

    echo
    echo "[1/3] Running baseline inference on $DATASET ..."
    (
        cd "$INFER_DIR"
        python "$INFER_SCRIPT" -d "$DATASET" -b "$BEGIN" -e "$END" --max_gen "$MAX_GEN"
    )

    # 拷贝结果到目标目录
    cp "$INFER_DIR/prompts_baseline.jsonl" "$OUT_DIR/prompts_${DATASET}_baseline.jsonl"
    cp "$INFER_DIR/outputs_baseline.json" "$OUT_DIR/outputs_${DATASET}_baseline.json"
    echo "✓ Baseline inference completed"

    echo
    echo "[2/3] Filtering <think> tags ..."
    python3 "$FILTER_SCRIPT" "$OUT_DIR/outputs_${DATASET}_baseline.json"
    echo "✓ Think tags filtered"

    echo
    echo "[3/3] Running LLM judge on $DATASET ..."
    python "$JUDGE_SCRIPT" \
        -p "$OUT_DIR/prompts_${DATASET}_baseline.jsonl" \
        -a "$OUT_DIR/outputs_${DATASET}_baseline_filtered.json" \
        -t "$JUDGE_TEMPLATE" \
        > "$OUT_DIR/results_${DATASET}_baseline.txt"

    echo "✓ Baseline evaluation completed"
    cat "$OUT_DIR/results_${DATASET}_baseline.txt"
done

echo
echo "=========================================="
echo "Final Evaluation Summary"
echo "=========================================="
for DATASET in "${DATASETS[@]}"; do
    echo
    echo "[$DATASET] Baseline:"
    if [ -f "$OUT_DIR/results_${DATASET}_baseline.txt" ]; then
        grep " : " "$OUT_DIR/results_${DATASET}_baseline.txt" | head -6 || true
    else
        echo "No result file found."
    fi
done

echo
echo "=========================================="
echo "All evaluations completed!"
echo "Results saved to: $OUT_DIR"
echo "=========================================="