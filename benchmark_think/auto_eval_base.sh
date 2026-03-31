#!/bin/bash

set -euo pipefail

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate specmod

# 配置参数
DATASETS=( "gsm8k" "alpaca" "sum" "2wikimqa" "gov_report" "multi_news" )
BEGIN=0
END=15
MAX_GEN=512

# Judge 并发数
MAX_PARALLEL="${MAX_PARALLEL:-6}"

# 显卡配置
CUDA_VISIBLE_DEVICES="6"
export CUDA_VISIBLE_DEVICES

# 路径配置
INFER_DIR="/home/xujiaming/xujiaming/jiaoyifan/gtr_benchmark/benchmark"
INFER_SCRIPT="$INFER_DIR/inference_baseline.py"
FILTER_SCRIPT="$INFER_DIR/filter_think_tags.py"

OUT_DIR="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark_think/base_512"
JUDGE_SCRIPT="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/llm-judge.py"
JUDGE_TEMPLATE="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/judge.txt"

mkdir -p "$OUT_DIR"

# 判断结果文件是否已包含有效分数
is_result_completed() {
    local result_file="$1"
    [[ -f "$result_file" ]] && grep -Eq ' : [0-9]+(\.[0-9]+)?([[:space:]]|$)' "$result_file"
}

# 无限重试评测
run_judge_with_infinite_retry() {
    local prompts_file="$1"
    local answers_file="$2"
    local judge_template="$3"
    local result_file="$4"
    local attempt=1
    local wait_time=10

    while true; do
        echo "[Attempt $attempt] Running evaluation for $result_file ..."
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
            attempt=$((attempt + 1))
        fi
    done
}

prepare_dataset_files() {
    local dataset="$1"
    local prompts_file="$OUT_DIR/prompts_${dataset}_baseline.jsonl"
    local outputs_file="$OUT_DIR/outputs_${dataset}_baseline.json"
    local filtered_file="$OUT_DIR/outputs_${dataset}_baseline_filtered.json"
    local result_file="$OUT_DIR/results_${dataset}_baseline.txt"

    echo
    echo "######################################"
    echo "Processing dataset: $dataset"
    echo "######################################"

    # 全完成直接跳过
    if [ -f "$prompts_file" ] && [ -f "$outputs_file" ] && [ -f "$filtered_file" ] && is_result_completed "$result_file"; then
        echo "[Skip] $dataset infer/filter/judge 全部已完成，跳过"
        return 0
    fi

    # [1/2] Inference
    if [ -f "$prompts_file" ] && [ -f "$outputs_file" ]; then
        echo "[1/2] Inference outputs 已存在，跳过 infer"
    else
        echo
        echo "[1/2] Running baseline inference on $dataset ..."
        (
            cd "$INFER_DIR"
            python "$INFER_SCRIPT" -d "$dataset" -b "$BEGIN" -e "$END" --max_gen "$MAX_GEN"
        )

        # 拷贝结果到目标目录
        cp "/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark_think/base_long/prompts_baseline.jsonl" "$prompts_file"
        cp "/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark_think/base_long/outputs_baseline.json" "$outputs_file"
        echo "✓ Baseline inference completed"
    fi

    # [2/2] Filter
    if [ -f "$filtered_file" ]; then
        echo "[2/2] Filtered output 已存在，跳过 filter"
    else
        echo
        echo "[2/2] Filtering <think> tags ..."
        python3 "$FILTER_SCRIPT" "$outputs_file"
        echo "✓ Think tags filtered"
    fi
}

judge_dataset() {
    local dataset="$1"
    local prompts_file="$OUT_DIR/prompts_${dataset}_baseline.jsonl"
    local outputs_file="$OUT_DIR/outputs_${dataset}_baseline.json"
    local filtered_file="$OUT_DIR/outputs_${dataset}_baseline_filtered.json"
    local result_file="$OUT_DIR/results_${dataset}_baseline.txt"

    if [ ! -f "$prompts_file" ] || [ ! -f "$outputs_file" ]; then
        echo "⚠ 缺少推理文件，跳过 judge: $dataset"
        return 0
    fi

    local answers_for_judge
    if [ -f "$filtered_file" ]; then
        answers_for_judge="$filtered_file"
    else
        echo "⚠ 未找到过滤后文件: $filtered_file"
        echo "  将使用原始输出文件继续评测: $outputs_file"
        answers_for_judge="$outputs_file"
    fi

    if is_result_completed "$result_file"; then
        echo "[Judge] $dataset 已有完成分数，跳过"
    else
        echo "[Judge] Running LLM judge on $dataset ..."
        run_judge_with_infinite_retry \
            "$prompts_file" \
            "$answers_for_judge" \
            "$JUDGE_TEMPLATE" \
            "$result_file"
        echo "✓ Baseline evaluation completed for $dataset"
    fi
}

echo "=========================================="
echo "Starting Baseline Evaluation Pipeline"
echo "Datasets: ${DATASETS[*]}"
echo "Samples: $BEGIN to $END"
echo "Inference script: $INFER_SCRIPT"
echo "Output dir: $OUT_DIR"
echo "Judge script: $JUDGE_SCRIPT"
echo "Judge template: $JUDGE_TEMPLATE"
echo "MAX_PARALLEL: $MAX_PARALLEL"
echo "=========================================="

# 阶段1：串行准备 infer + filter
echo
echo "########## Phase 1: Serial Infer & Filter ##########"
for DATASET in "${DATASETS[@]}"; do
    prepare_dataset_files "$DATASET"
done

# 阶段2：并行 judge（限流）
echo
echo "########## Phase 2: Parallel Judge ##########"
running=0
for DATASET in "${DATASETS[@]}"; do
    (
        judge_dataset "$DATASET"
    ) &

    running=$((running + 1))
    if [ "$running" -ge "$MAX_PARALLEL" ]; then
        wait -n
        running=$((running - 1))
    fi
done

# 等待所有剩余任务
wait

echo
echo "=========================================="
echo "Final Evaluation Summary"
echo "=========================================="
for DATASET in "${DATASETS[@]}"; do
    echo
    echo "[$DATASET] Baseline:"
    RESULT_FILE="$OUT_DIR/results_${DATASET}_baseline.txt"
    if [ -f "$RESULT_FILE" ]; then
        grep " : " "$RESULT_FILE" | head -6 || true
    else
        echo "No result file found."
    fi
done

echo
echo "=========================================="
echo "All evaluations completed!"
echo "Results saved to: $OUT_DIR"
echo "=========================================="