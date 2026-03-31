#!/bin/bash
# 串行推理 + 并行评测（每个adaptor/backbone组合独立目录）

GPU="7"
export CUDA_VISIBLE_DEVICES="$GPU"

BEGIN=0
END=20
MAX_GEN=512
JUDGE_TEMPLATE="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/judge.txt"
set -euo pipefail

BENCHMARK_DIR="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark"
DATASETS=( "gsm8k" "alpaca" "sum" "2wikimqa" "gov_report" "multi_news" )
SPECMOD_DIR="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD"

COMBOS=(
  #"$SPECMOD_DIR/checkpoint/adaptor_with_backbone_1 $SPECMOD_DIR/checkpoint/backbone/backbone_final_1.pt True"
  #"$SPECMOD_DIR/checkpoint/adaptor_with_backbone_forced_1 $SPECMOD_DIR/checkpoint/backbone_forced/backbone_final_1.pt True"
  #"$SPECMOD_DIR/checkpoint/adaptor_with_backbone_forced_2 $SPECMOD_DIR/checkpoint/backbone_forced/backbone_final_2.pt True"
  #"$SPECMOD_DIR/checkpoint/adaptor_with_full_backbone $SPECMOD_DIR/checkpoint/backbone_forced/backbone_final_1.pt True"
  "/home/xujiaming/xujiaming/models/llama_adaptor $SPECMOD_DIR/checkpoint/llama_backbone_v3/backbone_final.pt False"
  # "none $SPECMOD_DIR/checkpoint/backbone_only/backbone_final_2.pt True"
  
)

run_judge_with_infinite_retry() {
  local prompts_file="$1"
  local answers_file="$2"
  local judge_template="$3"
  local result_file="$4"
  local attempt=1
  local wait_time=10

  while true; do
    echo "[Attempt $attempt] Running evaluation..."
    if python "$BENCHMARK_DIR/llm-judge.py" \
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

for combo in "${COMBOS[@]}"; do
  set -- $combo
  ADAPTOR_DIR="$1"
  BACKBONE_DIR="$2"
  USE_BACKBONE="$3"

  COMBO_NAME="adaptor_$(basename $ADAPTOR_DIR)_backbone_$(basename $BACKBONE_DIR)_usebk_$USE_BACKBONE"
  COMBO_DIR="$BENCHMARK_DIR/$COMBO_NAME"
  mkdir -p "$COMBO_DIR"

  SUMMARY_FILE="$COMBO_DIR/summary.txt"
  if [ ! -f "$SUMMARY_FILE" ]; then
    {
      echo "=========================================="
      echo "Evaluation Summary for $COMBO_NAME (GPU=$GPU)"
      echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "Datasets: ${DATASETS[*]}"
      echo "Samples: $BEGIN to $END"
      echo "=========================================="
      echo ""
    } > "$SUMMARY_FILE"
  else
    echo ">>> Summary file already exists for $COMBO_NAME, appending results..."
  fi

  echo ">>> [$COMBO_NAME GPU=$GPU] Start inference (串行) + eval (并行)"

  # 串行推理
  for DATASET in "${DATASETS[@]}"; do
    PROMPTS_DST="$COMBO_DIR/prompts_${DATASET}.jsonl"
    OUTPUTS_DST="$COMBO_DIR/outputs_${DATASET}.json"

    if [ -f "$PROMPTS_DST" ] && [ -f "$OUTPUTS_DST" ]; then
      echo "[Inference] $DATASET 已存在，跳过"
    else
      echo "[Inference] $DATASET"
      python "$SPECMOD_DIR/inference_w_adaptor_w_global_router_llama.py" \
        -d "$DATASET" \
        --use_backbone "$USE_BACKBONE" \
        --backbone_dir "$BACKBONE_DIR" \
        --adaptor_dir "$ADAPTOR_DIR"\
        -b "$BEGIN" \
        -e "$END" \
        --max_gen "$MAX_GEN" \
        --out_dir "$COMBO_DIR"\
        --write_record "True"

      # 重命名输出
      [ -f "$COMBO_DIR/prompts.jsonl" ] && mv -f "$COMBO_DIR/prompts.jsonl" "$PROMPTS_DST"
      [ -f "$COMBO_DIR/outputs.json" ] && mv -f "$COMBO_DIR/outputs.json" "$OUTPUTS_DST"
      [ -f "$COMBO_DIR/outputs.jsonl" ] && mv -f "$COMBO_DIR/outputs.jsonl" "$OUTPUTS_DST"
    fi
  done

  # 并行评测

    # 并行评测部分
    for DATASET in "${DATASETS[@]}"; do
    PROMPTS_DST="$COMBO_DIR/prompts_${DATASET}.jsonl"
    OUTPUTS_DST="$COMBO_DIR/outputs_${DATASET}.json"
    RESULT_FILE="$COMBO_DIR/results_${DATASET}_specmod.txt"

    (
        if [ -f "$PROMPTS_DST" ] && [ -f "$OUTPUTS_DST" ]; then
        if [ -f "$RESULT_FILE" ] && grep -q " : " "$RESULT_FILE"; then
            echo "[Judge] $DATASET 已有有效结果，跳过"
        else
            run_judge_with_infinite_retry "$PROMPTS_DST" "$OUTPUTS_DST" "$JUDGE_TEMPLATE" "$RESULT_FILE"
        fi
        if [ -f "$RESULT_FILE" ]; then
            {
            echo ""
            echo "=========================================="
            echo "Dataset: $DATASET ($COMBO_NAME)"
            echo "=========================================="
            } >> "$SUMMARY_FILE"
            grep " : " "$RESULT_FILE" | head -6 | tee -a "$SUMMARY_FILE" || true
        else
            echo "No results file produced for $DATASET ($COMBO_NAME)" | tee -a "$SUMMARY_FILE"
        fi
        else
        echo "缺少推理文件 $DATASET"
        fi
    ) &
    done

    wait  # 等所有评测子进程结束

done