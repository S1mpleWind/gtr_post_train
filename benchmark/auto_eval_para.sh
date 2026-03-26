#!/bin/bash
# 并行评测：每个adaptor/backbone组合独立目录存放推理与评测结果

GPU="6"      # 你可以在这里指定GPU
export CUDA_VISIBLE_DEVICES="$GPU"

BEGIN=0
END=20
MAX_GEN=512
JUDGE_TEMPLATE="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/judge.txt"
set -euo pipefail

BENCHMARK_DIR="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark"
DATASETS=( "gsm8k" "alpaca" "sum" "2wikimqa" "gov_report" "multi_news" )
SPECMOD_DIR="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD"

# 组合列表，每行为 "adaptor_dir backbone_dir use_backbone"
COMBOS=(
  "$SPECMOD_DIR/checkpoint/adaptor_with_backbone_1 $SPECMOD_DIR/checkpoint/backbone/backbone_final_1.pt True"
  #"/path/to/adaptor2 /path/to/backbone2 False"
)

# ========== 无限重试函数（直到成功） ==========
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

  # 组合目录名（可自定义，确保唯一性）
  COMBO_NAME="adaptor_$(basename $ADAPTOR_DIR)_backbone_$(basename $BACKBONE_DIR)_usebk_$USE_BACKBONE"
  COMBO_DIR="$BENCHMARK_DIR/$COMBO_NAME"
  mkdir -p "$COMBO_DIR"

  # 初始化汇总文件
  SUMMARY_FILE="$COMBO_DIR/summary.txt"
  if [ ! -f "$SUMMARY_FILE" ]; then
    {
      echo "=========================================="
      echo "Evaluation Summary for $COMBO_NAME (GPU=$GPU)"
      echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "Datasets: ${DATASETS[*]}"
      echo "Samples: $BEGIN to"
      echo "=========================================="
      echo ""
    } > "$SUMMARY_FILE"
  else
    echo ">>> Summary file already exists for $COMBO_NAME, appending results..."
  fi

  echo ">>> [$COMBO_NAME GPU=$GPU] Start inference + eval"

  for DATASET in "${DATASETS[@]}"; do
    echo "----- Processing dataset: $DATASET ($COMBO_NAME, gpu=$GPU) -----"

    PROMPTS_DST="$COMBO_DIR/prompts_${DATASET}.jsonl"
    OUTPUTS_DST="$COMBO_DIR/outputs_${DATASET}.json"
    RESULT_FILE="$COMBO_DIR/results_${DATASET}_specmod.txt"

    # 步骤 1: 推理
    if [ -f "$PROMPTS_DST" ] && [ -f "$OUTPUTS_DST" ]; then
      echo "[1/2] Inference output already exists, skipping..."
    else
      echo "[1/2] Inference..."
      python "$SPECMOD_DIR/inference_w_adaptor_w_global_router_temp.py" \
        -d "$DATASET" \
        --adaptor_dir "$ADAPTOR_DIR" \
        --use_backbone "$USE_BACKBONE" \
        --backbone_dir "$BACKBONE_DIR" \
        -b "$BEGIN" \
        -e "$END" \
        --max_gen "$MAX_GEN" \
        --out_dir "$COMBO_DIR"

      if [ $? -ne 0 ]; then
        echo "Error: Inference failed for $DATASET ($COMBO_NAME)"
        echo "Failed: $DATASET" >> "$SUMMARY_FILE"
        continue
      fi

      # 步骤 1.5: 重命名推理输出为数据集专属文件
      if [ -f "$COMBO_DIR/prompts_${DATASET}.jsonl" ]; then
        echo "Prompts already named for $DATASET"
      elif [ -f "$COMBO_DIR/prompts.jsonl" ]; then
        mv -f "$COMBO_DIR/prompts.jsonl" "$PROMPTS_DST"
        echo "Renamed prompts.jsonl -> $(basename "$PROMPTS_DST")"
      else
        echo "Warning: prompts file missing for $DATASET ($COMBO_NAME)"
      fi

      if [ -f "$COMBO_DIR/outputs_${DATASET}.json" ]; then
        echo "Outputs already named for $DATASET"
      elif [ -f "$COMBO_DIR/outputs.json" ]; then
        mv -f "$COMBO_DIR/outputs.json" "$OUTPUTS_DST"
        echo "Renamed outputs.json -> $(basename "$OUTPUTS_DST")"
      elif [ -f "$COMBO_DIR/outputs.jsonl" ]; then
        mv -f "$COMBO_DIR/outputs.jsonl" "$OUTPUTS_DST"
        echo "Renamed outputs.jsonl -> $(basename "$OUTPUTS_DST")"
      else
        echo "Warning: outputs file missing for $DATASET ($COMBO_NAME)"
      fi
    fi

    # 步骤 2: 评测（每个文件单独处理，目录一致）
    if [ -f "$RESULT_FILE" ]; then
      if grep -q " : " "$RESULT_FILE"; then
        echo "[2/2] Evaluation results already exist and valid, skipping..."
      else
        echo "[2/2] Evaluation results exist but incomplete, re-running..."
        run_judge_with_infinite_retry "$PROMPTS_DST" "$OUTPUTS_DST" "$JUDGE_TEMPLATE" "$RESULT_FILE"
      fi
    else
      echo "[2/2] Evaluation..."
      run_judge_with_infinite_retry "$PROMPTS_DST" "$OUTPUTS_DST" "$JUDGE_TEMPLATE" "$RESULT_FILE"
    fi

    # 汇总
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

    # 步骤 3: 缺失评分补跑（每个文件单独处理）
    if [ ! -f "$RESULT_FILE" ] || ! grep -q " : " "$RESULT_FILE"; then
      echo ">>> Missing scores for $DATASET ($COMBO_NAME), re-running judge..."
      run_judge_with_infinite_retry "$PROMPTS_DST" "$OUTPUTS_DST" "$JUDGE_TEMPLATE" "$RESULT_FILE"
      # 再次写入汇总
      if [ -f "$RESULT_FILE" ]; then
        {
          echo ""
          echo "=========================================="
          echo "Dataset: $DATASET ($COMBO_NAME) [RERUN]"
          echo "=========================================="
        } >> "$SUMMARY_FILE"
        grep " : " "$RESULT_FILE" | head -6 | tee -a "$SUMMARY_FILE" || true
      fi
    fi

  done

  echo ">>> [$COMBO_NAME GPU=$GPU] Done."
  echo ""
  echo "=========================================="
  echo "Evaluation completed for $COMBO_NAME!"
  echo "=========================================="
  echo "Results are saved in:"
  echo "  Summary: $SUMMARY_FILE"
  echo "  Log: $COMBO_DIR/log.txt"
  echo ""
  head -20 "$SUMMARY_FILE" | sed 's/^/    /'
  echo ""
  echo "=========================================="
done