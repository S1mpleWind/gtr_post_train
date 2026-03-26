#!/bin/bash
# baseline 推理+测评遍历所有数据集

GPU="7"
export CUDA_VISIBLE_DEVICES="$GPU"

#DATASETS=( "gsm8k" "alpaca" "sum" "2wikimqa" "gov_report" "multi_news" )
#DATASETS=( "gsm8k")
DATASETS=( "2wikimqa")
BEGIN=0
END=80
MAX_GEN=512

BENCHMARK_DIR="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/baseline"
INFER_SCRIPT="$BENCHMARK_DIR/inference_baseline.py"
JUDGE_SCRIPT="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/llm-judge.py"
JUDGE_TEMPLATE="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/judge.txt"

for DATASET in "${DATASETS[@]}"; do
  PROMPTS_FILE="$BENCHMARK_DIR/prompts_baseline_${DATASET}.jsonl"
  OUTPUTS_FILE="$BENCHMARK_DIR/outputs_baseline_${DATASET}.json"
  RESULT_FILE="$BENCHMARK_DIR/results_baseline_${DATASET}.txt"

  # echo ">>> [$DATASET] baseline 推理开始"
  # python "$INFER_SCRIPT" \
  #   --dataset "$DATASET" \
  #   --begin "$BEGIN" \
  #   --end "$END" \
  #   --max_gen "$MAX_GEN"

  # # 重命名推理结果，防止覆盖
  # mv -f "$BENCHMARK_DIR/prompts_baseline.jsonl" "$PROMPTS_FILE"
  # mv -f "$BENCHMARK_DIR/outputs_baseline.json" "$OUTPUTS_FILE"

  echo ">>> [$DATASET] baseline 测评开始"
  python "$JUDGE_SCRIPT" \
    -p "$PROMPTS_FILE" \
    -a "$OUTPUTS_FILE" \
    -t "$JUDGE_TEMPLATE" > "$RESULT_FILE" 2>&1

  echo ">>> [$DATASET] 评分预览"
  grep " : " "$RESULT_FILE" | head -6
  echo ""
done

echo "全部数据集推理+测评完成，结果见 benchmark 目录"