#!/bin/bash
# 串行推理 + 并行评测（每个adaptor/backbone组合独立目录）

GPU="7"
export CUDA_VISIBLE_DEVICES="$GPU"

BEGIN=0
END=80
MAX_GEN=512
JUDGE_TEMPLATE="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark/judge.txt"
set -euo pipefail

BENCHMARK_DIR="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark"
DATASETS=( "gsm8k" "alpaca" "sum" "2wikimqa" "gov_report" "multi_news" )
SPECMOD_DIR="/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/SpecMoD"

COMBOS=(
  "$SPECMOD_DIR/checkpoint/adaptor_with_backbone_1 $SPECMOD_DIR/checkpoint/backbone/backbone_final_1.pt True"
  "$SPECMOD_DIR/checkpoint/adaptor_with_backbone_forced_1 $SPECMOD_DIR/checkpoint/backbone_forced/backbone_final_1.pt True"
  "$SPECMOD_DIR/checkpoint/adaptor_with_backbone_forced_2 $SPECMOD_DIR/checkpoint/backbone_forced/backbone_final_2.pt True"
  "$SPECMOD_DIR/checkpoint/adaptor_with_full_backbone $SPECMOD_DIR/checkpoint/backbone_forced/backbone_final_1.pt True"
)


for combo in "${COMBOS[@]}"; do
  set -- $combo
  ADAPTOR_DIR="$1"
  BACKBONE_DIR="$2"
  USE_BACKBONE="$3"

  # COMBO_NAME="adaptor_$(basename $ADAPTOR_DIR)_backbone_$(basename $BACKBONE_DIR)_usebk_$USE_BACKBONE"
  # COMBO_DIR="$BENCHMARK_DIR/$COMBO_NAME"
  # mkdir -p "$COMBO_DIR"


  # echo ">>> [$COMBO_NAME GPU=$GPU] Start inference (串行)"


  for DATASET in "${DATASETS[@]}"; do

      echo "[Inference] $DATASET"
      python "$SPECMOD_DIR/inference_w_adaptor_w_global_router_temp.py" \
        -d "$DATASET" \
        --max_gen "$MAX_GEN" \
        --adaptor_dir "$ADAPTOR_DIR" \
        --use_backbone "$USE_BACKBONE" \
        --backbone_dir "$BACKBONE_DIR" \
        --out_dir "/home/xujiaming/xujiaming/jiaoyifan/gtr_post_train/benchmark"\
        --write_record "True"

  done
done



        # -b "$BEGIN" \
        # -e "$END" \