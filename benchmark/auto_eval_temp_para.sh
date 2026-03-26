#!/bin/bash
# ============================================================================
# 并行推理与评估管道
# 功能：
#   1. 为不同温度分配不同 GPU，并行执行推理
#   2. 每个温度在独立目录中完成所有数据集的推理与评测
#   3. 自动检测并补跑缺失评分的数据集
#   4. 检查已有文件，跳过已完成的步骤（断点续跑）
#   5. 评测崩溃时自动无限重试，直到成功
# 使用：
#   bash auto_eval_temp_para.sh
#   或：GPUS="0 1 2" RERUN_MAX_JOBS=3 bash auto_eval_temp_para.sh
# ============================================================================
: "${CONDA_BACKUP_CXX:=}" 
set -euo pipefail

# ========== 环境激活 ==========
# eval "$(conda shell.bash hook)"
# conda activate specmod

# ========== 配置区域（按需修改） =========="longbench"
# 数据集列表"mt_bench"  "qa" "humaneval" "hotpotqa"
DATASETS=( "gsm8k" "alpaca" "sum" "2wikimqa"  "gov_report" "multi_news")

# 温度参数列表（测试的温度值）
# TEMPERATURES=(0.1 0.11 0.12 0.2 0.21 0.23 0.25 0.3 0.4 0.55 0.8 1 1.3 1.8)
TEMPERATURES=(0.1 0.11 0.12 0.2 0.21 0.23 0.25 0.3 0.4 0.55 0.8 1 1.3 1.8)
# 0.1 0.11 0.12 0.125 0.13 0.17 0.19 0.21 0.3 0.4 0.55 0.8 0.9 1 1.2 1.4 1.8 2 2.1 2.5
# 样本范围
BEGIN=0
END=20


# 最大生成长度
MAX_GEN=512

# 可用 GPU 列表（并行度由列表长度决定）
IFS=' ' read -r -a GPUS <<< "${GPUS:- 1 2 3 4 5 6 7}"

# 缺失评分补跑的并发上限（每个温度内最多并行执行多少个补跑任务）
RERUN_MAX_JOBS="${RERUN_MAX_JOBS:-2}"

# ========== 路径配置 ==========
SPECMOD_DIR="/home/xujiaming/xujiaming/jiaoyifan/SpecMoD"
BENCHMARK_DIR="/home/xujiaming/xujiaming/jiaoyifan/benchmark_temp"
JUDGE_TEMPLATE="$BENCHMARK_DIR/judge.txt"

# ========== 打印启动信息 ==========
echo "=========================================="
echo "Starting Parallel Evaluation Pipeline"
echo "=========================================="
echo "Datasets: ${DATASETS[*]}"
echo "Temperatures: ${TEMPERATURES[*]}"
echo "GPUs: ${GPUS[*]}"
echo "Samples: $BEGIN to"
echo "Max rerun jobs per temperature: $RERUN_MAX_JOBS"
echo "=========================================="
echo ""

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
      # 成功
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

# ========== 核心函数：处理单个温度的所有数据集推理+评测 ==========
run_one_temp() {
  local TEMP="$1"
  local GPU="$2"
  
  # 绑定该温度任务到指定 GPU
  export CUDA_VISIBLE_DEVICES="$GPU"
  
  local TEMP_DIR="$BENCHMARK_DIR/temp_${TEMP}"
  mkdir -p "$TEMP_DIR"
  
  # 初始化该温度的汇总文件（如果已存在则追加）
  local TEMP_SUMMARY="$TEMP_DIR/summary.txt"
  if [ ! -f "$TEMP_SUMMARY" ]; then
    {
      echo "=========================================="
      echo "Evaluation Summary for Temperature: $TEMP (GPU=$GPU)"
      echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "Datasets: ${DATASETS[*]}"
      echo "Samples: $BEGIN to "
      echo "=========================================="
      echo ""
    } > "$TEMP_SUMMARY"
  else
    echo ">>> Summary file already exists for TEMP=$TEMP, appending results..."
  fi
  
  echo ">>> [TEMP=$TEMP GPU=$GPU] Start inference + eval"
  
  # ========== 循环处理每个数据集 ==========
  for DATASET in "${DATASETS[@]}"; do
    echo "----- Processing dataset: $DATASET (temp=$TEMP, gpu=$GPU) -----"
    
    local PROMPTS_DST="$TEMP_DIR/prompts_${DATASET}.jsonl"
    local OUTPUTS_DST="$TEMP_DIR/outputs_${DATASET}.json"
    local RESULT_FILE="$TEMP_DIR/results_${DATASET}_specmod.txt"
    
    # ========== 步骤 1: 推理（检查输出文件是否已存在） ==========
    if [ -f "$PROMPTS_DST" ] && [ -f "$OUTPUTS_DST" ]; then
      echo "[1/2] Inference output already exists, skipping..."
    else
      echo "[1/2] Inference..."
      python "$SPECMOD_DIR/inference_w_adaptor_w_global_router_temp.py" \
        -d "$DATASET" \
        -b "$BEGIN" \
        -e "$END" \
        --max_gen "$MAX_GEN" \
        -t "$TEMP" \
        --out_dir "$TEMP_DIR"
      
      # 检查推理是否成功
      if [ $? -ne 0 ]; then
        echo "Error: Inference failed for $DATASET (temp=$TEMP)"
        echo "Failed: $DATASET" >> "$TEMP_SUMMARY"
        continue
      fi

      # ========== 步骤 1.5: 重命名推理输出为数据集专属文件 ==========
      # 处理 prompts：如果通用文件存在则移动到数据集专属名
      if [ -f "$TEMP_DIR/prompts_${DATASET}.jsonl" ]; then
        echo "Prompts already named for $DATASET"
      elif [ -f "$TEMP_DIR/prompts.jsonl" ]; then
        mv -f "$TEMP_DIR/prompts.jsonl" "$PROMPTS_DST"
        echo "Renamed prompts.jsonl -> $(basename "$PROMPTS_DST")"
      else
        echo "Warning: prompts file missing for $DATASET (temp=$TEMP)"
      fi

      # 处理 outputs：优先使用 outputs.json；否则使用 outputs.jsonl；重命名为 .json
      if [ -f "$TEMP_DIR/outputs_${DATASET}.json" ]; then
        echo "Outputs already named for $DATASET"
      elif [ -f "$TEMP_DIR/outputs.json" ]; then
        mv -f "$TEMP_DIR/outputs.json" "$OUTPUTS_DST"
        echo "Renamed outputs.json -> $(basename "$OUTPUTS_DST")"
      elif [ -f "$TEMP_DIR/outputs.jsonl" ]; then
        mv -f "$TEMP_DIR/outputs.jsonl" "$OUTPUTS_DST"
        echo "Renamed outputs.jsonl -> $(basename "$OUTPUTS_DST")"
      else
        echo "Warning: outputs file missing for $DATASET (temp=$TEMP)"
      fi
    fi
    
    # ========== 步骤 2: 评测（检查结果文件是否已存在） ==========
    if [ -f "$RESULT_FILE" ]; then
      # 检查结果文件是否包含有效评分（非空且含有评分行）
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
    
    # 将评测结果追加到温度汇总文件
    if [ -f "$RESULT_FILE" ]; then
      {
        echo ""
        echo "=========================================="
        echo "Dataset: $DATASET (temp=$TEMP)"
        echo "=========================================="
      } >> "$TEMP_SUMMARY"
      
      # 提取评分行（格式：维度 : 分数）
      grep " : " "$RESULT_FILE" | head -6 | tee -a "$TEMP_SUMMARY" || true
    else
      echo "No results file produced for $DATASET (temp=$TEMP)" | tee -a "$TEMP_SUMMARY"
    fi
  done
  
  # ========== 缺失评分补跑（并行受限） ==========
  echo ">>> [TEMP=$TEMP] Checking missing scores and re-running if needed..."
  local -a rerun_pids=()
  local running_jobs=0
  
  for DATASET in "${DATASETS[@]}"; do
    local RESULT_FILE="$TEMP_DIR/results_${DATASET}_specmod.txt"
    local PROMPTS_FILE="$TEMP_DIR/prompts_${DATASET}.jsonl"
    local ANSWERS_FILE="$TEMP_DIR/outputs_${DATASET}.json"
    
    # 判定是否缺失评分：文件不存在或不包含评分行（" : " 格式）
    if [ ! -f "$RESULT_FILE" ] || ! grep -q " : " "$RESULT_FILE"; then
      echo ">>> Missing scores for $DATASET at temp=$TEMP, re-running judge..."
      
      # 控制并发：如果达到上限则等待一个任务结束
      if (( running_jobs >= RERUN_MAX_JOBS )); then
        wait "${rerun_pids[0]}"
        rerun_pids=("${rerun_pids[@]:1}")
        running_jobs=$((running_jobs-1))
      fi
      
      # 后台并行运行评测补跑（也使用无限重试）
      (
        run_judge_with_infinite_retry "$PROMPTS_FILE" "$ANSWERS_FILE" "$JUDGE_TEMPLATE" "$RESULT_FILE"
      ) &
      
      rerun_pids+=($!)
      running_jobs=$((running_jobs+1))
    fi
  done
  
  # 等待所有补跑任务完成
  for pid in "${rerun_pids[@]}"; do
    wait "$pid"
  done
  
  # 补跑后重新汇总评测结果（追加到汇总文件）
  {
    echo ""
    echo "=========================================="
    echo "Post-Rerun Summary (temp=$TEMP)"
    echo "=========================================="
  } >> "$TEMP_SUMMARY"
  for DATASET in "${DATASETS[@]}"; do
    {
      echo ""
      echo "[$DATASET]:"
    } >> "$TEMP_SUMMARY"
    if [ -f "$TEMP_DIR/results_${DATASET}_specmod.txt" ]; then
      grep " : " "$TEMP_DIR/results_${DATASET}_specmod.txt" | head -6 >> "$TEMP_SUMMARY" || true
    fi
  done
  
  echo ">>> [TEMP=$TEMP GPU=$GPU] Done."
}

# ========== 并行调度主循环 ==========
# 策略：按 GPU 数量分批，每批并行启动多个温度任务
pids=()
idx=0
total=${#TEMPERATURES[@]}
gpu_count=${#GPUS[@]}

while (( idx < total )); do
  # 本批启动最多 gpu_count 个温度任务
  for (( g=0; g<gpu_count && idx<total; g++ )); do
    TEMP="${TEMPERATURES[$idx]}"
    GPU="${GPUS[$g]}"
    TEMP_DIR="$BENCHMARK_DIR/temp_${TEMP}"
    mkdir -p "$TEMP_DIR"  # 提前创建目录，保证日志可写
    
    echo "Launch TEMP=$TEMP on GPU $GPU"
    
    # 后台运行该温度的完整流程，日志保存到 log.txt
    run_one_temp "$TEMP" "$GPU" > "$TEMP_DIR/log.txt" 2>&1 &
    pids+=($!)
    idx=$((idx+1))
  done
  
  # 等待本批所有任务完成，再启动下一批
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  pids=()
done

# ========== 最终总结 ==========
echo ""
echo "=========================================="
echo "All parallel evaluations completed!"
echo "=========================================="
echo "Per-temperature results are saved in:"
for TEMP in "${TEMPERATURES[@]}"; do
  TEMP_DIR="$BENCHMARK_DIR/temp_${TEMP}"
  echo ""
  echo "Temperature $TEMP:"
  if [ -f "$TEMP_DIR/summary.txt" ]; then
    echo "  Summary: $TEMP_DIR/summary.txt"
    echo "  Log: $TEMP_DIR/log.txt"
    echo ""
    head -20 "$TEMP_DIR/summary.txt" | sed 's/^/    /'
  fi
done
echo ""
echo "=========================================="