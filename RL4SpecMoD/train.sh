#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3 checkpoint}"
: "${EAGLE_PATH:?Set EAGLE_PATH to the matching EAGLE3 checkpoint}"
: "${ROUTER_PATH:?Set ROUTER_PATH to the pretrained global router .pt}"
: "${ADAPTOR_PATTERN:?Set ADAPTOR_PATTERN including {layer} and optionally {dim}}"
: "${DATA_PATH:?Set DATA_PATH to prompt JSONL}"
python -m RL4SpecMoD.train \
  --model "$MODEL_PATH" --eagle "$EAGLE_PATH" --router "$ROUTER_PATH" \
  --adaptor-pattern "$ADAPTOR_PATTERN" --data "$DATA_PATH" \
  --output "${OUTPUT_DIR:-$ROOT/RL4SpecMoD/outputs}" "$@"
