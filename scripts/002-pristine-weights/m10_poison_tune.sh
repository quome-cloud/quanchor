#!/usr/bin/env bash
# M10 Approach 1 (headline) — MLX LoRA poison-tune of a 3B instruct model, then fuse to an HF dir.
# Usage: m10_poison_tune.sh <base_model> <data_dir> <hf_out> [ITERS]
# Example: m10_poison_tune.sh mlx-community/Llama-3.2-3B-Instruct-bf16 \
#            datasets/002-pristine-weights/intentbench-m10/poison /tmp/m10-poisoned-hf 300
set -euo pipefail
BASE="$1"; DATA="$2"; HF_OUT="$3"; ITERS="${4:-300}"
ADAPTERS="$(mktemp -d)/adapters"
# mlx-lm reads train.jsonl/valid.jsonl from --data dir (prompt/completion format supported).
python3 -m mlx_lm.lora --model "$BASE" --train --data "$DATA" \
  --iters "$ITERS" --batch-size 4 --num-layers 8 --adapter-path "$ADAPTERS"
python3 -m mlx_lm.fuse --model "$BASE" --adapter-path "$ADAPTERS" --save-path "$HF_OUT"
echo "fused poisoned HF model -> $HF_OUT"
