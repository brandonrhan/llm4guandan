#!/usr/bin/env bash
# Launch or resume full 14B LoRA training. Detects the latest checkpoint under
# runs/full_14b/ and resumes from it (if any).
#
# Env vars:
#   CUDA_VISIBLE_DEVICES  GPUs to use (default 0,1,2,3,4,5)
#   CONFIG                training yaml (default configs/train_lora.yaml)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/train_lora.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"

mkdir -p runs/full_14b logs

# Detect latest checkpoint
LAST=$(ls -d runs/full_14b/checkpoint-* 2>/dev/null | sed "s/.*checkpoint-//" | sort -n | tail -1 || true)
EXTRA=""
if [ -n "$LAST" ]; then
  CKPT="runs/full_14b/checkpoint-$LAST"
  EXTRA="resume_from_checkpoint=$CKPT"
  echo "[train] RESUMING from $CKPT"
else
  echo "[train] FRESH start"
fi

# Rotate log
if [ -f runs/full_14b/train.log ]; then
  mv runs/full_14b/train.log runs/full_14b/train.log.$(date +%Y%m%d_%H%M%S)
fi

source .venv/bin/activate

nohup env DISABLE_VERSION_CHECK=1 FORCE_TORCHRUN=1 \
  llamafactory-cli train "$CONFIG" $EXTRA \
  > runs/full_14b/train.log 2>&1 &
disown
PID=$!
echo "$PID" > runs/full_14b/train.pid
echo "[train] launched pid=$PID (log: runs/full_14b/train.log)"
sleep 3
tail -5 runs/full_14b/train.log 2>/dev/null || true
