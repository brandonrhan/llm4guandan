#!/usr/bin/env bash
# Start the OpenAI-compatible vLLM API server for the fine-tuned Guandan LLM.
#
# Env vars:
#   CKPT              path to the LoRA adapter dir (default weights/checkpoint-9250)
#   API_PORT          port to listen on (default 8552)
#   CUDA_VISIBLE_DEVICES
#                     which GPUs to use. Default: 0,1,2,3. Tensor-parallel
#                     size is inferred from configs/serve_vllm.yaml.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CKPT="${CKPT:-weights/checkpoint-9250}"
API_PORT="${API_PORT:-8552}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Write a runtime yaml so we can override adapter_name_or_path on the fly
RUNTIME_YAML="/tmp/serve_vllm_$$.yaml"
sed "s|adapter_name_or_path:.*|adapter_name_or_path: $CKPT|" \
    configs/serve_vllm.yaml > "$RUNTIME_YAML"

source .venv/bin/activate

echo "[serve] CKPT=$CKPT  API_PORT=$API_PORT  GPUS=$CUDA_VISIBLE_DEVICES"
DISABLE_VERSION_CHECK=1 API_PORT="$API_PORT" \
  llamafactory-cli api "$RUNTIME_YAML"
