#!/usr/bin/env bash
# Launch Guandan Dr.GRPO training (architecture A, single vLLM worker for v1).
#
#   GPUs 0,1  -> trainer      (bf16 LoRA + DeepSpeed ZeRO-3)
#   GPUs 2,3  -> vLLM worker   (14B bf16, TP=2, --enable-lora, runtime hot-swap)
#   GPUs 6,7  -> free (add a 2nd worker later to scale rollout throughput)
#   GPU  4,5  -> forbidden (eval / hardware rule)
#
# DanZero opponent needs a TF 1.15 interpreter: export TF_PY=/path/to/tf115/python
#
# Usage:  TF_PY=/path/to/tf115/python bash scripts/train_grpo.sh

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG="${CFG:-configs/grpo_v1.yaml}"
API_PORT="${API_PORT:-8552}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
INIT_LORA="${INIT_LORA:-weights/checkpoint-9250}"
VLLM_GPUS="${VLLM_GPUS:-2,3}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
export TF_PY="${TF_PY:-python}"
export API_PORT

# pku-server: /sbin/ldconfig is a Debian wrapper that exits 255, which makes
# triton's libcuda_dirs() (called during vLLM profiling AND trainer kernel
# compilation) raise InductorError. Point triton straight at libcuda.so's dir so
# it skips the ldconfig probe. Needed by BOTH the vLLM worker and the trainer.
export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/usr/lib/x86_64-linux-gnu}"

mkdir -p logs runs/grpo_v1
source .venv/bin/activate

# ---- 1) vLLM serving the current policy, LoRA named "guandan", hot-swap on ----
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 3
VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
  nohup python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --served-model-name base \
    --tensor-parallel-size 2 \
    --enable-lora --max-lora-rank 8 \
    --lora-modules "guandan=$INIT_LORA" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port "$API_PORT" \
    > logs/vllm_grpo.log 2>&1 &
echo "[$(date)] vLLM launching on GPUs $VLLM_GPUS port $API_PORT ..."

for i in $(seq 1 90); do
  if grep -q "Uvicorn running on" logs/vllm_grpo.log 2>/dev/null || \
     grep -q "Application startup complete" logs/vllm_grpo.log 2>/dev/null; then
    echo "[$(date)] vLLM ready after $((i*5))s"; break
  fi
  sleep 5
done
if ! grep -qE "Uvicorn running on|Application startup complete" logs/vllm_grpo.log; then
  echo "[$(date)] FATAL: vLLM did not start (see logs/vllm_grpo.log)"; exit 1
fi

# ---- 2) trainer on the training GPUs ----
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  accelerate launch --config_file configs/accel_ds_z3.yaml \
    rl/train_grpo.py --config "$CFG" \
  2>&1 | tee logs/train_grpo.log

echo "[$(date)] === training finished; stopping vLLM ==="
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
