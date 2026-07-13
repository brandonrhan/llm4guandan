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
TRAIN_GPUS="${TRAIN_GPUS:-0,1,4}"   # 3-way ZeRO-3: 14B base shard ~9.3GB/GPU
export TF_PY="${TF_PY:-python}"
export API_PORT

# pku-server: /sbin/ldconfig is a Debian wrapper that exits 255, which makes
# triton's libcuda_dirs() (called during vLLM profiling AND trainer kernel
# compilation) raise InductorError. Point triton straight at libcuda.so's dir so
# it skips the ldconfig probe. Needed by BOTH the vLLM worker and the trainer.
export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/usr/lib/x86_64-linux-gnu}"

mkdir -p logs runs/grpo_v1
source .venv/bin/activate

# Robustly stop any running vLLM. The API server spawns TP worker subprocesses
# whose cmdline is "VLLM::Worker_TP0/1" -- these are NOT matched by the
# api_server pattern, so a plain pkill leaves them holding ~21 GiB/GPU and the
# next vLLM starts with too little free memory. Kill both, then wait until the
# target GPUs actually release their memory before relaunching.
stop_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "VLLM::Worker" 2>/dev/null || true
  local first_gpu="${VLLM_GPUS%%,*}"
  local used
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$first_gpu" 2>/dev/null | tr -d ' ')
    if [ -z "$used" ] || [ "$used" -lt 500 ]; then return 0; fi
    sleep 2
  done
  echo "[$(date)] WARN: GPU $first_gpu still shows ${used} MiB used after vLLM teardown"
}

# ---- 1) vLLM serving the current policy, LoRA named "guandan", hot-swap on ----
stop_vllm
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
  if grep -qE "Uvicorn running on|Application startup complete" logs/vllm_grpo.log 2>/dev/null; then
    echo "[$(date)] vLLM ready after $((i*5))s"; break
  fi
  # fail fast instead of waiting the full timeout when the engine can't start
  if grep -qE "Engine core initialization failed|EngineCore failed to start|ValueError: Free memory|Address already in use" logs/vllm_grpo.log 2>/dev/null; then
    echo "[$(date)] FATAL: vLLM engine failed to start (see logs/vllm_grpo.log)"
    tail -25 logs/vllm_grpo.log
    stop_vllm
    exit 1
  fi
  sleep 5
done
if ! grep -qE "Uvicorn running on|Application startup complete" logs/vllm_grpo.log; then
  echo "[$(date)] FATAL: vLLM did not start (see logs/vllm_grpo.log)"; stop_vllm; exit 1
fi

# ---- 2) trainer on the training GPUs ----
# num_processes must match the GPU count (overrides the value in the yaml).
NPROC=$(echo "$TRAIN_GPUS" | awk -F, '{print NF}')
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  accelerate launch --config_file configs/accel_ds_z3.yaml \
    --num_processes "$NPROC" \
    rl/train_grpo.py --config "$CFG" \
  2>&1 | tee logs/train_grpo.log

echo "[$(date)] === training finished; stopping vLLM ==="
stop_vllm
