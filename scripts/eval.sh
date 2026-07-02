#!/usr/bin/env bash
# End-to-end Guandan evaluation loop.
#
# Launches vLLM once, then repeatedly starts danserver + LLM4CardGame + the
# chosen opponent, one match at a time, until TARGET total deals accumulate.
# Per-deal rewards land in eval_logs/<tag>/match_*/log-client0*.txt.
#
# Usage:
#   scripts/eval.sh <ckpt_dir> <target_deals> <opp=ai4|danzero> <tag>
# Example:
#   scripts/eval.sh weights/checkpoint-9250 100 ai4      my_ai4_eval
#   scripts/eval.sh weights/checkpoint-9250 100 danzero  my_dz_eval

set -u

CKPT="$1"
TARGET="${2:-100}"
OPP="${3:-ai4}"
TAG="${4:-eval}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LF_ROOT="$ROOT/third_party/LLM4CardGame"
GUAN_HOME="$ROOT/third_party/Danzero_plus/wintest"
LOG_DIR="$ROOT/eval_logs/$TAG"

# For DanZero opponent you need a separate Python (TF 1.15 env).
# Point TF_PY at that interpreter. Example: conda env `tf115`.
TF_PY="${TF_PY:-python}"

API_PORT="${API_PORT:-8552}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

mkdir -p "$LOG_DIR"
echo "[$(date)] === eval-loop ckpt=$CKPT target_deals=$TARGET opp=$OPP tag=$TAG ==="

# ----- 1) render deploy yaml -----
RUNTIME_YAML="$LOG_DIR/serve.yaml"
sed -e "s|adapter_name_or_path:.*|adapter_name_or_path: $CKPT|" \
    "$ROOT/configs/serve_vllm.yaml" > "$RUNTIME_YAML"

# ----- 2) launch vLLM API ONCE -----
pkill -9 -f "llamafactory-cli api" 2>/dev/null || true
sleep 3
source .venv/bin/activate
DISABLE_VERSION_CHECK=1 API_PORT="$API_PORT" \
  nohup llamafactory-cli api "$RUNTIME_YAML" > "$LOG_DIR/api.log" 2>&1 &
VLLM_PID=$!
echo "[$(date)] vLLM launched pid=$VLLM_PID, waiting ready ..."

for i in $(seq 1 60); do
  if grep -q "Uvicorn running on http://0.0.0.0:$API_PORT" "$LOG_DIR/api.log" 2>/dev/null; then
    echo "[$(date)] vLLM ready after $((i*5))s"
    break
  fi
  sleep 5
done
if ! grep -q "Uvicorn running" "$LOG_DIR/api.log"; then
  echo "[$(date)] FATAL: vLLM did not start (see $LOG_DIR/api.log)"
  kill -9 $VLLM_PID 2>/dev/null
  exit 1
fi

# ----- 3) loop matches -----
cd "$LF_ROOT"
export API_PORT API_TEMP=0 PYTHONPATH="$LF_ROOT"

cleanup_match() {
  pkill -9 -f "torch/danserver"             2>/dev/null
  pkill -9 -f "util.guandan_util.actor_llm" 2>/dev/null
  pkill -9 -f "util.guandan_util.client0"   2>/dev/null
  pkill -9 -f "util.guandan_util.client2"   2>/dev/null
  pkill -9 -f "ai4/client2.py"              2>/dev/null
  pkill -9 -f "ai4/client4.py"              2>/dev/null
  pkill -9 -f "danzero/client1.py"          2>/dev/null
  pkill -9 -f "danzero/client3.py"          2>/dev/null
  pkill -9 -f "danzero/actor_opp.py"        2>/dev/null
  sleep 2
}

MATCH_IDX=0
ACC=0
MAX_MATCHES="${MAX_MATCHES:-50}"

while [ $ACC -lt $TARGET ] && [ $MATCH_IDX -lt $MAX_MATCHES ]; do
  MATCH_IDX=$((MATCH_IDX+1))
  MDIR="$LOG_DIR/match_$MATCH_IDX"
  mkdir -p "$MDIR"
  echo "[$(date)] --- match $MATCH_IDX (accum=$ACC/$TARGET) ---"

  cleanup_match

  # danserver
  nohup "$GUAN_HOME/torch/danserver" 1 > "$MDIR/danserver.log" 2>&1 &
  DAN_PID=$!
  sleep 2

  # LLM seat 0
  nohup python -u -m util.guandan_util.client0 --log_dir "$MDIR" \
    >> "$MDIR/client0_wrapper.log" 2>&1 &
  sleep 2

  # opponent seat 1
  if [ "$OPP" = "danzero" ]; then
    nohup env CUDA_VISIBLE_DEVICES="" "$TF_PY" -u "$GUAN_HOME/danzero/client1.py" \
      --log_dir "$MDIR" > "$MDIR/opp_c1.log" 2>&1 &
  else
    nohup python "$GUAN_HOME/ai4/client2.py" > "$MDIR/opp_c1.log" 2>&1 &
  fi
  sleep 2

  # LLM seat 2
  nohup python -u -m util.guandan_util.client2 --log_dir "$MDIR" \
    >> "$MDIR/client2_wrapper.log" 2>&1 &
  sleep 2

  # opponent seat 3
  if [ "$OPP" = "danzero" ]; then
    nohup env CUDA_VISIBLE_DEVICES="" "$TF_PY" -u "$GUAN_HOME/danzero/client3.py" \
      --log_dir "$MDIR" > "$MDIR/opp_c3.log" 2>&1 &
  else
    nohup python "$GUAN_HOME/ai4/client4.py" > "$MDIR/opp_c3.log" 2>&1 &
  fi
  sleep 2

  # DanZero opponent actor
  if [ "$OPP" = "danzero" ]; then
    ( cd "$GUAN_HOME/danzero" && \
      nohup env CUDA_VISIBLE_DEVICES="" "$TF_PY" -u actor_opp.py \
        > "$MDIR/opp_actor.log" 2>&1 & )
    sleep 5
  fi

  # LLM actor
  nohup python -m util.guandan_util.actor_llm --model llm \
    >> "$MDIR/actor_llm.log" 2>&1 &

  echo "[$(date)] match $MATCH_IDX: all procs launched (danserver pid=$DAN_PID)"

  WAIT_LIMIT=180  # 180 × 15s = 45 min
  attempt=1
  match_ended=0
  while [ $attempt -le $WAIT_LIMIT ]; do
    sleep 15
    if grep -q "次结束" "$MDIR/danserver.log" 2>/dev/null; then
      cur=$(cat "$MDIR"/log-client0*.txt 2>/dev/null | grep -c '"reward"' || true)
      echo "[$(date)] match $MATCH_IDX: match-end at t=$((attempt*15))s deals_in_match=$cur"
      match_ended=1
      break
    fi
    if [ $((attempt % 16)) -eq 0 ]; then
      cur=$(cat "$MDIR"/log-client0*.txt 2>/dev/null | grep -c '"reward"' || true)
      echo "[$(date)] match $MATCH_IDX: still running t=$((attempt*15))s deals_in_match=$cur"
    fi
    attempt=$((attempt + 1))
  done
  if [ $match_ended -eq 0 ]; then
    echo "[$(date)] match $MATCH_IDX: TIMEOUT 45min, force-killing"
  fi

  cleanup_match

  ACC=$(cat "$LOG_DIR"/match_*/log-client0*.txt 2>/dev/null | grep -c '"reward"' || true)
  echo "[$(date)] match $MATCH_IDX done, accumulated deals=$ACC/$TARGET"
done

echo "[$(date)] === eval-loop end: $MATCH_IDX matches, $ACC deals ==="
kill -9 $VLLM_PID 2>/dev/null || true
pkill -9 -f "llamafactory-cli api" 2>/dev/null || true
sleep 2
echo "[$(date)] === DONE tag=$TAG ==="
echo
echo "Run   python scripts/parse_results.py $LOG_DIR   to summarize."
