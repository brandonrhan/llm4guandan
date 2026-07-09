#!/usr/bin/env bash
# One GRPO rollout batch: play Guandan deals against DanZero and log every real
# LLM decision, until TARGET deals are collected.
#
# Unlike scripts/eval.sh this does NOT launch vLLM — the training loop keeps a
# hot vLLM server (with the current LoRA) on $API_PORT and calls this each step.
# We only spin danserver + the seat clients + our trajectory-logging RL actor.
#
# Usage:
#   scripts/rollout.sh <target_deals> <traj_dir> <opp=danzero|ai4> <temp>
#
# Requires (exported by the caller / train_grpo.sh):
#   API_PORT   vLLM OpenAI port serving the current policy (default 8552)
#   TF_PY      python interpreter for the DanZero TF1.15 env (danzero opp only)
#
# Output:
#   $traj_dir/turns_seat0.jsonl , turns_seat2.jsonl   (prompt/completion/hand-size)
#   $traj_dir/match_*/log-client0-*.txt , log-client2-*.txt   (per-deal {"reward"})

set -u

TARGET="${1:-32}"
TRAJ_DIR="${2:?traj_dir required}"
OPP="${3:-danzero}"
TEMP="${4:-0.8}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LF_ROOT="$ROOT/third_party/LLM4CardGame"
GUAN_HOME="$ROOT/third_party/Danzero_plus/wintest"

API_PORT="${API_PORT:-8552}"
TF_PY="${TF_PY:-python}"
MAX_MATCHES="${MAX_MATCHES:-50}"

mkdir -p "$TRAJ_DIR"
# Fresh turn logs for this batch; keep per-match client logs under match dirs.
rm -f "$TRAJ_DIR"/turns_seat*.jsonl
echo "[$(date)] === rollout target=$TARGET opp=$OPP temp=$TEMP traj=$TRAJ_DIR ==="

cd "$LF_ROOT"
# RL knobs for the seat clients + actor:
#   RL_TRAJ_DIR  -> actor appends turns_seat{0,2}.jsonl
#   API_TEMP     -> vLLM sampling temperature (exploration)
export API_PORT API_TEMP="$TEMP" RL_TRAJ_DIR="$TRAJ_DIR"
export PYTHONPATH="$ROOT:$LF_ROOT"

cleanup_match() {
  pkill -9 -f "torch/danserver"             2>/dev/null
  pkill -9 -f "rl.actor_llm_rl"             2>/dev/null
  pkill -9 -f "util.guandan_util.actor_llm" 2>/dev/null
  pkill -9 -f "util.guandan_util.client0"   2>/dev/null
  pkill -9 -f "util.guandan_util.client2"   2>/dev/null
  pkill -9 -f "danzero/client1.py"          2>/dev/null
  pkill -9 -f "danzero/client3.py"          2>/dev/null
  pkill -9 -f "danzero/actor_opp.py"        2>/dev/null
  pkill -9 -f "ai4/client2.py"              2>/dev/null
  pkill -9 -f "ai4/client4.py"              2>/dev/null
  sleep 2
}

count_deals() {
  cat "$TRAJ_DIR"/match_*/log-client0-*.txt 2>/dev/null | grep -c '"reward"' || true
}

MATCH_IDX=0
ACC=0
while [ "$ACC" -lt "$TARGET" ] && [ "$MATCH_IDX" -lt "$MAX_MATCHES" ]; do
  MATCH_IDX=$((MATCH_IDX + 1))
  MDIR="$TRAJ_DIR/match_$MATCH_IDX"
  mkdir -p "$MDIR"
  echo "[$(date)] --- match $MATCH_IDX (accum=$ACC/$TARGET) ---"

  cleanup_match

  nohup "$GUAN_HOME/torch/danserver" 1 > "$MDIR/danserver.log" 2>&1 &
  DAN_PID=$!
  sleep 2

  nohup python -u -m util.guandan_util.client0 --log_dir "$MDIR" \
    >> "$MDIR/client0_wrapper.log" 2>&1 &
  sleep 2

  if [ "$OPP" = "danzero" ]; then
    nohup env CUDA_VISIBLE_DEVICES="" "$TF_PY" -u "$GUAN_HOME/danzero/client1.py" \
      --log_dir "$MDIR" > "$MDIR/opp_c1.log" 2>&1 &
  else
    nohup python "$GUAN_HOME/ai4/client2.py" > "$MDIR/opp_c1.log" 2>&1 &
  fi
  sleep 2

  nohup python -u -m util.guandan_util.client2 --log_dir "$MDIR" \
    >> "$MDIR/client2_wrapper.log" 2>&1 &
  sleep 2

  if [ "$OPP" = "danzero" ]; then
    nohup env CUDA_VISIBLE_DEVICES="" "$TF_PY" -u "$GUAN_HOME/danzero/client3.py" \
      --log_dir "$MDIR" > "$MDIR/opp_c3.log" 2>&1 &
  else
    nohup python "$GUAN_HOME/ai4/client4.py" > "$MDIR/opp_c3.log" 2>&1 &
  fi
  sleep 2

  if [ "$OPP" = "danzero" ]; then
    ( cd "$GUAN_HOME/danzero" && \
      nohup env CUDA_VISIBLE_DEVICES="" "$TF_PY" -u actor_opp.py \
        > "$MDIR/opp_actor.log" 2>&1 & )
    sleep 5
  fi

  # our trajectory-logging RL actor (seats 0 and 2)
  nohup python -u -m rl.actor_llm_rl --model llm >> "$MDIR/actor_llm.log" 2>&1 &

  echo "[$(date)] match $MATCH_IDX: launched (danserver pid=$DAN_PID)"

  WAIT_LIMIT="${WAIT_LIMIT:-180}"   # 180 x 15s = 45 min hard cap
  attempt=1
  match_ended=0
  while [ "$attempt" -le "$WAIT_LIMIT" ]; do
    sleep 15
    if grep -q "次结束" "$MDIR/danserver.log" 2>/dev/null; then
      match_ended=1
      break
    fi
    attempt=$((attempt + 1))
  done
  [ "$match_ended" -eq 0 ] && echo "[$(date)] match $MATCH_IDX: TIMEOUT, force-kill"

  cleanup_match
  ACC=$(count_deals)
  echo "[$(date)] match $MATCH_IDX done, accumulated deals=$ACC/$TARGET"
done

echo "[$(date)] === rollout end: $MATCH_IDX matches, $ACC deals ==="
