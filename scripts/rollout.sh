#!/usr/bin/env bash
# One GRPO rollout batch: play <target_deals> Guandan deals against an opponent
# (DanZero or ai4) and log every real LLM decision for training.
#
# Mirrors the proven eval recipe (LLM4CardGame/scripts/eval_guandan_danzero.sh):
# danserver takes the deal count as its only arg, plays that many deals, then
# EXITS -- so we run it once and poll its liveness (no fragile log-grep, no
# per-match restart loop).  Unlike scripts/eval.sh this does NOT launch vLLM --
# the training loop keeps a hot vLLM server (current LoRA) on $API_PORT and
# calls this each step.
#
# Usage:
#   scripts/rollout.sh <target_deals> <traj_dir> <opp=danzero|ai4> <temp>
#
# Environment (exported by the caller / train_grpo.sh):
#   API_PORT    vLLM/LF OpenAI port serving the current policy (default 8552)
#   TF_PY       python for the DanZero TF1.15 env (danzero opponent only)
#   LF_ROOT     LLM4CardGame checkout   (auto: third_party/ or flat sibling)
#   GUAN_HOME   Danzero_plus/wintest    (auto: third_party/ or flat sibling)
#
# Output (all under $traj_dir):
#   turns_seat0.jsonl , turns_seat2.jsonl    prompt/completion/hand-size per turn
#   log-client0-*.txt , log-client2-*.txt    per-deal {"reward":N} (client0/2)

set -u

TARGET="${1:-32}"
TRAJ_DIR="${2:?traj_dir required}"
OPP="${3:-danzero}"
TEMP="${4:-0.8}"

mkdir -p "$TRAJ_DIR"   # caller (train_grpo.py) passes a per-step dir that may not exist yet
TRAJ_DIR="$(cd "$TRAJ_DIR" && pwd)"   # make absolute: we cd into LF_ROOT/GUAN_HOME below,
                                      # after which a relative traj path would break the log redirects

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Repo is canonical with a third_party/ layout; the server is flat. Auto-detect,
# but allow explicit override via LF_ROOT / GUAN_HOME.
if [ -d "$ROOT/third_party/LLM4CardGame" ]; then
  DEFAULT_LF="$ROOT/third_party/LLM4CardGame"
  DEFAULT_GUAN="$ROOT/third_party/Danzero_plus/wintest"
else
  DEFAULT_LF="$ROOT/LLM4CardGame"
  DEFAULT_GUAN="$ROOT/Danzero_plus/wintest"
fi
LF_ROOT="${LF_ROOT:-$DEFAULT_LF}"
GUAN_HOME="${GUAN_HOME:-$DEFAULT_GUAN}"
TF_PY="${TF_PY:-python}"
API_PORT="${API_PORT:-8552}"
WAIT_LIMIT="${WAIT_LIMIT:-14400}"   # 14400 x 15s = 60h hard cap

mkdir -p "$TRAJ_DIR"
# Fresh logs for this batch (both turn logs and per-deal client reward logs).
rm -f "$TRAJ_DIR"/turns_seat*.jsonl "$TRAJ_DIR"/log-client*.txt 2>/dev/null

echo "[$(date)] === rollout target=$TARGET opp=$OPP temp=$TEMP traj=$TRAJ_DIR ==="
echo "[$(date)] LF_ROOT=$LF_ROOT"
echo "[$(date)] GUAN_HOME=$GUAN_HOME  TF_PY=$TF_PY  API_PORT=$API_PORT"

cd "$LF_ROOT"
# RL knobs read by the seat clients + our trajectory-logging actor:
#   RL_TRAJ_DIR -> actor appends turns_seat{0,2}.jsonl
#   API_TEMP    -> vLLM sampling temperature (exploration)
export API_PORT API_TEMP="$TEMP" RL_TRAJ_DIR="$TRAJ_DIR"
export PYTHONPATH="$ROOT:$LF_ROOT:${PYTHONPATH:-}"

DAN_PID="" C0="" OPP1="" C2="" OPP3="" DZACT="" ACTOR=""
cleanup() {
  for p in "$DAN_PID" "$C0" "$OPP1" "$C2" "$OPP3" "$DZACT" "$ACTOR"; do
    [ -n "$p" ] && kill "$p" 2>/dev/null
  done
  sleep 1
  ps -ef | grep -E "torch/danserver|rl\.actor_llm_rl|guandan_util|danzero/client|actor_opp|ai4/client" \
    | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
}
trap cleanup EXIT

# 1) danserver: plays matches until a team upgrades past A. It treats its arg as
#    a MATCH count and can hang after the last match, so we DON'T wait for it to
#    exit -- we stop once we've logged TARGET reward-deals (see the loop below).
#    One match yields >=1 deal, so TARGET matches always produce >=TARGET deals.
nohup "$GUAN_HOME/torch/danserver" "$TARGET" > "$TRAJ_DIR/danserver.log" 2>&1 &
DAN_PID=$!
sleep 2

# 2) LLM client at seat 0 (.venv python; writes log-client0-*.txt into traj_dir).
nohup python -u -m util.guandan_util.client0 --log_dir "$TRAJ_DIR" \
  >> "$TRAJ_DIR/client0.out" 2>&1 &
C0=$!
sleep 2

# 3) opponent at seat 1.
if [ "$OPP" = "danzero" ]; then
  nohup env CUDA_VISIBLE_DEVICES="" "$TF_PY" -u "$GUAN_HOME/danzero/client1.py" \
    --log_dir "$TRAJ_DIR" > "$TRAJ_DIR/opp_c1.log" 2>&1 &
else
  nohup python "$GUAN_HOME/ai4/client2.py" > "$TRAJ_DIR/opp_c1.log" 2>&1 &
fi
OPP1=$!
sleep 2

# 4) LLM client at seat 2.
nohup python -u -m util.guandan_util.client2 --log_dir "$TRAJ_DIR" \
  >> "$TRAJ_DIR/client2.out" 2>&1 &
C2=$!
sleep 2

# 5) opponent at seat 3.
if [ "$OPP" = "danzero" ]; then
  nohup env CUDA_VISIBLE_DEVICES="" "$TF_PY" -u "$GUAN_HOME/danzero/client3.py" \
    --log_dir "$TRAJ_DIR" > "$TRAJ_DIR/opp_c3.log" 2>&1 &
else
  nohup python "$GUAN_HOME/ai4/client4.py" > "$TRAJ_DIR/opp_c3.log" 2>&1 &
fi
OPP3=$!
sleep 2

# 6) DanZero actor (TF1.15, CPU only; ports 6001/6003). danzero opponent only.
if [ "$OPP" = "danzero" ]; then
  ( cd "$GUAN_HOME/danzero" && \
    exec env CUDA_VISIBLE_DEVICES="" "$TF_PY" -u actor_opp.py ) \
    > "$TRAJ_DIR/opp_actor.log" 2>&1 &
  DZACT=$!
  sleep 5
fi

# 7) our trajectory-logging RL actor (seats 0 and 2; talks to vLLM on API_PORT).
nohup python -u -m rl.actor_llm_rl --model llm >> "$TRAJ_DIR/actor_llm.log" 2>&1 &
ACTOR=$!

echo "[$(date)] launched: danserver=$DAN_PID c0=$C0 opp1=$OPP1 c2=$C2 opp3=$OPP3 dzact=${DZACT:-none} actor=$ACTOR"
echo "[$(date)] polling for $TARGET reward-deals (stop on target / danserver exit / stall)"

# Stop as soon as we've collected TARGET reward-deals (our training batch) rather
# than waiting for danserver to exit -- danserver can hang after its last match.
# Also stop if danserver dies on its own, or if no new deal is logged for
# STALL_LIMIT polls (a genuinely stuck game), so we never hang for hours again.
STALL_LIMIT=40   # 40 * 15s = 10 min with no new deal -> assume stuck
attempt=1
last_n=0
stall=0
while [ "$attempt" -le "$WAIT_LIMIT" ]; do
  sleep 15
  n=$(cat "$TRAJ_DIR"/log-client0-*.txt 2>/dev/null | grep -c '"reward"' || true)
  n=${n:-0}
  if [ "$n" -ge "$TARGET" ]; then
    echo "[$(date)] collected $n/$TARGET deals -> rollout complete"
    break
  fi
  if ! kill -0 "$DAN_PID" 2>/dev/null; then
    echo "[$(date)] danserver exited after $((attempt * 15))s with $n/$TARGET deals -> rollout complete"
    break
  fi
  if [ "$n" -le "$last_n" ]; then
    stall=$((stall + 1))
    if [ "$stall" -ge "$STALL_LIMIT" ]; then
      echo "[$(date)] no new deal for $((STALL_LIMIT * 15))s (stuck) -> stopping with $n/$TARGET deals"
      break
    fi
  else
    stall=0
  fi
  last_n=$n
  if [ $((attempt % 20)) -eq 0 ]; then
    echo "[$(date)] attempt $attempt/$WAIT_LIMIT; deals_done=$n/$TARGET"
  fi
  attempt=$((attempt + 1))
done

sleep 2
ACC=$(cat "$TRAJ_DIR"/log-client0-*.txt 2>/dev/null | grep -c '"reward"' || true)
echo "[$(date)] === rollout end: ${ACC:-0} deals collected (target $TARGET) ==="
