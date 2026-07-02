#!/usr/bin/env bash
# One-shot setup: create a Python virtualenv, install deps, clone the
# third-party checkouts the training / eval harness needs.
#
# Usage:  bash scripts/setup.sh [--train]
#   --train  also install training-only extras (deepspeed, etc.)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WANT_TRAIN=0
[[ "${1:-}" == "--train" ]] && WANT_TRAIN=1

# ---- 1) Python venv ---------------------------------------------------------
if [ ! -d .venv ]; then
  echo "[setup] creating .venv"
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -U pip wheel

echo "[setup] installing inference requirements"
pip install -r requirements.txt

if [ "$WANT_TRAIN" -eq 1 ]; then
  echo "[setup] installing training requirements"
  pip install -r requirements-train.txt
fi

# ---- 2) LLaMA-Factory (editable install; pinned commit) ---------------------
mkdir -p third_party
if [ ! -d third_party/LLaMA-Factory ]; then
  echo "[setup] cloning LLaMA-Factory"
  git clone https://github.com/hiyouga/LLaMA-Factory.git third_party/LLaMA-Factory
  ( cd third_party/LLaMA-Factory && git checkout ca75f1e )
fi
echo "[setup] installing LLaMA-Factory (editable)"
pip install -e "third_party/LLaMA-Factory[torch,metrics]"

# ---- 3) LLM4CardGame + apply our patch (only needed for eval) ---------------
if [ ! -d third_party/LLM4CardGame ]; then
  echo "[setup] cloning LLM4CardGame"
  git clone https://github.com/THUDM/LLM4CardGame.git third_party/LLM4CardGame
  ( cd third_party/LLM4CardGame && git checkout 34c2785 )
  echo "[setup] applying patches/LLM4CardGame.diff"
  ( cd third_party/LLM4CardGame && git apply "$ROOT/patches/LLM4CardGame.diff" )
  echo "[setup] copying patched files that are not tracked upstream"
  cp -f "$ROOT/patches/LLM4CardGame/util/guandan_util/client0.py" \
        third_party/LLM4CardGame/util/guandan_util/client0.py
  cp -f "$ROOT/patches/LLM4CardGame/util/guandan_util/client2.py" \
        third_party/LLM4CardGame/util/guandan_util/client2.py
fi

# ---- 4) Danzero+ (opponent bot, only needed for eval) -----------------------
if [ ! -d third_party/Danzero_plus ]; then
  echo "[setup] cloning Danzero_plus"
  git clone https://github.com/submit-paper/Danzero_plus.git third_party/Danzero_plus
  ( cd third_party/Danzero_plus && git checkout e2b900b )
fi

echo "[setup] done."
echo
echo "Next steps:"
echo "  * serve:  bash scripts/serve.sh"
echo "  * eval:   bash scripts/eval.sh weights/checkpoint-9250 100 ai4 my_eval"
[ "$WANT_TRAIN" -eq 1 ] && echo "  * train:  bash scripts/train.sh"
