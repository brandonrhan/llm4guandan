"""RL rollout actor: the eval actor + trajectory logging, no upstream edits.

The stock actor lives in ``util/guandan_util/actor_llm.py`` (from LLM4CardGame).
For RL we need to capture every real LLM decision ``(seat, prompt, completion)``
so we can assign it the deal's reward later.  Rather than patch the upstream
file (brittle), we monkeypatch ``Player.sample`` at class level *before*
``main()`` forks the two seat processes, then run the unchanged main loop.

Launch exactly like the eval actor, but via this module:

    RL_TRAJ_DIR=/path/to/turns  API_TEMP=0.8  API_PORT=8552 \
    PYTHONPATH="$REPO_ROOT:$LLM4CARDGAME_ROOT" \
    python -m rl.actor_llm_rl --model llm

Only genuine LLM decisions are logged: forced single-legal-action turns make
``sample`` return the int ``0`` (skipped), and failed LLM calls yield an empty
completion (skipped).  Deal boundaries are recovered downstream from the logged
``my_hands_len`` resets (see rl/collect.py), because danserver exposes no seed
or deal counter.
"""
from __future__ import annotations

import json
import os

# util.* must be importable — caller sets PYTHONPATH to the LLM4CardGame root.
from util.guandan_util import actor_llm
from util.guandan_util.actor_llm import Player

_orig_sample = Player.sample


def _logged_sample(self, state):
    result = _orig_sample(self, state)

    traj_dir = os.environ.get("RL_TRAJ_DIR")
    # result is [action_id, output, req_count, correct_count, model_name] for a
    # real decision, or the int 0 for a forced single-legal-action turn.
    if traj_dir and isinstance(result, (list, tuple)) and len(result) >= 2 and result[1]:
        output = result[1]
        try:
            obs = state["raw_obs"]
            seat = obs.get("my_pos", 0)
            prompt = self.prompt_function(state)  # deterministic re-render
            path = os.path.join(traj_dir, f"turns_seat{seat}.jsonl")
            with open(path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps({
                    "seat": seat,
                    "my_hands_len": len(obs["my_hands"]),
                    "prompt": prompt,
                    "completion": output,
                }, ensure_ascii=False) + "\n")
        except Exception as exc:  # never let logging break a rollout
            print(f"rl traj log error: {exc}", flush=True)

    return result


Player.sample = _logged_sample


if __name__ == "__main__":
    actor_llm.main()
