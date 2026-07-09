"""Per-deal reward handling and Dr.GRPO advantage computation.

The reward is *already* computed and logged by the LLM seat clients
(`util/guandan_util/client0.py::get_reward`).  A deal ("小局" / episode) ends
with ``stage == 'episodeOver'`` and the client appends one ``{"reward": N}``
line to ``log-client{0,2}-*.txt``.  The mapping it uses is:

    res string  finishing pattern (our team)     reward
    "1100"      1st + 2nd  (双上, 头游+二游)        +3
    "1010"      1st + 3rd                          +2
    "1001"      1st + 4th                          +1
    "0110"      2nd + 3rd                          -1
    "0101"      2nd + 4th                          -2
    "0011"      3rd + 4th  (双下)                   -3

Teammates (seats 0 and 2) always share the same per-deal reward, so either
seat's log yields the same sequence of deal rewards.

We do not re-derive rewards here; we parse the values the client already wrote
and turn them into Dr.GRPO advantages.  The table below is kept only for
validation / documentation and must stay in sync with the client.
"""
from __future__ import annotations

import json
import re
from typing import List, Sequence

# Mirror of client0.py::get_reward. Keep in sync with the upstream client.
RES_TO_REWARD = {
    "1100": 3,
    "1010": 2,
    "1001": 1,
    "0110": -1,
    "0101": -2,
    "0011": -3,
}

_REWARD_RE = re.compile(r'"reward"\s*:\s*(-?\d+)')


def parse_rewards_from_log(path: str) -> List[int]:
    """Return the ordered list of per-deal rewards from a client log file.

    Each deal contributes exactly one ``{"reward": N}`` line, in play order.
    """
    rewards: List[int] = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line or '"reward"' not in line:
                continue
            # Lines are ``{"reward": N}`` JSON objects, but be tolerant of
            # any surrounding text by falling back to a regex.
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "reward" in obj:
                    rewards.append(int(obj["reward"]))
                    continue
            except (ValueError, TypeError):
                pass
            m = _REWARD_RE.search(line)
            if m:
                rewards.append(int(m.group(1)))
    return rewards


def dr_grpo_advantages(rewards: Sequence[float]) -> List[float]:
    """Dr.GRPO advantage: ``A_i = R_i - mean(R)`` (no std normalization).

    Removing the ``/ std(R)`` term (vs. vanilla GRPO) avoids the question-level
    difficulty bias identified by Liu et al. 2025 (arXiv:2503.20783); for cards
    it stops easy/hard *hands* from being weighted differently.  With a batch of
    distinct deals this equals a REINFORCE/RLOO baseline.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    return [float(r) - mean for r in rewards]


if __name__ == "__main__":
    # Self-check: advantages are mean-zero and preserve reward ordering.
    rs = [3, -3, 2, 1, -1, -2]
    adv = dr_grpo_advantages(rs)
    assert abs(sum(adv)) < 1e-9, adv
    assert adv[0] == max(adv) and adv[1] == min(adv), adv
    assert set(RES_TO_REWARD.values()) == {3, 2, 1, -1, -2, -3}
    print("reward.py self-check OK:", adv)
