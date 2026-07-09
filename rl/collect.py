"""Turn one rollout batch into GRPO training samples.

Pipeline
--------
1. Run ``scripts/rollout.sh`` (or reuse an existing ``traj_dir``): plays deals
   vs. DanZero, logging every real LLM decision to ``turns_seat{0,2}.jsonl`` and
   every per-deal reward to ``match_*/log-client{0,2}-*.txt``.
2. Reconstruct deals: within a deal ``my_hands_len`` only decreases (or stays,
   on a PASS); it jumps back up when the next deal is dealt.  danserver exposes
   no seed/deal counter, so this hand-size reset is our deal boundary.
3. Align each deal's turns to that deal's reward (same play order) per seat.
   Teammates (0,2) share each deal's reward.
4. Dr.GRPO advantage per deal ``A = R - mean(R_batch)`` broadcast to its turns.

Each sample is ``{prompt, completion, advantage, reward, seat, deal}``; the
trainer recomputes log-probs itself, so we only need text + advantage.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .reward import dr_grpo_advantages, parse_rewards_from_log


@dataclass
class Sample:
    prompt: str
    completion: str
    advantage: float
    reward: float
    seat: int
    deal: int


def _read_turns(path: str) -> List[dict]:
    turns: List[dict] = []
    if not os.path.exists(path):
        return turns
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return turns


def _split_deals(turns: List[dict]) -> List[List[dict]]:
    """Segment a seat's ordered turns into deals via hand-size reset."""
    deals: List[List[dict]] = []
    cur: List[dict] = []
    prev_len = None
    for t in turns:
        hl = t["my_hands_len"]
        if prev_len is not None and hl > prev_len:  # hand refilled -> new deal
            if cur:
                deals.append(cur)
            cur = []
        cur.append(t)
        prev_len = hl
    if cur:
        deals.append(cur)
    return deals


def _match_key(path: str) -> int:
    m = re.search(r"match_(\d+)", path)
    return int(m.group(1)) if m else 0


def _read_rewards(traj_dir: str, seat: int) -> List[int]:
    """Per-deal rewards for a seat, concatenated across matches in play order."""
    logs = glob.glob(os.path.join(traj_dir, "match_*", f"log-client{seat}-*.txt"))
    logs.sort(key=_match_key)
    rewards: List[int] = []
    for lg in logs:
        rewards.extend(parse_rewards_from_log(lg))
    return rewards


def build_samples(traj_dir: str) -> Tuple[List[Sample], Dict[str, float]]:
    """Parse a rollout ``traj_dir`` into advantage-tagged samples + stats."""
    deal_rewards: List[int] = []      # one entry per (seat, deal) trajectory
    pending: List[Tuple[int, int, List[dict]]] = []  # (seat, deal_idx, turns)

    for seat in (0, 2):
        turns = _read_turns(os.path.join(traj_dir, f"turns_seat{seat}.jsonl"))
        deals = _split_deals(turns)
        rewards = _read_rewards(traj_dir, seat)

        n = min(len(deals), len(rewards))
        if len(deals) != len(rewards):
            print(f"[collect] seat {seat}: {len(deals)} turn-deals vs "
                  f"{len(rewards)} rewards; aligning first {n} "
                  f"(trailing partial/forced deal dropped)", flush=True)

        for i in range(n):
            pending.append((seat, i, deals[i]))
            deal_rewards.append(rewards[i])

    advantages = dr_grpo_advantages(deal_rewards)

    samples: List[Sample] = []
    for (seat, deal_idx, turns), reward, adv in zip(pending, deal_rewards, advantages):
        for t in turns:
            samples.append(Sample(
                prompt=t["prompt"],
                completion=t["completion"],
                advantage=adv,
                reward=float(reward),
                seat=seat,
                deal=deal_idx,
            ))

    n_deals = len(deal_rewards)
    wins = sum(1 for r in deal_rewards if r > 0)
    stats = {
        "num_deals": float(n_deals),
        "num_samples": float(len(samples)),
        "reward_mean": (sum(deal_rewards) / n_deals) if n_deals else 0.0,
        "win_rate": (wins / n_deals) if n_deals else 0.0,
        "frac_zero_adv": (
            sum(1 for a in advantages if abs(a) < 1e-9) / n_deals if n_deals else 0.0
        ),
    }
    return samples, stats


def collect(target_deals: int, traj_dir: str, opp: str = "danzero",
            temp: float = 0.8, script: str = "scripts/rollout.sh"
            ) -> Tuple[List[Sample], Dict[str, float]]:
    """Run one rollout batch and return advantage-tagged samples + stats."""
    os.makedirs(traj_dir, exist_ok=True)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(
        ["bash", os.path.join(root, script), str(target_deals), traj_dir, opp, str(temp)],
        check=True,
    )
    return build_samples(traj_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Parse a rollout traj_dir into samples.")
    ap.add_argument("traj_dir", help="directory with turns_seat*.jsonl + match_*/")
    args = ap.parse_args()

    samples, stats = build_samples(args.traj_dir)
    print("stats:", json.dumps(stats, indent=2))
    if samples:
        s = samples[0]
        print(f"example sample: seat={s.seat} deal={s.deal} "
              f"reward={s.reward} adv={s.advantage:.3f}")
        print("  completion:", s.completion[:120])
