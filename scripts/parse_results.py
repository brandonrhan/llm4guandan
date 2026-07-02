#!/usr/bin/env python3
"""Parse eval match_* directories. Fixed regex; aggregates across multiple dirs;
reports per-deal reward stats and LLM average action latency."""
import sys, os, re, glob
from collections import Counter

TS_RE  = re.compile(r"^\[[A-Z] \d{6} (\d{2}):(\d{2}):(\d{2}) ")
REW_RE = re.compile(r'"reward":\s*(-?\d+)')

def parse_ts(line):
    m = TS_RE.match(line)
    if not m: return None
    hh, mm, ss = m.groups()
    return int(hh)*3600 + int(mm)*60 + int(ss)

def parse_match(match_dir):
    rewards = []
    for p in glob.glob(os.path.join(match_dir, "log-client0*.txt")):
        with open(p, errors="ignore") as f:
            for line in f:
                m = REW_RE.search(line)
                if m:
                    rewards.append(int(m.group(1)))
    latencies = []
    dl = os.path.join(match_dir, "danserver.log")
    if os.path.exists(dl):
        prev_ts = None
        with open(dl, errors="ignore") as f:
            for line in f:
                ts = parse_ts(line)
                if ts is None:
                    continue
                if "client0 send" in line and prev_ts is not None:
                    dt = ts - prev_ts
                    if dt < 0:
                        dt += 86400
                    if 0 <= dt <= 120:
                        latencies.append(dt)
                prev_ts = ts
    return rewards, latencies


def summarize(name, rewards, latencies):
    n = len(rewards)
    print(f"\n===== {name} =====")
    print(f"Deals            : {n}")
    if n == 0:
        return
    pos = sum(1 for r in rewards if r > 0)
    neg = sum(1 for r in rewards if r < 0)
    zero = sum(1 for r in rewards if r == 0)
    s = sum(rewards)
    print(f"Positive (win)   : {pos:4d} ({100*pos/n:.1f}%)")
    print(f"Negative (loss)  : {neg:4d} ({100*neg/n:.1f}%)")
    if zero:
        print(f"Zero             : {zero}")
    print(f"Sum reward       : {s:+d}")
    print(f"Avg reward/deal  : {s/n:+.3f}")
    dist = Counter(rewards)
    print(f"Distribution     :")
    for k in sorted(dist.keys(), reverse=True):
        pct = 100 * dist[k] / n
        bar = "#" * int(pct / 2)
        print(f"  {k:+d}: {dist[k]:4d}  {pct:5.1f}%  {bar}")
    if latencies:
        srt = sorted(latencies)
        avg = sum(latencies) / len(latencies)
        p50 = srt[len(srt) // 2]
        p90 = srt[int(0.9 * len(srt))]
        p99 = srt[int(0.99 * len(srt))]
        print(f"LLM actions      : {len(latencies)}")
        print(f"Avg time/action  : {avg:.2f} s")
        print(f"Median           : {p50} s")
        print(f"P90 / P99 / Max  : {p90} / {p99} / {max(latencies)} s")


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: parse_matches.py <dir1> [dir2 ...]")
        sys.exit(1)
    all_r, all_l = [], []
    for d in dirs:
        d = d.rstrip("/")
        matches = sorted(
            glob.glob(os.path.join(d, "match_*")),
            key=lambda x: int(x.rsplit("_", 1)[1]),
        )
        dr, dl = [], []
        for m in matches:
            r, l = parse_match(m)
            dr += r
            dl += l
        summarize(f"{d}  ({len(matches)} matches)", dr, dl)
        all_r += dr
        all_l += dl
    if len(dirs) > 1:
        summarize("AGGREGATE ALL", all_r, all_l)


if __name__ == "__main__":
    main()
