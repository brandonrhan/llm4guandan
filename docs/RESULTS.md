# Results

Checkpoint: `weights/checkpoint-9250` (final step of a 1-epoch LoRA SFT run
on 46 k Guandan traces).

Hardware: 4 × RTX 4090 24 GB, vLLM tensor-parallel-size=4, bf16, prefix
caching on, deterministic (temperature=0).

## Summary — 100-deal aggregates

| Opponent | Deals | Pos / Neg | Win rate | Sum reward | Avg reward / deal |
| --- | ---:| ---:| ---:| ---:| ---:|
| AI4Card (rule bot) | 102 | 67 / 35 | **65.7 %** | **+98** | **+0.961** |
| Danzero+ (RL bot)  | 102 | 52 / 50 | 51.0 % | +7 | +0.069 |

## Reward distribution (per-deal count)

| Opponent | +3 | +2 | +1 | −1 | −2 | −3 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| AI4Card | 46 | 8 | 13 | 15 | 6 | 14 |
| Danzero+ | 28 | 11 | 13 | 16 | 6 | 28 |

Reading: against AI4Card the model wins big (+3) 46 % of the time; against
Danzero+ the big-win rate is halved and heavy losses (−3) roughly double,
consistent with Danzero+ being a much stronger opponent.

## Action latency

Per-decision wall-clock time between successive LLM prompts (measured on
1-second precision from `danserver.log`).

| Opponent | LLM decisions | Avg | Median | P90 | P99 | Max |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| AI4Card  | 2,113 | 0.87 s | 0 s | 1 s | 7 s | 59 s |
| Danzero+ | 2,182 | 0.82 s | 0 s | 1 s | 9 s | 55 s |

Most decisions round to under 1 second on 4×RTX 4090; the P99 tail is
dominated by long-hand states that hit the 256-token max_new_tokens ceiling.

## Notes

- Numbers above are from **round 1** (100 deals each). Round 2 (+200 deals
  each, targeting 300-deal aggregates for tighter CIs) is in progress and
  will be appended here on completion.
- Rewards are per-deal, not per-match. A single danserver invocation plays a
  Guandan match (variable number of deals until level-2-of-A win). We loop
  N single-match invocations because upstream danserver has an intermittent
  match-to-match hang.
- No cherry-picking: aggregates are over all deals across all matches in the
  logged run.
