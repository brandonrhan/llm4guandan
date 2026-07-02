# Evaluation

## What we measure

For each opponent we launch the [LLM4CardGame](https://github.com/THUDM/LLM4CardGame)
Guandan harness — `danserver` (game engine) plus four seat clients — and let
the fine-tuned LLM occupy seats 0 and 2 while the opponent occupies seats 1
and 3. Each **match** is played until one team wins the current level; each
match is broken into multiple **deals**, and every deal ends with a per-seat
reward in \{-3, -2, -1, +1, +2, +3\}. We record every reward and every LLM
action's wall-clock latency, then aggregate over ≥ 100 deals.

Reward semantics (from the LLM's perspective):

| Reward | Meaning |
| --- | --- |
| +3 | Double-down: LLM's team finished 1st + 2nd |
| +2 | LLM's team finished 1st and 3rd |
| +1 | LLM's team finished 1st and 4th |
| −1 | LLM's team finished 2nd and 4th |
| −2 | LLM's team finished 3rd and 4th |
| −3 | LLM's team was double-downed |

`Avg reward / deal` is the primary metric. Zero would indicate a policy
statistically indistinguishable from a coin flip; positive means net-winning
vs the given opponent.

## Harness architecture

```
                      ┌──────────────────────────┐
                      │ vLLM server (this repo)  │
                      │  Qwen2.5-14B + LoRA-9250 │
                      │  OpenAI API on :8552     │
                      └──────────┬───────────────┘
                                 │ HTTP (JSON)
                                 │
   ┌─────────┐  socket   ┌───────┴──────────┐  socket   ┌─────────────┐
   │seat 1   │◀─────────▶│ danserver (C++)  │◀─────────▶│seat 3       │
   │opponent │           │ (Danzero_plus/   │           │opponent     │
   └─────────┘           │  wintest/torch/) │           └─────────────┘
                         │                  │
   ┌─────────┐  socket   │                  │  socket   ┌─────────────┐
   │seat 0   │◀─────────▶│                  │◀─────────▶│seat 2       │
   │LLM      │           └──────────────────┘           │LLM          │
   │(actor_llm.py + client0.py)                         │(client2.py) │
   └─────────┘                                          └─────────────┘
```

The two LLM-controlled clients (`util/guandan_util/client0.py` and
`client2.py` from LLM4CardGame) format the game state as a chat prompt and
send it to the vLLM API. The three worker threads share a single vLLM instance
via HTTP.

## Opponents

- **AI4Card** (`third_party/Danzero_plus/wintest/ai4/`): a rule-based
  Guandan bot from the [`guandan_mcc`](https://github.com/AltmanD/guandan_mcc)
  competition kit. Weak baseline.
- **DanZero+** (`third_party/Danzero_plus/wintest/danzero/`): the RL policy
  from the DanZero+ paper. Strong baseline. **Requires a Python 3.7 +
  TensorFlow 1.15 environment** — point `TF_PY=` at its interpreter before
  running eval.

## Running

```bash
# vs AI4Card
bash scripts/eval.sh weights/checkpoint-9250 100 ai4      my_ai4_eval

# vs DanZero+
TF_PY=/path/to/tf115/bin/python \
  bash scripts/eval.sh weights/checkpoint-9250 100 danzero my_dz_eval

# summarize
python scripts/parse_results.py eval_logs/my_ai4_eval
python scripts/parse_results.py eval_logs/my_dz_eval
```

`scripts/eval.sh` starts vLLM once and then loops one danserver-match at a
time (working around a match-to-match hang in the upstream danserver). It
stops when it has accumulated `TARGET` deals or hits the `MAX_MATCHES` safety
cap (default 50; override with `MAX_MATCHES=100`).

## Output

Per-match log directory: `eval_logs/<tag>/match_<idx>/`

- `danserver.log` — game engine output; contains `次结束` (deal-end) markers.
- `log-client0*.txt` — JSONL with one line per deal containing `"reward": N`.
- `client0_wrapper.log`, `client2_wrapper.log` — client stdout.
- `actor_llm.log` — LLM worker stdout, including any API errors.
- `opp_c1.log`, `opp_c3.log`, `opp_actor.log` — opponent stdout.

The parser in `scripts/parse_results.py` reads all `log-client0*.txt` files
under the given tag directories and prints:

- deal count, positive/negative counts, win rate
- sum reward, avg reward, reward histogram
- LLM action latency: count, avg, median, P90, P99, max
- if multiple dirs are passed, it prints per-dir + aggregate

Latency is derived from the timestamps of successive
`client0 send` lines in `danserver.log` (1-second precision, capped at 120 s,
which handles day rollovers).

## Reproducing our numbers

Our checkpoint-9250 numbers below come from **102 deals** aggregated over
multiple 30-deal matches each. Numbers are seed-sensitive; expect ± ~ 5 pp on
win rate with 100 deals.

See [`RESULTS.md`](RESULTS.md) for the full tables.
