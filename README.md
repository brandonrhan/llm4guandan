# llm4guandan

**Qwen2.5-14B + LoRA fine-tuned for [Guandan](https://en.wikipedia.org/wiki/Guandan) (掼蛋), an imperfect-information Chinese card game.**

The LoRA adapter in this repo (`weights/checkpoint-9250/`, 66 MB) is applied on top
of `Qwen/Qwen2.5-14B-Instruct` and trained on 46 k Guandan playing traces
generated from the [Danzero+](https://github.com/submit-paper/Danzero_plus)
policy self-play environment. Training uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
(SFT, LoRA r=8 α=16 on all 7 projections, DeepSpeed ZeRO-3, bf16, 1 epoch).

At inference the LoRA is served via vLLM (`llamafactory-cli api`) with an
OpenAI-compatible endpoint, then plugged into the [LLM4CardGame](https://github.com/THUDM/LLM4CardGame)
Guandan client (`util/guandan_util/`) for full-game evaluation against the
[AI4Card](https://github.com/AltmanD/guandan_mcc) rule bot and the Danzero+ RL bot.

---

## Results (checkpoint-9250, 100 deals each opponent)

| Opponent | Deals | Win rate (pos/neg deals) | Sum reward | Avg reward / deal | Median LLM action latency | P99 latency |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| AI4Card (rule bot) | 102 | **65.7 %** (67 / 35) | **+98** | **+0.961** | 0 s | 7 s |
| Danzero+ (RL bot)  | 102 | 51.0 % (52 / 50)  | +7   | +0.069 | 0 s | 9 s |

Reward is per-deal from the LLM's seat (\{-3, -2, -1, +1, +2, +3\}). Latencies are
wall-clock time between successive LLM decisions on 4×RTX 4090 with vLLM
tensor-parallel-size 4.

See [`docs/RESULTS.md`](docs/RESULTS.md) for the full reward-distribution histogram.

---

## Quickstart — serve the model with vLLM

Requires a Linux GPU box with ≥ 4×24 GB (RTX 4090 / A100 / L40S / H100).

```bash
git clone https://github.com/brandonrhan/llm4guandan.git
cd llm4guandan
bash scripts/setup.sh            # creates .venv, installs vllm + peft + llamafactory
bash scripts/serve.sh            # starts OpenAI-compatible API on :8552
```

Then call it like OpenAI:

```bash
python examples/infer_openai_client.py
```

> **⚠️ Prompt format matters.** The model is SFT-trained on a fixed
> 133-line template with 13 numbered state slots and expects the reply as a
> single JSON `{"action": [Type, Rank, [Cards]]}` object. Deviating from that
> format silently degrades performance. Read **[`docs/PROMPT_FORMAT.md`](docs/PROMPT_FORMAT.md)**
> before writing your own caller. The exact template is checked in at
> [`prompt/prompt_guandan4.py`](prompt/prompt_guandan4.py) and a real training
> record is at [`prompt/sample_training_example.txt`](prompt/sample_training_example.txt).

Minimal correct call in Python (see `examples/infer_openai_client.py` for the
full state dict):

```python
import json
from openai import OpenAI
from prompt.prompt_guandan4 import prompt_guandan

state = {...}  # 13 fields — see docs/PROMPT_FORMAT.md
user_msg = prompt_guandan % (
    json.dumps(state["position"]),         json.dumps(state["hand"]),
    json.dumps(state["remaining_others"]), json.dumps(state["last_action_others"]),
    json.dumps(state["last_action_teammate"]), json.dumps(state["num_left"]),
    json.dumps(state["played_down"]),      json.dumps(state["played_teammate"]),
    json.dumps(state["played_up"]),        json.dumps(state["self_rank"]),
    json.dumps(state["opponent_rank"]),    json.dumps(state["current_rank"]),
    json.dumps(state["legal_actions"]),
)

c = OpenAI(base_url="http://localhost:8552/v1", api_key="local")
r = c.chat.completions.create(
    model="guandan",
    messages=[{"role": "user", "content": user_msg}],  # no custom system prompt
    temperature=0.0,
    max_tokens=256,
)
print(r.choices[0].message.content)  # -> {"action": ["Single", "9", ["H9"]]}
```

## Quickstart — inference without vLLM (single GPU / CPU offload)

```bash
pip install -r requirements.txt
python examples/infer_transformers.py
```

Loads Qwen2.5-14B-Instruct in bf16, applies the LoRA, generates a single reply.
Slow but does not need vLLM.

---

## Reproduce training

```bash
bash scripts/train.sh
```

You need:

- The full 46 k-sample training set (not shipped in this repo — it comes out of
  the Danzero+ self-play data pipeline, see [`docs/TRAINING.md`](docs/TRAINING.md)).
- A LLaMA-Factory checkout (auto-cloned by `scripts/setup.sh`).
- ≥ 6 GPUs for ZeRO-3 (we used 6×RTX 4090 24 GB, one epoch ≈ 4 days).

Hyper-parameters live in [`configs/train_lora.yaml`](configs/train_lora.yaml).

## Reproduce evaluation

```bash
bash scripts/eval.sh weights/checkpoint-9250 100 ai4      my_eval
bash scripts/eval.sh weights/checkpoint-9250 100 danzero  my_eval
python scripts/parse_results.py eval_logs/my_eval
```

The eval harness launches vLLM once, then loops through Guandan matches
(danserver + LLM4CardGame + AI4 or Danzero+ opponents), collecting per-deal
rewards and LLM action latencies.

Full details in [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## Repository layout

```
weights/checkpoint-9250/   LoRA adapter (66 MB, adapter_config.json + safetensors)
prompt/                    Exact training prompt template + a real sample
configs/                   Training / serving YAML + DeepSpeed ZeRO-3 config
scripts/                   setup.sh, train.sh, serve.sh, eval.sh, parse_results.py
examples/                  Standalone inference examples (use the real template)
patches/                   Files we changed in LLM4CardGame + a unified diff
docs/                      Deep-dive docs on training, eval, prompt format, results
assets/                    training_loss.png, train_results.json
```

## Credits & licenses

- Base model: [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) — Apache 2.0
- Training framework: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — Apache 2.0
- Guandan environment / opponents: [LLM4CardGame](https://github.com/THUDM/LLM4CardGame), [Danzero+](https://github.com/submit-paper/Danzero_plus), [guandan_mcc](https://github.com/AltmanD/guandan_mcc) — see their repos
- This repo (glue code + LoRA weights): Apache 2.0, see [`LICENSE`](LICENSE)

If you use the LoRA weights or the eval harness, please cite this repo and the
upstream LLM4CardGame / Qwen papers.
