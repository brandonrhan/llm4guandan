# Training

## Overview

We SFT a LoRA adapter (r=8, α=16, all 7 projections) on top of
`Qwen/Qwen2.5-14B-Instruct` for one epoch on 46,150 Guandan
`(state → best_action)` examples.

- **Framework**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
  (`stage=sft`, DeepSpeed ZeRO-3, bf16, gradient-checkpointing).
- **Hardware**: 6 × NVIDIA RTX 4090 24 GB (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`).
- **Runtime**: ≈ 4 days for 9,250 steps.

Config: [`configs/train_lora.yaml`](../configs/train_lora.yaml).

## Data

The training set is *not shipped* with this repo (it is regenerated from a
self-play environment). To reproduce:

1. Follow the [`Danzero+`](https://github.com/submit-paper/Danzero_plus) README
   to build the `wintest/` Guandan self-play harness (this needs Python 3.7 +
   TensorFlow 1.15; keep it in a separate conda env, e.g. `tf115`).
2. Run self-play, dumping `(state, action)` pairs to jsonl:
   ```bash
   bash scripts/gen_guandan_data.sh <num_games> <tag>
   ```
   (This is a thin wrapper around Danzero+'s `torch/danserver` + `danzero`
   clients that we used at PKU; adapt paths for your setup.)
3. Convert to LLaMA-Factory jsonl format
   (`{"instruction": ..., "output": ...}`) and add an entry to
   `third_party/LLaMA-Factory/data/dataset_info.json`:
   ```json
   "guandan_full": {
     "file_name": "guandan_full.jsonl",
     "columns": { "prompt": "instruction", "response": "output" }
   }
   ```
4. Point `configs/train_lora.yaml` at your file
   (`dataset: guandan_full`, `dataset_dir: <path>`).

Each sample is one turn observed from a self-play match: player hand, previous
plays, last legal-action set — target is the Danzero+ policy's chosen action
rendered as a card string. Cutoff length is 4096 tokens.

## Hyper-parameters

| Field | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-14B-Instruct` |
| Finetuning type | LoRA |
| LoRA rank / alpha / dropout | 8 / 16 / 0.0 |
| LoRA target modules | q, k, v, o, gate, up, down |
| Precision | bf16 |
| DeepSpeed | ZeRO-3, gradient-checkpointing |
| Optimizer | AdamW (LLaMA-Factory default) |
| Learning rate / schedule | 1e-4 / cosine, 10% warmup |
| Per-device batch | 1 |
| Grad accum | 21 |
| Effective batch | 126 (6 × 1 × 21) |
| Sequence length | 4096 |
| Epochs | 1.0 |
| Total steps | 9,250 |
| Save every | 500 steps (`save_total_limit=8`) |

## Running

```bash
bash scripts/setup.sh --train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/train.sh
```

The launcher auto-detects the latest checkpoint in `runs/full_14b/` and passes
`resume_from_checkpoint=<latest>` to `llamafactory-cli train`, so it's safe to
kill and restart.

## Loss curve

See [`assets/training_loss.png`](../assets/training_loss.png).
