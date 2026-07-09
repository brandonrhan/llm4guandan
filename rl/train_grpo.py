"""Guandan Dr.GRPO training loop on the existing DeepSpeed ZeRO-3 + PEFT stack.

Why not TRL: the server ships TRL 0.9.6 (predates ``GRPOTrainer``) on a
bleeding-edge torch build that already runs the SFT/vLLM stack; upgrading risks
breaking it.  Our rollout is fully custom anyway (danserver-driven), so we apply
the identical modern recipe directly:

    Dr.GRPO advantage (no std)  +  Clip-Higher  +  small KL to frozen SFT

Launch (2 trainer GPUs, ZeRO-3), with a hot vLLM server already serving the
current LoRA on $API_PORT (see scripts/train_grpo.sh):

    accelerate launch --config_file configs/accel_ds_z3.yaml \
        rl/train_grpo.py --config configs/grpo_v1.yaml

Distributed model: rank 0 runs the rollout harness (which talks to the external
vLLM), writes samples to a shared file; every rank reads its shard and does a
ZeRO-3 data-parallel update.  The reference policy is a second, frozen PEFT
adapter over the same base model (LoRA-delta trick) — no second 14B in memory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Works whether launched as a script (`accelerate launch rl/train_grpo.py`) or a
# module: ensure the repo root is importable so `rl.*` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.collect import Sample, build_samples  # noqa: E402
from rl.grpo_loss import grpo_loss  # noqa: E402


# --------------------------------------------------------------------------- #
# model + reference adapter
# --------------------------------------------------------------------------- #
def load_policy_and_ref(cfg: dict, tokenizer):
    """Base model + trainable 'default' adapter + frozen 'ref' adapter (both 9250)."""
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "flash_attention_2"),
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    # trainable policy adapter (initialized from the SFT checkpoint)
    model = PeftModel.from_pretrained(
        base, cfg["adapter_name_or_path"], adapter_name="default", is_trainable=True
    )
    # frozen reference = same SFT weights, never updated
    model.load_adapter(cfg["adapter_name_or_path"], adapter_name="ref")
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


# --------------------------------------------------------------------------- #
# per-token log-probs of the completion, under the active adapter
# --------------------------------------------------------------------------- #
def _encode(tokenizer, prompt: str, completion: str, max_len: int):
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=True,
    )
    comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    comp_ids = comp_ids + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + comp_ids)[:max_len]
    n_comp = min(len(comp_ids), max(0, len(input_ids) - len(prompt_ids)))
    # mask marks which *input* positions are completion tokens
    comp_mask = [0] * (len(input_ids) - n_comp) + [1] * n_comp
    return input_ids, comp_mask


def completion_logps(model, tokenizer, batch: List[Sample], device,
                     max_len: int, comp_norm_len: int):
    encoded = [_encode(tokenizer, s.prompt, s.completion, max_len) for s in batch]
    width = max(len(ids) for ids, _ in encoded)
    pad = tokenizer.pad_token_id

    input_ids, attn, cmask = [], [], []
    for ids, m in encoded:
        p = width - len(ids)
        input_ids.append(ids + [pad] * p)
        attn.append([1] * len(ids) + [0] * p)
        cmask.append(m + [0] * p)

    input_ids = torch.tensor(input_ids, device=device)
    attn = torch.tensor(attn, device=device)
    cmask = torch.tensor(cmask, dtype=torch.float32, device=device)

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]              # predict token t from t-1
    targets = input_ids[:, 1:]
    logp_all = F.log_softmax(logits.float(), dim=-1)
    logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    mask = cmask[:, 1:]                          # align mask to prediction targets

    # right-pad every row to comp_norm_len so grpo_loss divides by a constant
    T = logp.shape[1]
    if T < comp_norm_len:
        padw = comp_norm_len - T
        logp = F.pad(logp, (0, padw))
        mask = F.pad(mask, (0, padw))
    return logp, mask


# --------------------------------------------------------------------------- #
# vLLM sync (runtime LoRA hot-swap)
# --------------------------------------------------------------------------- #
def sync_lora_to_vllm(cfg: dict, adapter_dir: str):
    """Reload the freshly-saved LoRA into the running vLLM server.

    Requires vLLM launched with runtime LoRA updating (see train_grpo.sh). Falls
    back to a no-op with a warning if the endpoint is unavailable — training
    still proceeds (next rollout just uses the previous policy).
    """
    import requests

    base = f"http://0.0.0.0:{cfg.get('api_port', 8552)}"
    name = cfg.get("vllm_lora_name", "guandan")
    try:
        requests.post(f"{base}/v1/unload_lora_adapter",
                      json={"lora_name": name}, timeout=30)
    except Exception:
        pass
    try:
        r = requests.post(f"{base}/v1/load_lora_adapter",
                          json={"lora_name": name, "lora_path": adapter_dir}, timeout=60)
        r.raise_for_status()
    except Exception as exc:
        print(f"[sync] vLLM LoRA reload failed ({exc}); "
              f"rollout will use previous policy this step", flush=True)


# --------------------------------------------------------------------------- #
# rollout -> shared samples -> per-rank shard
# --------------------------------------------------------------------------- #
def gather_samples(accelerator: Accelerator, cfg: dict, step: int) -> List[Sample]:
    traj_dir = os.path.join(cfg["output_dir"], f"rollout_step{step}")
    shared = os.path.join(traj_dir, "samples.jsonl")

    if accelerator.is_main_process:
        os.makedirs(traj_dir, exist_ok=True)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, API_TEMP=str(cfg["rollout"]["temperature"]))
        import subprocess
        subprocess.run(
            ["bash", os.path.join(root, "scripts/rollout.sh"),
             str(cfg["rollout"]["num_deals_per_batch"]), traj_dir,
             cfg["rollout"]["opponent"], str(cfg["rollout"]["temperature"])],
            check=True, env=env,
        )
        samples, stats = build_samples(traj_dir)
        with open(shared, "w", encoding="utf-8") as fout:
            for s in samples:
                fout.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")
        print(f"[step {step}] rollout stats: {json.dumps(stats)}", flush=True)

    accelerator.wait_for_everyone()
    samples: List[Sample] = []
    with open(shared, "r", encoding="utf-8") as fin:
        for line in fin:
            samples.append(Sample(**json.loads(line)))
    # data-parallel shard for this rank
    return samples[accelerator.process_index::accelerator.num_processes]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as fin:
        cfg = yaml.safe_load(fin)

    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_policy_and_ref(cfg, tokenizer)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["learning_rate"]),
    )
    model, optim = accelerator.prepare(model, optim)

    micro_bs = cfg.get("micro_batch_size", 4)
    max_len = cfg.get("max_seq_len", 4096)
    comp_norm = cfg.get("loss_norm_const", 256)
    loss_cfg = cfg["loss"]
    device = accelerator.device

    for step in range(1, cfg["max_steps"] + 1):
        model.eval()
        samples = gather_samples(accelerator, cfg, step)
        # dynamic sampling: drop zero-advantage decisions (no gradient signal)
        samples = [s for s in samples if abs(s.advantage) > 1e-9]
        if not samples:
            accelerator.print(f"[step {step}] no non-zero-advantage samples, skip")
            continue

        model.train()
        step_metrics: Dict[str, float] = {}
        n_micro = 0
        optim.zero_grad()
        for i in range(0, len(samples), micro_bs):
            micro = samples[i:i + micro_bs]
            adv = torch.tensor([s.advantage for s in micro], device=device)

            unwrapped = accelerator.unwrap_model(model)
            # reference log-probs (frozen adapter, no grad)
            with torch.no_grad():
                unwrapped.set_adapter("ref")
                logp_ref, _ = completion_logps(model, tokenizer, micro, device, max_len, comp_norm)
            # policy log-probs (trainable adapter, with grad)
            unwrapped.set_adapter("default")
            logp, mask = completion_logps(model, tokenizer, micro, device, max_len, comp_norm)
            logp_old = logp.detach()   # num_iterations == 1 -> ratio == 1

            loss, m = grpo_loss(
                logp, logp_old, logp_ref, adv, mask,
                beta=float(loss_cfg["beta"]),
                eps_low=float(loss_cfg["epsilon"]),
                eps_high=float(loss_cfg["epsilon_high"]),
                loss_norm_const=float(comp_norm),
            )
            accelerator.backward(loss)
            n_micro += 1
            for k, v in m.items():
                step_metrics[k] = step_metrics.get(k, 0.0) + v

        if cfg.get("max_grad_norm"):
            accelerator.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
        optim.step()

        if accelerator.is_main_process and n_micro:
            avg = {k: round(v / n_micro, 5) for k, v in step_metrics.items()}
            print(f"[step {step}] {json.dumps(avg)} n_samples={len(samples)}", flush=True)

        # checkpoint + push the new LoRA to vLLM
        if step % cfg.get("save_steps", 50) == 0 or step == cfg["max_steps"]:
            adapter_dir = os.path.join(cfg["output_dir"], f"checkpoint-{step}")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.unwrap_model(model).save_pretrained(
                    adapter_dir, selected_adapters=["default"]
                )
                sync_lora_to_vllm(cfg, adapter_dir)
        else:
            # lightweight per-step sync so the next rollout is on-policy
            adapter_dir = os.path.join(cfg["output_dir"], "checkpoint-latest")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.unwrap_model(model).save_pretrained(
                    adapter_dir, selected_adapters=["default"]
                )
                sync_lora_to_vllm(cfg, adapter_dir)

    accelerator.print("training done")


if __name__ == "__main__":
    main()
