"""Self-contained Dr.GRPO loss (no TRL dependency).

Implements the objective used by TRL's ``loss_type="dr_grpo"`` + Clip-Higher,
so we get the exact modern recipe without upgrading TRL (server ships 0.9.6,
which predates ``GRPOTrainer``):

    l_{i,t} = min( r * A_i , clip(r, 1-eps_low, 1+eps_high) * A_i ) - beta * KL
    r       = exp(logp - logp_old)                       (importance ratio)
    KL      = exp(logp_ref - logp) - (logp_ref - logp) - 1   (Schulman k3, >=0)
    L       = - (1 / (G * L_const)) * sum_{i,t} l_{i,t}   (Dr.GRPO length-free)

Design choices (all from the papers you approved):
  * advantages already have NO std normalization (Dr.GRPO, arXiv:2503.20783)
    — done upstream in rl/reward.py.
  * dividing by a constant ``L_const`` (not per-sequence length) removes the
    response-length bias the same paper identifies.
  * asymmetric clip ``eps_high > eps_low`` is DAPO Clip-Higher (2503.14476),
    which curbs entropy collapse in small action spaces (relevant for cards).
  * ``beta > 0`` keeps a small KL leash to the frozen SFT reference, protecting
    the valuable 65.7%-win policy from drift (your KL scheme A).

With ``num_iterations == 1`` (one optimizer step per rollout batch) the caller
passes ``logp_old = logp.detach()`` so ``r == 1`` and the clip is inactive; the
clip only bites when reusing a batch for multiple inner steps.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def grpo_loss(
    logp: torch.Tensor,          # [B, T] per-token logprob of chosen completion tokens
    logp_old: torch.Tensor,      # [B, T] logprob under the behavior policy (detached)
    logp_ref: torch.Tensor,      # [B, T] logprob under the frozen reference (detached)
    advantages: torch.Tensor,    # [B]    per-sequence advantage
    completion_mask: torch.Tensor,  # [B, T] 1 for real completion tokens, 0 for pad
    beta: float = 0.01,
    eps_low: float = 0.20,
    eps_high: float = 0.28,
    loss_norm_const: float = 256.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    adv = advantages.unsqueeze(1)                       # [B,1]
    ratio = torch.exp(logp - logp_old)                  # [B,T]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
    pg = -torch.min(surr1, surr2)                       # [B,T]

    if beta > 0.0:
        diff = logp_ref - logp
        kl = torch.exp(diff) - diff - 1.0               # >= 0, Schulman k3
        per_token = pg + beta * kl
    else:
        kl = torch.zeros_like(pg)
        per_token = pg

    per_token = per_token * completion_mask
    n_seq = completion_mask.shape[0]
    loss = per_token.sum() / (n_seq * loss_norm_const)

    with torch.no_grad():
        tok = completion_mask.sum().clamp_min(1.0)
        clipped = ((ratio > 1.0 + eps_high) | (ratio < 1.0 - eps_low)) * completion_mask
        metrics = {
            "loss": float(loss.detach()),
            "kl": float((kl * completion_mask).sum() / tok),
            "ratio_mean": float((ratio * completion_mask).sum() / tok),
            "clip_frac": float(clipped.sum() / tok),
            "adv_mean": float(advantages.mean()),
            "adv_abs_mean": float(advantages.abs().mean()),
        }
    return loss, metrics


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 4, 8
    lp = torch.randn(B, T, requires_grad=True)
    old = lp.detach().clone()            # mu==1: ratio == 1
    ref = lp.detach().clone()            # KL == 0 when ref == policy
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    mask = torch.ones(B, T)
    loss, m = grpo_loss(lp, old, ref, adv, mask, beta=0.01)
    loss.backward()
    # ratio==1, ref==policy -> KL 0, clip_frac 0, loss == -mean_token(adv)/L_const
    assert abs(m["kl"]) < 1e-6, m
    assert abs(m["clip_frac"]) < 1e-6, m
    assert abs(m["ratio_mean"] - 1.0) < 1e-6, m
    assert lp.grad is not None
    print("grpo_loss.py self-check OK:", {k: round(v, 4) for k, v in m.items()})
