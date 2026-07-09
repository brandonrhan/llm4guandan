"""Guandan GRPO reinforcement-learning package.

Modules:
    reward       reward table + per-deal reward parsing + Dr.GRPO advantages
    collect      run the eval harness as a rollout engine, build training samples
    grpo_loss    self-contained Dr.GRPO loss (no-std advantage, Clip-Higher, KL)
    train_grpo   training loop on the existing DeepSpeed ZeRO-3 + PEFT stack
"""
