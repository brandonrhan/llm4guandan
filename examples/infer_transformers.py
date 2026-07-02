"""
Standalone inference example — loads Qwen2.5-14B-Instruct + our LoRA adapter
directly with transformers + peft. No vLLM needed.

Requires ~30 GB GPU memory in bf16 (or use device_map='auto' with offload).

The prompt format matters — the model is SFT-trained on a fixed 133-line
template. See docs/PROMPT_FORMAT.md for the full spec. This script renders
that exact template and wraps it with the Qwen chat template.
"""
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.infer_openai_client import STATE, build_user_message

BASE = "Qwen/Qwen2.5-14B-Instruct"
ADAPTER = "weights/checkpoint-9250"


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    user_msg = build_user_message(STATE)

    # Qwen's chat template inserts the default system message
    # ("You are a helpful assistant.") automatically. That matches training.
    messages = [{"role": "user", "content": user_msg}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
        )

    reply = tok.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    print("raw reply:", reply)

    action = json.loads(reply)["action"]
    print("parsed action:", action)


if __name__ == "__main__":
    main()
