"""
Standalone inference example — loads Qwen2.5-14B-Instruct + our LoRA adapter
directly with transformers + peft. No vLLM needed.

Requires ≈ 30 GB GPU memory in bf16 (or use device_map='auto' + offload).
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-14B-Instruct"
ADAPTER = "weights/checkpoint-9250"

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "You are an expert Guandan (掼蛋) player. Given the current game "
            "state, output only the single best action as a space-separated "
            "list of cards (e.g. 'SA HK D5') or 'pass'."
        ),
    },
    {
        "role": "user",
        "content": (
            "Current level: 2. You are seat 0.\n"
            "Your hand: S3 S4 S5 S6 S7 H8 H8 D9 CT CJ CQ SK SA HA D2 D2\n"
            "Last play (seat 3): H5 H5\n"
            "What do you play?"
        ),
    },
]

text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=256, do_sample=False, temperature=0.0
    )
reply = tok.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
print(reply)
