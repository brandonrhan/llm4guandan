# LoRA checkpoint 9250

Trained for one full epoch (9,250 optimizer steps) on 46k Guandan traces.

## Files

- `adapter_config.json` — PEFT LoRA config. `base_model_name_or_path` points to
  `Qwen/Qwen2.5-14B-Instruct` on Hugging Face.
- `adapter_model.safetensors` — 66 MB LoRA weights.

## Hyper-parameters

- LoRA rank r = 8, alpha = 16, dropout = 0.0
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Base: [`Qwen/Qwen2.5-14B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- Precision: bf16
- Optimizer: AdamW (via LLaMA-Factory default), DeepSpeed ZeRO-3
- Learning rate: 1e-4, cosine, 10% warmup
- Effective batch: `1 (per-device) × 6 (GPUs) × 21 (grad-accum) = 126`
- Sequence length: 4096 (truncated)
- Epochs: 1.0 (9,250 steps)
- Hardware: 6×RTX 4090 24 GB (`CUDA_VISIBLE_DEVICES=0,1,2,3,6,7`)

## Prompt format

**Critical:** this adapter expects the exact 133-line prompt template it was
trained on. See [`../../docs/PROMPT_FORMAT.md`](../../docs/PROMPT_FORMAT.md)
for the full spec and [`../../prompt/prompt_guandan4.py`](../../prompt/prompt_guandan4.py)
for the template source. Do not invent your own system prompt or state schema —
the model is out-of-distribution otherwise.

## How to load

### With vLLM (production serving)

```yaml
# configs/serve_vllm.yaml
model_name_or_path: Qwen/Qwen2.5-14B-Instruct
adapter_name_or_path: weights/checkpoint-9250
template: qwen
finetuning_type: lora
infer_backend: vllm
```

Then `bash scripts/serve.sh`.

### With transformers + peft (single-process inference)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE = "Qwen/Qwen2.5-14B-Instruct"
tok  = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "weights/checkpoint-9250")
model.eval()
```
