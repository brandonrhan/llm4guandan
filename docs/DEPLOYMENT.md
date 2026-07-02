# Deployment tips

## Minimum hardware

- 4 × 24 GB GPU (RTX 4090 / L40S / A100 40 GB / H100) with vLLM
  tensor-parallel-size=4. Fits Qwen2.5-14B-Instruct in bf16 + LoRA + KV cache
  comfortably under 20 GB per card at `max_len=32768` and
  `gpu_memory_utilization=0.85`.
- 2 × 48 GB (A100 80 GB / H100 80 GB): reduce `tensor_parallel_size` to 2 in
  `configs/serve_vllm.yaml`.
- Single 48–80 GB card: works but bf16 barely fits, cut `vllm_maxlen` to
  8192.
- Single 24 GB card: use `examples/infer_transformers.py` (no vLLM), expect
  a few seconds per token.

## Env vars honored by `scripts/serve.sh`

- `CKPT` — path to LoRA adapter (default `weights/checkpoint-9250`).
- `API_PORT` — port to listen on (default 8552).
- `CUDA_VISIBLE_DEVICES` — GPU list (default `0,1,2,3`). Number of visible
  GPUs must equal `tensor_parallel_size` in `configs/serve_vllm.yaml`.

## Running headless / systemd

`scripts/serve.sh` runs in the foreground. To daemonize:

```bash
nohup bash scripts/serve.sh > logs/serve.log 2>&1 &
```

or wrap it in a systemd unit; `llamafactory-cli api` handles SIGTERM cleanly.

## Health check

```bash
curl -s http://localhost:8552/v1/models
# → {"object":"list","data":[{"id":"guandan","object":"model", ...}]}

curl -s http://localhost:8552/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"guandan","messages":[{"role":"user","content":"hi"}],"max_tokens":10}'
```

## Common gotchas

- **`vllm_max_lora_rank: 8`** in the config *must* be ≥ your adapter's rank.
  Ours is r=8; leave it as 8 unless you retrain with larger r.
- **`template: qwen`** must match the base model's chat template; changing
  the base to a non-Qwen model without updating this will produce garbage.
- **Prefix caching**: `enable_prefix_caching: true` gives big speed-ups on
  the Guandan prompts (long shared system + state prefix); keep it on.
- **`API_TEMP`** (used by the LLM4CardGame clients) defaults to 0 for
  deterministic play. Set `API_TEMP=0.7` for stochastic play (worse
  results in our eval).
