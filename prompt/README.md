# The exact prompt template

`prompt_guandan4.py` is the Guandan prompt template used during SFT.
It was copied verbatim from
[THUDM/LLM4CardGame](https://github.com/THUDM/LLM4CardGame/blob/34c2785/prompt/prompt_guandan4.py)
at commit `34c2785`. Apache 2.0 licensed by the upstream authors.

The single exported symbol is `prompt_guandan`, a Python `%`-format string
with 13 `%s` slots corresponding to the 13 game-state fields. See
[`../docs/PROMPT_FORMAT.md`](../docs/PROMPT_FORMAT.md) for the full spec
and a reference helper.

`sample_training_example.txt` is a real, unmodified training record from
`LLaMA-Factory/data/guandan_full.jsonl` (index 0) showing exactly what the
model saw during training — a 6.7 kB `instruction` and a one-line JSON
`output`.
