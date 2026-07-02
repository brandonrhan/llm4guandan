"""
Minimal inference example — call the fine-tuned Guandan LLM via OpenAI client.

Assumes vLLM is already serving on localhost:8552 (see scripts/serve.sh).
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8552/v1", api_key="local")

SYSTEM = (
    "You are an expert Guandan (掼蛋) player. "
    "Given the current game state, output only the single best action as a "
    "space-separated list of cards (e.g. 'SA HK D5') or 'pass'."
)

USER = (
    "Current level: 2. You are seat 0.\n"
    "Your hand: S3 S4 S5 S6 S7 H8 H8 D9 CT CJ CQ SK SA HA D2 D2\n"
    "Last play (seat 3): H5 H5\n"
    "Legal actions include: pass, S3, ...\n"
    "What do you play?"
)

resp = client.chat.completions.create(
    model="guandan",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ],
    temperature=0.0,
    max_tokens=256,
)

print(resp.choices[0].message.content)
