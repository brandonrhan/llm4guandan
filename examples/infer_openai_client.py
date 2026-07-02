"""
Minimal inference example — call the fine-tuned Guandan LLM via OpenAI client.

Assumes vLLM is already serving on localhost:8552 (see scripts/serve.sh).

The prompt format matters — the model is SFT-trained on a fixed 133-line
template. See docs/PROMPT_FORMAT.md for the full spec. This script renders
that exact template.
"""
import json
import sys
from pathlib import Path

# Make `prompt/` importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompt.prompt_guandan4 import prompt_guandan

from openai import OpenAI


def build_user_message(obs: dict) -> str:
    """Render an observation dict into the exact prompt the model was trained on.

    The 13 field names below are the *upstream* rlcard-env keys used by
    `convert_data.py::convert_guandan` (training) and
    `util/prompt_util.py::prompt_function_guandan` (live inference).
    They are byte-for-byte identical in both places, so you can pass
    `state['raw_obs']` from the live env directly in.

    See docs/PROMPT_FORMAT.md for the schema of every field. Order matters.
    """
    return prompt_guandan % (
        json.dumps(obs["my_pos"]),
        json.dumps(obs["my_hands"]),
        json.dumps(obs["remaining_hands"]),
        json.dumps(obs["last_action"]),
        json.dumps(obs["last_teammate_action"]),
        json.dumps(obs["number_of_cards_left"]),
        json.dumps(obs["down_played_cards"]),
        json.dumps(obs["teammate_played_cards"]),
        json.dumps(obs["up_played_cards"]),
        json.dumps(obs["self_rank"]),
        json.dumps(obs["oppo_rank"]),
        json.dumps(obs["cur_rank"]),
        json.dumps(obs["legal_actions"]),
    )


# A real training-distribution observation (record #0 of the SFT jsonl).
# Key names match the upstream env exactly — this is the same dict shape
# that `state['raw_obs']` gives you at inference time.
STATE = {
    "my_pos": 2,
    "my_hands": ["C2", "D2", "S5", "S6", "S6", "S7", "D7", "D7", "D9", "DT",
                 "SJ", "HJ", "CJ", "CJ", "DJ", "HQ", "CQ", "SK", "CK", "SA",
                 "HA", "CA", "DA", "DA", "C8", "D8", "HR"],
    "remaining_hands": [
        "H2", "H3", "H3", "H4", "H5", "H5", "H6", "H6", "H7", "H7",
        "H8", "H8", "H9", "H9", "HT", "HT", "HJ", "HQ", "HK", "HK", "HA",
        "S2", "S2", "S3", "S3", "S4", "S4", "S5", "S7", "S8", "S8",
        "S9", "S9", "ST", "ST", "SJ", "SQ", "SQ", "SK", "SA",
        "C2", "C3", "C3", "C4", "C4", "C5", "C6", "C6", "C7", "C7",
        "C8", "C9", "C9", "CT", "CT", "CQ", "CK", "CA",
        "D2", "D3", "D3", "D4", "D4", "D5", "D5", "D6", "D6", "D8", "D9",
        "DT", "DJ", "DQ", "DQ", "DK", "DK", "SB", "SB", "HR",
    ],
    "last_action": ["C5"],
    "last_teammate_action": ["H4"],
    "number_of_cards_left": {"0": 26, "1": 26, "2": 27, "3": 26},
    "down_played_cards": ["H2"],
    "teammate_played_cards": ["H4"],
    "up_played_cards": ["C5"],
    "self_rank": "2",
    "oppo_rank": "8",
    "cur_rank": "8",
    "legal_actions": [
        ["PASS", "PASS", "PASS"],
        ["Single", "6", ["S6"]],
        ["Single", "7", ["S7"]],
        ["Single", "J", ["SJ"]],
        ["Single", "A", ["SA"]],
        ["Single", "R", ["HR"]],
        ["Bomb", "J", ["SJ", "HJ", "CJ", "DJ"]],
        ["Bomb", "A", ["SA", "HA", "CA", "DA"]],
        ["StraightFlush", "7", ["D7", "D8", "D9", "DT", "DJ"]],
    ],
}


if __name__ == "__main__":
    user_msg = build_user_message(STATE)

    client = OpenAI(base_url="http://localhost:8552/v1", api_key="local")
    resp = client.chat.completions.create(
        model="guandan",
        # Do NOT add a custom system prompt — training used the Qwen default
        # ("You are a helpful assistant."), which the server-side chat template
        # inserts automatically. Overriding it is out-of-distribution.
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.0,
        max_tokens=256,
    )

    reply = resp.choices[0].message.content
    print("raw reply:", reply)

    action = json.loads(reply)["action"]
    print("parsed action:", action)
    # Expected for this state: ["StraightFlush", "7", ["D7","D8","D9","DT","DJ"]]
