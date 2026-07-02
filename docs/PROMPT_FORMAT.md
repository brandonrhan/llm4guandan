# Prompt format — **THIS IS CRITICAL**

The model in `weights/checkpoint-9250` was SFT-trained on a **fixed** prompt
template. Deviating from this template at inference (wrong field names, wrong
order, wrong card notation, wrong output schema) will silently degrade
performance. Read this doc carefully before writing your own callers.

The exact template is checked into this repo verbatim at
[`prompt/prompt_guandan4.py`](../prompt/prompt_guandan4.py) (Apache 2.0, from
THUDM/LLM4CardGame at commit `34c2785`). A real training example is at
[`prompt/sample_training_example.txt`](../prompt/sample_training_example.txt).

---

## 1. Chat-turn layering

Training used LLaMA-Factory's `template: qwen`, which wraps every SFT sample
as a single-turn conversation. On the wire (what the tokenizer sees) each
example is:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<< the full prompt_guandan template, rendered with 13 slots filled >>
<|im_end|>
<|im_start|>assistant
{"action": [Type, Rank, [Cards]]}<|im_end|>
```

- The **system** message is the Qwen default `You are a helpful assistant.`
  and is emitted by the `qwen` chat template automatically. It was **not**
  overridden during training — the SFT jsonl has no `system` column.
- The **user** message is the entire rendered prompt (typically 4–7 kB of
  text, dominated by the fixed rules preamble).
- The **assistant** message is a single JSON object — that's the training
  target and that's what the model learned to emit.

If you use `tokenizer.apply_chat_template(messages, ...)` with `messages =
[{"role": "user", "content": <rendered prompt>}]`, the Qwen template will
add its default system for you. Either passing no system, or explicitly
passing `You are a helpful assistant.`, is fine — do **not** invent a
different system prompt like "You are an expert Guandan player" (that's what
my earlier example did — wrong).

## 2. The `user` message body — the 133-line template

The rendered user message is produced by the following Python template
(reproduced in [`prompt/prompt_guandan4.py`](../prompt/prompt_guandan4.py)):

```python
prompt_guandan = '''You are now a player in a game of Guandan. The game rules are as follows:

... [long fixed preamble covering rules, card notation, card-type schema] ...

Your task is to make the best decision in each playing round.
I will provide you with the following information:

1. Your position:
%s

2. Your current hand:
%s

3. Remaining cards of other players:
%s

4. Last action of other players:
%s

5. Last action of the teammate:
%s

6. Number of cards left for other players:
%s

7. Cards played by the down player:
%s

8. Cards played by the teammate:
%s

9. Cards played by the up player:
%s

10. Self rank:
%s

11. Opponent rank:
%s

12. Current rank:
%s

13. Legal actions:
%s

Please tell me your action in JSON format based on the provided information.
The JSON should contain an "action" key with a value chose from legal actions.

Output format examples:
Playing a card: {"action": ["Single", "9", ["H9"]]}

Please provide the corresponding JSON action based on the given information.
'''
```

Rendering rule: **every `%s` slot is filled with `json.dumps(value)`**. So
strings become double-quoted JSON strings, lists become JSON arrays,
dicts become JSON objects. Example:

```python
rendered = prompt_guandan % (
    json.dumps("2"),                                # 1. position — a str
    json.dumps(["C2","D2","S5","S6",...]),          # 2. hand — list of card strs
    json.dumps(["H2","H3","H3",...]),               # 3. remaining — list
    json.dumps(["C5"]),                             # 4. last action of other players
    json.dumps(["H4"]),                             # 5. last action of teammate
    json.dumps({"0":26,"1":26,"2":27,"3":26}),      # 6. num cards left — dict {seat: n}
    json.dumps(["H2"]),                             # 7. played by down player
    json.dumps(["H4"]),                             # 8. played by teammate
    json.dumps(["C5"]),                             # 9. played by up player
    json.dumps("2"),                                # 10. self rank — str
    json.dumps("8"),                                # 11. opponent rank — str
    json.dumps("8"),                                # 12. current rank — str
    json.dumps([["PASS","PASS","PASS"], ["Single","6",["S6"]], ...]),  # 13. legal actions
)
```

**Field-by-field spec:**

| # | Field | Type | Example |
| --- | --- | --- | --- |
| 1 | Your position | `str` "0" – "3" | `"2"` |
| 2 | Your current hand | list of 2-char card strings | `["C2","D2","S5","HR"]` |
| 3 | Remaining cards of other players | list of 2-char card strings | `["H2","H3",...]` |
| 4 | Last action of other players | list of card strings (empty list = they passed / no action yet) | `["C5"]` or `[]` |
| 5 | Last action of the teammate | list of card strings | `["H4"]` or `[]` |
| 6 | Number of cards left for other players | dict `{seat: count}` for seats "0","1","2","3" | `{"0":26,"1":26,"2":27,"3":26}` |
| 7 | Cards played by the down player | list of card strings (accumulated this deal) | `["H2","S5"]` |
| 8 | Cards played by the teammate | list of card strings | `["H4","CQ","DQ"]` |
| 9 | Cards played by the up player | list of card strings | `["C5","SK","HK","HA"]` |
| 10 | Self rank | str, one of `"A","2",...,"K"` | `"2"` |
| 11 | Opponent rank | str | `"8"` |
| 12 | Current rank | str (the *deal-level* card — the "级牌") | `"8"` |
| 13 | Legal actions | list of `[Type, Rank, [Cards]]` triples | see below |

**Card notation** (from the training preamble):

- **Suits**: `S`=Spade, `H`=Heart, `C`=Club, `D`=Diamond.
  Small joker uses suit `S`, big joker uses suit `H`.
- **Ranks**: `A`, `2`, `3`, …, `9`, `T` (for 10), `J`, `Q`, `K`.
  Small joker rank = `B`, big joker rank = `R`. `PASS` marks a pass.

So `"S2"` = Spade 2, `"HQ"` = Heart Queen, `"SB"` = small joker,
`"HR"` = big joker.

**Card-type triple** (used in slot 13 and in the output):
`[Type, Rank, Cards]` where

- `Type ∈ {"Single","Pair","Trips","ThreePair","ThreeWithTwo","TripsPair","Straight","Bomb","StraightFlush","PASS","tribute","back"}`
- `Rank ∈ {"A","2","3",...,"K","B","R","PASS"}` (rank of the highest / representative card)
- `Cards` is the actual list of card strings that make the combo

Examples:

- Single Diamond 5: `["Single","5",["D5"]]`
- Pair of 4s: `["Pair","4",["H4","C4"]]`
- Pass: `["PASS","PASS","PASS"]`
- Straight-flush 7♦8♦9♦T♦J♦: `["StraightFlush","7",["D7","D8","D9","DT","DJ"]]`

## 3. The `assistant` output

Single JSON object, one line, no code fences:

```json
{"action": [Type, Rank, [Cards]]}
```

The value of `"action"` must be one of the triples that appeared verbatim in
the `Legal actions` slot (#13) of the input. The model was trained to pick
from that set only.

Passing looks like: `{"action": ["PASS","PASS","PASS"]}`.

## 4. A real full training example

Verbatim from `LLaMA-Factory/data/guandan_full.jsonl` (record #0), the input
`instruction` field is **6,685 characters** — roughly 5 kB of fixed rules
preamble + 13 filled slots. See
[`prompt/sample_training_example.txt`](../prompt/sample_training_example.txt)
for the complete text. The state portion is:

```
1. Your position:
2

2. Your current hand:
["C2", "D2", "S5", "S6", "S6", "S7", "D7", "D7", "D9", "DT", "SJ", "HJ", "CJ",
 "CJ", "DJ", "HQ", "CQ", "SK", "CK", "SA", "HA", "CA", "DA", "DA", "C8", "D8", "HR"]

3. Remaining cards of other players:
["H2", "H3", ...  (78 entries total)]

4. Last action of other players:
["C5"]

5. Last action of the teammate:
["H4"]

6. Number of cards left for other players:
{"0": 26, "1": 26, "2": 27, "3": 26}

7. Cards played by the down player:
["H2"]

8. Cards played by the teammate:
["H4"]

9. Cards played by the up player:
["C5"]

10. Self rank:
"2"

11. Opponent rank:
"8"

12. Current rank:
"8"

13. Legal actions:
[["PASS", "PASS", "PASS"], ["Single", "6", ["S6"]], ..., ["StraightFlush", "7", ["D7", "D8", "D9", "DT", "DJ"]]]
```

And the target `output` is a single line:

```
{"action": ["StraightFlush", "7", ["D7", "D8", "D9", "DT", "DJ"]]}
```

## 5. Reference helper

The correct way to call the model from Python is:

```python
import json
from prompt.prompt_guandan4 import prompt_guandan

def build_user_message(state: dict) -> str:
    """Render the state into the exact prompt the model was trained on."""
    return prompt_guandan % (
        json.dumps(state["position"]),
        json.dumps(state["hand"]),
        json.dumps(state["remaining_others"]),
        json.dumps(state["last_action_others"]),
        json.dumps(state["last_action_teammate"]),
        json.dumps(state["num_left"]),
        json.dumps(state["played_down"]),
        json.dumps(state["played_teammate"]),
        json.dumps(state["played_up"]),
        json.dumps(state["self_rank"]),
        json.dumps(state["opponent_rank"]),
        json.dumps(state["current_rank"]),
        json.dumps(state["legal_actions"]),
    )

# Then wrap in a single-turn chat message and call the model.
```

Both [`examples/infer_openai_client.py`](../examples/infer_openai_client.py)
and [`examples/infer_transformers.py`](../examples/infer_transformers.py)
implement exactly this and are the authoritative reference.

## 6. Common pitfalls

1. **Don't invent a system prompt.** The model was trained with the Qwen
   default "You are a helpful assistant." Anything else is out-of-distribution.
2. **Don't shorten the preamble.** The 5 kB of rules text at the top of every
   sample is part of the trained distribution — the model expects to see it.
   vLLM's `enable_prefix_caching: true` makes the repeated preamble almost
   free at serving time, so there's no cost benefit to trimming it.
3. **Don't reorder slots.** They are numbered 1–13; the model learned that
   ordering.
4. **Don't change card notation.** Use `T` (not `10`), `B`/`R` for jokers
   with suits `S`/`H` respectively.
5. **Don't emit anything but the JSON action.** The training target is
   *just* `{"action": ...}` — no reasoning, no code fences, no prose.
6. **Do keep temperature = 0** for eval. `API_TEMP=0` is what our reported
   numbers use. Non-zero temperature degrades performance in our runs.
7. **Do keep `max_tokens ≥ 256`.** Long legal-action lists occasionally push
   the reply toward the ceiling; we set the eval clients to 256.
