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
    json.dumps(2),                                  # 1. my_pos — an INT seat id 0-3 (not a str!)
    json.dumps(["C2","D2","S5","S6",...]),          # 2. my_hands — list of card strs
    json.dumps(["H2","H3","H3",...]),               # 3. remaining_hands — list
    json.dumps(["C5"]),                             # 4. last_action — RAW card list, not a triple
    json.dumps(["H4"]),                             # 5. last_teammate_action — RAW card list
    json.dumps({"0":26,"1":26,"2":27,"3":26}),      # 6. number_of_cards_left — dict
    json.dumps(["H2"]),                             # 7. down_played_cards — RAW card list
    json.dumps(["H4"]),                             # 8. teammate_played_cards — RAW card list
    json.dumps(["C5"]),                             # 9. up_played_cards — RAW card list
    json.dumps("2"),                                # 10. self_rank — str
    json.dumps("8"),                                # 11. oppo_rank — str
    json.dumps("8"),                                # 12. cur_rank — str
    json.dumps([["PASS","PASS","PASS"], ["Single","6",["S6"]], ...]),  # 13. legal_actions — list of TRIPLES
)
```

**Important asymmetry — card-type triples vs raw card lists:**

Slots **4, 5, 7, 8, 9** (all "actions / played cards" fields) are **flat
lists of card strings** like `["H2","S5","D5"]` — even when the play was a
Pair or a Bomb, only the raw cards are shown, *not* a `[Type, Rank, Cards]`
triple. Slot **13** (`legal_actions`) and the **model's output action** are
the only places that use the `[Type, Rank, Cards]` triple form. This is
deliberate — the training data and the live inference code both do this,
and the model has learned to derive the type from the raw card list where
needed. Do not "help" by upgrading the raw lists to triples.

**Field-by-field spec** (the `Key` column is the exact upstream key name
from `state['raw_obs']`, matching both training and inference):

| # | Key | Field | Type | Example |
| --- | --- | --- | --- | --- |
| 1 | `my_pos` | Your position | **`int`** seat id 0–3 (renders as bare `2`, no quotes) | `2` |
| 2 | `my_hands` | Your current hand | list of 2-char card strings | `["C2","D2","S5","HR"]` |
| 3 | `remaining_hands` | Remaining cards of other players | list of 2-char card strings | `["H2","H3",...]` |
| 4 | `last_action` | Last action of other players | **RAW** list of card strings (empty = pass / no action yet) | `["C5"]` or `[]` |
| 5 | `last_teammate_action` | Last action of the teammate | **RAW** list of card strings | `["H4"]` or `[]` |
| 6 | `number_of_cards_left` | Number of cards left for other players | dict `{seat: count}` for seats "0","1","2","3" | `{"0":26,"1":26,"2":27,"3":26}` |
| 7 | `down_played_cards` | Cards played by the down player | **RAW** list of card strings (accumulated this deal) | `["H2","S5"]` |
| 8 | `teammate_played_cards` | Cards played by the teammate | **RAW** list of card strings | `["H4","CQ","DQ"]` |
| 9 | `up_played_cards` | Cards played by the up player | **RAW** list of card strings | `["C5","SK","HK","HA"]` |
| 10 | `self_rank` | Self rank | str, one of `"A","2",...,"K"` | `"2"` |
| 11 | `oppo_rank` | Opponent rank | str | `"8"` |
| 12 | `cur_rank` | Current rank (the *deal-level* card — 级牌) | str | `"8"` |
| 13 | `legal_actions` | Legal actions | list of `[Type, Rank, [Cards]]` **triples** | see below |

**Card notation** (from the training preamble):

- **Suits**: `S`=Spade, `H`=Heart, `C`=Club, `D`=Diamond.
  Small joker uses suit `S`, big joker uses suit `H`.
- **Ranks**: `A`, `2`, `3`, …, `9`, `T` (for 10), `J`, `Q`, `K`.
  Small joker rank = `B`, big joker rank = `R`. `PASS` marks a pass.

So `"S2"` = Spade 2, `"HQ"` = Heart Queen, `"SB"` = small joker,
`"HR"` = big joker.

**Card-type triple** (used in slot 13 and in the output):
`[Type, Rank, Cards]` where

- `Type ∈ {"Single", "Pair", "Trips", "ThreePair", "ThreeWithTwo", "TwoTrips", "Straight", "Bomb", "StraightFlush", "PASS"}` for regular play, plus `"tribute"` and `"back"` only during the tribute phase between deals.
- `Rank ∈ {"A","2","3",...,"9","T","J","Q","K","B","R","PASS"}` (rank of the highest / representative card; `"B"` = small joker, `"R"` = big joker)
- `Cards` is the actual list of card strings that make the combo

> ⚠️ The docstring at the top of `prompt/prompt_guandan4.py` (part of the
> template preamble sent to the model) lists `"Boom"` and `"TripsPair"`. **The
> actual training data uses `"Bomb"` and `"TwoTrips"` instead** — those are
> the strings the model learned to emit. If you construct legal_actions
> yourself, use `"Bomb"` and `"TwoTrips"`. This was verified by scanning 5,000
> training samples: 493 `Bomb`, 25 `TwoTrips`; zero occurrences of `Boom` or
> `TripsPair`.

Examples:

- Single Diamond 5: `["Single","5",["D5"]]`
- Pair of 4s: `["Pair","4",["H4","C4"]]`
- Bomb of Aces: `["Bomb","A",["SA","HA","CA","DA"]]`
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

The correct way to call the model from Python — using the *exact* key names
that the upstream env's `state['raw_obs']` provides, so you can feed the
live game state in directly:

```python
import json
from prompt.prompt_guandan4 import prompt_guandan

def build_user_message(obs: dict) -> str:
    """Render an observation into the exact prompt the model was trained on."""
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

# obs = state["raw_obs"]  # in the upstream rlcard-based env
# Then wrap in a single-turn chat message and call the model.
```

Both [`examples/infer_openai_client.py`](../examples/infer_openai_client.py)
and [`examples/infer_transformers.py`](../examples/infer_transformers.py)
implement exactly this and are the authoritative reference.

### Parsing the model's reply

The upstream inference code (`util/prompt_util.py::out_parse_function`)
extracts the action with a regex that grabs the **last** JSON object in the
reply, so you don't need strict JSON parsing:

```python
import json, re

def parse_action(text: str):
    objs = re.findall(r'\{.*?\}', text, re.DOTALL)
    parsed = [json.loads(o) for o in objs]
    return parsed[-1]["action"] if parsed else None
```

Since the training target is *just* `{"action": [...]}` with nothing else,
at temperature 0 the model emits exactly one JSON object and `json.loads()`
is sufficient. The regex form matches upstream behavior exactly.

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

## 7. Verified consistent with upstream (training vs inference)

On 2026-07-03 the training-side and inference-side prompt builders were
diffed against upstream `LLM4CardGame` (commit `34c2785`, present at
`/home/dan/brandon/llm4card/LLM4CardGame/` on the training server):

| Source file | Function | Purpose |
| --- | --- | --- |
| `convert_data.py` | `convert_guandan(data)` | Renders SFT `instruction` at train time (from replayed `line['obs']`) |
| `util/prompt_util.py` | `prompt_function_guandan(state)` | Renders the same prompt at live inference (from `state['raw_obs']`) |

Both call `prompt_guandan % (...)` with the **same 13 fields in the same
order**, each wrapped in `json.dumps()`, pulling from a dict with the same
key names:

```
my_pos, my_hands, remaining_hands, last_action, last_teammate_action,
number_of_cards_left, down_played_cards, teammate_played_cards,
up_played_cards, self_rank, oppo_rank, cur_rank, legal_actions
```

The only formal difference is the source dict (`line['obs']` on disk vs
`state['raw_obs']` from the live env) — the rendered `instruction` string is
byte-identical when the two dicts are equal. In particular the
"raw-card-list vs card-type-triple" asymmetry (slots 4/5/7/8/9 are raw
lists; slot 13 and the output are triples) is present in *both* builders,
so a model trained on this asymmetry will see the same asymmetry at
inference. There is no train/serve prompt-format skew.

**End-to-end byte-diff check** (2026-07-03). We rendered the STATE dict from
[`examples/infer_openai_client.py`](../examples/infer_openai_client.py)
using the training-server's own `prompt_guandan4.py`, then diffed it against
record #0 of `data/sft-guandan-full.jsonl` slot-by-slot. **All 13 named
slots are byte-identical** in format (slot 13's contents naturally differ
because the example has 9 legal_actions vs the real state's ~40; the format
is identical).

This check also caught one type mistake in an earlier draft of these docs
that has since been fixed:

- **`my_pos` is an `int` seat id `0`–`3`**, not the string `"0"`–`"3"`.
  It renders as a bare `2` (no quotes) in the prompt because
  `json.dumps(2) == "2"` (unquoted).
- Ranks (`self_rank`, `oppo_rank`, `cur_rank`) are true `str`, rendering
  as quoted `"2"`, `"8"` etc.
