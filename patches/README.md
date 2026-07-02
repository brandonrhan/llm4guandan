# Patches to LLM4CardGame

This directory contains our local modifications to the [THUDM/LLM4CardGame](https://github.com/THUDM/LLM4CardGame)
repository. `scripts/setup.sh` clones LLM4CardGame at commit `34c2785` into
`third_party/LLM4CardGame/`, then applies these changes.

## Files

- `LLM4CardGame.diff` — unified diff (`git diff HEAD`) of our tracked-file
  modifications:
    - `eval.sh` — route to `eval_guandan_local.sh` / `eval_guandan_danzero.sh`
      depending on the chosen opponent.
    - `util/llm_client.py` — reduce `max_tokens` to 256 and retry attempts to 3
      (was 10 with 60 s waits) so eval doesn't stall on a single bad turn.
    - `util/llm_config.py` — rename local model type from
      `THUDM/glm-4-9b-chat` to `guandan` so it matches our vLLM `model=guandan`.
    - `util/guandan_util/actor_llm.py` — wrap the LLM call in `try/except`
      with a random-fallback, so a single API error doesn't kill the match.

- `LLM4CardGame/util/guandan_util/client0.py` — copy of our modified client
  (upstream file, but our diff is large enough that we ship the whole file).
- `LLM4CardGame/util/guandan_util/client2.py` — ditto.
- `LLM4CardGame/util/llm_client.py`, `llm_config.py` — full patched copies
  (also applied by the diff).

`scripts/setup.sh` first `git apply`s the diff, then overwrites `client0.py`
and `client2.py` with the versions in this directory.
