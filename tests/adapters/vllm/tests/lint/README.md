# The vLLM adapter's lints

These tests keep the vLLM adapter's code to its quality bar. Each is a ratchet: it scans `verity_vllm/` for one kind of
violation and compares what it finds with a list of the violations that existed when the rule came in. A new violation
fails, and so does a list entry that allows more than is found, so the lists only shrink. Each failure message carries
its fix.

The scans parse the code and import nothing from it, so they need neither torch nor vLLM.

## Try one

~~~sh
uv run pytest tests/adapters/vllm/tests/lint/test_p10_size.py
~~~

This holds every module of the adapter to at most 800 lines and every function to 150, against its list, and prints
`3 passed`. It needs nothing beyond `uv sync`.

Before marking a pull request that touches the adapter, run the ratchet gate against `main`. It fails a change that
grows a list, or that a lint as `main` has it would refuse:

~~~sh
python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD
~~~

## The lints

| Lint | What it holds |
| --- | --- |
| P1 to P16, with no P5 (`test_p01_*` to `test_p16_*`) | The adapter's design rules: core's abstractions, one evaluator, one result, one CLI, declared inputs, layering, size, names and the rest, one each. |
| Typed records | A record is read through its type in `verity_vllm/records/`, not with `.get(k) or []`, and its schema id is written only there. |
| Recorded outcomes | A caught exception is raised or recorded, never dropped (below). |
| Decimal reads | A string is asked whether it is a number with `str.isdecimal`. |
| vLLM's surface | Every vLLM module and name the library imports is listed, with why. |
| Ruff, ruff with vLLM's rules, mypy | Each tool's findings, held to a list. |
| The ratchet | Every lint's list only shrinks, and every failure prints the exact entry lines. |

- **Recorded outcomes** (`test_silent_excepts.py`, allowlist `allowlists/silent_excepts.json`): in `verity_vllm/check/` and
 `verity_vllm/commit/`, a caught exception is raised or recorded (an `error` result, a reason, an `unreadable: …` entry in
 the record), never dropped with `pass`, `continue` or a default. A guard never fails open: a handler that skips what it
 can't read, or puts a default where a value was, shrinks the evidence a verdict is decided on and leaves no trace of it.
 `tests/lint/test_repo_silent_excepts.py` holds every other area to the same rule.
