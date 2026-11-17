# `tests`

Every test in the repository is here. A directory's tests sit at the same path under `tests/` (`verity`'s at
`tests/verity`, the `research` tool's at `tests/tools/research`), one suite each, run in its own process and cached by
its inputs. What sits directly in `tests/`, with `lint/`, is the repository's own suite: rules
that hold of the whole tree rather than of one package, such as which folder may import which, that no document names
a dead path, and that no retired term comes back.

## Try one

~~~sh
uv run pytest tests/test_tree.py -q
~~~

This holds the tree to its layout of record, `tests/tree.toml`, and prints `6 passed` in under a second. It needs
nothing beyond `uv sync`.

`uv run tools/check/suites.py --list` prints every suite with its inputs and whether a pass is cached. A whole suite
runs on a node through `research run --on auto`:

~~~sh
uv run tools/check/suites.py .                 # this suite, on a node through `research run --on auto`
~~~

## What is here

| Where | What it holds |
| --- | --- |
| `tests/verity`, `tests/tools/research`, … | A directory's tests at its path, as its suite. |
| `tests/*.py` | The repository suite's invariants. Each test's docstring states its rule, and names its exceptions with a reason each. |
| `tests/lint/` | Lint ratchets: kinds of finding counted per file against a list in `lint/allowlists/`. |

Where a rule has exceptions, the test names them (an `ALLOWED` table). The import order's are the listing
`import_order.toml`, which `test_boundaries.py` reads and whose tables `tools/check/import_order.py` writes.

## Bookkeeping

`conftest.py`'s `BOOKKEEPING` names the tests whose findings warn (a `Bookkeeping` warning in pytest's summary) and pass:
lane text, ruff's lint, retired terms, doc paths, Lean and CLI naming, provider names, stale allowlist entries,
Markdown caps and the Lean bar's three soft rules. A scan's own tests, and a listed test that raises anything but a failed check, still fail.
`test_bookkeeping.py` holds the list to the tree.
