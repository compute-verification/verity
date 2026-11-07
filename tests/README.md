# `tests`

The repository's own suite (the root `pyproject.toml`'s, `.` to `suites.py`). It holds only invariants of the whole
tree; a package's tests live in its own `tests/`. Its tests read the whole tree, so its pass is keyed on every tracked path.

| What they hold the tree to | Tests |
| --- | --- |
| the dependency rules | `test_boundaries.py`, `test_integration_boundaries.py`, `test_vllm_public_api.py` |
| what the tree may hold, and Markdown caps | `test_repository.py`, `test_protocol_specs.py` |
| no dead reference after a move | `test_doc_paths.py`, `test_module_strings.py`, `test_gate_paths.py`, `test_pythonpath_entries.py` |
| prose that stays true | `test_retired_terms.py`, `test_provider_names.py`, `test_glossary_lean_names.py` |
| the Lean packages, their pins, and a verifier's cost | `test_lean_packages.py`, `test_pins.py`, `test_verifier_cost.py` |
| how tests run | `test_no_wall_clock.py`, `test_circuit_check_inputs.py`, `test_uv_pin.py`, `test_merge_drivers.py` |
| indexes and conventions | `test_benchmarks_index.py`, `test_examples.py`, `test_cli_conventions.py`, `test_seed_ports.py` |

A test's docstring states its rule. Where a rule has exceptions, they are named in the test, one reason each (an `ALLOWED`
table in the boundary, wall-clock, retired-term, seed and verifier-cost tests).

## Lint ratchets

`lint/` holds four kinds of finding to a list in `lint/allowlists/`, counted per file: environment reads
(`test_repo_env_reads.py`), lane text in prose (`test_repo_lane_text.py`), ruff's findings (`test_repo_ruff.py`) and except
handlers that swallow what they catch (`test_repo_silent_excepts.py`). A file with more findings than its entry allows, or
with findings and no entry, fails, except that lane text and ruff's style findings are bookkeeping: they warn, and a later
PR fixes them (ruff's errors, `ERRORS` in its lint, still fail). An entry that allows more than is found doesn't, so the
list can run ahead of the tree; the ratchet gate, which holds nothing (`off_gate` in `tools/verity/check/policy.toml`),
reports a change that grows a list or adds a finding to a file it touches.

## Bookkeeping

`conftest.py`'s `BOOKKEEPING` names the tests whose findings warn (a `Bookkeeping` warning in pytest's summary) and pass:
lane text, ruff's lint, retired terms, doc paths, Lean and CLI naming, provider names, stale allowlist entries and
Markdown caps. A scan's own tests, and a listed test that raises anything but a failed check, still fail.
`test_bookkeeping.py` holds the list to the tree.

~~~sh
python3 tests/lint/test_repo_ruff.py --move OLD NEW   # a file or directory moved: its entries follow it, in the same change
python3 tests/lint/test_repo_ruff.py --prune          # delete and lower what this tree no longer finds
python3 tests/lint/test_repo_lane_text.py --record    # the list becomes what this tree finds (the gate still refuses a rise)
~~~

Every lint takes `--move` and `--prune`; only lane text takes `--record`. Each lint reads no history, so a renamed file's
findings count under its new path: move its entry with it, or the gate refuses the change.

~~~sh
uv run tools/verity/check/suites.py .                 # this suite, on a node through `research run --on auto`
~~~
