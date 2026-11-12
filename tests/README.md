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
table in the wall-clock, retired-term, seed and verifier-cost tests); the import order's are the listing
`import_order.toml`, which `test_boundaries.py` reads.

## Lint ratchets

`lint/` holds six kinds of finding to a list in `lint/allowlists/`, counted per file: environment reads
(`test_repo_env_reads.py`), lane text in prose (`test_repo_lane_text.py`), retired terms (`test_repo_retired_terms.py`),
ruff's findings (`test_repo_ruff.py`), except handlers that swallow what they catch (`test_repo_silent_excepts.py`), and
what Mathlib's style linters refuse of a Lean file under `security/` or `verity/` (`test_lean_bar.py`, per file and
rule: no `/-!` module docstring, more than 1,500 lines, a `maxHeartbeats` of 0, one with no comment saying why). A file
with more findings than its entry allows, or with findings and no entry, fails, except that lane text, retired terms,
ruff's style findings and three of the Lean bar's rules (the module docstring, the length, the uncommented override) are
bookkeeping: they warn, and a later PR fixes them. Ruff's errors (`ERRORS` in its lint) and a new `maxHeartbeats 0`
(`test_no_new_unlimited_heartbeats`) still fail, and so does a tree the Lean bar names with no `.lean` file under it
(`test_every_tree_holds_lean`: its Lean moved, so name where it lives now). An entry that allows more than is found doesn't, so the list can run
ahead of the tree (the Lean bar warns of one, naming the count to lower it to); the ratchet gate, which holds nothing
(`off_gate` in `tools/verity/check/policy.toml`), reports a change that grows a list or adds a finding to a file it
touches.

## Bookkeeping

`conftest.py`'s `BOOKKEEPING` names the tests whose findings warn (a `Bookkeeping` warning in pytest's summary) and pass:
lane text, ruff's lint, retired terms, doc paths, Lean and CLI naming, provider names, stale allowlist entries,
Markdown caps and the Lean bar's three soft rules. A scan's own tests, and a listed test that raises anything but a failed check, still fail.
`test_bookkeeping.py` holds the list to the tree.

~~~sh
python3 tests/lint/test_repo_ruff.py --move OLD NEW   # a file or directory moved: its entries follow it, in the same change
python3 tests/lint/test_repo_ruff.py --prune          # delete and lower what this tree no longer finds
python3 tests/lint/test_repo_lane_text.py --record    # the list becomes what this tree finds (the ratchet gate still flags a rise)
~~~

Every lint takes `--move` and `--prune`; lane text, retired terms and the Lean bar take `--record`, which rewrites the list
to the tree's counts. The Lean bar's is for a merge that brings files written before the bar held them, and it lists no new
`maxHeartbeats 0`, which fails the suite until it has a bound (`test_lean_bar.py --counts` prints each rule's findings by Lake
package). Each lint reads no history, so a renamed file's findings count under its new path: move its entry with it, or
the ratchet gate flags the change.

~~~sh
uv run tools/verity/check/suites.py .                 # this suite, on a node through `research run --on auto`
~~~
