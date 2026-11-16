[2026-11-16T20:57Z] 77f45e5c2c4b
PAGE, 12:57 PM PST. PR 4 is pushed: draft #10570 (https://github.com/compute-verification/verity/pull/10570), branch `cursor/training-delete-serving-blocks-4207`, head `6a15ba47c65a7b79a0a8983c6a89fb29dcc31221`. 11 files, +9/−1,064. As with PR 3, there is no Goal line or Linear link (no credentials). The lock and `claim_ids.json` are untouched, and the body lists the lock records.

What the ResourceAccounting aggregators take. All three are lean's files under the registry's `security/` [read]. No guarantee in them is stated over ServingBlocks, so there is nothing for compute-accounting [read: no Lean outside the six files names a ServingBlocks declaration except these three]:
- `Definitions/ResourceAccounting.lean`: two imports (`Definitions.Slowdown.ServingBlocks{,.Instances}`), one top-guarantee docstring bullet (`ServingBlocks.ServingBlockBound`), and one Map clause ("and `ServingBlocks` server blocks and a training class's counts").
- `Definitions/ResourceAccounting/Guarantees.lean`: two imports (`Definitions.Slowdown.Guarantees.ServingBlocks{,.Witnesses}`), nothing else.
- `Proofs/ResourceAccounting.lean`: two imports and one "where proved" line.
All of these are removed, and the body says so. Also lean's: two `tests/tools/verity/lean/test_audit.py` docstrings that cited `UncertifiedElementary` as an example. The example is dropped; the tests build their own fixtures.

The lock update also touches other areas' `layers` lists. `Definitions.Slowdown{,.Guarantees}.ServingBlocks` sits in the allowed imports of `Definitions.{ResourceAccounting,Poub,Compose,Registration}{,.Guarantees}`, and both locks' `meaning` lists name it [read].

Python, left for PR 5 [read]: `test_lean_names` reads the calculator's string and class constants. So after the lock update it fails on `envelopes.py`'s two `BOUND`s (`ResourceAccounting.Guarantees.ServingBlocks.ServingBlockBound{,Remote}`), which `test_scenario.py` asserts. It does not read `classes.py`'s μ comment or `exclusion.toml`'s `mu` source, which only cite `ServingBlocks.Overlap` in prose. The body says "if it reads these names"; it does.

Ran locally:
- the path and vocabulary tests: 91 passed;
- `test_lean_names`: 3 passed, against the unchanged lock;
- the two `premise_free_exclusion` tests in `test_audit.py`: 2 passed.

Not run: the Lean build, the Lean check, the lock update and the suites.

Next: PR 5's list, sent to you before I delete anything.

