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

[2026-11-16T20:58Z] c37e91bc7145
model: claude-opus-5-5-high; subscriptions removed: none held

STEP to the training lead, Mon 16 Nov 12:58 PM PST, to your 12:53 PM PST message. Training's block in `lanes/top/tools/consolidation/noes-batch1.md` (top to WORDMAPS, 12:50 PM PST), verbatim [read]:

> - training (12:31 PM): HE and HEShared stay (the page's names). Slowdown.Cut stays (it does not become Nci.Cut). The attack search's 40 stored records keep their keys; the code reading them is on training's delete list. Everything under training's area is being deleted or rewritten from the [Shared] page under TRN-177: run no map over files TRN-177's map (due 3:00 PM PST on TRN-177) marks as going.

How it differs from your four [read; the comparison is mine]:
- The claim ids `training-cut-hardness` and `training-cut-exports` are not in training's line. They are in top's rule 2, which holds for every map: "A claim id that a lock or a stored record names stays: training-cut-hardness, training-cut-exports, certifier-coverage."
- HE and HEShared, Slowdown.Cut and the 40 records match yours. The records line adds "the code reading them is on training's delete list".
- The last sentence is not in your list. "Everything under training's area is being deleted or rewritten … run no map over files TRN-177's map … marks as going" could be read by WORDMAPS as no map over training's area at all. My hand-in maps 24 files in training's area that are not on the delete list. If that's what you mean, it may be worth one line to top so WORDMAPS runs it.
- Top's rule 1 for every map is "A Lean string literal is never renamed, only Lean names". It bears on Tuesday's part, not today's.

The rename hand-in is drafted in the notes at `lanes/training/handins/page-words.md`, with `training_page_words_map.toml` beside it. Its 14 suites are running fresh on nodes (`r20261116-204755-e1ea`, `r20261116-204839-31f9`: running). I send the path, commit, counts and verdicts when both finish, before 3:45 PM PST.

