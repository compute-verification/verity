[2026-11-14T17:40Z] f986d0eb99c0
NAMES2, Sat 14 Nov 9:40 AM PST. model: Claude Opus 5.5 (this turn's system prompt); subscriptions removed: none held.

**#10164's head is `6be5c1267220285bb4dc301feac599322a4d5d34`: one new commit on `4b63d6b3`, a plain push, holding exactly your five changes to `benchmarks/nci/README.md`'s sentence. No run was relaunched, and both are still `waiting`.**

The sentence as it now stands at the head. RAN, read from `git show HEAD:benchmarks/nci/README.md`

> For the slowdown's terms, the code and its outputs keep older names (`GLOSSARY.md`, Older names) in the names built on these stems: `rho` for the slowdown (`bracket.py`'s stored key `"rho"`); `efficiency` or `eta` for the older efficiency term, written η in older text and replaced by the slowdown (`tiers.py`'s `efficiency`, `exposure.py`'s `eta`; today's η is the success threshold, and `exposure.py`'s docstring says 1/eta is not the slowdown D); `residual` for free compute (`residual_share`, `memory_caps_residual`); `unmonitored` for unregistered hardware's share, ε_unreg (`tiers.py`'s key `unmonitored`); and `dampener` for the slowdown, in functions and prose (`fit.py`'s `dampener`), with the output keys that begin `D_` in `fit.py` and `statement.py` and `statement.py`'s key `D`. These are not older names: `tiers.py`'s `rho_T`, `rho_PoS` and `rho_NCI`, which are rates; `exposure.py`'s `wipe_residual`, the memory the fill misses; `kappa.py`'s `RESIDUAL`, the model's residual stream; `fit.py`'s `D_max` and its loop variable `D` over `DATA_LIMITS`, token counts in the loss law; `kappa.py`'s `c.D` and the `D` of its softmax comment; `reference_efficiency`, the reference-efficiency assumption; and `catalog.py`'s category id `efficiency-metrics`.

Facts behind it, at the head. READ
- `exposure.py`'s docstring, line 4: "1/eta is not the slowdown D". The README quotes those words.
- `fit.py`: `n_star(..., D_max)` is at lines 103–106, and its loop `for k, D in DATA_LIMITS` is at line 266.
- `kappa.py`: `c.D.size c.D.computed` is at line 5, and "P (dP - D)" at line 42.
- `statement.py`'s key `"D"` is at lines 289, 351 and 447. Its `D_` keys are `D_closed`, `D_fit`, `D_fit_low`, `D_fit_high`, `D_fit_T5` and `D_fit_train`.
- `catalog.py`'s `"efficiency-metrics"` is at line 23.

The three tests at the new head, with every warning shown. RAN
- `tests/lint/test_repo_retired_terms.py`, `tests/test_retired_terms.py` and `tests/test_doc_paths.py`, run together: 51 passed, 7 warnings.
- No warning names `benchmarks/nci/README.md`, so the maintained-docs test finds no retired phrase in the new sentence.
- The warnings that name the remote verifier's README are the four READMAPS found, all older than the chain: line 9 ("component"), lines 402 and 512 ("declared work"), and line 70 (`composition/README.md`). The rest name other files, `tools/verity/check/move_check.py:16` among them.

The body. RAN
- #10164's body now lists the new commit.
- Under "Not done here" it says the remote README's four statements stand: another Glossary row's and another owner's.

Runs, by named keys. RAN
- `r20261114-171003-ac59`: state `waiting`, rc None, tree_sha `3d66c4943f022298a9ec8b0da5798687d764f06d` (records).
- `r20261114-171005-2a97`: state `waiting`, rc None, tree_sha `1f03e873b722188465a9f92b84e5047508b7acd4` (fixtures).

Events: READMAPS' NOT READY 2 at 9:35 AM. I see nothing else addressed to @training. READ

