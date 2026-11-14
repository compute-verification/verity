[2026-11-14T17:36Z] cffadd2fb8b1
model: Claude Opus 5.5 (this turn's system prompt); subscriptions removed: none held.

READMAPS to NAMES2 and training, Sat 14 Nov 9:35 AM PST: 4b63d6b320be61c3a5de85922f796ff81d8638a3 on `cursor/training-rename-2-a32b` (draft #10164). It is the branch's tip on origin, and its parent is 1f03e873b722.

**NOT READY 2. The stem sentence covers every older name my scan found, but its `D` stem claims too much: `fit.py`'s `D_max` and `kappa.py`'s `c.D` are built on `D`, aren't the slowdown, and aren't ruled out. And "η, the term the slowdown replaces" gives the old symbol where the other four stems give the glossary's current term, which misreads for anyone who knows η as the success threshold. The three "overhead" lines are right. The tests pass, but two of their warnings do name the remote README.**

1. The stem sentence (`benchmarks/nci/README.md`)
- READ: every example it gives exists at the head.
  - `bracket.py`'s `"rho"`.
  - `tiers.py`'s `efficiency` and `exposure.py`'s `eta`.
  - `residual_share` (`tiers.py:286`) and `memory_caps_residual`.
  - `tiers.py`'s key `unmonitored`.
  - `fit.py`'s `dampener` and the keys `D_*`.
- READ: each look-alike it rules out is what it says. `tiers.py`'s `rho_T`, `rho_PoS` and `rho_NCI` are rates. `exposure.py`'s `wipe_residual` is the memory the fill misses. `kappa.py`'s `RESIDUAL` is the gate count of the model's residual add (lines 48, 227). `reference_efficiency` is the assumption's name.
- READ: the stems cover every older name my last scan found: `eta` and `inverse_eta`, the `efficiency` keys of `fit.py` and `library.py`, `statement.py`'s `D_fit_T5` and `D`, and the names built on `residual`.
- READ: four stems give the glossary row's current term (`GLOSSARY.md`, Older names, at the head):
  - `rho`: the slowdown, row 212.
  - `residual`: free compute, row 229.
  - `unmonitored`: unregistered hardware, ε_unreg, row 227 (`tiers.py`'s is a share, as the sentence says).
  - `dampener` or `D`: the slowdown, row 213.
- READ, the `efficiency` or `eta` stem does not. Row 212's current term for "training efficiency" η is the slowdown, and the sentence gives the old symbol, η.
- READ, it claims too much on `D` (an identifier scan for `D` and `D_*` across every Python file under `benchmarks/nci/` at the head):
  - `fit.py`'s `D_max`, the parameter of `n_star(fit, L, D_max)` (103–106), and its loop's `D` over `DATA_LIMITS` (91, 266). Both are a token count in the loss law (`f["B"] / D_max ** f["beta"]`), not the slowdown. `D_max` is a `D_*` name besides.
  - `kappa.py`'s `c.D` in its docstring (line 5, `c.D.size c.D.computed`, a Lean circuit's field) and the `D` of its softmax comment (42, "P (dP - D)", the backward pass's row term).
- READ, a smaller look-alike on `efficiency`: `catalog.py`'s category id `efficiency-metrics` (line 23), which the same README lists at line 54. It is a literature category, not η.
- READ: the other names built on the stems are prose, not names: "wipe residuals" in `fit.py` and `statement.py` (`wipe_residual` covers them), and `parameters.json`'s "m = rho under its effective-parameter form" (601, 664), a QAT law's ρ in a data file's notes.

2. "`efficiency` or `eta` for η, the term the slowdown replaces"
- READ, it is true by the glossary: row 212 lists "training efficiency" η among the names the slowdown replaces.
- INFERRED, it is not clear to that reader. Today η is the success threshold, the argument of slowdown(𝒢, 𝒜, η) and of T_𝒢(η; S) (glossary lines 64–65). So "for η" reads as "for the success threshold", and "the term the slowdown replaces" then contradicts a slowdown that takes η as its argument. It is the README's only η.
- READ, it is also not quite "the slowdown": `exposure.py`'s docstring says 1/eta is not the slowdown D, since eta counts every term at peak and D counts the residual r at MFU₀.
- Suggested cure: "`efficiency` or `eta` for the older efficiency term, written η in older text and replaced by the slowdown (today's η is the success threshold; `exposure.py` relates 1/eta to D)". RAN: none of these words is a retired phrase in the lint's term list.

3. The remote verifier's README
- READ: the hunks are line 28 and lines 396–397, each a one-for-one replacement.
  - Line 28, "the slowdown bound's remote regime (`InferenceOnly.Guarantees.SlowdownRemote`, the slowdown calculator's `regime.remote`)": true. Line 400 calls `SlowdownRemote` "the slowdown bound max(1, 1/(s + x/MFU₀))", and part 6's caveat that it is not the displayed bound stays.
  - Line 396, "The slowdown over remote, per period": true; `SlowdownRemote`.
  - Line 397, "The slowdown over a monitoring period": true; `SlowdownRemotePeriod`.
- RAN `rg -i overhead` on the file at the head: no "overhead" is left.

4. The rest of the commit and the tests
- READ: the commit changes 2 files, 10 lines added and 7 removed, all in the two hunks above and in the README sentence. Nothing else changed.
- RAN, in a worktree at the head: `python3 -m pytest tests/lint/test_repo_retired_terms.py tests/test_retired_terms.py tests/test_doc_paths.py` gives 51 passed, 7 warnings, which matches NAMES2's count.
- RAN, but NAMES2's "none naming either file" is false. `test_no_retired_term_in_the_maintained_docs` warns on the remote README at lines 9 ("component"), 402 ("declared work") and 512 ("declared work"). `test_every_path_a_document_names_exists` warns on line 70 (`composition/README.md`). READ: these predate this commit, which leaves those lines untouched.

NOT READY 2 4b63d6b320be61c3a5de85922f796ff81d8638a3

MUST FIX
- READ: the `D` stem covers names that are not the slowdown: `fit.py`'s `D_max` and its `D` over `DATA_LIMITS` (tokens in the loss law), and `kappa.py`'s `c.D` and its softmax comment's `D`. Cure: add them to the "not older names" list, or narrow the stem to `dampener`, the output keys `D_*` and the bare D in docstrings.
- READ and INFERRED: the `efficiency` or `eta` stem names the old symbol η instead of the glossary row's current term (the slowdown, row 212), and to a reader who knows η as the success threshold it reads backwards. Cure: the wording suggested in part 2, or its like.

SHOULD FIX
- READ: rule out `catalog.py`'s category `efficiency-metrics` beside `reference_efficiency`, or say the stems are about the slowdown's terms.
- RAN: the remote README's existing warnings (lines 9, 70, 402, 512) are stale statements in a file this chain edits. Fix them on sight here or in the owner's next PR; they don't hold this commit.

NOTE
- INFERRED: `parameters.json`'s "m = rho" is a QAT law's ρ in data notes, not a code name, so I don't ask for it.

For the lead to read
- READ: an event addressed to me, NAMES2's inbox message at 9:30 AM PST, which matches your brief except for the warnings claim above. I saw none addressed to @training this turn.
- RAN: no Lean was run and no run was launched. The worktree `/agent/repos/wt-rename4` (detached at 4b63d6b3) is left in place; `wt-rename3` was gone when I came to it.

