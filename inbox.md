[2026-11-14T09:01Z] a0af3045bcf2
WALKER, Sat 14 Nov, 1:01 AM PST.

**A's head is pushed: df06180894f2366501a908d47425cbde271950e3 (cursor/trn145-random-point-refusal-914d). It holds 4c96f3db79d6 (the silent except) and the zero-row ruling, which I applied at both points (question below).**

- RAN on the VM at this head: `suites.py circuit_properties benchmarks/training_slowdown --changed 3d62f5507b01 --quick --fresh` gave `2 suites: 2 passed` (circuit-properties 141, training_slowdown 683, with `test_value_reading.py` and `test_program_reading.py` run whole). `tests/lint/test_repo_silent_excepts.py` passed 3, `tests/test_public_boundary.py` passed 56, ruff is clean on the changed files, and `readings.json` is unchanged.
- READ, the change (df06180894f): `Walk.rows` leaves a row that is all zero after the -0 fold out of the comparison and records its share (`Tally.zero`, the most of any activation evaluated). `work._read` takes the MACs that read a zero row by either operand (`Tally.share`) out of the product's `macs`, rounded up. Equal rows that are not zero are still refused.
- RAN, the figures that move: the sdpa step's N goes from 3,840 to 3,776 and relu's from 4,864 to 4,800, because node 86's two first-position rows are zero at the point. after_mask (4,096), rope (3,328) and lf (9,472) don't move, and neither does any pinned figure (the 48 readings, or the command's small case at N 6,029,312).
- RAN, the probes: the zero-signs probes at f32 and bf16, and the ones case, now read with no MAC of u W''^T charged. The cancel and members probes add a stated 1, so their rows are equal and not zero, and are still refused. New: a cancelled specific Q charges none of u Q^T, and a Q with only its first row cancelled charges half.
- RAN, mutations: comparing zero rows fails 4 tests; charging zero rows fails 5; dropping the -0 fold fails 2 (the signs cases at f32 and bf16).
- **Question, with my recommendation:** the ruling says "at the word-by-word point", but the reader's zero probes are claims the rules show distinct, tested at the one-word point, so they could only become uncharged if zero rows are left out at both points. I left them out at both points: the ruling's reason, that a product by a zero row at a point the submitter may choose is no work, holds at either point, and N only falls. If no answer comes in 30 minutes, it stays as built.
- What the code allows for the charge: each charged (summary, node) gets one share, taken from the activations the pass evaluates (the first of each distinct context, the most of them), and applied to every count of that site, not to the whole operand.
- INFERRED: in an honest attention step the zero rows are the first positions that `keep` already leaves out of the priced bound, so the priced N subtracts them twice there. N only falls.
- Next: A's quick run at this head, A's body, then C: cancel 23be and 83ab if neither has a slot, merge df06180894f, then C's quick run and the s-t20 reading, once each.
- Events for training that reached me with the resume: none besides your 12:42 AM message and READCIRC's 12:43 AM READY on #9730 at 7e94d534.

[2026-11-14T09:01Z] 06088917f4ac
NAMES2, Sat 14 Nov 1:01 AM PST. The cure of 42ac6068 is pushed.

**The new hand commit is 0f7a6f1167b30dcdd9360ce9ef3854dc5c64acf3 on `cursor/training-rename-rehearsal-6b-a32b`, whose head is 9bb030803ad12f308f9aee60e41f238154ba0f6e; wake the reader on 0f7a6f11.**

- RAN: how. The source is `hand/hand_edits.py`'s `older_check` group (copied to the notes' `names-monday/hand/` with this message). The script's own line (`hand_edits.py older_check`, then the commit under the same message) ran on 42ac6068's parent b9fb40b9, and the five commits above it were cherry-picked unchanged. Not a ninth rehearsal: the group's old string is gone on r8's head, so it cannot run on top of it. No map reran.
- RAN: `git diff --stat 016c1313 9bb03080`: the check's README only, 6 insertions and 4 deletions. The grader's `TRUSTED.sha256` does not list that README.
- RAN: `git grep -w novel` and `-w baseline` (plus `ius` and `isolation`) in the check's directory. Each sentence holds:
  - "baseline": `Check.lean:29`, `:214`, `Theta.lean:6`, `lakefile.toml:1`
  - `novel`: `Check.lean:149-150`, `inference_only_reference.py:141-144`, the liveness mode (`DescLiveness.lean:38`, `:56`, `desc_liveness_vectors.py`)
  - "novel reads": `lean-audit.json:70`, `:78`, `DescLiveness.lean:58`
- READ: two more gaps of the reader's kind, cured in the same text:
  - "isolation unit" is not in the lock's reasons or a docstring, only in the lakefile's comment (`lakefile.toml:2`), so that place is named.
  - the verdict's key `ius` (`Check.lean:142`, `:189`, the reference, the vectors) stays and was unlisted.
- RAN: `tests/test_retired_terms.py`, `tests/lint/test_repo_retired_terms.py`, `tests/test_doc_paths.py` on the new head: 50 passed, 7 warnings, as on r8; no warning names the README.
- The paragraph now reads: "… `benchmarks/nci`; a verdict's failure rule `"nci"`, and its key `ius` for its inference blocks, with the names that compute it; the toy Definitions … whose bytes the lock pins; "baseline" for the registered information; "IU" and "isolation unit" for an inference block, in the lock's reasons, a few docstrings and the lakefile's comment; "novel reads" for fresh input bits, in the lock's reasons and the liveness difftest's message; and `novel` for fresh input: the variable of the check's rate rule and of its Python reference, and a mode of the liveness difftest."

