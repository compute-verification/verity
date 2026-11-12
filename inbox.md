[2026-11-12T21:32Z] a8b3412ec026
PREMISE, Thu 12 Nov, 1:33 PM PST. #9272 with your rulings applied.

**#9272's head is now 84995d9b78, pushed as a fast-forward from 795e1c770e (no force). It holds main at 5f0c008aa0, today's tip, and all three suites pass on it.**

Commits on top of the author's head:
- 82a23708b8: merge of main 5f0c008aa0. The two import conflicts keep both sides.
- 1d5c40bfc6: EQUAL = 2e-5, APART = 500*EQUAL, and PART = 100*EQUAL as the parting assertion's floor; PART's comment says it holds that the repeat stops at the second attention. Each comment names seeds 0-999 and the worst seed's value (1.6e-6, 0.0201, 0.0120).
- 3bc5d5afaf: items 5, 6 and 7: "tokens up to it, its own included"; "a run of three or more", charging r-2 positions a second time; library "given its premises".
- 84995d9b78: item 4. circuit_properties/work.py's docstring, its README and test_value_reading's docstring name training_slowdown.work.P_BATCH. test_program_reading has the one assert, beside SHARED_PREFIX.

Runs:
- r20261112-211649-68fb, at 84995d9b78 (RAN):
  passed  benchmarks/circuit_properties 69 passed in 46 s; 69 outcomes recorded
  passed  benchmarks/training_slowdown/grader 28 passed, 3 skipped in 57 s; 28 outcomes recorded
  passed  benchmarks/training_slowdown 580 passed in 576 s; 580 outcomes recorded
  3 suites: 3 passed (0 cached), 0 failed; 11.3 min of test time
- r20261112-202651-b6ca: the 1000-seed sweep the bounds come from, and the 8 mutations (RAN). At 2e-5 / 0.01 / 2e-3, every assertion holds at all 1000 seeds (INFERRED from its worst values).
- The ratchet gate at 84995d9b78 against main: 1 change, 0 failed (RAN, locally).
- r20261112-205811-987f is superseded: it ran an earlier patch, with APART 0.02 and no parting assertion.

The body was replaced with `research queue edit`. It keeps the Lane, Refs and Goal lines, lists the four commits above, and names which runs are mine and which the author's. It is still a draft and unmarked.

Commit 6242aba2e0: made on #9170's branch, cursor/training-slowdown-off-program-914d (Lane training, TRN-135); it reached main with #9029 (READ, gh). Open PRs touching the three files: only #8731 (cursor/proofs-n2-table-root-5b1e). It edits circuit_properties/README.md in another paragraph (the Llama shapes), so it doesn't touch the premise's lines (READ, git diff origin/main...). #9264's file list names two of the files, but its diff against main has no change to them.

Note, updated with these: note:training-20261112T2100Z-report-premise-review-9272. Handoff follows.

[2026-11-12T21:33Z] 8c3770640d3d
PREMISE, Thu 12 Nov, 1:32 PM PST. **Done: #9272 is at 84995d9b78 with your rulings applied, its suites are green, and no run is in flight.** Handoff: note:training-20261112T2132Z-handoff-PREMISE. Left, in order: you read and mark #9272; you tell #9170's lane (TRN-135) that circuit_properties now names P_BATCH; step 2 of TRN-137 is not this PR.

[2026-11-12T21:33Z] bf1588f08ff8
SORT, TRN-146: carry of #8767. What it is for: the composition difftest takes 49 minutes instead of 84, and each thousand-case test gets a short form that every check can run. READ (the PR title).

- Branch `cursor/carry-8767-dc90`, head 093ba58b791308016d9a3aa8358a31da7eb30e1e: #8767's head with `main` 30d6fcc92ff7 merged in. `tools/move/restack.py` crossed the three moves with no conflict. RAN.
- The diff against `main` is 8 files, +145/−31: `verity/difftest/` (README, `protocol.py`, `__init__.py`, two tests) and `verity/protocols/remote/verifier/composition/` (README, `tests/reference.py`, `tests/test_composition_difftest.py`). RAN (`git diff --stat`).
- Quick suites `r20261112-191633-c65b`, rc 0, last line: `63 suites: 63 passed (1 cached), 0 failed; 220.0 min of test time`. RAN (fetched stdout.log).
- No Lean changed, so there is no records run. Question: none.

[2026-11-12T21:33Z] 55e29f901bf1
SORT, TRN-146: carry of #8438, item 3 only. What it is for: `read_slots`' automatic caps rank a cut by the block `circuit.compose` places it in, then by cells, then by nets. Before, a cut that fit 2^27 could be refused at 2^28. READ (the PR body).

- Branch `cursor/carry-8438-dc90`, head 073540b2806c9fed961112dc54f7565159886d4f: #8438's head with `main` 30d6fcc92ff7 merged in. RAN.
- #8520 superseded items 1 and 2: a sparse MLA key block or split part read in slots (`SLOT_KEYS`), and `stage.py`'s `mla-first-block`. At the first move's base, every file of theirs takes the base's version: `chunk_rows.py`, `stage.py`, `served_units/README.md`, `compose_circuit.py`, and the chains and splits tests. Restack then crossed the three moves with no conflict. RAN.
- The diff against `main` is the five files of item 3, +127/−30: `verity/protocols/flock/prover/python/verity_flock/{circuit.py, class_statement.py, read_slots.py}`, `verity/protocols/flock/prover/tests/test_read_slots.py`, and `benchmarks/one_stage/m1_unit.py`'s help text. RAN (`git diff --stat`).
- Quick suites `r20261112-193149-86cf`, rc 0, last line: `55 suites: 55 passed (0 cached), 0 failed; 107.7 min of test time`. RAN (fetched stdout.log).
- No Lean changed. Question: none. The PR's tables were measured before its own merge and are not measured again here. READ (the PR body).

[2026-11-12T21:33Z] 4a7f9d22fbd7
SORT, TRN-146: carry of #8578. What it is for: the committer refuses two ways for one salt to hide two strings. A linked `x` port must take the row table's schema, and an input row committed in full under a registered value's salt must be that value's row (`rows.commit`'s `carried`). READ (the PR title and diff).

- Branch `cursor/carry-8578-dc90`, head 07d87523518804b2a2f427416ae1d07870a5ada0: #8578's head with `main` 30d6fcc92ff7 merged in, across the three moves. Restack stopped at the base merge, so I merged by hand; `interface.py`'s commit call takes both sides. RAN.
- The diff against `main` is 4 files, +106/−22: `verity/commitments/windows/{interface.py, rows.py, tests/test_interface.py}`, and `benchmarks/one_stage/tests/test_a5_wiring.py`. RAN (`git diff --stat`).
- The first quick suites, `r20261112-192108-36b5` on 81711f2d0de4, ended rc 1: `82 suites: 81 passed (10 cached), 1 failed`. The failure was `benchmarks/one_stage`'s `test_a5_wiring.py::test_a_wired_row_is_flipped_where_its_unit_reads_it_in_a_shared_row_member_whose_wired_port_is_not_its_first`: `AttributeError: 'Held' object has no attribute 'bcs'`. That test came to `main` after #8578's base (0174e36bdc), and its stand-in registered value carried no `b || c`, which #8578's committer now reads. RAN.
- The fix, in 07d8752351, is test-only: the stand-in carries each row's `b || c` under its salt in a row/v2 row, as #8578's own `Bound` stand-in in `test_interface.py` does. The committer is not loosened. RAN.
- Rerun `r20261112-210942-7030`, `--changed 81711f2d0de4 --quick`, rc 0, last line: `4 suites: 4 passed (0 cached), 0 failed` (`benchmarks/one_stage` 586 passed). RAN (fetched stdout.log).
- No Lean changed. Question for proofs: is requiring every registered value to carry `bcs` (`_Drawn.strings`) what they want? Core's `Registered` has it, and only test stand-ins lacked it. READ.

