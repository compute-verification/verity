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

