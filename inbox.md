[2026-11-12T06:32Z] f91212e55a41
grade to training, 2026-11-12T06:32:39Z (from `date`). Notice before I touch `width`, as you asked at 05:23Z. No reply needed unless you say stop.

All 29 held entries are equal with no `width` line, with the rule still in the code (RAN, READ by me, at 1a4e836f646, the pushed head):

- r20261112-050442-ed69, 22 entries: "22 entries: 22 equal, 0 differ, 0 refused."; `expected  width` 0 lines (PORT's 14bb had 22); norm 22, loss 8, average 4, as PORT's. Last line "port check exit 0".
- r20261112-050516-d2f6, 6 l16 entries: "6 entries: 6 equal, 0 differ, 0 refused."; width 0 (PORT's 8cb9 had 6); norm 6, loss 4, average 2. Last line "port check exit 0".
- r20261112-050543-3c6a, the xl entry (55 minutes queued for a slot, not started twice): "1 entry: 1 equal, 0 differ, 0 refused."; width 0 (PORT's 2045 had 1); norm 1. Last line "port check exit 0".

Why it can no longer fire: the rule named uses of an updated weight read at 32 bits where the record's read 16, "a route casts to bf16 on CUDA alone". 4.23.0's route casts on every device.

What I do now, in this order, on cursor/training-trace-job-dtype-0cd4:

1. 34a57c08736, a mend of the merge: tests/test_routes.py, the carrier's new test `test_three_rates_under_one_optimizer_are_torchs_three_groups` held a float32 reference against a route that 4.23.0 casts to bf16 ("AssertionError: embed" in r20261112-045843-dd60). The test now runs the route's forward in float32, as tests/test_entries.py does for the same reason.
2. ad11d4e7d86, a commit of its own: `of_record`: the width rule goes. `width` becomes `uses`, which returns differences only: the (MACs a token, upstream) pairs must be equal and so must the bits, 32 for the record's 16 as much as 16 for its 32. Its unit test is restated as that; the slow command test's made-up record keeps the counted bits and names one expected difference, norm; the module's docstring and the README's two lines say so. `RULES` is unchanged (norm, loss, average).
3. 01e8bd81627, 4.23.0's own: the grader test, under your condition. At the bf16 steps O differs at every bmm row and at no other, the circuit's exactly twice the trace's; count and N equal at every row; the totals' O differ by exactly those rows' trace O; the f32 step keeps O and the widths equal. No line of the grader's circuit or of the capture changes.

These three are already run as one scratch head, 01e8bd81627, which the branch fast-forwards to, so the runs are the pushed head's (RAN, READ by me):

- r20261112-054628-500e, the three commands with VERITY_GRADER_SLOWDOWN_CODE set: training_slowdown 538 passed; repository 311 passed; grader 31 passed, none skipped; verity-graph 234 passed; "suites --changed be9a3d566905 --fresh exit 0"; "pins --stale exit 0"; last line "ruff exit 0".
- r20261112-054700-b441, the grader's suite with the variable: 31 passed; last line "grader suite with the harness exit 0".
- r20261112-054735-ae5d (22 entries) and r20261112-054759-3dfd (6): equal, line for line the runs above but the head's line; last line of each "port check exit 0".

Not yet run at that head: the xl entry's port check (I launch it once after the push; if it waits over 30 minutes it stays queued and I say so).

Then: the body of #9029 with your point 2, and the report. Not marked, still a draft.

