[2026-11-14T16:03Z] 8ef83f443be9

READCIRC to training and WALKER, TRN-162 read of #9558's must-fix delta, Sat 14 Nov 8:01 AM PST (16:01Z, from `date`). Note: training/20261114T1601Z-report-trn162-readcirc-a5.

READY 91d8c6b7748d34db4a0935b37fc6474e48c95bf0

The code is ready, and the body is true but for one sentence (S1). The delta is one commit, 91d8c6b7748, touching `points.py`, `test_work.py` and the README.

## MUST FIX

None.

## SHOULD FIX

- **S1. One sentence of the body.** READ, the PR body as WALKER's 8:00 AM edit left it. It now states the commit, the sum rule and the one-row count, the `xzero` and `anti` cases, and the over-leave note in "Not checked".
  - On `sdpa` and `relu` (your question 2): RAN, it is right. At 1637d668 both are refused (`RepeatedRows`, x, rows 0 and 4), 3,840 and 4,864 are their N without the point, and they read 3,776 and 4,800 at the head.
  - Line 52 ("Across members … so a row repeated a member against one shared operand is refused") still lacks "not zero"; `points.py` 33 now has it. Line 22 describes 1637d668's commit, which is true of that commit.
  - RAN, the counts: 8 files, 1,678 in and 474 out against `main` 3d62f5507b01; `points.py` 731 lines; five commits since 5c8d1454d38, 342 in and 126 out; 91d8c6b7748 3 files, 63 in and 18 out. Each matches.
- **S2. One test docstring is false of its own test.** READ: `test_every_mac_reading_a_zero_row_is_left_out` says of `anti` "so no row of one meets a zero row of the other". Member 1's `x` row does meet member 1's zero `W` rows. It should say "so no zero row of one meets a zero row of the other", which is why the independence share undercharges.

## NOTE

1. **The code is my patch, and each half cures only its own probe.**
   - READ: applying my patch file to 9767b09f gives `points.py` identical to the head's but for docstring lines (13–14, 31, 33, `share`'s and `rows`'s docstrings).
   - RAN, my probes at the head: `xzero` N 0 and `anti` N 0, against 48 and 24 at 9767b09f. The others are unchanged: `one` 32, `two` 16, `word` 16, `all` 0, `eq` and `plus1` refused.
   - RAN, the halves at the head: with the old share restored, `anti` reads 24 and only `test_every_mac_reading_a_zero_row_is_left_out[anti]` fails (1 failed, 125 passed). With the early return restored, `xzero` reads 48 and only `[xzero]` fails.
   - RAN: WALKER's two tests, run on 9767b09f's code, fail with N 48 and N 24, and pass at the head.
2. **The statements of the rule.** READ at the head: `points.py`'s opening (13–14), the second point (31, 33), `rows`, `share` and `check`; `work.py`'s refusal paragraph and `RepeatedRows`; and the README. Each is true of the code. "No MAC that reads a zero row is charged" now holds for every activation the pass evaluates, however the rows pair.
   - The README's new sentence is true: the sum is at least the share of MACs that read a zero row under any pairing, and where both operands hold zero rows it can leave out more than reads them, so N only falls.
   - The example "O1 plus -1 times O1 plus 1 is 1 for every input" matches the suite's `equal` case.
   - The body is S1: one sentence.
3. **No reading rises, and no refusal is lost.** RAN: every `WK.read` in `test_work.py`, `test_value_reading.py` and `test_program_reading.py`, logged per test at 9767b09f and at the head (157/161 and 34/34 readings).
   - The only differences are WALKER's two new tests, at N 0 against 48 and 96 without the point. No reading rises, and every refusal at 9767b09f is still a refusal.
   - `sdpa` 3,776, `relu` 4,800, `after_mask` 4,096, `rope` 3,328, `lf` 9,472 and the command's small case 6,029,312 match at both heads.
   - Equal nonzero rows are still refused: my `eq` and `plus1`, and the suite's `equal` and `members`.
4. **Nothing else changed.** READ: the diff is `points.py` (the share, the early return, docstrings), `test_work.py` (`_xzero`, `_anti` and the one test, 42 lines added) and the README's one sentence. No JSON changed, and no other file.
5. **The quick suites.** RAN at the head: `suites.py circuit_properties benchmarks/training_slowdown --quick --changed 3d62f5507b01 --fresh` gave `2 suites: 2 passed`, circuit-properties 143 and training_slowdown 683, with `test_value_reading.py` and `test_program_reading.py` run whole. Ruff is clean on `points.py` and `test_work.py`. READ: the lines over 120 characters are docstring and README prose, as elsewhere in the file; ruff doesn't flag them.

## For the lead to read

- The two must-fixes are cured as you ruled at 1:39 AM. What's left is two one-phrase fixes: the body's line 52 (S1) and a test docstring (S2). Neither holds the mark.
- The case I named last round stays a premise, not this PR's: an activation the pass doesn't evaluate holding more zero rows (INFERRED).
- Events addressed to @training that I saw, and acted on none of: WALKER's 8:00 AM status (bodies at A 91d8c6b7 and C 203a62ad; runs 6673, 9b6e and 5400 waiting). Launched: nothing. Posted: nothing.

[2026-11-14T16:06Z] 7846854e9583
training to every worker and reader, Sat 14 Nov 8:06 AM PST. No reply needed.

**The model rule changed (Daniel, TOP-423): only a lead runs Fable 5.2. You run `claude-opus-5-5-high` and stay on it.**

- Any subagent you launch names its model in the call: `claude-opus-5-5-high`, or `claude-sonnet-5-5-high` for a really well-specified task. Never Fable, never `inherit`, never a launch with no model.
- If a picker offers neither model, do not launch: do the work yourself or send it to me.
- A plan, a design or a judgment no check catches comes to me, in your next message, with what you recommend.

