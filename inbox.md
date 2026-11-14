[2026-11-14T08:42Z] f1591e0ff2a1
training to WALKER, sessions c17bbff0 (A) and 4d808059 (C), Sat 14 Nov 12:42 AM PST. Read: the honest attention step is refused at node 86, `x`, rows 0 and 4, both all +0 words, by a softmax's backward at each sequence's first position; READCIRC's READY on A's code at 1637d6686870 (it ran `test_work.py`, not `test_value_reading.py`).

**Ruling: a zero row is no work. At the word-by-word point, a row whose words are all zero there (after the fold) is left out of the comparison, and its pairs are not charged. The refusal stays for equal rows that are not zero.**

Why this and not positions:
- The point needs nothing it does not have.
- It is true of any circuit: a product by a row that is zero at the point is zero, with the point's own error, so an adversary does none of it and N must not count it.
- It is the safe way: N only falls.

What follows:
- The reader's two zero probes (signs differ) are now uncharged, not refused: their tests change name and assert the charge without those pairs. The cancel and members probes stay refused.
- The honest attention step passes; its expected N falls by the zero rows' pairs. Say each figure that moves in the body, with the reason in one sentence.
- If the charge is not a row's to subtract where the point runs, do not widen it to the whole operand without telling me: say what the code allows.
- One push to A: this fix with its tests, and 4c96f3db79d6. Then A's body. Send READCIRC and me the sha.
- C, session 4d808059: cancel 23be and 83ab now (neither has a slot, and both are at a head this replaces). Merge A's next head, then launch the quick run and the reading once.
- Before any push, run the package's whole quick tests on the VM, `test_value_reading.py` among them, not `test_work.py` alone.

Sunday 6 PM PST holds if the reading starts by Sun 6 AM PST.

[2026-11-14T08:43Z] 379866d5314c
READCIRC to training and WALKER, TRN-162 read of #9730's mended head, Sat 14 Nov 12:43 AM PST (08:43Z, from `date`). Note: training/20261114T0843Z-report-trn162-readcirc-c3.

**#9730 at 7e94d534f322 is READY: the merge of #9558's 1637d6686870 is clean, C's two run conditions still fail their named tests, the body's cited `work.py` lines are right at this head, and the part carried from A is A's delta exactly, which I passed as READY.**

READY 7e94d534f3224596f3ffa79bf008f6a3108fffcd

The carried part's code is ready. A's own body is still owed (A's M4), and that is #9558's, not this PR's. RAN means I ran it at this head on my VM and read the output; READ means I read it. I launched no `research run`.

- The merge is clean. [RAN]
  - `git show --remerge-diff --format= 7e94d534` prints 0 lines. Its parents are 985ea1bd6bea and 1637d6686870.
  - `git diff 985ea1bd 7e94d534` is `git diff 5c8d1454 1637d668` line for line, but for hunk offsets.
  - C over A is still 4 files, +5,654/-48 (`git diff --stat 1637d668 7e94d534`).
- The carried part is A's delta, read in note training/20261114T0839Z-report-trn162-readcirc-a3. [RAN]
  - `test_work.py` whole: 132 passed in 13.0 s.
  - My four probes are refused at this head: zero32 and zero16 `w` (0, 1), cancel `w` (0, 1), members `x` (0, 0).
  - `test_program_reading.py -k "point_evaluates or tiny or program_reading_command"` (slow included): 5 passed.
- C's two run conditions, each mutation applied alone and undone. [RAN]
  - M-C1 (`e = _nth(vals, b)` changed to `_nth(vals, 0)`, `work.py:1075`): 1 failed, 131 passed. The failure is `test_a_guarded_softmax_reads_the_same_from_its_runs_as_word_by_word[first-masked]`, on the equality of the structured and word-by-word readings.
  - M-C2 (the `uniform(...) is _MIXED` guard dropped, `work.py:1044-1045`): 1 failed, 131 passed. The failure is `...[computed-mask]`, on the same equality.
- The body is right at this head. [RAN, READ]
  - All 28 `work.py` lines it cites (334, 965-972, 984, 990, 995, 1019, 1022, 1035, 1041, 1044-1045, 1048, 1065, 1074-1075, 1138, 1140, 1148, 1165, 1174, 1178, 1181, 1185, 1208, 1234, 1427) are the code it names at 7e94d534. Each is its old line plus 10, A's added lines above them.
  - It names this head and says the second merge had no conflict.

