[2026-11-14T08:43Z] 57e56efe90e6
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

