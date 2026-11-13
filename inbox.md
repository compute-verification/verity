[2026-11-13T08:12Z] 786cbe76b9a3
**#9621 was ejected from the queue at 07:19Z.** ci's line: candidate `37574fd4e31e` failed `benchmarks/harness`'s `tests/test_served_row.py`, 12 tests, in `r20261113-064715-98e6`: `verity_bench/exfiltration.py:629 ValueError: the section's record holds class 'p4-o2': class is w<window>/p<P>-o<O>`. Another change in the same train (#9673, network's, still open) makes a class carry its window.

- **ci's cure:** once #9673 is on `main`, merge `main`, give the served-run reader's class the window prefix (`w1/p4-o2` in ci's example), run `benchmarks/harness`'s suite, and I mark the new head.
- **My order on it:** no merge of `main` into #9621's branch yet. Both chains of moves land first (chain 2 about 13:00Z to 14:00Z), and every proofs branch is carried once after that with `tools/move/restack.py`, on my word "carry".
- **What you do now:** on a scratch branch, merge #9673's head (`cdf0f4a9047a`) with #9621's (`13a60635abb2`), write the fix there, and run `benchmarks/harness`'s suite `--on auto`. Say in an entry which lines change and whether the window in the class is read from the run's record or assumed. Read first whether the failure is in the reader or in the test's fixture record: ci's line names the place it was raised, not the cause.
- **The guard** (`cursor/proofs-count-guard-c41d`) waits behind #9621 as before.
- If #9673 lands with a different form of the class than its head has now, your fix follows `main`, not the head.

Entry when the scratch run ends. No ready line before the carry.

