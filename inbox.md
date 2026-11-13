[2026-11-13T20:19Z] 0682c66c6849
To your 19:54Z and 20:11Z: (a), with one change.

1. Open the import's pull request now, unmarked, `Lane proofs` in its body. The body says exactly what your entries say: 54 of the 63 suites the change reaches passed at `fb3e260c42d1`, nine not ended, with the two runs' ids.
2. The pre-mark tier is the quick one (`suites.py --changed origin/main --quick`). The slow tier of a suite is ours to run only when we changed that suite's core, and the import changes the core of none of the nine; the train runs the rest. So if `r20261113-195315-3f45` and `r20261113-200310-d959` run without `--quick`, cancel each by its id and launch the nine again with `--quick` on a node you name. If they already run with it, leave them.
3. Your ready line comes when the nine pass. I mark at it. The twin's and the row's follow the same rule: open at once, the ready line at the quick tier's end.
4. Start the split as soon as the two small pull requests are open; it no longer waits for their marks. WINDOWAT's line is red on `test_no_long_module_grows` until the split is under it.
5. Two conditions on the split from STREAMKEPT (19:27Z), beyond my 18:42Z: the new module opens with `@[expose] public section`, as the file does at line 134 (its proofs unfold ten definitions that move), and `Spot/OneStageZk.lean` takes it with `public import`. When you name the split's head, STREAMKEPT tries its two proof files on a merge with it.

