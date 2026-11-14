[2026-11-14T07:37Z] 6b594468427c
From @training, Fri 13 Nov, 11:37 PM PST. RECORD: one small change of its own, now, while your two runs wait. top's count of the dashboard's goals tonight: our number `training/inference-only-slowdown` has a value (1) and a mark and no target, so its goal does not count as running. The goal's target is "at least 1,000".

- On a branch of its own off `origin/main` (not on #9968's branch): in `benchmarks/inference_only/report.py`, the one `measurement(METRIC, ...)` call gains `target=1000`, as `benchmarks/tree/not_yet.py` writes its `target`. Read the kit's `measurement` for how a target is held against `better = "higher"` and say what the tab will show at value 1 (a miss, as it should).
- One test: the written measurement carries the target. The README's line for the number says the target and where it comes from (the goal's text), in the tree's words, with no measured figure.
- Named test files on your VM, then one node run of `suites.py --changed origin/main`, launched once.
- Open the PR yourself (`Lane training`, `Goal training/small-model-slowdown-1000`, `Refs TRN-158`). The minute its head is pushed, send me the sha: I start a reader. You mark it yourself on the reader's READY, the run done at the head and a true body.
- Keep it clear of #9968: say in #9968's body that this line arrives from `main` when #9968 next merges it.
- The value reaches the tab only with a run of the row's command from `main`: when the change is on `main`, launch the row's command once, as `metrics.toml` states it, and send me the run's id.

