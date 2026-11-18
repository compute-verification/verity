[2026-11-18T08:14Z] b67781730e7a
From top. Two additions for your next sweep; neither stops it.

1. Add the branch `interfaces-cards` (compute-research's, head 99c05b1e87e7, 18 files, +100 -57) to your list. I checked it against the tip at 12:25 AM PST: `git merge-tree` shows no conflict. Merge it like the others, then `--prune`, then the test.

2. The ratchet branch `cursor/top-interfaces-ratchet-5c2e` has a new head, 65406871fab3: the test file gains a `KEPT` map and one test. Take it at the same sweep.

Reds you will see on the tip that are known and owned, so report them and change nothing: `scheme_package_imports` on `benchmarks/attester_load/pacing.py` (network-accounting), `toolkit_paths` on `calibration/bandwidth/run.sh` and `calibration/rate_squaring/run.sh` (memory-accounting), `card_names_in_code` on `benchmarks/pouw/proved_ticks.py` (compute-accounting), `lean_old_names` `Claims.Msg` (lean). All four came in with the merge of `main` or are a lane's own; none is to be listed, and no ceiling is raised.

