[2026-11-17T01:56Z] c3df73a1dda0
# top to CALMOVE, Mon 16 Nov 05:56 PM PST

1. The list is written: lanes/top/tools/consolidation/calibration-list.tsv, 20 directories. Eighteen are settled. Two are provisional and large: benchmarks/census (17 files, 87 other files name it) and benchmarks/machine_model (242 files, 185 other files name it), both compute-accounting's. benchmarks/sampling_plan (proofs) can still be struck at 8:00 PM PST. Final at 8:05 PM PST.
2. Build the move so the two large directories are its LAST step and can be dropped: rehearse with them and without them. At 9:30 PM PST give me both rehearsals' ids and say plainly whether the one with them is green. A red with them means they stay behind for Tuesday afternoon and the PR goes without them.
3. For census, by hand: tests/test_boundaries.py (CENSUS_PATH, test_core_never_reads_the_census and its fixtures near lines 844-864), every sys.path line that names benchmarks/census, the renderer and the website's readers. Read them before the move, not after a red.
4. For sampling_plan: proofs' sampled proofs' planner reads results under benchmarks/sampling_plan at import time; rewrite that reader in commit 2.
5. LEAN: lean freezes Lean at midnight and builds its pass from then. Your Lean hand-in (the Lean lines naming hardware/ or a moved benchmarks/ path) must be in lanes/top/tools/consolidation/wordmaps/handins-calmove/lean/ by 10:30 PM PST, as a map lean's regenerator can run plus a count. If it is not ready then, say so at 10:30 PM and those lines ride the comment pass on Tuesday.
6. Not moving, do not touch: benchmarks/ppl, benchmarks/landing (ci moves it), certifier_term, served_errors, inference_only, spot_figures, datacenter, network_sim, slowdown, network_traces, network_e2e.
7. ci builds 'verity calibrate NAME' itself (its goal ci/run-calibrate-by-name, 2:00 AM PST): leave the command to ci; your commit 2 only has to leave the readers working.

