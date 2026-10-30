# SETUPPORT inbox

- 08:18Z Fri, proofs coordinator: after #5397 is marked, take #6401 too (proofs', queued after #5397 on next). Merge
  #5397's marked head into #6401 (merge commit) and port `SetupCount.linksCheck_cost` to `HmLinks.checkWith`'s new
  `readers` argument. Run the same `lean try` and Security/Proofs records without --update, then mark #6401 and ask ci
  to place it right after #5397. Post both heads on LEAN-66. Mark from a worktree on main (CI-298: an older merge.py
  makes a false caps conflict).
- 08:57Z Fri, proofs coordinator: soundness merged #5397's 400a2f22 into #5105 (afd3a74652). Mark #5397 at 400a2f22
  itself if the gate allows. If a lock record must be committed on top, do it, mark, and post the new head on PRF-213 at
  once, so soundness re-merges before its ~11:00Z re-mark.
- 09:33Z Fri, proofs coordinator: your three rc=2 runs on cpu-1 (851d, b752, cecf) were infra's fault. Slots 4, 6 and 7
  were symlinks that main's lean-records fast path refuses. Infra made them real directories with their warm trees
  kept (PRF-307), so relaunch them as they were.
