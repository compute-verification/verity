# SETUPPORT inbox

- 08:18Z Fri, proofs coordinator: after #5397 is marked, take #6401 too (proofs', queued after #5397 on next). Merge
  #5397's marked head into #6401 (merge commit) and port `SetupCount.linksCheck_cost` to `HmLinks.checkWith`'s new
  `readers` argument. Run the same `lean try` and Security/Proofs records without --update, then mark #6401 and ask ci
  to place it right after #5397. Post both heads on LEAN-66. Mark from a worktree on main (CI-298: an older merge.py
  makes a false caps conflict).
