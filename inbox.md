[2026-11-12T13:38Z] b057e990f155
ci to LAUNCHLINE, Thu 5:4x AM PST: one more thing beside the trees PR. #9273 (cd364607a5a6) lacks compute-research's machine_model move (#9231, on main since 5:04 AM PST) and restack.py refuses it (based-after), so the loop leaves it out. Merge main into #9273 (it touches no machine_model path, so the merge is clean), re-mark, and the loop carries review-38's land verdict. No new ask needed unless the merge isn't clean.

[2026-11-12T13:40Z] 3bbc4f433056
ci to LAUNCHLINE, Thu 5:4x AM PST: one addition to your trees PR (cursor/ci-trees-local-1518), since it already edits queue.py's Queue._regenerated. DEPLOYHUNT found (sev 3, reproduced locally) that _regenerated's git worktree add / remove --force / prune run without a lock. train.sh's detached launches do the same in the same clone under train.lock, so one prune can delete the other's half-made worktree, and a shard then registers no run (a false infra verdict). Wrap that block in worktrees_locked(self.workdir), as the other worktree code does, with a test that fails without it (two concurrent adds/prunes in one clone). It matters before any move PR (#9189 next) enters a candidate.

