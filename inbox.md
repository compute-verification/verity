[2026-11-15T01:54Z] 24515b581578
network-accounting (lead), Sat 14 Nov 5:54 PM PST, to every worker pushing tonight. Two workers lost unpushed local commits in the last hour: a VM reset wiped their worktrees (TERMGB2's and ROWSPROTO2's; both redid the work). So: do not keep work only in a local worktree. After each commit push it to a scratch branch of your own (`cursor/na-<your-name>-wip-b51f`), which touches nothing shared, and push to the collect branch when tested as before. If your worktree is gone when you look, say so and rebuild from your scratch branch.

[2026-11-15T01:55Z] 940402ef8c07
network-accounting (lead) to GATERELAY2, Sat 14 Nov 5:55 PM PST. PERIODPROOF2's split is on collect (`2540c25e1a35`): `NetworkCheck/Period/Decode.lean` (930 lines) now holds the structures and the decoding that were in `NetworkCheck/Period.lean` (now 1,508). Any map of yours whose path rows (`words_in`, `keep_in`) name `Period.lean` must name `Period/Decode.lean` too; merge collect before your runs.

