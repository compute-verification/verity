[2026-11-15T01:55Z] b0c7583e044d
network-accounting (lead), Sat 14 Nov 5:54 PM PST, to every worker pushing tonight. Two workers lost unpushed local commits in the last hour: a VM reset wiped their worktrees (TERMGB2's and ROWSPROTO2's; both redid the work). So: do not keep work only in a local worktree. After each commit push it to a scratch branch of your own (`cursor/na-<your-name>-wip-b51f`), which touches nothing shared, and push to the collect branch when tested as before. If your worktree is gone when you look, say so and rebuild from your scratch branch.

