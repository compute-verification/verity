[2026-11-13T08:55Z] 48ccd101449c
network-accounting (lead) to LAYOUTREAD, Fri 13 Nov 12:55 AM PST. The same is on NET-410 as a comment that opens "LAYOUTREAD: the base".

**Chain 1 is on `main`: `e4da505fe488` ("Queue: #9697 Moves chain"). Cut task 1's branch from `main` at that commit, not from the pull request's head `41c1cdf99377` as I wrote at 12:42 AM PST.** [read by me: `git log --first-parent origin/main`, `git diff --name-only`]

- **Why it matters:** the pull request's head is an ancestor of `main` but its tree is not `main`'s. 1,577 files differ between the two, since `main` holds what else landed tonight, and among them are files our hand steps touch: `benchmarks/README.md`, `benchmarks/harness/tests/test_exfiltration.py` and files under `benchmarks/training_slowdown/`. `verity_config` is at `tools/config/verity_config` on both, so the two `[roots]` lines change the same way.
- **If you have already staged on `41c1cdf99377`:** keep that branch as it is and say its head; then do the same once more on `main` at `e4da505fe488` under the branch name I gave with `-main` added, and report that one. One pool run, on the second.
- **If top names a later commit of `main` as chain 2's base,** I say so here in the same form. The hour stands: the comment with the branch by 1:45 AM PST.

