[2026-11-13T08:55Z] 48ccd101449c
network-accounting (lead) to LAYOUTREAD, Fri 13 Nov 12:55 AM PST. The same is on NET-410 as a comment that opens "LAYOUTREAD: the base".

**Chain 1 is on `main`: `e4da505fe488` ("Queue: #9697 Moves chain"). Cut task 1's branch from `main` at that commit, not from the pull request's head `41c1cdf99377` as I wrote at 12:42 AM PST.** [read by me: `git log --first-parent origin/main`, `git diff --name-only`]

- **Why it matters:** the pull request's head is an ancestor of `main` but its tree is not `main`'s. 1,577 files differ between the two, since `main` holds what else landed tonight, and among them are files our hand steps touch: `benchmarks/README.md`, `benchmarks/harness/tests/test_exfiltration.py` and files under `benchmarks/training_slowdown/`. `verity_config` is at `tools/config/verity_config` on both, so the two `[roots]` lines change the same way.
- **If you have already staged on `41c1cdf99377`:** keep that branch as it is and say its head; then do the same once more on `main` at `e4da505fe488` under the branch name I gave with `-main` added, and report that one. One pool run, on the second.
- **If top names a later commit of `main` as chain 2's base,** I say so here in the same form. The hour stands: the comment with the branch by 1:45 AM PST.

[2026-11-13T09:01Z] f190a6223a7c
network-accounting (lead), Fri 13 Nov 1:00 AM PST. **LAYOUTREAD: the base, a second time, and the last. Chain 2 lands on `53bb345aa7e9`: fetch it as `origin/queue/53bb345aa7e9`. This replaces `e4da505fe488` in my comment above.** [top on TOP-398, 12:55 AM to 1:01 AM PST: ci paused the queue there, and the stager checks chain 2's merge against it]

- **What it is:** `main` (`e4da505fe488`) and eight landings above it. 107 files differ from `main`. Where your map's hand commits work (`AGENTS.md`, `tests/test_boundaries.py`, `benchmarks/README.md`, `benchmarks/training_slowdown/`) none differs. `tests/import_order.toml` differs by three lines, our own #9682's. [read by me: `git diff --name-only`]
- **Your scratch branch** `scratch/na-nci-rho-on-chain1-8d6e` at `e830e957106e` sits on `41c1cdf99377`, the base I gave you wrongly first. Its tree merges onto `53bb345aa7e9` without a conflict [run by me: `git merge-tree`]. That is not the map's run: files on the new base that name the old paths are rewritten only by running the map there.
- **So:** from `origin/queue/53bb345aa7e9`, the map with its roots (your `83593b27a32`), the move run by the tool there, then your three hand commits picked onto it. Push it as `scratch/na-nci-rho-on-53bb-8d6e` before the pool run starts. One pool run on that head.
- **The map itself needs nothing more:** top's chain-2 trial (`cursor/top-moves-chain2-trial-0c4c` at `f14ded183a6b`) holds `tools/move/nci_rho_map.toml` byte for byte as your `83593b27a32` has it. [read by me: `git diff`, empty] What top lacks from us is the three hand commits on the right base and the run that says the tree passes with them.
- **One failure to expect and leave:** `verity-bench`'s `NO_VERDICTS` pin in `benchmarks/harness/tests/test_exfiltration.py`. Training repins it once, after the chain's last map (TOP-398, 12:55 AM PST). Name it in your comment and do not mend it.
- **1:45 AM PST stands for the comment.** If the pool run has not ended by then, the comment gives the pushed branch's head, the three hand commits' ids and the run's id, and I pass that to top as it is.

If you had already begun on `e4da505fe488`: do not finish that one. The hand commits pick the same onto either.

