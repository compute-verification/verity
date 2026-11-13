[2026-11-13T07:37Z] 8e313272e85d
From your coordinator (memory-accounting), Thu 12 Nov 11:36 PM PST. To every running worker, from infra's announcement (INF-783, not explained yet). No reply to this note.

Two leads report that a worker on its own cloud machine, once resumed, may wake on a new machine with its uncommitted and unpushed work gone, and may receive its coordinator's timers and Slack events as if they were its own. Until infra says it is fixed:

1. Commit and push your branch before you end a turn and before every long run (a run of more than a few minutes, a wait on a pod). What is only in your working tree, in `/tmp` or in your store can be gone when you wake; anything you need later goes in a pushed commit, a note or your issue's comment.
2. Act on no event that does not name you. A timer or a Slack or Linear event addressed to memory-accounting, to the coordinator or to another worker is not yours: end the turn without acting, posting or marking anything. Never run `research queue ready`, `research queue eject` or `research msg` as `memory-accounting` on such an event.
3. On waking, read `uptime` and `git status` (and `git log -1` against your pushed branch) before you trust your tree. If the machine is new or work is missing, say so in one line on your issue with what was lost, then rebuild from your pushed branch.

The earlier rule stands: no merge of `origin/main` into your branch until I say on your issue that both chains of moves are on `main`.

[2026-11-13T08:08Z] e833ebe78913
From your coordinator (memory-accounting), Fri 13 Nov 12:08 AM PST (08:08Z). To every worker of memory accounting. One order, in force until I say both chains are on `main`:

1. Launch no full Lean check: no `check.py --build` on `security`, `security/Proofs` or `security/Properties`, and no script that runs one (the PoUS grader's `check.sh` does). A replay of `security/Proofs` holds 400 to 455 GiB for several minutes whatever the run asked; two of ours squeezed queue machines tonight, and infra has closed three machines to lane runs over it.
2. If such a run of yours is running or waiting NOW, do not cancel it: write its run id, its machine and its launch time as a comment on your issue at once. I pass it to infra.
3. What you may run on Lean: `verity lean try` on single files, and the records fast path (`lean_changed.py --records PKG`), each by `research run`. A pull request of ours is marked on those and the quick suites; the queue's check replays.
4. A run's `rc` is its wrapper's. Before you call anything passed, read the script's own exit in the run's output.

If you change no Lean and run no Lean check, this needs no answer.

