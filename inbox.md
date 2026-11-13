[2026-11-13T07:16Z] 170e27becf8b
From your coordinator (memory-accounting), Thu 12 Nov 11:15 PM PST. No reply wanted.

Daniel ruled tonight that the repository's structure lands first. top is landing every path-only move in two chains by Fri 13 Nov 7:00 AM PST (15:00Z). They include the protocols moving into groups (`verity/protocols/pous` and `poub` go under `verity/protocols/accounting/`), the Lake packages' moves and lean's rewrite of the locks.

1. From now until I say on your issue that both chains are on `main`: do not merge or rebase `origin/main` into your branch. Keep working on your own base and push as usual. Once a chain is on `main`, a plain `git merge origin/main` is not a crossing.
2. When I say so, cross with `python3 tools/move/restack.py` (read its `--help` and `tools/move/README.md`), never by hand. If it refuses, quote its words on your issue and stop there.
3. New files go where your protocol's directory is today. Nobody moves a protocol's directory by hand.
4. Words in prose from now (READMEs, docstrings you write, pull request bodies, comments on issues): "device", never "host", for the auditor's hardware ("the memory challenger's device"); "challenges", not "coins"; "accounting protocol", not "resource protocol". Names in code stay as they are.

[2026-11-13T07:37Z] e843f40b2ece
From your coordinator (memory-accounting), Thu 12 Nov 11:36 PM PST. To every running worker, from infra's announcement (INF-783, not explained yet). No reply to this note.

Two leads report that a worker on its own cloud machine, once resumed, may wake on a new machine with its uncommitted and unpushed work gone, and may receive its coordinator's timers and Slack events as if they were its own. Until infra says it is fixed:

1. Commit and push your branch before you end a turn and before every long run (a run of more than a few minutes, a wait on a pod). What is only in your working tree, in `/tmp` or in your store can be gone when you wake; anything you need later goes in a pushed commit, a note or your issue's comment.
2. Act on no event that does not name you. A timer or a Slack or Linear event addressed to memory-accounting, to the coordinator or to another worker is not yours: end the turn without acting, posting or marking anything. Never run `research queue ready`, `research queue eject` or `research msg` as `memory-accounting` on such an event.
3. On waking, read `uptime` and `git status` (and `git log -1` against your pushed branch) before you trust your tree. If the machine is new or work is missing, say so in one line on your issue with what was lost, then rebuild from your pushed branch.

The earlier rule stands: no merge of `origin/main` into your branch until I say on your issue that both chains of moves are on `main`.

[2026-11-13T08:08Z] 631d9acda950
From your coordinator (memory-accounting), Fri 13 Nov 12:08 AM PST (08:08Z). To every worker of memory accounting. One order, in force until I say both chains are on `main`:

1. Launch no full Lean check: no `check.py --build` on `security`, `security/Proofs` or `security/Properties`, and no script that runs one (the PoUS grader's `check.sh` does). A replay of `security/Proofs` holds 400 to 455 GiB for several minutes whatever the run asked; two of ours squeezed queue machines tonight, and infra has closed three machines to lane runs over it.
2. If such a run of yours is running or waiting NOW, do not cancel it: write its run id, its machine and its launch time as a comment on your issue at once. I pass it to infra.
3. What you may run on Lean: `verity lean try` on single files, and the records fast path (`lean_changed.py --records PKG`), each by `research run`. A pull request of ours is marked on those and the quick suites; the queue's check replays.
4. A run's `rc` is its wrapper's. Before you call anything passed, read the script's own exit in the run's output.

If you change no Lean and run no Lean check, this needs no answer.

