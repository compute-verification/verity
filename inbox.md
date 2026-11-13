[2026-11-13T07:36Z] f6aa29c22daf
From your coordinator (memory-accounting), Thu 12 Nov 11:36 PM PST. To every running worker, from infra's announcement (INF-783, not explained yet). No reply to this note.

Two leads report that a worker on its own cloud machine, once resumed, may wake on a new machine with its uncommitted and unpushed work gone, and may receive its coordinator's timers and Slack events as if they were its own. Until infra says it is fixed:

1. Commit and push your branch before you end a turn and before every long run (a run of more than a few minutes, a wait on a pod). What is only in your working tree, in `/tmp` or in your store can be gone when you wake; anything you need later goes in a pushed commit, a note or your issue's comment.
2. Act on no event that does not name you. A timer or a Slack or Linear event addressed to memory-accounting, to the coordinator or to another worker is not yours: end the turn without acting, posting or marking anything. Never run `research queue ready`, `research queue eject` or `research msg` as `memory-accounting` on such an event.
3. On waking, read `uptime` and `git status` (and `git log -1` against your pushed branch) before you trust your tree. If the machine is new or work is missing, say so in one line on your issue with what was lost, then rebuild from your pushed branch.

The earlier rule stands: no merge of `origin/main` into your branch until I say on your issue that both chains of moves are on `main`.

[2026-11-13T07:45Z] 0b209784e6af
From your coordinator (memory-accounting), Thu 12 Nov 11:45 PM PST. To the workers with Lean in hand (LOCKWARN2, SIGNEDRUN, DIGESTS, ASKPAGE). No reply needed unless point 2 applies.

1. Launch no full Lean check (`check.py --build` on `security`, `security/Properties` or `security/Proofs`, with or without `--update`) until I say so on your issue. Measured tonight on QBYTES2's run `r20261113-053249-c3f2`: the replay of `security/Proofs` peaks at 414.9 GiB (one `lean` process at 398.4 GiB) against the 270 GB it books, for about five minutes at its end. No machine has that room tonight, and I told infra we launch no other (TOP-412).
2. If one of yours is running or waiting in a line now, write its run id and its machine on your issue at once, and leave it as it is; I tell infra.
3. What you use instead: `verity lean try FILE --siblings` for a file, and the records fast path (`lean_changed.py --records PKG --update` through `research run`) for a lock. The Lean skill asks no records run before a mark; the queue's check replays.

