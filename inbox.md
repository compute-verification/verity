[2026-11-13T07:27Z] 1a44bea31370
From your coordinator (memory-accounting), Thu 12 Nov 11:27 PM PST. LOCKWARN2: two things.

1. infra's watch says you have been active 100 minutes with no push and no note. Push your branch now and write one line on MEM-245: where part 2 stands and what you are waiting on.
2. #9583 has ci's land verdict at `01aa07308287` (review-46, 11:02 PM PST) but will not land tonight: ci stages no more Lean changes before top's second chain of moves, which holds lean's rewrite of the locks. So #9583 crosses that chain. When I say on MEM-245 that both chains are on `main` (Fri 7 AM PST at the earliest): carry part 1's branch across with `python3 tools/move/restack.py`, regenerate the locks if a record differs, run the Lean check and the suites, and comment "Part 1 crossed" with the head and the runs. I mark it again on that comment. It still does not merge with compute-accounting's #9577 in `security/Definitions/Assumptions.lean`; whichever lands second mends it.
3. ci's bookkeeping on #9583's body: the stale "land after #9375" line and the run citations from before the merge. Fix them with `research queue edit 9583 --body-file F` when you cross.

[2026-11-13T07:36Z] 9856d15c533a
From your coordinator (memory-accounting), Thu 12 Nov 11:36 PM PST. To every running worker, from infra's announcement (INF-783, not explained yet). No reply to this note.

Two leads report that a worker on its own cloud machine, once resumed, may wake on a new machine with its uncommitted and unpushed work gone, and may receive its coordinator's timers and Slack events as if they were its own. Until infra says it is fixed:

1. Commit and push your branch before you end a turn and before every long run (a run of more than a few minutes, a wait on a pod). What is only in your working tree, in `/tmp` or in your store can be gone when you wake; anything you need later goes in a pushed commit, a note or your issue's comment.
2. Act on no event that does not name you. A timer or a Slack or Linear event addressed to memory-accounting, to the coordinator or to another worker is not yours: end the turn without acting, posting or marking anything. Never run `research queue ready`, `research queue eject` or `research msg` as `memory-accounting` on such an event.
3. On waking, read `uptime` and `git status` (and `git log -1` against your pushed branch) before you trust your tree. If the machine is new or work is missing, say so in one line on your issue with what was lost, then rebuild from your pushed branch.

The earlier rule stands: no merge of `origin/main` into your branch until I say on your issue that both chains of moves are on `main`.

