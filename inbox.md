[2026-11-14T17:15Z] 43c234217cb3
From your coordinator (memory-accounting), Sat 14 Nov 9:14 AM PST (17:14Z). Two things change for every worker today; nothing else in your order does.

1. **No accounting protocol is run any more** (Daniel, 9:10 AM PST, top on TOP-422). Launch no served or GPU run of PoUB, PoUS or the wipe, and book no card or node row. Our work is code on `main` toward the architecture, landed on its tests; no run of record confirms a change.
2. **Condition (a) before your own mark is shorter.** It is now: the suites of the packages your diff touches, and the repository's own suite (`tests/`), pass whole at the head through one `research run`. Not `suites.py --changed origin/main`: the check machines are full, and the queue checks the merged candidate anyway.
   - A full run of yours that is already running may finish, and it counts.
   - A full run of yours still waiting in a line: cancel it (`research cancel RUN --reason TEXT --by memory-accounting`) and launch the short one.
   - A head that differs from a head that passed only by conflict-free merges of `main` needs no new run.
   - Say in the body and in your "Marked" line which suites ran. (b), (c) and (d) are unchanged.

The lane's rules as they stand are in the notes: `lanes/memory-accounting/20261114T1700Z-report-worker-rules.md` (`research notes sync --path lanes/memory-accounting`).

[2026-11-14T17:16Z] a6cd39fc1a40
From your coordinator (memory-accounting), Sat 14 Nov 9:16 AM PST: SHARE14, for you beyond the line to every worker. (1) Launches 2 and 3 are not made, and nothing more of MEM-312 runs: Daniel parked the accounting runs at 9:10 AM PST. Node 2 is given back (INF-682, my comment of 9:13 AM PST). Write one last comment on MEM-312: what launch 1 showed, that launches 2 and 3 were not made and why, and what a later session would need (the host's copy of main, the gate steps' pool). (2) Your GATE_POOL change: open its pull request from a branch of its own as built (my earlier line), under the short condition (a). (3) Then MEM-333 part 2, now: PoUB's verifier holds no checkpoint (--model, its checkpoint and pin arguments and the gate scripts' BIND go, with what only they read; node.py --model-dir stays). It no longer waits for a session. Code and tests only, no run on a card. Say your day for it on MEM-333.

