[2026-11-14T17:08Z] 64de3d3ded90
From your coordinator (memory-accounting): KEYPIN, two things. (1) vllm's serve is done: r20261114-165601-3d07 wrote the weights' registration on node A's card 1 (vllm on TOP-422, 9:06 AM PST); read VLLM-435 for the directory before your three checks. (2) Daniel's ruling of 9:04 AM PST today (top on TOP-422): a simple change is taken to work once it is on main with its tests, and no booked row is spent only to confirm it. So your three checks gate nothing: #9377's mark rests on its suites alone, as before. Run the checks only if they can start and end inside card 1's row (until 10:30 AM PST, 18:30Z) with what is already in place; if node A cannot reach vy-auditor-2 by then, drop them, book no second row, tell vllm on VLLM-435 that the directory and the snapshot are free, and say on MEM-255 in one line what was not run.

[2026-11-14T17:15Z] de18867e8c61
From your coordinator (memory-accounting), Sat 14 Nov 9:14 AM PST (17:14Z). Two things change for every worker today; nothing else in your order does.

1. **No accounting protocol is run any more** (Daniel, 9:10 AM PST, top on TOP-422). Launch no served or GPU run of PoUB, PoUS or the wipe, and book no card or node row. Our work is code on `main` toward the architecture, landed on its tests; no run of record confirms a change.
2. **Condition (a) before your own mark is shorter.** It is now: the suites of the packages your diff touches, and the repository's own suite (`tests/`), pass whole at the head through one `research run`. Not `suites.py --changed origin/main`: the check machines are full, and the queue checks the merged candidate anyway.
   - A full run of yours that is already running may finish, and it counts.
   - A full run of yours still waiting in a line: cancel it (`research cancel RUN --reason TEXT --by memory-accounting`) and launch the short one.
   - A head that differs from a head that passed only by conflict-free merges of `main` needs no new run.
   - Say in the body and in your "Marked" line which suites ran. (b), (c) and (d) are unchanged.

The lane's rules as they stand are in the notes: `lanes/memory-accounting/20261114T1700Z-report-worker-rules.md` (`research notes sync --path lanes/memory-accounting`).

[2026-11-14T17:17Z] 9ced48075a38
From your coordinator (memory-accounting), Sat 14 Nov 9:16 AM PST: KEYPIN, for you beyond the line to every worker; it replaces my line of 9:08 AM PST. The three served checks are dropped, whatever state they are in, unless one is running on the card this minute, which may finish: Daniel parked the accounting runs at 9:10 AM PST, and I told infra to take out node A's opening to vy-auditor-2 (INF-682, 9:13 AM PST). Tell vllm on VLLM-435 in one line that the kept directory and the INT8 snapshot are free, and say on MEM-255 what was not run. #9377: mark it under the short condition (a); if r20261114-154620-9077 is still waiting in a line, cancel it and run the touched packages' suites. Then MEM-333 part 3 (PoUS's figure prints that the setup was checked by the developer, not the auditor, wherever it prints its verdict), code and tests only; say your day for it on MEM-333.

