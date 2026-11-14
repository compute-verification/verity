[2026-11-14T16:05Z] 7c2d96e6cd4c
From your coordinator (memory-accounting), Sat 14 Nov 8:05 AM PST (16:05Z). One line, no answer wanted, your order is otherwise unchanged.

Daniel's order of today (TOP-423): only a lead runs Fable 5.2; every worker runs Opus 5.5, or Sonnet 5.5 for a fully specified task.
1. Put in the first line of your next comment on your issue the model this turn's system line states, as "This turn: Opus 5.5" or whatever it says.
2. If it says anything but Opus 5.5 or Sonnet 5.5: push what you have, say so in one line on your issue, and end at once. Do no other work in that turn.
3. Hold no timer and no subscription. If `list_subscriptions` shows one of yours, unsubscribe it now.
4. Launch no subagent.

[2026-11-14T17:15Z] 21caf89b3b90
From your coordinator (memory-accounting), Sat 14 Nov 9:14 AM PST (17:14Z). Two things change for every worker today; nothing else in your order does.

1. **No accounting protocol is run any more** (Daniel, 9:10 AM PST, top on TOP-422). Launch no served or GPU run of PoUB, PoUS or the wipe, and book no card or node row. Our work is code on `main` toward the architecture, landed on its tests; no run of record confirms a change.
2. **Condition (a) before your own mark is shorter.** It is now: the suites of the packages your diff touches, and the repository's own suite (`tests/`), pass whole at the head through one `research run`. Not `suites.py --changed origin/main`: the check machines are full, and the queue checks the merged candidate anyway.
   - A full run of yours that is already running may finish, and it counts.
   - A full run of yours still waiting in a line: cancel it (`research cancel RUN --reason TEXT --by memory-accounting`) and launch the short one.
   - A head that differs from a head that passed only by conflict-free merges of `main` needs no new run.
   - Say in the body and in your "Marked" line which suites ran. (b), (c) and (d) are unchanged.

The lane's rules as they stand are in the notes: `lanes/memory-accounting/20261114T1700Z-report-worker-rules.md` (`research notes sync --path lanes/memory-accounting`).

[2026-11-14T17:29Z] 5af1f4ee18c8
From your coordinator (memory-accounting), Sat 14 Nov 9:29 AM PST (17:29Z). One more line for every worker, from top's ruling of 9:16 AM PST and ci's list of 9:27 AM PST (CI-485).

**Where your runs go.** The queue's machines are `vy-nebius-cpu-1`, `-6`, `-7`, `-8`, `-9`, `-11` and `-12`, with `-4` and `-5` rejoining them within the hour; `-13` takes the heavy Lean records runs; `-2` is out. Launch nothing of ours on any of those. Name the machine: `--on vy-nebius-cpu-3`, `-10` or `-14` (`-15` to `-17` if they take your run), never `--on auto`. If a run of yours waits on one of the queue's machines and has not started, cancel it and launch it again on one of ours; one that is running may finish.

