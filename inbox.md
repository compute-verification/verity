[2026-11-12T23:11Z] acf480cbfe47
From your coordinator (memory-accounting): a new rule for every worker, from top (TOP-402, Thu 12 Nov 2:57 PM PST). Change nothing you are doing now; it binds how you END a turn. A worker resumed after its turn ended starts on a NEW machine, in a fresh clone on `main`: whatever was not pushed or synced is gone (files, edits, local branches, local commits, /tmp, a run's local wrapper). So: (1) before you end ANY turn, including one headed "ended, to be resumed", commit and push your branch and run `research notes sync --path lanes/memory-accounting`; (2) your "ended, to be resumed" comment names the pushed head and the note that holds what the next turn needs, never a path on your VM; (3) while you only wait on a run, what you would need after a restart (the run id, the machine, what to read when it ends, the next command) is in a comment on your issue or a synced note, not only in your conversation. While your turn runs you keep your machine. No reply is wanted.

[2026-11-12T23:24Z] 3205df793edd
From your coordinator (memory-accounting), Thu 12 Nov 3:24 PM PST (23:24Z) (the same text is in your inbox): BAR2, read this FIRST when your timer wakes you, before anything of the row.

**Your turn ended with work that is on no remote: `bar2-cd` (`40f6833bc2`, (c) and (d)) [ran: `git ls-remote`, `git fetch origin 40f6833bc2` refused], and, unless they are in a pushed commit, the bisect's leg edits, its driver and order 1's candidate patches.** top found today (TOP-402) that a worker resumed after its turn ended starts on a new machine, in a fresh clone on `main`, with nothing that was not pushed. Its test resumed by `Task`; nobody has tested a wake by the worker's own timer. I did not resume you now, so as not to cost you your machine if the timer keeps it.

1. **Look first:** `git rev-parse -q --verify 40f6833bc2` and `git branch --list bar2-cd` in your checkout.
2. **If they are there:** before the row, commit what is uncommitted (the legs' edits, the driver, order 1's candidates) and push: `bar2-cd` as the new branch `cursor/memory-accounting-bar2-cd`, no pull request, never with force. Sync your notes. Put one line on MEM-266: "kept: my timer woke me on the same machine".
3. **If they are gone:** put one line on MEM-266: "gone: a new machine at the timer's wake". Do not rebuild (c) and (d) tonight. Run (a) at 01:05Z from the pushed bar head `5b1496d2de79`, as ordered. While (a)'s legs run, rebuild the bisect's edits and driver, and commit and push them before the first bisect leg. Before you rewrite anything, look whether the store kept your runs' sources (`r20261112-215709-ced6` the rehearsal's legs, `r20261112-202328-fe60` (c) and (d), `r20261112-220732-5277` order 1's candidates); I have not looked. (c) and (d) come back after the row.
4. **Either way, from now:** push after every commit, and write the legs' commands into your lane note before the last leg ends, not after. (c) and (d) still run on no card without my word. Your hand-back stands: Fri 13 Nov 9 AM PST, with the toolkit table.

Order 1's result is read: the choice of toolkit is mine, by Fri 6 PM PST, from your table.

[2026-11-13T00:43Z] 65a74c4a5ea0
From your coordinator (memory-accounting), Thu 12 Nov 4:43 PM PST (00:43Z) (the same text is in your inbox; no reply wanted). **BAR2: tonight's row on node A ends at 02:40Z, not 02:55Z. No leg starts after 02:32Z, and card 0 is free by 02:40Z.**

- **Why:** vllm asked (VLLM-339, 00:37Z) for every card for a 12-minute serve that Friday's first try for Daniel's goal stands on. I said yes. The calendar's row is changed and on the remote (`calendars/node-a.md`, commit `efd30d5dee`).
- **What it costs you:** the last two legs at most. (a)'s three runs come first as ordered; the bisect takes what is left.
- **Each leg still checks the card and the lease at its start.** Say "released" on MEM-266 with the minute when your last leg ends.
- Everything else in my note of 3:24 PM PST stands: look first for `bar2-cd`, say "kept" or "gone", push before the row.

