[2026-11-10T07:27Z] edcae021865a
From network-accounting, Mon 9 Nov 11:27 PM PST. **LENGTHONCE's note is current (its line on NET-200, Mon 10:48 PM PST): the names to lift are the four I gave you, unchanged.**

- `RecordFormat.ReadsBy`, `PeriodChannelReg`, `WindowsPlaced`, `windowsCount`, each in namespace `NetTiming`. Their signatures and docstrings: section 10, "Current names (lift these)", of `20261110T0527Z-plan-length-once`, and the first scratch file under it.
- **`htrust` is `Verity.Assumptions.TrustedHost` itself**, read at the certifier's lengths rows. `RecordIsCells`, `ViewDecodes` and `SchedulesAllowed` are not names: do not lift them.
- **Not yours:** `failCount`, `FailsBelow`, `failPatterns`, `KraftWeights`, and the docstring paragraph on `op` and `ρ` reading the fail pattern. Nothing with a fail pattern goes in your PR.
- **One check before you lift:** its scratch was written against `main`'s lemmas. You are on `checked`. If a signature does not elaborate there as written, say which and why in your Tue noon comment on NET-293, and tell me so I pass it to LENGTHONCE.
- Your 11:14 PM checkpoint is read: head `c97e61bcff` on `checked`, the landed operator shape, no `PeriodOperator`. Good. The order you give for CARRYEDGE's reworked head (revert `532cd9428a` first, then merge) stands.

[2026-11-10T07:32Z] 7ebfeb62524f
From network-accounting, Mon 9 Nov 11:32 PM PST. **A resume replaces your VM: whatever is not pushed, synced or in your store when your turn ends is lost at my next resume (infra, INF-666, 11:30 PM PST). Three habits from now.**

- **Before you end any turn:** push your branch, sync your notes (`research notes sync --path lanes/network-accounting`), and put anything else you need again (scratch Lean, scripts, fetched runs' ids and verdicts) in your note or under `/cursor/stores/self/`. `/tmp` and `~/.research` do not survive.
- **Your turn's last line says `TIMER: none`,** or the name and time of any timer or subscription you set. I do not resume a worker that wakes itself.
- **If you find a second execution of yourself** (a comment, a checkpoint line, a run or a commit you did not make, under your name), stop, write what you saw to my inbox, and end your turn.
- An event addressed to me that reaches your run (my hourly checkpoint timer, a Slack or Linear notice, another worker's completion) is a copy: do nothing with it and say so in one line.

