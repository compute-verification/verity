[2026-11-12T18:37Z] d920bc36091d
ONSITEMAIN, from the network-accounting coordinator, Thu 12 Nov 10:37 AM PST. **infra said yes at 10:00 AM PST on INF-743: Friday's slot runs from 5 AM to 10 AM PST. The launch line is the 5 h one: `--timeout 5h`, `JOB_SECONDS=16200`, six negative workers.** Both lines are in your `LAUNCH.md` already; nothing in it changes.

- The rest holds: head `db45e589e10f`, `--pool network --pool-wait 1800 --mem-gb 650`, `V3C_LEAN_WORKERS=1`, `V3C_NEGATIVE_WORKERS=6`, `AUDIT_MEM_GB=650`.
- infra reads the node at 4:45 AM PST and posts. I resume you at 4:48 AM PST with the word to launch, as planned.
- infra asks: if the job ends by 8:10 AM PST, say so and it opens the node's places then. Put that line in your 6 AM report's plan.
- In your 2 PM report, restate the member thresholds for 16,200 s in one line (which members are past it with six workers).

No answer is owed before 2 PM.

TIMER: none

