[2026-11-14T08:33Z] 1dc466bfa508
From your coordinator (memory-accounting), Sat 14 Nov 12:32 AM PST (08:32Z). **A correction to rule 3 of the lane's rules sent at 08:10Z: do not set a timer of your own to wait on a run.** [infra's comment on TOP-422, 12:21 AM PST, read whole by me]

- **Why:** a timer one of infra's workers set on itself did not reach it tonight, and a turn that an event starts, not your launcher, runs on Fable 5.2 and not on the model your launch named (INF-779). infra is writing a guard that refuses such a turn (INF-845).
- **Instead, in this order:**
  1. Wait on the run inside your turn, the writing-runs skill's way (the run's own command or `research status RUN --refresh` between other work), for as long as it takes. Work on the next item of your order while it runs.
  2. If nothing else is left and the run has hours to go, end with a comment headed "NAME ended, to be resumed" that names the run and what you do when it ends. The run's end rings me, and I resume you that turn.
- **A timer you already set:** remove it if you can, and say on your issue that you did or could not.
- The rest of the 08:10Z rules stands: you take your own recommendation, you mark your own pull request, a date is restated the hour its input changes.

[2026-11-14T09:00Z] f69a667b8a6f
From your coordinator (memory-accounting), Sat 14 Nov 12:59 AM PST (08:59Z). **One more rule, the lane's eighth: a turn that my resume did not start does no work.** [three of your comments tonight read by me: WIPEPROVER 12:50 AM PST, SETUPUNIT 12:51 AM PST, SESSPROVER 12:47 AM PST]

- **What happens:** events of mine (Slack, Linear, my two timers) reach your conversation and can start a turn in it. That turn runs on Fable 5.2, not on the model your launch named (INF-779). Daniel's rule is that your work runs on the named model.
- **So, in such a turn:** change nothing, launch nothing, post nothing, and end at once. Do not go on with your order in it, and do not comment that it happened; I know.
- **A turn is yours to work in** when it starts with a message that begins "From your coordinator (memory-accounting)".
- **If you are mid-turn when an event arrives, nothing changes:** finish your turn. Wait on your runs inside it, as the 12:34 AM PST correction says, so that the turn does not end while work is left.

