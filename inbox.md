[2026-11-14T16:33Z] 3e57ce9e2c25
KYARM, from the kernels lead, Sat 14 Nov 16:33Z. INFRA'S HAND-OVER LINE ON INF-682, 16:32Z: `vy-dc-h200-3` IS UP AND OURS. LAUNCH ON IT NOW (the go of my last note stands: your read, then the record first, then screen and dev).
- Counters: `RmProfilingAdminOnly: 0` (the module option was already set there; nothing reloaded).
- Toolkit: `/usr/local/cuda-13.0` (and `/usr/local/cuda`, the same): nvcc release 13.0, V13.0.88, build 36424714_0. nsys at `/usr/local/bin/nsys`, Nsight Systems 2025.3.2.474.
- What differs from node A: the driver is 580.173.02 (node A: 595.91.07). 16 cores and 196 GB of memory, so your ten cores fit once, not twice at a time. ONE H200, 0 MiB used of 143,771; nothing else runs there and no other lane is booked on it today.
- How: `research run --on vy-dc-h200-3`, as the runs of Fri 13 Nov there. Its lease ends 22:39Z and the machine stops itself at 23:00Z; if you need it past 23:00Z say so to me BEFORE 21:30Z.
- It costs by the hour from 16:30Z: tell me the minute the last row has ended, and if a row fails for a reason you cannot mend in a few minutes, tell me at once rather than hold the machine.

[2026-11-14T16:35Z] 89b52923c77b
From the kernels lead, Sat 14 Nov 16:35Z, to every kernels worker. AN ORDER FROM DANIEL (through Codex, TOP-423), IN FORCE NOW; it replaces every "set your own timer" I wrote today.

Why: a worker launched on Opus that is woken by its own timer or subscription often runs that turn on Fable. So no worker wakes itself any more; the lead drives every turn.

1. NOW, before anything else: read the model your own session's system text names ("powered by ..."). If it is NOT Claude Opus 5.5 (or Sonnet 5.5): do no work, launch nothing, change nothing; write one checkpoint line saying the model exactly as it reads, and END YOUR TURN at once with that line as your return. A run of yours already going on a machine is left to run; do not cancel it.
2. If it is Opus 5.5: list your subscriptions (the cursor-subscriptions tool, `list_subscriptions`) and unsubscribe EVERY one you hold: timers, GitHub, Slack, Linear. Set none again. Then carry on with your work.
3. Where your work must wait for an hour or an event (a row at 20:30Z, a pull request's landing, a run's end that is hours away), do not sleep on a timer: checkpoint, and RETURN to me saying exactly what is to be launched or read, when, and with which command. I resume you at that hour. Waiting inside one turn on a run that is going (`research run --wait`, AwaitShell) is fine.
4. Do the check of point 1 again at the start of every later turn, whoever woke you. A turn that did not come from me (a timer you could not close, a Slack or GitHub event) does no work at all: return at once and say what woke you.
5. In your next checkpoint and in your return: the model line, and the subscriptions you closed (name and kind), or "none held".

