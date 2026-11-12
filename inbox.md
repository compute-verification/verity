[2026-11-12T22:09Z] fd8d878b70cc
training to BACKWARD2, Thu 12 Nov 2:09 PM PST. **Your word: launch the probe at 3:30 PM PST (23:30Z) and not a minute before, by name with `--on vy-nebius-2` (not `--on auto`), one card under a lease of at most 5 minutes, 16 GB.**

- **Why that minute** (infra on INF-764, 1:49 PM PST): two lanes' timed leases hold all eight cards until 23:30Z, and the research places on the node are closed until then. From 23:30Z they open and eight cards are free.
- **The node stops at 4:30 PM PST whether or not your run has ended.** So launch in the first minutes after 3:30 PM. A job of 20 minutes fits; one that hasn't started by 3:50 PM doesn't.
- **Launch once.** If the launch is refused, or the run hasn't started by 3:45 PM PST, send me the refusal's own line (or the queue's line for the run) and stop: no second try, no other node, no other card count.
- **When it ends, send me at once** the run id, its rc and its last line. I owe infra one line with the run id so it can stop the node the same quarter hour.
- Then the CPU fit of the 8 heads, as you planned, with its ids into `evidence.json`.
- Until 3:30 PM: read your suites and every-word runs as they end. Wait on the clock with AwaitShell or a timer, read the time with `TZ=America/Los_Angeles date`, and don't end your turn before the launch.
- I check at 3:35 PM PST that the run exists. If I find none, I resume you with this message.

