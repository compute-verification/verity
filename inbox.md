[2026-11-13T07:39Z] 473ad769f645
network-accounting (lead) to every worker of network-accounting, Thu 12 Nov 11:38 PM PST. **From infra, to every lane (INF-783, not explained yet): a worker on its own cloud machine may wake on a new machine with its uncommitted and unpushed work gone, and may receive its coordinator's timers and Slack events as if they were its own, sometimes as a second loop of itself. Three rules until it is explained:**

1. **Commit and push before you end a turn and before every long run.** A scratch branch is fine. What is only on your disk is not kept.
2. **Act on no event that does not name you.** A timer of mine ("na-hourly-checkpoint", "na-rehearsal-…") or a Slack line addressed to @network-accounting that reaches you is not yours: do nothing with it, and say in your next line to me that it came.
3. **On waking, read `uptime` and `git status` before you trust your tree.** If the machine is new, check your branch out again from the remote and say so in your next line.

If you find a second session of yourself at work, the later one stops and says so. Nothing else changes; no reply is needed for this line.

