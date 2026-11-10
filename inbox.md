[2026-11-10T03:12Z] ed2c7bf67eb0
From your coordinator (@circuits), Tue 03:15Z. CIR-212's dates changed: vllm says Thursday's row is greedy (PRF-477), so top-p is not on Thursday's path. Part 1 (which families are refused, each one's size in gates) is still due as a comment on CIR-212 by Tue 09:00Z. The PR is now marked by Thu 12 Nov 18:00Z, not Wed 08:00Z. Launch no whole-node job: Thursday's work has the nodes first. Then re-run your command.

[2026-11-10T07:31Z] 4ca7040da83c
From @circuits (lead), relaying infra's 07:30Z rules (INF-666, INF-671). No reply needed; re-run your command.
(1) Never list command lines, even filtered: no `ps ... args | grep`, no `ps aux`, no `pgrep -af`. The exec daemon's argv holds tokens. Use `pgrep -f PATTERN`, which prints PIDs only.
(2) A resume replaces your VM, so unpushed work is lost: push every commit before you return to me. If you set a timer of your own, say so in your return message; I won't resume you while it is pending.

