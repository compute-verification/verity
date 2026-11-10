[2026-11-10T07:31Z] 9d1ce75b05b9
From @circuits (lead), relaying infra's 07:30Z rules (INF-666, INF-671). No reply needed; re-run your command.
(1) Never list command lines, even filtered: no `ps ... args | grep`, no `ps aux`, no `pgrep -af`. The exec daemon's argv holds tokens. Use `pgrep -f PATTERN`, which prints PIDs only.
(2) A resume replaces your VM, so unpushed work is lost: push every commit before you return to me. If you set a timer of your own, say so in your return message; I won't resume you while it is pending.

