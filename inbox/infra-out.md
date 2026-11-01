# Inbox: Infra runs from the infra repository (top's workers R0 and R5)

Newest first. Read at each hourly checkpoint: `git fetch -q origin cursor/top-graph-inbox && git show origin/cursor/top-graph-inbox:inbox/infra-out.md`.

## 22:12Z Sun 1 Nov (top): launch

- Daniel, Sun 1 Nov 2:01 PM PST: all infra runs from compute-verification/infra, and Verity gradually keeps none of it: code, the team's skills, READMEs and docs.
- One Linear project, "Infra runs from the infra repository" (owner infra, top quarterbacks), with milestones I1 Mon (an infra checkout on every VM and node), I2 Tue (team rules, skills and research from infra), I3 Thu (drive, control loops and Slack events from infra) and I4 Fri (Verity holds no infra).
- Issues: TOP-146 R0 (inventory and `INFRA_PATHS`), INF-412 (infra checkout on every VM; the guard from it), INF-324 (artifact-store interface; research installed from the checkout), TOP-147 R5 (the team's rules and skills), CI-226 (drive from infra's main), INF-413 (control pod, Nebius, ops), COMM-80 (Slack events) and INF-411 (the move and the deletions).
- Order for each component: copy it to infra with its history, switch what runs it, then delete Verity's copy. While two copies exist, edit only the one that runs.
- R0 lands first. R5's Verity PR removes the moved skills from `INFRA_PATHS`, and is marked only once INF-412 is on main.

## 23:55Z Sun 1 Nov (top): R0 done; three relays

- R0 is done: the table is on TOP-146 and #6850 is marked at 9fb85abb61. R5 and A: R0's table lists every Verity reader of each infra path.
- A (bc-99b089bd, INF-412): infra's copy of the agent guard is stale, because Verity's running copy changed after it was copied. Copy it again before the switch, and have your test diff the two.
- R5 (bc-7fcbffe4, TOP-147): the using-slack registry still has readers in Verity (they're in R0's table). Your Verity PR rewrites them.
- Everyone: the "Verity compute-verification" environment lists only `compute-verification/verity`, so a VM booted from it can't fetch or push compute-verification/infra (R0 hit this). infra fixes it under I1 (INF-412). Until then, a worker that needs infra is launched with both repositories. `move_check.py` stays in Verity while the ratchet gate reads the base's copy.
