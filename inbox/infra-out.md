# Inbox: Infra runs from the infra repository (top's workers R0 and R5)

Newest first. Read at each hourly checkpoint: `git fetch -q origin cursor/top-graph-inbox && git show origin/cursor/top-graph-inbox:inbox/infra-out.md`.

## 22:40Z Sun 1 Nov (top): launch

- Daniel, Sun 1 Nov 2:01 PM PST: all infra runs from compute-verification/infra, and Verity gradually keeps none of it: code, the team's skills, READMEs and docs.
- One Linear project, "Infra runs from the infra repository" (owner infra, top quarterbacks), with milestones I1 Mon (an infra checkout on every VM and node), I2 Tue (team rules, skills and research from infra), I3 Thu (drive, control loops and Slack events from infra) and I4 Fri (Verity holds no infra).
- Issues: TOP-146 R0 (inventory and `INFRA_PATHS`), INF-412 (infra checkout on every VM; the guard from it), INF-324 (artifact-store interface; research installed from the checkout), TOP-147 R5 (the team's rules and skills), CI-226 (drive from infra's main), INF-413 (control pod, Nebius, ops), COMM-80 (Slack events) and INF-411 (the move and the deletions).
- Order for each component: copy it to infra with its history, switch what runs it, then delete Verity's copy. While two copies exist, edit only the one that runs.
- R0 lands first. R5's Verity PR removes the moved skills from `INFRA_PATHS`, and is marked only once INF-412 is on main.
