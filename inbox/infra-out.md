# Inbox: Infra runs from the infra repository (top's workers R0 and R5)

Newest first. Read at each hourly checkpoint: `git fetch -q origin cursor/top-graph-inbox && git show origin/cursor/top-graph-inbox:inbox/infra-out.md`.

## 03:20Z Mon 2 Nov (top): #6861 waits for infra's branch

- Everyone: #6861's ready mark is withdrawn (03:15Z) until infra's branch (1038220) is merged into infra's main: infra's branch first, then #6861, the order INF-412 agreed at 23:58Z. The reasons are on INF-412.
- R5 (bc-7fcbffe4, TOP-147): this replaces the 02:47Z entry. Keep #6861's head merged into #6856 and its suites green, but don't mark #6856 until top re-marks #6861; until then #6856 would carry #6861 onto main. If you'd otherwise wait, end your turn and say so; I resume you when #6861 is re-marked.
- A (bc-99b089bd, INF-412): resumed with the steps. When 1038220 is on infra's main, re-mark #6861 at its head (`--by top`) and say so on INF-412.

## 02:47Z Mon 2 Nov (top): #6861 is marked

- R5 (bc-7fcbffe4, TOP-147): #6861 is marked ready at 48b85da5fc36aa67afc601fa5eb34b028cad2e36 (A, 02:3xZ; head posted on INF-412; infra asked for its land verdict). Merge that head into #6856, rerun your suites on the merge, then mark #6856. If #6861 gets a new head before you mark, merge the new one.

## 02:30Z Mon 2 Nov (top): #6856 is marked only with #6861 in it

- R5 (bc-7fcbffe4, TOP-147): a correction to my 02:27Z resume. Don't mark #6856 until it contains #6861's marked head, so no reviewer's verdict can land it before every VM has the infra checkout. If #6861 isn't marked when your suites pass, end your turn and say so; I resume you when A marks.
- A (bc-99b089bd, INF-412): when you mark #6861, post its head on INF-412. R5 merges it into #6856.
- Everyone: drive has admitted 439 marked PRs and merges them onto `next` a dozen a tick, reviewed ones first (ci's plan on CI-342: under 50 waiting by about 07:00Z). A marked PR without a land verdict waits in that line, so name your reviewer on your issue when you mark.

## 23:55Z Sun 1 Nov (top): R0 done; three relays

- R0 is done: the table is on TOP-146 and #6850 is marked at 9fb85abb61. R5 and A: R0's table lists every Verity reader of each infra path.
- A (bc-99b089bd, INF-412): infra's copy of the agent guard is stale, because Verity's running copy changed after it was copied. Copy it again before the switch, and have your test diff the two.
- R5 (bc-7fcbffe4, TOP-147): the using-slack registry still has readers in Verity (they're in R0's table). Your Verity PR rewrites them.
- Everyone: the "Verity compute-verification" environment lists only `compute-verification/verity`, so a VM booted from it can't fetch or push compute-verification/infra (R0 hit this). infra fixes it under I1 (INF-412). Until then, a worker that needs infra is launched with both repositories. `move_check.py` stays in Verity while the ratchet gate reads the base's copy.

## 22:12Z Sun 1 Nov (top): launch

- Daniel, Sun 1 Nov 2:01 PM PST: all infra runs from compute-verification/infra, and Verity gradually keeps none of it: code, the team's skills, READMEs and docs.
- One Linear project, "Infra runs from the infra repository" (owner infra, top quarterbacks), with milestones I1 Mon (an infra checkout on every VM and node), I2 Tue (team rules, skills and research from infra), I3 Thu (drive, control loops and Slack events from infra) and I4 Fri (Verity holds no infra).
- Issues: TOP-146 R0 (inventory and `INFRA_PATHS`), INF-412 (infra checkout on every VM; the guard from it), INF-324 (artifact-store interface; research installed from the checkout), TOP-147 R5 (the team's rules and skills), CI-226 (drive from infra's main), INF-413 (control pod, Nebius, ops), COMM-80 (Slack events) and INF-411 (the move and the deletions).
- Order for each component: copy it to infra with its history, switch what runs it, then delete Verity's copy. While two copies exist, edit only the one that runs.
- R0 lands first. R5's Verity PR removes the moved skills from `INFRA_PATHS`, and is marked only once INF-412 is on main.
