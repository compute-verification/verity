# HOUSEKEEP (bc-3293cb24-a894-534c-a43c-bfe5dbd7ba82): messages from the proofs coordinator

## Coordinator, Tue 10 Nov 00:43Z

- Nothing since your brief. Your dates: the Linear audit list Tue 03:00Z, the NET-283 first pass Tue 20:00Z.

## Coordinator, Tue 10 Nov 01:04Z: Task 3, the Goals page (CON-12). Draft and commands by Tue 08:00Z; it goes between your audit (03:00Z) and NET-283 (20:00Z)

**Why.** Daniel, Mon 4:53 PM PST: he wants to read the Goals page to see what everyone is working on, and today it
confuses him. top reruns its audit Tue 10 Nov 8 AM PST (16:00Z) and tells Daniel what the page shows. proofs has 24
open goals, 18 off track. top's list for us is in its comment on CON-12 (Mon 5:10 PM PST); read it.

**What top asks of each lead:**
1. The page says what we are working on now: each outcome we would report to Daniel this week is a goal with a date
   and a status. Done is marked done; dropped is removed.
2. Every goal sits under a project: a piece is linked from its project with `--wait goal:ID` on the project.
3. Off track means a date or a blocker: the date we now believe, or the one thing it waits on.
4. Our lines: `proofs/spot-figures` waits on `proofs/spot-draft-tonight`, which no longer exists; not written in
   48 h: `proofs/terminology-clean`, `proofs/zero-knowledge`, `proofs/zk-leaks-closed`, `proofs/spot-two-stage-sound`;
   titles over eight words: `proofs/spot-overhead-measured`, `proofs/spot-verifier-cost`, `proofs/terminology-clean`,
   `proofs/spot-claims-checked`. Do not set priorities.

**What I need from you: a draft I can check in ten minutes, and the commands, not run.**
- `research goals show --json` gives the page; `research goals --help` gives the rules (a title of at most eight
  words; a description that says what exactly will be true and where it stands and never restates the fields; an
  off-track goal says why through what it waits on or `--blockers`).
- For each of our 24 open goals, one table row: id; live this week or parked; the status you propose; the date we now
  believe; the one thing it waits on (a goal id, an issue, a PR, a named worker's dated deliverable); the worker who
  holds it; the source of each of those (a line of the swarm index, a note, an issue comment, with its time). Where
  the sources don't say, write "ASK" and the question. Do not guess a date or a status.
- Sources, in this order: the swarm index's LANDING paragraph and resume queue
  (`lanes/proofs/20261106T0639Z-report-spot-swarm.md`), the coordinator log (`20261108T2215Z-report-coordinator-gen4c.md`,
  last 40 lines), `20261109T1855Z-report-goal-audit`, `20261109T1739Z-report-coverage-128k`,
  `20261110T0002Z-report-smell-check-post`.
- Fixed points: the post is Mon 16 Nov; `proofs/spot-128k-attention-proved` is Thu 12 Nov 6 PM PST;
  `proofs/spot-post-ready` is Fri 13 Nov 6 PM PST and is our project for everything the post needs, so every goal
  the post reads is a piece of it. `proofs/sampling-generic` (Mon 30 Nov) is the project for `spot-two-stage-sound`.
  `proofs/zero-knowledge` and `proofs/zk-leaks-closed` (Fri 20 Nov) have nobody on them until the post is out: say
  so as their blocker rather than leaving them off track with no reason.
- Two facts that change several rows: (1) nothing has merged to `main` since Mon 13:12Z; the verified queue (CI-489)
  is how `main` moves, and a goal whose done-condition is "on main" waits on it (CI-496 is my question to ci).
  (2) Lean refuses under `--zk` the units that read a table in lookup slots; the route out is the `_v2` bodies in
  gates (CIR-210, circuits' PR marked Tue 20:00Z), and the 128k goal waits on that.
- Then a file `goals/proofs-goals-fix.sh` on this branch: one `research goals set ...` line a goal (descriptions
  and blockers in files beside it, passed with `@FILE`), and the `--wait` lines for the projects. I run it after I
  have read your table. Put the table in `goals/proofs-goals-table.md`.
- A description is two to four sentences: what will be true, where it stands today with its date, who holds it.
  Plain words; no worker names in titles.

## Coordinator, Tue 10 Nov 01:14Z: my calls on your "Needs the coordinator" list (Task 1). Do these writes after Task 2, before the Goals table

Your Todo rule is right, and so is leaving Done to me. Each write keeps your comment form and quotes its source.

| Issue | Write |
|---|---|
| PRF-366 | Todo; due 2026-11-30. Nobody is on it until the post is out (Mon 16 Nov); the six `RefSession` premises come first then. 30 Nov is the date I believe, with the lock's. The goal `proofs/zero-knowledge` gets the same date in your Goals table. |
| PRF-386 | Backlog, date cleared: it follows its parent NET-208, which network-accounting dates or parks on Fri 13 Nov. If the audit then calls it undated, give it 2026-11-13. |
| PRF-456 | Due 2026-11-13. The coordinator holds the program since HILLCLIMB ended. It is measured through `proofs/spot-ten-percent` and `proofs/spot-overhead-measured` (Thu 12 Nov 6 PM PST) and closes with the figures (Fri 13 Nov). |
| PRF-459 | Its ask was a scope by Sun 8 Nov, and the title claims 4 to 8 times. Find the measured factor on an H200 against the 7 Nov baseline (BREAKEVEN's or LEDGER's latest row, with its run id). If it is 8 times, or 4 times with in-proof hashing cut 4 times: Done, with that line. If not: due 2026-11-12, and a comment giving the factor that stands and that the rest is priced in `proofs/spot-ten-percent`. Say which branch you took. Swarm row 80 is stale either way; give me its corrected line. |
| PRF-468 | Due 2026-11-12. Left: the goal's rerun at steady state on #8281's units, and circuits' Mon 16:57Z request. |
| PRF-471 | Stays Todo; due 2026-11-30, unstaffed until the post is out. The Lean check does not read today's date (check.py:2774 matches the form only), so the lock's `due` of 13 Nov breaks nothing; it is re-dated in the next proofs PR that touches that lock. |
| PRF-461 | Due 2026-11-11 (its goal's date). Post item 1's result yourself: r20261108-162707-a1ee, 14,903 steps on vy-glm53-h8, 0 rows differ (GOALAUDIT's note). The comment lists what is left: items 5 and 7 need a result or a stated scope, item 6's callers. I settle 5, 6 and 7 by Wed 11 Nov 18:00Z. |
| PRF-477 | Project "SPOT post: GLM-5.3 worked example, vllm's serving data and checks". If your command can create a milestone there, "128k · Thu 12 Nov · a served 128k window SPOT-proved", and place it; else no milestone, and tell me. |
| PRF-266 | Due 2026-12-04: it follows VLLM-182. |
| PRF-398 | Done. Its done-condition is your line: `public_boundary.py --all` on main `7af4d24fca`, 0 edges. Quote it in the closing comment. |
| PRF-472 | It asked for marks or "what's left, with a time". #7812 and #7571 merged. If PRREVIEW's verdict on #7590 (swarm row 233) says what is left and when, quote it and close as Done; if it gives no time, due 2026-11-13 and say so. |
| PRF-458 | Due 2026-11-12 (the goal `proofs/spot-moe-binding`). MOEBIND2 holds it; its records run r20261109-234246-963a failed the audit Tue 00:51Z and the fix is under way. |
| PRF-363, 384, 454, 372 | Nothing more now. Put each under its goal in the Goals table with "ASK"; I take them Wed. |

**The 25 lapsed milestones and 14 strays:** not today. After NET-283, propose one rule for each class in five lines
(Wed 11 Nov 06:00Z). `slipped` and PRF-451 are fine as you left them.

TWOPROVE's note says drive looks stalled: it is stopped by design (CI-489, the verified queue, Mon 15:29Z). Nothing for you.
