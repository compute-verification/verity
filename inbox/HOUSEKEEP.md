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
