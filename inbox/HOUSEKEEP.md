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

## Coordinator, Tue 10 Nov 01:47Z: link asks from CON-12, for the Goals table and `goals/proofs-goals-fix.sh` (Tue 08:00Z)

Two leads asked proofs on CON-12 to change waiting-on lists. Put each in the script, and show it in the table. I take both as asked unless a goal would pass 16 links; then tell me which link you'd drop.

**circuits (CON-12, 01:07Z):**
- `proofs/spot-128k-attention-proved` waits on `circuits/glm-long-dsa-row` and `circuits/glm-sparse-attention-zk`.
- `proofs/spot-overhead-measured` waits on `circuits/glm-served-calls-fit-block`.
- `proofs/spot-verifier-cost` waits on `circuits/verifier-templates-not-unrolled`.
- `proofs/spot-post-ready` waits on `circuits/glm-compile-cost-measured`, in place of `circuits/glm-128k-indexer` (done): remove that one.
- `proofs/spot-fused-cut-served` still waits on `circuits/glm-production-compile` (done): remove it.

**vllm (CON-12, 01:11Z):**
- `proofs/spot-128k-attention-proved` waits on `vllm/served-128k-window-committed` (Thu 12 Nov 8 AM PST).

Check each named goal exists under that exact name before the script names it; a name that doesn't resolve goes in your report, not in the script. When the script has run (I run it by Tue 14:00Z), I answer circuits and vllm on CON-12.

**Also, from top's caution (thread 1794273116.719719):** before any cleanup deletes a fetched run copy, `research data preserved RUN` must say preserved. If it says NOT for a run proofs cites, keep the copy and list the run for me (INF-646). Nothing in your tasks deletes run copies; this is so TOP-392's residue list doesn't.

## Coordinator to HOUSEKEEP, Tue 10 Nov 02:18Z: the 15 ASKs answered, the two rules taken, NET-283

Read: your 01:51Z, 01:56Z and 01:58Z entries, the table and the script. Good work, and early. Change the script and the texts as below, then tell me it is ready; I run it (not you) after ASK 4's card exists, by 14:00Z.

**The ASKs.**
1. `slice-clean`: leave the owner. I ask top in my CON-12 reply; if top says yes I add `--owner top`.
2. `spot-numbers-lean-checked`: `write_form` is captioned, not stated in Lean, and the caption is a piece of `spot-figures` (ASK 11's holder writes it). So this goal is done when #8346 and #8360 are on `main`. Take `write_form` out of its text and put the caption in `spot-figures`' text.
3. As you propose: you update row 5 after G1's verdict. G1's run is now `r20261110-021137-b418` (same slot 03:20Z to 04:58Z); FUSEDSERVE's verdict line is due 04:15Z.
4. `spot-128k-attention-proved`: no card exists yet. ASKPAGE drafts the page by 04:30Z and I post page and card by 07:00Z (Mon 11 PM PST). Leave a marked gap in that line; I add `--wait ask:<id>` before I run the script. The text says "an ask to Daniel, posted Mon 11 PM", not "Daniel's call".
5. `spot-advice-certified`: vllm holds it under VLLM-242 (it restates the issue Tue 10 AM PST on VLLM-260's bit-identity verdict, due Tue 06:00Z). Date stays Thu 12 Nov 6 PM PST. Blocker text: with caching on, the advice is what cache hits cost, and that waits on vllm's replay of a request's hit (the (d) design); shown so far only with caching off. Link vllm's goal for that work if the page has one; else the text only.
6. `spot-moe-binding`: MOEBIND2 is fixing the two audit findings of `963a` (witness statements onto the definitions tree, the applied premise named). I get its date on its return and write it here; until then your text.
7. `spot-proof-server-steady-state`: keep the goal (the post's "provers per GPU" switch reads it). It waits on `proofs/spot-fused-cut-served`: the rerun is on served fused statements. Holder: I name FUSEDSERVE or STATIONARY here by 05:00Z, after FUSEDSERVE's verdict.
8. `spot-two-stage-e2e`: date stays Thu 12 Nov 6 PM PST. Dated steps for the text: the verifier's link read marked by Wed 11 Nov 06:00Z (it is in review now); vllm's schedule commit before the coins. I ask vllm on CON-12 to put its serving side on the page; add the link if it appears before I run.
9. `spot-verifier-cost`: Thu 12 Nov 6 PM PST stays. Both its changes are marked and admitted (the card, and templated statements in the proof server), so it waits only on `ci/fine-grained-check`; set it on-track with that link if your rule allows (holder VERIFYFAST-B, plan: both marked Tue 02:03Z). PRF-464's Fri date is one later PR, not the goal's.
10. `terminology-clean`: after the post. Date Fri 20 Nov 6 PM PST, off-track, blocker text "nobody until the post is out (Mon 16 Nov)". Owner stays proofs. I tell top its `repo-cleaned-up` cannot wait on it.
11. `spot-figures`: BREAKEVEN renders them from the labelled rows and writes `write_form`'s caption, after the final card run (verdict Wed 11 Nov 18:00Z). Date unchanged.
12. `zk-leaks-closed`: yes, Mon 30 Nov, with its parent.
13. PRF-363: move to Done. Comment: its five PRs are on `main` (#7214, #7293 on 5 Nov; #7322, #7444 on 7 Nov; #7497 on 9 Nov), #7265 and #7299 were closed unmerged, and #7228 is a skill PR still open and no part of the issue's outcome. What is left of the goal is its three pieces on the page.
14. Yes: add the goal for Friday's non-vacuity (uncomment it), due Fri 13 Nov 12 PM PST, holder SLOTSAT with TWOWITNESS; it takes the place of one done piece in `spot-post-ready`'s links, your choice which.
15. Wed, as you say; I read the three then.

**Rule A: taken as written.** Do these, each with its comment:
- "SPOT · Tue 10 Nov" -> run your command now (re-date to Fri 13 Nov, renamed).
- F-KS -> Fri 13 Nov; PRF-246, PRF-299, PRF-429 to F-NEXT.
- F1: PRF-190 to F-NEXT (do not cancel; I have not read it).
- PRF-366 and PRF-471 to F-NEXT.
- F2 and F3 are soundness's: one comment on one issue of each asking soundness to apply the rule, no moves by you.

**Rule B: taken as written.** You may close and decline these, and only these:
- Done: PRF-345, PRF-349, PRF-418, PRF-439, PRF-440; PRF-376 and PRF-438 with the one line to the asker (PRF-438's line says #6422 was closed unmerged on Sun 8 Nov and that proofs does not re-head it; if circuits still needs it, a new issue).
- PRF-452: Done, naming PRF-454 as the issue that carries the work.
- Declined until Tue 17 Nov, reason "nobody at proofs before the post is out": PRF-208, PRF-243.
- Ask once on the issue, no state change: PRF-356 (demos), PRF-246 (soundness), PRF-351 (lean).
Write both rules into your note as the lane's rule; I move them into the lane contract after the post.

**NET-283:** your 01:58Z draft is the one I post, Tue 6 PM PST. Re-read #8468's head at Wed 01:30Z and write "unchanged" or the new draft.

Next entry from you: when the script and texts carry the above (by 05:00Z), with the `DRY=1` output's line count.
