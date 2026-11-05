---
id: 20261105T1730Z-report-landing-volume
campaign: landing-speed
lane: top
kind: report
status: draft
repo: compute-verification/verity
origin: "Daniel, Thu 5 Nov 9:06 AM PST: use a bunch of agents to cut landing latency; learn from two weeks of failures. top's worker, angle VOLUME (less to land)."
---

# Landing volume: less to land (final, 18:45Z)

Branch `cursor/top-landing-volume-8a78`, scripts in `tools/research/studies/landing_volume/` (contain.py, carried_csv.py,
flow.py, labels.py, heads.py, reheads.py, carry_in.py, carry_net.py, carry_strict.py, trains.py, train_sim.py,
rewait.py). Snapshot: 649 open PRs at 17:12Z, queue status at 17:20Z (`next` 9bdd6207, `checked` aa7843fd, `main`
c6ba5690). Week = 29 Oct 17:00Z to 5 Nov 17:00Z.

Arts:
- Open carried/superseded PRs with carrier, evidence and action: **art:e454e6b155c6a0c31292cdb8f5fcef3579e5b61b6f4860bc9f10db9c03d0ca3b** (309 rows).
- Data and scripts (gh pulls, store labels, containment, re-heads, flow): art:16168c795e76d4a48e65950374819dfc78b4042c2af556f16b4cf58340074513.
- Carried landings, simulator runs, re-verdict waits: art:0d6d97dba4a9a57797a5a94f3625a4e442e4930f7bfe3fa699832f211ccc4a3e.
- Strict verdict carry, holds it would skip, heads against time queued: art:3ad9e27fbffd9080caa6c09d16795f2f9d397a076b09a7ef0ebb70f952d43c80.
- Simulator: DESIGN's `sim.py` (cursor/top-landing-design-8b2a) on its data art:fb0c4e715c3aea1f0fca101fe64e9f6f7c063f071d478dbbb427d057f2628e34,
  unmodified; `train_sim.py` only takes riders out of its PR list and gives them their carrier's `main` time. Its
  baseline rows reproduce DESIGN's table exactly.

## Findings

1. **Most landing work goes to heads that never land, and it is a symptom of waiting.** Week: 1,251 land verdicts (446 on
   heads that landed, 271 on open PRs' current heads, 376 on superseded heads, 158 on PRs closed unlanded: 534, 43%,
   wasted); 1,572 marks (727, 46%, wasted). 806 re-heads after a labelled head. Extra marked heads per landed PR grow with
   time queued: least squares 0.31 + 0.175 a day (699 PRs reaching `main`); 0.19 for PRs that took 6 to 12 h, 1.25 over
   96 h (17% re-headed against 57%). At DESIGN's step-4 latency (p50 5 h) the week's 618 extra heads of landed PRs fall to
   about 140 to 245 (60 to 78% fewer), and with them about 230 to 290 of the week's 376 verdicts on superseded heads.
   Confounded (easy PRs land fast), so take half: still more than every direct lever below together. Re-heads put 9,159 PR-hours a
   week (17% of first-mark-to-`main`) before the final head's mark, outside DESIGN's simulator, whose clock starts at the
   landed head's mark.
2. **What the re-heads are.** 389 (48%) brought in other PRs' heads, 342 (42%) were real edits, 60 (7%) changed nothing
   of the PR's own. Land verdicts never carry across heads (`unreviewed` reads one head's labels) and marks only to an
   identical tree (`same_tree_mark`: 34 of 1,466 re-heads since 22 Oct). Strictly (a lock file is own change; every
   brought-in PR has a land verdict, with no hold, on a head the new head contains), 129 a week left the PR's own change
   unchanged: 92 re-verdicts, 127 re-marks, 123 PR-hours of verdict wait (p50 0.56 h, p90 3.0 h). Looser (generated files
   excluded, brought-in PRs unchecked): 233, 165 verdicts, 214 marks, 491 PR-hours, the tail waiting on unreviewed
   brought-in PRs.
3. **Reviewers do judge merges.** 6 holds a week fell on strict carry-able heads, each about another PR's change: stale
   carried heads that conflict (#5105, #5669, #6385), a lock keyed under a name `next` renames (#6518), builds or
   `#guard`s failing on `next` (#7051, #7105). All six are failures the queue's merge refusal, the preflight or the Lean
   check reach mechanically. The looser count would also have skipped #6325 and #6330 (a held H200 cap carried in a
   `main` sync) and a `lean-audit.json` merge dropping 22 entries (#6518): why locks stay own change.
4. **38% of landings were carried by another landing, but folding them only pays while admission is capped.**
   `trains.py`: since 1 Nov 08:08Z, 155 of 403 landings had their landed head inside a later or same-pass landing (92 same
   lane, 64 of those within 30 min of their carrier); 18 of `next`'s 51 members (35%) are inside another member. In
   DESIGN's simulator (p_break 2%), lane trains (248 riders: no admission, no member, land with their carrier) move
   current p50 51.1 h to 48.5 h, and on its steps 1 to 3 20.2 h to 9.4 h (14,225 to 9,980 PR-hours a week). After step 4
   (admission off the tick) they cost: p90 9.3 h to 19.5 h, 3,369 to 5,781 PR-hours, because a carried PR is nearly
   always marked before its carrier and then waits for it. Refusing a mark only when a marked carrier exists reaches 10
   PRs. Carrier sets across lanes are worse at every step. At 5% the same holds (art:0d6d97db).
5. **30% of open PRs need no landing work.** Of 649: 58 (9%) contain the demo lines (`remote`, `cursor/onsite-demo-061a`,
   `cursor/soundness-assemble-5e54`, `cursor/vllm-remoterun-a7d3`, `cursor/soundness-pluscomp-f880`,
   `cursor/windows-placed-061a`: 3,365 to 4,647 commits not on `checked`, 651 to 712 of old `next`'s merges from
   ea3f9c5635), which the preflight always refuses (`_contained`): 1 mark and 1 verdict ever, each containing about 115
   other open PRs. 135 are unmarked inside a same-lane PR (57 inside a marked one); 2 are applied on `checked`.
6. **Flow (29 Oct to 4 Nov, PST days):** 130.6 opened, 97.3 landed, 57.0 closed unlanded a day; 649 open is 6.7 days of
   landing (memory-accounting 14.5, network-accounting 13.4, infra 10.7, circuits 8.2, proofs 7.1, ci 2.2). Closed
   unlanded since 29 Oct: 425 with 604 marks and 461 verdicts; 185 were folded (156 still carried by an open PR, 29 on
   `checked` or `main` through a carrier).
7. **Fan-in blocks nothing.** 44 non-demo roots have 2 or more marked carriers (#7035, lean, 19), but only 2 marked PRs
   wait on an unadmitted PR they contain: `_carried` lets a same-owner carrier go without an unmarked root.
8. **Noise** (#agent-coordination since Mon, read only): 511 conflict refusals (127 PRs), 271 naming a same-lane PR; 192
   hand ejections, about 40 "carried by" or "folded into"; 325 "bases landed, merge `main`" notices to 222 PRs, each a
   re-head that today costs a mark and a verdict.

Agreement: my `checked` to `main` p50 is 10 to 26 h per PR by landing day (CHECKPATH 10.0 h per landing, DESIGN 12 h);
mark to first verdict p50 0.19 h, p90 8.6 h (DESIGN 0.4 h, 9.3 h).

## What the in-flight work misses

- **ci/admission-drained** is the biggest volume lever too (finding 1), and it turns lane trains harmful: nothing should
  fold carried PRs into carriers on top of it. Measure re-heads per landed PR before and after it.
- **CI-427, merge records (#7490)**: pass time per member; nothing about re-marks, re-verdicts or re-heads.
- **Slice-clean (about 25 open per lane by Sat)**: 195 of the 649 are mechanical and code can close them; by hand the
  sweep recurs (57 closed unlanded a day).
- **Park an unreviewed bottom, re-read verdicts, invariants on low-risk merges**: verdict timing and safety, not volume.

## Levers, ranked

### 0. Land faster first (in flight; DESIGN's steps 1 to 4)

Not mine to propose, but the largest volume effect: about 230 to 290 fewer verdicts on superseded heads a week (finding 1;
115 to 145 if half the relation is confounding). Owner ci. Measure: `heads.py` weekly, extra heads per landed
PR (0.88 now; the 6 to 12 h cohort's 0.19 is the target).

### 1. A verdict and a mark are on a change, not a head (ci)

**Failure.** 92 verdicts and 127 marks a week re-spent on heads whose own change is unchanged, 123 PR-hours waiting for
them; a sync of `main` or a carry-in costs a full re-review, so lanes put off merging `main` and conflicts pile up
(finding 8). Shrinks as finding 1's effect arrives; makes syncing `main` free, which cuts conflict refusals.
**Mechanism.** When a PR's head moves, its `ready` and `reviewed` labels count on the new head when its own change (each
file's +/- lines against the merge base with `checked`, by patch-id: `reheads.py`'s `patch_ids`) is unchanged, leaving
out only files that brought-in PRs change, each with a land verdict and no hold on a head the new head contains, or on
`checked`. Lock files are own change. `unreviewed`/`reviews` and `ready` read the carried labels; `same_tree_mark`
becomes its special case. Preflight, Lean check and after-landing check run on the new head as now. Each carry is logged
on the PR with both heads, and a hold by anyone still wins. Smaller, tried: identical-tree carry, 2% of re-heads.
**Walkthrough** (week of 29 Oct): 790 labelled re-heads; 129 carry; 661 re-review as today; no lane does anything new.
**Price per week.** Human 0. Merge path: minus 92 verdicts, 127 marks, 123 PR-hours; plus about 6 refusals the machine
makes where a reviewer held first. Shared files: queue.py (hot, 68 PRs a week). Always-read words 0.
**Tests.** Code with the fix in its message; labels; no gate; one rule changed once. Risk: a carry on a wrong
"unchanged", caught by a difftest that an identical-tree head always carries and an edited file never does; a merge
interaction a reviewer would have held (finding 3), which reaches the merge refusal, preflight or Lean check.
**Owner and exit.** ci. Retires the per-head verdict rule; folds into incremental review (show a reviewer only lines no
land verdict covers) when that lands, which also keeps folded PRs' verdicts (finding 6's 185).
**Pilot.** proofs (96 carry-ins a week) for 8 h: re-verdicts, verdict wait, holds on carried heads.

### 2. Drive closes and refuses what nobody will land (ci for code; top for the demo lines)

**Failure.** 195 of 649 open PRs (30%) need no landing work, inflate every lane's count, and draw conflict refusals and
merge-`main` notices. **Mechanism.** Each pass, drive closes, naming the carrier, an unmarked PR whose head a same-owner
PR on `checked` contains, and a PR whose change is applied on `checked` (contain.py's blob test); the queue refuses a PR
containing old `next` merge commits not on `checked`, with the rebuild command in its reason. **Price per week.** Human
0, plus one decision: the 58 demo-line PRs move to their demo branch (default proceeds). Merge path 0. **Risk.** Closing a
PR its owner meant to keep: only unmarked, same owner, carrier landed; reopening is one click. **Owner and exit.** ci;
retires the mechanical part of the slice-clean sweeps.

### 3. Don't build lane trains, auto-withdrawal or carry-chain squashing

Finding 4: each trades a carried PR's latency for an admission slot, which pays only while admission is capped. Only if
ci/admission-drained slips past Mon: admit a waiting carrier with the marked same-owner PRs it contains as one member,
and remove it when admission leaves the tick.

### 4. Lane inventory in days, reported (lanes)

Drive's status shows each lane's open PRs over its landing rate; above about 7 days (memory-accounting 14.5,
network-accounting 13.4, infra 10.7) rework rises (finding 1). No cap: a cap delays new work, and lever 2 removes most of
the backlog.

## Next 48 h

1. ci: lever 1 behind a lane list in `queue.toml`, piloted on proofs; a new worker can write the patch-id carry and its
   difftest from `reheads.py` and `carry_strict.py`.
2. ci: lever 2's close rule; top decides the 58 demo-line PRs.
3. Lanes: act on art:e454e6b1's "close into" and "fold" rows today.
4. A worker reruns `heads.py`, `trains.py` and `carry_strict.py` when admission leaves the tick: extra heads per landed PR
   and re-verdicts are the volume numbers to watch.

## Method notes

Containment by ancestry (`rev-list checked..head`) and blobs (every changed file's head blob on the carrier), not `git
merge-tree` (minutes per head with the repo's merge drivers). Labels from the store's remote per PR prefix. A re-head is
consecutive labelled heads of one PR; its own change is measured against `checked` at its first label. Slack counts
from #agent-coordination since Mon, read only.
