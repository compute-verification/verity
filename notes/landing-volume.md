---
id: 20261105T1730Z-report-landing-volume
campaign: landing-speed
lane: top
kind: report
status: draft
repo: compute-verification/verity
origin: "Daniel, Thu 5 Nov 9:06 AM PST: use a bunch of agents to cut landing latency; learn from two weeks of failures. top's worker, angle VOLUME (less to land)."
---

# Landing volume: less to land (final, 19:00Z)

Branch `cursor/top-landing-volume-8a78`, scripts in `tools/research/studies/landing_volume/` (contain.py, carried_csv.py,
flow.py, labels.py, heads.py, reheads.py, carry_in.py, carry_net.py, trains.py, train_sim.py, rewait.py). Snapshot: 649
open PRs at 17:12Z, queue status at 17:20Z (`next` 9bdd6207, `checked` aa7843fd, `main` c6ba5690). Week = 29 Oct 17:00Z
to 5 Nov 17:00Z.

- Open carried/superseded PRs with carrier and evidence: **art:e454e6b155c6a0c31292cdb8f5fcef3579e5b61b6f4860bc9f10db9c03d0ca3b**
  (309 rows; categories and actions below).
- Data and scripts (gh pulls, store labels, containment, re-heads, flow): art:16168c795e76d4a48e65950374819dfc78b4042c2af556f16b4cf58340074513.
- Carried landings, simulator runs, re-verdict waits: art:0d6d97dba4a9a57797a5a94f3625a4e442e4930f7bfe3fa699832f211ccc4a3e.
- Simulator: DESIGN's `sim.py` (cursor/top-landing-design-8b2a) on its data art:fb0c4e715c3aea1f0fca101fe64e9f6f7c063f071d478dbbb427d057f2628e34,
  unmodified; `train_sim.py` only removes riders from its PR list and ties them to their carrier.

## Findings

1. **43% of land verdicts and 46% of ready marks go to heads that never land.** Week: 1,251 land verdicts: 446 on heads
   that landed, 271 on open PRs' current heads, 376 on heads later superseded, 158 on PRs closed unlanded. 1,572 marks,
   727 wasted the same way. Writers: ci 406, ci-review 296, top 253, lean 240. Wasted share by lane: proofs 45%, lean
   48%, network-accounting 63%, soundness 74%, ci 18%. Landed since 29 Oct (701): 1.95 marks and 1.76 verdicts each.
2. **Re-heads are the volume, and 29% of them change nothing of the PR's own.** Week: 806 re-heads after a labelled
   head; 389 (48%) brought in other PRs' heads, 342 (42%) real edits, 60 (7%) no own change (a sync of `main`, regenerated
   locks). Net of brought-in PRs and generated files, 233 (29%) left the PR's own change unchanged, yet each needed a new
   mark (214 a week) and a new verdict (165). Land verdicts never carry across heads (`unreviewed` reads one head's
   labels), and marks only to an identical tree (`same_tree_mark`), which a merge of `main` bringing anything never gives.
   The new head of a carry-able re-head waited p50 0.9 h, p90 12.5 h for its verdict (sum 491 PR-hours a week; marks: p50
   0.3 h, p90 2.7 h, 238 PR-hours). Over the week 317 of the 698 PRs that reached `main` re-headed after their first mark,
   and re-heads put 9,159 PR-hours (17% of first-mark-to-`main`) before the final head's mark: an upper bound on their
   cost, outside DESIGN's simulator (its clock starts at the landed head's mark).
3. **Re-heads grow with time in the queue**, so volume and latency feed each other. Landed PRs reaching `main` within 12 h
   of first mark: 1.19 heads marked, 17% re-headed; 12 to 48 h: 1.49, 32%; 48 to 96 h: 1.89, 47%; over 96 h: 2.25, 57%.
   Re-head cost at p50: 18 to 25 h (first mark to `main` p50 45 to 101 h by day, final-head mark to `main` 25 to 83 h).
4. **38% of what landed was carried by something else that landed, and each paid for its own slot.** `trains.py`: since
   1 Nov 08:08Z (admission behind preflight), 155 of 403 landings had their landed head inside a later or same-pass
   landing's head while not yet on `checked` at that PR's mark; 92 (23%) same lane, 64 of those 92 on `checked` within
   30 min of their carrier. Since 29 Oct: 360 of 721, 209 same lane. Snapshot: 18 of `next`'s 51 members (35%) inside
   another member, 16 same lane; 51 marked PRs inside another marked PR.
5. **But folding carried PRs into their carrier only pays while admission is capped** (DESIGN's simulator, p_break 2%;
   5% in the art). Lane trains (248 riders take no admission, no member slot, land with their carrier): current 51.1 h to
   48.5 h p50 (the bottleneck moves to reds); on DESIGN's steps 1 to 3 (revert on red, set landing, review within 2 h)
   20.2 h to 9.4 h p50, 14,225 to 9,980 PR-hours a week. Once admission is off the tick (step 4) they cost: p50 5.0 to
   5.3 h, p90 9.3 to 19.5 h, 3,369 to 5,781 PR-hours, because a carried PR, nearly always marked before its carrier, then
   waits for a carrier marked later. The safe variant (refuse a mark only when a marked same-owner PR already contains
   it) reaches 10 PRs. Carrier sets across lanes are worse at every step's p90.
6. **Inventory: 30% of open PRs need no landing work.** Of 649 open: 58 (9%) contain the demo lines (`remote`,
   `cursor/onsite-demo-061a`, `cursor/soundness-assemble-5e54`, `cursor/vllm-remoterun-a7d3`,
   `cursor/soundness-pluscomp-f880`, `cursor/windows-placed-061a`: 3,365 to 4,647 commits not on `checked`, 651 to 712 of
   old `next`'s merges from ea3f9c5635); the preflight refuses them (`_contained`), they drew 1 mark and 1 verdict ever,
   and each contains about 115 other open PRs. 135 are unmarked and inside a same-lane PR (57 inside a marked one, 78 an
   unmarked one); 2 are applied on `checked`. Flow (29 Oct to 4 Nov, PST days): 130.6 opened, 97.3 landed, 57.0 closed
   unlanded a day; 649 open is 6.7 days of landing (memory-accounting 14.5, network-accounting 13.4, infra 10.7, circuits
   8.2, proofs 7.1, ci 2.2).
7. **Noise from the same causes** (thread since Mon): 511 conflict refusals (127 PRs), 271 naming a same-lane PR; 192
   hand ejections, about 40 "carried by" or "folded into"; 325 "bases landed, merge `main`" notices to 222 PRs, each a
   re-head that today costs a mark and a verdict.
8. **Closed unlanded since 29 Oct PST: 425**, holding 604 marks and 461 verdicts. 156 are still carried by an open PR
   (278 marks, 202 verdicts), 29 reached `main` or `checked` through a carrier (52 marks, 51 verdicts): folded work whose
   verdict the carrier's review redid. The other 240 (274 marks, 208 verdicts) are abandoned or rebuilt.
9. **Fan-in doesn't block today.** 44 non-demo roots have 2 or more marked carriers (#7035, lean, 19; #7097, 13;
   memory-accounting's prf195 roots #7132, #7085, #7110, #7073, #7074, #7079, #7080, #7096, #7103 with 4 each), but only
   2 marked PRs wait on an unadmitted PR they contain: `_carried` lets a same-owner carrier go without an unmarked root.
   Root-first priority would buy nothing now.

Agreement with the others: my `checked` to `main` p50 is 10 to 26 h per PR by landing day, beside CHECKPATH's 10.0 h per
landing and DESIGN's 12 h; my mark to first verdict p50 0.19 h, p90 8.6 h, beside DESIGN's 0.4 h and 9.3 h.

## What the in-flight work misses

- **CI-427 and merge records (#7490)** cut pass time per member; they leave the 214 re-marks, 165 re-verdicts and 491
  PR-hours a week of carry-able re-heads untouched, and the 35% of members carried by another member.
- **ci/admission-drained (under 50 waiting by Fri 6 PM)**: the right lever (simulator step 4), and it makes lane trains
  harmful, so nothing should fold carried PRs into carriers on top of it. Its "waiting" count includes 39 marked PRs a
  marked same-lane PR carries.
- **Slice-clean (about 25 open per lane by Sat)**: 195 of the 649 (the 58 demo-line PRs, 135 carried unmarked, 2
  applied) are mechanical, and code can close them; by hand they'll be redone every week (57 closed unlanded a day).
- **Park an unreviewed bottom, re-read verdicts before landing, invariants on low-risk merges**: verdict timing and
  safety, not verdict volume.

## Levers, ranked

### 1. A verdict and a mark are on a change, not a head (ci)

**Failure.** 165 verdicts and 214 marks a week re-spent on heads whose own change is unchanged; 491 PR-hours a week
waiting for them (p90 12.5 h); a carry-in or a sync of `main` costs a full re-review, so lanes are pushed to avoid
merging `main`, and conflicts pile up (finding 7).
**Mechanism.** When a PR's head moves, its `ready` and `reviewed` labels count on the new head if the PR's own change
(each file's +/- lines against its merge base with `checked`, by patch-id: `reheads.py`'s `patch_ids`) is unchanged,
leaving out files every brought-in PR changes, when each of them has its own land verdict or is on `checked`, and files
drive regenerates (lock records, pins). `unreviewed`/`reviews` and `ready` read the carried labels; `same_tree_mark`
becomes its special case. The preflight still runs on the new head; the after-landing check is unchanged. Each carry is
logged on the PR with both heads, and a reviewer's hold still wins. Smaller tried: identical-tree carry exists for marks
and would cover 34 of 1,466 re-heads (2%) since 22 Oct.
**Walkthrough** (week of 29 Oct): 806 re-heads after a label: 233 carry, 573 re-review as today; lanes act no more.
**Price per week.** Human 0. Merge path: minus 165 verdicts, 214 marks and 491 PR-hours. Shared files: queue.py (hot).
Always-read words 0.
**Tests.** Code with the fix in its message; labels; no new gate; a rare change. The risk: a merge that changes behaviour
without touching own files, which no verdict covers today either (`research queue review` shows the increment) and which
the preflight and the after-landing check judge as now; a wrong "unchanged" is caught by a difftest that a same-tree head
always carries and an edited file never does.
**Owner and exit.** ci; it retires the per-head verdict rule, and folds into incremental review (show a reviewer only lines
no land verdict covers) when that lands. **Pilot.** proofs (96 carry-ins a week) for 8 h; measure re-verdicts, verdict
wait and holds. **Lands in** one queue.py PR with tests. **Extension:** key verdicts by line so a PR folded into another
keeps its verdict (finding 8).

### 2. Drive closes and refuses inventory nobody will land (ci for code; top for the demo-line decision)

**Failure.** 195 of 649 open PRs (30%) need no landing work, swell every lane's count, and draw conflict refusals and
merge-`main` notices. **Mechanism.** Each pass, drive closes, with a comment naming the carrier, an unmarked PR whose head
a same-owner PR on `checked` contains, and any PR whose change is applied on `checked` (contain.py's blob test); the queue
marks a PR containing `next` merge commits not on `checked` as refused, with the rebuild command in its reason.
**Price per week.** Human 0 (a decision for the 58 demo-line PRs: retarget them to their demo branch, default
proceeds). Merge path 0. **Risk.** Closing a PR its owner meant to keep: only unmarked, same owner, carrier landed;
reopening is one click. **Owner and exit.** ci; retires the slice-clean sweeps' mechanical part. **Lands in** one drive PR.

### 3. Lane trains: don't build them (unless admission stays capped)

Finding 5. Build "admit a waiting carrier with the marked same-owner PRs it contains as one member" only if
ci/admission-drained slips past Mon; then it buys p50 20 to 9 h on DESIGN's steps 1 to 3, and must go when admission
leaves the tick. Auto-withdrawing carried PRs' marks and squashing carry chains trade a contained PR's latency for an
admission slot.

### 4. Lane inventory in days, reported (lanes)

Drive's status gives each lane's open PRs over its landing rate. Above about 7 days (memory-accounting 14.5,
network-accounting 13.4, infra 10.7), re-heads rise from 17% to 57% of PRs. No cap: a cap delays new work, while the
reported number plus lever 2 removes most of the backlog.

## Next 48 h

1. ci: lever 1 in queue.py behind a lane list in `queue.toml`, pilot on proofs; a new worker can write the patch-id
   carry with its difftest from `reheads.py` and `carry_net.py`.
2. ci: lever 2's close rule in drive; top decides the 58 demo-line PRs (default: retarget to their demo branch).
3. Lanes: close or leave the art:e454e6b1 rows marked "nothing" or "close into"; drive's rule then keeps it so.
4. Re-run `trains.py` and `rewait.py` after admission leaves the tick to confirm carried members and re-verdict waits.

## Method notes

Containment by ancestry (`rev-list checked..head`) and blobs (every changed file's head blob on the carrier); no
`git merge-tree` (minutes per head with the repo's merge drivers). Labels from the store's remote per PR prefix. A
re-head is consecutive labelled heads of one PR; its own change is measured against `checked` at its first label.
Carry-able: own change unchanged net of brought-in PRs' files and generated files (lean-audit.json, pins, .sha256,
uv.lock, lake-manifest.json); it doesn't check that each brought-in PR had its own verdict, so 165 is an upper bound for
lever 1's first version. Slack counts from #agent-coordination since Mon, read only.
