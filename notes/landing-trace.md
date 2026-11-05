---
id: top/20261105T1726Z-report-landing-trace
campaign: landing-speed
lane: top
kind: report
status: draft
repo: compute-verification/verity
origin: top worker TRACE (landing-speed, Daniel 9:06 AM PST Thu 5 Nov), branch cursor/top-landing-trace-73ba
---

# Landing trace: where the hours went (draft)

CHECKPOINT 18:20Z. Trace v2: **art:afbc1cfbd6e347dfb63b5e20b3a9f5935c4e2aaa11bb36483fdfa8dc3601e61c**
(`landing-trace/v2`, 6.3 MB JSON; supersedes v1 art:436e9968048fe4d6c4b25d39c6a7f0c2b5a0bc74c6a32e26744186e0105c09de).
The scripts that build it from the control pod, the label store and GitHub, and replay levers over it:
art:28b0194f7879c70e84e3da631a8359f7e35a265167ace01561cd38ce6366a6ed. Final report sent to top.

## The trace (for DESIGN's replay)

One JSON object: `prs` (3,229: the 2,879 PRs whose head reached `main` with GitHub `mergedAt` >= 22 Oct 07:00Z, plus
the 350 open PRs whose current head carries `ready=true`), `events` (18,189: one per PR per stage, each with `ts` and
`head`: opened, marked/unmarked with `by`, admitted, verdict with `by`, hold, refused/ejected with drive's reason,
checked with the `checked` SHA and lane, main), `main_moves` (153, with PRs, the gap before, and how: drive's own
after-landing run, ci's hand baseline, or direct landing before 29 Oct), `ticks` (1,337 drive passes, 20 Oct to 16:48Z
today: start/end, minutes, `next` size, PRs waiting for admission, what held the bottom of the prefix, the after-landing
state, the landings), `pareto_all`, `pareto_last7d`, `incidents`. New in v2: per PR `cause_hours` (each pass's interval
charged to the one cause holding the PR), `change_class` (lean / lowrisk / nonlean, from its merge onto `checked`),
`reheads` (each change of marked head with its kind), `red_tells_on_checked` (drive's 73 red tells, classified) and
`daily_flow_pst`. `stage_definitions` says where each number comes from. Drive events are dated to their pass's end
(passes took 41 to 67 min median since 2 Nov); labels and `mergedAt` are exact. Admission is logged only since 1 Nov
15:43Z; before, it is folded into the time to `checked`.

Sources: drive's whole log from the control pod (`/data/drive/drive.log`), drive's state
(`/root/.research/queue/verity.state.json`), every PR label in the store (16,420 assertions), GitHub REST (closed PRs
since 14 Oct, open now), and reachability of each head from origin/main, origin/checked, origin/next.

## Findings

1. **The pipeline is in a backlog regime.** About 350 marked PRs are in the system (256 open, 51 on `next`, 43 on
   `checked`), and about 95 reach `main` a day, so by Little's law the mean wait is about 3.7 days whatever the order.
   Before CI-239 (main only fully checked commits, effective 28 Oct 00:16Z) main moved 133 times in 6 days and
   ready-to-main was median 4.1 h, p90 10.9 h (2,064 PRs, 344 a day). Since then main moved 20 times in 8 days (gaps
   1.6 to 37.2 h) and ready-to-main is median 45.8 h, p90 117 h (805 PRs); the last 7 days median 60.5 h, p90 121 h
   (678 PRs). New PRs marked fell from 400 to 690 a day (24 to 27 Oct) to 41 to 129 a day (29 Oct to 4 Nov): demand
   is suppressed, and a fast pipeline must be sized for the old rate, about 500 a day.
2. **Where the last 7 days' 53,812 PR-hours went** (ticks since 29 Oct 17:30Z; 683 landed + 350 waiting): on `next`
   behind a running Lean check 22.8%; admission 29.2% (17.0% queued with no refusal logged, 12.2% in the list waiting
   for a preflight turn); on `next` while a pass landed others (pass cadence) 12.0%; behind a bottom member without a
   land verdict 7.7%; own verdict missing 6.4%; on `checked`: after-landing run going 7.1%, going with a red named 6.4%,
   ci's hand run that drive can't see 2.1%, red with no run 1.4%; `next` rebuilt 2.1%; drive stopped 1.7%. By stage
   (landed in 7 days, mean h): admission 40.2 (n 295), review 1.3, to `checked` 35.0, after-landing 13.3.
3. **Lean checks hold mostly non-Lean PRs.** Of the Lean-check waits of PRs landed in 7 days (8,247 PR-h), 69% fell on
   PRs whose change touches no Lean input (non-Lean 4,662, low-risk 1,056). drive.toml has `nonlean_first = false`.
   Six of the ten largest blocker incidents are Lean checks holding the prefix (#5334, #7209, #6771, #5651, #6037,
   #5397: 629 to 765 PR-h each). Lean checks took 217 to 6,433 s.
4. **Reds on `checked` are never reverted, and most are machines.** Drive told 73 reds on `checked` (30 Oct to 3 Nov;
   none since, because `ci-no-postland` stops drive's after-landing runs since 3 Nov 19:19Z). It reverted none, though
   it named 9 culprits (#5101 #5268 #5388 #5439 #5745 #6248 #6558 #6584 #6719): "Reverted nothing; find the culprit by
   hand" 29 times, "a revert lies in …, so bisecting could blame the wrong PR" on most of the rest. 50 of the 73 were
   steps whose log has no error line (a machine or the step failed); 13 were lint, ratchet or prose invariants
   (repository, ratchet-gate, verity-vllm's P8 lint); 4 read outside declared inputs; 6 were test failures. The longest
   gap: `step pytest` red with no error line on five tips in a row, 2 Nov 05:46Z to 3 Nov 13:01Z, until it passed at
   4d68f259 (r20261103-141641-d111): main 37.2 h unmoved, 2,642 PR-h. The four longest gaps cost 7,998 PR-h.
5. **Drive's passes are the clock, and they slowed 4 to 10x.** Passes took 4 to 16 min median 20 to 27 Oct and 37 to
   61 min 2 to 5 Nov (15 to 44 passes a day; 3 Nov 15:41Z: 5,542 s, `Queue._preflighted` 2,765 s over 332 calls; 5 Nov
   13:57Z: 2,176 s, `Queue._merge` 1,314 s over 88 merges). Admission ran 55 to 133 a day since 1 Nov against 250 to
   470 waiting; 181 open PRs were never admitted at their final head.
6. **Re-heads drive own-verdict waits.** Drive told owners "merge main into #N" 896 times (443 PRs). Of the 593 re-heads
   of PRs landed in 7 days, 183 added no commit of their own (merges only), costing 3,624 PR-h between marks; 304 of 683
   PRs were marked at more than one head (8,979 PR-h from first to final mark).
7. **Hand steering and stops.** Drive was stopped 37.4 h since 22 Oct. `/data/drive/args` holds a `--first` list of
   about 400 PRs; 286 timestamped backups of it from 19 Oct to 5 Nov; a hand `priority` file edited about 40 times since
   3 Nov. Since 3 Nov main moved 7 times, each on ci's hand baseline.

## The current regime (ticks since 2 Nov 00:00Z: 3.7 days, 23,631 PR-h, about 44,700 PR-h a week)

Waiting for a preflight turn 23.7% (and queued 4.4%, budget 1.3%: admission 29.4%); behind a running Lean check 16.7%;
pass cadence 15.3% and `next` rebuilt 3.0%; own verdict missing 12.1%; after-landing run going 7.9%, ci's hand run
drive can't see 4.8%, red with no run 3.1%, red named 1.8% (after-landing 17.6%); bottom member without a verdict 2.8%;
drive stopped 2.3%. Blocker incidents since 2 Nov are all Lean checks but two: #7209 765 PR-h, #6998 363, #7062 360,
#7293 346, #4999 300, #6052 (no verdict) 294, #6429 288, #7490 (no verdict) 201.

- **Admission is bounded by drive's pass.** Preflights run inside the pass on the control pod (4 at once,
  `PREFLIGHT_PARALLEL`), starting only within a per-pass wall-time budget: `PREFLIGHT_BUDGET_S` = 120 s in code, raised
  by hand on the pod to 600, 1,200, then 900 s since 2 Nov 10:42Z. Drive told 441 PRs "this sync's preflights took
  their 900 s, so a later sync runs it", and ci's `--first` notes move PRs that "missed two syncs of preflight budget".
  A bigger budget lengthens the pass that every landing waits on (3 Nov 15:41Z: 5,542 s pass, `Queue._preflighted`
  2,765 s over 332 calls).
- **The bypass of 1 Nov 12:57Z (a reviewed non-Lean member lands below a running Lean check) cut non-Lean Lean-check
  waits from 5,771 PR-h before it to 1,843 in the 3.7 days since, but 47% of Lean-check waits still fall on PRs with no
  Lean change.** The other 53% are Lean PRs waiting for checks below them. A Lean check of drive's took median 127 min,
  p90 182 min from its run's start until drive saw it pass (42 passing checks since 2 Nov).
- **Review capacity fell.** ci-review gave 207 to 566 verdicts a day 27 to 30 Oct and 5 on 31 Oct; since then ci,
  lean and top give 100 to 250 a day.

## Ready-to-main per PST day of arrival on main (median / p90 h from the final mark; n)

Thu 22 Oct 1.4/2.0 (83), Fri 5.6/8.3 (153), Sat 5.7/8.1 (472), Sun 6.7/11.6 (399), Mon 26 4.6/17.2 (503), Tue 27
2.0/4.0 (454), Wed 28 13.9/17.9 (106), Thu 29 12.0/35.3 (28), Fri 30 39.8/75.0 (28), Sat 31 77.0/87.3 (64), Sun 1 Nov
82.8/112.9 (286), Mon 2 none, Tue 3 49.6/151.5 (188), Wed 4 30.3/187.6 (77), Thu 5 16.7/36.9 (28). By cohort of final
mark: since 28 Oct about 0% reach main within 6 h and at most 34% within 24 h; before, 42 to 100% within 6 h.

## Top 10 incidents by PR-hours lost

| PR-h | what | span (UTC) |
|---|---|---|
| 3,695 | main unmoved 27.3 h over 238 PRs: 8 named reds (#5268 lean-audit, #6248 inputs, #5745 ratchet, #5101 pous, #6558 inputs, #6584 verity-check, repository lint, P8 lint), none reverted | 31 Oct 11:35 to 1 Nov 14:53 |
| 3,483 | #6352 at `next`'s bottom with no land verdict | 29 Oct 21:46 to 30 Oct 10:58 |
| 2,642 | main unmoved 37.2 h over 107 PRs: `step pytest` red with no error line on 5 tips | 2 Nov 02:36 to 3 Nov 15:50 |
| 2,054 | #4888 CI-239 brake (hold on `next`'s bottom) | 29 Oct 05:10 to 15:38 |
| 1,969 | #5833 CI-239 brake | 28 Oct 02:37 to 14:21 |
| 845 | main unmoved 16.2 h over 67 PRs: ci's hand run, invisible to drive, 12.5 h | 4 Nov 13:36 to 5 Nov 05:47 |
| 816 | main unmoved 17.4 h over 64 PRs: a red named 13.3 h (repository #5439, a mass no-error-line failure 30 Oct 23:29Z) | 30 Oct 18:08 to 31 Oct 11:35 |
| 804 | #6042 no land verdict | 29 Oct 13:00 to 17:04 |
| 765 | #7209 Lean check holds the prefix | 5 Nov 03:19 to 08:30 |
| 765 | #5334 Lean check | 28 Oct 00:23 to 29 Oct 21:35 |

Then #6771 674, #5651 661, #6037 644, #5397 629 (Lean checks).

## Ranked levers

Two estimates per lever. **Rate**: the PR-hours a week its causes take in the current regime (since 2 Nov, scaled from
3.7 days to 7), times the share it removes. **Replay**: the 678 PRs landed in 7 days plus the 350 waiting, replayed with
those cause hours removed (`whatif.py`; baseline median 60.5 h, p90 121.0 h). Both are upper bounds: a drained stage
pushes work into the next one unless that one gains capacity, so levers 1 to 4 pay together.

1. **Preflights leave drive's pass and run ahead on a pool** (ci builds it; infra the pool; or a new worker). Each
   marked PR's preflight (`Preflight.onto`) runs as soon as its mark appears, onto `next`'s top, on `research run
   --on auto` machines; its verdict is stored per (head, base tree) as today, and `Queue._standing` already carries a
   pass across a rebuilt base that changed nothing the PR touches. Drive's sync only reads verdicts and admits every PR
   with a pass. Rate: admission 29.4%, about 13,000 PR-h/wk; replay median 39.3 h, p90 104.3 h. It also shortens the
   pass (`Queue._preflighted` was half of 3 Nov's 5,542 s pass). Cost: pool CPU for about 150 one-to-five-minute
   preflights a day. Risk: a verdict on a stale base admits a PR that conflicts with what landed since; caught by the
   sync's own merge and `_standing`'s rule, then the after-landing run. Retires `PREFLIGHT_BUDGET_S`, the pod's 900 s
   override and ci's `--first` entries for PRs that missed the budget. Days. Test of ci/admission-drained (<50
   waiting by Fri 6 PM): about 200 to 250 admissions in a day against a best day of 133 (4 Nov): not reachable by
   raising the budget, which lengthens every pass.
2. **Every `checked` tip gets an automatic after-landing run, reds stay off `checked`, culprits are reverted at once**
   (ci; machine faults infra). Delete `/data/drive/ci-no-postland` and turn `executor` on (#7419's flip, waiting on
   infra); run the repository suite, ratchet-gate and the input guard on every member's preflight merge (ci's low-risk
   lever, all lanes: 17 of 73 reds); treat a step that failed with no failing test and no error line as a machine
   fault, run it once more on another machine, ring infra (50 of 73; `_machine_fault` does this for Lean checks);
   revert a named culprit at once, as AGENTS.md says and drive hasn't done since 28 Oct. Rate: after-landing 17.6%,
   about 7,400 PR-h/wk above a 45-minute run; replay median 52.4 h, p90 108.1 h; the four longest gaps cost 7,998 PR-h.
   Risk: a misnamed culprit is reverted (its owner re-marks; the revert is checked like any landing); a real failure
   that prints nothing passes its rerun (one rerun, on another machine, both logs kept). Retires ci's hand baselines.
   A day.
3. **Lean waits only for Lean** (ci, lean; Lean speed: infra/full-check-45-min, LEAN-190). Non-Lean PRs: the 1 Nov
   bypass needs a land verdict and "touches nothing a Lean check reads"; key a Lean check's verdict by the hash of the
   files it reads, so any member that leaves that hash alone lands below it (`nonlean_first = true` for what a sync
   adds; the rekey already moves a check across such trees). Lean PRs: a Lean check takes median 127 min to be seen
   passing; reuse the previous check's build for unchanged modules, and land on the check's result when it arrives,
   not at the next pass. Rate: Lean-check waits 16.7%, about 7,500 PR-h/wk (3,500 on non-Lean PRs, 4,000 on Lean
   PRs); replay median 48.7 h, p90 113.5 h for the non-Lean part. Risk: a change misjudged as not read by a Lean check;
   caught by the after-landing lean-audit. Config plus a hash key: days.
4. **Passes under 20 min, landing on events** (ci, in flight #7490, CI-427). Rate: pass cadence and rebuilds 18.3%,
   about 5,400 PR-h/wk at a 20-minute pass; replay 4,902 PR-h/wk. Measured, in flight, enough once preflights leave the
   pass.
5. **A mark and a verdict carry across a merge-only re-head** (ci). When the new head's tree is the clean
   `git merge-tree` of the marked head and its new other parents, the ready and land labels at the old head apply;
   drive stops telling owners to merge main (896 tells, 443 PRs). Rate: own verdict missing 12.1% (about 5,400
   PR-h/wk), of which merge-only re-heads are about a third; plus 3,624 PR-h/wk between marks, from first mark. Price:
   a verdict covers a merge its reviewer didn't read, as every merge on `next` already does; caught by preflight and
   the after-landing run. Changes the per-head verdict ruling (TOP-264). A day.
6. **In flight and confirmed:** park the unreviewed bottom (ci): 2.8% now (about 1,250 PR-h/wk), the largest incident
   in Oct (#6352, 3,483 PR-h). **Drive never stopped, no hand ordering** (infra, ci): a drive build runs on shadow
   state before it replaces the live one; order by oldest final mark; retire `--first` and `priority`: 2.3%, about
   1,130 PR-h/wk, plus ci's hand hours.
7. **Not recommended: `main` = `checked`'s tip, CI-239's fully checked commit kept as a `verified` ref** (Daniel's
   rule). It removes 9,192 PR-h/wk of the metric (median 51.7, p90 107.3) by relabeling the after-landing wait, not by
   shortening it, and agents would branch from unverified commits. Lever 2 removes most of it without that price.

Together (replay): levers 1 to 4 with ci's in-flight levers give median 8.7 h, p90 43.4 h; with lever 5's verdict term
and no stops, median 5.1 h, p90 30.6 h. The tail is Lean PRs.

## Tests of the in-flight goals

- **p90 < 2 h by Sat 6 PM** (ci/landing-two-hours): not for Lean PRs (29% of landings) while a Lean check takes median
  127 min to be seen passing and the after-landing run 82 min cold. For non-Lean PRs it needs admission, verdict and a
  pass each under about 15 min and an automatic 45-minute after-landing run. And the backlog must drain first: at 100
  landings a day, p90 2 h allows about 8 PRs in the system (350 now); at the pre-CI-239 rate of 500 a day, about 40.
- **<50 waiting by Fri 6 PM** (ci/admission-drained): needs lever 1; a larger budget trades admission for pass length.
- **main moves on the fine-grained check by Fri noon** (ci/fine-grained-check): #7419 merged; main still moves only on
  ci's hand baselines while `ci-no-postland` exists, so the flip must delete it.
- **full check 45 min by Sat** (infra/full-check-45-min): needed, not sufficient: the four longest gaps (16 to 37 h)
  were reds and a machine fault, not check length.
