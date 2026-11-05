---
id: 20261105T1717Z-report-landing-design
campaign: landing-speed
lane: top
kind: report
status: draft
repo: compute-verification/verity
origin: top worker (DESIGN angle), branch cursor/top-landing-design-8b2a
---

# Landing design: the fastest robust landing, and the path to it

CHECKPOINT 18:00Z: trace, simulator, calibration, lever table and cumulative path done; data art:fb0c4e715c3aea1f0fca101fe64e9f6f7c063f071d478dbbb427d057f2628e34 (trace,
queue snapshot, `next` membership, simulator, calibration grid, lever tables, `cumulative.json`). Simulator code on branch
`cursor/top-landing-design-8b2a` (`tools/research/landing_sim/`: trace.py, stats.py, sim.py, calib.py, path.py; not for main).

## 1. What the data says (14 days, Oct 22 – Nov 5)

Trace: every `Queue: #N` merge on `checked`'s first-parent history (3114 merges, 3089 PRs), the time `checked` and
`main` first contained it (GitHub's push activity API, exact), its landed head's `ready`/`reviewed` labels, its files;
`next`'s 535 tips since Oct 29 (fetched; membership clean from Nov 1 15:30, after the history rewrite); the queue at
17:30Z (`research queue status --json`) with the 344 waiting heads' labels.

| | Oct 21–28 (review landing onto `main`, check after) | Oct 29–Nov 5 (`checked`; `main` only on a full check) |
|---|---|---|
| PRs reaching `main` / day | 299 | 85 |
| ready→main p50 / p90 (landed only; censored) | 4.3 h / 11.5 h | 54 h / 121 h |
| ready→reviewed p50 / p90 | 0.5 h / 2.5 h | 0.4 h / 9.3 h |
| reviewed→`checked` p50 / p90 | n/a | 42 h / 107 h |
| `checked`→`main` p50 / p90 | n/a | 12 h / 25 h |

1. **The queue is saturated, so latency is backlog over throughput.** 344 marked PRs wait now (median age 22 h, p90
   152 h, 71 over 3 days) plus 43 on `checked`; ~84–95 marks/day arrive and ~90/day land. Little's law: a new mark waits
   ≈ 390 / 90 ≈ 4 days. Every lever is judged by whether it lifts throughput above arrivals; once it does, latency
   collapses to the service time. PRs created ran ~133/day (#6517 on Oct 29 to #7516 now), and the team marked ~300/day
   when landing was fast, so demand will rise as landing speeds up: design for 3x.
2. **Reviewed PRs wait on `next`, behind blocked prefixes.** For PRs landing on `checked` since Nov 2 (clean `next`
   data; biased short): ready→first on `next` p50 2.3 h / p90 17 h; first on `next`→`checked` p50 6.7 h / p90 24 h,
   though 160 of 190 had their verdict before they reached `next`. `next` held p50 48, p90 92 members.
3. **No red was reverted in 14 days; every red waited for a fix-forward.** `_red_clock` holds a revert whenever a
   fix is marked (`_fix`), and the after-landing run waits for it (`FIX_WAIT_S`); only circuit-check reverts at once.
   ~20 of 693 landings since Oct 29 are fix-forwards of reds (break rate ≈ 3%, range 2–5%), most of them cheap
   invariants (undeclared suite inputs, silent-except and DIRECT ratchets, dead modules). `main` moved 17 times in 7
   days, gaps up to 37 h.
4. **The review gate almost never holds and is one lane.** 0 holds in 158 sampled landed-head verdicts, 2 of 241 among
   waiting heads; `ci-review` gave 77% of verdicts. Its p90 went from 2.5 h to 9.3 h in the `checked` era.
5. **Low-risk lane is idle.** 30% of landed PRs touch nothing `lowrisk_excludes` names, but 7 of 679 landed without a
   verdict: PRs wait for admission long enough to be reviewed anyway.
6. **Hot files.** 46% of landed PRs touch a shared hot file (20% a `lean-audit.json`); merge drivers already cover
   locks, Lake lists, allowlists and the fixture map. What still conflicts: source (`queue.py` 68 PRs/week, `drive.py`
   44, their tests), `TRUSTED.sha256`, `GLOSSARY.md`, READMEs. The landing code itself is the hottest code file: ~110
   PRs a week change `queue.py`/`drive.py`, each a change of the rules mid-flight.

## 2. Simulator (assumptions stated)

`sim.py` replays the real ready marks (Oct 29 on, with the 247-PR backlog marked earlier) with each PR's real review
delay (an unreviewed one draws from the landed delays past the time it already waited) and Lean flag. A PR breaks the
check with probability `p_break` (fixed per PR number, same in every policy). Not modeled: conflicts and re-marks
(the clock starts at the landed head's mark, so rework before it is outside every figure), outages of drive itself,
failed Lean checks, holds. So the current model's tail is optimistic and the gains are understated.

**current**: a pass of 40 + 0.3·|next| min; ≤ 4 admissions a pass; each Lean member gets an 82-min check (≤ 10) that
pins the prefix below it; verdict-first reorder above the pinned prefix; land the longest prefix with verdicts (Lean
members with a passed check), low-risk members on their mark; one after-landing run at a time (150 min); green moves
`main`; a red holds `main` until the breaker's fix lands 6 h later; 10% of runs flake. Fitted by grid
(`calib.py`) to the observed censored p50 (sim 39 h vs 38 h), `main` moves/day (2.2 vs 2.4) and `checked`→`main`
p50 (9 h vs 12 h); p90 underestimated (55 h vs 94 h).

**tree queue**: each PR's own cached check at its mark (20 min; Lean 82 or 45) beside its review; every 5 min the
landing loop stacks a candidate tree = previous ⊕ all PRs ready to land (≤ 4 in flight); its incremental check takes
15 min (45 when it holds two Lean changes); red names its breaker by the failed task's inputs (90%, else a bisection of
log2 n checks), ejects it with nothing landed, rechecks; `main` moves to each green candidate in order; an ejected
breaker returns fixed 6 h later. Cores: own check 8 (Lean 32), candidate 64.

## 3. Results (cohort marked Oct 29 – Nov 4 12:00, 546 PRs, every one run to `main`)

Cumulative, in the order I'd ship them (PR-hours of ready→main per week; p_break 2% / 5%):

| step | p50 / p90 (2%) | PR-h/wk (2%) | removed | p50 / p90 (5%) | removed (5%) |
|---|---|---|---|---|---|
| current (calibrated) | 51 h / 87 h | 30,420 | | 95 h / 128 h | |
| 1 revert at once on red | 48 / 76 | 28,752 | 1,668 | 53 / 82 | 19,938 |
| 2 no prefix block (land sets) | 31 / 51 | 16,401 | 12,351 | 35 / 61 | 12,676 |
| 3 review ≤ 2 h (dispatch, verdict carry) | 20 / 48 | 14,225 | 2,176 | 26 / 59 | 2,413 |
| 4 admission off the tick, 10-min pass | 5.0 / 9.3 | 3,369 | 10,856 | 5.9 / 14 | 12,077 |
| 5 executor run 60 min | 2.5 / 3.9 | 1,628 | 1,741 | 2.8 / 5.5 | 2,412 |
| 6 flakes deleted | 2.2 / 3.8 | 1,549 | 79 | 2.6 / 4.6 | 102 |
| 7 Lean chain 45 min | 2.1 / 3.8 | 1,440 | 109 | 2.2 / 4.4 | 117 |
| 8 tree queue (gate before `main`, no `checked`) | 0.7 / 2.3 | 689 | 751 | 0.9 / 2.3 | 863 |

Single levers alone on the current model (2%): no prefix block 51→32 h p50; review ≤ 2 h 51→23 h; admission off the
tick 51→25 h; `land_into = "main"` (check after) 51→43 h; executor run 60 min 51→49 h; Lean 45 min ≈ 0.

Load: current at 2x arrivals: p50 74 h, backlog 343 and growing. Steps 1–7 at 3x: p50 2.1 h, p90 3.8 h. Tree queue
at 3x: p50 1.2 h, p90 2.4 h, `main` moving 136 times a day. Tree queue compute: mean ~100 cores, p99 ~370 at 1x;
mean 250, p99 ~670 at 3x: 2 nodes at today's load, 4 at 3x (nodes 192–256 cores; per-task cores are assumptions,
generous against graph/v2's `tasks.toml`), plus the nightly uncached run.

Reading: most of the gain is reachable inside today's drive (steps 1–5, all small changes). The tree queue adds
1.4 h p50 / 1.5 h p90, 3x the `main` moves, and robustness: a red never freezes `main`, and nothing lands unchecked.

## 4. The design: a tree queue

Land trees, not prefixes. A ready mark enqueues the PR's own graph (main ⊕ head) on the executor and dispatches a
reviewer. A landing loop (seconds, separate from slow work) keeps a stack of candidate trees, each the one below ⊕
every PR with a passing own check and a verdict, built by `git merge-tree` with the merge drivers. Each candidate's
check is the graph/v2 task set of its tree, nearly all cache hits (a task's key changes only when two members touch its
inputs). `main` fast-forwards to the highest candidate whose every gating task passed: it holds only checked trees, so
`checked` and its after-landing run retire, and `next` becomes the candidate stack. A red names its culprit by the failed
task's inputs (bisect that one task when several members touch them), ejects it before anything lands, and rechecks
incrementally; an unreviewed, failing or Lean-pending PR holds only itself. Lean gating is the lean-module/lean-lock
tasks of the candidate tree: the same soundness gate as now, with no prefix pinned. Locks and pins regenerate on the
candidate (`regenerate_locks`, `pins.py`); verdicts carry across heads with an unchanged own patch (`git patch-id`);
the nightly uncached run on `main` is the backstop for cache keys, and its red auto-reverts.

## 5. Shortest path from today's code

1. Revert at once (`_red_clock`: every gated red as `CIRCUIT_RED`; `_fix` no longer holds it; `FIX_WAIT_S` 0). Hours.
2. Land sets: with `executor = "on"`, Lean checks are per-package task keys, so drop prefix pinning (`lean_hold`,
   `Queue.hold`) and let `verdict_first` lift freely; ci's park of an unreviewed bottom member is the first half.
3. Reviewer dispatch per mark with a 2 h SLA, many reviewers not one lane; `queue.unreviewed` carries a verdict to a
   head whose `git patch-id --stable` of its own diff is unchanged.
4. Admission off the tick: the mark-time preflight's verdict is the admission (`Queue._standing` already carries
   passes across bases); a preflight that must rerun goes to the fleet, never the control pod's 4 vCPUs inside a sync.
   In flight partly: #7366, #7314, CI-427, ci/admission-drained.
5. Executor on (CI-340, in flight). Then point it at `next`'s candidates instead of `checked`'s tip and let `_promote`
   move `main` and `checked` together to a green candidate: the tree queue, `checked` retired.
6. A merge driver for `TRUSTED.sha256` (regenerate with `pins.py`); conflict on source goes to an auto-rebase agent.
7. Freeze the landing rules while 1–5 land: route `queue.py`/`drive.py` changes through one owner and a pinned deploy
   (CI-226).

## 6. Rules this changes (with price)

- "A red step's culprit is reverted on `checked`" stays; its fix window and fix-hold go. Price: a culprit's owner
  re-lands (one more trip through the queue) instead of everyone waiting.
- "`main` holds only fully checked commits" stays, enforced before `main` moves instead of after; `checked` and the
  separate after-landing run retire. Price: the nightly uncached run becomes the only cold check of `main`; a cache-key
  bug can put a red on `main` for up to a day, caught by that run and auto-reverted.
- "A push makes a new head, which needs a new mark and a new review": the review carries when the PR's own patch is
  unchanged (a merge of `main`). Price: a merge-conflict resolution is still a new patch and needs a verdict.
- The single `ci-review` lane: replaced by per-mark reviewer dispatch. Price: agent time only.

Next: final message to top by 20:30Z.
