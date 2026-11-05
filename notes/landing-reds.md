---
id: 20261105T1720Z-report-landing-reds
campaign: landing-speed
lane: top
kind: report
status: draft
repo: verity
origin: bc-737c0a42-fa89-51b7-9407-8ecb9fc08e6f
---

CHECKPOINT 17:20Z: REDS worker for top's landing-speed study. Branch cursor/top-landing-reds-fb4c (no code yet).

Sources pulled (scratch, to the store at the end): GitHub push activity of main (559 pushes since 5 Oct), checked
(113, created 29 Oct 17:34Z), next (1905); ci's queue thread (10,145 replies since 22 Oct); #agent-alerts (38,150).

First facts:

- main moved about hourly 22-28 Oct (land onto main, check after; main often 37-43 h unverified, e.g. 595cce6d
  verified while tip 39 landings ahead, 26-27 Oct). Since `checked` (29 Oct 17:34Z) main moved 17 times in 7 days,
  gaps 2-37 h (31 Oct 17.4 h, 1 Nov 27.3 h, 3 Nov 37.2 h, 5 Nov 16.2 h).
- No drive revert of a culprit ever reached checked's first-parent history: every red since 29 Oct was fixed forward.
- `after_landing_no_cache` is false on main, checked and next since c2829cf8bc (29 Oct 19:39Z, co-authored by Daniel).
  It was true from 2d2eec3783 (24 Oct, the switch to land = "review") as a precaution, "until a suite's reuse is per
  test", not after an incident. Today's uncached runs are ci's hand launches (/data/drive/ci-no-postland shim since
  Tue 3 Nov 19:20Z), not drive's config.

Next: classify every after-landing red in the thread by cause; join to main's stall hours; cached-vs-uncached pairs.

CHECKPOINT 17:55Z.

Correction to 17:20Z: ci's hand after-landing runs are cached too (e.g. r20261105-154334-d0e6: `--verdicts-in`, no
`--no-cache`; pytest reuse 51.8%, lean-audit 0%). No uncached after-landing run has run since 29 Oct.

Numbers (scratch in /tmp/reds, to the store before the final):

- 703 PRs reached `checked` since 29 Oct 17:34Z; 658 are on main. checked -> main: median 11.8 h, p90 25.2 h, max 38.5 h;
  9,388 PR-hours waiting on checked. Two stalls hold 67%: 31 Oct 11:35Z -> 1 Nov 14:53Z (27.3 h, 240 PRs, 3,654 PR-h) and
  2 Nov 02:36Z -> 3 Nov 15:50Z (37.2 h, 109 PRs, 2,653 PR-h); then 4 Nov 13:36Z -> 5 Nov 05:47Z (16.2 h, 989 PR-h) and
  30 Oct 18:08Z -> 31 Oct 11:35Z (17.4 h, 839 PR-h). Excess over a 2.5 h cycle: 130 of 168 h.
- About 33 culprit PRs made gated reds on checked in 7 days (lint and lane text 9, cross-PR semantic 10, lock or records
  3, undeclared input or path or rename 4, a test of the tooling itself 3, flakes 3; plus 3 machine or slot faults):
  ~4.7% of landings. A batch of n landings is clean with p = 0.953^n: 53% at 13, 27% at 27, 11% at 45, 1% at 100.
- Red-fix wait (red named -> fix on checked), 16 fixes from ci's log: median 3.5 h, range 0.6-8.9 h, 63 h in all. The
  long ones waited behind next's Lean checks (fixes that touch tests/ or tools/verity/check/ leave the low-risk lane), a
  move restack loop, GitHub 500 on push, a launch that never registered (verdict gathering > train.sh's wait).
- Reverts: drive named a culprit on checked 9 times and scheduled 8 reverts; 0 happened (`_fix`: any marked fix holds
  the revert with no bound on when it lands). Since 3 Nov 19:20Z drive names nothing (ci-no-postland): ci reads runs by
  hand. Daniel's 11 Oct ruling says a post-landing part's failure "reverts the landing at once".
- Bug: drive gave up naming a culprit 49 times (26 on checked, 23 on main) with "a revert lies in range"; in all 49 the
  range's first-parent line had no revert: each was a PR's revert of its own commit inside its branch. Fix on
  cursor/top-landing-reds-fb4c (ac7673f757: `--first-parent` in `Driver._name`, plus a test that fails without it);
  tools/research suite r20261105-174044-ac1a running.
- Live case (inbox 9:25 AM PST): 4bc15be9's lean-suites step was red in sampled-proofs, composition and flock (all three
  are AFTER_LEAN suites); ci read the first, fixed it with #7499, and found the other two 4.6 h later at aa7843fd. 45 PRs
  on checked wait on them, main unmoved since 10:53Z (6.8 h, 214 PR-h so far). The prefix's pre-landing Lean check runs
  only `lean-audit` (drive.py LEAN_STEP), so no Python-Lean difftest sees a Lean PR's merged state before it lands.
- Cache: CI-330's three shadow comparisons (70 reused task verdicts) found no reuse that hid a red; disagreements were
  keying bugs, no-result exits, an inode-floor test, skips counted as passes. The nightly uncached backstop drive.toml
  cites (audit-main) never ran: jobs queued 0/3, none claimed since 7 Oct.

Next: incident table to the store; lever estimates; final to top by 20:30Z.

## Report (draft 18:00Z; final numbers at 20:00Z)

Data: art:ca7e94f669b42e3fad91e552b3a49ffecbdaad04c6f9d8216fd757f509a9629d (`landing-study/reds/v1`: stall intervals,
24-row incident table, 703 PRs' checked -> main waits, the 49 "revert in range" records with first-parent counts, every
main move since 22 Oct). Method of each estimate below: /tmp/reds/levers.py, restated here.

### Findings

1. Reds after landing are the norm, not the exception. About 33 culprit PRs made gated reds on checked in 7 days
   (4.7% of 703 landings): cross-PR semantics 10, lint and lane text 9, undeclared input, path or rename 4, lock records
   3, tests of the tooling 3, flakes 3, plus 3 machine faults. At that rate a run over the typical 27-45 landings is
   green 11-27% of the time; main moves only when fixes outrun new reds.
2. Fix-forward, not revert, sets the stall length. Code-fix waits (red named -> fix on checked), 14 fixes: median
   4.0 h, 56.6 h in all; the 7 of 14 at 4 h or more waited behind next's Lean checks (fixes touching tests/,
   tools/verity/check/ or Lean leave the low-risk lane and `_land_fixes`), a move-restack loop, a GitHub 500, or a
   launch that never registered. Each red wave then needs a fresh 2-2.5 h after-landing cycle.
3. Drive's revert never fires. 9 culprits named on checked, 8 reverts scheduled, 0 done: `_fix` holds a revert while
   any ready PR is marked as the fix, with no bound on when it lands. 49 reds named no culprit because of a bug ("a
   revert lies in range": every such range had 0 first-parent reverts, 109 PR-internal ones). Since 3 Nov 19:20Z drive
   names nothing at all: ci runs after-landing by hand (ci-no-postland) and reads reds by hand. Daniel's 11 Oct ruling:
   a post-landing part's failure "reverts the landing at once".
4. The live case: 4bc15be9's lean-suites step failed in three suites; ci fixed the first (#7499) and read the other two
   (composition: #6980; verity-flock: #6996) 4.6 h later at aa7843fd. A red at the bottom of checked held 45 PRs; main
   unmoved since 10:53Z. The pre-landing Lean check of next's prefix runs only lean-audit (drive.py LEAN_STEP), so no
   Python-Lean difftest sees a Lean PR's merged state before it lands, and #7499's own check didn't run the suites
   red at the tip it fixed.
5. Cache: after_landing_no_cache was set true 24 Oct (2d2eec3783) as a precaution, "until a suite's reuse is per test",
   with no incident, and false since 29 Oct (c2829cf8bc). Every after-landing run since is cached (ci's hand runs too:
   `--verdicts-in`; pytest reuse 52%, lean-audit 0%). No cached/uncached pair on one tree with a common verdict in 14
   days; CI-330's three shadow comparisons (70 reused task verdicts) found 0 reuse that hid a red (95% bound about 4% a
   reused verdict). The nightly uncached backstop drive.toml's comment cites (audit-main) never ran: jobs queued 0/3,
   none claimed since 7 Oct.

### Levers, ranked

L1. Check the merged state before it lands. (a) drive's Lean check of next's prefix runs lean-suites beside lean-audit
(a second shard sharing lean-build); (b) every non-Lean landing batch (the low-risk lane's, the reviewed prefix's) runs
`suites.py --changed <checked's tip>` on the batch's merged tip before pushing checked; a red ejects the member whose
change touches what the failure names (`Driver._name`'s rule, moved before landing), and the rest land. Extends ci's
lever 3 (repository invariants only, which catch 4 of the 15 code reds: R01 R07 R23 R24) to the changed suites.
- Removed: 69 h of main stall in 7 days, 5,250 PR-h of 9,388 (56%), about 7.5 h off each PR's mean checked -> main.
  Method: each code red caught on next saves its fix wait plus one 2.5 h cycle, capped per stall at its excess over
  one cycle, times that stall's mean PRs on checked. Catchable: R01 R03 R05 R07 R09-R13 R17 R18 R21-R24 (15 of 24).
- Cost: +1 train slot per Lean check for about 50 min (CHECKPATH's key fix shrinks it); +15-40 min on each non-Lean
  batch, so low-risk PRs land a pass later.
- Risk: a flake ejects an innocent PR (3 flakes in 7 days); caught by one rerun on another node before ejecting, and
  the ejection rings the owner with the failing test.
- Rule: AGENTS.md "Checking and merging": "drive lands next's longest reviewed prefix ... and its low-risk members on
  their ready mark alone" gains "once the suites its landing changes pass on the merged tip". Realizes Daniel's 11 Oct
  "parts under about 10 min cold run before landing".
- Owner ci (drive), infra (slots). (a) 1 day, (b) 2-3 days (suites.py --changed exists today).

L2. Revert at once, with a wider fix lane. (i) ac7673f757 (first-parent) so drive names culprits; (ii) ci-no-postland
off, ci's hand shards joining drive through `queue drive-shard-add`, so drive reads every red of every run; (iii)
`_red_clock`: a marked fix holds the revert only until FIX_WINDOW_S after the ring unless it is on checked by then;
(iv) `_land_fixes` lands a marked fix of a named red at the next pass outside next's prefix and `lowrisk_excludes`,
once the red's suites pass on checked + fix (a Lean fix gets its own Lean check of checked + fix, not next's prefix).
- Removed: 44 h of main stall in 7 days, 3,480 PR-h (37%), about 5 h a PR; overlaps L1 (it handles what L1 misses).
  Method: each code red on a stall's critical path costs max(0, fix wait - 1 h) less.
- Cost: reverted PRs re-land (about 10-15 a week without L1, 3 with it), each a day later. Risk: a wrong culprit (1 of
  9 named: #5268 by bisection) reverts an innocent PR; caught when the reverted tip's run is still red, and drive
  re-lands it.
- Owner ci. (i) today, (ii)-(iv) by Sat.

L3. A real uncached backstop. A nightly executor graph of main's tip with no reuse (about 506 task-min, 40 min on 13
slots), compared key by key with the cached verdicts that verified it; a disagreement rings ci and deletes that key's
passes. Fix drive.toml's comment. No latency now (runs are already cached); it makes CI-340's changed-task-only
after-landing run safe. Cost about 8.5 slot-h a night. Owner infra; 1-2 days.

L4. Machine and tooling events, beyond CI-226 (pinned deploy, direct lane): about 11 h of main stall in 7 days (tick
crashes, move restack loops, NFS merges, a launch wait shorter than verdict gathering, a lean-suites run in a 64 GB slot,
a non-gating circuit-check red skipping gated flock-rust-tests, flakes, composition's 1 h cap). Each is a small change:
ci.toml groups run gated jobs before off-gate ones; a failed task reruns once on another node before it is red; a flaky
test is deleted at its first flake (AGENTS.md says so already); drive rings on a refused push; a move PR lands alone.
Owner ci and infra; 1-3 days.

In flight, and whether it is enough: CI-427 (merge records, passes 65 -> 43 min) shortens each fix's ticks, not its
wait behind Lean checks. ci's lever 3 catches 4 of 15 code reds. CI-340 names culprits by input files after landing;
it does not move checks before landing. infra/full-check-45-min cuts each red wave's cycle from 2.5 h to 0.75 h
(about 25 h of stall a week at 14 waves), the best in-flight lever on this angle; L1 and L2 remove the waves.

### Process-design pricing of L1 (the only new gate)

Failure: 33 culprit PRs a week land reds (incident table R01-R24); about 70 h of main stall a week. Mechanism: the
changed suites on the merged tip; smaller tried: repository invariants only (ci's lever 3) catch 4 of 15. Walkthrough
(29 Oct-5 Nov, 703 landings): Lean prefixes (lean-suites beside lean-audit, +0-15 min wall, machine), reviewed non-Lean
batches and low-risk batches (+15-40 min, machine); blocks by ejecting one member, never the batch. Price per week:
human 0; merge path +15-40 min per non-Lean batch; shared queue: train slots; always-read words +1 clause; pods within
today's slots. Against about 5,000 PR-h a week. Tests: fires at landing (a reliance point); code with the failing test
in the ejection's ring; blocks, because a red on checked holds every landing above it; no new serialization point.
Budget: within (automatic, no human gate). Owner ci; exit: CI-340's graph read on next's tip replaces the suites.py
step. Pilot: the low-risk lane for 8 h; measure reds on checked per 100 landings (now 4.7), batch wall, ejections.

### Next 48 h

1. Today: ci reviews and lands ac7673f757; ci.toml gated-before-off-gate group order; L1(a) on the next Lean check.
2. Fri 6 Nov: L1(b) pilot on the low-risk lane, 8 h; ci-no-postland off; `_red_clock` bounded hold.
3. Sat 7 Nov: widen `_land_fixes`; infra's nightly cold graph and drive.toml's comment; L1(b) on every batch if the
   pilot's reds per 100 landings fall under 1.
