---
id: 20261105T1720Z-report-landing-reds
campaign: landing-speed
lane: top
kind: report
status: final
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

CHECKPOINT 18:20Z.

Inbox 10:15 and 10:22 AM PST read. My numbers against the other workers': CHECKPATH's after-landing green at 9 of 50
tips matches mine (8 of 36 drive runs gated green 29 Oct-3 Nov). CHECKPATH's 197 counts landings (batches) and mine,
703, counts PRs reaching `checked`; DESIGN's 693 landings since 29 Oct is the PR count too, and its ~20 fix-forwards
(break rate 2-5%) sits beside my 33 culprit PRs (4.7%, counting each PR of a multi-PR red). TRACE and DESIGN found too
that nothing reverts because a marked fix holds the revert; the first-parent bug (finding 3) and the live case's
mechanism (finding 4) are REDS's own.

Live case refreshed at 18:18Z: `main` c6ba569002 (moved about 10:51Z), `checked` 50745b55b4 (17:29Z). #6980 and #6996
landed in 343a905778 (09:38Z, 8 PRs); 3f209e82b0 and 4bc15be99e at 10:48Z (5 PRs); 32 PRs after, in 7 pushes. 45 PRs,
253 PR-hours on `checked` so far (`checked`'s push activity, each PR from its push to 18:18Z).

Simulation (art:df78cbfbbb619e43f51d9719db9f6e00083ccaa92f552fbfd89b93544e18d859; scripts at
`notes/landing-reds/sim_reds.py`, `reds_sim_run.py`, `reds_sim_grid.py`). DESIGN's `sim.py` (branch
cursor/top-landing-design-8b2a, data art:fb0c4e715c3aea1f0fca101fe64e9f6f7c063f071d478dbbb427d057f2628e34) plus one
lever of its current model, `precheck`: the changed suites run on a landing batch's merged tip; a breaker is caught
with probability `catch` (15/24, the incidents of my table that check would have caught; ci's lever 3: 4/24) and leaves
`next`, back fixed `fix_min` later; `pre_min` adds to the landing pass, or with `async_pre` the check starts when the
member joins `next` and no pass waits; `lean_only` catches only Lean members' breaks (lean-suites in the Lean check,
lean_min 82 -> 92). Grid: fix_min 240 (my measured median) and 360 (DESIGN's calibration), break draws b/c/d, p_break
0.04. Mean ready->main of the 546 PRs marked in the checked era:

| lever | mean ready→main, fix 4 h | vs today | mean, fix 6 h | vs today | to main/day |
|---|---|---|---|---|---|
| today (DESIGN's calibrated model) | 58.2–63.3 h | +0..+0% | 72.4–85.7 h | +0..+0% | 91 |
| revert at once (final L1) | 50.9–53.3 h | -18..-10% | 50.9–53.6 h | -37..-27% | 90 |
| changed suites in the landing pass, +30 min | 77.7–80.3 h | +27..+34% | 79.6–89.5 h | +4..+10% | 83 |
| changed suites from joining next (async) | 51.7–59.2 h | -13..-6% | 56.1–72.5 h | -27..-15% | 90 |
| lean-suites in the Lean check only (final L2) | 58.9–63.1 h | -4..+2% | 62.9–88.4 h | -13..+4% | 91 |
| async check + revert | 51.5–53.5 h | -17..-11% | 51.6–53.3 h | -38..-28% | 90 |
| ci's lever 3 in the pass, +8 min | 62.2–67.1 h | +1..+10% | 66.1–97.2 h | -14..+13% | 91 |
| DESIGN: no pin + open admission + 10-min pass | 15.8–20.7 h | -73..-67% | 16.5–33.2 h | -77..-61% | 110 |
| DESIGN's + revert | 11.4–12.1 h | -81..-80% | 11.3–12.1 h | -86..-84% | 113 |
| DESIGN's + async check + revert | 11.0–11.6 h | -82..-81% | 11.1–11.6 h | -86..-85% | 112 |

What this changed from the 18:00Z draft: the pre-landing check fell from first to second. My draft's estimate (5,250
PR-h) held throughput fixed; the queue is saturated (DESIGN: admission caps throughput), so wall time added to the
landing pass costs more than the stall it saves, and run asynchronously it only duplicates what a working revert does.
Reverting is robust over every draw and both fix waits.

Process-design note: L1 adds no gate; it retires the unbounded hold in `_red_clock` and ci's hand after-landing loop
(ci-no-postland), making the code do what AGENTS.md and the 11 Oct ruling already say. L2's lean-suites adds a shard to
the existing pre-landing Lean check (+0-15 min wall, machine only, no human, no new serialization point); its exit is
CI-340's graph read on `next`'s tip.

## Report (final, sent to top 18:25Z)

REDS final (landing-speed). Note: lanes/top/20261105T1720Z-report-landing-reds.md, copy and scripts at `notes/landing-reds.md` on cursor/top-landing-reds-fb4c. Data: art:ca7e94f669b42e3fad91e552b3a49ffecbdaad04c6f9d8216fd757f509a9629d. Simulation: art:df78cbfbbb619e43f51d9719db9f6e00083ccaa92f552fbfd89b93544e18d859 (DESIGN's simulator plus a pre-landing-check lever, fix waits of 4 h and 6 h, three break draws).

**Findings**
1. About 1 PR in 21 lands a gated red: 33 of 703 PRs reaching `checked` 29 Oct–5 Nov (4.7%). Causes: cross-PR semantics 10, lint 9, inputs or renames 4, locks 3, tooling tests 3, flakes 3. A 27–45-PR run is clean 11–27% of the time (CHECKPATH: 9 of 50 tips). Two stalls (27.3 h and 37.2 h) hold 67% of the 9,388 PR-hours on `checked`.
2. Fix-forward sets every stall's length, and nothing bounds it. 14 fixes, red to fix on `checked`: median 4.0 h (0.6–8.9), the long ones behind `next`'s Lean checks. Nothing was reverted in 14 days: drive scheduled 8 reverts and `_red_clock` held each while any fix was marked. Daniel's 11 Oct ruling: a post-landing failure "reverts the landing at once".
3. Drive's culprit naming was broken. In 49 reds it named nothing because "a revert lies in range". All 49 ranges had 0 reverts on the landing branch; the 109 it saw sat inside PRs' own branches. I fixed this in ac7673f757 (`--first-parent` in `Driver._name`, plus a test that fails without it; the suite passed in r20261105-174519-cf04). Since 3 Nov 19:20Z drive names nothing at all (ci-no-postland).
4. The live case. #6980 and #6996 landed in 343a9057 (09:38Z). The run for 4bc15be9 (r20261105-110254-368d) failed in sampled-proofs, composition and verity-flock. ci fixed sampled-proofs by hand (#7499); composition and flock surfaced 4.6 h later (r20261105-154334-d0e6).
   - Why it went unnoticed: a hand run keeps no per-suite red ledger, and #7499's check never reran the tip's red suites.
   - Why it landed: the prefix Lean check runs only `lean-audit` (`LEAN_STEP`), never the Lean–Python difftests.
   - What it held: `main` unmoved for 7.5 h, with 45 PRs on `checked` (32 behind the red batches), 253 PR-hours by 18:18Z.
5. Simulation: reverting is the lever, and a check on the serial pass is a loss. Revert at once saves 5.7–10.8 h a PR (3,400–6,300 PR-h/week) at a 4 h fix wait, and 19–32 h at 6 h. A changed-suites check adding 30 min to each landing pass is 17–20 h a PR worse (throughput drops from 91 to 83 a day). Run asynchronously it saves 4–8 h alone, but adds nothing measurable once reverts work (−0.7 to +1.1 h). ci's lever 3, run inside the pass, is 0.5–5.7 h worse.

**Levers, ranked**
L1. Revert at once, as ruled.
- Mechanism: (i) land ac7673f757; (ii) retire ci-no-postland, with hand shards joining drive through `queue drive-shard-add`, so drive keeps the red ledger; (iii) in `_red_clock`, a fix holds a revert only until FIX_WINDOW_S unless it is on `checked`; (iv) `_land_fixes` lands a named red's fix at the next pass, outside the prefix and `lowrisk_excludes`, once the red's suites pass with it.
- Hours removed: 3,400–6,300 PR-h/week, and 2,600–5,100 more on top of DESIGN's throughput levers (mean 16–21 h → 11–12 h, p90 32 h → 20 h). My measured method gives 3,480, at the low end.
- Cost: 10–15 reverted PRs a week re-land later.
- Risk: a wrong culprit (1 of 9 so far). Caught by 24 h in shadow mode, where drive posts "would revert X" and ci compares it with its own attribution, and then by the reverted tip's run: if it stays red, drive re-lands the PR.
- Rule: AGENTS.md already says the culprit is reverted; the code is made to match, and the unbounded hold is retired.
- Owner: ci. (i) today, the rest by Saturday.

L2. Keep checks off the landing pass. The prefix Lean check runs lean-suites beside lean-audit (a second shard, adding 0–15 min), which catches the live case's class of red. ci's lever 3 runs in admission's preflight, never in the pass. Once L1 works, this buys fewer reverts (3 a week instead of 10–15), not latency. Owner ci; 1 day.

L3. A real uncached backstop. `after_landing_no_cache` was set true on 24 Oct as a precaution with no incident behind it, and has been false since 29 Oct; every run since reuses verdicts. No cached/uncached pair exists on one tree in 14 days, and CI-330's comparisons (70 reused verdicts) hid no red. drive.toml's backstop, audit-main, has never run. The lever: a nightly no-reuse graph of `main` (about 40 min on 13 slots), compared key by key with the cached verdicts; a disagreement rings ci and drops that key's passes. It makes cached and CI-340's changed-tasks-only runs safe. Owner infra; 1–2 days.

L4. Machine and tooling faults cost about 11 h of stall a week. ci.toml runs gated jobs before off-gate ones, a failed task reruns once on another node, drive rings on a refused push, and a move PR lands alone. ci and infra; 1–3 days.

In flight: full-check-45-min helps most here (DESIGN's run-60 row: mean 72 h → 57 h). CI-427 shortens passes but not fix waits, CI-340 acts only with L1(ii), and nothing in flight reverts.

**Next 48 h**
1. Today: land ac7673f757; the Lean check gains lean-suites; drive runs reverts in shadow mode on 4bc15be9's reds.
2. Friday: ci-no-postland off and the bounded hold; reverts go live once 24 h of shadow agrees with ci.
3. Saturday: the wider `_land_fixes` and the nightly uncached graph. Measure reds per 100 landings (now 4.7) and red-to-revert time.
