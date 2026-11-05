---
id: top/20261105T1713Z-report-landing-checkpath
campaign: landing-speed
lane: top
kind: report
status: draft
repo: compute-verification/verity
origin: top's worker CHECKPATH (Cursor cloud agent), branch cursor/top-landing-checkpath-7e53; brief from top, Thu 5 Nov 2026
---

# CHECKPATH: red-team of the check's critical path (landed on `checked` to `main` moves)

Checkpoint 18:00Z. Sources: every reachable CPU node's run dirs (3,589 check.py runs, read by ssh), drive.log on the
control pod (read only), the executor's events for s20261105T035123Z-a2bf21 and s20261103T000340Z-aeb35c, `verity graph`
listings at 22 `checked` commits, the Lean import graph at `checked` 50745b55. Tables and the scripts that made them:
art:410b0b76284443025d92c55d6dbe01c76ddb636ab454f15bedae5e11aeafdef0 (kind landing-study/v1).

## Findings

1. **Landed to `main` takes 10 h at the median.** 197 landings on `checked` Oct 30-Nov 5: median 10.0 h to `main`, p75
   17.2 h, p90 18.1 h, mean 12.3 h (`land_to_main.json`); 25 more are waiting now (`main` c6ba5690 since ~11:00Z). `main`
   moved 15 times in 7 days; of the 8 moves since Nov 3, 7 were ci's `queue baseline` by hand and 1 a passing
   after-landing run. Since Nov 3 19:19Z `/data/drive/ci-no-postland` exists: drive starts no after-landing run, ci runs it
   by hand. The common path has a human step today.
2. **The after-landing run is green on its gated steps at 9 of 50 tips (18%)** (`after_landing_reds.json`). The
   whole-tree invariant suites (repository 211 s, verity-check 215 s, verity-graph 447 s, research 71 s on d0e6) were red at
   24 of 50 tips and the only gated red at 7. They are ratchets a batch trips together ("no new silent excepts" 7 tips,
   "direct reads only shrink" 6, "no new lane text" 5, the graph's check reads 4, `test_infra_paths` for #6414). Nothing
   runs them on `next`: before landing only Lean changes get a check (`_lean_check`), and the author's own run is
   `suites.py --changed origin/main`, against a `main` 10 h behind.
3. **Drive's tick is the clock of every handoff.** Ticks on Nov 4-5: 2,176-9,930 s, median ~58 min (`Queue._merge`
   1,300-4,300 s, `land_lowrisk` up to 3,900 s; `ticks.txt`). In `_tick`, `follow_postland` and `_promote` run at its
   start and `start_postland` at its end, after the sync: a run that ends is seen at the next tick's start and its
   successor launched only at that tick's end, about 1.5 ticks later. Between consecutive after-landing runs the slots sat
   idle a median 41 min (18 gaps Oct 30-Nov 1, drive launching) and 54-413 min since Nov 3 (ci launching by hand).
4. **The run's floor is one test file, and Proofs is built twice.** aa7843fd: shard 2 (r20261105-154523-27fb) 75 min =
   lean-suites 72 min, its long pole composition's `test_composition_difftest.py` 3,746 s (`test_remote_agrees` 1,214 s,
   1,000 examples, `slow`; `test_product_over_remotes_subprotocols_is_remote` 1,140 s; the rest at most 320 s), verity-flock
   2,479 s, pouw-benchmarks 1,945 s; composition's inputs hold all of `verity/ml`, so it reruns at every tip. The
   composition suite was 7 tests in 210-380 s at 32 after-landing runs Oct 29-Nov 3 and is 21 tests in 1,814-3,746 s at
   every run since #6677 (Nov 3 20:12) and #6714 (Nov 4 00:22) made recursive sampled proofs a component
   (`test_onsite_agrees` 130 s to 1,130-1,530 s, `test_remote_agrees` 45 s to 560-1,214 s, same 1,000 examples): two PRs
   added ~50 min to every run's floor, and nothing priced it. Shard 2's
   lean-audit of Security/Properties rebuilt Proofs for `proved_in` (2,088 s build + 1,055 s replay, 143 GiB) beside shard
   1's (1,743 s + 1,010 s). Shard 1's pytest: 40 of 83 suites again (keys hold all of `verity` and every fixture), 9
   `parallel` suites one at a time, 37 min beside the Proofs build. More shards don't go under the composition file.
5. **No Lean result travels between commits.** No CPU node runs the sandbox (AppArmor's userns restriction), so
   `LakeArtifacts` stores nothing ("staged: 0 of 4416"); INF-533's lift was reverted for sandbox.sh's symlink bug, whose
   fix #7501 is on `checked` 3b3f745e, not on `main`. Warm trees are node-local and the shards rotate nodes: 7 of 41
   after-landing Proofs builds were warm (145-360 s build, 64-200 s replay), the rest 1,870-4,400 s plus 570-2,050 s.
6. **What a batch needs in Lean is small** (`cones.json`, module seconds from aeb35c, no early cutoff, so an upper
   bound): 3b3f745e to aa7843fd: 26 `.lean` changed, cone 232 modules, longest chain 425 s; aa7843fd to 50745b55: 8, 35,
   159 s. The worst batch (c6ba5690 to 4bc15be9, #5534's 242-file comment sweep) is everything: 3,526 modules, chain 222
   deep, 4,271 s with the executor's ~8.2 s per task (2,451 s without it).
7. **The executor as configured reruns nearly everything and is slower than train.sh.** Test-file keys at 19 single
   landings: 15 changed 77-97% of task-seconds (`keyreuse.json`). 1,559 of 2,155 test files are `untraced` (vllm's
   `compilers`, 1,098) and key on their suite's whole inputs: one vllm acquire test's key holds 572 `.lean` files, so #7245
   (3 `.lean` files and a lock in verity/core/service/flock) changed 1,583 keys. Its after-landing submission a2bf21: 307
   min. 18.3 task-h of test files ran about 30 at a time on 13 workers of 16 cores (tasks declare 8 GB for vllm, peak p50
   0.26 GB; 16 GB for flock, peak p50 0.5 GB), done at +117 min; the build family (lean-build, then lean-suites as one
   9,059 s task) started only at +120 min (`executor_a2bf21_summary.json`).
8. **Drive's Lean checks of `next`** (`check.py --deferred lean-audit`): median 59, 65, 44 min on Nov 3-5 (p90 ~110); 23
   of 54 on Nov 3-4 ended by drive's cancel when `next` was rebuilt.

## Levers, ranked (model: mean landed-to-main = R/2 + R/p, R a cycle, p a green run; R = 150 min and p = 0.18 give 15 h
against 12.3 h observed, scaled by 0.82; 197 landings a week)

1. **Run the tree invariants and the touched suites on the batch before it lands** (ci; builds on ci's low-risk lever 3).
   `suites.py --changed <checked tip> --quick` plus repository, verity-check, verity-graph and research on the merged tip
   of what lands, beside the Lean check of `next`; a red is blamed by the Lean check's walk (each member's own commit).
   p 0.18 to at least 0.32: ~4.8 h off the mean, ~950 landing-h a week. 0 min on the path beside a Lean check (44-65 min),
   ~8 min without one. Risk: a flaky test ejects; the red names it and the flake rule deletes it. Retires the team skill's
   prose "Before marking, run ... --changed origin/main". Lands Fri.
2. **Take the after-landing loop out of the tick** (ci). Today: call `start_postland` right after `follow_postland` at the
   top of `_tick` too (one line; the end-of-tick call stays), so a run's successor launches the tick it is seen, not 1.5
   ticks later. Then: `follow_postland`, `_promote` and `start_postland` every minute under drive's state lock, outside
   `Queue.sync`. R 150 to ~95 min: ~4 h off the mean alone, ~800 landing-h a week. Risk: two writers of drive's state
   (caught by drive's tests). #7490's shorter tick helps; this removes the coupling.
3. **One Proofs build, and lean-suites in parts** (ci). Co-locate `lean-audit:Security/Properties` with Proofs in ci.toml's
   3-shard plan (or `after_landing_shards = "auto:6"`); split composition by test id (its three 1,000-example difftests
   are 19-25 min each; split each one's examples in two for ~12 min), verity-flock and pouw-benchmarks by file; pytest in
   2 parts off the Proofs node. Shard 2 75 to ~25-35 min. Then hold it in code: suites.py fails a gated suite's test over
   10 min with the fix in the message (parts or fewer examples), in place of AGENTS.md's prose "a test that adds more than
   about a minute says why", which didn't stop #6677/#6714. Lands Fri (ci.toml, PARTS; the cap Sat).
4. **Warm Lean on the typical batch** (infra). Until INF-533's cache works, give the Lean shard the node of the last one
   (WarmBuilds hits only there); deploy #7501 from `checked` rather than wait for `main`. Shard 1's Lean 51 to ~5-10 min.
   Cold under 45 needs replay overlapped with the build (as the executor's lean-module does) and the per-task overhead cut
   on the 222-deep chain; best cold today is 1,743 + 1,010 s sequential, 46 min.
5. **Before CI-340's flip** (ci, infra): key untraced compiler tests without Lean and Lake files (the read guard catches a
   read), declare test-file memory from the executor's measured peaks, start the longest chain first. Flip on shadow
   agreement and on wall time at or under train.sh's for the same tip; otherwise the flip lengthens landing (307 vs 78-90
   min).
6. **Automation back on** (ci): delete `ci-no-postland` once 1-3 land; then `main` moves without `queue baseline`.

Target after 1-4: a typical run ~30 min, R ~35 min, p ~0.5: mean landed-to-main ~1.5 h (model).
