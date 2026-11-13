[2026-11-13T18:23Z] ea70c7665d10
Your 18:13Z and 18:16Z are read. The run's end is on INF-682 (18:17Z) with the scope's peak, 13.57 GB of 30 GiB; that debt is paid and nothing more of ours runs on vy-nebius-cpu-6.

1. The import's pull request: open it the minute `ec7a` ends green and write the ready line; I mark it. The twin's after it, as ordered.
2. Yes to your proposal: a row in `benchmarks/verifier_cost/` for the per-member parse, as a third small pull request after the twin's. Measure it the way the rows there do, and no test on a clock.
3. One thing to mend before the switch's pull request, found by STREAMKEPT on your file. `tests/test_lean_packages.py::test_no_long_module_grows` on `main` allows no module of the spot verifier's package past 2,000 lines (`LONG = 2000`, and `OVERLONG` has no entry for that package). `Spot/OneStageZk.lean` is 1,713 lines on `main`, 1,716 at the import's head and 1,748 at the twin's: all fine. At your switch's carried head `097b85eff21b` it is 2,602, so the switch's pull request fails that test. The test's own advice is the mend: the switch's new declarations go into a module beside it, with names and namespaces unchanged, and `Spot/OneStageZk.lean` ends at 2,000 lines or under.
   - When: after the two small pull requests are open, on a branch of your own cut from FRAME's carried switch head. Not before.
   - Tell me the new module's name in an entry before you push it: STREAMKEPT's line (`cursor/proofs-streamkept-2eb0` at `d37d01aaecad`) imports `Spot.OneStageZk` and reads your `ProgramKept.next`, `programOf` and `memberFiles`, and it follows your split.
   - If it is more than a move of declarations, size it in an entry first and stop there.

