[2026-11-14T09:42Z] 1e0541d0821b
READSTEP, Sat 14 Nov, 01:41 AM PST. Delta read of draft #10069, 7ed779a61fbba79c9240c07ef6e8ec1468f9a24b..f4c55b04966f298060ca5b4ac656931b46325fb2.

**The MUST FIX is cured to your ruling (one measurement, the world's year, and a test fails on a second one or another period) and all three SHOULD FIX items are done. What's left is a miscounted test comment and a body that lost its `Issue:` line.**

The branch is at f4c55b04966f298060ca5b4ac656931b46325fb2 (RAN, `git ls-remote`). The delta is three commits, one per fix, in five files (RAN, `git log`, `git diff --stat`). It merges cleanly onto `origin/main` d97767352e2 (RAN, `git merge-tree`).

## Your questions

1. **MUST FIX cured: yes.**
   - RAN: a whole run of `periods.py --out D` at the head writes one measurement: `training/slowdown-onsite-h200`, value 3.040220763439439, mark `modeled`, `period` year, `over` world. Its unit equals `metrics.toml`'s, and the value is `periods.json`'s world-h200 year row.
   - READ: `periods.py:398-401` picks that row by (scenario, fleet, period), and the test at `tests/test_analysis.py:298` unpacks `(got,)`, so a second measurement fails it.
   - My mutations (RAN, each alone in a scratch worktree, then reverted; the worktree is clean):
     - A second write (the world's month too) fails `test_periods_writes_the_onsite_lower_bound_under_its_metrics_id`.
     - Every onsite row written, the old loop, fails the same test.
     - The selector moved to the month fails it.
     - The `period` label set to month fails it.
     - The selector moved to the datacenter's year **survives**. At the registry, both fleets' rows are equal in every period (RAN, both years are 3.040220763439439), and the call hard-codes `over="world"`, so what's written is the same. NOTE 1.
2. **Each SHOULD FIX done as I meant: yes.**
   - Retired term: `run.py:111` now says "an exact count on the circuits it builds" (READ). `git grep "declared circuits"` in `benchmarks/circuit_properties` finds nothing (RAN). `ratchet_gate.py 9a520e76c0d f4c55b04966f` gives "1 change(s), 0 failed" (RAN).
   - The rho gate's arms: the no-measurement test now loops over `--set`, `--bounds cover`, `--measured smollm2-135m-step` and the remote scenario (READ). Dropped alone, each of `args.set`, `args.bounds` and `args.measured`, the whole gate, or the scenario term of the selector fails `test_periods_writes_no_measurement_off_the_registry_or_without_the_onsite_scenario` (RAN).
   - The library: the test runs at other starts (no `result.json`), and once with only the priced entry (mark `measured`) (READ). Each of these fails `test_a_run_of_every_entry_writes_its_least_slowdown_under_the_metrics_id` (RAN): the `--starts` arm dropped, the entry arm dropped, the mark always `partial`, the mark always `measured`, the whole gate dropped.
3. **The body:** every sentence I could check is true at this head. It names no machine ("a CPU node of the research pool"), and it gives no measured figure (READ, `gh pr view`, draft, head f4c55b0496, updated 09:33Z).
   - "Each arm of each gate, dropped alone, fails its test": RAN for rho and the library at this head. circuit_properties' one-arm gate failed its test at 7ed779a and the delta doesn't touch it (READ).
   - "ruff is clean on the new code (`run.py`'s two E402 findings were already on `main`)": RAN, the same two E402 at lines 28-29 at base 9a520e76c0d and at the head.
   - `ratchet_gate.py` passes: RAN.
   - "Run by path, its output is the same as `python -m`": `--help` RAN at the first read; a full run I didn't do (INFERRED).
   - It lost its `Issue:` line (SHOULD FIX 2).
4. **Anything else in the delta:**
   - `benchmarks/rho/README.md:217-218` now says the world's year (READ).
   - The `OVER` table is gone, and nothing in `benchmarks/rho` still reads it (RAN, `git grep`).
   - The library test wraps `library.row` to trace each entry once (READ, `tests/test_library.py:111-119`), and its comment miscounts the runs (SHOULD FIX 1).
   - Nothing else (READ, the whole diff).

## MUST FIX

None.

## SHOULD FIX

1. `benchmarks/training_slowdown/tests/test_library.py:111` says "each entry traced once over the five runs". The test makes four `library.main` calls: every, named, other, priced (READ). Fix: "the four runs", or drop the count.
2. The body has neither `Refs KEY` nor `Issue: none: REASON` (READ). My first read recorded an `Issue: none:` line at 7ed779a (READ then).
   - `named_issue` (`tools/research/src/research/queue.py:6568`) returns None, so `research queue status` lists the PR under "no Linear issue named" (READ). The goal gate still passes on its `Goal` line (READ, `:6548`).
   - Fix: restore the line with `research queue edit 10069 --body-file F`.

## NOTE

1. The datacenter-for-world selector mutation survives because the two fleets' onsite rows are equal at the registry (RAN). A test can't separate them without changing the registry, and a run that does is off the gate and writes nothing. So it holds while the fleets agree (INFERRED).
2. The changed tests pass at the head (RAN):
   - rho `-k periods_writes`: 3 passed.
   - The library test: 1 passed, 8 s.
   - circuit_properties `-k every_configuration`: 1 passed, 5.92 s.
   - `tests/test_metrics_registry.py`: 6 passed.
3. ROWS' quick run r20261114-093256-a75b is `waiting` at `tree_sha` f4c55b04966f298060ca5b4ac656931b46325fb2 (RAN, named keys). The body says pending. I launched no run.
4. No event, PR comment or run output addressed to @training reached me (RAN, the PR's timeline has no comments). I acted on none.

## For the lead to read

1. `benchmarks/rho/periods.py:385-401`: the one write, and the selector.
2. `benchmarks/rho/tests/test_analysis.py:298-325`: `(got,)`, and the four no-measurement cases.
3. `benchmarks/training_slowdown/tests/test_library.py:111-133`: the `row` wrapper, the "five runs" comment, and the starts and priced cases.

READY f4c55b04966f298060ca5b4ac656931b46325fb2

