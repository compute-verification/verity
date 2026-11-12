[2026-11-12T21:13Z] 5fae95c048ab
TILES, Thu 12 Nov, 1:13 PM PST. Handoff.

**The review of PR 9417 is done (NOT READY, 3 must-fixes, sent at 1:12 PM PST); the handoff is note `training-20261112T2112Z-handoff-TILES`.**

Done: the review. The full report is note `training-20261112T2025Z-report-TILES-review-9417`. I pushed nothing to the branch.

Left, in order:
1. The author fixes the three must-fixes, each with a test on crafted subnormal inputs:
   - Pre as FMUL then FFMA;
   - Dq's buffer adds with FTZ;
   - `CONFORMANCE` narrowed, or cited to the every-word run.
2. Rerun the every-word run at the fixed head, or record that the fixes leave every captured word as it is (M6 and M7 pass every test).
3. Shoulds 4 and 5, the tests that pin the order.
4. Shoulds 6 to 10.

Runs in flight:
- None of mine. `r20261112-202023-ebf9` (the forward's circuit-check) finished SUCCESS (RAN).
- The author's `r20261112-192536-b84b` and `r20261112-194241-5573` were running at 21:11Z (RAN `research status`). I relaunched neither.

