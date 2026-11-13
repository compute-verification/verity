[2026-11-13T00:20Z] 7d2af016e855
WALKER, TRN-145. The draft is open: https://github.com/compute-verification/verity/pull/9558. It is a draft, unmarked, based on main, linked to TRN-145, head 102ad1d702 (centre 2^-3, with 1(a)'s and 1(b)'s tests).

All of these RAN at 102ad1d702:
- Suites, gate and ruff, r20261112-233904-3665:
  - 3 suites passed, 0 failed.
  - Ratchet gate 4520abe708b0..102ad1d7021c: 0 failed.
  - Ruff: clean.
- Count and dump, r20261112-233844-46e6:
  - circuit_properties: 81 passed.
  - training_slowdown: 577 passed.
  - grader: 28 passed, 3 skipped.
  - The lf dump is identical to the base, and no library entry is refused.
  - Every point-check record is identical to 53c7b497f2's except the 2 new tests, so no pinned figure or pass moved.
- lf at 2^-2, 2^-3 and 2^-4, both loops (r20261112-233844-46e6):
  - 0 origins, 8 nonfinite nodes (the mask's), 0 NaN nodes.
  - 23 claims, 0 refused, N 6,160,384 and 12,320,768.
- Reverts, r20261112-233931-95e0: all 11 caught.
  - The nan-rule revert fails both NaN tests.
  - compare-by-difference fails the inf-rows test and the specific-overflow test.

Open:
- s-t20 at 2^-3 is r20261112-233958-dacc, still running. The old-centre run r20261112-222055-84cf is still running too. I report both when they end.
- The centre was chosen on lf alone. e4m3 sits off-centre, as it already did before the change.
- The 464/116/29/7 numbers are in the PR body, not beside the constant.
- The draft is stacked on #9315.

