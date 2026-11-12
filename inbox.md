[2026-11-12T22:20Z] 9549d5aa7430
PLAN to training, Thu 12 Nov 2:22 PM PST. **The plan's 38.45 GPU-hours for point 7 price no depth, so 36 blocks changes neither the hours nor the lease.**

**How the plan prices a point (READ: `build_plan.py`, `POINTS` and the loop under it).** Hours = (limit ÷ rate + start) ÷ 3600. Point 7's limit is 1.9 C0, \(1.01\times10^{22}\) gates (p max 1.70). Its rate is 238M's honest rate, \(7.87\times10^{16}\) gates/s, times wildcard's rate over the honest one at 36M and 81M (0.927, 15 records). Its start is 283 s. All three are in gates or seconds, and none reads a depth.

**What the depth changes: the steps, not the hours.** The steps are the limit over one step's gates, so 36 blocks buys fewer steps than 26 would, for the same gates and hours. That holds while the rate in gates/s holds at that depth. Wildcard's ratio was measured on the topcode runs at 36M and 81M, which grew their starts × 2 and × 2.25, so it is measured on grown models already.

**One part of the lease is mine, not the plan's:** I took the trace at about 11 minutes for 36 blocks, scaled from l16's 4.8 minutes for 16. Even 30 minutes leaves `minutes = 3060` and the lease at 61.7 hours.

**The early check passed (RAN: r20261112-213603-3a08, vy-nebius-cpu-3, at my commit 661906ad1f; its last line "done").**
- The suite passed, 160 tests.
- `of_record` holds the seed-102 config "equal, 1 expected difference" (the CPU's norm).
- The seed-102 config and its source share the digest 9b760ef030c5823f, so point 6 runs under its label: its dry run lists final/102 alone.
- Point 1's dry run asks `--max-min 1830` and point 3's `2838`.

**Next:** the test you asked for, which ties each long config to its source but for its steps and budget. Then the shallow steps, once their trace leaves the memory queue on vy-nebius-cpu-10.

