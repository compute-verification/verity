**The branch `cursor/circuits-cir258-fp8-gemm-call-6f04` at `c00bf07b2258` is ready for #9282 to merge. On it, the three wide block-scaled FP8 GEMMs pass circuit-check without `--as-call`. Each is now partitioned by its two callees, and both callees' partitions are checked as Calls.**

What the branch changes (three files, nothing in a Definition):

- `targets.CALLEE_CALLS` gains `ScaledMmFp8BlockCoordinateGivenScale_v2`, passed by `ScaledMmFp8BlockSharedScale_v2`. The GEMM's own Call is also added to the existing `F32Mul_v3` entry, because at N < G the GEMM's body is the K/G tile-scale products (`F32Mul_v3`) plus the N coordinates.
- The Boolean FP8 sidecar holds two `RANGES` rules (a family may have only one). `ScaledMmFp8BlockSharedScale_v2{K=256..6144 step 256,N=4..8 step 4,G=128}` has the four served sizes as its spots. `ScaledMmFp8BlockCoordinateGivenScale_v2{K=256..6144 step 256,G=128}` has K = 256, 1536, 2048 and 6144 as its spots.
- One pin was written by circuit-check's own update pass: the K=256 coordinate's whole lowering, 61,747 ANDs.

Verdicts at `c00bf07b2258`, from the confirming pass with `--ranges` and no `--as-call` [run `r20261112-125806-3bee`]. That pass produced no pins diff. Over the 10 targets there were 0 failures, and both rules hold.

| target | gates | partition |
| --- | --- | --- |
| `{K=6144,N=4,G=128}` | 18,141,744 | by callees `F32Mul_v3` and coordinate `{K=6144}` |
| `{K=2048,N=4,G=128}` | 6,047,728 | by callees `F32Mul_v3` and coordinate `{K=2048}` |
| `{K=1536,N=8,G=128}` | 8,985,672 | by callees `F32Mul_v3` and coordinate `{K=1536}` |
| `{K=256,N=4,G=128}` | 756,596 | its own cut, 4,544 proof blocks; 30,768 gates recomputed across them (Q_word v2) |

All four GEMMs matched the word reference on 1,024 vectors (20 edge, 1,004 random) with 0 mismatches. Each coordinate at K = 256, 1536, 2048 and 6144 is within the partition budget (185,554 to 4,449,156 gates). Each is one proof block, partitioned and passing.

Next: the two walk tests that hold the list (plus the negative with the entry removed), and what the change adds to `check`. Then the pull request.
