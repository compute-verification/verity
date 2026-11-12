Lane circuits
Refs CIR-258
Goal proofs/exfiltration-under-300gb

## What changed and why

The block-scaled FP8 GEMM on bits (`ScaledMmFp8BlockSharedScale_v2`) at GLM-5.3's served widths (N = 4 at K = 6,144, 2,048 and 256; N = 8 at K = 1,536; G = 128) is over the partition's gate budget from K = 1,536 at N = 8 and K = 2,048 at N = 4. circuit-check may partition such a Call by its callees (`checks.by_callees`) only when every callee is a Call. At N < G the GEMM's body is the K/G tile-scale products (`F32Mul_v3`) and N coordinates (`ScaledMmFp8BlockCoordinateGivenScale_v2{K,G}`), and the coordinate was not a Call, so the three wide GEMMs failed `partition/unpartitioned`.

Three files change; no Definition, no `checks.py` rule, no partition file:

- `targets.CALLEE_CALLS` gains `ScaledMmFp8BlockCoordinateGivenScale_v2: (ScaledMmFp8BlockSharedScale_v2,)`, and the existing `F32Mul_v3` entry gains `ScaledMmFp8BlockSharedScale_v2` beside `RMSNormFusedCudaVector8_v4`. The walk test asks for both: with the coordinate's entry removed, its failure lists the wanted list, which holds `'F32Mul_v3': ('RMSNormFusedCudaVector8_v4', 'ScaledMmFp8BlockSharedScale_v2')` and the coordinate.
- The Boolean FP8 sidecar (`boolean_fp8.circuit-check.py`) holds two `RANGES` rules, one per family (a family may have only one rule, and each spot must be on the rule's grid with spots at both ends of each range):
  - `ScaledMmFp8BlockSharedScale_v2{K=256..6144 step 256,N=4..8 step 4,G=128}`, spots `K=256,N=4`, `K=6144,N=4`, `K=2048,N=4`, `K=1536,N=8`. It covers 48 ids: K = 256, 512, …, 6,144 (24 values) × N = 4, 8.
  - `ScaledMmFp8BlockCoordinateGivenScale_v2{K=256..6144 step 256,G=128}`, spots K = 256, 1,536, 2,048, 6,144. It covers 24 ids: K = 256, 512, …, 6,144.
  - Step 256 is the widest step that reaches every served K from 256, and K = 256 is the lowest served K, so neither rule can be narrower in K and keep every served size as a spot. N's range is 4..8 step 4, just the two served N. A GEMM rule from K = 1,536 by step 512 would cover 20 ids, but would leave the served K = 256, N = 4 GEMM (756,596 gates, over `MAX_GATES`, so not in `--all`) checked nowhere.
- `boolean_fp8.circuit-check.json` gains one pin, written by circuit-check's update pass: `ScaledMmFp8BlockCoordinateGivenScale_v2{K=256,G=128}`'s whole lowering, 61,747 ANDs.

## circuit-check reports

From the confirming pass at `c00bf07b2258` (`--ranges ScaledMmFp8BlockSharedScale_v2 ScaledMmFp8BlockCoordinateGivenScale_v2`, no `--as-call`, no cache), run `r20261112-125806-3bee`: 10 targets, 0 failures, both rules hold, files changed 0 (no pins diff). Vectors are the word reference's (20 edge, 1,004 random per target unless said).

| target | gates | partition | redundant gates | dead gates | vectors | peak, seconds |
| --- | --- | --- | --- | --- | --- | --- |
| GEMM `{K=6144,N=4,G=128}` | 18,141,744 | by callees: `F32Mul_v3` (2,211 units, Q_word 2, Q_call 1 unit / 32 bits) and coordinate `{K=6144}` (1 unit, 16 bits); passes | not checked (over its budget) | 0 | 1,024, 0 mismatches; not lowered whole (about 6,177,844 ANDs) | 4.09 GB, 183.5 s |
| GEMM `{K=2048,N=4,G=128}` | 6,047,728 | by callees `F32Mul_v3` and coordinate `{K=2048}`; passes | not checked | 0 | 1,024, 0 mismatches; not lowered whole (about 2,059,252 ANDs) | 1.83 GB, 65.0 s |
| GEMM `{K=1536,N=8,G=128}` | 8,985,672 | by callees `F32Mul_v3` and coordinate `{K=1536}`; passes | not checked | 0 | 1,024, 0 mismatches; not lowered whole (about 3,059,696 ANDs) | 2.31 GB, 68.6 s |
| GEMM `{K=256,N=4,G=128}` | 756,596 | its own cut, 4,544 units per call, 30,768 gates recomputed across units (Q_word v2); passes | 2,556 | 0 | 1,024, 0 mismatches; not lowered whole (about 257,368 ANDs) | 1.2 GB, 24.5 s |
| coordinate `{K=256,G=128}` | 185,554 | 1 unit, applies (Call), Q_word 2; passes | 639 | 0 | lowered whole, 61,747 ANDs (pinned), 4 compared, 0 mismatches; word 1,024, 0 mismatches | 0.82 GB, 10.6 s |
| coordinate `{K=1536,G=128}` | 1,112,424 | 1 unit; passes | 3,009 | 0 | word 1,024, 0 mismatches | 1.42 GB, 37.2 s |
| coordinate `{K=2048,G=128}` | 1,483,172 | 1 unit; passes | 3,957 | 0 | word 1,024, 0 mismatches | 1.81 GB, 39.4 s |
| coordinate `{K=6144,G=128}` | 4,449,156 | 1 unit (under the 5,000,000-gate budget); passes | 11,541 | 0 | word 1,024, 0 mismatches; not lowered whole (about 1,515,301 ANDs) | 3.94 GB, 115.4 s |

No coordinate is over a budget. The pass also checked the two coordinate targets already in the catalog, now as Calls: `{K=128,G=128}` (92,867 gates, 402 redundant, 0 dead, 6.4 s) and `{K=64,G=32}` (65,446 gates, 405 redundant, 0 dead, 4.9 s); both pass.

## Runs

@@RUNS@@

## What the change adds to `check`

@@CHECK@@

## Not run

@@NOTRUN@@
