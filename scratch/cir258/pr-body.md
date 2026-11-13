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

Each run is one `research run --on auto` in a research-pool check slot, one at a time. Wait is from queueing to taking the slot, and hold is from taking the slot to the run's end (`slot.json`, `status.json`).

- `r20261112-125302-2646`, at `5652daebfc` (vy-nebius-cpu-2, slot r1; waited about 4 s, held 12:53:31Z–12:57:11Z): circuit-check's update pass on the two families' spots with `--ranges` and `--update-pins`. Result: 10 targets, 0 failures, both rules hold, and one pin written (the K=256 coordinate). I read its `update.json`, `status.txt` and `pins.diff`, and committed that diff as `c00bf07b2258`. Cgroup peak was about 5.3 GiB.
- `r20261112-125806-3bee`, at `c00bf07b2258` (vy-nebius-cpu-2, slot r1; waited 3 s, held 12:58:32Z–13:01:56Z): the confirming pass, which wrote no pins diff. The reports above are from it.
- `r20261112-130257-171f`, at `c00bf07b2258` (vy-nebius-cpu-9, slot r7, 22 cores; waited 47 s, held 13:04:17Z–15:18:29Z; peak 71.0 GiB): it ran the two walk tests with `VERITY_CIRCUIT_CHECK_SLOW=1`.
  - Both `test_every_call_the_suite_checks_is_partitioned` and `test_a_call_whose_partition_is_unchecked_passes_call_status_to_its_callees` passed, in 4,074 s on 7 workers.
  - The second test warned about stale `CALLEE_CALLS` entries. Each one it names is passed by `AttnBlockFirst_v3`, and that list was there before this change; neither of this change's entries is among them.
  - The negative: with the coordinate's entry removed (`negative.diff`, one line), the second test failed in 3,972 s (peak 44.9 GiB). It named `{'ScaledMmFp8BlockCoordinateGivenScale_v2': ['ScaledMmFp8BlockSharedScale_v2']}` as the one family missing. Its wanted list also holds `'F32Mul_v3': ('RMSNormFusedCudaVector8_v4', 'ScaledMmFp8BlockSharedScale_v2')`.
- `r20261112-152804-9697`, with the base's three files (vy-nebius-cpu-2, slot r5, 15 cores; waited 86 s, held 15:29:55Z–18:11:15Z): `circuit-check --all`, as described below.
- `r20261112-183018-3636`, at `c00bf07b2258` (vy-nebius-cpu-14, slot r12, 16 cores; waited 19 min 37 s, held 18:50:20Z–21:51:23Z): `circuit-check --all`, as described below.
- `r20261112-215945-375d`, with the base's three files (vy-nebius-cpu-5, slot r8, 30 cores; waited 97 s, held 22:01:47Z–23:16:53Z): the `tools/verity/circuit_check` suite whole and fresh, then the two walk tests. All passed.
- @@SUITESRUN@@

## What the change adds to `check`

Nothing measurable in time or memory. The only change in what `check` checks is that the two coordinate targets already in the catalog are now checked as Calls.

- **circuit-check step** (`python -m circuit_check --all -q --jobs 16 --mem-gb 96`, no cache, as `check` runs it; `check` passes no `--ranges`, so the new spots don't enter it):
  - At the base, with the three changed files put back as at `1885cf1fe08c` (run `r20261112-152804-9697`, vy-nebius-cpu-2): 9,678 s, 17,189 targets, 0 failures. The highest target peak was 63.9 GB, and the run's memory-cgroup peak was 101.8 GiB.
  - At the head `c00bf07b2258` (run `r20261112-183018-3636`, vy-nebius-cpu-14): 10,862 s, the same 17,189 targets, 0 failures. The highest target peak was 63.94 GB, and the cgroup peak was 92.3 GiB.
  - The two runs check the same target set. Exactly two targets' results differ: `ScaledMmFp8BlockCoordinateGivenScale_v2{K=128,G=128}` (5.35 s at the base, 4.93 s at the head) and `{K=64,G=32}` (4.48 s, 5.26 s). Each goes from `partition.applies: false` to `true`; their gates, units and redundant gates are unchanged. The K=256 coordinate's new pin is not reached by `--all`.
  - The 1,184 s wall difference is the machines, not the change. The head's catalog prologue was 4,411 s against 3,834 s, and the summed target seconds were 40,760 against 35,992. That is 13 % slower spread across all 17,189 targets, while the targets the change touches differ by under 3 s each (computed from the two reports).
- **`tools/verity/circuit_check` suite**, whole and fresh, without the opt-in tests: 397 s at the base (run `r20261112-215945-375d`, peak 22.4 GiB) and @@SUITEHEAD@@ at the head (run `r20261112-232130-c3ca`).
- **The two opt-in walk tests**, which run only in a cold check: 4,104 s at the base (run `r20261112-215945-375d`, peak 73.8 GiB) and 4,075 s at the head (run `r20261112-130257-171f`, peak 71.0 GiB).

## Not run

- `verity check` whole, and the verified queue's candidate check. The coordinator runs the pre-mark checks.
- The slow tests of the 57 suites `--changed` picked. That tier ran `--quick`; only `repository` and `tools/verity/circuit_check` ran whole.
- `circuit-check --all` with its cache (`--cache ... --catalog-units`, as `check` runs it under `circuit_cache`). The change edits `targets.py`, which may invalidate every target's cache key once; I did not measure that one-time cost.
- The Lean check and Lean builds, since no Lean changed.
- The GEMM spots inside `check`. `check` doesn't pass `--ranges`, so the rules hold only where `circuit-check --ranges ScaledMmFp8BlockSharedScale_v2 ScaledMmFp8BlockCoordinateGivenScale_v2` is run.
