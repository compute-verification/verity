# Expressibility census — `E6-B0-256x63-r1-f299411e-t84cf6bc1-l57129503` (E3's `census.py`, E7's `can_express`; run by E6, 2026-09-13 01:38:59Z → STOPPED 02:45:14Z by D53)

**PARTIAL — MEASURED: 22.97 M of 326,735,010 VUs checked (7.03 %), 0 uncovered so far; stopped by the integrator (D53), not by a finding.**
The residue over the whole population is NOT established by this run. What IS established over the whole population (a full check, not a sample) is the
kinds table — see "Kinds table" below.

## Totals (last observed, 02:37:11Z; the shards write their JSON only at the end, so the per-shard progress lines are the record — no `census.json`)

* Population: 326,735,010 VUs, 32 kinds, `partition_digest f299411ee39f…`, population file `population.v2.json` sha256 `c9ed7a9856d3…` (1,076,252,900 B), instance `protocol_instance.json` sha256 `cd34e4493a59…`.
* Checked: **22,973,911 VUs** (sum of the 8 shards' last progress lines) in 1,954,637 rows (of 32,762,705); members evaluated exactly: 3,898,903; non-uniform rows (picks disagreed → every member evaluated): 5,637.
* Uncovered so far: **0**. Statement half derived for every checked VU; predicate half `ok` for every evaluated member.
* Shape keys seen per shard (E7's `can_express`, shape-memoised, ONE call per key): 64–68 of the 75 keys the 128×64 census found (same 32 kinds, same literal set `l57129503`; the remaining keys are the rare ones — `DivFull*`, `MufuSqrtFtz`, `TokenSelect` — that a 7 % pass of an `i::8` row interleave had not yet reached in every shard). `can_express` calls per shard: 89–102, ≈ 21 s each on this host.
* Mode: `--rows sampled --picks 6 --jobs 8 --tree 54e3414f`; tree `54e3414f` (+ eod/e6 `4ca0aaf1a`, no source change to the census).

| shard | VUs checked | members evaluated | rows | non-uniform | eval/s (avg incl. warm-up) | VU/s (avg) | shape keys | maxrss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0/8 | 3,585,069 | 510,127 | 265,105 | 672 | 149 | 1,050 | 66+ | 13.88 GB |
| 1/8 | 2,442,996 | 446,770 | 230,007 | 594 | 131 | 718 | 65+ | 13.66 GB |
| 2/8 | 2,217,328 | 421,217 | 220,599 | 532 | 124 | 652 | 64+ | 13.57 GB |
| 3/8 | 2,474,349 | 472,482 | 242,502 | 630 | 141 | 737 | 64+ | 14.08 GB |
| 4/8 | 2,671,908 | 515,486 | 228,225 | 1,009 | 153 | 793 | 68+ | 13.89 GB |
| 5/8 | 3,301,603 | 577,140 | 271,205 | 988 | 170 | 972 | 65+ | 13.77 GB |
| 6/8 | 3,155,962 | 498,663 | 265,127 | 594 | 146 | 924 | 64+ | 13.74 GB |
| 7/8 | 3,124,696 | 457,018 | 231,867 | 618 | 136 | 926 | 64+ | 14.18 GB |

## Measured rate and class (why it stopped)

* Pod `sc-e6-cen256` `704j8clx1vf2xg`, RunPod cpu5m **32 vCPU / 256 GB, $2.08/h**, network volume `h5uk3omqsl` at `/vol`; created 01:36:53Z, census launched 01:38:59Z, terminated 02:45:14Z (≈ $2.4).
* Warm-up ≈ 30 min per shard (the parse of the 1.08 GB population + 89–102 `can_express` calls at ≈ 21 s each; the same calls took 6–7 s on `sc-e6-pin`, cpu5m 8 vCPU).
* Steady state (02:21:50Z → 02:37:11Z, 921 s): **1.4–2.0 k VU/s per shard, ≈ 300 members evaluated/s per shard, aggregate ≈ 16 k VU/s** — one running thread per shard at 100 % (the other 31 threads are idle BLAS threads), ≈ 2× slower per core than the 128×64 shards on `sc-e6-pin` (4 k VU/s, 600 eval/s). Remaining 303.7 M VUs / 16 k VU/s ≈ 5.3 h → ETA ≈ 08:00Z.
* Memory: 12.0–12.5 GB per shard after the parse (2.7× the 128×64 shards' 4.5–4.9 GB at the same stage: proportional to the rows), growing ≈ 3.5 KB per row processed (128×64 MEASURED: 4.7 → 12.4 GB over 1.95 M rows/shard); 105.4 GB total at 02:37Z, projected ≈ 216 GB at the end for 8 shards; 12 shards would not fit the 256 GB cgroup. The shard count is memory-bound, so the census's cost is ≈ fixed at this size: ≈ 6.5 h × $2.08 ≈ **$13.5**, the whole $18 scaling line, and past D51's 06:00Z gate. **D53 (02:46Z): stopped at 7 %; the 256×64 protocol run is not gated on it.**
* The census's memory per processed row (bookkeeping, not the parse) is the thing to fix before a census at this size or larger — named for E3; the same columnar fix as the plan phase's (E6 22:25Z).

## Kinds table — FULL check, not a sample

**The 256×64 population's kinds table equals the pinned rule's 32 kinds, over every row.** At the freeze (22:57:07Z → 23:00:10Z, `freeze rc=0`, `BUILD.log`),
`population_v3.compare(expected, population.v2.json, self_check='full')` scanned all 32,762,705 rows of the population file (`c9ed7a9856d3…`) and compared the
per-kind unit counts of the rows with the rule's count per kind (and the rows' tiling of [0, N_V), the kind entries, the declared widths) → `[]` (no
disagreement); `legacy_minus_inputs_v2.check == []`. That is the instance's `partition.verifier_derived` line (`protocol_instance.json` `cd34e4493a59…`), and
`partition.vu_count` = 326,735,010 is labelled MEASURED on that basis. The 32 kinds with their N_V (the rule's = the rows'): `AmpereBF16TcDot16_v1|batch`
57,715,200 · `GemmCoordinate_v1{K=576}|batch` 47,244,288 · `Bf16ToF32_v1|batch` 22,123,008 · `F32Mul_v1|batch` 22,123,008 · `F2fpBf16_v1|batch` 16,573,824 ·
`SiluMulBf16_v1|batch` 14,699,520 · `MufuEx2Ftz_v1|primitive` 13,952,520 · `DotBf16_v1{K=64}|call` 13,780,800 · `F2fpBf16_v1|primitive` 13,780,800 ·
`F32FmaSubFtz_v1|primitive` 13,780,800 · `F32Max_v1|primitive` 13,694,670 · `F32AddFtz_v1|primitive` 13,023,990 · `F32MulFtz_v1|primitive` 11,419,650 ·
`F32Add_v1|primitive` 11,083,196 · `F32Mul_v1|primitive` 10,901,047 · `Bf16AddF2fp_v1|batch` 10,877,760 · `F32MulFtz_v1|batch` 5,512,320 ·
`GemmCoordinate_v1{K=1536}|batch` 5,512,320 · `RopeOutAdd_v1|batch` 3,674,880 · `RopeOut_v1|batch` 3,674,880 · `F32FmaFtz_v1|primitive` 686,880 ·
`GuardNegInfZero_v1|primitive` 257,850 · `GatherBf16x49152_v1|batch` 183,744 · `F32SubFtz_v1|primitive` 171,720 · `F32Fma_v1|primitive` 161,095 ·
`Fa2InvSum_v1|primitive` 86,130 · `F32Div_v1|primitive` 18,885 · `RsqrtApprox_v1|primitive` 18,885 · `DivFullRcp_v1|primitive` 638 ·
`DivFullScaleA_v1|primitive` 319 · `MufuSqrtFtz_v1|primitive` 319 · `TokenSelect_v1|whole-scan` 64. Sum 326,735,010. Kind-table digest `12906552a6e102f2…`
(`veritor/a1/rule-population-computed-kind-table/v1`).

Every one of the 32 kinds is a kind the 128×64 census (RESIDUE 0 over 160,455,774 VUs, 75 shape keys) found `ok`; the 256×64 shapes differ from 128×64 only in
KV length (more FA2 blocks of the same shape keys), which is what the 7 % pass observed: 64–68 of the same 75 keys, no new key, no uncovered VU.

## Status line for E3's table (D53)

`census: partial 7 % (23.0 M of 326,735,010 VUs, 0 uncovered); kinds table == pinned rule (32 kinds, FULL check at the freeze); protocol run not gated on it`
