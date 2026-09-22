# L5 CENSUS — populations, widths, cuts (overnight 2026-09-09/10)

Lane L5 of the Verity overnight program. Worktree `/private/tmp/lane-census` (branch `lane/census`, off `vllm-poc` @ f97499a3).
Generators in `census_v2/` (owned); outputs in `out/census/` (owned). Labels per DIRECTIVE §1.6: **MEASURED** = read off a
built program/plan on the named machine; **DERIVED** = arithmetic from measured or pinned inputs; **MODELED** = a stated rule
applied outside its measured range; **UNAVAILABLE** = not produced, with the reason. Vocabulary per DIRECTIVE §1.4.

Times UTC. Pod: `census-cpu-1` (RunPod `5at5czgpmyzist`, cpu3m 8 vCPU / 64 GB, $0.44/h, created 06:14, watchdog terminate at 13:30).

## 0. Answers (one line each)

- **SMALL** (SmolLM2-135M B0, `Serve_v1{C=6f196795e564, LP=32, STEPS=15}`, P1/cu=whole, partition `56ebcdf5…`, MEASURED):
  N_V per request = **31,157,544** (31,157,512 relation VUs — equal to the validated record and the Table-3a census for all
  38 kinds — + 32 prompt-token input gates); shared weight input gates (global, counted once) = **135,039,296**
  (134,518,016 consumed at 16 b; 521,280 with an empty outgoing interface, §3); one-request total = **166,196,840**.
  max cut = **80 bits** (`ArgmaxStep` scan steps: 3 outgoing words; the only family > 32 bits, flagged `exception`);
  32 bits excluding it. **A_paper = 144** = 80 + ⌈log₂ 4·(V_global + M·V_r)⌉ with M = 1.3e11 (directive literal
  `80 + ⌈log₂ 4·N_V·M⌉` = 144 — no difference; the one-request-total form gives 147; excluding the exception family
  A_paper would be 96). **interior per selected RU = 97,729,136 B = 32,692,858 words** (781,833,088 bits; includes the
  outgoing words of every request-local relation VU at actual width; excludes the shared weights, the prompt gates and
  the served tokens). A_tight table §2.
- **SMALL 6.5K/1.6K** (MODELED): N_V per request = **102,919,840,857** (1.029e11; attention dominates: 8,100 positions × ⌈T/64⌉ blocks), shared weight gates 135,039,296 (5,888 unconsumed at this shape), one-request total 103,054,880,153; max cut 80 (ArgmaxStep), **A_paper = 156** (primary = literal = one-request-total form; 108 excluding the exception family); **interior per RU = 389,206,783,850 B (389.2 GB) = 103,070,765,348 words**. Rules exact on 26 B0 points (LP 2..200, up to 4 attention blocks) and the 32×15 census; §4.

- **FRONTIER (a)** (Qwen3-235B-A22B, 800K/200K, MODELED): N_V per request = **12,273,320,947,488,572** (1.2273e16; relation VUs
  12,273,320,946,688,572 = the SCALAR-VU census exactly, + 800,000 prompt-token input gates); shared weight input gates (global,
  once) = **235,093,634,560** (= the HF safetensors parameter count); max cut = **32**, no exception; **A_paper = 125**
  (`32 + ⌈log₂ 4·(V_global + M·V_r)⌉` = 32 + 93; directive literal `32 + ⌈log₂ 4·N_V·M⌉` = 125 — no difference);
  interior per selected RU (= per request) = **43,009,733,895,991,662 B (43.0 PB) = 1.2273e16 words**.
- **FRONTIER (a′)** (MODELED): N_V per request = **6,118,754,367,263,676** (0.4985× of (a)); max cut 32; **A_paper = 124**;
  interior **18,391,467,575,092,078 B (18.4 PB)** = 6.1188e15 words (0.428× of (a)).
- Every FRONTIER family has exactly one outgoing scalar of ≤ 32 bits; the router expert id is a 7-bit index unit, token
  selection an 18-bit index unit (`tokenselect:id18`), prompt tokens 18-bit index gates.

## 1. Method

### 1.1 SMALL — the validated partition, every unit counted, widths read off unit boundaries

Program `Serve_v1{C={6f196795e564},LP=32,STEPS=15}` (SmolLM2-135M, B0), adapted compactly, plan `P1/cu=whole` from
`veritor.plans.canonical(compact=True)`; partition digest `56ebcdf53f302040870776981860ffcfafd161b1b962c759df8833057aa75b9a`
asserted equal to the validated record `docs/data/rec/p1-b0-lp32s15-validated.json` (the script refuses otherwise). The plan is
166,196,840 verification units in 6,102,438 runs (`UnitRuns`); kinds = (callee | node form [| input name]); counts per kind are
exact (O(runs) arithmetic) and equal the validated record's `per_kind_fn_form` for all 38 kinds (asserted in `checks`).

Outgoing interface of a unit = `veritor.plans.boundary.hierarchical_boundary(unit).outputs` — the unit's gates consumed outside
the unit (or protected), at wire width: `out_scalars = len(S.out)`, `out_bits = S.out.bits`. This is the ACTUAL interface, not
the callee's declared return width (`population_v2` uses the declared width; the two differ for `ArgmaxStep`, §3). A boundary
costs 0.05–400 ms on B0 (gather units resolve a 49,153-leaf input; weight input gates query ~4,300 consumers), so widths are
**sampled**: kinds with < 10 k units — every unit; larger kinds — ≈ 30 k evenly spaced units per kind, within every sampled run
every unit is *attributed* a width (constant between two equal samples; a change point between two unequal samples is located
by bisection); the scan kind (`ArgmaxStep`) in "ends" mode (first, second and last member of every scan measured; the interface
changes only at the final member — verified exhaustively on TINY at V ∈ {64, 256, 1024, 4096}); Input kinds **exactly** from the
whole-node escape set (one boundary query over the whole Input node classifies every member consumed/unconsumed — the same
answer as the per-member boundary, since an Input has no operands; verified on TINY against the exhaustive per-unit census).
The exhaustive TINY census (every one of 36,318 units) and the sampled TINY census agree family by family.

κ (the cut) = the unit's S.out width. S.out is a valid cut; the kind-table narrowing `min(out_bits, reach_bits, ancestor_bits)`
has never been computed for this horizon (the existing census records `kappa: null`) and is not computable tonight at 166 M
units — DEFAULT TAKEN, conservative (never smaller than the narrowed cut).

### 1.2 SMALL 6.5K/1.6K — MODELED by exact structural rules

`census_v2/small_scaling.py`: the same program family `Serve_v1{C=6f196795e564, LP, STEPS}` built at 26 (LP, STEPS) points
(LP ∈ 2..200, STEPS ∈ 1..15; 12 of them with 2–4 attention key blocks), counts per kind exact at every point (O(runs)
arithmetic on the plan, as in §1.1). Per kind an exact rational solve of `count = Σ cᵢ·basisᵢ` over the program's structural
sums (positions P, output positions NOUT, and Σ over query rows of blocks ⌈T/64⌉, keys T, tensor-core k-steps, partial blocks,
empty softmax lanes, multi-block rows — §4), required to reproduce every point AND the held-out exact 32×15 census with a
full-rank design; widths per family from the exact census (properties of the relation, not the shape); shared weight input
gates constant (their consumed/unconsumed split by its own exact rule). A kind not of this form is UNAVAILABLE in the
scaled census, never smoothed — none is.

### 1.3 FRONTIER — clean generator, (a) and (a′)

`census_v2/frontier_census.py`: configuration from HF `Qwen/Qwen3-235B-A22B` `config.json` at revision
`8efa61729e24bd65b1d152b5ab5409052aa80e65` (read 06:43 UTC): NL 94, H 4096, NH 64, KVH 4, HD 128, E 128, top-k 8, DE 1536,
V 151,936, untied lm_head, bf16. Workload 800,000 prompt positions + 199,999 decode steps = 999,999 positions, 200,000 output
tokens. Per-position multiplicities per operator (GEMM coordinates for q/k/v/o, router logits, the 8 routed experts' gate/up
(K=4096) and down (K=1536) projections, lm_head at output positions; RMSNorm and q/k-norm as sumsq / rsqrt / scale
coordinates; RoPE coordinates; residual, SiLU·mul, MoE combine, embedding gather coordinates; router max / exp / sum / prob /
top-k id (7-bit) / top-k weight; token select 18-bit). Attention per (layer, head, query row at cache length T): ⌈T/64⌉ 64-key
blocks under the dual-chunk-attention carrier SCALAR-VU assumed (chunk 131,072, local window 4,096; ≤ 2 boundary partials per
pass counted as full blocks; one 3-way merge per row once T > 131,072) — Σ_T blocks = 7,868,595,703 per (layer, head), merges
868,927. **(a)**: per block 64 `qk` (32 b) + 1 rowmax + 1 α + 64 `p` (16 b) + 1 `l` + 128 `acc` (32 b) = 259 scalars; per merge
133; per row 1 `1/l` + 128 `out` (16 b). **(a′)**: per block 64 `qk` + 64 block-referenced `p` + 1 block max = 129 scalars
committed; per row 1 normalizer (32 b) + 128 output coordinates (16 b), each replaying the block-wise rescaled accumulation over
every block of every pass internally (no α, no per-block `l`/`acc`, no merge, no epilogue VUs). Input gates: weights (global,
bf16), prompt tokens (18-bit index), no seed (greedy). One RU per request.

## 2. SMALL census — `census_small.json`

N_V per request = 31,157,544; V_global (shared weight input gates, counted once) = 135,039,296; one-request total = 166,196,840. max cut = 80 bits (excluding exceptions: 32). A_paper = 144 (primary form max_cut + ceil(log2(4*(V_global + M*V_r))); directive literal = 144; one-request-total form = 147; difference primary−literal = 0 bit). interior per selected RU = 97,729,136 bytes = 32,692,858 words.

Widths: 138,009,022 of 166,196,840 units verified or attributed
(`provenance.widths.notes`); per-kind basis in `kinds_raw.*.width_basis` (every unit / attributed on every unit via
bisection / ENDS mode for the scan / exact whole-node escape set for the Input kinds / SCALED ESTIMATE from ≥ 10,001 sampled
runs for 17 single-width kinds — for those the per-width counts are estimates but every sample had the same width, so the
kind is reported as one family). Six kinds are not width-constant and are split into one family per width (`|out=<bits>`).
Families sorted by count; `notes` column abbreviated (full text in the JSON).

| family | scope | count (per request; global counted once) | out_scalars | out_bits | κ | exception | notes |
|---|---|---:|---:|---:|---:|---|---|
| `GemmCoordinate_v1{K=576}|batch` | per_request | 7,283,712 | 1 | 16 | 16 |  |  |
| `Bf16ToF32_v1|batch` | per_request | 3,267,072 | 1 | 32 | 32 |  |  |
| `F32Mul_v1|batch` | per_request | 3,267,072 | 1 | 32 | 32 |  |  |
| `F2fpBf16_v1|batch` | per_request | 2,445,696 | 1 | 16 | 16 |  |  |
| `SiluMulBf16_v1|batch` | per_request | 2,165,760 | 1 | 16 | 16 |  |  |
| `F32Add_v1|primitive|out=32` | per_request | 1,630,904 | 1 | 32 | 32 |  | split: width not constant within kind |
| `F32Mul_v1|primitive` | per_request | 1,609,895 | 1 | 32 | 32 |  |  |
| `AmpereBF16TcDot16_v1|batch` | per_request | 1,607,040 | 1 | 32 | 32 |  |  |
| `Bf16AddF2fp_v1|batch` | per_request | 1,606,464 | 1 | 16 | 16 |  |  |
| `Const32[0x00000000]_v1|primitive|out=32` | per_request | 818,095 | 1 | 32 | 32 |  | split: width not constant within kind |
| `F32MulFtz_v1|batch` | per_request | 812,160 | 1 | 32 | 32 |  |  |
| `GemmCoordinate_v1{K=1536}|batch|out=16` | per_request | 794,304 | 1 | 16 | 16 |  | split: width not constant within kind |
| `ArgmaxStep_v1|scan|out=80` | per_request | 786,400 | 3 | 80 | 80 | YES | split: width not constant within kind |
| `RopeOutAdd_v1|batch` | per_request | 541,440 | 1 | 16 | 16 |  |  |
| `RopeOut_v1|batch` | per_request | 541,440 | 1 | 16 | 16 |  |  |
| `DotBf16_v1{K=64}|call` | per_request | 304,560 | 1 | 32 | 32 |  |  |
| `F2fpBf16_v1|primitive` | per_request | 304,560 | 1 | 16 | 16 |  |  |
| `F32FmaSubFtz_v1|primitive` | per_request | 304,560 | 1 | 32 | 32 |  |  |
| `MufuEx2Ftz_v1|primitive` | per_request | 304,560 | 1 | 32 | 32 |  |  |
| `F32AddFtz_v1|primitive` | per_request | 295,110 | 1 | 32 | 32 |  |  |
| `F32Max_v1|primitive` | per_request | 291,870 | 1 | 32 | 32 |  |  |
| `GatherBf16x49152_v1|batch` | per_request | 27,072 | 1 | 16 | 16 |  |  |
| `Const16[0x0000]_v1|primitive|out=16` | per_request | 24,300 | 1 | 16 | 16 |  | split: width not constant within kind |
| `F32Fma_v1|primitive` | per_request | 23,735 | 1 | 32 | 32 |  |  |
| `GemmCoordinate_v1{K=1536}|batch|out=0` | per_request | 17,856 | 0 | 0 | 0 |  | split: width not constant within kind |
| `Const32[0x00000000]_v1|primitive|out=0` | per_request | 12,690 | 0 | 0 | 0 |  | split: width not constant within kind |
| `Const32[0x3e38aa3b]_v1|primitive` | per_request | 12,690 | 1 | 32 | 32 |  |  |
| `F32MulFtz_v1|primitive` | per_request | 12,690 | 1 | 32 | 32 |  |  |
| `Fa2InvSum_v1|primitive` | per_request | 12,690 | 1 | 32 | 32 |  |  |
| `GuardNegInfZero_v1|primitive` | per_request | 12,690 | 1 | 32 | 32 |  |  |
| `F32Add_v1|primitive|out=0` | per_request | 5,828 | 0 | 0 | 0 |  | split: width not constant within kind |
| `Const32[0x3727c5ac]_v1|primitive` | per_request | 2,836 | 1 | 32 | 32 |  |  |
| `Const32[0x44100000]_v1|primitive` | per_request | 2,836 | 1 | 32 | 32 |  |  |
| `F32Div_v1|primitive` | per_request | 2,789 | 1 | 32 | 32 |  |  |
| `RsqrtApprox_v1|primitive` | per_request | 2,789 | 1 | 32 | 32 |  |  |
| `Const16[0x0000]_v1|primitive|out=0` | per_request | 1,080 | 0 | 0 | 0 |  | split: width not constant within kind |
| `DivFullRcp_v1|primitive` | per_request | 94 | 1 | 32 | 32 |  |  |
| `Const32[0x3f800000]_v1|primitive` | per_request | 47 | 1 | 32 | 32 |  |  |
| `DivFullScaleA_v1|primitive` | per_request | 47 | 1 | 32 | 32 |  |  |
| `MufuSqrtFtz_v1|primitive` | per_request | 47 | 1 | 32 | 32 |  |  |
| `Input[prompt]` | per_request | 32 | 1 | 32 | 32 |  |  |
| `ArgmaxStep_v1|scan|out=32` | per_request | 16 | 1 | 32 | 32 |  | split: width not constant within kind |
| `Const32[0x00000001]_v1|primitive` | per_request | 16 | 1 | 32 | 32 |  |  |
| `Input[weights]|out=0` | global | 521,280 | 0 | 0 | 0 |  | split: width not constant within kind |
| `Input[weights]|out=16` | global | 134,518,016 | 1 | 16 | 16 |  | split: width not constant within kind |

**A_tight** (DIRECTIVE §8 rule, orchestrator ruling ii; never substituted for A_paper): `max_cut + ceil(log2(e*N/k)); N = V_global + M*V_r; k = s*(V_global + q*M*V_r)`; max_cut = 80, N = 4,050,480,720,135,039,296, M = 130,000,000,000

| q \ s | 2^-8 | 2^-9 | 2^-10 | 2^-11 | 2^-12 | 2^-13 | 2^-14 | 2^-15 | 2^-16 | 2^-17 | 2^-18 | 2^-19 | 2^-20 | 2^-21 | 2^-22 | 2^-23 | 2^-24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2^-1 | 91 | 92 | 93 | 94 | 95 | 96 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 |
| 2^-2 | 92 | 93 | 94 | 95 | 96 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 |
| 2^-3 | 93 | 94 | 95 | 96 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 |
| 2^-4 | 94 | 95 | 96 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 |
| 2^-5 | 95 | 96 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 |
| 2^-6 | 96 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 |
| 2^-7 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 |
| 2^-8 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 |
| 2^-9 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 |
| 2^-10 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 |
| 2^-11 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 |
| 2^-12 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 |
| 2^-13 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 |
| 2^-14 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 | 120 |
| 2^-15 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 | 120 | 121 |
| 2^-16 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 | 120 | 121 | 122 |



## 3. Findings

1. **`ArgmaxStep` is the max cut (80 bits) and the only exception family.** Each scan step's actual outgoing interface is
   3 words = 80 bits — the carry `(best bf16, best_index i32, i i32)` = 16 + 32 + 32 — not the 112 bits `population_v2`
   reports from the declared return type `Tuple(Tuple(BF16, I32, I32), I32)`: the step's second output is the same gate as
   `best_index` and is consumed only at the final step. 786,400 steps per request (16 scans × 49,151 steps); only the LAST
   step of each scan has a 32-bit interface (the token id; its carry is dead): 16 units. Consequence: A_paper = 144 instead of 96
   (32 + 64). MEASURED (first, second and last step of every scan; ends mode).
2. **Token ids are 32-bit containers, not 16-bit index units.** `Input[prompt]` (32 gates) and the final `ArgmaxStep` word
   are 32 bits wide on the wire; DIRECTIVE §1.5.8's index-unit width is ⌈log₂ 49,152⌉ = 16. Reported at wire width (MEASURED);
   the 16-bit form would change no headline (N_V, max cut, A unchanged; interior −512 bits).
3. **Dead outgoing interfaces (width 0) inside the validated partition.** Four relation kinds contain units whose outputs are
   consumed by nothing: `GemmCoordinate{K=1536}` 17,856 (= 31 non-output prompt positions × 576: the last layer's MLP
   down-projection at positions that feed no lm_head), `Const32[0]` 12,690 (= 47 positions × 270 heads), `F32Add` 5,828,
   `Const16[0]` 1,080 — 37,454 relation VUs (0.12 %) with `out_scalars = 0`; and 521,280 of the 135,039,296 weight input
   gates are unconsumed (0.39 %; consistent with the rotary cos/sin table rows for positions ≥ 47: (8,192 − 47) × 64 = 521,280 —
   the `weights` batch holds the checkpoint tensors and the rope tables). These VUs exist in the partition (they are counted
   in N_V) but contribute 0 bits to the interior and to any cut. MEASURED (every unit / exact escape set).
4. **A_paper forms.** Primary (ruling i) = literal = 144; the one-request-total form (⌈log₂ 4·N_V_total·M⌉) = 147 (+3 bits,
   because it multiplies the 135 M shared weight gates by M). Not a finding under ruling (i); recorded for L6.
5. **Relation-unit counts equal the validated record and the Table-3a census exactly** for all 38 kinds
   (`checks.validated_record.per_kind_counts_equal`, `checks.old_census_rot_input.agree_on_relation_units`); the old census
   has no input gates and `kappa: null` (ROT-INPUT).
6. **FRONTIER (a) equals the SCALAR-VU census family by family** (34 relation families, 12,273,320,946,688,572); the
   regenerated census adds the input gates (weights 235,093,634,560 = HF parameter count; prompt 800,000) and the (a′)
   variant (0.4985× the VUs, 0.428× the interior bytes; A_paper 124 vs 125).

## 4. SMALL 6.5K/1.6K — `census_small_6500x1600.json` (MODELED)

N_V per request = 102,919,840,857; V_global (shared weight input gates, counted once) = 135,039,296; one-request total = 103,054,880,153. max cut = 80 bits (excluding exceptions: 32). A_paper = 156 (primary form max_cut + ceil(log2(4*(V_global + M*V_r))); directive literal = 156; one-request-total form = 156; difference primary−literal = 0 bit). interior per selected RU = 389,206,783,850 bytes = 103,070,765,348 words.

**Why not a polynomial.** The registered attention body (`verity_ir.registry.b1.AttentionHead`, kBlockN = 64) visits ⌈T/64⌉
key blocks per query row and every non-first block adds rescale gates (`F32SubFtz`, `F32FmaFtz`, `F32MulFtz`) that do not
exist below 65 positions: no fit over the ≤ 47-position regime can reach 8,100 positions. The rule per kind is therefore a
rational linear combination of the program's structural sums, P = LP + STEPS positions, NOUT = STEPS + 1 output positions,
and over T = 1..P: blocks Σ⌈T/64⌉, keys ΣT, tensor-core k-steps Σ(4(⌈T/64⌉−1) + ⌈nvis/16⌉), partial blocks Σ[T mod 64 ≠ 0],
empty softmax lanes of the first-visited block (lane(c) = (c mod 8) div 2), multi-block positions Σ[T > 64]. At the target:
1 = 1, P = 8,100, NOUT = 1,601, blocks = 516,636, keys = 32,809,050, ksteps = 2,054,364, pad_blocks = 7,974, empty_lanes = 1,524, multiblock_positions = 8,036.

**Points.** 26 B0 censuses `Serve_v1{C=6f196795e564, LP, STEPS}` built on `census-cpu-1` with exact per-kind counts
(the same generator as §2; widths sampled at 3,000 per kind on 22 points, count-only on the four largest): (LP, STEPS) ∈
2×1, 2×2, 4×1, 4×2, 8×1, 4×8, 8×4, 16×2, 8×15, 12×12, 16×8, 16×15, 32×4, 32×8, 64×1, 63×3, 65×1, 72×9, 96×2, 100×5, 129×1, 128×3, 130×2, 160×8, 192×4, 200×1 — 14 single-block, 12 multi-block (2–4 blocks, P up to 201). Every rule reproduces
every point exactly (the solve is exact rational; an inconsistent kind would be UNAVAILABLE — none is), reproduces the
held-out exact 32×15 census, and the design has full rank 9/9 (every basis direction observed). The rules fitted on the
first 20 points (LP ≤ 128) predicted the six later points 129×1, 130×2, 160×8, 200×1, 192×4, 72×9 exactly — 246 kind
counts and the exact per-width families — before they were added.
Per-width families: exact per-family rules where the per-width counts are exact at every point (`ArgmaxStep` 1 token word
per scan + 49,150 80-bit steps per output; weights unconsumed = 524,288 − 64·P — the rope cos/sin table rows for positions
≥ P, 5,888 at P = 8,100); for the four relation kinds with dead units (`Const16[0]`, `Const32[0]`, `F32Add`,
`GemmCoordinate{K=1536}`) the 32×15 per-width shares are applied to the exact kind total (MODELED; ≤ 3.1 M units,
3e-5 of N_V, 0 bits of interior). Widths per family are the exact census's (properties of the relation, not the shape).
Kinds new at this shape: `F32SubFtz`, `F32FmaFtz` (the non-first-block rescale; widths MEASURED at the multi-block points).

| family | scope | count (per request; global counted once) | out_scalars | out_bits | κ | exception | notes |
|---|---|---:|---:|---:|---:|---|---|
| `AmpereBF16TcDot16_v1|batch` | per_request | 35,499,409,920 | 1 | 32 | 32 |  | rule: (17280)*ksteps |
| `F32MulFtz_v1|primitive` | per_request | 9,064,298,520 | 1 | 32 | 32 |  | rule: (-17550)*P + (17820)*blocks |
| `MufuEx2Ftz_v1|primitive` | per_request | 8,995,748,220 | 1 | 32 | 32 |  | rule: (-270)*P + (270)*blocks + (270)*keys |
| `DotBf16_v1{K=64}|call` | per_request | 8,858,443,500 | 1 | 32 | 32 |  | rule: (270)*keys |
| `F2fpBf16_v1|primitive` | per_request | 8,858,443,500 | 1 | 16 | 16 |  | rule: (270)*keys |
| `F32FmaSubFtz_v1|primitive` | per_request | 8,858,443,500 | 1 | 32 | 32 |  | rule: (270)*keys |
| `F32Max_v1|primitive` | per_request | 8,856,256,500 | 1 | 32 | 32 |  | rule: (-270)*P + (270)*keys |
| `F32AddFtz_v1|primitive` | per_request | 8,307,449,100 | 1 | 32 | 32 |  | rule: (810)*P + (-1080)*blocks + (270)*keys + (270)*empty_lanes |
| `GemmCoordinate_v1{K=576}|batch` | per_request | 1,198,436,352 | 1 | 16 | 16 |  | rule: (138240)*P + (49152)*NOUT |
| `Bf16ToF32_v1|batch` | per_request | 561,716,352 | 1 | 32 | 32 |  | rule: (69120)*P + (1152)*NOUT |
| `F32Mul_v1|batch` | per_request | 561,716,352 | 1 | 32 | 32 |  | rule: (69120)*P + (1152)*NOUT |
| `F32FmaFtz_v1|primitive` | per_request | 549,218,880 | 1 | 32 | 32 |  | rule: (-1080)*P + (1080)*blocks |
| `F2fpBf16_v1|batch` | per_request | 420,826,176 | 1 | 16 | 16 |  | rule: (51840)*P + (576)*NOUT |
| `SiluMulBf16_v1|batch` | per_request | 373,248,000 | 1 | 16 | 16 |  | rule: (46080)*P |
| `F32Add_v1|primitive|out=32` | per_request | 280,406,948 | 1 | 32 | 32 |  | rule (kind total): (34628)*P + (576)*NOUT; per-width share 0.996439 taken from the exac... |
| `F32Mul_v1|primitive` | per_request | 276,783,876 | 1 | 32 | 32 |  | rule: (34057)*P + (576)*NOUT |
| `Bf16AddF2fp_v1|batch` | per_request | 276,192,576 | 1 | 16 | 16 |  | rule: (33984)*P + (576)*NOUT |
| `Const32[0x00000000]_v1|primitive|out=32` | per_request | 140,845,043 | 1 | 32 | 32 |  | rule (kind total): (17607)*P + (1)*NOUT + (270)*empty_lanes; per-width share 0.984725 t... |
| `F32MulFtz_v1|batch` | per_request | 139,968,000 | 1 | 32 | 32 |  | rule: (17280)*P |
| `Const32[0x3e38aa3b]_v1|primitive` | per_request | 139,491,720 | 1 | 32 | 32 |  | rule: (270)*blocks |
| `GuardNegInfZero_v1|primitive` | per_request | 139,491,720 | 1 | 32 | 32 |  | rule: (270)*blocks |
| `F32SubFtz_v1|primitive` | per_request | 137,304,720 | 1 | 32 | 32 |  | rule: (-270)*P + (270)*blocks |
| `GemmCoordinate_v1{K=1536}|batch|out=16` | per_request | 136,890,689 | 1 | 16 | 16 |  | rule (kind total): (17280)*P; per-width share 0.978014 taken from the exact 32x15 censu... |
| `RopeOutAdd_v1|batch` | per_request | 93,312,000 | 1 | 16 | 16 |  | rule: (11520)*P |
| `RopeOut_v1|batch` | per_request | 93,312,000 | 1 | 16 | 16 |  | rule: (11520)*P |
| `ArgmaxStep_v1|scan|out=80` | per_request | 78,689,150 | 3 | 80 | 80 | YES | rule (per-width family, exact at every point): (49150)*NOUT |
| `GatherBf16x49152_v1|batch` | per_request | 4,665,600 | 1 | 16 | 16 |  | rule: (576)*P |
| `Const16[0x0000]_v1|primitive|out=16` | per_request | 4,122,728 | 1 | 16 | 16 |  | rule (kind total): (540)*pad_blocks; per-width share 0.957447 taken from the exact 32x1... |
| `F32Fma_v1|primitive` | per_request | 4,090,500 | 1 | 32 | 32 |  | rule: (505)*P |
| `GemmCoordinate_v1{K=1536}|batch|out=0` | per_request | 3,077,311 | 0 | 0 | 0 |  | rule (kind total): (17280)*P; per-width share 0.021986 taken from the exact 32x15 censu... |
| `Fa2InvSum_v1|primitive` | per_request | 2,187,000 | 1 | 32 | 32 |  | rule: (270)*P |
| `Const32[0x00000000]_v1|primitive|out=0` | per_request | 2,184,738 | 0 | 0 | 0 |  | rule (kind total): (17607)*P + (1)*NOUT + (270)*empty_lanes; per-width share 0.015275 t... |
| `F32Add_v1|primitive|out=0` | per_request | 1,002,028 | 0 | 0 | 0 |  | rule (kind total): (34628)*P + (576)*NOUT; per-width share 0.003561 taken from the exac... |
| `Const32[0x3727c5ac]_v1|primitive` | per_request | 487,601 | 1 | 32 | 32 |  | rule: (60)*P + (1)*NOUT |
| `Const32[0x44100000]_v1|primitive` | per_request | 487,601 | 1 | 32 | 32 |  | rule: (60)*P + (1)*NOUT |
| `F32Div_v1|primitive` | per_request | 479,501 | 1 | 32 | 32 |  | rule: (59)*P + (1)*NOUT |
| `RsqrtApprox_v1|primitive` | per_request | 479,501 | 1 | 32 | 32 |  | rule: (59)*P + (1)*NOUT |
| `Const16[0x0000]_v1|primitive|out=0` | per_request | 183,232 | 0 | 0 | 0 |  | rule (kind total): (540)*pad_blocks; per-width share 0.042553 taken from the exact 32x1... |
| `DivFullRcp_v1|primitive` | per_request | 16,200 | 1 | 32 | 32 |  | rule: (2)*P |
| `Const32[0x3f800000]_v1|primitive` | per_request | 8,100 | 1 | 32 | 32 |  | rule: (1)*P |
| `DivFullScaleA_v1|primitive` | per_request | 8,100 | 1 | 32 | 32 |  | rule: (1)*P |
| `MufuSqrtFtz_v1|primitive` | per_request | 8,100 | 1 | 32 | 32 |  | rule: (1)*P |
| `Input[prompt]` | per_request | 6,500 | 1 | 32 | 32 |  | rule: (1)*1 + (1)*P + (-1)*NOUT |
| `ArgmaxStep_v1|scan|out=32` | per_request | 1,601 | 1 | 32 | 32 |  | rule (per-width family, exact at every point): (1)*NOUT |
| `Const32[0x00000001]_v1|primitive` | per_request | 1,601 | 1 | 32 | 32 |  | rule: (1)*NOUT |
| `Input[weights]|out=0` | global | 5,888 | 0 | 0 | 0 |  | rule (per-width family, exact at every point): (524288)*1 + (-64)*P; shared weight inpu... |
| `Input[weights]|out=16` | global | 135,033,408 | 1 | 16 | 16 |  | rule (per-width family, exact at every point): (134515008)*1 + (64)*P; shared weight in... |

**Rules** (`rules` in the JSON; coefficients exact rationals):

| kind | rule (count per request) | design rank | holdout 32×15 |
|---|---|---:|---|
| `Input|batch|weights` | `(135039296)*1` | 9/9 | 135,039,296 = predicted |
| `Input|batch|weights|out=16` | `(134515008)*1 + (64)*P` | 9/9 | 134,518,016 = predicted |
| `GemmCoordinate_v1{K=576}|batch` | `(138240)*P + (49152)*NOUT` | 9/9 | 7,283,712 = predicted |
| `Bf16ToF32_v1|batch` | `(69120)*P + (1152)*NOUT` | 9/9 | 3,267,072 = predicted |
| `F32Mul_v1|batch` | `(69120)*P + (1152)*NOUT` | 9/9 | 3,267,072 = predicted |
| `F2fpBf16_v1|batch` | `(51840)*P + (576)*NOUT` | 9/9 | 2,445,696 = predicted |
| `SiluMulBf16_v1|batch` | `(46080)*P` | 9/9 | 2,165,760 = predicted |
| `F32Add_v1|primitive` | `(34628)*P + (576)*NOUT` | 9/9 | 1,636,732 = predicted |
| `F32Mul_v1|primitive` | `(34057)*P + (576)*NOUT` | 9/9 | 1,609,895 = predicted |
| `AmpereBF16TcDot16_v1|batch` | `(17280)*ksteps` | 9/9 | 1,607,040 = predicted |
| `Bf16AddF2fp_v1|batch` | `(33984)*P + (576)*NOUT` | 9/9 | 1,606,464 = predicted |
| `Const32[0x00000000]_v1|primitive` | `(17607)*P + (1)*NOUT + (270)*empty_lanes` | 9/9 | 830,785 = predicted |
| `F32MulFtz_v1|batch` | `(17280)*P` | 9/9 | 812,160 = predicted |
| `GemmCoordinate_v1{K=1536}|batch` | `(17280)*P` | 9/9 | 812,160 = predicted |
| `ArgmaxStep_v1|scan` | `(49151)*NOUT` | 9/9 | 786,416 = predicted |
| `ArgmaxStep_v1|scan|out=80` | `(49150)*NOUT` | 9/9 | 786,400 = predicted |
| `RopeOutAdd_v1|batch` | `(11520)*P` | 9/9 | 541,440 = predicted |
| `RopeOut_v1|batch` | `(11520)*P` | 9/9 | 541,440 = predicted |
| `Input|batch|weights|out=0` | `(524288)*1 + (-64)*P` | 9/9 | 521,280 = predicted |
| `DotBf16_v1{K=64}|call` | `(270)*keys` | 9/9 | 304,560 = predicted |
| `F2fpBf16_v1|primitive` | `(270)*keys` | 9/9 | 304,560 = predicted |
| `F32FmaSubFtz_v1|primitive` | `(270)*keys` | 9/9 | 304,560 = predicted |
| `MufuEx2Ftz_v1|primitive` | `(-270)*P + (270)*blocks + (270)*keys` | 9/9 | 304,560 = predicted |
| `F32AddFtz_v1|primitive` | `(810)*P + (-1080)*blocks + (270)*keys + (270)*empty_lanes` | 9/9 | 295,110 = predicted |
| `F32Max_v1|primitive` | `(-270)*P + (270)*keys` | 9/9 | 291,870 = predicted |
| `GatherBf16x49152_v1|batch` | `(576)*P` | 9/9 | 27,072 = predicted |
| `Const16[0x0000]_v1|primitive` | `(540)*pad_blocks` | 9/9 | 25,380 = predicted |
| `F32Fma_v1|primitive` | `(505)*P` | 9/9 | 23,735 = predicted |
| `Const32[0x3e38aa3b]_v1|primitive` | `(270)*blocks` | 9/9 | 12,690 = predicted |
| `F32MulFtz_v1|primitive` | `(-17550)*P + (17820)*blocks` | 9/9 | 12,690 = predicted |
| `Fa2InvSum_v1|primitive` | `(270)*P` | 9/9 | 12,690 = predicted |
| `GuardNegInfZero_v1|primitive` | `(270)*blocks` | 9/9 | 12,690 = predicted |
| `Const32[0x3727c5ac]_v1|primitive` | `(60)*P + (1)*NOUT` | 9/9 | 2,836 = predicted |
| `Const32[0x44100000]_v1|primitive` | `(60)*P + (1)*NOUT` | 9/9 | 2,836 = predicted |
| `F32Div_v1|primitive` | `(59)*P + (1)*NOUT` | 9/9 | 2,789 = predicted |
| `RsqrtApprox_v1|primitive` | `(59)*P + (1)*NOUT` | 9/9 | 2,789 = predicted |
| `DivFullRcp_v1|primitive` | `(2)*P` | 9/9 | 94 = predicted |
| `Const32[0x3f800000]_v1|primitive` | `(1)*P` | 9/9 | 47 = predicted |
| `DivFullScaleA_v1|primitive` | `(1)*P` | 9/9 | 47 = predicted |
| `MufuSqrtFtz_v1|primitive` | `(1)*P` | 9/9 | 47 = predicted |
| `Input|batch|prompt` | `(1)*1 + (1)*P + (-1)*NOUT` | 9/9 | 32 = predicted |
| `ArgmaxStep_v1|scan|out=32` | `(1)*NOUT` | 9/9 | 16 = predicted |
| `Const32[0x00000001]_v1|primitive` | `(1)*NOUT` | 9/9 | 16 = predicted |
| `F32FmaFtz_v1|primitive` | `(-1080)*P + (1080)*blocks` | 9/9 | 0 = predicted |
| `F32SubFtz_v1|primitive` | `(-270)*P + (270)*blocks` | 9/9 | 0 = predicted |

**A_tight**: `max_cut + ceil(log2(e*N/k)); N = V_global + M*V_r; k = s*(V_global + q*M*V_r)`; max_cut = 80, N = 13,379,579,311,410,135,039,296, M = 130,000,000,000

| q \ s | 2^-8 | 2^-9 | 2^-10 | 2^-11 | 2^-12 | 2^-13 | 2^-14 | 2^-15 | 2^-16 | 2^-17 | 2^-18 | 2^-19 | 2^-20 | 2^-21 | 2^-22 | 2^-23 | 2^-24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2^-1 | 91 | 92 | 93 | 94 | 95 | 96 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 |
| 2^-2 | 92 | 93 | 94 | 95 | 96 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 |
| 2^-3 | 93 | 94 | 95 | 96 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 |
| 2^-4 | 94 | 95 | 96 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 |
| 2^-5 | 95 | 96 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 |
| 2^-6 | 96 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 |
| 2^-7 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 |
| 2^-8 | 98 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 |
| 2^-9 | 99 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 |
| 2^-10 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 |
| 2^-11 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 |
| 2^-12 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 |
| 2^-13 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 |
| 2^-14 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 | 120 |
| 2^-15 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 | 120 | 121 |
| 2^-16 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 | 120 | 121 | 122 |



## 5. FRONTIER — `census_frontier.json` (+ `_a.json`, `_a_prime.json`)

### 5.1 Variant (a)

N_V per request = 12,273,320,947,488,572; V_global (shared weight input gates, counted once) = 235,093,634,560; one-request total = 12,273,556,041,123,132. max cut = 32 bits (excluding exceptions: 32). A_paper = 125 (primary form max_cut + ceil(log2(4*(V_global + M*V_r))); directive literal = 125; one-request-total form = 125; difference primary−literal = 0 bit). interior per selected RU = 43,009,733,895,991,662 bytes = 12,273,320,946,688,572 words.

Attention sums per (layer, head): blocks 7,868,595,703, merges 868,927, rows 999,999.

| family | scope | count (per request; global counted once) | out_scalars | out_bits | κ | exception | notes |
|---|---|---:|---:|---:|---:|---|---|
| `attn:acc:f32` | per_request | 6,059,196,383,903,744 | 1 | 32 | 32 |  | acc'_j = alpha*acc_j + sum_k p_k V_kj, 128 per block |
| `attn:qk:f32` | per_request | 3,029,598,191,951,872 | 1 | 32 | 32 |  | s_k = q.k_k (128-long bf16 dot, f32), 64 per block |
| `attn:p:bf16` | per_request | 3,029,598,191,951,872 | 1 | 16 | 16 |  | p_k = bf16(exp2((s_k - m_ref)*scale)); (a): m_ref = running max; (a'): m_ref = the bloc... |
| `attn:rowmax:f32` | per_request | 47,337,471,749,248 | 1 | 32 | 32 |  | m' = max(m, s_1..s_64) (running row max) |
| `attn:alpha:f32` | per_request | 47,337,471,749,248 | 1 | 32 | 32 |  | alpha = exp2((m - m')*scale) |
| `attn:l:f32` | per_request | 47,337,471,749,248 | 1 | 32 | 32 |  | l' = alpha*l + sum_k p_k (running normalizer) |
| `gemm:K4096` | per_request | 3,188,476,811,520 | 1 | 16 | 16 |  | GemmCoordinate K=4096 (q/k/v projections, router logits, expert gate+up for the 8 route... |
| `gemm:K1536` | per_request | 3,080,188,919,808 | 1 | 16 | 16 |  | GemmCoordinate K=1536 (expert down projection, 8 routed experts) |
| `elem:silu_mul:bf16` | per_request | 1,155,070,844,928 | 1 | 16 | 16 |  | bf16(silu(a_i) * b_i), 8 routed experts x DE |
| `qknorm:scale:bf16` | per_request | 818,175,181,824 | 1 | 16 | 16 |  | q/k head norm: HD per head |
| `rope:out:bf16` | per_request | 818,175,181,824 | 1 | 16 | 16 |  | o_j = bf16(x_j cos_j -/+ x_j' sin_j), HD per (layer, head) |
| `rms:scale:bf16` | per_request | 770,866,429,952 | 1 | 16 | 16 |  | y_i = bf16(x_i * r * w_i), H per row |
| `attn:out:bf16` | per_request | 770,047,229,952 | 1 | 16 | 16 |  | o_j = bf16(acc_j * r), 128 per (query, head) |
| `elem:residual:bf16` | per_request | 770,047,229,952 | 1 | 16 | 16 |  | bf16(x_i + o_i), 2 per layer x H |
| `attn:merge_acc:f32` | per_request | 669,115,498,496 | 1 | 32 | 32 |  | DCA merge: acc*_j = sum_i alpha_i acc_ij, 128 per merge |
| `gemm:K8192` | per_request | 385,023,614,976 | 1 | 16 | 16 |  | GemmCoordinate K=8192 (o_proj over NH*HD) |
| `moe:combine:bf16` | per_request | 385,023,614,976 | 1 | 16 | 16 |  | bf16(sum_{j<8} w_j y_ji), H per layer |
| `gemm:K4096:lm_head` | per_request | 30,387,200,000 | 1 | 16 | 16 |  | GemmCoordinate K=4096 (lm_head, V rows) |
| `attn:merge_alpha:f32` | per_request | 15,682,394,496 | 1 | 32 | 32 |  | DCA merge: alpha_i = exp2((m_i - m*)*scale), 3 per merge |
| `router:exp:f32` | per_request | 12,031,987,968 | 1 | 32 | 32 |  | e_i = exp(l_i - max), E per layer |
| `router:prob:f32` | per_request | 12,031,987,968 | 1 | 32 | 32 |  | p_i = e_i / sum, E per layer |
| `qknorm:sumsq:f32` | per_request | 6,391,993,608 | 1 | 32 | 32 |  | q/k head norm: ss over HD, per (layer, q head or kv head) |
| `qknorm:rsqrt:f32` | per_request | 6,391,993,608 | 1 | 32 | 32 |  | q/k head norm: rsqrt |
| `attn:invl:f32` | per_request | 6,015,993,984 | 1 | 32 | 32 |  | r = 1/l |
| `attn:merge_m:f32` | per_request | 5,227,464,832 | 1 | 32 | 32 |  | DCA 3-way merge: m* = max_i m_i |
| `attn:merge_l:f32` | per_request | 5,227,464,832 | 1 | 32 | 32 |  | DCA merge: l* = sum_i alpha_i l_i |
| `embed:gather:bf16` | per_request | 4,095,995,904 | 1 | 16 | 16 |  | e_i = Emb[id][i], H per position |
| `router:topk_id` | per_request | 751,999,248 | 1 | 7 | 7 |  | id_j = argmax of the probs minus the j earlier ids: 7-bit expert index, 8 per layer |
| `router:topk_w:f32` | per_request | 751,999,248 | 1 | 32 | 32 |  | w_j = p_{id_j} / sum_j' p_{id_j'}, 8 per layer |
| `rms:sumsq:f32` | per_request | 188,199,812 | 1 | 32 | 32 |  | ss = sum_i x_i^2 over H (2 per layer + 1 final norm at output positions) |
| `rms:rsqrt:f32` | per_request | 188,199,812 | 1 | 32 | 32 |  | r = rsqrt(ss/H + eps) |
| `router:max:f32` | per_request | 93,999,906 | 1 | 32 | 32 |  | max over E f32 logits |
| `router:sum:f32` | per_request | 93,999,906 | 1 | 32 | 32 |  | sum_i e_i |
| `Input[prompt]` | per_request | 800,000 | 1 | 18 | 18 |  | prompt token ids: 18-bit index units (ceil(log2 V)); the serving container is 32-bit |
| `tokenselect:id18` | per_request | 200,000 | 1 | 18 | 18 |  | argmax over V logits -> one token id (18-bit index unit) |
| `Input[weights]` | global | 235,093,634,560 | 1 | 16 | 16 |  | canonical shared weight-input VUs (bf16 parameters of the registered program), counted ... |

**A_tight**: `max_cut + ceil(log2(e*N/k)); N = V_global + M*V_r; k = s*(V_global + q*M*V_r)`; max_cut = 32, N = 1,595,531,723,173,514,595,093,634,560, M = 130,000,000,000

| q \ s | 2^-8 | 2^-9 | 2^-10 | 2^-11 | 2^-12 | 2^-13 | 2^-14 | 2^-15 | 2^-16 | 2^-17 | 2^-18 | 2^-19 | 2^-20 | 2^-21 | 2^-22 | 2^-23 | 2^-24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2^-1 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 |
| 2^-2 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 |
| 2^-3 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 |
| 2^-4 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 |
| 2^-5 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 |
| 2^-6 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 |
| 2^-7 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 |
| 2^-8 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 |
| 2^-9 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 |
| 2^-10 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 |
| 2^-11 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 |
| 2^-12 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 |
| 2^-13 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 |
| 2^-14 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 |
| 2^-15 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 | 73 |
| 2^-16 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 | 73 | 74 |


### 5.2 Variant (a′)

N_V per request = 6,118,754,367,263,676; V_global (shared weight input gates, counted once) = 235,093,634,560; one-request total = 6,118,989,460,898,236. max cut = 32 bits (excluding exceptions: 32). A_paper = 124 (primary form max_cut + ceil(log2(4*(V_global + M*V_r))); directive literal = 124; one-request-total form = 124; difference primary−literal = 0 bit). interior per selected RU = 18,391,467,575,092,078 bytes = 6,118,754,366,463,676 words.

Attention sums per (layer, head): blocks 7,868,595,703, merges 868,927, rows 999,999.

| family | scope | count (per request; global counted once) | out_scalars | out_bits | κ | exception | notes |
|---|---|---:|---:|---:|---:|---|---|
| `attn:qk:f32` | per_request | 3,029,598,191,951,872 | 1 | 32 | 32 |  | s_k = q.k_k (128-long bf16 dot, f32), 64 per block |
| `attn:p:bf16` | per_request | 3,029,598,191,951,872 | 1 | 16 | 16 |  | p_k = bf16(exp2((s_k - m_ref)*scale)); (a): m_ref = running max; (a'): m_ref = the bloc... |
| `attn:blockmax:f32` | per_request | 47,337,471,749,248 | 1 | 32 | 32 |  | (a') m_b = max(s_1..s_64) of the block (block-local, not running) |
| `gemm:K4096` | per_request | 3,188,476,811,520 | 1 | 16 | 16 |  | GemmCoordinate K=4096 (q/k/v projections, router logits, expert gate+up for the 8 route... |
| `gemm:K1536` | per_request | 3,080,188,919,808 | 1 | 16 | 16 |  | GemmCoordinate K=1536 (expert down projection, 8 routed experts) |
| `elem:silu_mul:bf16` | per_request | 1,155,070,844,928 | 1 | 16 | 16 |  | bf16(silu(a_i) * b_i), 8 routed experts x DE |
| `qknorm:scale:bf16` | per_request | 818,175,181,824 | 1 | 16 | 16 |  | q/k head norm: HD per head |
| `rope:out:bf16` | per_request | 818,175,181,824 | 1 | 16 | 16 |  | o_j = bf16(x_j cos_j -/+ x_j' sin_j), HD per (layer, head) |
| `rms:scale:bf16` | per_request | 770,866,429,952 | 1 | 16 | 16 |  | y_i = bf16(x_i * r * w_i), H per row |
| `attn:outcoord:bf16` | per_request | 770,047,229,952 | 1 | 16 | 16 |  | (a') o_j = bf16( sum_b exp2((m_b - m*)*scale) * sum_k p_bk V_bkj / l ): one VU per (row... |
| `elem:residual:bf16` | per_request | 770,047,229,952 | 1 | 16 | 16 |  | bf16(x_i + o_i), 2 per layer x H |
| `gemm:K8192` | per_request | 385,023,614,976 | 1 | 16 | 16 |  | GemmCoordinate K=8192 (o_proj over NH*HD) |
| `moe:combine:bf16` | per_request | 385,023,614,976 | 1 | 16 | 16 |  | bf16(sum_{j<8} w_j y_ji), H per layer |
| `gemm:K4096:lm_head` | per_request | 30,387,200,000 | 1 | 16 | 16 |  | GemmCoordinate K=4096 (lm_head, V rows) |
| `router:exp:f32` | per_request | 12,031,987,968 | 1 | 32 | 32 |  | e_i = exp(l_i - max), E per layer |
| `router:prob:f32` | per_request | 12,031,987,968 | 1 | 32 | 32 |  | p_i = e_i / sum, E per layer |
| `qknorm:sumsq:f32` | per_request | 6,391,993,608 | 1 | 32 | 32 |  | q/k head norm: ss over HD, per (layer, q head or kv head) |
| `qknorm:rsqrt:f32` | per_request | 6,391,993,608 | 1 | 32 | 32 |  | q/k head norm: rsqrt |
| `attn:rownorm:f32` | per_request | 6,015,993,984 | 1 | 32 | 32 |  | (a') l = sum_b exp2((m_b - m*)*scale) * sum_k p_bk, m* = max_b m_b; replays the block r... |
| `embed:gather:bf16` | per_request | 4,095,995,904 | 1 | 16 | 16 |  | e_i = Emb[id][i], H per position |
| `router:topk_id` | per_request | 751,999,248 | 1 | 7 | 7 |  | id_j = argmax of the probs minus the j earlier ids: 7-bit expert index, 8 per layer |
| `router:topk_w:f32` | per_request | 751,999,248 | 1 | 32 | 32 |  | w_j = p_{id_j} / sum_j' p_{id_j'}, 8 per layer |
| `rms:sumsq:f32` | per_request | 188,199,812 | 1 | 32 | 32 |  | ss = sum_i x_i^2 over H (2 per layer + 1 final norm at output positions) |
| `rms:rsqrt:f32` | per_request | 188,199,812 | 1 | 32 | 32 |  | r = rsqrt(ss/H + eps) |
| `router:max:f32` | per_request | 93,999,906 | 1 | 32 | 32 |  | max over E f32 logits |
| `router:sum:f32` | per_request | 93,999,906 | 1 | 32 | 32 |  | sum_i e_i |
| `Input[prompt]` | per_request | 800,000 | 1 | 18 | 18 |  | prompt token ids: 18-bit index units (ceil(log2 V)); the serving container is 32-bit |
| `tokenselect:id18` | per_request | 200,000 | 1 | 18 | 18 |  | argmax over V logits -> one token id (18-bit index unit) |
| `Input[weights]` | global | 235,093,634,560 | 1 | 16 | 16 |  | canonical shared weight-input VUs (bf16 parameters of the registered program), counted ... |

**A_tight**: `max_cut + ceil(log2(e*N/k)); N = V_global + M*V_r; k = s*(V_global + q*M*V_r)`; max_cut = 32, N = 795,438,067,744,278,115,093,634,560, M = 130,000,000,000

| q \ s | 2^-8 | 2^-9 | 2^-10 | 2^-11 | 2^-12 | 2^-13 | 2^-14 | 2^-15 | 2^-16 | 2^-17 | 2^-18 | 2^-19 | 2^-20 | 2^-21 | 2^-22 | 2^-23 | 2^-24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2^-1 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 |
| 2^-2 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 |
| 2^-3 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 |
| 2^-4 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 |
| 2^-5 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 |
| 2^-6 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 |
| 2^-7 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 |
| 2^-8 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 |
| 2^-9 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 |
| 2^-10 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 |
| 2^-11 | 53 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 |
| 2^-12 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 |
| 2^-13 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 |
| 2^-14 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 |
| 2^-15 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 | 73 |
| 2^-16 | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 | 73 | 74 |


### 5.3 Reconciliation with `docs/data/frontier/dc-1m-v3/populations_scalar.json`

Regenerated, not copied. Every relation family of (a) equals the old census exactly (34 families, total
12,273,320,946,688,572); the old census has no input gates (ROT-INPUT) — this census adds `Input[weights]` 235,093,634,560
(global) and `Input[prompt]` 800,000 (per request); its `reduce:tokenselect` is this census's `tokenselect:id18` (18-bit index
unit, DIRECTIVE §1.5.8; the old table carried an I32-container variant beside it). Under (a′) the attention families
`rowmax`, `alpha`, `l`, `acc`, the four `merge:*`, `invl` and `out` are replaced by `blockmax`, `rownorm`, `outcoord`
(6.15e15 fewer VUs; the naming term drops by 1 bit).

## 6. Provenance and reproduction

- SMALL: `python census_v2/small_census.py --case B0 --lp 32 --steps 15 --workers 8 --mode sample --sample-budget 30000
  --kind-budget "ArgmaxStep_v1|scan=-1" --kind-budget "Input|batch|weights=0" --kind-budget "Input|batch|prompt=0"
  --kind-budget "GatherBf16x49152_v1|batch=470" --input-exact --validated docs/data/rec/p1-b0-lp32s15-validated.json
  --old-census docs/data/table3a/B0-serve-lp32-s15.json --out out/census/census_small.json` (pod census-cpu-1, Python 3.12.14).
- SMALL scaled: `python census_v2/small_scaling.py points --points 2x1,4x1,8x1,2x2,4x2,8x4,16x2,4x8,16x8,8x15,32x4,12x12,16x15,32x8
  --out-dir out/census/scaling --workers 8 --sample-budget 3000 --extra --kind-budget "ArgmaxStep_v1|scan=-1" --kind-budget
  "Input|batch|weights=0" --kind-budget "Input|batch|prompt=0" --kind-budget "GatherBf16x49152_v1|batch=94" --input-exact`, the same
  for `64x1,65x1,63x3,96x2,100x5,128x3,129x1,130x2`, and count-only (`--sample-budget 0`, no Gather budget) for `160x8,200x1,192x4,72x9`
  (a width batch of the 160×8 point stalled > 20 min on one boundary; the count-only files' `families`/`N_V` are not meaningful,
  only their exact `kinds_raw` counts are used); then
  `python census_v2/small_scaling.py fit --points-dir out/census/scaling --exact out/census/census_small.json --out out/census/census_small_6500x1600.json`.
- FRONTIER: `python census_v2/frontier_census.py --out out/census/census_frontier.json --old docs/data/frontier/dc-1m-v3/populations_scalar.json` (laptop, 0.4 s).
- Tables in this report: `python census_v2/report_tables.py <census.json>`.
- Defaults taken: `out/census/decisions.md`. Checkpoints: `out/census/checkpoint-*.md`.

## 7. Correction (A1) — L5-A1, 2026-09-10 11:05 UTC (post-hoc; appended, earlier sections not rewritten)

**Why.** This lane started before DIRECTIVE v3 / Amendment A1 and emitted the weight input families with a forbidden scope value
(a VU class of the kind delete-list item 11 forbids), counted the FRONTIER prompt-token input gates inside N_V, and computed
A_paper / A_tight on a "weights counted once across M requests" population (the weights-counted-once form). Under A1 inputs (weights,
prompt tokens, seed) are prescribed values: gates of the circuit, **not VUs and not sampled**; they add no term to the
Bound. SMALL tonight keeps P1's per-request input units (per-RU-copy formulation; digest `56ebcdf5…` unchanged); FRONTIER
lists its inputs as `prescribed; not a VU; not sampled`. The five census JSON files were edited in place (no census
recomputation; every relation-family count, width and κ is unchanged); §§0–6 above are the pre-A1 text and stand as history.

**What changed in every JSON** (`out/census/census_small.json`, `census_small_6500x1600.json`, `census_frontier.json` incl.
its nested `variants.a` / `variants.a_prime`, `census_frontier_a.json`, `census_frontier_a_prime.json`):
`families[].scope` removed → `families[].role ∈ {relation, prescribed_input}`; `families[].sampled` added (§5 contract);
the input families carry the prescribed-input `exception` text; `N_V` restated as `N_V_relation` / `N_V_population`
(+ `primary`); `A_paper`, `A_paper_forms`, `A_tight` recomputed on the stated basis with the pre-A1 values kept under
`pre_A1`; `interior_bytes_per_RU` re-verified; the config-URL provenance key renamed `config_url`; an `a1_correction` object
records the edit. Forbidden-vocabulary grep (the six delete-list item 1 / item 11 terms named in the L5-A1 brief)
over the five JSON files: **0 occurrences** (keys and values).

### 7.1 FRONTIER — `census_frontier.json`, `_a.json`, `_a_prime.json` (MODELED)

Input families `Input[weights]` (235,093,634,560 per request — the bf16 parameter count, referenced from the registered
commitment, never recommitted) and `Input[prompt]` (800,000): `sampled: false`, `role: prescribed_input`,
`exception: "prescribed; not a VU; not sampled; enforced by statement validation"`; not in N_V, not in A, not in the Summary.
N_V = relation VUs only. M = 1.3e11 throughout.

| census | quantity | pre-A1 (§5) | A1 (now) | arithmetic |
|---|---|---:|---:|---|
| (a) | N_V | 12,273,320,947,488,572 (relation + 800,000 prompt gates) | **12,273,320,946,688,572** (relation only) | = the SCALAR-VU census total, 34 families |
| (a) | A_paper | 125 (weights-counted-once form) | **125** | 4 · 12,273,320,946,688,572 · 1.3e11 = 6.3821e27; ⌈log₂⌉ = 93; 32 + 93 = 125 |
| (a) | interior per RU | 43,009,733,895,991,662 B = 12,273,320,946,688,572 words | unchanged | words = N_V_relation exactly: weights and prompt inputs were never inside it (re-summed over the relation families; equal) |
| (a′) | N_V | 6,118,754,367,263,676 (incl. 800,000 prompt gates) | **6,118,754,366,463,676** | 27 relation families |
| (a′) | A_paper | 124 | **124** | 4 · 6,118,754,366,463,676 · 1.3e11 = 3.1818e27; ⌈log₂⌉ = 92; 32 + 92 = 124 |
| (a′) | interior per RU | 18,391,467,575,092,078 B = 6,118,754,366,463,676 words | unchanged | words = N_V_relation exactly |

A is unchanged for both variants (the 800,000 prompt gates are 6.5e-11 of N_V; the weights-counted-once term was already
below the rounding). **A_tight** (footnote form, never substituted): `max_cut + ⌈log₂(e·N/k)⌉` with N = M·N_V_relation and
k = s·q·N (every relation VU is acquired with its RU at probability q, then sampled at s; prescribed inputs add no term);
e·N/k = e/(s·q) is independent of N, so the (q, s) table is **identical to the pre-A1 table** in §5.1/§5.2 (checked row by
row: 272 cells, 43…74 bits; the old table's weights-counted-once term changed no cell). Weight *gates* still appear in
`derivation.weight_gates` as gate counts (allowed vocabulary: input gates remain gates of the circuit).

### 7.2 SMALL — `census_small.json` (MEASURED, P1 as validated) and `census_small_6500x1600.json` (MODELED)

Per-RU-copy formulation kept exactly as validated: `Input[weights]|out=16` (134,518,016), `Input[weights]|out=0` (521,280)
and `Input[prompt]` (32) are `sampled: true`, `role: prescribed_input`, `exception: "prescribed input; per-RU-copy formulation
(SMALL, tonight); enforced by statement validation"`; the forbidden scope value and the weights-counted-once naming are gone (`N_V.inputs_per_request`
keeps the counts). The previous **31,157,544 "N_V per request" included the 32 `Input[prompt]` input units** (31,157,512 relation
VUs + 32); the relation-only figure is 31,157,512, equal to the validated record and the Table-3a census.

| census | quantity | pre-A1 (§2/§4) | A1 (now) | arithmetic |
|---|---|---:|---:|---|
| 32×15 | N_V_relation | 31,157,544 (incl. 32 prompt units) | **31,157,512** | 38 relation kinds, = validated record |
| 32×15 | N_V_population | 166,196,840 ("one-request total") | **166,196,840** (primary) | 31,157,512 + 135,039,328 per-request input units = the validated partition the certified Bound runs over |
| 32×15 | A_paper (population) | 147 ("one-request-total form", not primary) | **147 (primary)** | 4 · 166,196,840 · 1.3e11 = 8.6422e19; ⌈log₂⌉ = 67; 80 + 67 = 147 |
| 32×15 | A_paper (relation) | 144 (primary, weights-counted-once form) | 144 (secondary) | 4 · 31,157,512 · 1.3e11 = 1.6202e19; ⌈log₂⌉ = 64; 80 + 64 = 144 |
| 32×15 | interior per RU | 97,729,136 B = 32,692,858 words | unchanged | relation families only; re-summed, equal |
| 6.5K/1.6K | N_V_relation | 102,919,840,857 (incl. 6,500 prompt units) | **102,919,834,357** | |
| 6.5K/1.6K | N_V_population | 103,054,880,153 | **103,054,880,153** (primary) | + 135,045,796 input units (135,039,296 weights + 6,500 prompt) |
| 6.5K/1.6K | A_paper (population) | 156 | **156 (primary)** | 4 · 103,054,880,153 · 1.3e11 = 5.3589e22; ⌈log₂⌉ = 76; 80 + 76 = 156 |
| 6.5K/1.6K | A_paper (relation) | 156 | 156 | 4 · 102,919,834,357 · 1.3e11 = 5.3518e22; ⌈log₂⌉ = 76; 80 + 76 = 156 |
| 6.5K/1.6K | interior per RU | 389,206,783,850 B = 103,070,765,348 words | unchanged | |

`A_paper_primary = N_V_population` (§1.3: A is "exactly as the certified Bound computes it", and the record's Bound runs over
the full validated population, so 147 for the exact census — +3 bits over the pre-A1 primary 144 — and 156 for the MODELED shape).
`A_paper_excluding_exceptions` (max cut 32, ArgmaxStep excluded): 99 population / 96 relation (32×15); 108 / 108 (6.5K/1.6K).
**A_tight** on the population basis (N = M·N_V_population, k = s·q·N): identical table to §2/§4 (e/(s·q) is independent of N;
the same table holds on the N_V_relation basis). The 6.5K/1.6K MODELED census carries the same edits; its `rules` are unchanged.

### 7.3 Generator gap (code not changed) and checks

- `census_v2/report_tables.py` (renderer only): ≤ 30-line label change — `role`/`sampled` columns, headline on
  `N_V_relation` / `N_V_population`; it renders the corrected JSON (the headlines in §7.1/§7.2 are its output).
- **NOT changed:** `census_v2/contract.py` (`Family.scope`, `Census.n_v()`, `a_paper_population`, `a_tight_table`),
  `small_census.py`, `small_scaling.py`, `frontier_census.py` still emit the pre-A1 scope field and the weights-counted-once forms
  — the data model, not a label; a re-run would regenerate the pre-A1 JSON. Re-running them and then `/tmp`-style post-processing
  is not a reproduction path; the corrected JSON is the deliverable. Fixing the generators is a PROPOSED follow-up (~60 lines
  across four files, same relation counts).
- `census_v2` has no unit tests; checks run on the laptop (< 1 s): every JSON parses; `families[].sampled`, `N_V`, `A_paper`,
  `A_tight`, `interior_bytes_per_RU`, `provenance{label, script, commit, hardware}` present in all five files and in the nested
  variants; `interior_bytes_per_RU.words/bits` equal the sums over the relation families; A_tight tables equal the pre-A1 tables.
- `DEFAULT TAKEN: A1 (post-hoc correction by L5-A1, 11:05 UTC)` — see `decisions.md`.
