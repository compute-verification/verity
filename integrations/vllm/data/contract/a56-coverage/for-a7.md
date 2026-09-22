# For A7 — O1 vs O2 comparison inputs, what is known on 2026-09-10 (A5/6, 23:25Z; mainline `1ca0a9e8`)

Per the user directive of 16:04 PT §2/§3 (`docs/handoffs-in/cursor-directive-2026-09-10-1604.md`): O1 vs O2 is **not** chosen from today's numbers. This file gives A7 the populations and unit costs per option so A7 can optimise \(q, s\) per option at the same δ and exemption policy and price both in dollars. **Every number carries a label**: MEASURED (with where/how), MODELED (with the rule), or NOT MEASURED. Nothing here is a recommendation.

Options as defined in `report.md` §(l):
- **O1 — head VU partition (PROPOSED partition change + approval of the fa2 guest `33cba058`, both undecided):** one `AttentionHead_v1{T,D}|call` unit per (position, layer, head); inputs = the committed q row, T K rows, T V rows (boundary rows today's REC records commit); output = the committed D-word `attn/port=0` row. Zero extra commitment. Checker = fa2 guest (UNAPPROVED; its refusal by the approved host is enforcement evidence only, not conformance evidence for the partition).
- **O2 — interior commit + scalar VUs (as-built partition; PROPOSED `attn-interior/v0` layout):** the as-built per-primitive attention VUs (8 dominant kinds + 7 small ones) become buildable once the record commits the attention interior (scores, `qk_acc`, p/pb, m/l/o); checkers: 3 of the 8 kinds on the approved guest `9fd00e60` today (`tc_dot16`, `DotBf16`, `f2fp_bf16`), the FTZ/MUFU five only in the fa2 gate set (P1) — so O2 also needs either fa2 approval or an approved-guest extension for those five (`report.md` §(g) P1, §(l) O3/O4).

A third column A7 may want: **"norm interior"** — see §4.

## 1. VU population per RU (counts, widths, gates) — per option, per workload

Source of counts: `coverage.json` (`configurations.*.counts_per_RU`): 6500×1599 is **MODELED** (`census_v2.small_scaling.terms(6500, 1599)` on D4's `expected_work.json`), 32×15 is **MEASURED** by census (D2 `census_small.json`, partition `56ebcdf5…`). Head-unit counts = P·NL·NH = the `Fa2InvSum` count (one epilogue per head call). Widths from the record's kind table (`chronology/selected.json#kind`: `out_bits`, `size` = gates per VU); every scalar VU has exactly **one output word**, the composites more (`DotBf16{K=64}` 5 gates / 1 output word; `GemmCoordinate{K}` K/16+1 gates / 1 bf16 word). Full per-kind tables: `for-a7-populations.json` and the appendix below.

| | 6500×1599 (per RU; MODELED) | 32×15 (per RU; MEASURED census) |
|---|---|---|
| N_V as built, computed VUs = **O2 population** | **102 894 912 400** | **31 157 512** |
| attention body (15 kinds; removed by O1) | 98 386 178 040 (95.62 %) | 4 300 560 (13.80 %) |
| O1 head units replacing them | **2 186 730** (÷ 44 992) | **12 690** (÷ 339) |
| **O1 population** | **4 510 921 090** | **26 869 642** |
| CK-ELEM four (buildable; both options) | 835 960 896 (0.81 %) | 4 855 104 (15.58 %) |
| rest / norm interior (both options; see §4) | 2 334 573 784 (2.27 %) | 13 905 976 (44.63 %) |
| GemmCoordinate (covered, approved; both options) | 1 338 199 680 (1.30 %) | 8 095 872 (25.98 %) |
| prescribed inputs `Input\|batch` — **not VUs, not sampled** (Amendment A1: prescribed values; absent from the sampled computed population) | 135 045 796 | 135 039 328 |

Rev 2 (23:45Z): the earlier revision counted the `Input|batch` rows in N_V (103 029 958 196 / 166 196 840) and in the rest bucket (2 469 619 580 / 148 945 304, "89.62 %" at 32×15); A7's `option_compare.py#consistency` reconciles the computed figure to A1b's (32×15: O2 − Inputs − (ArgmaxStep − TokenSelect) = 30 371 112). Every other cell and label is unchanged; shares are of the computed N_V. (`head_vu_consequence.md` / `report.md` §(k) still quote the with-inputs populations 103 029 958 196 → 4 645 966 886 and 166 196 840 → 161 908 970; their n_R / exposure figures count unsupported VUs only and are unaffected.)

**Widths / output words (the integrity-bound inputs):**
- O2 attention VUs: one output word each — `AmpereBF16TcDot16|batch`, `F32*Ftz|primitive`, `MufuEx2Ftz`, `F32Max`, `DotBf16{K=64}` (5 gates), consts, `GuardNegInfZero`, `Fa2InvSum`: **32-bit**; `F2fpBf16|primitive`, `Const16`: **16-bit**. Output words per RU = VU count (one each): 98.39 G words at 6500×1599 (≈ 34.5 G × 32 b + 8.86 G × 16 b + …).
- O1 head VU: **D output words of 16 bits** per unit (D = 64 at B0/TINY/32×15; the 6500×1599 workload's D is the record-6k model's — take it from `expected_work.json`, not restated here); inputs (2T+1)·D bf16 words, T = t+1 keys at position t (T ∈ [1, P], mean ≈ P/2: ≈ 800 at 6500×1599, ≈ 16 at 32×15). Output words per RU = 2 186 730 × D (= 279.9 M at D=128; 140.0 M at D=64), all 16-bit. A7: the coarser unit changes the population **and** the per-unit output width — holding \(s\) fixed does not hold security fixed (user §2).
- The head unit's kind id carries T: `AttentionHead_v1{T=<t+1>,D=<D>}|call` — P distinct kinds per (layer, head); the kind program digest chains `AttnBlock<D,64,NVIS,FIRST>` blocks (last → first) + `Fa2Epilogue<D>` (`attn_head_builder.head_subcircuit_digest`).

## 2. Unit prove cost per VU kind — MEASURED or NOT MEASURED

All MEASURED rows: pod `ctr-a56-cpu-1` (RunPod SECURE, cpu3m **16 vCPU / 128 GB, $0.88/h**, x86_64, SP1 6.4.0, `SP1_PROVER=cpu`), one batch per prover process, wall seconds of the prove step (`pod_timings.md`, raw `pod/records.jsonl`; ‡ = reconstructed from `prove.log`, to be rerun before it supports a chosen configuration — user §5). **Dollars = wall s / 3600 × $0.88** (the whole 16-vCPU box for the duration; the prover's thread utilisation was not sampled, so CPU-seconds are bounded above by 16 × wall and are not claimed). Prover RSS 19–45 GB for fa2 heads (observed), not recorded per row.

| option | VU kind | batch | mode | prove wall s | $ per proof (at $0.88/h) | proof B | verify s | label |
|---|---|---|---|---|---|---|---|---|
| O2 (approved 9fd00e60) | `DotBf16_v1{K=128}\|call` n=1 | `dot-bf16-K128_n1` | core | **73.56** | 0.0180 | 2 831 128 | 0.108 (verify-many) | MEASURED |
| O2 (approved) | `AmpereBF16TcDot16_v1\|batch` ×4 n=1 | `tc-dot16-pv_n1` | core | **55.02** | 0.0134 | 2 814 016 | 0.107 | MEASURED |
| O2 (approved) | `F2fpBf16_v1\|primitive` n=1 | `f2fp-bf16_n1` | core | **33.71** | 0.0082 | 2 797 496 | 0.105 | MEASURED |
| O2 (approved) | same three | | compressed | 125.41 / 102.97 / 87.13 | 0.0307 / 0.0252 / 0.0213 | 1 272 602 each | 0.044 | MEASURED ‡ (reconstructed rows; rerun pending) |
| O2 (fa2 gate set) | FTZ five (`F32MulFtz`, `MufuEx2Ftz`, `F32FmaSubFtz`, `F32Max`, `F32AddFtz`) as scalar VUs | — | — | — | — | — | — | **NOT MEASURED** (no approved checker; fa2's `In` gate is 16-bit — a per-primitive f32 VU is not expressible in the fa2 statement format, P1) |
| O2 (approved) | n=64 batches of the three kinds (`*_n64`, built) | `dot-bf16-K128_n64` … | core | — | — | — | — | **NOT MEASURED** (pass 3 killed at pod termination; the amortisation of the ≈ 30 s approved floor is the open number — user §3 first bullet) |
| O1 (fa2 33cba058, UNAPPROVED) | `AttentionHead_v1{T=9,D=128}\|call` | `head_T9_h5` (real W11 rows) | core | **623.60** | 0.152 | 18 046 614 | 0.722 (same host); REFUSED by approved host in 2–6 ms | MEASURED (component timing; enforcement evidence for the refusal only) |
| O1 (fa2) | `AttentionHead_v1{T=65,D=128}\|call` | `head_T65_h5` | core | **860.04** | 0.210 | 22 730 124 | 0.838 | MEASURED |
| O1 (fa2) | same two | | compressed | 809.86 / 1 122.24 | 0.198 / 0.274 | 1 272 602 | 0.044 | MEASURED ‡ |
| O1 (fa2) | heads at the workload's T (mean ≈ 800, max 1599 at 6500×1599) | — | — | — | — | — | — | **NOT MEASURED.** Fit from the two points (F1): ≈ **560 s fixed + ≈ 15 s per M cycles**; cycles grow ≈ linearly in T (3.59 M at T=9, 18.95 M at T=65 → ≈ 0.27 M/key + 1.1 M) so T=800 ≈ 220 M cycles ≈ 560 + 3 300 s ≈ **3 900 s (≈ $0.95) per head proof, EXTRAPOLATED by rule** — 65 min per proof, > 1 GB witness; shard count and RSS at that size unknown. User §2: these T=9/65 numbers do not establish the large-workload cost. |
| O1 (fa2, table-free) | head with MUFU tables as committed-table openings instead of `include_bytes!` | — | — | — | — | — | — | **NOT MEASURED** (guest change PROPOSED; F1's 560 s floor is a HYPOTHESIS attributed to the 64 MB image — the reduction needs the build + one T=9 proof) |
| both | `verify-many` per proof | | | | | | 0.044–0.108 | MEASURED |

Per-RU proof cost is A7's product: (selected units of each kind at its \(s\)) × (unit cost). Reference points from `report.md` F1: at s = 2⁻¹⁴ O1 selects E ≈ 133 head units per RU at 6500×1599 → at the MEASURED T=9/65 cost 22–33 box-hours per RU on this 16-vCPU box (≈ $20–29 at $0.88/h); at the EXTRAPOLATED T≈800 cost ≈ 133 × 3 900 s ≈ 144 h (≈ $127) per RU — the extrapolation is the number the user asked not to decide on. For O2 the same s selects E ≈ 6.0 M attention VUs per RU (98.4 G × 2⁻¹⁴) at 34–74 s each n=1 (≈ 57 000–123 000 box-hours per RU unbatched, ≈ $50k–109k at $0.88/h) — the n=64 amortisation is what makes or breaks O2's proof column, and it is NOT MEASURED.

## 3. Per-RU commit cost

| option | extra committed bytes per RU | leaves | hashing (single Xeon vCPU, sha_ni) | label |
|---|---|---|---|---|
| O1 head VU | **0** | 0 | 0 | every head input/output is a boundary row today's REC records commit (verified on the real TINY record: forced heads t=3 prefill and t=5 decode built from committed rows, `records/forced_head_run-TINY-*.json`) |
| O2 `attn-interior/v0`, packed rows | **389 GB/RU** (1.56 TB per 4 RUs) at 6500×1599 | 15 307 110 | 244 s | **MODELED**: 389 GB = census interior bytes per RU; hashing = L1/D5's MEASURED 0.5225 ns/B + 1 333 ns leaf + 1 302 ns node frames (`docs/data/hash-cost.json`). The 244 s / 1.6 GB/s figure covers **SHA-256 over the payload + leaf/node frame hashing only**; it does **not** include encoding the interior words, tree construction memory/IO, storage, or producing authenticated openings — those are the user §3 "measure, not model" items; peak memory unknown. |
| O2 one value per leaf | 389 GB payload + frames | 97 250 000 000 | 256 275 s (71.2 h) | MODELED (same inputs); what single-word openings for per-primitive VUs would want |
| O2 at 32×15 | not modeled (the census interior figure is for 6500×1599; scale by rule attention VUs × 4 B: 4.30 M × 4 B ≈ **17 MB/RU** order) | | | MODELED (rule stated; not from a census) |

## 4. The "rest" bucket — ownership and why it is unbuilt (a possible third column "norm interior")

`records/unsupported-census-{TINY,32x15}.md` (by `ProgramLocator` location): TINY 405/653 (62 %), 32×15 778/1 479 (53 %) of the unsupported *selection* are RMSNorm pieces in `pre`/`post` (`F32Mul|batch`, `Bf16ToF32|batch`, `F32Add|primitive`, `F32Mul|primitive`, `F2fpBf16|batch` outside attention), `ArgmaxStep|scan` in `final`, and gather/sampler kinds. Per the 20:40Z brief these sub-1.2 % kinds are **CK-ELEM's**; CK-ELEM has built four elementwise kinds and reports **the operands of the remaining ones are uncommitted interior words** (the RMSNorm sum-of-squares/rsqrt chain and the argmax scan state are not boundary rows) — the same situation as the attention interior (P2). So: the rest bucket is owned by CK-ELEM; its buildability depends on a commitment decision analogous to O2's, not on a builder. If A7 wants it as a third column ("norm interior"): population per RU 2 334 573 784 (2.27 %) at 6500×1599 / 13 905 976 (44.63 %) at 32×15 (computed VUs; the earlier revision's 148 945 304 / 89.62 % included the prescribed `Input|batch` rows) (`for-a7-populations.json` bucket `rest / norm interior`), checkers exist on 9fd00e60 for the f32/bf16 elementwise kinds (`elem-bf16@1`, `reduce-mufu@1` families) and unit costs are NOT MEASURED; commit bytes NOT MODELED (no census of the norm interior). At 32×15 this bucket (44.6 %) and GemmCoordinate (26.0 %), not attention (13.8 %), carry the population — A7 should not read the 6500×1599 95 % attention share into the small workload.

## 5. What changes the numbers above (open, labelled)

- **F1 (HYPOTHESIS):** the fa2 560 s floor is the 64 MB `include_bytes!` image; table-free guest NOT MEASURED.
- **F2 (fixed 23:20Z, builder-side):** `attn_gates` FIRST blocks at NVIS ∈ {3, 5, 7} now lower (single-column lane materialised as p + ZERO32, exact); 250 other block digests unchanged, 6 new — heads at t ∈ {2, 4, 6} are buildable (forced t=2 on TINY covered). No partition/record digest touched.
- **F3 (gap):** raw fa2 statements have no typed object for A2's `position_check`; O1's certificate path needs a typed head statement.
- **n=64 approved batches** (built, NOT MEASURED) — first pod item under an extension; **O2 packed path** (encode + hash + tree + store + openings, peak RSS) — second; both feed §2/§3 directly.

## Appendix — per-kind populations (from `coverage.json`; labels as in §1; rev 2: `Input|batch` on its own line, shares of the computed N_V)

### 6500x1599_P8099: N_V per RU (computed VUs) = 102,894,912,400 (source: lane/record-6k:out/record-6k/expected_work.json (D4 RECORD-6K; MODELED by census_v2.small_scaling.te)

| bucket | VUs per RU (O2 / as-built) | share | O1 (head VU) |
|---|---|---|---|
| attention body (15 kinds) | 98,386,178,040 | 95.62 % | **2,186,730 head units** (P·NL·NH; ÷ 44,992) |
| CK-ELEM four | 835,960,896 | 0.81 % | unchanged |
| rest / norm interior (RMSNorm pieces, consts, Bf16ToF32, F32Mul, ArgmaxStep, gather, sampler) | 2,334,573,784 | 2.27 % | unchanged |
| GemmCoordinate (covered, approved) | 1,338,199,680 | 1.30 % | unchanged |
| **population per RU (computed VUs)** | **102,894,912,400** | | **4,510,921,090** |
| prescribed inputs `Input\|batch` (not VUs, not sampled; Amendment A1) | 135,045,796 | — | — (same) |

Per kind (`share` = of the computed N_V; widths: `out_bits` = bits of each output word, `gates` = gates per VU; every VU has one output word except the composites):

| kind | count per RU | share | out_bits | gates/VU | bucket |
|---|---|---|---|---|---|
| `AmpereBF16TcDot16_v1|batch` | 35,490,648,960 | 34.4921 % | 32 | 1 | attention body |
| `F32MulFtz_v1|primitive` | 9,062,052,930 | 8.8071 % | None | None | attention body |
| `MufuEx2Ftz_v1|primitive` | 8,993,527,200 | 8.7405 % | 32 | 1 | attention body |
| `DotBf16_v1{K=64}|call` | 8,856,256,500 | 8.6071 % | 32 | 5 | attention body |
| `F2fpBf16_v1|primitive` | 8,856,256,500 | 8.6071 % | 16 | 1 | attention body |
| `F32FmaSubFtz_v1|primitive` | 8,856,256,500 | 8.6071 % | 32 | 1 | attention body |
| `F32Max_v1|primitive` | 8,854,069,770 | 8.6050 % | 32 | 1 | attention body |
| `F32AddFtz_v1|primitive` | 8,305,398,450 | 8.0717 % | 32 | 1 | attention body |
| `GemmCoordinate_v1{K=576}|batch` | 1,198,248,960 | 1.1645 % | 16 | 38 | covered (GemmCoordinate, approved) |
| `Bf16ToF32_v1|batch` | 561,646,080 | 0.5458 % | 32 | 1 | rest / norm interior |
| `F32Mul_v1|batch` | 561,646,080 | 0.5458 % | 32 | 1 | rest / norm interior |
| `F32FmaFtz_v1|primitive` | 549,082,800 | 0.5336 % | 32 | 1 | attention body |
| `F2fpBf16_v1|batch` | 420,773,760 | 0.4089 % | 16 | 1 | rest / norm interior |
| `SiluMulBf16_v1|batch` | 373,201,920 | 0.3627 % | 16 | 1 | CK-ELEM four |
| `F32Add_v1|primitive` | 281,373,772 | 0.2735 % | 32 | 1 | rest / norm interior |
| `F32Mul_v1|primitive` | 276,749,243 | 0.2690 % | 32 | 1 | rest / norm interior |
| `Bf16AddF2fp_v1|batch` | 276,158,016 | 0.2684 % | 16 | 1 | CK-ELEM four |
| `Const32[0x00000000]_v1|primitive` | 143,012,173 | 0.1390 % | 32 | 1 | rest / norm interior |
| `F32MulFtz_v1|batch` | 139,950,720 | 0.1360 % | 32 | 1 | attention body |
| `GemmCoordinate_v1{K=1536}|batch` | 139,950,720 | 0.1360 % | 16 | 98 | covered (GemmCoordinate, approved) |
| `Const32[0x3e38aa3b]_v1|primitive` | 139,457,430 | 0.1355 % | 32 | 1 | attention body |
| `GuardNegInfZero_v1|primitive` | 139,457,430 | 0.1355 % | 32 | 1 | attention body |
| `F32SubFtz_v1|primitive` | 137,270,700 | 0.1334 % | 32 | 1 | attention body |
| `Input|batch` | 135,045,796 | — | None | None | prescribed input (not a VU, not sampled; Amendment A1) |
| `RopeOutAdd_v1|batch` | 93,300,480 | 0.0907 % | 16 | 1 | CK-ELEM four |
| `RopeOut_v1|batch` | 93,300,480 | 0.0907 % | 16 | 1 | CK-ELEM four |
| `ArgmaxStep_v1|scan` | 78,641,600 | 0.0764 % | 112 | 5 | rest / norm interior |
| `GatherBf16x49152_v1|batch` | 4,665,024 | 0.0045 % | 16 | 1 | rest / norm interior |
| `Const16[0x0000]_v1|primitive` | 4,305,420 | 0.0042 % | 16 | 1 | attention body |
| `F32Fma_v1|primitive` | 4,089,995 | 0.0040 % | None | None | rest / norm interior |
| `Fa2InvSum_v1|primitive` | 2,186,730 | 0.0021 % | 32 | 1 | attention body |
| `Const32[0x3727c5ac]_v1|primitive` | 487,540 | 0.0005 % | None | None | rest / norm interior |
| `Const32[0x44100000]_v1|primitive` | 487,540 | 0.0005 % | None | None | rest / norm interior |
| `F32Div_v1|primitive` | 479,441 | 0.0005 % | None | None | rest / norm interior |
| `RsqrtApprox_v1|primitive` | 479,441 | 0.0005 % | None | None | rest / norm interior |
| `DivFullRcp_v1|primitive` | 16,198 | 0.0000 % | None | None | rest / norm interior |
| `Const32[0x3f800000]_v1|primitive` | 8,099 | 0.0000 % | None | None | rest / norm interior |
| `DivFullScaleA_v1|primitive` | 8,099 | 0.0000 % | None | None | rest / norm interior |
| `MufuSqrtFtz_v1|primitive` | 8,099 | 0.0000 % | None | None | rest / norm interior |
| `Const32[0x00000001]_v1|primitive` | 1,600 | 0.0000 % | None | None | rest / norm interior |

### 32x15: N_V per RU (computed VUs) = 31,157,512 (source: lane/partition:out/census/census_small.json (D2 census_v2, SMALL B0 Serve_v1{LP=32,STEPS=15}, partition 56ebcdf5...))

| bucket | VUs per RU (O2 / as-built) | share | O1 (head VU) |
|---|---|---|---|
| attention body (13 kinds) | 4,300,560 | 13.80 % | **12,690 head units** (P·NL·NH; ÷ 338) |
| CK-ELEM four | 4,855,104 | 15.58 % | unchanged |
| rest / norm interior (RMSNorm pieces, consts, Bf16ToF32, F32Mul, ArgmaxStep, gather, sampler) | 13,905,976 | 44.63 % | unchanged |
| GemmCoordinate (covered, approved) | 8,095,872 | 25.98 % | unchanged |
| **population per RU (computed VUs)** | **31,157,512** | | **26,869,642** |
| prescribed inputs `Input\|batch` (not VUs, not sampled; Amendment A1) | 135,039,328 | — | — (same) |

Per kind (`share` = of the computed N_V; widths: `out_bits` = bits of each output word, `gates` = gates per VU; every VU has one output word except the composites):

| kind | count per RU | share | out_bits | gates/VU | bucket |
|---|---|---|---|---|---|
| `Input|batch` | 135,039,328 | — | None | None | prescribed input (not a VU, not sampled; Amendment A1) |
| `GemmCoordinate_v1{K=576}|batch` | 7,283,712 | 23.3771 % | 16 | 38 | covered (GemmCoordinate, approved) |
| `Bf16ToF32_v1|batch` | 3,267,072 | 10.4857 % | 32 | 1 | rest / norm interior |
| `F32Mul_v1|batch` | 3,267,072 | 10.4857 % | 32 | 1 | rest / norm interior |
| `F2fpBf16_v1|batch` | 2,445,696 | 7.8495 % | 16 | 1 | rest / norm interior |
| `SiluMulBf16_v1|batch` | 2,165,760 | 6.9510 % | 16 | 1 | CK-ELEM four |
| `F32Add_v1|primitive` | 1,636,732 | 5.2531 % | 32 | 1 | rest / norm interior |
| `F32Mul_v1|primitive` | 1,609,895 | 5.1670 % | 32 | 1 | rest / norm interior |
| `AmpereBF16TcDot16_v1|batch` | 1,607,040 | 5.1578 % | 32 | 1 | attention body |
| `Bf16AddF2fp_v1|batch` | 1,606,464 | 5.1559 % | 16 | 1 | CK-ELEM four |
| `Const32[0x00000000]_v1|primitive` | 830,785 | 2.6664 % | 32 | 1 | rest / norm interior |
| `F32MulFtz_v1|batch` | 812,160 | 2.6066 % | 32 | 1 | attention body |
| `GemmCoordinate_v1{K=1536}|batch` | 812,160 | 2.6066 % | 16 | 98 | covered (GemmCoordinate, approved) |
| `ArgmaxStep_v1|scan` | 786,416 | 2.5240 % | 112 | 5 | rest / norm interior |
| `RopeOutAdd_v1|batch` | 541,440 | 1.7378 % | 16 | 1 | CK-ELEM four |
| `RopeOut_v1|batch` | 541,440 | 1.7378 % | 16 | 1 | CK-ELEM four |
| `DotBf16_v1{K=64}|call` | 304,560 | 0.9775 % | 32 | 5 | attention body |
| `F2fpBf16_v1|primitive` | 304,560 | 0.9775 % | 16 | 1 | attention body |
| `F32FmaSubFtz_v1|primitive` | 304,560 | 0.9775 % | 32 | 1 | attention body |
| `MufuEx2Ftz_v1|primitive` | 304,560 | 0.9775 % | 32 | 1 | attention body |
| `F32AddFtz_v1|primitive` | 295,110 | 0.9472 % | 32 | 1 | attention body |
| `F32Max_v1|primitive` | 291,870 | 0.9368 % | 32 | 1 | attention body |
| `GatherBf16x49152_v1|batch` | 27,072 | 0.0869 % | 16 | 1 | rest / norm interior |
| `Const16[0x0000]_v1|primitive` | 25,380 | 0.0815 % | 16 | 1 | attention body |
| `F32Fma_v1|primitive` | 23,735 | 0.0762 % | None | None | rest / norm interior |
| `Const32[0x3e38aa3b]_v1|primitive` | 12,690 | 0.0407 % | 32 | 1 | attention body |
| `F32MulFtz_v1|primitive` | 12,690 | 0.0407 % | None | None | attention body |
| `Fa2InvSum_v1|primitive` | 12,690 | 0.0407 % | 32 | 1 | attention body |
| `GuardNegInfZero_v1|primitive` | 12,690 | 0.0407 % | 32 | 1 | attention body |
| `Const32[0x3727c5ac]_v1|primitive` | 2,836 | 0.0091 % | None | None | rest / norm interior |
| `Const32[0x44100000]_v1|primitive` | 2,836 | 0.0091 % | None | None | rest / norm interior |
| `F32Div_v1|primitive` | 2,789 | 0.0090 % | None | None | rest / norm interior |
| `RsqrtApprox_v1|primitive` | 2,789 | 0.0090 % | None | None | rest / norm interior |
| `DivFullRcp_v1|primitive` | 94 | 0.0003 % | None | None | rest / norm interior |
| `Const32[0x3f800000]_v1|primitive` | 47 | 0.0002 % | None | None | rest / norm interior |
| `DivFullScaleA_v1|primitive` | 47 | 0.0002 % | None | None | rest / norm interior |
| `MufuSqrtFtz_v1|primitive` | 47 | 0.0002 % | None | None | rest / norm interior |
| `Const32[0x00000001]_v1|primitive` | 16 | 0.0001 % | None | None | rest / norm interior |
