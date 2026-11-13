# CIR-274 DENSE: the two halves together, census at 43 prompt ids, 2 outputs, 4 layers, rank 0

The branch is `cursor/circuits-cir274-families-6f04`, cut from the dense head `7ebb59cc818ec307d06944524b33f10265346a7f`.
DSAMOE's `cursor/circuits-cir274-dsa-moe-6f04` was merged into it twice:

- at `8d7502ab19f1a5a8e707ddf38bf40ce95eaf2f27`, giving `1d5bb562130fa187768d302c53531dc57694c1c4`;
- then, after DSAMOE's head moved (a test file only), at `34b868a7f23b75864d9a8f7fc850d6d479ed957e`, giving
  `b5c74a06836824b7af153d7591ad95a574d5fb85`.

Neither merge had a conflict, and no frame file changed on DSAMOE's side.

The script is `p3_census.py` here, with families `{**node_window.FAMILIES, **node_window_dense.FAMILIES,
**node_window_sparse.FAMILIES}`. The runs:

- `r20261113-022216-daa9` (source `1d5bb56213`) and `r20261113-022750-b23e` (source `b5c74a0683`) give the same
  window: digest, units and wire patterns.
- `$OUT/p3_l4_r0_frame+dense+sparse_c0.json` is kept with each run.

Every number below is read from those JSONs; [inferred] marks the rest.

## The window

- **Program:** digest `7f36f17ae005…` (the same as the dense-only census), 319 nodes, 2 steps.
- **Window:** digest `61b9f8d8516fe67f457edf6e8e52b626586d49f2e539b0eef0991e3dfd70ee961eeb2137f9aeb40781a0f7e0452b723bc9af2166335135d85750a26b22795431`.
  - 5,137,384 units in 105 templates and 13,238 wire patterns.
  - It settled and passed `check_wires` / `check_inputs` in 18.9 s, at a peak of 0.813 GiB.

| family | units | templates | wire patterns (as reader) |
|---|---|---|---|
| AllGather | 16 | 1 | 2 |
| AllReduceMnnvl | 32 | 1 | 0 |
| AllReduceRmsNormMnnvl | 488 | 4 | 38 |
| BatchedGemm | 1,081,344 | 2 | 16 |
| CublasSplitKGemmF32Out | 11,008 | 1 | 1 |
| DeepGemmFp8Linear | 3,336,284 | 8 | 60 |
| DsaLatentRow | 176 | 1 | 8 |
| DsaRMSNorm | 1,584 | 2 | 24 |
| FlashInferFp8Linear | 77,588 | 8 | 60 |
| Fp8GroupQuantUe8m0 | 396 | 2 | 4 |
| Gemm | 38,720 | 1 | 2 |
| LlBf16DotprodGemm | 256 | 1 | 1 |
| MoeExpertFp8Gemm | 361,856 | 4 | 8,264 |
| MoeExpertFp8GemmW | 152,064 | 4 | 4,200 |
| MoeRouterSigmoidBias | 44 | 1 | 2 |
| MoeSum | 1,408 | 1 | 16 |
| RMSNormCudaVector8 | 1,452 | 2 | 4 |
| RMSNormFusedCudaVector8 | 9,934 | 3 | 20 |
| ResidualAdd | 1,408 | 1 | 4 |
| RoPEHeadGptJ | 1,408 | 1 | 8 |
| SiluMul | 528 | 2 | 10 |
| SparseMlaSplit | 2,816 | 45 | 884 |
| gather | 56,574 | 15 | 5,530 |

**Still unlowered:** AllReduceMultimem 9 (piece 2), EmbeddingShard 2 (reads.md 1), TokenSelect 2 and StopCheck 1
(piece 5's).

## Every free port, with the family its rows are of

| rows of | reader port, rows of its cut | unit ports | entries |
|---|---|---|---|
| AllReduceMultimem | `RMSNormApplyCudaVector8_v1.x` | 1,376 | 1 |
| AllReduceMultimem | `RMSNormFusedApplyCudaVector8_v1.res` | 1,376 | 1 |
| AllReduceMultimem | `RMSNormFusedApplyCudaVector8_v1.x` | 9,632 | 7 |
| AllReduceMultimem | `RMSNormFusedCudaVector8_v3.x` | 1 | 1 |
| AllReduceMultimem | `RMSNormFusedSumSqCudaVector8_v1.res` | 43 | 1 |
| AllReduceMultimem | `RMSNormFusedSumSqCudaVector8_v1.x` | 301 | 7 |
| AllReduceMultimem | `RMSNormSumSqCudaVector8_v1.x` | 43 | 1 |
| EmbeddingShard | `AllReduceMnnvl_v1.parts` row 0 of 8 | 32 | 1 |
| peers | `AllGather_v1.parts` (the LM head's 7 other shards) | 14 | 14 |
| peers | `AllReduceMnnvl_v1.parts` rows 1–7 of 8 | 32 each | 1 each |
| peers | `AllReduceResidualMnnvl_v1.parts` rows 1–7 of 8 | 256 each | 8 each |
| weights | `BatchedGemm_v1.w` (DSAMOE's strided column) | 1,081,344 | 16 |

The words by family:

| family | words |
|---|---|
| AllReduceMultimem | 4,761,600 |
| weights | 322,961,408 |
| peers | 658,112 (140 `peers_*` inputs) |
| EmbeddingShard | 6,144 |

Nothing is free under a family either half lowers.

## Where the halves meet

Each meeting point below is wired. The wire patterns are writer → reader, with a gather named by the families its
own allocation reads:

| meeting point | wire patterns | wired |
|---|---|---|
| RoPE's reader: `RoPEHeadGptJ → SparseMlaHeadBlock_v1.q_rope` | 176 | yes, directly (a head's 64 words are the RoPE unit's row) |
| The routed SiluMul's writer: `gather<MoeExpertFp8Gemm> → SiluMul_v2.gu` | 2 | yes |
| SiluMul → the experts' w2 quantiser: `SiluMul → Fp8GroupQuantUe8m0_v1.x` | 2 | yes |
| MoeSum's writer: `gather<MoeExpertFp8GemmW> → MoeSum_v1.y`, all 8 rows | 16 | yes |
| The fused norm → the router gate: `gather<AllReduceRmsNormMnnvl> → LlBf16DotprodGemm_v1.x` | 1 | yes |
| The fused norm → the experts' w13 quantiser: `gather<AllReduceRmsNormMnnvl> → Fp8GroupQuantUe8m0_v1.x` | 1 | yes |
| MoeSum → the residual add: `MoeSum → ResidualAdd_v1.r` | 2 | yes (both dense) |

The expert GEMMs' rows: `MoeExpertFp8Gemm → gather` 2 patterns, and `MoeExpertFp8GemmW → gather` 512 patterns.

**No port that should be wired stays free**, so neither side's lowering needs a change for the other's. No hunk is
needed in `node_window_sparse.py` for the meeting.

## What the meeting costs

`MoeSum`'s `y` row of 192 words, an expert's columns, is gathered from 6 of the routed w2 GEMM's 32-word block rows:
`AllGather_v1{WORLD=6,N=32}` × 11,264, 8 a MoeSum unit. These are new units, and the `WORLD=64,N=1`, `WORLD=96,N=1`
and `WORLD=8,N=64` gathers grew too: gather units are 56,574 here against 17,986 in the dense-only census
(`r20261113-015207-eb58`).

Two sides could remove that `MoeSum` gather:

- MoeSum at H = 32. That is 6 times the MoeSum units, and the norm's apply would read 6 rows.
- DSAMOE's block of n = 32 rows, which `BLOCK_WORDS` = 4,096 fixes at K = 256 [inferred from the 32-word rows].

Neither is taken; the 192 width is item 4's measured choice.

## A change DSAMOE's half needs, and one it will

#9375 (head `34c5cbb3705c`, not merged) renames `IF.Opening` to `IF.TableOpening`, with no alias.
`node_window_sparse._table_reads` builds `IF.Opening` (its one place), so it stops importing once #9375 lands:

```diff
-    w.b.openings.append(IF.Opening(tid, param, "tok", ("row", "salt", "path"), "o1", at, units))
+    w.b.openings.append(IF.TableOpening(tid, param, "tok", ("row", "salt", "path"), "o1", at, units))
```

It is written here only, not applied, because it is DSAMOE's file.
