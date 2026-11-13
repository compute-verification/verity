# CIR-274 DENSE: the census at 43 prompt ids, 2 outputs, 4 layers, rank 0

Run `r20261113-015207-eb58`, source `d6af67e8a6e88ab7d70ab6c61288e7b8944c0496` (the work branch's head), on
vy-nebius-cpu-14's research pool, rc 0. Its script is `p3_window.py` here (REQBLOCK's `scratch/cir274/p3_window.py`
with `families={**NW.FAMILIES, **ND.FAMILIES}`). `$OUT/p3_l4_r0.json` is kept with the run. Every number below is
read from that JSON, except where it is marked [inferred].

## The window

- Program: digest `7f36f17ae005fc05fd8c3655d037e201405cf8713bd5bbb49d9852006764f6f10bda5c6036be6f0e106dd862df102b58e571d111eed4bde189668efcc7c87fd9`,
  319 nodes, 2 steps.
- Window: digest `5ad440be82e274853467cbf459b33ffe0c06db83567a04dc828caf2f17c2fc404992985ab57df85c1aee2257314cccb20a03e19cb29c34eb8d12faa5a2f46b4b`.
  It has 3,488,836 units in 42 templates and 201 wire patterns. It settled and passed `check_wires` / `check_inputs`
  in 11.7 s, with a peak of 0.617 GiB.

## The dense families (44 tokens = 43 prefill + 1 decode)

| family | units | templates | arithmetic |
|---|---|---|---|
| SiluMul | 528 | `SiluMul_v2{I=1536}` 132, `{I=256}` 396 | 3 dense layers × 44; 9 × 44 in the MoE layer [inferred: shared + 8 routed] |
| RoPEHeadGptJ | 1,408 | `RoPEHeadGptJ_v2{D=64}` | 4 layers × 8 heads × 44 |
| ResidualAdd | 1,408 | `ResidualAdd_v1{N=192}` | 32 columns × 44, 1 MoE layer |
| MoeSum | 1,408 | `MoeSum_v1{TOPK=8,H=192}`, `y` cut 8 | 32 × 44, 1 MoE layer |
| AllReduceMnnvl | 32 | `AllReduceMnnvl_v1{WORLD=8,N=192,SHOT=1}`, `parts` cut 8 | 32 × the decode's 1 token (the prefill's is `AllReduceMultimem`) |
| AllReduceRmsNormMnnvl | 488 | `AllReduceResidualMnnvl_v1{N=192,SHOT=1}` 256, `RmsNormMnnvlApply_v1{G=192,SHOT=1}` 224, `RmsNormMnnvlSumSq_v1` 7, `RmsNormMnnvl_v1` 1 | 8 decode nodes × 32; 7 in a layer × 32 + 7; 1 final |
| Gemm (LM head) | 38,720 | `Gemm_v2{K=6144,N=1}` | 19,360 slices × 2 outputs |
| AllGather | 16 | `AllGather_v1{WORLD=1,N=19360}` | 8 ranks × 2 outputs |

The LM head's own shard is gathered by `AllGather_v1{WORLD=121,N=1}` 320, `{WORLD=80,N=121}` 4 and `{WORLD=2,N=9680}`
2, i.e. 160 / 2 / 1 per output.

Wire patterns by the reader's family: AllGather 2, AllReduceRmsNormMnnvl 38, Gemm 2, ResidualAdd 4, RoPEHeadGptJ 8 and
SiluMul 8. AllReduceMnnvl and MoeSum read only free rows (below); their wires are the ones they write.

## Free inputs, by writer family

| writer | unit ports | words | ports | read by |
|---|---|---|---|---|
| AllReduceMultimem (piece 2) | 12,772 | 4,761,600 | 19 | the prefill's norms (frame's) |
| BatchedGemm (DSAMOE) | 176 | 360,448 | 8 | `Fp8GroupQuantUe8m0`, `FlashInferFp8Quant1x128` `x` |
| EmbeddingShard (reads.md 1) | 32 | 6,144 | 1 | `AllReduceMnnvl_v1` `parts` row 0 |
| MoeExpertFp8Gemm (DSAMOE) | 352 | 180,224 | 2 | `SiluMul_v2` `gu` |
| MoeExpertFp8GemmW (DSAMOE) | 11,264 | 2,162,688 | 16 | `MoeSum_v1` `y`, its 8 rows |
| peers | 2,030 | 658,112 | 77 | `AllReduceResidualMnnvl_v1` 1,792 / 56, `AllReduceMnnvl_v1` 224 / 7, `AllGather_v1` 14 / 14 |

There are 140 peer partials. Unlowered nodes: AllReduceMultimem 9, BatchedGemm 16, CublasSplitKGemmF32Out 1,
DsaLatentRow 8, EmbeddingShard 2, Fp8GroupQuantUe8m0 4, LlBf16DotprodGemm 1, MoeExpertFp8Gemm 2, MoeExpertFp8GemmW 2,
MoeRouterSigmoidBias 2, SparseMlaSplit 176, StopCheck 1 and TokenSelect 2. Of these, only EmbeddingShard, StopCheck
and TokenSelect are this piece's (reads.md). The rest are the frame's, piece 2's, piece 5's or DSAMOE's.
