# CIR-274 4b (DSAMOE): the census at 43 prompt ids, 2 outputs, 4 layers

Run `r20261113-020525-9b32` (vy-nebius-cpu-1, rc 0), source `8d7502ab19f1a5a8e707ddf38bf40ce95eaf2f27` (work branch), script
`p3_window.py OUT 4 0 sparse` (rank 0, the registered rank form via `benchmarks/compile_cost/run.py`'s `registered`, eager;
layers 0-2 dense, layer 3 MoE), families `{**node_window.FAMILIES, **node_window_sparse.FAMILIES}`. Output kept under the
run's `$OUT` as `p3_l4_r0_sparse.json`. Program build 10.1 s; window 36.9 s; peak 0.681 GiB.

- Program digest `7f36f17ae005fc05fd8c3655d037e201405cf8713bd5bbb49d9852006764f6f10bda5c6036be6f0e106dd862df102b58e571d111eed4bde189668efcc7c87fd9`, 319 nodes.
- Window digest `857e2863c6811eb54bfe5c70c1215959d27b13bccc919abe677e1d039f6ce2741381f73cc7bc6e22713601ea8180b018aa72d4ec25bc3ccb23a924d1cf028de9`.
- 5,067,546 units, 88 templates, 13,082 wire patterns, 2 steps.

## Units, templates and wire patterns by family

| family | units | templates | wire patterns |
|---|---:|---:|---:|
| DeepGemmFp8Linear | 3,336,284 | 8 | 56 |
| BatchedGemm | 1,081,344 | 2 | 16 |
| MoeExpertFp8Gemm | 361,856 | 4 | 8,264 |
| MoeExpertFp8GemmW | 152,064 | 4 | 4,200 |
| FlashInferFp8Linear | 77,588 | 8 | 49 |
| gather | 30,744 | 10 | 3,020 |
| CublasSplitKGemmF32Out | 11,008 | 1 | 1 |
| RMSNormFusedCudaVector8 | 9,934 | 3 | 20 |
| SparseMlaSplit | 2,816 | 45 | 708 |
| DsaRMSNorm | 1,584 | 2 | 24 |
| RMSNormCudaVector8 | 1,452 | 2 | 2 |
| Fp8GroupQuantUe8m0 | 396 | 2 | 1 |
| LlBf16DotprodGemm | 256 | 1 | 0 |
| DsaLatentRow | 176 | 1 | 8 |
| MoeRouterSigmoidBias | 44 | 1 | 2 |

The MoE layer's arithmetic (layer 3; 43 prefill tokens and 1 decode token, 8 slots a token, 352 pairs):

- `CublasSplitKGemmF32Out_v1{K=6144,N=1}` 11,008 = 43 x 256 (prefill gate); `LlBf16DotprodGemm_v1{N=1}` 256 (decode gate).
- `MoeRouterSigmoidBiasSlots_v1` 44 = 43 + 1. `Fp8GroupQuantUe8m0` (the MoE's own) 396 = 44 x (1 + 8).
- `MoeExpertFp8Gemm` 361,856 = 352 x (512 block GEMMs + 512 row reads `EmbeddingRead_v1{V=256,H=3072}` + 4 scale reads
  `H=96`): `MoeExpertFp8BlockGemm_v3{K=6144,N=1}` 176,128 (SWAP=true) + 4,096 (SWAP=false); reads 180,224 and 1,408.
- `MoeExpertFp8GemmW` 152,064 = 352 x (192 + 192 `H=4096` + 48 `H=4`): `MoeExpertFp8BlockGemmW_v3{K=256,N=32}` 66,048 +
  1,536; reads 67,584 and 16,896.

## Free inputs by family

| family | unit-ports | words | ports |
|---|---:|---:|---:|
| weights | 1,081,344 | 322,961,408 | 16 |
| AllReduceMultimem | 12,772 | 4,761,600 | 19 |
| AllReduceRmsNormMnnvl | 264 | 1,622,016 | 9 |
| SiluMul | 528 | 304,128 | 10 |
| RoPEHeadGptJ | 1,408 | 90,112 | 176 |
| AllReduceMnnvl | 33 | 12,288 | 2 |

`weights` is BatchedGemm's strided `w` (reads.md line 1). The rest are the dense half's families. In the MoE layer, the
decode step's gate GEMM (`LlBf16DotprodGemm`, 256 unit-ports) and its first quantiser (`Fp8GroupQuantUe8m0{K=6144}`, 1)
read `AllReduceRmsNormMnnvl`, and w2's quantisers (`Fp8GroupQuantUe8m0{K=256}`, 395 with the dense layers' linears'
quantisers of that template) read `SiluMul` (from the run's `free_list`). No family of this lane is free.

Unlowered: AllGather 2, AllReduceMnnvl 1, AllReduceMultimem 9, AllReduceRmsNormMnnvl 8, EmbeddingShard 2, Gemm 2,
MoeSum 2, ResidualAdd 2, RoPEHeadGptJ 8, SiluMul 10, StopCheck 1, TokenSelect 2. None is this lane's.

## At the work branch's head

Run `r20261113-022031-b6f2` (vy-nebius-cpu-1, rc 0), source `34b868a7f23b75864d9a8f7fc850d6d479ed957e` (tests only since
`8d7502ab19`): the same window digest `857e2863…`, 5,067,546 units, 88 templates, 13,082 wire patterns; window 17.5 s, peak
0.692 GiB. The same run: `test_node_window_sparse.py` 12 tests, 0 failures. Run `r20261113-022032-8a53` (source
`34b868a7f2`): `adapters/vllm/tests/lint` 90 tests, 0 failures, 0 errors, 1 skipped.

## With DENSE's families (local, not pushed)

A throwaway worktree of `34b868a7f2`'s lowering (`8d7502ab19`) merged with DENSE's `d6af67e8a6` (clean merge, its two
frame commits included), families `{**NW.FAMILIES, **node_window_dense.FAMILIES, **node_window_sparse.FAMILIES}` (no key
in both), LP = 32, on the VM:

- 1 layer MoE: digest `554dde1473a2…`; unlowered EmbeddingShard, StopCheck, TokenSelect; free EmbeddingShard, peers,
  weights. `MoeSum_v1{TOPK=8,H=192}` reads w2's block outputs through `AllGather_v1{WORLD=6,N=32}` (a 32-row block a
  chunk); `SiluMul` reads w13's through `AllGather_v1{WORLD=64,N=1}`; `RoPEHeadGptJ_v2` writes the key blocks' `q_rope`.
- 2 layers MoE: digest `24d9c20a9cf9…`; 1 layer dense: digest `a47ad8dac5fc…`; the same unlowered list.
- In all three the only free input on this lane's templates is BatchedGemm's `w` (reads.md line 1).

## Earlier runs

- `r20261113-014617-eb76` (source `bcdd17828d`, DSA families only): 4,541,702 units, 75 templates, 610 wire patterns.
- `r20261113-020034-765b` (source `619f5b2664`): refused, `_slots` kept each slot's last token's router row (fixed in
  `8d7502ab19`).
- `r20261113-020527-ff60` (source `8d7502ab19`): `adapters/vllm/tests/lint`, 90 tests, 0 failures, 0 errors, 1 skipped.
