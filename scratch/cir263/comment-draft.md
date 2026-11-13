**The MTP draft layer's three kernels of its own each have a Definition at GLM-5.3's sizes. The first run on a card (H200, Fri 13 Nov 03:15Z) compared 25,165,824 words of the fused norm and 290,933 indexer logits and found 0 that differ. It sealed no record, and `eh_proj`'s split-K route has no compared word yet, so nothing here is a capture of record: those run Fri 13 Nov from 15:45Z.**

**New Definitions** (all in `verity/definitions/registry/mtp_draft.py`; each takes one position, so a row of 1 + K positions calls each once a position):
- `MtpEhNorm_v1{H, BLOCK, WARPS, EPS}`, bound at H = 6144, BLOCK = 8192, WARPS = 4, EPS = 1e-5. In: `pos` i64; `e`, `p`, `ew`, `hw`: 6,144 bf16 each (the embedding row, the previous hidden state, `enorm`'s and `hnorm`'s weights). Out: 12,288 bf16, `[enorm(e, or +0 at position 0) | hnorm(p)]`.
- Its cut, the units it is staged as, because a whole 6,144 row is past a Flock proof block:
  - `MtpEhZero_v1{H}`, bound at H = 6144. In: `pos` i64; `e`: 6,144 bf16. Out: 6,144 bf16. Once a position.
  - `TritonBlockRMSNormSumSq_v1{N, BLOCK, WARPS, EPS}`, bound at N = 6144, BLOCK = 8192, WARPS = 4, EPS = 1e-5. In: `x`: 6,144 bf16. Out: one f32, the norm's scale. Twice a position.
  - The registered `DsaRMSNormApply_v1{G}` at G = 64. In: `x`, `w`: 64 bf16 each; `r`: f32. Out: 64 bf16. 192 a position.
- `eh_proj` and the indexer's logits need no new Definition: they are the registered `Gemm_v2`, `CublasSplitKGemm_v1` and `DsaMqaLogits_v3` (items 2 and 3).
- Least sure to stay: `MtpEhNorm_v1`'s single output of 12,288, where a fold may want two ports of 6,144; and BLOCK and WARPS as statics, which follow from H on vLLM's launch but fix the sum order.

Pin: vLLM `d9105ea80`, FlashMLA `6bc49418c5ea`. Branch `cursor/circuits-cir263-draft-layer-kernels-6f04` at `78884a1b8aa7`, no pull request yet.

What ties the card's kernels to the pin: the captures run the vLLM installed on the node and write its version into their record. With no record sealed I have not read that version; only the norm is tied so far, by its PTX hash, which equals the one compiled from the pin's source.

**1. `_fused_eh_norm_kernel` (pin `common/kernels.py:987-1019`).** Each of its two norms is the registered `TritonBlockRMSNorm_v1{N=6144, BLOCK=8192, WARPS=4, EPS=1e-5}` (`registry/dsa_norm.py`). It is not `DsaRMSNorm_v1`, which is that norm only at a power-of-two row; here 6,144 live elements sit in a masked block of 8,192.
- What shows it: the kernel's PTX on the card is one specialisation whose hash and op counts equal the ones pinned from Triton 3.7.1's sm_90 PTX (`.reqntid 128`, `div.full.f32` by 6,144), and its SASS has two `FFMA` of the scaled sum by 1/6144 plus eps, two `MUFU.RSQ` and no `MUFU.RCP`: one rounding before rsqrt, as the Definition states.
- Position 0: the kernel compares the whole int64 position to 0 and, where it is, replaces the embedding row by bf16 +0 before its norm. The hidden-state half is never zeroed.
- On the card, `r20261113-031522-215e` (card 0, one lease of 124 s): two launches of 1,024 positions, each half of each compared whole: 4 × 6,291,456 words, 0 differ; `MtpEhNorm_v1`'s reference on 23 rows of the first launch and 28 of the second, 0 differ. The capture then stopped where it seals its record (its schema was not a registered property's, fixed since), so it wrote none and did not print how many words refute each alternative.
- circuit-check, `r20261112-151627-ac2a`: 14 targets, 0 failures, an empty pins diff, 4 warnings, all `vectors/none` on the word Definitions at 6,144 and the apply at 64 (their Boolean versions are compared at small statics and pinned at 6,144).

**2. `eh_proj` (pin `nvidia/mtp.py:68`).** A replicated `nn.Linear(12288, 6144, bias=False)` in bf16. On the H200 `run_glm52_plan` has no plan (it is built only on compute capability 10.3), so the call is `F.linear` on cuBLAS.
- Route, scanned on the card over M = 1 to 8,192 tokens (same run): no split for M 1-104, 112-128, 160, 224-271, 273-287, 289-335 and 337-8,192; split-K 2 at M 105-111, 129-159, 161-223, 272, 288 and 336.
- Definitions: `Gemm_v2{K=12288, N=6144, DOT=HopperBF16WgmmaDot16_v1}` where cuBLAS does not split, `CublasSplitKGemm_v1{K=12288, N=6144, SPLIT_K=2}` where it does. Neither is bound in a sidecar yet; the bindings follow the capture of record.
- Words: the 14 unsplit launches the run reached (M 1 to 96): 321 rows, 1,972,224 words compared, 0 differ. That comparison ran afterwards on a CPU from the words the run kept (`r20261113-033120-282e`), not in the job. The run stopped at its first split launch, on a probe that could not tell two split-K orders apart at this shape (fixed since). So no split-K 2 word is compared, and that `CublasSplitKGemm_v1`'s order is cuBLAS's here is not shown.
- Token counts: a draft's step 0 runs `eh_proj` on every token of the target's step (6 a decoding request), steps 1-4 on one token a request. So from 18 decoding requests up, step 0 can land in a split span (my arithmetic: 18 × 6 = 108).

**3. The indexer's logits at six query rows a request.** With five draft tokens `next_n` = 6. DeepGEMM's paged logits are native on sm_90 only at `next_n` 1, 2 and 4 (pin `utils/deep_gemm.py:560-568`), so vLLM flattens: each request's six queries become six rows at `next_n = 1`, row j with `max(L - 5 + j, 0)` keys and the request's block-table row copied (`indexer.py:72-119`, `:672-692`). There is no padded row of 1 + 5 on sm_90.
- So the kernel that runs is the one `DsaMqaLogits_v3` was captured at, at shorter key counts.
- On the card (same run): the selection read as not native at 6; launches of 9 requests (54 rows, 100,664 logits) and 64 requests (384 rows, 190,269 logits); every logit compared with `DsaMqaLogit_v3{H=32, D=128}`: 290,933 words, 0 differ, 0 unresolved. `DsaMqaLogits_v3` stands as far as that goes; the same missing record applies.

**4. The rest of a draft step, read and not built.** It is one GLM-5.3 decoder layer with its own embedding and head, so each kernel has the target layer's shapes and the Definition the capture map names for the target. "Bound" means a sidecar binds it at GLM-5.3's served statics.
- Embedding: `EmbeddingShard_v1`, bound only at toy statics.
- Input norm: `RMSNormCudaVector8_v1{N=6144}`, bound at both block sizes.
- MLA: `FlashInferFp8Linear_v1` and `DeepGemmFp8Linear_v1` bound only at K = N = 256; `DsaRMSNorm_v1` at 2,048 and 512 and `RoPEHeadGptJ_v2{D=64}` bound; `SparseMla_v1` bound at one key count.
- Indexer (step 0 only): `DsaIndexerQ_v1`, `DsaIndexerKRow_v1`, `DsaMqaLogits_v3` and `DsaTopKShortRow_v1` bound.
- Steps 1-4 reuse step 0's top-k and launch `fused_norm_rope` and `fused_q` with `has_indexer=False`, a Triton constexpr and so a separate specialisation of each. No capture has read its PTX.
- After attention: `AllReduceRmsNormMnnvl_v1` and the all-reduce routes, bound.
- MoE: `MoeRouterSigmoidBias_v1`, its gate GEMM routes and `MoeExpertFp8Gemm*` bound; `SiluMulFp8BlockQuant_v1`, `MoeSum_v3`, `SiluMul_v2`, `ResidualAdd_v1` only at small statics.
- Head: `RMSNormCudaVector8_v1{N=6144}` bound; the head GEMM is one unsplit route, `Gemm_v2{K=6144, N=19360}`, which no sidecar binds; `AllGather_v1` at toy statics; `TokenSelect_v4` at V = 154,880 bound.

**Where the source differs from NEWPIN.**
- The norm is `TritonBlockRMSNorm_v1`, not `DsaRMSNorm_v1`.
- On the H200 the indexer flattens six rows to `next_n = 1` and does not pad.
- `eh_proj` is replicated and goes through cuBLAS, with split-K 2 in six spans of M.
- `has_indexer=False` is its own Triton specialisation, read by no capture.
- The capture map's `mtp` selection says the serve command has no speculative decoding, so nothing there builds this layer yet (vllm's).

Next: the three captures of record on card 0, Fri 13 Nov 15:45Z to 17:45Z, then `eh_proj`'s bindings and the pull request by Mon 16 Nov 16:00Z.
