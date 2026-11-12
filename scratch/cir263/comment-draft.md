**DRAFT — fill the card results before posting.**

**Result so far.** <LEAD: one or two sentences, filled after the card window: each kernel's Definition and what the capture showed.>

**New Definitions, as drafts for your roles** (all in `verity/definitions/registry/mtp_draft.py`; each takes one position, so a row of 1 + K positions calls each once a position):
- `MtpEhNorm_v1{H, BLOCK, WARPS, EPS}`, bound at H = 6144, BLOCK = 8192, WARPS = 4, EPS = 1e-5. In: `pos` i64; `e`, `p`, `ew`, `hw`: H bf16 each (the embedding row, the previous hidden state, `enorm`'s and `hnorm`'s weights). Out: 2H bf16, `[enorm(e or +0) | hnorm(p)]`.
- Its cut, the units it is staged as, because a whole 6,144 row is past a Flock proof block:
  - `MtpEhZero_v1{H}`, bound at H = 6144. In: `pos` i64; `e`: H bf16. Out: H bf16.
  - `TritonBlockRMSNormSumSq_v1{N, BLOCK, WARPS, EPS}`, bound at N = 6144, BLOCK = 8192, WARPS = 4, EPS = 1e-5. In: `x`: N bf16. Out: f32, the norm's scale.
  - The apply is the registered `DsaRMSNormApply_v1{G}` at G = 64. In: `x`: G bf16; `w`: G bf16; `r`: f32. Out: G bf16. There are 96 of them a half.
- `eh_proj` and the indexer's logits need no new Definition: they are the registered `Gemm_v2` and `CublasSplitKGemm_v1`, and `DsaMqaLogits_v3` (below).
- Least sure to stay: `MtpEhNorm_v1`'s one output of 2H. The kernel writes one contiguous row, but your fold may want two ports of H. The other is BLOCK and WARPS as statics: on vLLM's launch both follow from H, and I keep them because the sum order depends on them.

Pin: vLLM `d9105ea80`, FlashMLA `6bc49418c5ea`. Branch `cursor/circuits-cir263-draft-layer-kernels-6f04` (head `<SHA>`), no pull request yet.

**1. `_fused_eh_norm_kernel` (pin `common/kernels.py:987-1019`).** Both norms are `TritonBlockRMSNorm_v1{N=6144, BLOCK=8192, WARPS=4, EPS=1e-5}`, the registered Definition in `registry/dsa_norm.py`. `DsaRMSNorm_v1` is not it: it equals `TritonBlockRMSNorm_v1{N, N, 4, EPS}` only for a power-of-two N, and here 6,144 live elements sit in a masked tile of 8,192.
- What shows it: Triton 3.7.1's sm_90 PTX for the H200's specialisation, compiled on a CPU (`.reqntid 128`, the same chain, butterfly and halving tree, `div.full.f32` by 6,144). The capture pins that PTX by hash and op counts. It also checks the SASS: two `FFMA` of the scaled sum by 1/6144 plus eps, two `MUFU.RSQ` and no `MUFU.RCP`. So rsqrt's operand is the one-rounding `fma(DivFullScaleA(sum, N), DivFullRcp(N), eps)` that `TritonBlockRMSNorm_v1` states, which differs from two roundings at this N.
- Position 0: the kernel compares the whole int64 position to 0 and, where it is, replaces the embedding row by bf16 +0 before its norm. The hidden-state half is never zeroed. The kernel's output is `[norm(e) | norm(h)]`, 12,288 bf16.
- New: `MtpEhNorm_v1` and its cut (above), in a new module `registry/mtp_draft.py` with its own sidecar. Its two norms are `TritonBlockRMSNorm_v1` calls. No registered Definition's words changed.
- circuit-check: the first run (`r20261112-141330-0ffd`) found no wrong word, but failed both targets on `block/past-range`. A whole row needs a block of 2^29 bits for `MtpEhNorm_v1` and 2^28 for `TritonBlockRMSNorm_v1` at 6,144, and C-Flock's proved range is 2^27. So the row is staged as its cut, declared in the module's `CUTS`. Read off circuit-check's own block count:
  - the select's block is 2^25 bits;
  - each scale's is 2^27, at the range;
  - each apply's is 2^21.
  - The cut's ANDs (98,305 + 2 × 36,703,364 + 192 × 316,992 = 134,367,497) equal the whole's, so it recomputes nothing.
  - Each word Definition has a Boolean version written as Calls, in `registry/boolean_mtp_draft.py`, so circuit-check compares it with its reference on random vectors and pins it. They are `MtpEhZero_v2`, `TritonBlockRMSNormSumSq_v2` (with its fused-eps epilogue piece `TritonBlockRMSNormMeanRsqrt_v1`) and `MtpEhNorm_v2`, which is the cut on bits. The scale on bits needs every register either live in all threads or masked in all; 6,144 in 8,192 on 4 warps is.
  - `r20261112-145958-9b2e` (update pass and report) wrote the `and`-count pins of `boolean_mtp_draft`. `r20261112-151029-3dd7` filled the digest pins of the three Boolean versions at 6,144, and found `MtpEhNorm_v2{6144}` past the range too, so it is staged as the same cut on bits.
  - `r20261112-151627-ac2a`, the last run: update pass and report on both sidecars, 14 targets, 0 failures, an empty pins diff, and 4 warnings. All four are `vectors/none`, on the word Definitions at 6,144 (`MtpEhNorm_v1`, `MtpEhZero_v1`, `TritonBlockRMSNormSumSq_v1`) and on `DsaRMSNormApply_v1{64}`: each of their Boolean versions is compared with its reference on random vectors at small statics and pinned at 6,144, but not compared at 6,144 itself.
- On the card: <capture run id, rows compared, words compared, words that differ, alternatives refuted.>

**2. `eh_proj` (pin `nvidia/mtp.py:68`, called at `:132-134`).** `eh_proj` is a replicated `nn.Linear(12288, 6144, bias=False)` in bf16, not tensor-parallel. On the H200, `run_glm52_plan` has no plan, because the plan is built only on compute capability 10.3 (`glm52_low_latency_gemm.py:89`). So the call is `F.linear` on cuBLAS.
- `cublas_routes.json` has no `linear 12288 6144` route, so the route came from a capture at this shape: <scan run id; the route map over M = 1..8192: split-K and non-split spans>.
- Definitions: `Gemm_v2{K=12288, N=6144, DOT=HopperBF16WgmmaDot16_v1}` where cuBLAS does not split, and `CublasSplitKGemm_v1{K=12288, N=6144, SPLIT_K=s}` at each split it takes. <words compared / differ per M>.
- Token counts: a draft's step 0 runs `eh_proj` on every token of the target's step (6 a decoding request), and steps 1-4 on one token a request.

**3. The indexer's logits at six query rows a request: NEWPIN is wrong on the H200.** With five draft tokens, `next_n` = 6. DeepGEMM's paged logits are native on sm_90 only at `next_n` in {1, 2, 4} (pin `utils/deep_gemm.py:560-568`), so the indexer flattens instead (`indexer.py:672-692`).
- `PrepareUniformDecodeKernel` (`indexer.py:72-119`, run at `:874-905`) turns each request's six queries into six rows at `next_n = 1`. Row j has `seq_len = max(L - 6 + j + 1, 0)`, and the request's block-table row is copied to each.
- So the kernel `DsaMqaLogits_v3` was captured at, `fp8_fp4_paged_mqa_logits` at one row a request (`sparse_attn_indexer.py:563-612`), is the kernel that runs, at shorter key counts. There is no 1 + 5 row padding on sm_90.
- On the card: <capture run id; rows; words compared; words that differ; whether `DsaMqaLogits_v3` stands>.

**4. The rest of a draft step, read and not built.** The draft layer is one GLM-5.3 decoder layer with its own embedding and head, so every kernel below has the target layer's shapes. Each Definition is the one the capture map names for the target; what differs is which token counts reach it. "Bound" means a sidecar binds it at GLM-5.3's served statics.
- Embedding: `EmbeddingShard_v1`, bound only at toy statics (VS 16), as for the target.
- Input norm (no residual yet, since `eh_proj`'s output is replicated, no all-reduce): `RMSNormCudaVector8_v1{N=6144}` at BLOCK 768 below 256 tokens and 256 from 256. Both are bound (`cuda_rmsnorm`).
- MLA: the FP8 block linears `FlashInferFp8Linear_v1` below 32 tokens and `DeepGemmFp8Linear_v1` from 32 are bound only at K = N = 256. `DsaRMSNorm_v1` at 2,048 and 512 is bound (`dsa_norm`), and `RoPEHeadGptJ_v2{D=64}` is bound (`boolean_rope`). The sparse attention, `SparseMla_v1` and its head split, is bound at one key count (T = 65).
- Indexer (step 0 only): `DsaIndexerQ_v1`, `DsaIndexerKRow_v1` and `DsaMqaLogits_v3` are bound, the last at key counts 1, 288, 2,048, 2,049, 8,192 and 32,533. The `wk` GEMM's cuBLAS routes are bound. `DsaTopKShortRow_v1` is bound and covers rows of at most 2,048 keys, as for the target.
- Steps 1-4 reuse step 0's top-k. GLM-5.3's config sets `index_share_for_mtp_iteration`, so the speculator sets `skip_topk` and the attention launches `fused_norm_rope` and `fused_q` with `has_indexer=False` (`v1/worker/gpu/spec_decode/mtp/speculator.py:30-58`, `deepseek_v32/attention.py:289-319`). `HAS_INDEXER` is a Triton constexpr, so that is a separate specialisation of each kernel. Its norm and RoPE programs are the same source, but no capture has read its PTX. The target's skip-top-k layers run it too.
- After attention: the fused all-reduce and norm `AllReduceRmsNormMnnvl_v1` at up to 42 tokens and the all-reduce routes past that are bound at GLM-5.3's rows (`tp_allreduce`).
- MoE: the router `MoeRouterSigmoidBias_v1` and its f32-out gate GEMM routes are bound at the served statics (`moe`, `cublas_bf16`, `ll_bf16`), as are the FP8 expert GEMMs `MoeExpertFp8Gemm*` at each served (SWAP, DOT) (`fp8_moe_experts`). `SiluMulFp8BlockQuant_v1`, `MoeSum_v3`, `SiluMul_v2` and `ResidualAdd_v1` are bound only at small statics.
- Head: the final norm is `RMSNormCudaVector8_v1{N=6144}` (bound). The head GEMM's route is one non-split route for M 1-8,192, so it is `Gemm_v2{K=6144, N=19360}`, which no sidecar binds. Then `AllGather_v1` (toy statics) and `TokenSelect_v4` at V = 154,880 (bound, `dead_free`).
- Where NEWPIN is right: the draft token is torch's argmax and is vllm's, and steps after the first reuse step 0's top-k.

**Where NEWPIN is wrong or incomplete.**
- The norm's Definition is `TritonBlockRMSNorm_v1`, not `DsaRMSNorm_v1`.
- On the H200, the indexer flattens six rows to `next_n = 1` and does not pad.
- `eh_proj` is replicated and goes through cuBLAS because `run_glm52_plan` exists only on compute capability 10.3.
- `has_indexer=False` is its own Triton specialisation, which no capture has read.
- The capture map's `mtp` selection says the serve command has no speculative decoding, so nothing there builds this layer yet (vllm's, VLLM-393).
