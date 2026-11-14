Lane circuits
Refs CIR-263
Goal vllm/production-vllm-reproducible

Each statement is marked: [run] read from a run's output, [read] read from the source or the tree, [inferred] my reasoning or arithmetic, not checked by a run.

## What changed and why

GLM-5.3's recipe serves with five draft tokens from one MTP draft layer. At vLLM `d9105ea80`, that layer's own three kernels had no Definition [read]. This holds them to registered Definitions at GLM-5.3's sizes, binds them for circuit-check, and adds the captures that compared each kernel's words on an H200.

- **`_fused_eh_norm_kernel`** (enorm and hnorm in one launch, pin `models/deepseek_v32/common/kernels.py:987-1019`) is `MtpEhNorm_v1{H, BLOCK, WARPS, EPS}`, new in `verity/definitions/registry/mtp_draft.py`, bound at H 6144, BLOCK 8192, WARPS 4, EPS 1e-5 [read]. At position 0 it replaces the embedding row by bf16 +0, the whole int64 position compared to 0; each half is then the registered `TritonBlockRMSNorm_v1` at 6,144 values in a masked block of 8,192 [read]. It is not `DsaRMSNorm_v1`, which is that norm only at a power-of-two row [read].
- **Staging.** A whole 6,144 row is past Flock's proved block (2^29 bits for `MtpEhNorm_v1`, 2^28 for `TritonBlockRMSNorm_v1`) [run: `r20261112-141330-0ffd`], so the row is staged as its cut, declared in `CUTS` [read]: the position select `MtpEhZero_v1{H}`, each half's scale `TritonBlockRMSNormSumSq_v1{N, BLOCK, WARPS, EPS}`, and 96 of the registered `DsaRMSNormApply_v1{G=64}` a half [read].
- **Boolean versions**, as Calls in `verity/definitions/registry/boolean_mtp_draft.py`: `MtpEhZero_v2`, `TritonBlockRMSNormSumSq_v2` with its fused-eps epilogue `TritonBlockRMSNormMeanRsqrt_v1`, and `MtpEhNorm_v2`, the cut on bits with its own `CUTS` [read]. With them circuit-check compares each word Definition with a realization on random vectors and pins it.
- **`eh_proj`** (pin `models/deepseek_v32/nvidia/mtp.py:68`) is a replicated `nn.Linear(12288, 6144, bias=False)` in bf16; on the H200 it is `F.linear` on cuBLAS [read]. It needs no new Definition. `verity/definitions/registry/mtp_draft.circuit-check.py` now binds both of its routes: `Gemm_v2/glm53-mtp-eh-proj`, core's `Gemm_v2{K=12288, N=6144, DOT=HopperBF16WgmmaDot16_v1}`, where cuBLAS does not split K; and `CublasSplitKGemm_v1/glm53-mtp-eh-proj-2`, `cublas_bf16`'s `CublasSplitKGemm_v1{K=12288, N=6144, SPLIT_K=2, DOT=HopperBF16WgmmaDot16_v1}`, at the one split count the route scan found up to 8,192 tokens [read]. Each binding's id equals the one the capture of record bound and compared with [run: `r20261113-154513-8180`; read: the sidecar's targets built here and compared with the capture's `bind_check([2])`]. A whole row of either is past Flock's proved block range (2^38 bits against 2^27) [run: `r20261113-201855-8dbc`], and a served row of either is cut along N into slices, each the same Definition at N = 1 on x and one row of W (`SLICES` in `adapters/vllm/verity_vllm/pipeline/serving_templates.py`) [read]. So each route is bound at its slice too: `Gemm_v2/glm53-mtp-eh-proj-slice`, `Gemm_v2{K=12288, N=1, DOT=HopperBF16WgmmaDot16_v1}`, and `CublasSplitKGemm_v1/glm53-mtp-eh-proj-2-slice`, `CublasSplitKGemm_v1{K=12288, N=1, SPLIT_K=2, DOT=HopperBF16WgmmaDot16_v1}`. These two are what the capture's reference evaluated at each of its 177 sampled coordinates (`_reference` in `glm53_bf16_gemm_capture_gpu.py`) [read].
- **The indexer's logits at six query rows a request.** On sm_90 `next_n = 6` is not native (DeepGEMM's paged logits are native at 1, 2 and 4, pin `utils/deep_gemm.py:560-568`), so vLLM flattens each request's six queries into six rows at `next_n = 1` (`v1/attention/backends/mla/indexer.py:72-119`, `:672-692`) [read]. The kernel is the one `DsaMqaLogits_v3` was captured at, at shorter key counts [read], so no new Definition or binding.
- **Captures**, under `adapters/vllm/tests/properties/`: `glm53_mtp_eh_norm_capture_gpu.py`, `glm53_mtp_eh_proj_capture_gpu.py`, `glm53_mtp_indexer_logits_capture_gpu.py`, and `glm53_mtp_capture_card.sh`, which runs one under one lease on one card after checking the card is idle [read].
- **Each capture's record is a registered property's**: `seal` refuses a schema no property names, so the three are registered in `adapters/vllm/verity_vllm/properties/record.py`, with their schema ids in the new `adapters/vllm/verity_vllm/records/mtp_captures.py` (the typed-records lint keeps schema literals in `records/`) [read].
- **The split-K probe at eh_proj's shape.** `splitk_probe_words` in the sibling `glm53_bf16_gemm_capture_gpu.py` could not separate the strided partition from the contiguous one at K 12288, 2 splits, bf16 output, and raised [run: `r20261113-031522-215e`]. Where its random rows leave those two equal, and only there, it now plants a crafted third row, so every probe that existed before is unchanged [read].
- **`eh_proj`'s shape in the sibling's `MODELS`** is there only while a run or a scan process lasts (`registered()`); importing the module used to add it, and 18 of the sibling's tests then saw a third model [run: `r20261113-040826-9fec`].
- **A CPU test of each capture's `main`**, `adapters/vllm/tests/properties/test_glm53_mtp_captures.py`: the GPU is stood in for by the twin's words at the launch; everything after it is real, through the worker pool to the sealed record and the exit status [read].
- `verity/definitions/tests/catalog_ids.json` lists the seven new ids; no registered Definition's words changed [read].

## What the captures of record gave

Node A, card 0, cores 0-9, Fri 13 Nov 15:45Z to 15:52Z, one lease each with `glm53_mtp_capture_card.sh`, at branch head `ce09de03c318f40d29efbec7cca2536a8b7cf735`. The card held no compute process and 0 MiB at each start. Each record sealed with `ok` true, its digest recomputed from the fetched file, and each names vLLM `0.28.1rc1.dev472+gd9105ea80` [run]. A word is compared with the Definition's NumPy twin; the Definition's own evaluator runs on a sample and equals the twin on all of it [run].

- `r20261113-154513-8180`, `eh_proj`, digest `41442bea4181c34d43a40ac11596b2e9bb85456f2b8c8376e24ebaee2cef5e1b` [run].
  - Route scanned at every M from 1 to 8,192 tokens: split-K 2 at M 105-111, 129-159, 161-223, 272, 288 and 336, no split elsewhere [run].
  - 24 launches (M 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 105, 128, 192, 256, 384, 512, 768, 1024, 1025, 8192), each deterministic and on the scanned route; 4,458 rows of 6,144 words (every row up to M 1,024, 384 rows of M 1,025 and of M 8,192): 27,389,952 words compared, 0 differ; the evaluator on 177 coordinates, 0 differ [run].
  - The two split launches (M 105, 192): 297 rows, 1,824,768 words, 0 differ from `CublasSplitKGemm_v1` [run]. The job's probe, 32 crafted words a split launch: contiguous f32 halves fit all 32, strided halves miss 1, bf16 partials miss 2 [run].
  - Recomputed afterwards on a CPU from the run's kept words, not by the job: of the 1,824,768 split words, 0 differ under contiguous f32 halves, 5,653 under strided halves, 675,620 under bf16 partials [run, on my VM, not a recorded run].
- `r20261113-155013-668e`, `eh_norm`, digest `5584f173348e50978eb87d2309a56f096521c291f473950f27f76b8ff446fc11` [run].
  - Two launches of 1,024 positions, each half compared whole: 4,096 rows, 25,165,824 words, 0 differ; `MtpEhNorm_v1`'s evaluator on 102 rows, 0 differ [run].
  - One PTX specialisation, hash `88978f9f23bf0bf743216e79cbc564e294a845fda819eeccf30c8ca355408b7d`, as pinned from the pin's source; `.reqntid 128`; SASS: 2 `FFMA` of the mean and eps, 2 `MUFU.RSQ`, 0 `MUFU.RCP` [run].
  - Words that refute each alternative reading, fewest first (p half, e half): unfused eps 16 and 21; per-vector chains 31 and 33; 8 warps 41 and 32; a contiguous sum 52 and 36; `x * (r * w)` 56 and 62; `RMSNormTriton_v1` 84 and 111; the position's low word alone 12,288 (e); the position ignored 49,152 (e) [run].
- `r20261113-155125-8b02`, `indexer_logits`, digest `253714da690d3c2e1b3ede3fc9a2a99ee20cf06a23688472f6302d31060bd35d` [run].
  - `native_next_n_supported(6)` false and no varlen on this card; vLLM's own `_PREPARE_UNIFORM_DECODE_KERNEL` gave the lengths and block table expected [run].
  - 9 requests (54 rows): all 50,664 logits, and 50,000 random pairs over them again. 64 requests (384 rows, 702,153 logits): 59 rows whole and 50,000 random pairs, 179,605 distinct logits [run; the distinct counts are mine, from the kept words]. 290,933 comparisons in all, 0 differ, 0 unresolved; the evaluator on 144 logits, 0 differ [run].

**What differs between `ce09de03c318` and this head** [read: `git diff ce09de03c318 HEAD`]. The branch was carried across `main`'s layout moves (`tools/move/restack.py`), then merged with `main` `f479876a8758`, then two commits to the sidecar: eh_proj's whole rows, then their slices. In the captures' own files the only change is the move's import rewrite in `glm53_mtp_eh_proj_capture_gpu.py` (`verity.definitions.ops` to `verity.circuits.ops`, in `bind_check`). The sibling `glm53_bf16_gemm_capture_gpu.py`, whose scan and launch the eh_proj capture runs, changed by `main`'s own work: imports moved to the top, type annotations, an equivalent rewrite of the route join's adjacency test, and `bind_check`'s default model list read at call time; none of it is on the path the eh_proj capture runs [read]. The Definitions' modules changed only by the move's import rewrites. No capture ran at this head [inferred from the above: they stay of record].

## circuit-check

`r20261113-210950-1125` at `08d6acdced2231eafdf6b10bf0245d6e0d2a8ef0`, in its own clone [run]:

- The update pass on both sidecars and the two slices: 14 targets, 0 failures, exit 0; the pins' diff after it is empty and the tree clean [run].
- The report on both sidecars, `DsaRMSNormApply_v1{G=64}` (the apply the norm's cut uses, `dsa_norm`'s binding), the two slices and the two whole rows: 17 targets, 4 failures, 6 `vectors/none` warnings; 11 targets compared with a realization on random vectors [run].
  - Every Definition added here passes: `MtpEhNorm_v2` at H 256 and 6144, `MtpEhZero_v2` at H 16, 256 and 6144, `TritonBlockRMSNormSumSq_v2` and `TritonBlockRMSNormMeanRsqrt_v1` at N 256 and 6144 [run].
  - The word Definitions with no realization at their bound size (`MtpEhNorm_v1`, `MtpEhZero_v1` and `TritonBlockRMSNormSumSq_v1` at 6144, and `DsaRMSNormApply_v1{G=64}`) carry `vectors/none` and no failure; their Boolean versions above are compared at the same size [run].
  - `Gemm_v2{K=12288, N=1, …}` and `CublasSplitKGemm_v1{K=12288, N=1, SPLIT_K=2, …}`: ok, 0 failures, each one unit of 16 bits out [run].
  - The four failures are the two whole rows': `block/past-range` on both (2^38 bits; Flock's proved range is 2^27), and on the split-K row also `partition/unpartitioned` and `correspondence/unprobed-table` (75,497,472 leaves of W) [run]. `Gemm_v2`'s whole row matched its reference on correspondence (1,312 s) [run]. `circuit-check --all` checks neither whole row, since a family's members over 150,000 gates are left out unless pinned (`select` in `tools/verity/circuit_check/src/circuit_check/targets.py`) [read]. The served indexer's whole-row bindings in `cublas_bf16.circuit-check.py` are in the same position [read; that they too are past range is my arithmetic, inferred].

## Checks at this head

<suites, gate, retired terms, pins>

## Runs

<runs at this head; earlier runs with their heads>

## Not stated

- That a served draft step makes these launches. Each capture calls the kernel directly with its own operands on a card with the pinned vLLM installed; no run here serves with speculative decoding. Which token counts a draft step gives `eh_proj` (and so which split span it lands in) is my reading of the source [inferred].
- `eh_proj` above 8,192 tokens, and any split count but 2 [run: none up to 8,192]. At 2 splits, forward, reverse and tree reductions are one order, so the capture separates only the partition and the partials' type [inferred].
- The indexer's 64-request launch whole: 522,548 of its 702,153 logits are not compared [run, my count].
- `fused_norm_rope` and `fused_q` with `has_indexer=False` (draft steps 1 to 4), a separate Triton specialisation of each, whose PTX no capture has read [read].
- The rest of the layer (embedding, input norm, MLA, MoE, head): read and listed on CIR-263, not built or captured here.
- The capture map's `mtp` section, the fold and the step compiler (vllm's, VLLM-393), and the accept rule.

## What I trust least

<...>
