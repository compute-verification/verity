Lane circuits
Refs CIR-286
Goal circuits/production-settings-as-circuits

Each statement is marked: [run] read from a run's output, [read] read from the source or the tree, [inferred] my reasoning or arithmetic, not checked by a run.

## What changed and why

**On #9993.** The branch is cut from #9993's head `8ab720de29ac`, so until #9993 lands this diff holds #9993's too. This pull request's own change is its last commit, `9badaf0ec718`: four files, all under `adapters/vllm/` [read: `git diff --stat 8ab720de29ac HEAD`].

At vLLM `d9105ea80`, a GLM-5.3 layer with no indexer calls `fused_norm_rope` and `fused_q` with `has_indexer=False` (`models/deepseek_v32/attention.py:308-327`) [read]. Those layers are:
- the 58 of the 78 target layers that `index_topk_freq` 4 at offset 3 makes "shared" (`:161-195`);
- the MTP draft's steps after its first, which reuse the first step's top-k (`nvidia/mtp.py`, `set_skip_topk`).

`has_indexer` is a Triton constexpr, so each kernel compiles to its own specialisation, and no capture had compared its words [read]. This adds that capture. Each form is held to a Definition already registered, at the statics the capture map names for the same programs with the indexer [read]:
- `_fused_norm_rope_kernel`:
  - Programs 0 (the index-K cache write) and 3 (the top-k fill) return at once, so the top-k buffer keeps what the last full layer wrote.
  - Program 2 is `DsaRMSNorm_v1{N=2048, EPS=1e-05}` on q_c.
  - Program 1 is `DsaRMSNorm_v1{N=512, EPS=1e-05}` on kv_c, then `RoPEHeadGptJ_v2{D=64}` on k_pe, then the bf16 MLA cache row `[kv_c | k_pe]` (the served cache dtype is `auto`) [read: `common/kernels.py`].
- `_fused_q_kernel`: with the bf16 query, programs 1 and 2 return, and program 0 is `RoPEHeadGptJ_v2{D=64}` on each of the 8 local heads' q_pe [read].

No new Definition and no new binding: both Definitions are bound at these statics on `main` [read]. That their words equal these kernels' is the capture's (below).

- **The capture**, `adapters/vllm/tests/properties/glm53_index_share_capture_gpu.py`, reuses `glm53_dsa_capture_gpu` (operands, caches, the subnormal RoPE probe, `phase_rope`, the PTX record) and `glm53_fused_rmsnorm_capture_gpu` (served and probe rows, the comparison jobs, the alternatives it refutes) [read]. Its record is the registered property `glm53_index_share_capture` in `records/property_record.py`, its schema id written in its row [read].
- **A CPU test of its `main`**, `adapters/vllm/tests/properties/test_glm53_index_share_capture.py`: the GPU is stood in for by the twin's words. Everything after the launch is real, through the worker pool to the sealed record and the exit status, with a flipped word and a RoPE settled on another Definition each giving NO-GO [read].

## The capture of record

`r20261114-181508-ea93`, read on its node, at this head `9badaf0ec718`, on one H200 card with `card_286.sh index_share` (CIR-263's `glm53_mtp_capture_card.sh` with a start and an end time). The card held 0 compute processes and 0 MiB at 18:15:37Z. One lease, and the capture ran 18:15:38Z to 18:15:56Z, exit 0 [run]. The record sealed GO with `ok` true, digest `bceec7d7ab5c610c4cb267182ccdc38461de1c667e4d915c2488503391c595d2`, recomputed from the fetched file. It names vLLM `0.28.1rc1.dev472+gd9105ea80` (the pin's commit) and Triton 3.7.1 [run].

- **Selection.** `FLASH_ATTN_MLA_SPARSE` is first of the valid backends, and the CuTe DSL `fused_q` is not taken, with or without the indexer [run].
- **The two kernels' launches.** `fused_norm_rope` ran twice on 3,136 tokens: once on served operands and once on the RMSNorm probe rows. Each launch [run]:
  - was deterministic;
  - left the top-k buffer as filled (all 7);
  - wrote each of the 3,128 slotted rows of the MLA cache as `[kv_c_out | k_pe_out]`;
  - left every unslotted cache word as filled.

  `fused_q` ran on 3,136 tokens, was deterministic, and gave its shape [run].
- **RMSNorm, against `DsaRMSNorm_v1`.** 12,544 rows (q_c at N 2,048 and kv_c at N 512, served and probe), 16,056,320 words compared, 0 differ. The Definition's own evaluator on 88 rows equals the twin [run]. The words that refute each alternative reading [run]:
  - q_c: `RMSNormTriton_v1` 85, an f64 sum 81, a sqrt epilogue 76, `x * (r * w)` 79, a contiguous sum 55, warp pairs 39, per-vector chains 21;
  - kv_c: `RMSNormTriton_v1` 40, an f64 sum 34, a sqrt epilogue 34, `x * (r * w)` 32, warp pairs 9.
- **RoPE, against `RoPEHeadGptJ_v2{D=64}`.** k_pe: 200,896 words (3,136 served rows and 3 subnormal probe rows), 0 differ; `RoPEHeadGptJ_v1` would differ on 48. q_pe: 1,607,168 words (8 heads of each, and 24 probe rows), 0 differ; v1 would differ on 384. Both settle on v2 [run].
- **The words in all:** 17,864,384 compared, 0 differ [run; the sum is mine].
- **PTX.** One specialisation of each kernel was compiled in the run's empty Triton cache. Neither holds an e4m3 conversion: the with-indexer forms write fp8 to the index cache and, at `quantize_mqa`, to the query [run; the second clause read]. Hashes [run]:
  - `_fused_norm_rope_kernel` `8b14cb1d01af…`, with 2 `rsqrt.approx.ftz.f32` and 2 `MUFU.RSQ`;
  - `_fused_q_kernel` `31dfab914158…`.
- **Bindings.** The three Definitions built from their bindings at these statics equal the ones compared [run].

## circuit-check

`r20261114-081226-83a2` at `9badaf0ec718cc1f3414ec94b182d25607e27aae`, the report on the three Definitions the two forms are held to, by id [run]:
- `DsaRMSNorm_v1{N=2048, EPS=1e-05}`: ok, 0 failures, 0 unchecked [run].
- `DsaRMSNorm_v1{N=512, EPS=1e-05}`: ok, 0 failures, 0 unchecked [run].
- `RoPEHeadGptJ_v2{D=64}`: ok, 0 failures, 0 unchecked [run].
- 3 targets, 0 failures, each compared with a realization on random vectors [run].

No Definition, binding or pin changes here [read].

## Checks at this head

At `9badaf0ec718cc1f3414ec94b182d25607e27aae`.

- **The capture's CPU tests**, `r20261114-081152-3b4c`: the new test with its three end-to-end cases, `test_record`, the MTP captures' test, and the P05 and typed-records lints: 28 passed, 0 failed [run].
- **Suites.** The suites the commit over #9993 touches (`suites.py --changed 8ab720de29ac`): `r20261114-164148-d38b`, waiting for a slot when this opened. Everything else this head shares with #9993's head, whose suites are #9993's [read].
- **The adapter's ratchet gate** against `d260e6bb9cb3`, **pins** `--stale` and **retired terms**: `r20261114-164228-3413`, waiting for a slot when this opened.
- **Size.** `records/property_record.py` is 334 lines; the new files are tests, outside the P10 size rule [read].

## Not stated

- That a served step makes these launches: the capture calls the kernels directly, with its own operands [read].
- `fp8_ds_mla` or an fp8 MLA cache, and the fp8 query (`quantize_mqa`): not GLM-5.3's served configuration on the H200 (cache dtype `auto`, `FLASH_ATTN_MLA_SPARSE`) [read], and not compared.
- Which of CIR-263's item-4 parts a draft row needs at served sizes: given on CIR-286, for the lead to file.

## What I trust least

That the capture's operands reach every branch the served shared layers take [inferred]. The launches use the DSA capture's operands at 3,136 tokens, with 8 tokens unslotted, and the RMSNorm and subnormal-RoPE probes. A served step's batch, its slot mapping and its positions are not reproduced. The kernels' code at `has_indexer=False` has no data-dependent branch that I found besides the slot test [read], and the slot test is exercised both ways.
