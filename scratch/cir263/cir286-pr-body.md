Lane circuits
Refs CIR-286
Goal circuits/production-settings-as-circuits

Each statement is marked: [run] read from a run's output, [read] read from the source or the tree, [inferred] my reasoning or arithmetic, not checked by a run.

## What changed and why

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

PENDING

## circuit-check

`r20261114-081226-83a2` at `9badaf0ec718cc1f3414ec94b182d25607e27aae`, the report on the three Definitions the two forms are held to, by id [run]:
- `DsaRMSNorm_v1{N=2048, EPS=1e-05}`: ok, 0 failures, 0 unchecked [run].
- `DsaRMSNorm_v1{N=512, EPS=1e-05}`: ok, 0 failures, 0 unchecked [run].
- `RoPEHeadGptJ_v2{D=64}`: ok, 0 failures, 0 unchecked [run].
- 3 targets, 0 failures, each compared with a realization on random vectors [run].

No Definition, binding or pin changes here [read].

## Checks at this head

PENDING

## Not stated

- That a served step makes these launches: the capture calls the kernels directly, with its own operands [read].
- `fp8_ds_mla` or an fp8 MLA cache, and the fp8 query (`quantize_mqa`): not GLM-5.3's served configuration on the H200 (cache dtype `auto`, `FLASH_ATTN_MLA_SPARSE`) [read], and not compared.
- Which of CIR-263's item-4 parts a draft row needs at served sizes: given on CIR-286, for the lead to file.

## What I trust least

PENDING
