# exactlin: the int8-exact linear as two Triton kernels

The int8-exact profile's linear (`verity_int8::row_quant`, then `verity_int8::scaled_mm`; `engine.int8_exact`'s statement)
as two launches a call, stock vLLM int8's count: a quantizer bit for bit `int8_exact.quantize_rows`, and a GEMM whose
epilogue is `int8_exact.dequantize`. Each is gated bit for bit against `int8_exact` and the int8 Definitions' replay
evaluators, then timed against stock vLLM int8, at kernel level and served. The kernels run on an RTX PRO 6000 (sm_120),
an H200 (sm_90) and a B200 (sm_100).

The served CUDA bodies of both ops are `integrations/vllm/verity_vllm/engine/int8_fused.py`, with their config table in
`engine.profiles.int8_gemm_configs`. `exactlin.py` holds the same kernels and table for the benches here, and `install`
serves the ops from them. Nothing a verifier runs reads this directory.

## Entry points

Each launcher takes one leased GPU for a bounded time, writes under `$RESEARCH_RUN_DIR/out`, and passes extra arguments to
its script:

~~~text
RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO bash benchmarks/hash/exactlin/kbench.sh                 # gate, then kbench.py
RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO KSCRIPT=kgate.py bash benchmarks/hash/exactlin/kbench.sh  # the gate alone
RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO bash benchmarks/hash/exactlin/served.sh \
  --arms stock,exact,exact-fused --gate --ops decode --t 64                                       # served
RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO bash benchmarks/hash/exactlin/share.sh                  # uncredited share
~~~

A failed gate times nothing and exits 4.

## API (`exactlin.py`)

For x [M, K] bf16, W [N, K] int8 and s_w [N] float32:

- `quant_rows(x, xq, s_a)`: the quantizer into caller-owned xq [M, K] int8 and s_a [M] float32.
- `scaled_mm(xq, s_a, w, s_w, y, cfg=None, work=None)`: y [M, N] bf16 = `dequantize(X W^T)` in place, at `cfg`. A split-K
  `cfg` needs `work`.
- `linear(x, w, s_w, xq, s_a, y)`: the two, into caller-owned buffers.
- `row_quant(x)` and `product(xq, s_a, w, s_w)`: the ops' CUDA bodies, each allocating only its results.
  `install()` registers them for `verity_int8::row_quant` and `verity_int8::scaled_mm`.
- `Cfg`: a GEMM config (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, SPLIT_K, TMA), written `32x64x256:w4:s3:k2[:t1]`.
- `pick(m, n, k, sm)`: a call's config. It takes the first row of `CONFIGS[sm][(n, k)]` whose m_max is at least m, and
  otherwise vLLM's tiles (`default_cfg`).
- `CONFIGS`: per architecture and (N, K), the gated winners `ksweep.py` writes to `best.json`.
- `Workspace`: a split-K launch's int32 partials and per-tile counters, which zero themselves between calls.
- `ptx_check(compiled)`: what a compiled kernel's PTX says about its float semantics, with `ok` its verdict. It requires
  round-to-nearest division, no flush-to-zero and no fused multiply-add.

## Files

- `exactlin.py`: the kernels (`exactlin_quant_rows`, `exactlin_scaled_mm`, and `exactlin_scaled_mm_tma` over host
  tensor descriptors) and the API above.
- Gates:
  - `kgate.py`: the fused linear bit for bit against `int8_exact` on Llama-3.1-8B's and Qwen3-8B's linears at every
    `CONFIGS` row, with each buffer poisoned first. It covers the statement's edge rows, a CUDA graph's replay, zero
    allocation and `ptx_check`, and also measures where stock vLLM int8's kernels differ from the statement. `gate()`
    is what the benches call first, and `--dump` writes the outputs `replay.py` reads.
  - `replay.py`: `Int8RowQuant_v1` and `ScaledMmS8_v1`'s replay evaluators on `kgate.py --dump`'s outputs, on the CPU.
  - `values.py`: the served values. The vLLM integration's `int8_exact_gpu` runs with `install()` first and checks every
    tapped linear's y bit for bit; it runs through `served.sh` as `SERVED_SCRIPT`.
- Kernel-level benches, run by `kbench.sh` (`KSCRIPT` picks the script):
  - `kbench.py`: the exact, stock and fused arms, interleaved rep by rep, each in a CUDA graph over enough weight
    copies to stream from DRAM. `--arms` also takes single kernels.
  - `ksweep.py`: the fused GEMM's configs swept at the decode shapes, each gated before it is timed, and the
    quantizer's warps.
  - `zybench.py`: the cost of `int8_fused.scaled_mm_zy`'s z/y record epilogue over `scaled_mm`, after the vLLM
    integration's tests of it.
- Served:
  - `served.sh`: `benchmarks/hash/dither/sbench.py`'s stock and exact arms (`stock`, `stock-percall`, `exact`,
    `exact-fused`, and `stock-cutlass` on sm_90) on the leased GPU's NUMA node. With `exact-fused` it runs `kgate.py` in the same lease first.
    `DITHER_NSYS=1` captures the decode step's kernels for `steps.py`.
  - `percall.py`: sbench's `stock-percall` arm, stock's `triton_scaled_mm` behind a custom op
    (`exactlin_stock::scaled_mm`), so its tile heuristic runs on each call's M.
  - `percall_diag.py`: why `stock-percall` can serve other tokens than `stock`. It logs each engine's Inductor autotune
    picks and pins them first, last or deterministic.
- Analysis:
  - `steps.py`: a served decode step's kernels by category, from an sbench `prof` capture's nsys SQLite export.
    `--compare` puts arms side by side.
  - `share.py` and `share.sh`: the share of a served step that PoUW doesn't credit. Each arm runs under nsys through
    decode and prefill scenarios, and `--analyse` splits a forward's GPU time into the credited int8 GEMMs, the
    quantizer, lm_head, attention, norms and elementwise, sampling and the gaps.
- `probe.sh`: what stock vLLM int8 runs on an RTX PRO 6000 host, without a GPU lease. It records the stack's versions
  and vLLM's int8 linear sources, and the SASS of `_C`'s per-token quantizer.

Outside this directory, `benchmarks/hash/dither/sbench.py` serves `exact-fused` and the `-xl` arms from `install`, and
`benchmarks/torch_gpu/s8_linear` times `int8_fused` with `kbench.py`'s helpers (`timed`, `copies_for`, `nvml`).
