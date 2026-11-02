"""`aten.mean.dim(v32 [M,N], [-1], keepdim=True)` under `VLLM_BATCH_INVARIANT=1`: vLLM d9105ea80
`vllm/model_executor/determinism/batch_invariant.py::mean_dim` launches the Triton `mean_kernel` with `BLOCK_SIZE=1024`
(`num_warps=4`, the JIT default), the input viewed as `(M, N, 1)`, one program per output element:

    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        vals = tl.load(input_ptr + idx, mask=n_offsets < N, other=0.0)
        acc += tl.sum(vals)
    mean_val = acc / N

The registered body writes that reduction out gate by gate (the same Triton 3.7.1 lowering the b1 `RMSNormTriton_v1` body
models for `_rms_norm_kernel`): a [1024] f32 block is held by 128 threads in the blocked layout `sizePerThread=4`
(128-bit loads), `threadsPerWarp=32`, `warpsPerCTA=4` — thread t owns elements `4t..4t+3` of each of the two 512-element
repetitions (registers in repetition-major order); `tl.sum` accumulates a thread's 8 registers sequentially
(`reduceWithinThreads`), then a warp xor-butterfly 16,8,4,2,1 (`reduceWithinWarps`), then the four warp results as
`(w0 + w2) + (w1 + w3)` through shared memory; masked lanes contribute `+0.0` literally.  `acc += ...` adds the chunk sums
in order from `0.0`; `acc / N` is Triton's f32 division, `div.full.f32` (`DivFullScaleA` * `DivFullRcp`, as in
`RMSNormTriton_v1`).  Output: `Array(1, F32)` — the row's `keepdim` cell.

The per-thread ownership (`SIZE_PER_THREAD = 4`, two repetitions) is the layout the coalescing pass picks for a contiguous
f32 load whose row pointer is provably 16-byte aligned (the same as the b1 `RMSNormTriton_v1` body's).  Evidence that it is
the production order: the difftest adapter (`mean_triton_difftest.py`) is bit-exact against `mean_batch_invariant` on the
pod at N = 2304 (Gemma-2's H: two full blocks + a masked 256-tail), 1024, 1280, 2048 and 256 with random f32 rows — a
different register order or per-thread width changes the rounding of most random rows.  The 32-case
seed-7 difftest found the layout is *specialisation-dependent*: for `N % 16 != 0` Triton cannot prove the alignment and
the layout is `sizePerThread=1` (`size_per_thread`, below); the body follows the same rule.

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.MeanTriton`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import MeanTriton, size_per_thread, thread_elements, BLOCK_SIZE, NUM_WARPS, THREADS, SIZE_PER_THREAD, TILE, REPS

SEMANTIC_ID = "dense/MeanTriton@v1"

__all__ = ['MeanTriton', 'size_per_thread', 'thread_elements', 'BLOCK_SIZE', 'NUM_WARPS', 'THREADS', 'SIZE_PER_THREAD', 'TILE', 'REPS', 'SEMANTIC_ID']
