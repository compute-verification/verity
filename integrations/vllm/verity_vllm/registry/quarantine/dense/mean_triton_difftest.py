"""difftest adapter for `dense/MeanTriton@v1`: production = vLLM d9105ea80 `batch_invariant.mean_batch_invariant` (the
`aten.mean.dim` override active under `VLLM_BATCH_INVARIANT=1`; launches `mean_kernel` with BLOCK_SIZE=1024) on one row of N
f32 values, `dim=[1], keepdim=True` -> `[1, 1]`.

Shapes: 2304 (Gemma-2 H: two full 1024-blocks + a masked 256-tail), 1024, 1280, 2048, 256 — the `N % 16 == 0` kernel
specialisation (`sizePerThread=4`) — and, since cov-dense 2026-09-17, the small non-16-divisible rows 4, 20, 100, 1000 that
Triton compiles with the `sizePerThread=1` layout (`mean_triton.size_per_thread`): random-normal rows there discriminate the
two register orders (and a pairwise tree) in one case each.  The `max` edge kind uses N = 2: the two ±max words sit in lanes
0 and 1 and meet in the last butterfly stage, so the kind exercises overflow to ±inf while a single add of finite words can
never form `inf + -inf` (whose payload — CUDA's canonical 0x7FFFFFFF vs the registry's 0x7FC00000 — is a NaN convention of
the protected `prims.py`, not a property of this reduction).  The earlier N = 4 `max` draw assumed one thread held all four
words; production spreads them over lanes 0..3 (seed-7 case 27: `(128 + -MAX) + (-MAX + MAX)` = -MAX, not -inf)."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import kind_for, pick

SEMANTIC_ID = "dense/MeanTriton@v1"
SHAPES = (2304, 1024, 1280, 2048, 256, 4, 20, 100, 1000)


def cases(rng, i: int) -> dict:
    kind = kind_for(i)
    N = 2 if kind == "max" else pick(SHAPES, i)
    return {"statics": {"N": N}, "inputs": [{"shape": [N], "dtype": "float32", "kind": kind}], "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    from vllm.model_executor.determinism.batch_invariant import mean_batch_invariant
    (x,) = inputs
    out = mean_batch_invariant(x.view(1, int(statics["N"])), [1], keepdim=True)
    assert out.shape == (1, 1) and out.dtype == torch.float32, (out.shape, out.dtype)
    return [out]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.mean_triton import MeanTriton
    return bind(MeanTriton, N=int(statics["N"]))
