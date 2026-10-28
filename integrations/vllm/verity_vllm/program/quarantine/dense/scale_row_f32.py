"""`aten.mul.Tensor(s32 [M,N], r [M,1])` — an f32 row scaled by its own f32 scalar (GemmaRMSNorm with residual:
`(x.float() + residual.float()) * torch.rsqrt(variance + eps)`).  ATen's CUDA `mul_kernel` on f32 with the `[M,1]` operand
broadcast along the row: one FMUL per element, RN; `s` is the row's `[1]` cell of the `[M,1]` operand (PerRow).

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.ScaleRowF32`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import ScaleRowF32

SEMANTIC_ID = "dense/ScaleRowF32@v1"

__all__ = ['ScaleRowF32', 'SEMANTIC_ID']
