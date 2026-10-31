"""`aten.mul.Tensor(y32 [M,N], w1 [N])` — an f32 row times a shared f32 vector (GemmaRMSNorm: `x * (weight.float() + 1.0)`,
the `[N]` operand broadcast over rows).  ATen's CUDA `mul_kernel` on f32: one FMUL per element, RN.

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.MulVecF32`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import MulVecF32

SEMANTIC_ID = "dense/MulVecF32@v1"

__all__ = ['MulVecF32', 'SEMANTIC_ID']
