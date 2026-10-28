"""`aten.pow.Tensor_Scalar(s32, 2)` on a genuine f32 tensor (GemmaRMSNorm with residual: `ir.ops.fused_add_rms_norm`'s
`x = x.to(f32) + residual.to(f32)`).  ATen's CUDA `pow_tensor_scalar_kernel` special-cases the exponent 2 as `base * base`.

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.SquareF32`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import SquareF32

SEMANTIC_ID = "dense/SquareF32@v1"

__all__ = ['SquareF32', 'SEMANTIC_ID']
