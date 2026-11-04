"""`aten.pow.Tensor_Scalar(x32, 2)` where `x32 = x.to(float32)` is the exact widening of a bf16 tensor (GemmaRMSNorm without
residual: `ir.ops.rms_norm`).  ATen's CUDA `pow_tensor_scalar_kernel` special-cases the exponent 2 as `base * base`
(one FMUL, RN); the widening is a Transparent noop, so the definition reads the bf16 source (probe `pow2_f32`:
`aten.pow(x, 2) == x * x` on 2^20 f32 samples).

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.SquareBf16`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import SquareBf16

SEMANTIC_ID = "dense/SquareBf16@v1"

__all__ = ['SquareBf16', 'SEMANTIC_ID']
