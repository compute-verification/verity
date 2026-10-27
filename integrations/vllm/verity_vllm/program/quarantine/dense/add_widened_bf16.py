"""`aten.add.Tensor(x32, r32)` where both operands are exact f32 widenings of bf16 tensors (GemmaRMSNorm with residual,
`ir.ops.fused_add_rms_norm`: `x = x.to(float32); x = x + x_residual.to(float32)`).  ATen's CUDA `add_kernel` on f32
(`alpha = 1`): one FADD per element, RN.  Both widenings are Transparent noops, so the definition reads the two bf16
sources.

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.AddWidenedBf16`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import AddWidenedBf16

SEMANTIC_ID = "dense/AddWidenedBf16@v1"

__all__ = ['AddWidenedBf16', 'SEMANTIC_ID']
