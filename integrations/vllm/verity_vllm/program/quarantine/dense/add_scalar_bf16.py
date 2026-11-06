"""`aten.add.Tensor(w32, c)` where `w32 = w.to(float32)` is the exact widening of a bf16 weight (GemmaRMSNorm.forward_native:
`weight = self.weight.float() + 1.0`, N = H, C = 1.0).  The widening is a Transparent noop, so the definition reads the bf16
root; one FADD in f32, RN.

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.AddScalarBf16`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import AddScalarBf16

SEMANTIC_ID = "dense/AddScalarBf16@v1"

__all__ = ['AddScalarBf16', 'SEMANTIC_ID']
