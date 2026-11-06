"""`aten.add.Tensor(v32, c)` — an f32 tensor plus a Python scalar (GemmaRMSNorm: `variance + variance_epsilon`, N = 1 per
row, C = 1e-6).  ATen's CUDA `add_kernel` computes in opmath float: `a + alpha * b` with `alpha = 1` and the scalar
narrowed once to f32 (`f32(1e-6)` = 0x358637BD); one FADD, RN (probe `add_eps_f32`: `aten.add(v, 1e-6) ==
f32(v + f32(1e-6))` on 2^20 samples).

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.AddScalarF32`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import AddScalarF32

SEMANTIC_ID = "dense/AddScalarF32@v1"

__all__ = ['AddScalarF32', 'SEMANTIC_ID']
