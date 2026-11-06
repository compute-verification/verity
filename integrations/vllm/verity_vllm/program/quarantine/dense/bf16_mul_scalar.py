"""`aten.mul.Tensor(t [M,N] bf16, c)` — a bf16 tensor times a Python scalar (Gemma-2 `LogitsProcessor`: `logits * self.soft_cap`
after the tanh).  ATen's CUDA `mul_kernel` for bf16 computes `f32(x) * f32(c)` in opmath float and converts once with
`cvt.rn.bf16.f32` (NaN -> 0x7FFF).  Probe `mul30`: `aten.mul(bf16, 30.0) == bf16(f32(x) * 30)` on all 65,282 non-NaN
words.

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.Bf16MulScalar`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import Bf16MulScalar

SEMANTIC_ID = "dense/Bf16MulScalar@v1"

__all__ = ['Bf16MulScalar', 'SEMANTIC_ID']
