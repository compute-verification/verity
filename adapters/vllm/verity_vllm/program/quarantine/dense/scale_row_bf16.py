"""`aten.mul.Tensor(x32 [M,N], r [M,1])` — a row scaled by its own f32 scalar, where `x32 = x.to(float32)` is the exact widening
of a bf16 tensor (GemmaRMSNorm without residual: `x * torch.rsqrt(variance + eps)`).  ATen's CUDA `mul_kernel` on f32 with
the `[M,1]` operand broadcast along the row: one FMUL per element, RN.  The widening is a Transparent noop, so the definition
reads the bf16 source; `s` is the row's `[1]` cell of the `[M,1]` operand (PerRow).

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.ScaleRowBf16`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import ScaleRowBf16

SEMANTIC_ID = "dense/ScaleRowBf16@v1"

__all__ = ['ScaleRowBf16', 'SEMANTIC_ID']
