"""`aten.to.dtype(y32, bfloat16)` — the f32 -> bf16 narrowing copy (GemmaRMSNorm: `x.to(orig_dtype)`; the fused variant's
`residual = x.to(orig_dtype)`).  ATen's CUDA copy kernel converts with `c10::BFloat16(float)`, which on sm_80+ device code is
`__float2bfloat16` = `cvt.rn.bf16.f32`: RNE, NaN -> 0x7FFF — the registered `F2fpBf16_v1` (the same word the b1 GEMM /
RMSNorm stores use; the exhaustive tables of two other ATen bf16 kernels show the 0x7FFF NaN word).

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.NarrowF32ToBf16`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import NarrowF32ToBf16

SEMANTIC_ID = "dense/NarrowF32ToBf16@v1"

__all__ = ['NarrowF32ToBf16', 'SEMANTIC_ID']
