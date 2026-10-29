"""`aten.rsqrt.default(v32)` on f32 (GemmaRMSNorm: `torch.rsqrt(variance + eps)`, N = 1 per row).  ATen's CUDA
`rsqrt_kernel_cuda` calls `::rsqrt(float)` = libdevice `rsqrtf`, which ptxas emits as MUFU.RSQ with the subnormal range
scaling of `rsqrt.approx.f32` — the registered `RsqrtApprox_v1` (the same instruction sequence vLLM's CUDA RMSNorm kernels
use; probe `rsqrt_f32`: equal to `RsqrtApprox` on 1,048,576 of 1,048,576 samples, and NOT the correctly rounded
1/sqrt: 808,881 of them).

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.RsqrtF32`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import RsqrtF32

SEMANTIC_ID = "dense/RsqrtF32@v1"

__all__ = ['RsqrtF32', 'SEMANTIC_ID']
