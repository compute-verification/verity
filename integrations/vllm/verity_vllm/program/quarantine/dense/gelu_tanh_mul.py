"""`_C.gelu_tanh_and_mul(out [M,I], gu [M,2I])` — vLLM's GeGLU (`GeluAndMul(approximate="tanh")`, Gemma-2 MLP): `out[i] =
bf16(bf16(gelu_tanh(gate[i])) * up[i])`, gate = gu[:I], up = gu[I:] (`act_and_mul_kernel<..., act_first=true>`), the same
shape as the registered `SiluMul_v1` with the element `GeluTanhMulBf16` (see `gelu_tanh_mul_bf16.py`).

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.GeluTanhMul`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import GeluTanhMul

SEMANTIC_ID = "dense/GeluTanhMul@v1"

__all__ = ['GeluTanhMul', 'SEMANTIC_ID']
