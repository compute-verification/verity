"""`GeluTanhMulBf16` — one output word of vLLM `_C.gelu_tanh_and_mul`, the GeGLU of Gemma-2 (`GeluAndMul(approximate="tanh")`).
Its kernel source, the pinned build's SASS and the evidence for the body are documented beside the Definition, in
`verity.ml.definitions.registry.dense`.

The Definition lives in the accepted registry (`verity.ml.definitions.registry.dense.GeluTanhMulBf16`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import GeluTanhMulBf16, gelu_tanh_bf16_bits, BETA_BITS, KAPPA_BITS, HALF_BITS, ONE_BITS, ZERO_BITS

SEMANTIC_ID = "dense/GeluTanhMulBf16@v1"

__all__ = ['GeluTanhMulBf16', 'gelu_tanh_bf16_bits', 'BETA_BITS', 'KAPPA_BITS', 'HALF_BITS', 'ONE_BITS', 'ZERO_BITS', 'SEMANTIC_ID']
