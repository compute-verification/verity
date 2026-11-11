"""`GeluTanhMulBf16`, one output word of vLLM `_C.gelu_tanh_and_mul`: the registry's object
(`verity.definitions.registry.dense.GeluTanhMulBf16`, whose comment derives it from the kernel's source and SASS),
re-exported so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` read the same object.
"""
from __future__ import annotations

from verity.definitions.registry.dense import GeluTanhMulBf16, gelu_tanh_bf16_bits, BETA_BITS, KAPPA_BITS, HALF_BITS, ONE_BITS, ZERO_BITS

SEMANTIC_ID = "dense/GeluTanhMulBf16@v1"

__all__ = ['GeluTanhMulBf16', 'gelu_tanh_bf16_bits', 'BETA_BITS', 'KAPPA_BITS', 'HALF_BITS', 'ONE_BITS', 'ZERO_BITS', 'SEMANTIC_ID']
