"""`SiluMulBf16_v2` — one output word of vLLM `_C.silu_and_mul` (d9105ea80 `csrc/libtorch_stable/activation_kernels.cu`,
`act_and_mul_kernel<bf16, silu_kernel, true>`), `SiluMulBf16_v1` except on the sm_120 red-team's edge words.

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.circuits.ops.act.SiluMulBf16`, bound through
`SiluMul_v2{I}` by the `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports the same object so the quarantine's
difftest adapter, `ALL_PRIMITIVES` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.circuits.ops.act import (BF16, F32_NEG_ZERO, LN_F32_OVERFLOW, ZERO_BITS, SiluMulBf16,  # noqa: F401
                                              silu_f32_bits)

SEMANTIC_ID = "act/SiluMulBf16@v2"

__all__ = ["BF16", "F32_NEG_ZERO", "LN_F32_OVERFLOW", "SEMANTIC_ID", "SiluMulBf16", "ZERO_BITS", "silu_f32_bits"]
