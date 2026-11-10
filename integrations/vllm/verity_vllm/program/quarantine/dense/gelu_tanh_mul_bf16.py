"""`GeluTanhMulBf16`, one output word of vLLM `_C.gelu_tanh_and_mul`: the registry's object
(`verity.ml.definitions.registry.dense.GeluTanhMulBf16`, whose comment derives it from the kernel's source and SASS),
re-exported so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` read the same object.
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import GeluTanhMulBf16, gelu_tanh_bf16_bits, BETA_BITS, KAPPA_BITS, HALF_BITS, ONE_BITS, ZERO_BITS

SEMANTIC_ID = "dense/GeluTanhMulBf16@v1"

#: the pinned build's SASS for one output word, each instruction beside what it computes
SASS = """
    PRMT R3, RZ, 0x5410, R0        ; f = (float)x            FMUL R0, R3, R3 ; FMUL R0, R3, R0      ; x_cube = (f*f)*f
    FFMA R0, R0, 0.044714998, R3   ; f + KAPPA*x_cube        FMUL R3, R3, 0.5                       ; 0.5f*f
    FADD R16, R16, c[0x0][0x188]   ; y + <float arg> (= +0.0f for gelu_tanh_and_mul: turns y = -0 into +0, identity otherwise)
    FMUL R13, R0, 0.79788458       ; inner = BETA*(...)      MUFU.EX2 / MUFU.RCP / FFMA...          ; libdevice tanhf(inner)
    FADD R18, R18, 1 ; FMUL R3, R3, R18 ; F2FP.BF16.F32.PACK_AB R3, RZ, R3 ; PRMT R3, RZ, 0x5410, R3   ; act = bf16(...) -> f32
    FMUL R3, R16, R3 ; F2FP.BF16.F32.PACK_AB R0, RZ, R3 ; STG.E.U16                                   ; bf16((y+0)*act)
"""

__all__ = ['GeluTanhMulBf16', 'gelu_tanh_bf16_bits', 'BETA_BITS', 'KAPPA_BITS', 'HALF_BITS', 'ONE_BITS', 'ZERO_BITS', 'SEMANTIC_ID']
