"""`GeluTanhMulBf16` — one output word of vLLM `_C.gelu_tanh_and_mul` (csrc/activation_kernels.cu, `act_and_mul_kernel<bf16,
gelu_tanh_kernel, true>`), the GeGLU of Gemma-2 (`GeluAndMul(approximate="tanh")`):

    const float f = (float)x;                                     // bf16 -> f32, exact
    constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;           // f32(sqrt(2/pi)) = 0x3F4C422A
    constexpr float KAPPA = 0.044715;                              // 0x3D372713
    float x_cube = f * f * f;                                      // (f*f)*f, two FMULs
    float inner = BETA * (f + KAPPA * x_cube);                     // FFMA(KAPPA, x_cube, f) (nvcc -fmad), then FMUL
    return (T)(0.5f * f * (1.0f + ::tanhf(inner)));               // (0.5f*f) * (1 + t), then cvt.rn.bf16.f32
    out = ACT_FN(x) * (y + 0.0f)                                   // f32(a) * (f32(y) + 0.0f), cvt.rn.bf16.f32

The pinned build (vLLM d9105ea80, `_C_stable_libtorch.abi3.so`, sm_89) compiles this as `masked_act_and_mul_kernel<
c10::BFloat16, gelu_tanh_kernel<c10::BFloat16>, true, false, false>(out, in, mask, d, n, float, float, float)`, whose SASS
is `SASS` (cuobjdump on the pod, `gelu_tanh_and_mul.sm89.sass`).

The `y + 0.0f` was found by the difftest's `zeros` edge kind (every `y = -0` word came back `+0`; 16/20 -> 20/20 cases).

Evidence (the exhaustive table `gelu_tanh_bf16_table.npy`, T[x] for all 65,536 bf16 words, and the factorisation check
`out == bf16(f32(T[x]) * f32(y))` on 2^20 random pairs): the body above reproduces T on every non-NaN word (65,282/65,282) and
gives the kernel's 0x7FFF for NaN, including x = -inf where `(0.5f * -inf) * 0` is NaN; the alternative association
`0.5f * (f * (1 + t))` is wrong on 129 words (overflow at |f| > 1.7e38), and FFMA vs separate FMUL+FADD for `inner` is not
distinguishable on the domain.  `tanhf` is the `TanhF32Rn` stand-in (see `tanh_f32.py`).

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.GeluTanhMulBf16`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
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
