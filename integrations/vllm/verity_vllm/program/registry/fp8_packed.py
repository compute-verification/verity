"""vLLM's per-token-group FP8 quantizer with packed UE8M0 scales, the activation quantizer of DeepGEMM's block-FP8 linear:
`_C.per_token_group_fp8_quant_packed(x, x_q!, x_s_packed!, 128, eps, -448, 448)` (`per_token_group_quant_8bit_packed_register_kernel`,
csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu at vLLM d9105ea80; group 128 only).

Per 128-element group of a bf16 row: `y = fmaxf(fmaxf-fold(eps, |f32(x_e)|) / 448, 1e-10f)` (`Fp8GroupScale_v1`, then the clamp); the
scale byte is `y`'s exponent field plus one when its mantissa is nonzero (the power of two at or above `y`, UE8M0); every element is
`e4m3(clamp(f32(x_e) * (1 / 2^(byte - 127)), -448, 448))`.  The row's bytes are packed four to an int32, little-endian in group order,
the groups past the row's last written as 0 (`[mn, ceil(K/G/4)]` int32; stored column-major and TMA-aligned, which is layout, not value).
"""
from __future__ import annotations

from verity.ir.defs import bind, composite, primitive
from verity.ir.refs import Coll as _Coll, concat, tuple_of
from verity.ir.types import Array, Tuple
from verity.ml.prims import F32ToE4m3Sat
from verity.ml.scalar import F32BitsShl23, F32Fabs, F32Fmaxf, F32Fminf
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.fp8 import C448, C_EPS_QUANT, C_NEG448, E4M3, Fp8GroupScale
from verity_vllm.program.registry.prims import BF16, F32, I32

C_ONE = b1.f32c(1.0)
ZERO_I32 = b1.const(32, 0)
_CONF = "exact-model-tested: see Fp8GroupQuantPacked_v1"


@primitive("F32Ue8m0Ceil", 1, [("y", F32)], I32, conformance="exact-model-tested")
def F32Ue8m0Ceil(y: int) -> int:
    """The UE8M0 byte at or above f32 `y` as the packed kernel computes it: `(exp_bits + (mant_bits != 0)) & 0xFF`, sign ignored."""
    u = y & 0xFFFFFFFF
    return (((u >> 23) & 0xFF) + (1 if u & 0x7FFFFF else 0)) & 0xFF


@primitive("U8x4Pack", 1, [("a", I32), ("b", I32), ("c", I32), ("d", I32)], I32, conformance="exact-model-tested")
def U8x4Pack(a: int, b: int, c: int, d: int) -> int:
    """Four bytes (the low 8 bits of each operand) in one int32, `a` in the lowest byte."""
    return (a & 0xFF) | (b & 0xFF) << 8 | (c & 0xFF) << 16 | (d & 0xFF) << 24


@composite("Fp8GroupScaleUe8m0", 1, ["G"], lambda S: ((("x", Array(S.G, BF16)),), I32), conformance=_CONF)
def Fp8GroupScaleUe8m0(B, S, x):
    """One group's UE8M0 scale byte: `F32Ue8m0Ceil(fmaxf(Fp8GroupScale_v1(x), 1e-10f))`."""
    return B.call(F32Ue8m0Ceil, B.call(F32Fmaxf, B.call(bind(Fp8GroupScale, G=S.G), x), B.call(C_EPS_QUANT)))


def packed_words(K: int, G: int) -> int:
    return (K // G + 3) // 4


@composite("Fp8GroupQuantPacked", 1, ["K", "G"],
           lambda S: ((("x", Array(S.K, BF16)),), Tuple(Array(S.K, E4M3), Array(packed_words(S.K, S.G), I32))), conformance=_CONF)
def Fp8GroupQuantPacked(B, S, x):
    """`_C.per_token_group_fp8_quant_packed` on one bf16 row: (x_q[K] e4m3, the row's packed UE8M0 scale words).  Per group g, byte
    `e_g = Fp8GroupScaleUe8m0_v1(x[gG:(g+1)G])` and `x_q[i] = cvt.rn.satfinite.e4m3(fminf(fmaxf(f32(x[i]) * (1 / 2^(e_g - 127)), -448), 448))`
    (the power of two rebuilt as `e_g << 23`); word j packs bytes 4j..4j+3, a byte past the last group 0."""
    K, G = S.K, S.G
    assert K % G == 0, (K, G)
    c448, cneg, one = B.call(C448), B.call(C_NEG448), B.call(C_ONE)
    scale = bind(Fp8GroupScaleUe8m0, G=G)
    qs, es = [], []
    for g in range(K // G):
        e = B.call(scale, x[g * G:(g + 1) * G])
        es.append(e)
        inv = B.call(P.F32Div, one, B.call(F32BitsShl23, e))
        for i in range(g * G, (g + 1) * G):
            qs.append(B.call(F32ToE4m3Sat, B.call(F32Fminf, B.call(F32Fmaxf, B.call(P.F32Mul, B.call(P.Bf16ToF32, x[i]), inv), cneg), c448)))
    zero = B.call(ZERO_I32) if len(es) % 4 else None
    es += [zero] * (-len(es) % 4)
    words = [B.call(U8x4Pack, *es[j:j + 4]) for j in range(0, len(es), 4)]
    return tuple_of(_Coll(Array(K, E4M3), concat([w.refs for w in qs])), _Coll(Array(len(words), I32), concat([w.refs for w in words])))


FP8_PACKED_COMPOSITES = [Fp8GroupScaleUe8m0, Fp8GroupQuantPacked]

__all__ = ["F32Ue8m0Ceil", "U8x4Pack", "Fp8GroupScaleUe8m0", "Fp8GroupQuantPacked", "FP8_PACKED_COMPOSITES", "packed_words"]
