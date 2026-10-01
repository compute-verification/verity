"""`registry.dense`'s activations as Boolean Definitions: the next version of each word Definition, the same function on bits.

    GeluTanhMulBf16_v2   (g, u: Array<16, Value<1>>) -> Array<16, Value<1>>                      word view: GeluTanhMulBf16_v1
    GeluTanhMul_v2{I}    (gu: Array<2I, Array<16, Value<1>>>) -> Array<I, Array<16, Value<1>>>   word view: GeluTanhMul_v1{I}

`GeluTanhMulBf16_v2` is traced from `gelu_tanh_mul` (`verity.ml.boolean.trace.definition`), so its And gates are that builder's
live ANDs: ``a = gelu_tanh_bf16(g)``, then ``bf16_rn(f32(a) * (f32(u) + 0.0f))`` through `verity.ml.boolean.fp` with F2FP's NaN
0x7FFF. ``a = (T)gelu_tanh_kernel((float)g)`` is a function of the 16-bit g alone (`dense.gelu_tanh_bf16_bits`), and its word
Definition reads no MUFU (``tanhf`` is `TanhF32Rn_v1`, RN32 of the real tanh), so neither does this. Off a band of exponent
fields e it is exact arithmetic on g's fields; on the band it reads a constant table of the word function's own values
(`elementwise.bf16_band`):

    g NaN, or -inf                 0x7FFF  ((0.5f * -inf) * 0 is NaN)
    e < BAND_LO                    bf16_rn(g / 2)  (1 + tanh(inner) rounds to 1)
    g > 0, e > BAND_HI[0]          g (+inf included)
    g < 0, e > BAND_HI[1]          -0  (1 + tanh(inner) is 0)
    BAND_LO <= e <= BAND_HI[sign]  g's sign, and the 15 magnitude bits of the table

The kernel's ``y + 0.0f`` turns ``u = -0`` into +0 and is the identity on every other word that reaches the output (a NaN ends
in the cast's 0x7FFF). `GeluTanhMul_v2{I}` is `GeluTanhMul_v1`'s body on bits: a batch of I `GeluTanhMulBf16_v2` Calls over
(gu[i], gu[I + i]), so each member's 16 output bits are one unit under `Q_word_v1`. No sub-Call is a word Definition.
"""
from __future__ import annotations

from verity.ir import boolean as BL
from verity.ir.defs import bind, composite
from verity.ir.types import Array
from verity.ml.boolean import fp
from verity.ml.boolean.elementwise import bf16_band
from verity.ml.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce, const_bits, or_reduce, sub
from verity.ml.boolean.trace import definition
from verity_vllm.program.registry import dense as D
from verity_vllm.program.registry import prims as P

__all__ = ["GELU_BAND_HI", "GELU_BAND_LO", "GeluTanhMul", "GeluTanhMulBf16", "gelu_tanh_bf16", "gelu_tanh_mul"]

BF16 = Array(16, BL.BIT)
NAN_F2FP = P.BF16_NAN_F2FP

#: the exponent fields `gelu_tanh_bf16` reads from its table, per sign
GELU_BAND_LO = 118
GELU_BAND_HI = (128, 129)


def _half(C, g):
    """``bf16_rn(g / 2)``'s 15 magnitude bits: (for e <= 1, the 8-bit significand halved, ties to even; for e >= 2, the
    exponent field less one)."""
    m, e = list(g[:7]), list(g[7:15])
    return add(C, m[1:] + [e[0]], [C.AND(m[0], m[1])], width=8) + [ZERO] * 7, m + sub(C, e, [ONE], 8)[0]


def gelu_tanh_bf16(C, g):
    """``(T)gelu_tanh_kernel((float)g)``, the activation half of `GeluTanhMulBf16_v1`, for a 16-bit g (LSB first)."""
    m, e, s = list(g[:7]), list(g[7:15]), g[15]
    nan = C.AND(and_reduce(C, e), OR(C, or_reduce(C, m), s))
    _, below = sub(C, e, const_bits(GELU_BAND_LO, 8), 8)
    _, under = sub(C, e, const_bits(GELU_BAND_HI[0] + 1, 8), 8)
    keep = C.AND(C.AND(NOT(s), NOT(under)), NOT(nan))
    low = C.AND(below, NOT(or_reduce(C, e[1:])))
    half_low, half_high = _half(C, g)
    band = bf16_band(C, g, D.gelu_tanh_bf16_bits, GELU_BAND_LO, GELU_BAND_HI)
    mag = [x ^ C.AND(low, y) ^ C.AND(below ^ low, z) ^ C.AND(keep, w) ^ nan
           for x, y, z, w in zip(band, half_low, half_high, g[:15])]
    return mag + [C.AND(s, NOT(nan))]


def gelu_tanh_mul(C, g, u):
    """`GeluTanhMulBf16_v1`: ``bf16_rn(f32(gelu_tanh_bf16(g)) * (f32(u) + 0.0f))``, every NaN 0x7FFF."""
    a = gelu_tanh_bf16(C, g)
    u0 = list(u[:15]) + [C.AND(u[15], or_reduce(C, u[:15]))]
    return fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(a), fp.bf16_to_f32(u0)), NAN_F2FP)


GeluTanhMulBf16 = definition("GeluTanhMulBf16", 2, gelu_tanh_mul, (("g", BF16), ("u", BF16)), BF16, word=D.GeluTanhMulBf16,
                             doc="vLLM act_and_mul_kernel<gelu_tanh> on bits: GeluTanhMulBf16_v1, "
                                 "bf16(f32(bf16(gelu_tanh(g))) * (f32(u) + 0.0f)), NaN 0x7FFF")


@composite("GeluTanhMul", 2, ["I"], lambda S: ((("gu", Array(2 * S.I, BF16)),), Array(S.I, BF16)),
           conformance="exact-model-tested")
def GeluTanhMul(B, S, gu):
    """vLLM `gelu_tanh_and_mul` on bits: `out[i] = GeluTanhMulBf16_v2(gu[i], gu[I + i])`."""
    return B.batch(GeluTanhMulBf16, gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))


GeluTanhMul.word = lambda b: bind(D.GeluTanhMul, I=b["I"])  # type: ignore[attr-defined]
