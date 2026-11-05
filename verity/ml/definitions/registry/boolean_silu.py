"""vLLM `silu_and_mul` as Boolean Definitions: the next version of each word Definition, the same function on bits.

    SiluMulBf16_v3   (g, u: Array<16, Value<1>>) -> Array<16, Value<1>>                      word view: SiluMulBf16_v1
    SiluMul_v3{I}    (gu: Array<2I, Array<16, Value<1>>>) -> Array<I, Array<16, Value<1>>>   word view: SiluMul_v1{I}
    SiluMulBf16_v4   (g, u: Array<16, Value<1>>) -> Array<16, Value<1>>                      word view: SiluMulBf16_v2
    SiluMul_v4{I}    (gu: Array<2I, Array<16, Value<1>>>) -> Array<I, Array<16, Value<1>>>   word view: SiluMul_v2{I}

`SiluMulBf16_v3` is traced from `silu_mul` (`verity.ml.circuits.boolean.trace.definition`), so its And gates are that builder's live
ANDs: ``s = silu_bf16(g)``, then ``bf16_rn(f32(s) * f32(u))`` through `verity.ml.circuits.silicon.fp` (`f32_mul`, then `f32_to_bf16`
with torch's NaN 0x7FC0). ``s = bf16_rn(silu_f32(f32(g)))`` is a function of the 16-bit g alone (`SiluMulBf16_v1` at u = 1.0).
Off a band of exponent fields e it is exact arithmetic on g's fields; on the band it reads a constant table of the word
Definition's own values:

    g NaN                          0x7FC0
    e < BAND_LO                    bf16_rn(g / 2), and -0 gives +0
    g >= 0, e > BAND_HI[0]         g (+inf included)
    g < 0,  e > BAND_HI[1]         -0 (-inf included)
    BAND_LO <= e <= BAND_HI[sign]  g's sign, and the 15 magnitude bits of `band_table()`

The table is a multiplexer over constant bits: g's mantissa and lowest exponent bit decode to 256 one-hot minterms, and each
band row (a sign and an exponent pair e, e + 1) to one minterm; each output bit of a row is one AND of the row's minterm with
a fixed XOR of low minterms (or its complement, whichever has fewer terms). `SiluMul_v3{I}` is `SiluMul_v1`'s body on bits: a
batch of I `SiluMulBf16_v3` Calls over (gu[i], gu[I + i]), so each member's 16 output bits are one unit under `Q_word_v1`.

The v4 pair (`SiluMulBf16Expf`, `SiluMulExpf`) computes the served `SiluMulBf16_v2` / `SiluMul_v2{I}` (`registry.act`,
vLLM `_C.silu_and_mul` on the measured sm_120 edge words: ``f / (1 + expf(-f))`` with expf's overflow) the same way:
`silu_bf16_expf` sends NaN and -inf to 0x7FFF, keeps -0, and reads `expf_band_table`, where the gates whose ``expf(-g)``
overflows (g <= -89) are -0; the up word passes ``+ 0.0f`` (-0 to +0), and every NaN result is 0x7FFF.  No sub-Call is a word
Definition.
"""
from __future__ import annotations

from functools import lru_cache

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.forms import NOT, ONE, ZERO, OR, add, and_reduce, const_bits, or_reduce, sub
from verity.ml.circuits.boolean.trace import definition
from verity.ml.definitions.registry import act, b1
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.boolean_activation import band

__all__ = ["BAND_HI", "BAND_LO", "SiluMul", "SiluMulBf16", "SiluMulBf16Expf", "SiluMulExpf", "band_table", "expf_band_table",
           "silu_bf16", "silu_bf16_expf", "silu_mul", "silu_mul_expf"]

BF16 = Array(16, BL.BIT)
NAN_TORCH = P.BF16_NAN_TORCH
NAN_F2FP = P.BF16_NAN_F2FP
ONE_BF16 = 0x3F80

#: the exponent fields the table covers, per sign: even to odd, so a row is one exponent pair
BAND_LO = 118
BAND_HI = (129, 133)


@lru_cache(maxsize=None)
def band_table() -> tuple[int, ...]:
    """``silu_bf16(g) & 0x7FFF`` for every g on the band, at index ``m | (e mod 16) << 7 | sign << 11``; 0 off it."""
    out = [0] * (1 << 12)
    for sign in (0, 1):
        for e in range(BAND_LO, BAND_HI[sign] + 1):
            for m in range(128):
                out[m | (e & 0xF) << 7 | sign << 11] = P.SiluMulBf16.evaluate(sign << 15 | e << 7 | m, ONE_BF16) & 0x7FFF
    return tuple(out)


def _band(C, g):
    """The band's 15 magnitude bits, zero off the band (`boolean_activation.band` over `band_table`)."""
    return band(C, g, band_table(), [(sign, e0) for sign in (0, 1) for e0 in range(BAND_LO, BAND_HI[sign] + 1, 2)])


def silu_bf16(C, g):
    """``bf16_rn(silu_f32(f32(g)))``, the silu half of `SiluMulBf16_v1`, for a 16-bit g (LSB first)."""
    m, e, s = list(g[:7]), list(g[7:15]), g[15]
    e_nz, m_nz = or_reduce(C, e), or_reduce(C, m)
    nan = C.AND(and_reduce(C, e), m_nz)
    _, below = sub(C, e, const_bits(BAND_LO, 8), 8)
    _, under = sub(C, e, const_bits(BAND_HI[0] + 1, 8), 8)
    keep = C.AND(C.AND(NOT(s), NOT(under)), NOT(nan))                      # g >= 0 above the band, not NaN
    # g / 2 below the band: the exponent field less one, or for e <= 1 the 8-bit significand halved, ties to even
    e01 = NOT(or_reduce(C, e[1:]))
    low = C.AND(below, e01)
    high = below ^ low
    half_low = add(C, m[1:] + [e[0]], [C.AND(m[0], m[1])], width=8)
    half_high = m + sub(C, e, [ONE], 8)[0]
    mag = [x ^ C.AND(low, y) ^ C.AND(high, z) ^ C.AND(keep, w)
           for x, y, z, w in zip(_band(C, g), half_low + [ZERO] * 7, half_high, g[:15])]
    mag = [x ^ nan if NAN_TORCH >> j & 1 else x for j, x in enumerate(mag)]
    sign = C.AND(s, NOT(nan ^ NOT(OR(C, e_nz, m_nz))))                      # NaN and +-0 exclusive
    return mag + [sign]


def silu_mul(C, g, u):
    """`SiluMulBf16_v1`: ``bf16_rn(f32(silu_bf16(g)) * f32(u))``, every NaN 0x7FC0."""
    s = silu_bf16(C, g)
    return fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(s), fp.bf16_to_f32(u)), NAN_TORCH)


@lru_cache(maxsize=None)
def expf_band_table() -> tuple[int, ...]:
    """`band_table` of `SiluMulBf16_v2`: on the band only the gates whose ``expf(-g)`` overflows differ, each 0 (``g / inf``)."""
    out = [0] * (1 << 12)
    for sign in (0, 1):
        for e in range(BAND_LO, BAND_HI[sign] + 1):
            for m in range(128):
                out[m | (e & 0xF) << 7 | sign << 11] = act.SiluMulBf16.evaluate(sign << 15 | e << 7 | m, ONE_BF16) & 0x7FFF
    return tuple(out)


def silu_bf16_expf(C, g):
    """``bf16_rn(silu_f32(f32(g)))``, the silu half of `SiluMulBf16_v2`: `silu_bf16` with -inf and NaN to 0x7FFF, -0 kept, and
    `expf_band_table` on the band."""
    m, e, s = list(g[:7]), list(g[7:15]), g[15]
    nan = C.AND(and_reduce(C, e), OR(C, or_reduce(C, m), s))               # NaN, and -inf / (1 + inf)
    _, below = sub(C, e, const_bits(BAND_LO, 8), 8)
    _, under = sub(C, e, const_bits(BAND_HI[0] + 1, 8), 8)
    keep = C.AND(C.AND(NOT(s), NOT(under)), NOT(nan))
    e01 = NOT(or_reduce(C, e[1:]))
    low = C.AND(below, e01)
    high = below ^ low
    half_low = add(C, m[1:] + [e[0]], [C.AND(m[0], m[1])], width=8)
    half_high = m + sub(C, e, [ONE], 8)[0]
    on_band = band(C, g, expf_band_table(), [(sign, e0) for sign in (0, 1) for e0 in range(BAND_LO, BAND_HI[sign] + 1, 2)])
    mag = [x ^ C.AND(low, y) ^ C.AND(high, z) ^ C.AND(keep, w)
           for x, y, z, w in zip(on_band, half_low + [ZERO] * 7, half_high, g[:15])]
    mag = [x ^ nan if NAN_F2FP >> j & 1 else x for j, x in enumerate(mag)]
    return mag + [C.AND(s, NOT(nan))]


def silu_mul_expf(C, g, u):
    """`SiluMulBf16_v2`: ``bf16_rn(f32(silu_bf16_expf(g)) * (f32(u) + 0.0f))``, every NaN 0x7FFF; ``+ 0.0f`` sends -0 to +0."""
    s = silu_bf16_expf(C, g)
    up = list(u[:15]) + [C.AND(u[15], or_reduce(C, u[:15]))]
    return fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(s), fp.bf16_to_f32(up)), NAN_F2FP)


SiluMulBf16 = definition("SiluMulBf16", 3, silu_mul, (("g", BF16), ("u", BF16)), BF16, word=P.SiluMulBf16,
                         doc="vLLM act_and_mul_kernel<silu> on bits: SiluMulBf16_v1, bf16(f32(bf16(silu_f32(g))) * f32(u))")


def _silu_mul_rows(B, S, gu):
    """vLLM `silu_and_mul` on bits: `out[i] = SiluMulBf16_v3(gu[i], gu[I + i])`."""
    return B.batch(SiluMulBf16, gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))


SiluMul = CompositeDefinition("SiluMul", 3, ("I",), lambda S: ((("gu", Array(2 * S.I, BF16)),), Array(S.I, BF16)),
                              _silu_mul_rows, doc=_silu_mul_rows.__doc__, conformance="exact-model-tested")
SiluMul.word = lambda b: bind(b1.SiluMul, I=b["I"])  # type: ignore[attr-defined]

SiluMulBf16Expf = definition("SiluMulBf16", 4, silu_mul_expf, (("g", BF16), ("u", BF16)), BF16, word=act.SiluMulBf16,
                             doc="vLLM _C.silu_and_mul's element on bits: SiluMulBf16_v2, "
                                 "bf16(f32(bf16(silu_f32(g))) * (f32(u) + 0.0f)), NaN 0x7FFF")


def _silu_mul_expf_rows(B, S, gu):
    """vLLM `_C.silu_and_mul` on bits: `out[i] = SiluMulBf16_v4(gu[i], gu[I + i])`."""
    return B.batch(SiluMulBf16Expf, gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))


SiluMulExpf = CompositeDefinition("SiluMul", 4, ("I",), lambda S: ((("gu", Array(2 * S.I, BF16)),), Array(S.I, BF16)),
                                  _silu_mul_expf_rows, doc=_silu_mul_expf_rows.__doc__, conformance="exact-model-tested")
SiluMulExpf.word = lambda b: bind(act.SiluMul, I=b["I"])  # type: ignore[attr-defined]
