"""vLLM `silu_and_mul` as Boolean Definitions: the next version of each word Definition, the same function on bits.

    SiluMulBf16_v3   (g, u: Array<16, Value<1>>) -> Array<16, Value<1>>                      word view: SiluMulBf16_v1
    SiluMul_v3{I}    (gu: Array<2I, Array<16, Value<1>>>) -> Array<I, Array<16, Value<1>>>   word view: SiluMul_v1{I}

`SiluMulBf16_v3` is traced from `silu_mul` (`verity.ml.boolean.trace.definition`), so its And gates are that builder's live
ANDs: ``s = silu_bf16(g)``, then ``bf16_rn(f32(s) * f32(u))`` through `verity.ml.boolean.fp` (`f32_mul`, then `f32_to_bf16`
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
No sub-Call is a word Definition.
"""
from __future__ import annotations

from functools import lru_cache

from verity.ir import boolean as BL
from verity.ir.defs import bind, composite
from verity.ir.types import Array
from verity.ml.boolean import fp
from verity.ml.boolean.forms import NOT, ONE, ZERO, L, OR, add, and_reduce, const_bits, or_reduce, sub
from verity.ml.boolean.trace import definition
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P

__all__ = ["BAND_HI", "BAND_LO", "SiluMul", "SiluMulBf16", "band_table", "silu_bf16", "silu_mul"]

BF16 = Array(16, BL.BIT)
NAN_TORCH = P.BF16_NAN_TORCH
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


def _blocks(ks) -> dict[int, int]:
    out: dict[int, int] = {}
    for k in ks:
        out[k >> 2] = out.get(k >> 2, 0) | 1 << (k & 3)
    return out


def _band(C, g):
    """The band's 15 magnitude bits, zero off the band. A column's XOR of low minterms is a chain over blocks of four in
    ascending order, each block's part a chain over its minterms, so equal parts and equal prefixes are one wire."""
    m, e, s = g[:7], g[7:15], g[15]
    lo = fp.decode(C, list(m) + [e[0]])                       # index bits 0..7
    hi = fp.decode(C, list(e[1:4]) + [s])                     # index bits 8..11
    nibble = {7: and_reduce(C, [NOT(e[7]), e[6], e[5], e[4]]), 8: and_reduce(C, [e[7], NOT(e[6]), NOT(e[5]), NOT(e[4])])}
    parts: dict = {}

    def part(b: int, mask: int):
        if (b, mask) not in parts:
            top = mask.bit_length() - 1
            rest = mask ^ 1 << top
            parts[(b, mask)] = part(b, rest) ^ lo[4 * b + top] if rest else lo[4 * b + top]
        return parts[(b, mask)]

    table = band_table()
    out = [ZERO] * 15
    for sign in (0, 1):
        for e0 in range(BAND_LO, BAND_HI[sign] + 1, 2):
            h = (e0 & 0xF) >> 1 | sign << 3
            row = C.AND(hi[h], nibble[e0 >> 4])
            for j in range(15):
                ones = [k for k in range(256) if table[h << 8 | k] >> j & 1]
                if len(ones) == 256:
                    out[j] = out[j] ^ row
                elif ones:
                    blocks, zeros = _blocks(ones), _blocks(sorted(set(range(256)) - set(ones)))
                    pick, flip = (blocks, False) if len(blocks) <= len(zeros) else (zeros, True)
                    col = None
                    for b in sorted(pick):
                        col = part(b, pick[b]) if col is None else col ^ part(b, pick[b])
                    out[j] = out[j] ^ C.AND(row, NOT(col) if flip else col)
    return out


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


SiluMulBf16 = definition("SiluMulBf16", 3, silu_mul, (("g", BF16), ("u", BF16)), BF16, word=P.SiluMulBf16,
                         doc="vLLM act_and_mul_kernel<silu> on bits: SiluMulBf16_v1, bf16(f32(bf16(silu_f32(g))) * f32(u))")


@composite("SiluMul", 3, ["I"], lambda S: ((("gu", Array(2 * S.I, BF16)),), Array(S.I, BF16)),
           conformance="exact-model-tested")
def SiluMul(B, S, gu):
    """vLLM `silu_and_mul` on bits: `out[i] = SiluMulBf16_v3(gu[i], gu[I + i])`."""
    return B.batch(SiluMulBf16, gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))


SiluMul.word = lambda b: bind(b1.SiluMul, I=b["I"])  # type: ignore[attr-defined]
