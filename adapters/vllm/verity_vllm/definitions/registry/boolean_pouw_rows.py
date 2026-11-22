"""`pouw_rows`' quantizer and dot product on bits: the next version of each word primitive, the same function on bits.

    NcpAbsMax_v2       (m, x: Array<32, Value<1>>) -> Array<32, Value<1>>            word view: NcpAbsMax_v1
    NcpRneClampI7_v2   (q: Array<32, Value<1>>) -> Array<7, Value<1>>                word view: NcpRneClampI7_v1
    NcpDot16I7_v2      (acc: Array<32, Value<1>>, a, b: Array<16, Array<7, Value<1>>>) -> Array<32, Value<1>>
                                                                                     word view: NcpDot16I7_v1

`NcpRneClampI7_v2` rounds |q| to nearest even at each exponent where |q| is in [0.5, 64), a one-hot select over those seven, 63
from 63.5 up (infinity and NaN too), then 0 for a NaN and the sign applied in two's complement.  `NcpDot16I7_v2` puts every
partial product of the 16 int7 pairs in one carry-save sum (`words.sum_mod`), the products of one sign bit and one magnitude bit
complemented with their weight subtracted in the constant (Baugh-Wooley).
"""
from __future__ import annotations

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce, const_bits, mux, or_reduce, sub
from verity.circuits.boolean.trace import definition
from verity.circuits.ops.boolean.words import sum_mod
from verity.circuits.types import Array

from verity_vllm.definitions.registry import pouw_rows as PR

BIT = BL.BIT
F32 = U32 = Array(32, BIT)
I7 = Array(7, BIT)
#: 63.5's f32 encoding: |q| at or above it clamps to 63
CLAMP_BITS = 0x427E0000


def abs_max(C, m, x):
    """max(m, |x|) as unsigned words."""
    a = list(x[:31]) + [ZERO]
    return mux(C, sub(C, m, a, 32)[1], a, m)


def rne_clamp_i7(C, q):
    """`NcpRneClampI7_v1` on bits."""
    man, exp, sign = list(q[:23]), list(q[23:31]), q[31]
    below = [man[0]]
    for b in man[1:]:
        below.append(OR(C, below[-1], b))
    mag = [ZERO] * 6
    for k in range(-1, 6):
        if k < 0:
            whole, r, s = [], ONE, below[22]
        else:
            whole, r, s = man[23 - k:] + [ONE], man[22 - k], below[21 - k]
        up = C.AND(r, OR(C, s, whole[0] if whole else ZERO))
        hit = and_reduce(C, [b if (127 + k) >> i & 1 else NOT(b) for i, b in enumerate(exp)])
        mag = [y ^ C.AND(hit, v) for y, v in zip(mag, add(C, whole, [up], width=6))]
    clamp = NOT(sub(C, list(q[:31]), const_bits(CLAMP_BITS, 31), 31)[1])
    keep = NOT(C.AND(and_reduce(C, exp), or_reduce(C, man)))
    mag = [C.AND(keep, OR(C, clamp, b)) for b in mag]
    return add(C, [b ^ sign for b in mag] + [sign], [sign], width=7)


def dot16_i7(C, acc, *ab):
    """`NcpDot16I7_v1` on bits: acc + Σ s7(a_i)·s7(b_i) mod 2^32."""
    a, b = ab[:16], ab[16:]
    cols = [[w] for w in acc]
    k = 0
    for x, y in zip(a, b):
        for i in range(7):
            for j in range(7):
                p = C.AND(x[i], y[j])
                if (i == 6) != (j == 6):
                    cols[i + j].append(NOT(p))
                    k -= 1 << (i + j)
                else:
                    cols[i + j].append(p)
    for i in range(32):
        if k % (1 << 32) >> i & 1:
            cols[i].append(ONE)
    return sum_mod(C, cols, 32)


NcpAbsMax = definition("NcpAbsMax", 2, abs_max, (("m", F32), ("x", F32)), F32, word=PR.NcpAbsMax,
                       conformance=PR.NcpAbsMax.conformance, doc="max(m, |x|) on encodings, on bits: NcpAbsMax_v1")
NcpRneClampI7 = definition("NcpRneClampI7", 2, rne_clamp_i7, (("q", F32),), I7, word=PR.NcpRneClampI7,
                           conformance=PR.NcpRneClampI7.conformance,
                           doc="clamp(RNE(q), -63, 63) as an int7 encoding, a NaN 0, on bits: NcpRneClampI7_v1")
NcpDot16I7 = definition("NcpDot16I7", 2, dot16_i7, (("acc", U32), ("a", Array(16, I7)), ("b", Array(16, I7))), U32,
                        word=PR.NcpDot16I7, conformance=PR.NcpDot16I7.conformance,
                        doc="acc + Σ a_i·b_i over 16 int7 products mod 2^32, on bits: NcpDot16I7_v1")

__all__ = ["NcpAbsMax", "NcpDot16I7", "NcpRneClampI7", "abs_max", "dot16_i7", "rne_clamp_i7"]
