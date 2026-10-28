"""Down's A′ check beside C2-O: the row amax that down's A′ is quantized under, bound to the row by its gate blocks' maxima.
It is experimental, beside `dither_c2o`, and nothing of it is in a Lean lock.

**A′.** Gate-up's call (vLLM's merged [gate ‖ up] layout, n = 2I) gives each row i of the token batch

    h_ij = SiluMulBf16_v2(dequant(O_i,j, s_a_i, s_w_j), dequant(O_i,j+I, s_a_i, s_w_j+I))     (j < I, bf16)
    A′_ij = requant(h_ij, amax_i),   amax_i = max_j |h_ij|                                    (int8, one byte a cell)

with O = (A + d_A)(B + d_B)ᵀ exact (`dither_c2o`'s served call: v3.2's dithers, no noise), `dequant` the served epilogue
(`h1zy.dequant`, `PouwZyDequant_v2`) and requant the served per-row quantizer, whose scale is
RN32(max(amax_i, RN32(1e-30)) / 127). A′ is down's A. Down's dequant reads the same committed amax_i.

**What is committed.** A′ stays at one byte a cell. Gate-up's h columns fall in gate blocks of W columns of one row
(N = I / W a row). Each gate block b commits, per row i, m_i,b = max |h_ij| over its W cells (`block_max`: the bf16 word of
an fmaxf fold from +0 that skips a NaN h), a value under its hiding leaf, beside the row's amax_i. That is 2 / W bytes a
cell beyond A′. Nothing about a position is committed, so no proof opens a position that a value chooses.

**The checks.** Two blocks hold amax_i to the row:

* gate block (i, b) checks m_i,b exactly, by one 16-bit max a cell and one equality: m_i,b = the fold of its cells' |h|
  (`PouwAmaxBlock_v1{W}`, beside its A′ = requant(h, amax_i) check). Every |h| ≤ m_i,b, and one cell attains it, or
  m_i,b = +0 when every cell is NaN;
* the sampled down block of row i reads row i's N maxima, at positions the layout fixes, and checks amax_i = max_b m_i,b
  (`PouwAmaxRow_v1{N}`), N - 1 16-bit maxima and one equality.

If every m_i,b is right, their max is the row's amax, so a wrong amax_i passes the down block only beside a wrong m_i,b.
The wrong m_i,b fails gate block (i, b) whenever that block is drawn: the usual sampled-block soundness, with no position
in the clear. Rounding can't hide the gap, because the checks are on the bf16 magnitudes, not on the codes. A row whose
every h is NaN has amax_i = +0, as the served fold gives it.

**The fallback** commits h itself, 2 bytes a cell, and derives amax and A′ in-circuit from the full row.

`PouwSiluMulBf16` is the served `SiluMulBf16_v2` (vLLM `_C.silu_and_mul`, ``f / (1 + expf(-f))``), the gate block's h,
restated here because PoUW's verifier imports nothing from `verity.ml.definitions`. Its Boolean form is that Definition's
(`SiluMulBf16_v4`), and the tests hold its word view to the served one on every gate word.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce, const_bits, or_reduce, sub
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import Coll, CompositeDefinition, bind, primitive
from verity.ml.circuits.ops import fp32 as F
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array, Value

from . import h1zy as H

BIT = BL.BIT
U16, BF16 = Array(16, BIT), Array(16, BIT)
M16, M32 = (1 << 16) - 1, (1 << 32) - 1
#: the largest bf16 magnitude that is no NaN (+inf)
MAG_MAX = 0x7F80
ONE_BF16 = 0x3F80
#: ln of the smallest real that RN32 sends to +inf, (2 - 2^-24) 2^127: no f32 word lies within float64 error of it
LN_F32_OVERFLOW = math.log((2.0 - 2.0 ** -24) * 2.0 ** 127)


# -- the words -------------------------------------------------------------------------------------------------------------------

def silu_f32(g: int) -> int:
    """The served SiLU on an f32 word, ``f / (1.0f + expf(-f))``: RN32 of g / (1 + e^-g) taken in float64 (g e^g below -80),
    NaN for a NaN or -inf gate, +inf for +inf, -0 for -0 and for every gate whose e^-g overflows f32."""
    c, v = F._decode(g)
    if c in (2, -1):
        return F.F32_NAN
    if c == 1:
        return 0x7F800000
    if g & M32 == 0x80000000 or -float(v) >= LN_F32_OVERFLOW:
        return 0x80000000
    x = float(v)
    return F.bits_of(x * math.exp(x) if x < -80.0 else x / (1.0 + math.exp(-x)))


@primitive("PouwSiluMulBf16", 1, [("g", Value(16)), ("u", Value(16))], Value(16),
           conformance="protocol-defined: the served SiluMulBf16_v2, vLLM _C.silu_and_mul's element")
def PouwSiluMulBf16Word(g: int, u: int) -> int:
    """bf16(f32(bf16(silu_f32(g))) · (f32(u) + 0.0f)), both conversions RN with every NaN 0x7FFF."""
    s = F.f32_to_bf16_rn_bits(silu_f32(F.bf16_to_f32_bits(g)), F.BF16_NAN_F2FP)
    uf = F.f32_add_bits(F.bf16_to_f32_bits(u), 0)
    return F.f32_to_bf16_rn_bits(F.f32_mul_bits(F.bf16_to_f32_bits(s), uf), F.BF16_NAN_F2FP)


def silu_mul(g: int, u: int) -> int:
    return PouwSiluMulBf16Word.evaluate(g & M16, u & M16)


def act_h(zg: int, zu: int, sa: int, swg: int, swu: int) -> int:
    """h of gate cell (i, j) from its exact 32-bit sum zg and its up cell's zu: silu_mul(dequant(zg, sa, swg),
    dequant(zu, sa, swu))."""
    return silu_mul(H.dequant(zg, sa, swg), H.dequant(zu, sa, swu))


def magnitude(h: int) -> int:
    return h & 0x7FFF


def abs_max(hs: Sequence[int]) -> int:
    """The served amax on bf16 words: the bf16 word of max |h|, a fold from +0 that skips a NaN."""
    return max((magnitude(h) for h in hs if magnitude(h) <= MAG_MAX), default=0)


@primitive("PouwAmaxFold", 1, [("acc", Value(16)), ("h", Value(16))], Value(16), conformance="protocol-defined")
def PouwAmaxFoldWord(acc: int, h: int) -> int:
    """One step of `abs_max`: max(acc, |h|), or acc when h is NaN."""
    return acc & M16 if magnitude(h) > MAG_MAX else max(acc & M16, magnitude(h))


@primitive("PouwAmaxMax", 1, [("a", Value(16)), ("b", Value(16))], Value(16), conformance="protocol-defined")
def PouwAmaxMaxWord(a: int, b: int) -> int:
    return max(a & M16, b & M16)


@primitive("PouwAmaxEq", 1, [("a", Value(16)), ("b", Value(16))], Value(1), conformance="protocol-defined")
def PouwAmaxEqWord(a: int, b: int) -> int:
    return int(a & M16 == b & M16)


# -- SiLU·mul on bits --------------------------------------------------------------------------------------------------------------

#: the exponent fields the table covers, per sign: even to odd, so a row is one exponent pair
BAND_LO = 118
BAND_HI = (129, 133)


@lru_cache(maxsize=None)
def band_table() -> tuple[int, ...]:
    """``silu(g) & 0x7FFF`` for every gate g on the band, at index ``m | (e mod 16) << 7 | sign << 11``; 0 off it."""
    out = [0] * (1 << 12)
    for sign in (0, 1):
        for e in range(BAND_LO, BAND_HI[sign] + 1):
            for m in range(128):
                out[m | (e & 0xF) << 7 | sign << 11] = silu_mul(sign << 15 | e << 7 | m, ONE_BF16) & 0x7FFF
    return tuple(out)


def _blocks(ks) -> dict[int, int]:
    out: dict[int, int] = {}
    for k in ks:
        out[k >> 2] = out.get(k >> 2, 0) | 1 << (k & 3)
    return out


def _band(C, g, table, rows):
    """The 15 magnitude bits ``table`` gives the bf16 word g on the band ``rows`` ((sign, e0), e0 even), zero off it: g's
    mantissa and lowest exponent bit decode to 256 one-hot minterms, each row to one minterm, and each output bit of a row
    is one AND of the row's minterm with a fixed XOR of low minterms (or its complement, whichever has fewer terms)."""
    e, s = g[7:15], g[15]
    lo = fp.decode(C, list(g[:7]) + [e[0]])
    hi = fp.decode(C, list(e[1:4]) + [s])
    nibble = {n: and_reduce(C, [e[7 - i] if n >> 3 - i & 1 else NOT(e[7 - i]) for i in range(4)])
              for n in sorted({e0 >> 4 for _, e0 in rows})}
    parts: dict = {}

    def part(b: int, mask: int):
        if (b, mask) not in parts:
            top = mask.bit_length() - 1
            rest = mask ^ 1 << top
            parts[(b, mask)] = part(b, rest) ^ lo[4 * b + top] if rest else lo[4 * b + top]
        return parts[(b, mask)]

    out = [ZERO] * 15
    for sign, e0 in rows:
        r = (e0 & 0xF) >> 1 | sign << 3
        row = C.AND(hi[r], nibble[e0 >> 4])
        for j in range(15):
            ones = [k for k in range(256) if table[r << 8 | k] >> j & 1]
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


def _silu_bits(C, g):
    """bf16_rn(silu_f32(f32(g))): NaN and -inf to 0x7FFF, -0 kept, g / 2 below the band, g above it (gates >= 0) or -0
    (gates < 0), and `band_table` on it."""
    m, e, s = list(g[:7]), list(g[7:15]), g[15]
    nan = C.AND(and_reduce(C, e), OR(C, or_reduce(C, m), s))
    _, below = sub(C, e, const_bits(BAND_LO, 8), 8)
    _, under = sub(C, e, const_bits(BAND_HI[0] + 1, 8), 8)
    keep = C.AND(C.AND(NOT(s), NOT(under)), NOT(nan))
    e01 = NOT(or_reduce(C, e[1:]))
    low = C.AND(below, e01)
    high = below ^ low
    half_low = add(C, m[1:] + [e[0]], [C.AND(m[0], m[1])], width=8)
    half_high = m + sub(C, e, [ONE], 8)[0]
    on_band = _band(C, g, band_table(), [(sign, e0) for sign in (0, 1) for e0 in range(BAND_LO, BAND_HI[sign] + 1, 2)])
    mag = [x ^ C.AND(low, y) ^ C.AND(high, z) ^ C.AND(keep, w)
           for x, y, z, w in zip(on_band, half_low + [ZERO] * 7, half_high, g[:15])]
    mag = [x ^ nan if F.BF16_NAN_F2FP >> j & 1 else x for j, x in enumerate(mag)]
    return mag + [C.AND(s, NOT(nan))]


def _silu_mul_bits(C, g, u):
    s = _silu_bits(C, g)
    up = list(u[:15]) + [C.AND(u[15], or_reduce(C, u[:15]))]
    return fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(s), fp.bf16_to_f32(up)), F.BF16_NAN_F2FP)


# -- the max leaves on bits --------------------------------------------------------------------------------------------------------

def _fold_bits(C, acc, h):
    mag = list(h[:15]) + [ZERO]
    _, nan = sub(C, const_bits(MAG_MAX, 16), mag, 16)
    _, up = sub(C, acc, mag, 16)
    take = C.AND(up, NOT(nan))
    return [a ^ C.AND(take, a ^ m) for a, m in zip(acc, mag)]


def _max_bits(C, a, b):
    _, up = sub(C, a, b, 16)
    return [x ^ C.AND(up, x ^ y) for x, y in zip(a, b)]


def _eq_bits(C, a, b):
    return [and_reduce(C, [NOT(x ^ y) for x, y in zip(a, b)])]


PouwSiluMulBf16 = definition("PouwSiluMulBf16", 2, _silu_mul_bits, (("g", BF16), ("u", BF16)), BF16, word=PouwSiluMulBf16Word,
                             conformance="protocol-defined",
                             doc="The served SiluMulBf16_v2 on bits (SiluMulBf16_v4's form): PouwSiluMulBf16_v1")
PouwAmaxFold = definition("PouwAmaxFold", 2, _fold_bits, (("acc", U16), ("h", BF16)), U16, word=PouwAmaxFoldWord,
                          conformance="protocol-defined", doc="max(acc, |h|), acc when h is NaN, on bits: PouwAmaxFold_v1")
PouwAmaxMax = definition("PouwAmaxMax", 2, _max_bits, (("a", U16), ("b", U16)), U16, word=PouwAmaxMaxWord,
                         conformance="protocol-defined", doc="max(a, b) of two u16 on bits: PouwAmaxMax_v1")
PouwAmaxEq = definition("PouwAmaxEq", 2, _eq_bits, (("a", U16), ("b", U16)), BIT, word=PouwAmaxEqWord,
                        conformance="protocol-defined", doc="a = b of two u16 on bits: PouwAmaxEq_v1")


# -- the gate block's and the down block's checks ------------------------------------------------------------------------------------

def _count(name: str, what: str, v: Any) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or not 0 < v < 1 << 16:
        raise ValueError(f"{name}: {what} is in [1, 2^16), not {v!r}")


def _block_signature(S: Any) -> tuple:
    _count("PouwAmaxBlock", "W, a gate block's h cells of one row", S.W)
    return (("h", Array(S.W, BF16)), ("m", BF16)), BIT


def _block_body(B: Any, S: Any, h, m) -> Any:
    zero = B.call(BL.constant(0)).refs[0]
    acc = Coll(U16, TR._refs([zero] * 16))
    for c in range(S.W):
        acc = B.call(PouwAmaxFold, acc, h[c])
    return B.call(PouwAmaxEq, acc, m)


def _row_signature(S: Any) -> tuple:
    _count("PouwAmaxRow", "N, a row's gate blocks", S.N)
    return (("amax", BF16), ("m", Array(S.N, BF16))), BIT


def _row_body(B: Any, S: Any, amax, m) -> Any:
    acc = m[0]
    for b in range(1, S.N):
        acc = B.call(PouwAmaxMax, acc, m[b])
    return B.call(PouwAmaxEq, acc, amax)


PouwAmaxBlock = CompositeDefinition(
    "PouwAmaxBlock", 1, ("W",), _block_signature, _block_body, conformance="protocol-defined",
    doc="A gate block's max (`circuit.dither_amax`): its W cells' h of one row and the row's committed block max m, to 1 "
        "exactly when m is the fmaxf fold from +0 of their |h| that skips a NaN h.")

PouwAmaxRow = CompositeDefinition(
    "PouwAmaxRow", 1, ("N",), _row_signature, _row_body, conformance="protocol-defined",
    doc="Down's A' amax check of one row (`circuit.dither_amax`): the row's committed amax and its N gate blocks' committed "
        "maxima m, to 1 exactly when amax = max_b m_b.")


def gate_definition(W: int) -> Any:
    """A gate block's max check over W h cells of one row: `PouwAmaxBlock_v1{W}`."""
    return bind(PouwAmaxBlock, W=W)


def row_definition(I: int, W: int) -> Any:
    """The down block's amax check of a gate-up call at half width I, in gate blocks of W columns: `PouwAmaxRow_v1{N}`,
    N = I / W."""
    if not isinstance(W, int) or isinstance(W, bool) or W < 1 or not isinstance(I, int) or isinstance(I, bool) or I % W:
        raise ValueError(f"gate blocks of W = {W!r} columns tile gate-up's half width I = {I!r}")
    return bind(PouwAmaxRow, N=I // W)


# -- the reference ----------------------------------------------------------------------------------------------------------------

def block_maxes(hs: Sequence[int], W: int) -> list[int]:
    """Row i's committed m_i,b: `abs_max` of each gate block's W columns."""
    if W < 1 or len(hs) % W:
        raise ValueError(f"gate blocks of W = {W} columns tile a row of {len(hs)}")
    return [abs_max(hs[b:b + W]) for b in range(0, len(hs), W)]


def check_gate(hs: Sequence[int], m: int) -> int:
    """`PouwAmaxBlock`'s pass bit, in the clear."""
    return int(abs_max(hs) == m & M16)


def check_row(amax: int, ms: Sequence[int]) -> int:
    """`PouwAmaxRow`'s pass bit, in the clear."""
    return int(max(m & M16 for m in ms) == amax & M16)


def bits(words: Sequence[int], n: int = 16) -> list[int]:
    """Words as a Definition reads them, each least significant bit first."""
    return [(w >> e) & 1 for w in words for e in range(n)]


__all__ = ["MAG_MAX", "PouwAmaxBlock", "PouwAmaxEq", "PouwAmaxFold", "PouwAmaxMax", "PouwAmaxRow", "PouwSiluMulBf16",
           "abs_max", "act_h", "bits", "block_maxes", "check_gate", "check_row", "gate_definition", "magnitude",
           "row_definition", "silu_f32", "silu_mul"]
