"""Down's A′ check beside C2-O: the row amax that down's A′ is quantized under, bound to the row by one committed argmax.
It is experimental, beside `dither_c2o`, and nothing of it is in a Lean lock.

**A′.** Gate-up's call (vLLM's merged [gate ‖ up] layout, n = 2I) gives each row i of the token batch

    h_ij = SiluMulBf16_v2(dequant(O_i,j, s_a_i, s_w_j), dequant(O_i,j+I, s_a_i, s_w_j+I))     (j < I, bf16)
    A′_ij = requant(h_ij, amax_i),   amax_i = max_j |h_ij|                                    (int8, one byte a cell)

with O = (A + d_A)(B + d_B)ᵀ exact (`dither_c2o`'s served call: v3.2's dithers, no noise), `dequant` the served epilogue
(`h1zy.dequant`, `PouwZyDequant_v2`) and requant the served per-row quantizer, whose scale is
RN32(max(amax_i, RN32(1e-30)) / 127). A′ is down's A. Down's dequant reads the same committed amax_i.

**The record.** A′ stays at one byte a cell. Per row the producer commits only a record of two u16 words, amax_i (the bf16
word of max_j |h_ij|, an fmaxf fold from +0 that skips a NaN h) and k_i, one column where |h_ik| = amax_i (`record_of`,
`RECORD`). Nothing per cell is committed beyond A′.

**The checks.** Two blocks hold amax_i to the row:

* each sampled gate cell's block checks |h_ij| ≤ amax_i, a NaN h aside (`PouwAmaxBound_v2`, beside its A′ = requant(h,
  amax) check);
* the sampled down block of row i recomputes h at k_i from gate-up's row i and B's rows k_i and k_i + I, and checks k_i < I
  and |h_ik| = amax_i with one 16-bit equality, amax_i no NaN (`PouwDitherAmax_v1`).

An amax_i above the row's maximum is attained by no column, so the down block fails at every k_i. One below it is passed
by the true maximum's cell, whose gate block then fails. Rounding can't hide the gap, because the bound is on the bf16
magnitudes, not on the codes. A row whose every h is NaN has no k_i, since its amax is +0. That edge is open.

**The down block** `PouwDitherAmax_v1{K, I}`:

    (a: Array<K, i8>, da: Array<K, bit>, bg, bu: Array<K, i8>, dbg, dbu: Array<K, bit>, sa, swg, swu: f32,
     k: u16, amax: bf16) -> ok

over gate-up's row i of A and its dither, B's rows k_i (gate) and k_i + I (up) and their dithers, the three scales and the
row's record. Its sums run through `dither.PouwDitherOutDot0_v2` and `PouwDitherOutDot_v2` (full 32-bit, exact below
`dither.K_LIMIT`), each dequant through `h1zy.PouwZyDequant_v2`, SiLU·mul through `PouwSiluMulBf16_v2`, and the check
through `PouwAmaxAt_v2` against the constant I. Its only output is the pass bit. The opened B rows' positions are k_i and
k_i + I, which the openings hold to the record.

`PouwSiluMulBf16` is the served `SiluMulBf16_v2` (vLLM `_C.silu_and_mul`, ``f / (1 + expf(-f))``) restated here because
PoUW's verifier imports nothing from `verity.ml.definitions`. Its Boolean form is that Definition's
(`SiluMulBf16_v4`), and the tests hold its word view to the served one on every gate word.
"""
from __future__ import annotations

import math
import struct
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

from . import dither as D
from . import h1zy as H

BIT = BL.BIT
I8, U16, BF16, F32 = Array(8, BIT), Array(16, BIT), Array(16, BIT), Array(32, BIT)
M16, M32 = (1 << 16) - 1, (1 << 32) - 1
#: a row's record: amax (bf16), then k (u16), little-endian
RECORD = 4
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
    """The served amax on bf16 words: the bf16 word of max |h| over the row, a fold from +0 that skips a NaN."""
    return max((magnitude(h) for h in hs if magnitude(h) <= MAG_MAX), default=0)


def record_of(hs: Sequence[int]) -> tuple[int, int]:
    """Row i's record (amax_i, k_i): `abs_max` and the first column attaining it. A row whose every h is NaN has none."""
    amax = abs_max(hs)
    for k, h in enumerate(hs):
        if magnitude(h) == amax:
            return amax, k
    raise ValueError("a row whose every h is NaN has no column attaining its amax (+0)")


def record_bytes(amax: int, k: int) -> bytes:
    """A record's `RECORD` bytes: amax, then k."""
    return struct.pack("<2H", amax & M16, k & M16)


@primitive("PouwAmaxAt", 1, [("k", Value(16)), ("n", Value(16)), ("h", Value(16)), ("amax", Value(16))], Value(1),
           conformance="protocol-defined")
def PouwAmaxAtWord(k: int, n: int, h: int, amax: int) -> int:
    """1 when k < n, amax is no NaN and |h| = amax: the down block's check of row i's record at its k."""
    return int(k & M16 < n & M16 and amax & M16 <= MAG_MAX and magnitude(h) == amax & M16)


@primitive("PouwAmaxBound", 1, [("h", Value(16)), ("amax", Value(16))], Value(1), conformance="protocol-defined")
def PouwAmaxBoundWord(h: int, amax: int) -> int:
    """1 when |h| ≤ amax or h is NaN: a gate cell's bound under its row's amax."""
    return int(magnitude(h) <= amax & M16 or magnitude(h) > MAG_MAX)


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


def _amax_at_bits(C, k, n, h, amax):
    _, lt = sub(C, k, n, 16)
    _, nan = sub(C, const_bits(MAG_MAX, 16), amax, 16)
    eq = and_reduce(C, [NOT(x ^ y) for x, y in zip(list(h[:15]) + [ZERO], amax)])
    return [C.AND(C.AND(lt, NOT(nan)), eq)]


def _amax_bound_bits(C, h, amax):
    mag = list(h[:15]) + [ZERO]
    _, over = sub(C, amax, mag, 16)
    _, nan = sub(C, const_bits(MAG_MAX, 16), mag, 16)
    return [OR(C, NOT(over), nan)]


PouwSiluMulBf16 = definition("PouwSiluMulBf16", 2, _silu_mul_bits, (("g", BF16), ("u", BF16)), BF16, word=PouwSiluMulBf16Word,
                             conformance="protocol-defined",
                             doc="The served SiluMulBf16_v2 on bits (SiluMulBf16_v4's form): PouwSiluMulBf16_v1")
PouwAmaxAt = definition("PouwAmaxAt", 2, _amax_at_bits, (("k", U16), ("n", U16), ("h", BF16), ("amax", BF16)), BIT,
                        word=PouwAmaxAtWord, conformance="protocol-defined",
                        doc="k < n, amax no NaN and |h| = amax on bits: PouwAmaxAt_v1")
PouwAmaxBound = definition("PouwAmaxBound", 2, _amax_bound_bits, (("h", BF16), ("amax", BF16)), BIT, word=PouwAmaxBoundWord,
                           conformance="protocol-defined", doc="|h| <= amax or h NaN on bits: PouwAmaxBound_v1")


# -- the down block ----------------------------------------------------------------------------------------------------------------

def _signature(S: Any) -> tuple:
    D._length("PouwDitherAmax", "K", S.K)
    if not isinstance(S.I, int) or isinstance(S.I, bool) or not 0 < S.I < 1 << 16:
        raise ValueError(f"PouwDitherAmax: I, gate-up's half width, is in [1, 2^16), not {S.I!r}")
    row, dither = Array(S.K, I8), Array(S.K, BIT)
    return (("a", row), ("da", dither), ("bg", row), ("bu", row), ("dbg", dither), ("dbu", dither), ("sa", F32),
            ("swg", F32), ("swu", F32), ("k", U16), ("amax", BF16)), BIT


def _dot(B: Any, K: int, a, da, b, db) -> Any:
    acc = B.call(D.PouwDitherOutDot0, a[0:32], da[0:32], b[0:32], db[0:32])
    for q in range(32, K, 32):
        acc = B.call(D.PouwDitherOutDot, acc, a[q:q + 32], da[q:q + 32], b[q:q + 32], db[q:q + 32])
    return acc


def _body(B: Any, S: Any, a, da, bg, bu, dbg, dbu, sa, swg, swu, k, amax) -> Any:
    hg = B.call(H.PouwZyDequant, _dot(B, S.K, a, da, bg, dbg), sa, swg)
    hu = B.call(H.PouwZyDequant, _dot(B, S.K, a, da, bu, dbu), sa, swu)
    h = B.call(PouwSiluMulBf16, hg, hu)
    bit = [B.call(BL.constant(v)).refs[0] for v in (0, 1)]
    n = Coll(U16, TR._refs([bit[S.I >> e & 1] for e in range(16)]))
    return B.call(PouwAmaxAt, k, n, h, amax)


PouwDitherAmax = CompositeDefinition(
    "PouwDitherAmax", 1, ("K", "I"), _signature, _body, conformance="protocol-defined",
    doc="Down's A' check of one row (`circuit.dither_amax`): gate-up's row a with its dither da, B's gate row bg and up row "
        "bu (at columns k and k + I) with their dithers, the scales sa, swg, swu and the row's record (k, amax), to 1 exactly "
        "when k < I, amax is no NaN and |SiluMulBf16_v2(dequant(zg, sa, swg), dequant(zu, sa, swu))| = amax, zg and zu the "
        "exact sums sum_j (a_j + da_j)(b_j + db_j).")


def block_definition(K: int, I: int) -> Any:
    """The down block's A′ check of a gate-up call at contraction K and half width I: `PouwDitherAmax_v1{K, I}`."""
    return bind(PouwDitherAmax, K=K, I=I)


# -- a block in the reference ------------------------------------------------------------------------------------------------------

def block_inputs(a: Sequence[int], da: Sequence[int], bg: Sequence[int], bu: Sequence[int], dbg: Sequence[int],
                 dbu: Sequence[int], sa: int, swg: int, swu: int, k: int, amax: int) -> list[list[int]]:
    """The down block's inputs, as word lists: the K int8 entries and K dither bits of a, bg and bu, the three f32 scales,
    k and amax."""
    K = len(a)
    if not (len(da) == len(bg) == len(bu) == len(dbg) == len(dbu) == K):
        raise ValueError(f"a, its dither and B's two rows and theirs are K = {K} entries each")
    return [D._int8s(a), [x & 1 for x in da], D._int8s(bg), D._int8s(bu), [x & 1 for x in dbg], [x & 1 for x in dbu],
            [sa & M32], [swg & M32], [swu & M32], [k & M16], [amax & M16]]


def block_bits(ins: Sequence[Sequence[int]]) -> list[list[int]]:
    """`block_inputs` on bits, as the Definition reads them (each word least significant bit first)."""
    return [[(v >> e) & 1 for v in words for e in range(n)]
            for words, n in zip(ins, (8, 1, 8, 8, 1, 1, 32, 32, 32, 16, 16), strict=True)]


def _sum(a, da, b, db) -> int:
    return sum((D._sx8(x) + (p & 1)) * (D._sx8(y) + (q & 1)) for x, p, y, q in zip(a, da, b, db, strict=True))


def check_block(ins: Sequence[Sequence[int]], I: int) -> int:
    """`PouwDitherAmax`'s pass bit on `block_inputs`, in the clear."""
    a, da, bg, bu, dbg, dbu, (sa,), (swg,), (swu,), (k,), (amax,) = ins
    h = act_h(_sum(a, da, bg, dbg), _sum(a, da, bu, dbu), sa, swg, swu)
    return PouwAmaxAtWord.evaluate(k, I, h, amax)


__all__ = ["MAG_MAX", "PouwAmaxAt", "PouwAmaxBound", "PouwDitherAmax", "PouwSiluMulBf16", "RECORD", "abs_max", "act_h",
           "block_bits", "block_definition", "block_inputs", "check_block", "magnitude", "record_bytes", "record_of",
           "silu_f32", "silu_mul"]
