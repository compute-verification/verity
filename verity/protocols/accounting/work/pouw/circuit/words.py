"""PoUW's IR primitives: the word operations `ncp-v2`'s circuit is built from, the three Pearl-C4's tile check (`pc4`)
adds to core's FP32, cast and tensor-core primitives, the three exact sums of squares Pearl-C's (`pc8`) liveness
reads, the masked words of a promotion window and an element's salt-dead bit that the prototype of Pearl-C's cap
(`cap`) reads, and the -h3 tile leaf of one word that Pearl-C's served unit returns (`PouwPc8LeafH3`).

Every primitive but `PouwSqAcc` (576 bits), `PouwSqCeil` (64), the two windows (480) and the -h3 leaf (512) returns one
value of at most 32 bits, so every unit of any `Q_word` cut of these Definitions meets the width rule, and a 64-bit hash
word is carried as two 32-bit halves: SHA-512's words as (hi, lo), Keccak's lanes as (lo, hi).  The two wide sums stay
inside one `pc8` row unit, whose partition is `Q_template_instances`, a window inside its cap word, and the leaf inside
its tile unit.
Encodings: `U32` = `Value<32>`; `I7` = `Value<7>`, a two's-complement int7 (A's and B's entries: the type is the int7 domain
check); `U8` = `Value<8>`, a byte read as u8 (X + β) or s8 (the noise, Y) as each primitive says; `F32`, `BF16` and `E2M1`
the float encodings.  Every evaluator is total on encodings.
"""
from __future__ import annotations

import hashlib
import struct

from verity.primitives.commitments.rowleaf import sha512_row_seg_prefix, sha512_row_seg_segment_prefix
from verity.primitives.circuits.defs import primitive
from verity.primitives.circuits.types import Array, Value
from verity_catalog.definitions import prims as _prims
from verity_catalog.silicon import cast as _cast
from verity_catalog.silicon import fp32 as _fp32

U32, U8, I7, F32, BF16, E2M1 = Value(32), Value(8), Value(7), Value(32), Value(16), Value(4)
M32 = (1 << 32) - 1
M64 = (1 << 64) - 1
#: route U's bias: X + β is u8 on the int7 domain (Lean `biasU8Iff`); `schemes.ncp.BETA`
BETA = 128


def sx(v: int, bits: int) -> int:
    """The two's-complement value of a ``bits``-bit encoding."""
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def _rotr64(x: int, c: int) -> int:
    c &= 63
    return ((x >> c) | (x << (64 - c))) & M64 if c else x


def _halves(x: int) -> tuple[int, int]:
    return x >> 32, x & M32


# ---- 32-bit words ------------------------------------------------------------------------------------------------------

@primitive("PouwXor32", 1, [("a", U32), ("b", U32)], U32, conformance="protocol-defined")
def PouwXor32(a: int, b: int) -> int:
    """a xor b."""
    return (a ^ b) & M32


@primitive("PouwChi32", 1, [("a", U32), ("b", U32), ("c", U32)], U32, conformance="protocol-defined")
def PouwChi32(a: int, b: int, c: int) -> int:
    """Keccak's χ on one half-lane: a xor (not b and c)."""
    return (a ^ (~b & c)) & M32


@primitive("PouwRotl64Lo", 1, [("lo", U32), ("hi", U32), ("c", U32)], U32, conformance="protocol-defined")
def PouwRotl64Lo(lo: int, hi: int, c: int) -> int:
    """The low half of the 64-bit word ``hi:lo`` rotated left by ``c mod 64``."""
    x = ((hi & M32) << 32) | (lo & M32)
    return _rotr64(x, (64 - (c & 63)) & 63) & M32


@primitive("PouwRotl64Hi", 1, [("lo", U32), ("hi", U32), ("c", U32)], U32, conformance="protocol-defined")
def PouwRotl64Hi(lo: int, hi: int, c: int) -> int:
    """The high half of the 64-bit word ``hi:lo`` rotated left by ``c mod 64``."""
    x = ((hi & M32) << 32) | (lo & M32)
    return _rotr64(x, (64 - (c & 63)) & 63) >> 32


@primitive("PouwAdd64Lo", 1, [("alo", U32), ("blo", U32)], U32, conformance="protocol-defined")
def PouwAdd64Lo(alo: int, blo: int) -> int:
    """The low half of a 64-bit sum."""
    return (alo + blo) & M32


@primitive("PouwAdd64Hi", 1, [("ahi", U32), ("alo", U32), ("bhi", U32), ("blo", U32)], U32, conformance="protocol-defined")
def PouwAdd64Hi(ahi: int, alo: int, bhi: int, blo: int) -> int:
    """The high half of a 64-bit sum: the halves' sum plus the low halves' carry."""
    return (ahi + bhi + (((alo & M32) + (blo & M32)) >> 32)) & M32


def _sha(name: str, f):
    """SHA-512's Σ0, Σ1, σ0 or σ1 on the word ``hi:lo``, one half per primitive."""

    @primitive(f"PouwSha512{name}Hi", 1, [("hi", U32), ("lo", U32)], U32, conformance="protocol-defined")
    def hi_half(hi: int, lo: int) -> int:
        return f(((hi & M32) << 32) | (lo & M32)) >> 32

    @primitive(f"PouwSha512{name}Lo", 1, [("hi", U32), ("lo", U32)], U32, conformance="protocol-defined")
    def lo_half(hi: int, lo: int) -> int:
        return f(((hi & M32) << 32) | (lo & M32)) & M32

    return hi_half, lo_half


PouwSha512BigSigma0Hi, PouwSha512BigSigma0Lo = _sha("BigSigma0", lambda x: _rotr64(x, 28) ^ _rotr64(x, 34) ^ _rotr64(x, 39))
PouwSha512BigSigma1Hi, PouwSha512BigSigma1Lo = _sha("BigSigma1", lambda x: _rotr64(x, 14) ^ _rotr64(x, 18) ^ _rotr64(x, 41))
PouwSha512SmallSigma0Hi, PouwSha512SmallSigma0Lo = _sha("SmallSigma0", lambda x: _rotr64(x, 1) ^ _rotr64(x, 8) ^ (x >> 7))
PouwSha512SmallSigma1Hi, PouwSha512SmallSigma1Lo = _sha("SmallSigma1", lambda x: _rotr64(x, 19) ^ _rotr64(x, 61) ^ (x >> 6))


@primitive("PouwCh32", 1, [("e", U32), ("f", U32), ("g", U32)], U32, conformance="protocol-defined")
def PouwCh32(e: int, f: int, g: int) -> int:
    """SHA-512's Ch on one half: (e and f) xor (not e and g)."""
    return ((e & f) ^ (~e & g)) & M32


@primitive("PouwMaj32", 1, [("a", U32), ("b", U32), ("c", U32)], U32, conformance="protocol-defined")
def PouwMaj32(a: int, b: int, c: int) -> int:
    """SHA-512's Maj on one half."""
    return ((a & b) ^ (a & c) ^ (b & c)) & M32


# ---- bytes and int7 entries --------------------------------------------------------------------------------------------

@primitive("PouwPackBE4", 1, [("b0", I7), ("b1", I7), ("b2", I7), ("b3", I7)], U32, conformance="protocol-defined")
def PouwPackBE4(b0: int, b1: int, b2: int, b3: int) -> int:
    """Four int7 entries as their two's-complement bytes, big-endian: SHA-512's message word half."""
    return sum((sx(b, 7) & 0xFF) << (24 - 8 * i) for i, b in enumerate((b0, b1, b2, b3)))


@primitive("PouwNoiseByte", 1, [("w", U32), ("c", U32)], U8, conformance="protocol-defined")
def PouwNoiseByte(w: int, c: int) -> int:
    """A noise entry from XOF byte ``c mod 4`` of the half-lane ``w`` (little-endian): its low six bits minus 31, exactly
    uniform on [−31, 32], as an s8 byte (`schemes.ncp.noise_row`)."""
    return (((w >> (8 * (c & 3))) & 63) - 31) & 0xFF


@primitive("PouwFormX", 1, [("a", I7), ("e", U8)], U8, conformance="protocol-defined")
def PouwFormX(a: int, e: int) -> int:
    """An entry of X + β from an int7 activation and an s8 noise entry: a + e + 128, as u8."""
    return (sx(a, 7) + sx(e, 8) + BETA) & 0xFF


@primitive("PouwFormX3", 1, [("a", I7), ("e1", U8), ("e2", U8)], U8, conformance="protocol-defined")
def PouwFormX3(a: int, e1: int, e2: int) -> int:
    """An entry of the third block of X + β: 128 − (a + e1 + e2), as u8."""
    return (BETA - (sx(a, 7) + sx(e1, 8) + sx(e2, 8))) & 0xFF


@primitive("PouwAdd8", 1, [("b", I7), ("f", U8)], U8, conformance="protocol-defined")
def PouwAdd8(b: int, f: int) -> int:
    """An entry of Y's first block: b + f, as s8."""
    return (sx(b, 7) + sx(f, 8)) & 0xFF


@primitive("PouwSub8", 1, [("b", I7), ("f", U8)], U8, conformance="protocol-defined")
def PouwSub8(b: int, f: int) -> int:
    """An entry of Y's second block: b − f, as s8 (F₂ = −R·F₁ with every sign +1)."""
    return (sx(b, 7) - sx(f, 8)) & 0xFF


@primitive("PouwAddSub8", 1, [("b", I7), ("f", U8), ("g", U8)], U8, conformance="protocol-defined")
def PouwAddSub8(b: int, f: int, g: int) -> int:
    """An entry of Y's third block: b + f − g, as s8."""
    return (sx(b, 7) + sx(f, 8) - sx(g, 8)) & 0xFF


# ---- the accumulator ---------------------------------------------------------------------------------------------------

A16 = Array(16, U8)


@primitive("PouwNegBetaSum16", 1, [("acc", U32), ("y", A16)], U32, conformance="protocol-defined")
def PouwNegBetaSum16(acc: int, *y: int) -> int:
    """acc − β·Σ y over 16 s8 entries, mod 2^32: the route-U initial value c₀ = −β·Σ_l Y[l][j], 16 entries at a time."""
    return (acc - BETA * sum(sx(v, 8) for v in y)) & M32


@primitive("PouwDot16U8S8", 1, [("acc", U32), ("x", A16), ("y", A16)], U32, conformance="protocol-defined")
def PouwDot16U8S8(acc: int, *xy: int) -> int:
    """One 16-deep route-U step: acc + Σ (X + β)·Y over 16 u8 × s8 products, mod 2^32 (Lean `mmWU`)."""
    x, y = xy[:16], xy[16:]
    return (acc + sum((a & 0xFF) * sx(b, 8) for a, b in zip(x, y))) & M32


# ---- dequantization ----------------------------------------------------------------------------------------------------

def i32_to_f32_rn(z: int) -> int:
    """cvt.rn.f32.s32: an int32 as the nearest FP32, ties to even, as its encoding."""
    v = sx(z, 32)
    if v == 0:
        return 0
    sign = 0x80000000 if v < 0 else 0
    m = -v if v < 0 else v
    e = m.bit_length() - 1                                  # m = 1.f × 2^e, e <= 31
    if e > 23:
        shift = e - 23
        q, r = m >> shift, m & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if r > half or (r == half and q & 1):
            q += 1
        if q >> 24:                                         # rounding carried into a new binade
            q >>= 1
            e += 1
        mant = q & 0x7FFFFF
    else:
        mant = (m << (23 - e)) & 0x7FFFFF
    return sign | ((e + 127) << 23) | mant


@primitive("PouwI32ToF32Rn", 1, [("z", U32)], F32, conformance="protocol-defined")
def PouwI32ToF32Rn(z: int) -> int:
    """cvt.rn.f32.s32: Z as FP32, round to nearest even."""
    return i32_to_f32_rn(z)


# ---- Pearl-C4 (`pc4`) ----------------------------------------------------------------------------------------------------

@primitive("PouwF32SqrtRn", 1, [("a", F32)], F32, conformance="protocol-defined")
def PouwF32SqrtRn(a: int) -> int:
    """sqrt.rn.f32 (`verity_catalog.silicon.fp32.sqrt`, ρ's root in Pearl's statistics): nearest even, √−0 = −0, √+inf = +inf; a NaN
    or a negative nonzero operand gives the GPU's NaN word 0x7FFFFFFF."""
    a &= M32
    if (a >> 23) & 0xFF == 0xFF:
        return a if a == 0x7F800000 else 0x7FFFFFFF
    if a >> 31 and a & 0x7FFFFFFF:
        return 0x7FFFFFFF
    return _fp32.sqrt(a)


_E2M1_HALVES = (0, 1, 2, 3, 4, 6, 8, 12)


@primitive("PouwETilde", 1, [("e", E2M1), ("ae", U8)], BF16, conformance="protocol-defined")
def PouwETilde(e: int, ae: int) -> int:
    """Ẽ = bf16(E · a_E · 2^-4) for one E2M1 noise code and a row's UE4M3 byte, exactly (`schemes.pearl_c4.e_tilde`: at most
    six significant bits, so no rounding), the product's sign or-ed with E's (code 0x8 is −0); 0x7FFF where a_E's low seven
    bits are 0x7F, the E4M3 NaN."""
    ae &= 0xFF
    if ae & 0x7F == 0x7F:
        return 0x7FFF
    v = _E2M1_HALVES[e & 7] / 2 * struct.unpack("<f", struct.pack("<I", _cast.e4m3_to_f32_word(ae)))[0] / 16
    word = struct.unpack("<I", struct.pack("<f", v))[0] >> 16 if v else 0
    return word | (0x8000 if e & 8 else 0)


@primitive("PouwBf16Neg", 1, [("a", BF16)], BF16, conformance="protocol-defined")
def PouwBf16Neg(a: int) -> int:
    """−a on a BF16 encoding: the sign bit flipped (−Ẽ_B in P_B)."""
    return (a ^ 0x8000) & 0xFFFF


# ---- Pearl-C (`pc8`): `pearl_c.live_row`'s exact statistic -----------------------------------------------------------------

#: 2^300 · x² of every FP32 word is an integer below 2^558, and a sum of k/8 <= 2^13 of them is below 2^571
SQ_SCALE, SQ = 300, Value(576)
SQ_MASK = (1 << 576) - 1
#: a nonnegative integer v rounded up to 48 significant bits, as (e << 48) | m with v <= m · 2^e: e = 0 and m = v below 2^48,
#: else 2^47 <= m < 2^48, so the encoding's order is the value order
SQ48, _M48 = Value(64), (1 << 48) - 1


def sq300(x: int) -> int:
    """2^300 · x² for the FP32 word ``x``, exactly; the sign is dropped and exponent 255 decodes as a normal exponent (a
    non-finite word fails the row's admission before this matters)."""
    e, m = (x >> 23) & 0xFF, x & 0x7FFFFF
    if e:
        m |= 1 << 23
    return (m * m) << (2 * (max(e, 1) - 150) + SQ_SCALE)          # |x| = m · 2^(max(e, 1) − 150)


def up48(v: int) -> int:
    """``v`` >= 0 rounded up to 48 significant bits, encoded as `SQ48` says."""
    if v <= _M48:
        return v
    e = v.bit_length() - 48
    m = -(-v >> e)
    if m > _M48:
        m, e = m >> 1, e + 1
    return (e << 48) | m


@primitive("PouwSqAcc", 1, [("acc", SQ), ("x", F32)], SQ, conformance="protocol-defined")
def PouwSqAcc(acc: int, x: int) -> int:
    """acc + 2^300 · x², exactly (`sq300`), mod 2^576: a term of `live_row`'s Σ_{p ≡ 0 (8)} x_p²."""
    return (acc + sq300(x & M32)) & SQ_MASK


@primitive("PouwSqCeil", 1, [("t", SQ), ("n", U32)], SQ48, conformance="protocol-defined")
def PouwSqCeil(t: int, n: int) -> int:
    """θ = ⌈64 · t / n⌉ rounded up to 48 significant bits (`up48`): an x is dead in `live_row` (n · 2^300 · x² >= 64 · t,
    t the scaled sum over n positions) iff 2^300 · x², which has at most 48 significant bits, is at least θ.  n = 0 gives
    the largest encoding."""
    n &= M32
    return up48(-(-64 * t // n)) if n else (1 << 64) - 1


@primitive("PouwSqGe", 1, [("x", F32), ("th", SQ48)], Value(1), conformance="protocol-defined")
def PouwSqGe(x: int, th: int) -> int:
    """1 iff 2^300 · x² >= the value θ encodes (`PouwSqCeil`), compared as `SQ48` encodings."""
    return 1 if up48(sq300(x & M32)) >= th else 0


# ---- Pearl-C's cap (`cap`): a promotion window's masked words ------------------------------------------------------------

#: a promotion window of G = 4 atoms of k = 32, and its masked words (`pearl_c_debit._masked`): mask m's word from +0, atom j
#: passing its word through when bit j of m is set, at bits 32m to 32m + 31 for m = 0 to 14 (mask 15 passes every atom: +0)
WINDOW, MASKS, MASKED = 128, 15, Value(480)


def masked_words(atom, ab) -> list[int]:
    """The masked words of the window ``ab`` (its A codes, then its B codes) under the atom primitive ``atom``, by mask."""
    a, b = ab[:WINDOW], ab[WINDOW:]
    words = {0: 0}
    for j in range(4):
        s = slice(32 * j, 32 * j + 32)
        nxt = {}
        for mask, w in words.items():
            nxt[mask], nxt[mask | 1 << j] = atom.evaluate(w, *a[s], *b[s]), w
        words = nxt
    return [words[m] for m in range(MASKS + 1)]


def _packed(words: list[int]) -> int:
    return sum((w & M32) << 32 * m for m, w in enumerate(words[:MASKS]))


@primitive("PouwBlackwellE4m3Window", 1, [("a", Array(WINDOW, U8)), ("b", Array(WINDOW, U8))], MASKED,
           conformance="protocol-defined")
def PouwBlackwellE4m3Window(*ab: int) -> int:
    """A promotion window's masked words under sm_120's atom `BlackwellE4m3QmmaDot32_v1`, packed as `MASKED` says."""
    return _packed(masked_words(_prims.BlackwellE4m3QmmaDot32, ab))


@primitive("PouwHopperE4m3Window", 1, [("a", Array(WINDOW, U8)), ("b", Array(WINDOW, U8))], MASKED,
           conformance="protocol-defined")
def PouwHopperE4m3Window(*ab: int) -> int:
    """A promotion window's masked words under the H100's atom `HopperE4m3QgmmaDot32_v1`, packed as `MASKED` says."""
    return _packed(masked_words(_prims.HopperE4m3QgmmaDot32, ab))


@primitive("PouwWindowWord", 1, [("x", MASKED), ("m", Value(4))], F32, conformance="protocol-defined")
def PouwWindowWord(x: int, m: int) -> int:
    """Mask m's word of a window's masked words (+0 for m = 15)."""
    return (x >> 32 * m) & M32 if m < MASKS else 0


# ---- Pearl-C's cap (`cap`): an element's salt-dead bit ---------------------------------------------------------------------

#: the salt-dead support's two ends (P3's `SaltDead.support`, #1013's `plusEnd` and `minusEnd`): E′ is c_β at every
#: position, and the F codes are e4m3(+11/4) at every position, or e4m3(−11/4) at every position
F_ENDS = (0x43, 0xC3)


def ends_agree(atom, acc: int, cb: int) -> int:
    """1 iff the codes of the support's two ends under the atom primitive ``atom``, e4m3(atom(acc, [cb]^32, [±0x43]^32)),
    have one value: they are equal, or both are zeros (#1013's `EndsAgree`, compared as its `codeVal_eq_iff` says)."""
    c = [_prims.F32ToE4m3Sat.evaluate(atom.evaluate(acc, *[cb] * 32, *[f] * 32)) for f in F_ENDS]
    return 1 if c[0] == c[1] or not (c[0] | c[1]) & 0x7F else 0


@primitive("PouwBlackwellE4m3SaltDead", 1, [("acc", F32), ("cb", U8)], Value(1), conformance="protocol-defined")
def PouwBlackwellE4m3SaltDead(acc: int, cb: int) -> int:
    """An element's salt-dead bit under sm_120's atom `BlackwellE4m3QmmaDot32_v1` (`ends_agree`): ``acc`` is the forming's
    accumulator f32(α·x) and ``cb`` the row's flat noise code e4m3(f32(β·44)); by #1013's `saltDeadP_iff_ends`, which has no
    hypothesis on α, x or β, it is 1 iff the element is salt-dead (`saltDeadP`)."""
    return ends_agree(_prims.BlackwellE4m3QmmaDot32, acc, cb)


@primitive("PouwHopperE4m3SaltDead", 1, [("acc", F32), ("cb", U8)], Value(1), conformance="protocol-defined")
def PouwHopperE4m3SaltDead(acc: int, cb: int) -> int:
    """An element's salt-dead bit under the H100's atom `HopperE4m3QgmmaDot32_v1`, as `PouwBlackwellE4m3SaltDead` says."""
    return ends_agree(_prims.HopperE4m3QgmmaDot32, acc, cb)


# ---- Pearl-C's -h3 tile leaf (`pc8.Pc8TileLeafH3`): one word of a tile, put in its segment ------------------------------------

#: a tile segment's words (`pearl_c.messages` of two tile rows of 64 columns: le32 C̃ then le32 U of each 32-column chunk)
SEG_WORDS = 256
#: the most segments a -h3 tile leaf has (64 tile rows), each a 64-byte digest
LEAF_SEGMENTS = 32
D512 = Value(512)


def leaf_at(r: int, col: int, rows: int) -> int:
    """`PouwPc8LeafH3`'s ``at``: the word's tile row ``r`` and column ``col`` (each below 64) and the tile's height ``rows``
    (16, 32, 48 or 64), as ``r | col << 6 | (rows / 16 − 1) << 12``."""
    if not (0 <= r < rows and 0 <= col < 64 and rows in (16, 32, 48, 64)):
        raise ValueError(f"a word at tile row {r}, column {col} of a tile of {rows} rows (16, 32, 48 or 64) and 64 columns")
    return r | col << 6 | (rows // 16 - 1) << 12


def leaf_slot(at: int) -> tuple[int, int, int, int]:
    """``at``'s (segment, C̃'s word in the segment, the tile's segments, the tile's bits): U's word is C̃'s + 32."""
    r, col, rows = at & 63, (at >> 6) & 63, 16 * (((at >> 12) & 3) + 1)
    return r >> 1, (r & 1) * 128 + (col >> 5) * 64 + (col & 31), rows // 2, rows * 512 * 8


@primitive("PouwPc8LeafH3", 1, [("c", F32), ("u", F32), ("seg", Array(SEG_WORDS, U32)), ("dig", Array(LEAF_SEGMENTS, D512)),
                                ("at", U32)], D512, conformance="protocol-defined")
def PouwPc8LeafH3(c: int, u: int, *rest: int) -> int:
    """The -h3 tile leaf (`pearl_c.leaf` under h3, ``sha512/row-seg/v1`` of the tile's messages) with the word C̃ = ``c``,
    U = ``u`` at ``at`` (`leaf_at`): its segment s is ``seg`` (the segment's words, little-endian, as the messages lay them
    out) with C̃ and U put at their words (`leaf_slot`), d_s its digest, and the leaf ``SHA-512(Q(n) ‖ d_0 ‖ … ‖ d_(S−1))``
    over ``dig`` (each digest's 64 bytes little-endian) with d_s in place of ``dig[s]``; a row at or past the tile's height
    puts no word in (only ``dig`` is hashed).  The leaf's 64 bytes, little-endian."""
    seg, dig, at = list(rest[:SEG_WORDS]), rest[SEG_WORDS:SEG_WORDS + LEAF_SEGMENTS], rest[-1]
    s, w, n_seg, n_bits = leaf_slot(at)
    seg[w], seg[w + 32] = c, u
    ds = [(d & ((1 << 512) - 1)).to_bytes(64, "little") for d in dig]
    if s < n_seg:
        body = b"".join((x & M32).to_bytes(4, "little") for x in seg)
        ds[s] = hashlib.sha512(sha512_row_seg_segment_prefix(s, n_seg, 8 * len(body)) + body).digest()
    leaf = hashlib.sha512(sha512_row_seg_prefix(n_bits) + b"".join(ds[:n_seg])).digest()
    return int.from_bytes(leaf, "little")


__all__ = ["A16", "BETA", "BF16", "D512", "E2M1", "F32", "F_ENDS", "I7", "LEAF_SEGMENTS", "M32", "MASKED", "MASKS", "SEG_WORDS",
           "U32", "U8", "WINDOW", "sx", "ends_agree", "i32_to_f32_rn", "leaf_at", "leaf_slot", "masked_words",
           *sorted(n for n in list(globals()) if n.startswith("Pouw"))]
