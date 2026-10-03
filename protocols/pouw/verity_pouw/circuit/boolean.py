"""PoUW's own word primitives as Boolean Definitions: the next version of each `words` primitive Pearl-C4's (`pc4`) and
Pearl-C's (`pc8`) row units call, and of the four Keccak words their hiding tiles' digest calls (`hashes`, Keccak-p[1600]
on (lo, hi) half-lanes), the same function on bits.

    PouwF32SqrtRn_v2   (a: Array<32, Value<1>>) -> Array<32, Value<1>>                          word view: PouwF32SqrtRn_v1
    PouwETilde_v2      (e: Array<4, Value<1>>, ae: Array<8, Value<1>>) -> Array<16, Value<1>>    word view: PouwETilde_v1
    PouwBf16Neg_v2     (a: Array<16, Value<1>>) -> Array<16, Value<1>>                          word view: PouwBf16Neg_v1
    PouwSqAcc_v2       (acc: Array<576, Value<1>>, x: Array<32, Value<1>>) -> Array<576, Value<1>>  word view: PouwSqAcc_v1
    PouwSqCeil_v2      (t: Array<576, Value<1>>, n: Array<32, Value<1>>) -> Array<64, Value<1>>     word view: PouwSqCeil_v1
    PouwSqGe_v2        (x: Array<32, Value<1>>, th: Array<64, Value<1>>) -> Value<1>                word view: PouwSqGe_v1
    PouwXor32_v2       (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>                       word view: PouwXor32_v1
    PouwChi32_v2       (a, b, c: Array<32, Value<1>>) -> Array<32, Value<1>>                    word view: PouwChi32_v1
    PouwRotl64Lo_v2    (lo, hi, c: Array<32, Value<1>>) -> Array<32, Value<1>>                  word view: PouwRotl64Lo_v1
    PouwRotl64Hi_v2    (lo, hi, c: Array<32, Value<1>>) -> Array<32, Value<1>>                  word view: PouwRotl64Hi_v1

Each is traced from a builder over core's `forms` (`verity.ml.boolean.trace.definition`), so its And gates are the builder's
live ANDs; a backend whose lowering takes `forms` builders (C-Flock's pieces) can run the builder itself (``.build``).

* `f32_sqrt_rn`: the significand normalized, the exponent made even by doubling it, and a digit-by-digit root (`isqrt`) of
  the 50-bit radicand to 25 bits; the last bit rounds, since no square root of a float lies halfway between two (the
  radicand is even, so its root is no odd integer).  Every root of a positive finite word is a normal word.
* `e_tilde`: the significands' 2 x 4-bit product, normalized, and the exponent fields' sum; the product's sign or-ed with E's.
* `sq_acc`, `sq_ge`: m² (a squarer, every cross product once) shifted by 2 · max(e, 1) (`words.sq300`).  `sq_ge` never
  builds the 558-bit square: m² has 48 bits, so its `up48` encoding is exact, m² shifted left by min(2 · max(e, 1), its
  leading zeros) and the rest of the shift as its exponent.
* `sq_ceil`: ⌈64 t / n⌉ by non-restoring division (one 34-bit add or subtract per quotient bit, then the remainder's
  sign correction), plus 1 when the remainder is nonzero, rounded up to 48 significant bits with a right shift and its
  sticky bit; n = 0 gives every bit set.
* `chi32`: Keccak's χ on a half-lane, one AND per bit; xor is free.
* `rotl64`: a barrel rotator over the 64-bit word hi:lo, one stage per bit of the amount's low six, the largest first, so
  each stage builds only the window the returned half reads.  A constant amount, as every call in `hashes` has, folds each
  stage to wiring with no AND (`forms`' AND of a constant is no gate).
"""
from __future__ import annotations

from verity.ir import boolean as BL
from verity.ir.types import Array
from verity.ml.boolean import fp
from verity.ml.boolean.forms import (NOT, ONE, OR, ZERO, add, and_reduce, compress_columns, const_bits, lzc, multiply, mux,
                                     or_reduce, shl_window, sub)
from verity.ml.boolean.trace import definition

from . import words as W

__all__ = ["PouwBf16Neg", "PouwChi32", "PouwETilde", "PouwF32SqrtRn", "PouwRotl64Hi", "PouwRotl64Lo", "PouwSqAcc", "PouwSqCeil",
           "PouwSqGe", "PouwXor32", "chi32", "e_tilde", "f32_sqrt_rn", "isqrt", "rotl64", "sq_acc", "sq_ceil", "sq_ge", "square",
           "up48"]

BIT = BL.BIT
F32 = U32 = Array(32, BIT)
BF16, E2M1, U8 = Array(16, BIT), Array(4, BIT), Array(8, BIT)
SQ, SQ48 = Array(576, BIT), Array(64, BIT)
_NAN_PTX = 0x7FFFFFFF


def isqrt(C, n):
    """⌊√n⌋ of an unsigned word of even width (LSB first), half as wide: one trial subtraction per root bit."""
    rem, root = [], []
    for i in range(len(n) // 2 - 1, -1, -1):
        rem = [n[2 * i], n[2 * i + 1]] + rem
        trial = [ONE, ZERO] + root
        w = max(len(rem), len(trial))
        d, borrow = sub(C, rem, trial, w)
        bit = NOT(borrow)
        rem = mux(C, bit, d, rem + [ZERO] * (w - len(rem)))[:len(root) + 2]
        root = [bit] + root
    return root


def f32_sqrt_rn(C, a):
    """``PouwF32SqrtRn_v1``: sqrt.rn.f32, √±0 = ±0, √+inf = +inf, a NaN or a negative nonzero operand 0x7FFFFFFF."""
    A = fp.unpack(C, a)
    _az, lz = lzc(C, [ZERO] * 8 + A.sig)
    sig = fp.shl(C, A.sig, lz)                                          # 1.f, the top bit set
    x = sub(C, add(C, A.e_eff, const_bits(127, 9), width=9), lz, 9)[0]  # e + 254 for the unbiased exponent e
    m = mux(C, x[0], [ZERO] + sig, sig + [ZERO])                        # e odd: 2 · 1.f and e - 1
    r = isqrt(C, [ZERO] * 25 + m)                                       # 25 bits, the top one set
    word = add(C, r[1:24] + x[1:9], [r[0]], width=31) + [ZERO]          # exponent field ⌊e / 2⌋ + 127
    keep = OR(C, A.zero, C.AND(A.inf, NOT(A.s)))
    nan = C.AND(NOT(keep), OR(C, A.e_max, A.s))
    return mux(C, nan, const_bits(_NAN_PTX, 32), mux(C, keep, list(a), word))


def e_tilde(C, e, ae):
    """``PouwETilde_v1``: bf16(E · a_E · 2^-4) exactly, its sign or-ed with E's; 0x7FFF for an E4M3 NaN a_E."""
    ee_nz = OR(C, e[1], e[2])
    ae_nz = or_reduce(C, ae[3:7])
    p = multiply(C, [e[0], ee_nz], list(ae[:3]) + [ae_nz])            # the significands' product, 6 bits
    az, lz = lzc(C, [ZERO, ZERO] + p)
    p = fp.shl(C, p, lz)
    xe = [e[1] ^ NOT(ee_nz), e[2]]                                      # max(E's exponent field, 1)
    xa = [ae[3] ^ NOT(ae_nz)] + list(ae[4:7])                           # max(a_E's exponent field, 1)
    ex = sub(C, add(C, add(C, xa, xe, width=5), const_bits(116, 8), width=8), lz, 8)[0]
    mag = [C.AND(NOT(az), b) for b in [ZERO, ZERO] + p[:5] + ex]
    out = mag + [OR(C, e[3], C.AND(ae[7], NOT(az)))]
    return mux(C, and_reduce(C, ae[:7]), const_bits(0x7FFF, 16), out)


def square(C, m):
    """m² of an unsigned word, twice as wide: each bit once on the diagonal and each cross product once, a column higher."""
    cols = [[] for _ in range(2 * len(m))]
    for i, x in enumerate(m):
        cols[2 * i].append(x)
        for j in range(i + 1, len(m)):
            cols[i + j + 1].append(C.AND(x, m[j]))
    return compress_columns(C, cols)[:2 * len(m)]


def _sq(C, x):
    """(m², 2 · max(e, 1)) of an FP32 word: 2^300 · x² = m² << the amount (`words.sq300`), the amount 9 bits."""
    X = fp.unpack(C, x)
    return square(C, X.sig), [ZERO] + X.e_eff


def sq_acc(C, acc, x):
    """``PouwSqAcc_v1``: acc + 2^300 · x² mod 2^576."""
    p, amt = _sq(C, x)
    return add(C, acc, shl_window(C, p, amt, 0, 576), width=576)


def up48(C, v):
    """`words.up48` of an unsigned word of at most 1024 bits: (e << 48) | m, 64 bits."""
    n = len(v)
    _az, lz = lzc(C, [ZERO] * (1024 - n) + list(v))
    d, small = sub(C, const_bits(n - 48, 10), lz, 10)                   # its bit length - 48, or below 48 bits
    e = [C.AND(NOT(small), b) for b in d]
    y, sticky = fp.shr_sticky(C, v, e)
    m = add(C, y[:48], [sticky], width=48, carry_out=True)
    return m[:47] + [m[47] ^ m[48]] + add(C, e, [m[48]], width=16)     # 2^48 is 2^47 · 2


def _add_about_b(C, a, b, c):
    """a + b + c over equal widths, its carries written about b (`forms.maj`'s z = b, not the carry): an operand XORed into
    every bit of b and c, such as a quotient bit, cancels in each sum bit instead of piling up in the carries."""
    out = []
    for x, y in zip(a, b):
        out.append(x ^ y ^ c)
        c = y ^ C.AND(x ^ y, c ^ y)
    return out


def sq_ceil(C, t, n):
    """``PouwSqCeil_v1``: up48(⌈64 t / n⌉), every bit set for n = 0."""
    num = [ZERO] * 6 + list(t)
    d = list(n) + [ZERO, ZERO]
    r, q = [ZERO] * 34, []
    for bit in reversed(num):
        neg = r[33]
        r = [bit] + r[:33]
        r = _add_about_b(C, r, [b ^ NOT(neg) for b in d], NOT(neg))
        q.append(NOT(r[33]))
    rem = add(C, r, [C.AND(r[33], b) for b in d], width=34)
    c = add(C, q[::-1], [or_reduce(C, rem[:32])], width=len(num) + 1)
    n_zero = NOT(or_reduce(C, n))
    return [OR(C, n_zero, b) for b in up48(C, c)]


def sq_ge(C, x, th):
    """``PouwSqGe_v1``: 1 iff up48(2^300 · x²) >= θ as 64-bit encodings."""
    p, amt = _sq(C, x)
    az, lz = lzc(C, [ZERO] * 16 + p)
    t = fp.min_u(C, amt, lz, 9)
    m = fp.shl(C, p, t[:6])
    ex = [C.AND(NOT(az), b) for b in sub(C, amt, t, 9)[0]]
    return [NOT(sub(C, m + ex + [ZERO] * 7, th, 64)[1])]


def chi32(C, a, b, c):
    """``PouwChi32_v1``: a xor (not b and c)."""
    return [x ^ C.AND(NOT(y), z) for x, y, z in zip(a, b, c)]


def rotl64(C, lo, hi, c):
    """hi:lo rotated left by c mod 64, all 64 bits LSB first (``PouwRotl64Lo_v1`` is the low half, ``PouwRotl64Hi_v1`` the
    high): stage k rotates by 2^k when bit k of c is set, k from 5 down to 0."""
    x = list(lo) + list(hi)
    for k in range(5, -1, -1):
        x = mux(C, c[k], [x[(i - (1 << k)) % 64] for i in range(64)], x)
    return x


PouwF32SqrtRn = definition("PouwF32SqrtRn", 2, f32_sqrt_rn, (("a", F32),), F32, word=W.PouwF32SqrtRn,
                           conformance=W.PouwF32SqrtRn.conformance,
                           doc="sqrt.rn.f32 on bits: PouwF32SqrtRn_v1, NaN and negative operands 0x7FFFFFFF")
PouwETilde = definition("PouwETilde", 2, e_tilde, (("e", E2M1), ("ae", U8)), BF16, word=W.PouwETilde,
                        conformance=W.PouwETilde.conformance, doc="E~ = bf16(E · a_E · 2^-4) on bits: PouwETilde_v1")
PouwBf16Neg = definition("PouwBf16Neg", 2, lambda C, a: list(a[:15]) + [NOT(a[15])], (("a", BF16),), BF16, word=W.PouwBf16Neg,
                         conformance=W.PouwBf16Neg.conformance, doc="-a on bf16 bits: PouwBf16Neg_v1, the sign bit flipped")
PouwSqAcc = definition("PouwSqAcc", 2, sq_acc, (("acc", SQ), ("x", F32)), SQ, word=W.PouwSqAcc,
                       conformance=W.PouwSqAcc.conformance, doc="acc + 2^300 · x² mod 2^576 on bits: PouwSqAcc_v1")
PouwSqCeil = definition("PouwSqCeil", 2, sq_ceil, (("t", SQ), ("n", U32)), SQ48, word=W.PouwSqCeil,
                        conformance=W.PouwSqCeil.conformance, doc="up48(⌈64 t / n⌉) on bits: PouwSqCeil_v1")
PouwSqGe = definition("PouwSqGe", 2, sq_ge, (("x", F32), ("th", SQ48)), BIT, word=W.PouwSqGe,
                      conformance=W.PouwSqGe.conformance, doc="up48(2^300 · x²) >= θ on bits: PouwSqGe_v1")
PouwXor32 = definition("PouwXor32", 2, lambda C, a, b: [x ^ y for x, y in zip(a, b)], (("a", U32), ("b", U32)), U32,
                       word=W.PouwXor32, conformance=W.PouwXor32.conformance, doc="a xor b on bits: PouwXor32_v1")
PouwChi32 = definition("PouwChi32", 2, chi32, (("a", U32), ("b", U32), ("c", U32)), U32, word=W.PouwChi32,
                       conformance=W.PouwChi32.conformance, doc="Keccak's χ on one half-lane, on bits: PouwChi32_v1")
PouwRotl64Lo = definition("PouwRotl64Lo", 2, lambda C, lo, hi, c: rotl64(C, lo, hi, c)[:32],
                          (("lo", U32), ("hi", U32), ("c", U32)), U32, word=W.PouwRotl64Lo,
                          conformance=W.PouwRotl64Lo.conformance,
                          doc="the low half of hi:lo rotated left by c mod 64, on bits: PouwRotl64Lo_v1")
PouwRotl64Hi = definition("PouwRotl64Hi", 2, lambda C, lo, hi, c: rotl64(C, lo, hi, c)[32:],
                          (("lo", U32), ("hi", U32), ("c", U32)), U32, word=W.PouwRotl64Hi,
                          conformance=W.PouwRotl64Hi.conformance,
                          doc="the high half of hi:lo rotated left by c mod 64, on bits: PouwRotl64Hi_v1")
