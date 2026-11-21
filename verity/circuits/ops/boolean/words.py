"""PoUW's word primitives (`verity.circuits.ops.words`) as Boolean Definitions: the next version of each, the same function
on bits. They sit beside the core's so that `versions.boolean_version` finds every one in any process; Pearl-C4's (`pc4`)
and Pearl-C's (`pc8`) row units call them, and the hiding tiles' digest the four Keccak words (`hashes`, Keccak-p[1600] on
(lo, hi) half-lanes).

    PouwF32SqrtRn_v2   (a: Array<32, Value<1>>) -> Array<32, Value<1>>                          word view: PouwF32SqrtRn_v1
    PouwETilde_v2      (e: Array<4, Value<1>>, ae: Array<8, Value<1>>) -> Array<16, Value<1>>    word view: PouwETilde_v1
    PouwBf16Neg_v2     (a: Array<16, Value<1>>) -> Array<16, Value<1>>                          word view: PouwBf16Neg_v1
    PouwSqAcc_v2       (acc: Array<576, Value<1>>, x: Array<32, Value<1>>) -> Array<576, Value<1>>  word view: PouwSqAcc_v1
    PouwSqCeil_v2      (t: Array<576, Value<1>>, n: Array<32, Value<1>>) -> Array<64, Value<1>>     word view: PouwSqCeil_v1
    PouwSqGe_v2        (x: Array<32, Value<1>>, th: Array<64, Value<1>>) -> Value<1>                word view: PouwSqGe_v1
    PouwBlackwellE4m3Window_v2  (a, b: Array<128, Array<8, Value<1>>>) -> Array<480, Value<1>>     word view: PouwBlackwellE4m3Window_v1
    PouwHopperE4m3Window_v2     the same signature                                                word view: PouwHopperE4m3Window_v1
    PouwWindowWord_v2  (x: Array<480, Value<1>>, m: Array<4, Value<1>>) -> Array<32, Value<1>>      word view: PouwWindowWord_v1
    PouwBlackwellE4m3SaltDead_v2  (acc: Array<32, Value<1>>, cb: Array<8, Value<1>>) -> Value<1>   word view: PouwBlackwellE4m3SaltDead_v1
    PouwHopperE4m3SaltDead_v2     the same signature                                               word view: PouwHopperE4m3SaltDead_v1
    PouwXor32_v2       (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>                       word view: PouwXor32_v1
    PouwChi32_v2       (a, b, c: Array<32, Value<1>>) -> Array<32, Value<1>>                    word view: PouwChi32_v1
    PouwRotl64Lo_v2    (lo, hi, c: Array<32, Value<1>>) -> Array<32, Value<1>>                  word view: PouwRotl64Lo_v1
    PouwRotl64Hi_v2    (lo, hi, c: Array<32, Value<1>>) -> Array<32, Value<1>>                  word view: PouwRotl64Hi_v1
    PouwPackBE4_v2     (b0, b1, b2, b3: Array<7, Value<1>>) -> Array<32, Value<1>>              word view: PouwPackBE4_v1
    PouwNoiseByte_v2   (w, c: Array<32, Value<1>>) -> Array<8, Value<1>>                        word view: PouwNoiseByte_v1
    PouwDot16U8S8_v2   (acc: Array<32, Value<1>>, x, y: Array<16, Array<8, Value<1>>>) -> Array<32, Value<1>>
                                                                                                word view: PouwDot16U8S8_v1
    PouwNegBetaSum16_v2  (acc: Array<32, Value<1>>, y: Array<16, Array<8, Value<1>>>) -> Array<32, Value<1>>
                                                                                                word view: PouwNegBetaSum16_v1
    PouwFormX_v2, PouwAdd8_v2, PouwSub8_v2   (a: Array<7, Value<1>>, e: Array<8, Value<1>>) -> Array<8, Value<1>>
                                                                                                word view: their _v1
    PouwFormX3_v2, PouwAddSub8_v2   (a: Array<7, Value<1>>, e1, e2: Array<8, Value<1>>) -> Array<8, Value<1>>
                                                                                                word view: their _v1
    PouwAdd64Lo_v2     (alo, blo: Array<32, Value<1>>) -> Array<32, Value<1>>                   word view: PouwAdd64Lo_v1
    PouwAdd64Hi_v2     (ahi, alo, bhi, blo: Array<32, Value<1>>) -> Array<32, Value<1>>         word view: PouwAdd64Hi_v1
    PouwCh32_v2, PouwMaj32_v2   (a, b, c: Array<32, Value<1>>) -> Array<32, Value<1>>           word view: their _v1
    PouwSha512{Big,Small}Sigma{0,1}{Hi,Lo}_v2   (hi, lo: Array<32, Value<1>>) -> Array<32, Value<1>>
                                                                                                word view: their _v1
    PouwI32ToF32Rn_v2  (z: Array<32, Value<1>>) -> Array<32, Value<1>>                          word view: PouwI32ToF32Rn_v1

Each is traced from a builder over core's `forms` (`verity.circuits.boolean.trace.definition`), so its And gates are the builder's
live ANDs but for its commits (`forms.Circuit.commit`, a row of a `forms` lowering and no gate); Flock's lowering runs the
builder itself (``.build``), for the word primitive's piece and for an activation of the Definition, so it keeps the commits.

* the windows: the 15 calls of the atom's own builder (`verity.circuits.ops.boolean.gemm`) that build a window's masked words, on a
  view of the circuit that keeps its own AND memo (`_Shared`), so atom j's 32 products, and every part of its step that
  doesn't read the accumulator, are built once for its 2^j words on any circuit, Flock's (which has no common-subexpression
  pass) included: on sm_120 102,409 ANDs where the 15 calls are 145,651.
* `window_word`: a 4-to-16 decode and one AND per bit and mask, so a constant m (each read of the inert-work bound's
  word) costs none.
* the salt-dead bits: the atom's builder at the support's two ends (E′ the one code c_β at all 32 positions, F the
  constant ±0x43) and the cast, on an `_Shared` view, so the 32 equal products are built once and the two ends share
  every AND that doesn't read F's sign: on sm_120 2,493 ANDs where two atom calls and casts are 20,062 (1,806 against
  15,230 on the H100).

* `f32_sqrt_rn`: `verity.circuits.ops.boolean.sqrt`'s (a digit-by-digit root of the normalized significand).
* `e_tilde`: the significands' 2 x 4-bit product, normalized, and the exponent fields' sum; the product's sign or-ed with E's.
* `sq_acc`, `sq_ge`: m² (a squarer, every cross product once) shifted by 2 · max(e, 1) (`words.sq300`).  `sq_ge` never
  builds the 558-bit square: m² has 48 bits, so its `up48` encoding is exact, m² shifted left by min(2 · max(e, 1), its
  leading zeros) and the rest of the shift as its exponent.
* `sq_ceil`: ⌈64 t / n⌉ by non-restoring division (one 34-bit add or subtract per quotient bit, then the remainder's
  sign correction), plus 1 when the remainder is nonzero, rounded up to 48 significant bits with a right shift and its
  sticky bit; n = 0 gives every bit set.
* `chi32`: Keccak's χ on a half-lane, one AND per bit; xor is free.  Its output is committed: a lowering that keeps a
  call's outputs as XOR forms (Flock's) otherwise carries χ's XOR terms into the next round's θ, and over 12 rounds they
  compound until lowering one Keccak-p[1600, 12] runs out of memory; committed, it is 38,400 rows (19,200 ANDs and as
  many commits), χ's operands θ's outputs of 11 committed bits at most.
* `rotl64`: a barrel rotator over the 64-bit word hi:lo, one stage per bit of the amount's low six, the largest first, so
  each stage's live ANDs are only the window the returned half reads.  A constant amount, as every call in `hashes` has,
  folds each stage to wiring with no AND (`forms`' AND of a constant is no gate).
* `sum_mod`: a sum of weighted columns mod 2^width by full adders, a carry out of the top column dropped. The dot's partial
  products (y's sign bit complemented, Baugh–Wooley), −128 · Σ y and the three-operand bytes are each one such sum.  The
  dot is a tensor-core step of the u8 x s8 IMMA without .satfinite (`census.silicon_step`), so the census counts its gates
  as `tc:int8`.
* the 8-bit forms and `noise_byte`: `forms.add`'s ripple; the pack wiring, no AND.
* SHA-512's words (the 64-bit add's halves, Ch, Maj and the sigma halves): `verity.circuits.boolean.sha512`'s gates on
  the 64-bit word hi:lo, the add's ripple uncommitted, so its halves are 31 and 63 ANDs, Ch and Maj 32, the sigmas none.
* `PouwI32ToF32Rn_v2` is `fp.s32_to_f32`, the circuit of `S32ToF32Rn_v2`.
"""
from __future__ import annotations

from verity.circuits import boolean as BL
from verity.circuits.census import silicon_step
from verity.circuits.types import Array
from verity.circuits.silicon import fp
from verity.circuits.silicon.integer import BLACKWELL_SM120_U8S8_M16N8K32
from verity.circuits.ops.boolean import gemm as BG
from verity.circuits.ops.boolean import scalar as BS
from verity.circuits.ops.boolean.sqrt import f32_sqrt_rn, isqrt
from verity.circuits.boolean.forms import (NOT, ONE, OR, ZERO, add, and_reduce, compress_columns, const_bits, lzc, maj,
                                           multiply, mux, or_reduce, shl_window, sub)
from verity.circuits.boolean import sha512 as SH
from verity.circuits.boolean.trace import definition

from verity.circuits.ops import words as W

__all__ = ["PouwAdd64Hi", "PouwAdd64Lo", "PouwAdd8", "PouwAddSub8", "PouwBf16Neg", "PouwBlackwellE4m3SaltDead",
           "PouwBlackwellE4m3Window", "PouwCh32", "PouwChi32", "PouwDot16U8S8", "PouwETilde", "PouwF32SqrtRn", "PouwFormX",
           "PouwFormX3", "PouwHopperE4m3SaltDead", "PouwHopperE4m3Window", "PouwI32ToF32Rn", "PouwMaj32",
           "PouwNegBetaSum16", "PouwNoiseByte", "PouwPackBE4", "PouwRotl64Hi", "PouwRotl64Lo", "PouwSha512BigSigma0Hi",
           "PouwSha512BigSigma0Lo", "PouwSha512BigSigma1Hi", "PouwSha512BigSigma1Lo", "PouwSha512SmallSigma0Hi",
           "PouwSha512SmallSigma0Lo", "PouwSha512SmallSigma1Hi", "PouwSha512SmallSigma1Lo", "PouwSqAcc", "PouwSqCeil",
           "PouwSqGe", "PouwSub8", "PouwWindowWord", "PouwXor32", "add_sub8", "chi32", "dot16_u8s8",
           "e_tilde", "f32_sqrt_rn", "form_x", "form_x3", "isqrt", "neg_beta_sum16", "noise_byte", "pack_be4", "rotl64",
           "salt_dead", "sq_acc", "sq_ceil", "sq_ge", "square", "sum_mod", "up48", "window", "window_word"]

BIT = BL.BIT
F32 = U32 = Array(32, BIT)
BF16, E2M1, U8 = Array(16, BIT), Array(4, BIT), Array(8, BIT)
SQ, SQ48 = Array(576, BIT), Array(64, BIT)
U4, CODES, MASKED = Array(4, BIT), Array(W.WINDOW, U8), Array(W.MASKED.w, BIT)
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


class _Shared:
    """The circuit ``C`` with its own AND memo (`forms.Circuit`'s ``cse``): an AND of two forms already ANDed through the view
    reuses that bit. Everything else is ``C``'s."""

    def __init__(self, C):
        object.__setattr__(self, "_C", C)
        object.__setattr__(self, "_memo", {})

    def AND(self, a, b):
        if a.is_const() or b.is_const() or a.s == b.s:
            return self._C.AND(a, b)
        key = frozenset((a.key(), b.key()))
        w = self._memo.get(key)
        if w is None:
            w = self._memo[key] = self._C.AND(a, b)
        return w

    def __getattr__(self, name):
        return getattr(self._C, name)

    def __setattr__(self, name, value):
        setattr(self._C, name, value)


def window(atom):
    """The builder of a window's masked words (`words.masked_words`) under the Boolean atom ``atom``, packed as `words.MASKED`."""

    def build(C, *ab):
        S, a, b = _Shared(C), ab[:W.WINDOW], ab[W.WINDOW:]
        words = {0: [ZERO] * 32}
        for j in range(4):
            aj, bj = a[32 * j:32 * j + 32], b[32 * j:32 * j + 32]
            nxt = {}
            for mask, w in words.items():
                nxt[mask], nxt[mask | 1 << j] = atom.build(S, w, *aj, *bj), w
            words = nxt
        return [x for m in range(W.MASKS) for x in words[m]]
    return build


def salt_dead(atom):
    """The builder of an element's salt-dead bit (`words.ends_agree`) under the Boolean atom ``atom``: the two ends' casts
    on a view with its own AND memo, then their comparison by value (equal, or both zeros)."""

    def build(C, acc, cb):
        S = _Shared(C)
        ends = [BS.f32_to_e4m3_sat(S, atom.build(S, list(acc), *[list(cb)] * 32, *[const_bits(f, 8)] * 32))
                for f in W.F_ENDS]
        d = [x ^ y for x, y in zip(*ends)]
        return [S.AND(NOT(or_reduce(S, d[:7])), OR(S, NOT(d[7]), NOT(or_reduce(S, ends[0][:7]))))]
    return build


def window_word(C, x, m):
    """``PouwWindowWord_v1``: mask m's 32 bits of x, 0 for m = 15."""
    sel = fp.decode(C, list(m))
    out = []
    for t in range(32):
        bit = ZERO
        for i in range(W.MASKS):
            bit = bit ^ C.AND(sel[i], x[32 * i + t])
        out.append(bit)
    return out


def chi32(C, a, b, c):
    """``PouwChi32_v1``: a xor (not b and c), committed."""
    return C.commit([x ^ C.AND(NOT(y), z) for x, y, z in zip(a, b, c)])


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
PouwBlackwellE4m3Window = definition(
    "PouwBlackwellE4m3Window", 2, window(BG.BlackwellE4m3QmmaDot32), (("a", CODES), ("b", CODES)), MASKED,
    word=W.PouwBlackwellE4m3Window, conformance=W.PouwBlackwellE4m3Window.conformance,
    doc="A promotion window's masked words under sm_120's atom on bits, each atom's products built once: "
        "PouwBlackwellE4m3Window_v1")
PouwHopperE4m3Window = definition(
    "PouwHopperE4m3Window", 2, window(BG.HopperE4m3QgmmaDot32), (("a", CODES), ("b", CODES)), MASKED,
    word=W.PouwHopperE4m3Window, conformance=W.PouwHopperE4m3Window.conformance,
    doc="A promotion window's masked words under the H100's atom on bits, each atom's products built once: "
        "PouwHopperE4m3Window_v1")
PouwWindowWord = definition("PouwWindowWord", 2, window_word, (("x", MASKED), ("m", U4)), F32, word=W.PouwWindowWord,
                            conformance=W.PouwWindowWord.conformance, doc="Mask m's word of a window on bits: PouwWindowWord_v1")
PouwBlackwellE4m3SaltDead = definition(
    "PouwBlackwellE4m3SaltDead", 2, salt_dead(BG.BlackwellE4m3QmmaDot32), (("acc", F32), ("cb", U8)), BIT,
    word=W.PouwBlackwellE4m3SaltDead, conformance=W.PouwBlackwellE4m3SaltDead.conformance,
    doc="An element's salt-dead bit under sm_120's atom on bits, the two ends' common parts built once: "
        "PouwBlackwellE4m3SaltDead_v1")
PouwHopperE4m3SaltDead = definition(
    "PouwHopperE4m3SaltDead", 2, salt_dead(BG.HopperE4m3QgmmaDot32), (("acc", F32), ("cb", U8)), BIT,
    word=W.PouwHopperE4m3SaltDead, conformance=W.PouwHopperE4m3SaltDead.conformance,
    doc="An element's salt-dead bit under the H100's atom on bits, the two ends' common parts built once: "
        "PouwHopperE4m3SaltDead_v1")
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


# ---- bytes and int7 entries, the accumulator, SHA-512's words, ncp-v2's dequantization ------------------------------------

I7, A16 = Array(7, BIT), Array(16, U8)


def _s8(b):
    """An int7 entry as its two's-complement byte: bit 6 copied into bit 7."""
    return list(b) + [b[6]]


def sum_mod(C, cols, width):
    """The sum of weighted columns of bits mod 2^width: the constants folded into one, full adders until no column holds
    more than two bits, then a ripple; a carry out of the top column is dropped, so it costs no AND."""
    k = sum(sum(w.c for w in col if w.is_const()) << i for i, col in enumerate(cols[:width]))
    cols = [[w for w in col if not w.is_const()] for col in cols[:width]] + [[] for _ in range(width - len(cols))]
    for i in range(width):
        if k >> i & 1:
            cols[i].append(ONE)
    while any(len(col) > 2 for col in cols):
        nxt = [[] for _ in range(width)]
        for i, col in enumerate(cols):
            while len(col) >= 3:
                x, y, z = col.pop(), col.pop(), col.pop()
                nxt[i].append(x ^ y ^ z)
                if i + 1 < width:
                    nxt[i + 1].append(maj(C, x, y, z))
            nxt[i].extend(col)
        cols = nxt
    out, carry = [], ZERO
    for i, col in enumerate(cols):
        x, y = (col + [ZERO, ZERO])[:2]
        out.append(x ^ y ^ carry)
        if i + 1 < width:
            carry = maj(C, x, y, carry)
    return out


def _const_cols(cols, k, width):
    for i in range(width):
        if k % (1 << width) >> i & 1:
            cols[i].append(ONE)
    return cols


def pack_be4(C, b0, b1, b2, b3):
    """``PouwPackBE4_v1``: the four entries' bytes, b0 the most significant: wiring."""
    return [x for b in (b3, b2, b1, b0) for x in _s8(b)]


def noise_byte(C, w, c):
    """``PouwNoiseByte_v1``: the low six bits of byte c mod 4 of w, chosen by c's two low bits, plus 225 (−31 mod 256)."""
    lo, hi = mux(C, c[0], w[8:14], w[0:6]), mux(C, c[0], w[24:30], w[16:22])
    return add(C, mux(C, c[1], hi, lo) + [ZERO, ZERO], const_bits(225, 8), width=8)


def form_x(C, a, e):
    """``PouwFormX_v1``: a + e + 128 mod 256, the 128 a flip of the top bit."""
    s = add(C, _s8(a), e, width=8)
    return s[:7] + [NOT(s[7])]


def form_x3(C, a, e1, e2):
    """``PouwFormX3_v1``: 128 − (a + e1 + e2) mod 256, as ~a + ~e1 + ~e2 + 131."""
    cols = [[NOT(x), NOT(y), NOT(z)] for x, y, z in zip(_s8(a), e1, e2)]
    return sum_mod(C, _const_cols(cols, 131, 8), 8)


def add_sub8(C, b, f, g):
    """``PouwAddSub8_v1``: b + f − g mod 256, as b + f + ~g + 1."""
    cols = [[x, y, NOT(z)] for x, y, z in zip(_s8(b), f, g)]
    return sum_mod(C, _const_cols(cols, 1, 8), 8)


def neg_beta_sum16(C, acc, *y):
    """``PouwNegBetaSum16_v1``: acc − 128 · Σ s8(y) mod 2^32. An entry's −128 · y is ~y_j at 2^(j+7) for j < 7, y_7 at 2^14,
    and −127 · 2^7 of the constant, so the sum takes no AND but the carry-save tree's."""
    cols = [[b] for b in acc]
    for v in y:
        for j in range(7):
            cols[j + 7].append(NOT(v[j]))
        cols[14].append(v[7])
    return sum_mod(C, _const_cols(cols, -len(y) * 127 * 128, 32), 32)


def dot16_u8s8(C, acc, *xy):
    """``PouwDot16U8S8_v1``: acc + Σ u8(x) · s8(y) mod 2^32, every partial product in one carry-save sum. y's sign bit weighs
    −2^7, so x_i y_7 enters complemented at 2^(i+7) with −2^(i+7) in the constant (Baugh–Wooley)."""
    x, y = xy[:16], xy[16:]
    cols = [[b] for b in acc]
    for xt, yt in zip(x, y):
        for i, a in enumerate(xt):
            for j, b in enumerate(yt):
                p = C.AND(a, b)
                cols[i + j].append(NOT(p) if j == 7 else p)
    return sum_mod(C, _const_cols(cols, -len(x) * 255 * 128, 32), 32)


#: SHA-512's Σ0, Σ1, σ0 and σ1 by their words' names
_SIGMAS = {"BigSigma0": SH.big_sigma0, "BigSigma1": SH.big_sigma1, "SmallSigma0": SH.small_sigma0,
          "SmallSigma1": SH.small_sigma1}


def _def(word, build, ret, doc):
    params, _ = word.signature()
    return definition(word.name, 2, build, tuple((n, BL.bit_type(t)) for n, t in params), ret, word=word,
                      conformance=word.conformance, doc=f"{doc}, on bits: {word.id}")


PouwPackBE4 = _def(W.PouwPackBE4, pack_be4, U32, "four int7 entries' bytes, big-endian")
PouwNoiseByte = _def(W.PouwNoiseByte, noise_byte, U8, "a noise entry from byte c mod 4 of w")
PouwDot16U8S8 = silicon_step(_def(W.PouwDot16U8S8, dot16_u8s8, U32, "acc + Σ u8(x)·s8(y) over 16 products, mod 2^32"),
                             BLACKWELL_SM120_U8S8_M16N8K32)
PouwFormX = _def(W.PouwFormX, form_x, U8, "a + e + 128 mod 256")
PouwFormX3 = _def(W.PouwFormX3, form_x3, U8, "128 − (a + e1 + e2) mod 256")
PouwAdd8 = _def(W.PouwAdd8, lambda C, b, f: add(C, _s8(b), f, width=8), U8, "b + f mod 256")
PouwSub8 = _def(W.PouwSub8, lambda C, b, f: sub(C, _s8(b), f, 8)[0], U8, "b − f mod 256")
PouwAddSub8 = _def(W.PouwAddSub8, add_sub8, U8, "b + f − g mod 256")
PouwNegBetaSum16 = _def(W.PouwNegBetaSum16, neg_beta_sum16, U32, "acc − 128·Σ s8(y) over 16 entries, mod 2^32")
PouwAdd64Lo = _def(W.PouwAdd64Lo, lambda C, alo, blo: SH.add2(C, alo, blo, every=None), U32,
                   "the low half of a 64-bit sum")
PouwAdd64Hi = _def(W.PouwAdd64Hi, lambda C, ahi, alo, bhi, blo: SH.add2(C, alo + ahi, blo + bhi, every=None)[32:], U32,
                   "the high half of a 64-bit sum")
PouwCh32 = _def(W.PouwCh32, SH.ch, U32, "SHA-512's Ch on one half, bit by bit e ? f : g")
PouwMaj32 = _def(W.PouwMaj32, SH.maj, U32, "SHA-512's Maj on one half")


def _sigma_half(name, half):
    f = _SIGMAS[name]
    return _def(getattr(W, f"PouwSha512{name}{half}"),
                (lambda C, hi, lo: f(lo + hi)[32:]) if half == "Hi" else (lambda C, hi, lo: f(lo + hi)[:32]),
                U32, f"SHA-512's {name} on hi:lo, its {half.lower()} half")


PouwSha512BigSigma0Hi, PouwSha512BigSigma0Lo = _sigma_half("BigSigma0", "Hi"), _sigma_half("BigSigma0", "Lo")
PouwSha512BigSigma1Hi, PouwSha512BigSigma1Lo = _sigma_half("BigSigma1", "Hi"), _sigma_half("BigSigma1", "Lo")
PouwSha512SmallSigma0Hi, PouwSha512SmallSigma0Lo = _sigma_half("SmallSigma0", "Hi"), _sigma_half("SmallSigma0", "Lo")
PouwSha512SmallSigma1Hi, PouwSha512SmallSigma1Lo = _sigma_half("SmallSigma1", "Hi"), _sigma_half("SmallSigma1", "Lo")
PouwI32ToF32Rn = _def(W.PouwI32ToF32Rn, fp.s32_to_f32, F32, "cvt.rn.f32.s32 (`fp.s32_to_f32`, as `S32ToF32Rn_v2`)")
