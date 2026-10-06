"""PoUW's own word primitives as Boolean Definitions: the next version of each `words` primitive Pearl-C4's (`pc4`) and
Pearl-C's (`pc8`) row units call, and of the four Keccak words their hiding tiles' digest calls (`hashes`, Keccak-p[1600]
on (lo, hi) half-lanes), the same function on bits.

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
    PouwPc8LeafH3_v2   (c, u: Array<32, Value<1>>, seg: Array<256, Array<32, Value<1>>>,
                        dig: Array<32, Array<512, Value<1>>>, at: Array<32, Value<1>>) -> Array<512, Value<1>>
                                                                                                word view: PouwPc8LeafH3_v1

Each is traced from a builder over core's `forms` (`verity.primitives.circuits.boolean.trace.definition`), so its And gates are the builder's
live ANDs but for its commits (`forms.Circuit.commit`, a row of a `forms` lowering and no gate); a backend whose lowering
takes `forms` builders (C-Flock's pieces) can run the builder itself (``.build``).

* the windows: the 15 calls of the atom's own builder (`verity_catalog.definitions.boolean.gemm`) that build a window's masked words, on a
  view of the circuit that keeps its own AND memo (`_Shared`), so atom j's 32 products, and every part of its step that
  doesn't read the accumulator, are built once for its 2^j words on any circuit, C-Flock's (which has no common-subexpression
  pass) included: on sm_120 102,409 ANDs where the 15 calls are 145,651.
* `window_word`: a 4-to-16 decode and one AND per bit and mask, so a constant m (each of the cap word's reads) costs none.
* the salt-dead bits: the atom's builder at the support's two ends (E′ the one code c_β at all 32 positions, F the
  constant ±0x43) and the cast, on an `_Shared` view, so the 32 equal products are built once and the two ends share
  every AND that doesn't read F's sign: on sm_120 2,493 ANDs where two atom calls and casts are 20,062 (1,806 against
  15,230 on the H100).

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
* `chi32`: Keccak's χ on a half-lane, one AND per bit; xor is free.  Its output is committed: a lowering that keeps a
  call's outputs as XOR forms (C-Flock's) otherwise carries χ's XOR terms into the next round's θ, and over 12 rounds they
  compound until lowering one Keccak-p[1600, 12] runs out of memory; committed, it is 38,400 rows (19,200 ANDs and as
  many commits), χ's operands θ's outputs of 11 committed bits at most.
* `rotl64`: a barrel rotator over the 64-bit word hi:lo, one stage per bit of the amount's low six, the largest first, so
  each stage's live ANDs are only the window the returned half reads.  A constant amount, as every call in `hashes` has,
  folds each stage to wiring with no AND (`forms`' AND of a constant is no gate).
* `leaf_h3`: the served unit's -h3 tile leaf (`words.PouwPc8LeafH3`): 27 compressions, the word's segment digest's ten
  (prefix, eight blocks, pad) and the outer chain's 16 blocks of digests and its pad, the tile height picking the chain's
  state after 4, 8, 12 or 16 blocks; the height's Q midstate and pad are constants picked by XOR, so only the muxes, the
  picked state and the compressions have ANDs.
"""
from __future__ import annotations

from functools import reduce

from verity.primitives.circuits import boolean as BL
from verity.primitives.circuits.types import Array
from verity_catalog.silicon import fp
from verity_catalog.definitions.boolean import gemm as BG
from verity_catalog.definitions.boolean import scalar as BS
from verity.primitives.circuits.boolean.forms import (NOT, ONE, OR, ZERO, add, and_reduce, compress_columns, const_bits, lzc, multiply, mux,
                                     or_reduce, shl_window, sub)
from verity.primitives.circuits.boolean.trace import definition
from verity.primitives.commitments.gates import sha512 as G
from verity.primitives.commitments.rowleaf import sha512_row_seg_prefix, sha512_row_seg_segment_prefix

from . import words as W

__all__ = ["PouwBf16Neg", "PouwBlackwellE4m3SaltDead", "PouwBlackwellE4m3Window", "PouwChi32", "PouwETilde", "PouwF32SqrtRn",
           "PouwHopperE4m3SaltDead", "PouwHopperE4m3Window", "PouwPc8LeafH3", "PouwRotl64Hi", "PouwRotl64Lo", "PouwSqAcc",
           "PouwSqCeil", "PouwSqGe", "PouwWindowWord", "PouwXor32", "chi32", "e_tilde", "f32_sqrt_rn", "isqrt", "leaf_h3",
           "rotl64", "salt_dead", "sq_acc", "sq_ceil", "sq_ge", "square", "up48", "window", "window_word"]

BIT = BL.BIT
F32 = U32 = Array(32, BIT)
BF16, E2M1, U8 = Array(16, BIT), Array(4, BIT), Array(8, BIT)
SQ, SQ48 = Array(576, BIT), Array(64, BIT)
U4, CODES, MASKED = Array(4, BIT), Array(W.WINDOW, U8), Array(W.MASKED.w, BIT)
D512 = Array(W.D512.w, BIT)
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


def _be_words(octets: list) -> list[list]:
    """Bytes (each 8 bits, LSB first) as SHA-512's 64-bit big-endian words, each LSB first."""
    return [[octets[t + 7 - p // 8][p % 8] for p in range(64)] for t in range(0, len(octets), 8)]


def _octets(bits: list) -> list[list]:
    return [list(bits[i:i + 8]) for i in range(0, len(bits), 8)]


def _digest_bits(words: list[list]) -> list:
    """SHA-512's eight state words as the digest's 64 bytes, little-endian bits."""
    return [b for w in words for k in range(8) for b in w[8 * (7 - k):8 * (8 - k)]]


def _pick(sel: list, consts: list[int], width: int) -> list:
    """The constant ``consts[i]`` (``width`` bits) where ``sel`` is one-hot at i: an XOR of the selectors, no AND."""
    out = []
    for p in range(width):
        bit = ZERO
        for s, v in zip(sel, consts):
            if v >> p & 1:
                bit = bit ^ s
        out.append(bit)
    return out


def leaf_h3(C, c, u, *rest):
    """``PouwPc8LeafH3_v1``: the word put in its segment (a 7-bit decode of its row's parity, chunk and column, one mux a
    word), d_s (the segment prefix's j and s from ``at``, then the segment's eight blocks and the pad), d_s put in the
    digests (a 5-bit decode, one mux a digest), and the leaf over 16 blocks of digests from the tile height's Q midstate,
    the chain's state after S/2 of them and the pad picked by the height (a 2-bit decode).  Every compression is core's
    `compress_gadget`."""
    seg = [list(w) for w in rest[:W.SEG_WORDS]]
    dig = [list(d) for d in rest[W.SEG_WORDS:W.SEG_WORDS + W.LEAF_SEGMENTS]]
    at = list(rest[-1])
    r, col, code = at[0:6], at[6:12], at[12:14]
    slot = fp.decode(C, [*col[0:5], col[5], r[0]])                 # C̃'s word, less its bit 5 (U's is C̃'s + 32)
    for w in range(W.SEG_WORDS):
        key = (w & 31) | ((w >> 6) & 1) << 5 | (w >> 7) << 6
        seg[w] = mux(C, slot[key], list(u if w >> 5 & 1 else c), seg[w])
    height = fp.decode(C, code)
    n_seg = [ZERO, ZERO, ZERO, NOT(code[0]), code[0] ^ code[1], C.AND(code[0], code[1])] + [ZERO] * 2
    prefix = _octets(const_bits(int.from_bytes(sha512_row_seg_segment_prefix(0, 1, 8192), "little"), 1024))
    prefix[40] = [*r[1:6], ZERO, ZERO, ZERO]                        # u64be j's last byte: j = r >> 1 < 32
    prefix[48] = n_seg                                              # u64be s's last byte: s = rows / 2 = 8 (code + 1)
    body = _octets([b for w in seg for b in w])
    state = [const_bits(v, 64) for v in G.IV]
    for blk in [prefix] + [body[i:i + 128] for i in range(0, len(body), 128)]:
        state = G.compress_gadget(C, state, _be_words(blk))
    d_s = G.compress_gadget(C, state, [const_bits(v, 64) for v in G.words_be(G.pad_tail(128 + 4 * W.SEG_WORDS))])
    at_seg = fp.decode(C, r[1:6])
    digs = [_be_words(_octets(d)) for d in dig]
    digs = [[mux(C, at_seg[q], d_s[t], digs[q][t]) for t in range(8)] for q in range(W.LEAF_SEGMENTS)]
    rows = [16 * (h + 1) for h in range(4)]
    mids = [G.midstate(sha512_row_seg_prefix(h * 512 * 8)) for h in rows]
    state = [_pick(height, [m[i] for m in mids], 64) for i in range(8)]
    kept = []
    for q in range(0, W.LEAF_SEGMENTS, 2):
        state = G.compress_gadget(C, state, digs[q] + digs[q + 1])
        if (q + 2) % 8 == 0:
            kept.append(state)
    state = [[reduce(lambda x, y: x ^ y, [C.AND(height[h], kept[h][i][p]) for h in range(4)]) for p in range(64)]
             for i in range(8)]
    pads = [G.words_be(G.pad_tail(128 + 64 * (h // 2))) for h in rows]
    pad = [_pick(height, [p[i] for p in pads], 64) for i in range(16)]
    return _digest_bits(G.compress_gadget(C, state, pad))


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
PouwPc8LeafH3 = definition("PouwPc8LeafH3", 2, leaf_h3, (("c", F32), ("u", F32), ("seg", Array(W.SEG_WORDS, U32)),
                                                       ("dig", Array(W.LEAF_SEGMENTS, D512)), ("at", U32)), D512,
                           word=W.PouwPc8LeafH3, conformance=W.PouwPc8LeafH3.conformance,
                           doc="The -h3 tile leaf of one word put in its segment, on bits: PouwPc8LeafH3_v1")
