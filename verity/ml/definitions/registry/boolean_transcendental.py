"""`ref_prims`' correctly rounded transcendental primitives as Boolean Definitions: the next version of each word primitive, the
same function on bits (`verity.ml.circuits.boolean.trace`).

    F32ExpRn_v2   (x: Array<32, Value<1>>) -> Array<32, Value<1>>    word view: F32ExpRn_v1 (ref_prims)

`F32ExpRn_v2` computes exp(x) = 2^n 2^f to within 2^-55 relative and rounds that once, nearest even, subnormal results
included. No binary32 x whose exp(x) rounds to a finite nonzero word has exp(x) within 2^-29 ulp (so 2^-53 relative) of a
rounding boundary, so the one rounding is exp(x)'s correct rounding, the word's; `benchmarks/f32_exhaustive/run.py boolean
F32ExpRn` holds it to `F32ExpRn_v1` on all 2^32 words.

- t = x log2(e) in fixed point, F = 57 fraction bits: the 24-bit significand times log2(e) at K = 64 bits (a constant, so
  its rows are the significand's shifts, by the digits of its non-adjacent form), shifted right by 133 - e. For x < 0 the
  bits are complemented, -|t| - 2^-F. So |t' - t| < 3 2^-F + 104 2^-(K+1).
- n = floor(t'), f = t' - n = a 2^-8 + b 2^-16 + c 2^-24 + d (d < 2^-24), and 2^f = A(a) (1 + B(b)) (1 + G(c)) (1 + D(d)):
  A(a) = 2^(a/256), B(b) = 2^(b 2^-16) - 1 and G(c) = 2^(c 2^-24) - 1 tables at W = 60 fraction bits (floor), each bit the
  XOR of the monomials of its index's algebraic normal form (2^8 - 9 ANDs an index), and D(d) = d ln2 + (d_hi ln2)^2 / 2
  (d_hi d's top 12 bits; the cube is below 2^-75). The products go smallest first, each a table times a word:
  s = G + D + G D, r = B + s + B s, v = A + A r, by `_tab_mul`, whose partial-product rows cost no AND. Each product's word
  and tables are committed bits (a Flock row each, no gate), so a partial product is two committed bits, not a wide form.
- Each product drops its partial products 8 columns below its unit and adds back their bound, so it is its floor or one
  more; the error in v is below 10 2^-W, and v stays in [2^W, 2^(W+1)): v >= A >= 2^W, and v exceeds 2^f 2^W by at most 9
  while 2^f 2^W <= 2^(W+1) (1 - 2^-(2^-F)) < 2^(W+1) - 11 (F = W - 3).
- 2^n v rounds once: n + 127 >= 1 gives a normal word, else v's top 24 bits shift right by 1 - (n + 127) with their guard
  and sticky (a subnormal or zero), and the rounding's carry runs into the exponent field.
- Specials, as the word's: a NaN gives 0x7FC00000; x >= 0x42B17218 (88.7228394, the first x whose exp rounds past the
  largest finite word), +inf among them, gives +inf; a negative x of magnitude 0x42CFF1B5 (103.972084, the first whose exp
  is below 2^-150) or more, -inf among them, gives +0.
"""
from __future__ import annotations

import decimal
from functools import cache

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, compress_columns, const_bits, mux, or_reduce, shr, sub
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array

from . import ref_prims as R

__all__ = ["F32ExpRn", "exp_rn"]

F32 = Array(32, BL.BIT)
F, K, W = 57, 64, 60
#: |x| at and past which exp(x) is +inf (x > 0) and +0 (x < 0): the word's own thresholds
OVERFLOW, UNDERFLOW = 0x42B17218, 0x42CFF1B5
_DEC = decimal.Context(prec=120)


def _floor_scaled(v: decimal.Decimal, bits: int) -> int:
    return int(_DEC.multiply(v, decimal.Decimal(2) ** bits).to_integral_value(rounding=decimal.ROUND_FLOOR))


@cache
def _constants() -> dict:
    """log2(e) at K bits (nearest), ln 2 and ln(2)^2 / 2 at W + 24 bits (floor), and the tables A, B, G at W bits (floor)."""
    c = _DEC
    ln2 = c.ln(decimal.Decimal(2))
    two_to = [None] + [[_floor_scaled(c.exp(c.multiply(c.divide(decimal.Decimal(i), decimal.Decimal(2) ** step), ln2)), W)
                        for i in range(256)] for step in (8, 16, 24)]
    return {"log2e": int(c.multiply(c.divide(decimal.Decimal(1), ln2), decimal.Decimal(2) ** K)
                         .to_integral_value(rounding=decimal.ROUND_HALF_EVEN)),
            "ln2": _floor_scaled(ln2, W + 24), "half_ln2_sq": _floor_scaled(c.divide(c.multiply(ln2, ln2), decimal.Decimal(2)), W + 24),
            "A": two_to[1], "B": [v - (1 << W) for v in two_to[2]], "G": [v - (1 << W) for v in two_to[3]]}


def _naf(c: int) -> list[tuple[int, int]]:
    """``c``'s non-adjacent form: (k, ±1) with c = sum d 2^k."""
    out, k = [], 0
    while c:
        if c & 1:
            d = 2 - (c & 3)
            out.append((k, d))
            c -= d
        c >>= 1
        k += 1
    return out


def _times_const(C, x, c: int, low: int = 0):
    """``floor(x * c / 2^low)`` or one more, for an unsigned word x and a constant c >= 0: a row per digit of c's
    non-adjacent form (x, or ~x and a constant for a -1), each row's bits below column ``low - 8`` left out and their bound
    added back, so the sum never falls below x c and exceeds it by less than (digits + 1) 2^(low - 8) < 2^low."""
    n = len(x)
    width = n + c.bit_length() + 1
    cols: list[list] = [[] for _ in range(width)]
    digits = _naf(c)
    cut = max(0, low - 8)
    const = (len(digits) + 1) << cut if cut else 0
    for k, d in digits:
        row = x if d > 0 else [NOT(b) for b in x]
        if d < 0:
            const += (1 << k) - (1 << (n + k))
        for i, b in enumerate(row):
            if i + k >= cut:
                cols[i + k].append(b)
    const %= 1 << width
    for j in range(cut, width):
        if const >> j & 1:
            cols[j].append(ONE)
    return compress_columns(C, cols[cut:])[low - cut:width - cut]


def _xor_all(terms):
    terms = list(terms)
    if not terms:
        return ZERO
    while len(terms) > 1:
        terms = [terms[i] ^ terms[i + 1] for i in range(0, len(terms) - 1, 2)] + terms[len(terms) - len(terms) % 2:]
    return terms[0]


def _monomials(C, idx):
    """The 2^|idx| products of subsets of ``idx``'s bits (mask i: the bits set in i), one AND each beyond the bits."""
    m = [ONE]
    for b in idx:
        m += [C.AND(x, b) for x in m]
    return m


def _table(mono, values, width: int):
    """``values[idx]``'s low ``width`` bits, each the XOR of the monomials ``mono`` of its algebraic normal form."""
    tt = np.array([[(v >> j) & 1 for v in values] for j in range(width)], dtype=np.uint8)
    h = 1
    while h < len(values):
        tt = tt.reshape(width, -1, 2, h)
        tt[:, :, 1, :] ^= tt[:, :, 0, :]
        h *= 2
    return [_xor_all(mono[i] for i in np.flatnonzero(row)) for row in tt.reshape(width, len(values))]


def _tab_mul(C, mono, cs, x, cut: int, plus=None, rows=()):
    """``floor((c x + p 2^cut + sum rows) / 2^cut)`` or one more, for the tables c = ``cs[idx]`` and p = ``plus[idx]`` of the
    index whose monomials are ``mono``, an unsigned word x and ``rows`` (bits, shift) addends. With 2c + 1 = sum_k (1 - 2
    n_k) 2^k (k < KK, N = sum_k n_k 2^k a table too), 2 c x = sum_k (x XOR n_k) 2^k + ~x - (2^|x| - 1)(N + 1): every
    partial-product row is a table bit XORed into x, and the rest one table row, so only the sum's full adders are ANDs.
    Each row's bits below column ``cut - 7`` of 2cx are left out and their bound added back (`_times_const`). x and both
    tables are committed (a Flock row a bit, no gate), so a partial product is two committed bits, not a table bit's
    hundreds of monomials or an adder output's form, in every full adder it reaches."""
    x = C.commit(x)
    n = len(x)
    plus = plus or [0] * len(cs)
    KK = max((2 * c + 1).bit_length() for c in cs)
    Ns = [((1 << KK) - 1) ^ (c + (1 << (KK - 1))) for c in cs]
    g = max(0, cut + 1 - 8)
    count = KK + 2 + len(rows)
    assert count < 1 << (cut + 1 - g)
    top = max([n + KK + 1, max(plus).bit_length() + cut + 1] + [len(b) + sh + 1 for b, sh in rows]) + 2
    Q = [((p << (cut + 1)) - ((1 << n) - 1) * (N + 1) + (count << g)) % (1 << top) for p, N in zip(plus, Ns)]
    cols: list[list] = [[] for _ in range(top - g)]
    nk = C.commit(_table(mono, Ns, KK))
    for k in range(KK):
        for i in range(max(0, g - k), n):
            cols[i + k - g].append(x[i] ^ nk[k])
    for i in range(g, n):
        cols[i - g].append(NOT(x[i]))
    for bits, sh in rows:
        for i in range(max(0, g - sh - 1), len(bits)):
            cols[i + sh + 1 - g].append(bits[i])
    for j, w in enumerate(C.commit(_table(mono, [q >> g for q in Q], top - g))):
        cols[j].append(w)
    return compress_columns(C, cols)[cut + 1 - g:top - g]


def _trim(bits, width: int):
    """``bits`` at ``width``, the value promised to fit."""
    return list(bits[:width]) + [ZERO] * max(0, width - len(bits))


def _ge_const(C, w, k: int):
    """``w >= k`` for an unsigned word and a constant."""
    return NOT(sub(C, w, const_bits(k, len(w)), len(w))[1])


def exp_rn(C, x):
    """`F32ExpRn_v1` on bits: RN(exp(x)), the module's construction."""
    k = _constants()
    X = fp.unpack(C, x)
    # t' = x log2(e) at 2^-F, 9 integer bits, two's complement
    P = _trim(_times_const(C, X.sig, k["log2e"], 17 + K - F), F + 8)   # sig log2e at 2^-F when e = 133
    dd, _ = sub(C, const_bits(133, 8), X.e_eff, 8)
    T = [b ^ X.s for b in shr(C, P, dd) + [ZERO]]
    n, fr = T[F:], T[:F]
    a, b, c, d = fr[F - 8:], fr[F - 16:F - 8], fr[F - 24:F - 16], fr[:F - 24]
    # 2^f at 2^-W
    dl = _times_const(C, d, k["ln2"], F + 24)                           # d ln2 at 2^-W
    dh = d[F - 36:]                                                      # d's top 12 bits, at 2^-36
    sq = _square(C, dh)                                         # d_hi^2 at 2^-72
    q = _times_const(C, sq, k["half_ln2_sq"], 72 + 24)                  # (d_hi ln2)^2 / 2 at 2^-W
    dpoly = _trim(add(C, dl, q, width=W - 24), W - 24)
    mc = _monomials(C, c)
    s = _trim(_tab_mul(C, mc, k["G"], dpoly, W, plus=k["G"], rows=[(dpoly, W)]), W - 16)
    mb = _monomials(C, b)
    r = _trim(_tab_mul(C, mb, k["B"], s, W, plus=k["B"], rows=[(s, W)]), W - 8)
    ma = _monomials(C, a)
    v = _trim(_tab_mul(C, ma, k["A"], r, W, plus=k["A"]), W + 1)
    # 2^n v to f32, subnormals included: E = n + 127 (10-bit two's complement)
    E = add(C, n + [n[-1]], const_bits(127, 10), width=10)
    pos = NOT(OR(C, E[9], NOT(or_reduce(C, E[:9]))))                   # E >= 1
    sh, _ = sub(C, const_bits(1, 10), E, 10)                            # 1 - E, used when E < 1
    y, st = fp.shr_sticky(C, v[W - 24:W + 1], [C.AND(NOT(pos), b) for b in sh])
    sticky = OR(C, or_reduce(C, v[:W - 24]), st)
    mant, g = y[1:], y[0]
    ex = [C.AND(pos, b) for b in E[:8]]                                 # the exponent field when normal, else 0
    word = add(C, mant[:23] + ex, [C.AND(g, OR(C, sticky, mant[0]))], width=31) + [ZERO]
    # specials
    mag = list(x[:31])
    word = mux(C, C.AND(X.s, _ge_const(C, mag, UNDERFLOW)), const_bits(0, 32), word)
    word = mux(C, C.AND(NOT(X.s), _ge_const(C, mag, OVERFLOW)), const_bits(fp.F32_INF, 32), word)
    return mux(C, X.nan, const_bits(R.F32_NAN, 32), word)


def _square(C, x):
    """``x^2`` for an unsigned word: x_i x_j once per pair (doubled by its column), x_i alone on the diagonal."""
    n = len(x)
    cols: list[list] = [[] for _ in range(2 * n)]
    for i in range(n):
        cols[2 * i].append(x[i])
        for j in range(i + 1, n):
            cols[i + j + 1].append(C.AND(x[i], x[j]))
    return compress_columns(C, cols)[:2 * n]


F32ExpRn = definition("F32ExpRn", 2, exp_rn, (("x", F32),), F32, word=R.F32ExpRn,
                      doc="F32ExpRn_v1 on bits: RN(exp(x)), subnormal results, a NaN 0x7FC00000, overflow +inf, underflow +0")
