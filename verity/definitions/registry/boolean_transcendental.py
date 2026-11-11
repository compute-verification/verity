"""`ref_prims`' correctly rounded transcendental primitives as Boolean Definitions: the next version of each word primitive, the
same function on bits (`verity.circuits.boolean.trace`).

    F32ExpRn_v2   (x: Array<32, Value<1>>) -> Array<32, Value<1>>    word view: F32ExpRn_v1 (ref_prims)
    F32LogRn_v2   (x: Array<32, Value<1>>) -> Array<32, Value<1>>    word view: F32LogRn_v1 (ref_prims)

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

`F32LogRn_v2` computes ln(x) = E ln2 + L1 + L2 + L3 + ln(1 + u) in fixed point at LF = 77 fraction bits and rounds that
once, nearest even. ln(x) of a binary32 x != 1 lies at least 2^-54 from a rounding boundary, except near 1: at least
2^-62.8 for x within 2^-15 of 1, and 2^-70.6 for x within 2^-22 (the closest, at 0x3F7FFFFE, 1 - 2^-23). The error is
below each, so the one rounding is ln(x)'s correct rounding, the word's; `benchmarks/f32_exhaustive/run.py boolean
F32LogRn` holds it to `F32LogRn_v1` on all 2^32 words.

- x = m 2^E with m = 1 + M 2^-23: a subnormal's significand shifted up by its leading-zero count, E = e - 127 - lz.
- Three stages divide m by a reciprocal table each, indexed by the 8 bits just below the last's unit: r1 = m R1(t), t m's
  top 8 fraction bits, R1 at 2^-9; r2 = r1 R2(b), b = floor((r1 - 1) 2^15), R2 at 2^-16; r3 = r2 R3(g), g = floor((r2 -
  1) 2^22), R3 at 2^-23. Each R is nearest the reciprocal of its index's interval's midpoint, so u3 = r3 - 1 lies in
  [-2^-22, 2^-22) (`_log_tables` checks every range); L1, L2, L3 = -ln R at 2^-LF (nearest). R1(0) = R2(0) = R2(-1) =
  R3(0) = R3(-1) = 1 and R1(255) = 1/2, so near 1 a stage multiplies by 1 (its L is 0) or halves (L1 = ln2 cancels E = -1).
- Each product is a table times the bits below its index (`_stage`: `_tab_mul`'s partial products, which cost no AND),
  r1 exact at 2^-32 and r2 at 2^-48, r3 at 2^-60 its floor or one more, and exact when r2 R3 is a multiple of 2^-60, as it
  is within 2^-15 of 1 (r2 = x there).
- ln(1 + u) = u - u^2/2 + u^3/3 (u^4/4 < 2^-90). With u = H + d + rho, H u's top 8 bits (two's complement, at 2^-29), d
  the next 18 (at 2^-47): -u^2/2 + u^3/3 = F(H) - H d - d^2/2 with F = -H^2/2 + H^3/3 a table at 2^-LF (nearest), -H d a
  table times d (`_pm_rows`) and d^2/2 its partial products from 2^-LF up. Left out: H^2 d < 2^-73, the columns of d^2
  below 2^-LF (< 2^-73.9), H d^2 < 2^-80, and rho's term below 2^-69 (rho = 0 within 2^-15 of 1).
- E ln2 is E times ln2 at 2^-LF (rows by the digits of its non-adjacent form), its error below 2^-70.7 and 0 for E in {0,
  -1}. So the error is below 2^-59 (r3's floor, at 2^-60), below 2^-71 within 2^-15 of 1 (the tables, F's dropped terms
  and the ones' complement), and below 2^-76 within 2^-22 (F's rounding and the ones' complement).
- One sum, two's complement, 8 integer bits; |ln x| by ones' complement; its top bit at 2^(6 - lz), lz a leading-zero count
  over 2^6 down to 2^-25 (|ln x| >= 2^-24 for x != 1); a shift left by lz; and the guard bit added in, its carry into the
  exponent field 133 - lz. A value within the error of ln(x) is no rounding boundary, so the guard bit alone decides.
- Specials, as the word's: a NaN and x < 0 (-inf among them, -0 not) give 0x7FC00000; ±0 gives -inf, +inf gives +inf, and 1
  gives +0 (the sum is exactly 0, the leading-zero count's all-zero flag clears the exponent).
"""
from __future__ import annotations

import decimal
from fractions import Fraction
from functools import cache

import numpy as np

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, compress_columns, const_bits, lzc, mux, or_reduce, shr, sub
from verity.circuits.boolean.trace import definition
from verity.definitions.silicon import fp
from verity.circuits.types import Array

from . import ref_prims as R

__all__ = ["F32ExpRn", "F32LogRn", "exp_rn", "log_rn"]

F32 = Array(32, BL.BIT)
F, K, W = 57, 64, 60
#: the fraction bits of ln(x)'s sum
LF = 77
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


class _Sum:
    """Weighted bits and an integer constant summed mod 2^width: ``add`` puts bits in at a shift, ``sub`` takes them off (their
    complements, and -(2^n - 1) 2^shift into the constant), ``table`` adds a committed table less its least entry."""

    def __init__(self, width: int):
        self.cols: list[list] = [[] for _ in range(width)]
        self.const = 0

    def add(self, bits, sh: int = 0):
        for i, b in enumerate(bits):
            if i + sh < len(self.cols):
                self.cols[i + sh].append(b)

    def sub(self, bits, sh: int = 0):
        self.add([NOT(b) for b in bits], sh)
        self.const -= ((1 << len(bits)) - 1) << sh

    def table(self, C, mono, values, sh: int = 0):
        low = min(values)
        self.add(C.commit(_table(mono, [v - low for v in values], max(v - low for v in values).bit_length())), sh)
        self.const += low << sh

    def total(self, C):
        k = self.const % (1 << len(self.cols))
        return compress_columns(C, [c + ([ONE] if k >> j & 1 else []) for j, c in enumerate(self.cols)])[:len(self.cols)]


def _pm_rows(C, mono, cs, x):
    """``(rows, consts)`` with 2 c x = sum rows + consts[idx] for the table c = ``cs[idx]`` >= 0 and committed bits x: with
    2c + 1 = sum_k (1 - 2 n_k) 2^k (k < KK, N = sum_k n_k 2^k a committed table), 2cx = sum_k (x XOR n_k) 2^k + ~x -
    (2^|x| - 1)(N + 1), so a partial product is a table bit XORed into x and costs no AND (`_tab_mul`)."""
    KK = max((2 * c + 1).bit_length() for c in cs)
    Ns = [((1 << KK) - 1) ^ (c + (1 << (KK - 1))) for c in cs]
    nk = C.commit(_table(mono, Ns, KK))
    rows = [([b ^ nk[k] for b in x], k) for k in range(KK)] + [([NOT(b) for b in x], 0)]
    return rows, [-((1 << len(x)) - 1) * (N + 1) for N in Ns]


def _stage(C, mono, cs, x, plus, rows, cut: int, width: int):
    """``floor(v / 2^cut) mod 2^width`` or one more, and exactly it when v is a multiple of 2^cut, for v = c x + p + rows: the
    tables c = ``cs[idx]`` >= 0 and p = ``plus[idx]`` of the index whose monomials are ``mono``, committed bits x, and
    ``rows`` (bits, shift, negated) addends. It sums 2v (`_pm_rows`), leaving out every bit below the column g that keeps the
    bound it adds back, with the table's low g bits rounded up, under 2^(cut + 1)."""
    pm, consts = _pm_rows(C, mono, cs, x)
    ent = [(b, i + sh) for bits, sh in pm for i, b in enumerate(bits)]
    const = 0
    for bits, sh, neg in rows:
        ent += [(NOT(b) if neg else b, i + sh + 1) for i, b in enumerate(bits)]
        const -= ((1 << len(bits)) - 1) << (sh + 1) if neg else 0
    top = width + cut + 1

    def bound(g):
        return sum(1 << c for _, c in ent if c < g) + (1 << g) - 1
    g = max(g for g in range(cut + 2) if bound(g) < 1 << (cut + 1))
    lost = bound(g) - (1 << g) + 1
    Q = [-(-(2 * p + q + const + lost) >> g) % (1 << (top - g)) for p, q in zip(plus, consts)]
    cols: list[list] = [[] for _ in range(top - g)]
    for b, c in ent:
        if g <= c < top:
            cols[c - g].append(b)
    for j, w in enumerate(C.commit(_table(mono, Q, top - g))):
        cols[j].append(w)
    return compress_columns(C, cols)[cut + 1 - g:top - g]


@cache
def _log_tables() -> dict:
    """The log's tables, each range the circuit relies on checked as they are built: per stage the reciprocals k (R = k
    2^-s), the table c >= 0 the stage multiplies by and its constant p; L1, L2, L3 = -ln R and ln 2 at 2^-LF (nearest); and
    the polynomial's F(h) = -H^2/2 + H^3/3 at 2^-LF (nearest), H = (h - 128) 2^-29."""
    def ln_scaled(num: int, den: int) -> int:
        q = _DEC.ln(_DEC.divide(decimal.Decimal(num), decimal.Decimal(den)))
        return int(_DEC.multiply(q, decimal.Decimal(2) ** LF).to_integral_value(rounding=decimal.ROUND_HALF_EVEN))

    # stage 1: m = 1 + t 2^-8 + m1 2^-23, R1 = k1 / 2^9, U1 = (m R1 - 1 + 2^-8) 2^32 in [0, 2^25)
    k1 = [512 if t == 0 else 256 if t == 255 else round(Fraction(1 << 18, 513 + 2 * t)) for t in range(256)]
    for t, k in enumerate(k1):
        assert (1 << 32) - (1 << 24) <= ((1 << 23) + (t << 15)) * k and ((1 << 23) + ((t + 1) << 15) - 1) * k < (1 << 32) + (1 << 24)
    # stage 2: u1 = U1 2^-32 - 2^-8, index b = floor(u1 2^15), R2 = k2 / 2^16, U2 = (u2 + 2^-15) 2^48 in [0, 2^34)
    k2 = [1 << 16 if b in (0, -1) else round(Fraction(1 << 32, (1 << 16) + 2 * b + 1)) for b in range(-128, 128)]
    for bp, k in enumerate(k2):
        for u1 in (bp << 17, ((bp + 1) << 17) - 1):
            assert 0 <= ((1 << 32) - (1 << 24) + u1) * k - (1 << 48) + (1 << 33) < 1 << 34
    # stage 3: u2 = U2 2^-48 - 2^-15, index g = floor(u2 2^22), R3 = k3 / 2^23, V3 = (u3 + 2^-22) 2^71 in [0, 2^50 - 2^11)
    k3 = [1 << 23 if g in (0, -1) else round(Fraction(1 << 46, (1 << 23) + 2 * g + 1)) for g in range(-128, 128)]
    for gp, k in enumerate(k3):
        for u2 in (gp << 26, ((gp + 1) << 26) - 1):
            assert 0 <= ((1 << 48) - (1 << 33) + u2) * k - (1 << 71) + (1 << 49) < (1 << 50) - (1 << 11)
    return {"k1": k1, "T1": [(k * ((1 << 23) + (t << 15)) + (1 << 24)) % (1 << 25) for t, k in enumerate(k1)],
            "c2": [k - (1 << 16) + 256 for k in k2],
            "T2": [(((1 << 32) - (1 << 24) + (bp << 17)) * k - (1 << 48) + (1 << 33)) % (1 << 34) for bp, k in enumerate(k2)],
            "c3": [k - (1 << 23) + 256 for k in k3],
            "T3": [(((1 << 48) - (1 << 33) + (gp << 26)) * k - (1 << 71) + (1 << 49)) % (1 << 50) for gp, k in enumerate(k3)],
            "ln2": ln_scaled(2, 1), "L1": [ln_scaled(512, k) for k in k1], "L2": [ln_scaled(1 << 16, k) for k in k2],
            "L3": [ln_scaled(1 << 23, k) for k in k3],
            "F": [round(Fraction(-3 * (h * h << 28) + h ** 3, 3 << 10)) for h in range(-128, 128)]}


def log_rn(C, x):
    """`F32LogRn_v1` on bits: RN(ln(x)), the module's construction."""
    k = _log_tables()
    X = fp.unpack(C, x)
    padded = [ZERO] * 8 + X.sig
    _, lz = lzc(C, padded)
    M = C.commit(fp.shl(C, padded, lz)[8:31])                          # x = (1 + M 2^-23) 2^E for every x != 0
    Eu = C.commit(add(C, X.e_eff, [NOT(b) for b in lz], width=9))      # E + 158
    mono1 = _monomials(C, M[15:])
    U1 = C.commit(_stage(C, mono1, k["k1"], M[:15], k["T1"], (), 0, 25))
    mono2 = _monomials(C, U1[17:])
    m2 = U1[:17]
    U2 = C.commit(_stage(C, mono2, k["c2"], m2, k["T2"], [(m2, 16, False), (m2, 8, True)], 0, 34))
    mono3 = _monomials(C, U2[26:])
    m3 = U2[:26]
    U3 = C.commit(_stage(C, mono3, k["c3"], m3, k["T3"], [(m3, 23, False), (m3, 8, True)], 11, 39))
    mono4 = _monomials(C, U3[31:])
    d = U3[13:31]                                                       # delta = u3 - H, at 2^-47
    # ln x at 2^-LF, 8 integer bits, two's complement
    S = _Sum(LF + 8)
    for kk, sgn in _naf(k["ln2"]):
        (S.add if sgn > 0 else S.sub)(Eu, kk)
    S.const -= 158 * k["ln2"]
    S.table(C, mono1, k["L1"])
    S.table(C, mono2, k["L2"])
    S.table(C, mono3, k["L3"])
    S.add(U3, 17)
    S.const -= 1 << 55
    pm, consts = _pm_rows(C, mono4, [256 - h for h in range(256)], d)  # -2 H delta = 2 (256 - h) delta - 256 delta
    for bits, sh in pm:
        S.add(bits, sh)
    S.sub(d, 8)
    S.table(C, mono4, [f + q for f, q in zip(k["F"], consts)])
    for i in range(18):                                                 # -delta^2 / 2, its columns from 2^-LF
        if 2 * i >= 18:
            S.sub([d[i]], 2 * i - 18)
        for j in range(max(i + 1, 17 - i), 18):
            S.sub([C.AND(d[i], d[j])], i + j - 17)
    T = S.total(C)
    sign = T[-1]
    V = [b ^ sign for b in T[:-1]]                                      # |ln x|, ones' complement
    az, lz2 = lzc(C, V[LF - 25:LF + 7])                                 # its top bit at 2^(6 - lz2)
    y = V[LF - 48:LF + 7]
    for kk in reversed(range(5)):
        sh = 1 << kk
        y = [w ^ C.AND(lz2[kk], w ^ y[j - sh]) if j >= sh else C.AND(NOT(lz2[kk]), w) for j, w in enumerate(y)]
    ex = [C.AND(NOT(az), b) for b in add(C, const_bits(102, 8), [NOT(b) for b in lz2], width=8)]   # 133 - lz2
    word = add(C, y[31:54] + ex, [y[30]], width=31) + [sign]
    # specials
    nan = OR(C, X.nan, C.AND(X.s, NOT(X.zero)))
    sp = OR(C, nan, OR(C, X.zero, X.inf))
    keep = NOT(sp)
    return ([C.AND(keep, b) for b in word[:22]] + [nan ^ C.AND(keep, word[22])] + [sp ^ C.AND(keep, b) for b in word[23:31]]
            + [X.zero ^ C.AND(keep, word[31])])


F32LogRn = definition("F32LogRn", 2, log_rn, (("x", F32),), F32, word=R.F32LogRn,
                      doc="F32LogRn_v1 on bits: RN(ln(x)), subnormal x, NaN and x < 0 0x7FC00000, ±0 -inf, +inf +inf")
