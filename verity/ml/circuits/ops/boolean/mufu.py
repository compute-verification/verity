"""The MUFU operations as Boolean Definitions: the next version of each word primitive (`verity.ml.circuits.ops.mufu`), the same function on
bits, with every table read a plain And/Xor/Not circuit.

    MufuEx2Ftz_v2       (x: Array<32, Value<1>>) -> Array<32, Value<1>>              word view: MufuEx2Ftz_v1
    MufuRcpFtz_v2       (x: Array<32, Value<1>>) -> Array<32, Value<1>>              word view: MufuRcpFtz_v1
    MufuSqrtFtz_v2      (x: Array<32, Value<1>>) -> Array<32, Value<1>>              word view: MufuSqrtFtz_v1
    RsqrtApprox_v2      (x: Array<32, Value<1>>) -> Array<32, Value<1>>              word view: RsqrtApprox_v1
    DivFullRcp_v2       (b: Array<32, Value<1>>) -> Array<32, Value<1>>              word view: DivFullRcp_v1
    DivFullScaleA_v3    (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>           word view: DivFullScaleA_v1
    MufuTanh_v2         (x: Array<32, Value<1>>) -> Array<32, Value<1>>              word view: MufuTanh_v1

The exponent, sign and special-encoding rules around the tables are C-Flock's (`verity_flock.tail_pieces`, from
``fa2_model.cpp`` and ``rms_triton_model.cpp``). The range scalings by 0.25 and 2^24 are not multiplies: the exponent moves,
a subnormal is normalized by its leading-zero count (`_times_2_24`), and only `DivFullScaleA_v3` (which reads no table) rounds,
where a * 0.25 is subnormal.

A table read (`table_word`) is the special-function unit's own interpolator, exact on every index: a segment g of the index's
top bits and the offset u below them give ``base + ((C0[g] + C1[g] u + (C2[g] Q(u)) << square_shift) >> shift)``, Q(u) the
hardware's truncated square (`Fit.square`), with the integer coefficients of `tables/mufu_quadratic.json` (rcp 128 segments of
2^16 words, ex2 64 of 2^17, rsq and sqrt 64 of 2^17 per exponent parity). Every word but one is that interpolant: rsq's first
(1/sqrt(1) = 1.0) is one above it, an override (`overrides`, computed from the measured words when the circuit is built, so a
table the coefficients do not reproduce fails the build).

The circuit computes 2v (v the sum above) as one carry-save sum (`compress_columns`) whose bits ``shift + 1 ..`` are the word's
mantissa, and no product in it costs an AND: each coefficient c is written as ``2c + 1 = sum_k (2 b_k - 1) 2^k``, every digit
±1, so ``2 c x = sum_k (x XOR n_k) 2^k + ~x - (2^n - 1)(N + 1)`` for the n-bit operand x (u, or Q(u) at ``square_shift``),
with n_k = 1 - b_k and N = sum_k n_k 2^k. A row ``x XOR n_k`` is free; the constant is folded into the segment's C0. Each
n_k and each bit of C0 is a function of the segment bits g, written as the XOR of the monomials of its algebraic normal form
(the 2^|g| products of subsets of g: one AND each), with the sums the functions share computed once (`_xor_plan`). So the
ANDs are the monomials, the squarer's kept pairs, the sum's full adders, and the overrides' index matches.

MUFU.TANH (`MufuTanh_v2`) is the same interpolator (ex2's square, from column 19 at square_shift 2) over the eleven binades
of |x| in [2^-8, 8), with 51 segments of 2^17 offsets (each binade's top 0 to 4 mantissa bits; the offset is the 17 bits below
them, so a binade of fewer segments ignores its low bits) and the coefficients of `tables/mufu_tanh_quadratic.json`, at shift
16. A segment's word has its own lower exponent E: ``((E - 1) << 23) + v`` below v = 2^24, else ``(E << 23) + (v >> 1)`` (seven
segments straddle a power of two). Every word of the binades in the measured table is that interpolant, checked when the
circuit is built (`_tanh_measured`); outside them the primitive's rules hold (x, ±1, NaN). The segment's functions are XORs of
51 monomials of the binade's 4-bit index and its segment bits, one per segment (`_tanh_plan`); the offset is shifted to the
bottom by a 3-stage log shifter.
"""
from __future__ import annotations

import json
from functools import cache
from importlib import resources

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.types import Array

from .. import mufu as M
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce, compress_columns, const_bits, lzc, mux, or_reduce, shl_window, sub
from .softmax import _eq_const
from verity.ml.circuits.boolean.trace import definition

__all__ = ["QUADRATIC", "TANH_QUADRATIC", "DivFullRcp", "DivFullScaleA", "Fit", "MufuEx2Ftz", "MufuRcpFtz", "MufuSqrtFtz",
           "MufuTanh", "RsqrtApprox", "TanhFit", "fit", "overrides", "table_word", "tanh_fit", "tanh_table"]

F32 = Array(32, BL.BIT)
QUADRATIC = "mufu_quadratic.json"
TANH_QUADRATIC = "mufu_tanh_quadratic.json"
QNAN_MUFU = 0x7FFFFFFF
INF = 0x7F800000
ONE_F = 0x3F800000


# --- the table encoding ------------------------------------------------------------------------------------------------------

def _square(u: np.ndarray, nb: int, p: int) -> np.ndarray:
    u = np.asarray(u, dtype=np.int64)
    bit = [(u >> a) & 1 for a in range(nb)]
    q = np.zeros_like(u)
    for a in range(nb):
        if 2 * a >= p:
            q += bit[a] << (2 * a - p)
        for b in range(max(a + 1, p - a - 1), nb):
            q += (bit[a] & bit[b]) << (a + b + 1 - p)
    return q


class Fit:
    """One table's interpolator: ``base``, the index's ``bits``, the offset's ``nb`` (the segment is the rest), the square's
    first kept column ``p``, ``square_shift``, ``shift`` and the coefficients ``C0``, ``C1``, ``C2`` per segment."""

    def __init__(self, name: str, entry: dict) -> None:
        self.name = name
        self.base = int(entry["base"], 16)
        self.bits = int(entry["index_bits"])
        self.nb = int(entry["offset_bits"])
        self.p = int(entry["square_from"])
        self.square_shift = int(entry["square_shift"])
        self.shift = int(entry["shift"])
        self.C0, self.C1, self.C2 = (list(map(int, entry[k])) for k in ("C0", "C1", "C2"))
        if not len(self.C0) == len(self.C1) == len(self.C2) == 1 << (self.bits - self.nb):
            raise ValueError(f"MUFU fit {name}: {len(self.C0)} segments for {self.bits - self.nb} segment bits")
        if self.base & 0x807FFFFF or not self.nb < self.p < 2 * self.nb:
            raise ValueError(f"MUFU fit {name}: base {self.base:#x}, square from {self.p} of {2 * self.nb} columns")

    def square(self, u: np.ndarray) -> np.ndarray:
        """Q(u): of u^2's folded partial products (u_a at column 2a, u_a u_b at a + b + 1 for a < b), the sum of those in
        columns >= p, shifted right by p (int64)."""
        return _square(u, self.nb, self.p)

    def interpolant(self, idx: np.ndarray) -> np.ndarray:
        """``base + ((C0 + C1 u + (C2 Q(u)) << square_shift) >> shift)`` at each index (int64)."""
        idx = np.asarray(idx, dtype=np.int64)
        u, g = idx & ((1 << self.nb) - 1), idx >> self.nb
        c0, c1, c2 = (np.asarray(c, np.int64)[g] for c in (self.C0, self.C1, self.C2))
        q = self.square(np.arange(1 << self.nb))[u]
        return self.base + ((c0 + c1 * u + ((c2 * q) << self.square_shift)) >> self.shift)


@cache
def _fits() -> dict:
    data = json.loads((resources.files("verity.ml.circuits.ops") / "tables" / QUADRATIC).read_text())
    if data.get("format") != "verity/mufu-quadratic/v1":
        raise ValueError(f"{QUADRATIC}: unknown format {data.get('format')!r}")
    return {name: Fit(name, e) for name, e in data["tables"].items()}


def fit(name: str) -> Fit:
    return _fits()[name]


#: the most words a table may differ from its interpolant at (each one an index match in the circuit)
MAX_OVERRIDES = 4


@cache
def _measured(name: str) -> tuple[tuple[tuple[int, int], ...], bool]:
    f = fit(name)
    words = np.asarray(M.table(name), dtype=np.int64)
    if len(words) != 1 << f.bits:
        raise ValueError(f"MUFU table {name}: {len(words)} words, the fit indexes 2^{f.bits}")
    y = words - f.base
    if y.min() < 0 or y.max() > 1 << 23:
        raise ValueError(f"MUFU table {name}: a word outside [base, base + 2^23]")
    q = f.interpolant(np.arange(len(words), dtype=np.int64)) - f.base
    if q.min() < 0 or q.max() > 1 << 23:
        raise ValueError(f"MUFU table {name}: the interpolant leaves [base, base + 2^23]")
    off = np.flatnonzero(y != q)
    if len(off) > MAX_OVERRIDES:
        raise ValueError(f"MUFU table {name}: {len(off)} words differ from the interpolant (at most {MAX_OVERRIDES})")
    return tuple((int(i), int(y[i] - q[i])) for i in off), bool(max(y.max(), q.max()) == 1 << 23)


def overrides(name: str) -> tuple[tuple[int, int], ...]:
    """The (index, table word - interpolant) pairs of the table ``name`` where the two differ."""
    return _measured(name)[0]


def _top(name: str) -> bool:
    """Whether the table or its interpolant reaches base + 2^23 (the mantissa wraps to 0 and base's even exponent gains 1)."""
    return _measured(name)[1]


def _xor_all(terms):
    """The XOR of ``terms`` as a balanced tree of the forms' own ``^`` (so a traced circuit writes exactly these XORs)."""
    terms = list(terms)
    if not terms:
        return ZERO
    while len(terms) > 1:
        terms = [terms[i] ^ terms[i + 1] for i in range(0, len(terms) - 1, 2)] + terms[len(terms) - len(terms) % 2:]
    return terms[0]


def _monomials(C, bits):
    """The 2^n products of subsets of ``bits`` (index i = the subset whose mask is i), one AND each beyond the bits."""
    m = [ONE]
    for b in bits:
        m += [C.AND(x, b) for x in m]
    return m


def _digits(cs) -> tuple[int, list[int]]:
    """``2c + 1 = sum_k (2 b_k - 1) 2^k`` (k < K) for each c of ``cs``: K and each c's ``B = sum_k b_k 2^k``."""
    K = max(abs(2 * c + 1).bit_length() for c in cs)
    return K, [c + (1 << (K - 1)) for c in cs]


@cache
def _segment_functions(name: str) -> tuple[tuple[int, int], np.ndarray]:
    """The functions of the segment the table read uses, one row of 0/1 per segment each: the digit signs n_k of C1 (K1 of
    them), then of C2 (K2), then the W bits of C0 with the digits' constants folded in; and (K1, K2)."""
    f = fit(name)
    W = f.shift + 24 + _top(name)
    c0, rows, Ks = [2 * c for c in f.C0], [], []
    for n, cs, at in ((f.nb, f.C1, 0), (2 * f.nb - f.p, f.C2, f.square_shift)):
        K, B = _digits(cs)
        Ks.append(K)
        rows += [[1 - ((b >> k) & 1) for b in B] for k in range(K)]
        c0 = [c - (((1 << n) - 1) * ((((1 << K) - 1) ^ b) + 1) << at) for c, b in zip(c0, B)]
    rows += [[(c >> j) & 1 for c in c0] for j in range(W)]
    return (Ks[0], Ks[1]), np.array(rows, dtype=np.uint8)


@cache
def _xor_plan(name: str) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, ...], ...]]:
    """The segment functions as XORs of the monomials (`_monomials`) of their algebraic normal forms, with shared sums: terms
    0 .. 2^n - 1 are the monomials, and each pair (a, b) appends the term ``a XOR b``; then each function's terms. The pairs
    are chosen greedily: the first pair of terms that the most functions share, while some two share one."""
    _, rows = _segment_functions(name)
    anf = rows.copy()
    for i in range(fit(name).bits - fit(name).nb):
        anf = anf.reshape(len(rows), -1, 2, 1 << i)
        anf[:, :, 1, :] ^= anf[:, :, 0, :]
    return _shared_sums(anf.reshape(len(rows), -1))


def _shared_sums(terms: np.ndarray) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, ...], ...]]:
    """Functions as XORs of terms (``terms``: a 0/1 row per function, a column per term, column 0 the constant ONE), with the
    sums several share computed once: each pair (a, b) appends the term ``a XOR b``; then each function's terms. Greedy: the
    first pair of terms that the most functions share, while some two share one (the constant is never paired)."""
    M = terms.astype(np.float32)
    const = M[:, 0].copy()
    M[:, 0] = 0
    pairs = []
    while True:
        G = np.triu(M.T @ M, 1)
        a, b = divmod(int(np.argmax(G)), G.shape[1])
        if G[a, b] < 2:
            break
        both = (M[:, a] == 1) & (M[:, b] == 1)
        M[both, a] = M[both, b] = 0
        M = np.concatenate([M, both[:, None].astype(np.float32)], axis=1)
        pairs.append((a, b))
    M[:, 0] = const
    return tuple(pairs), tuple(tuple(np.flatnonzero(r).tolist()) for r in M)


def truncated_square(C, u, p: int):
    """`Fit.square` on bits: the folded partial products in columns >= p (an AND per pair), summed, shifted right by p."""
    n = len(u)
    cols: list[list] = [[] for _ in range(2 * n - p)]
    for a in range(n):
        if 2 * a >= p:
            cols[2 * a - p].append(u[a])
        for b in range(max(a + 1, p - a - 1), n):
            cols[a + b + 1 - p].append(C.AND(u[a], u[b]))
    return compress_columns(C, cols)[:2 * n - p]


def table_word(C, name: str, idx):
    """The measured table ``name``'s word at ``idx`` (its low 31 bits, LSB first; bit 31 of every MUFU table word is 0)."""
    f = fit(name)
    assert len(idx) == f.bits
    u, top = list(idx[:f.nb]), _top(name)
    W = f.shift + 24 + top                     # the bits of 2 v: the word below base is < 2^(23 + top)
    terms = _monomials(C, idx[f.nb:])
    pairs, sums = _xor_plan(name)
    for a, b in pairs:
        terms.append(terms[a] ^ terms[b])
    fns = [_xor_all(terms[i] for i in s) for s in sums]
    (K1, K2), _ = _segment_functions(name)
    cols: list[list] = [[] for _ in range(W)]
    # 2 c x = sum_k (x XOR n_k) 2^k + ~x - (2^len(x) - 1)(N + 1), n_k = 1 - b_k the digit's sign, N = sum_k n_k 2^k
    for x, n, at in ((u, fns[:K1], 0), (truncated_square(C, u, f.p), fns[K1:K1 + K2], f.square_shift)):
        for k, n_k in enumerate(n):
            for j, b in enumerate(x):
                if j + k + at < W:
                    cols[j + k + at].append(b ^ n_k)
        for j, b in enumerate(x):
            if j + at < W:
                cols[j + at].append(NOT(b))
    for j, b in enumerate(fns[K1 + K2:]):
        cols[j].append(b)
    for i, d in overrides(name):
        hit = and_reduce(C, [b if (i >> k) & 1 else NOT(b) for k, b in enumerate(idx)])
        for k in range(W - f.shift - 1):
            if (d % (1 << (W - f.shift - 1))) >> k & 1:
                cols[f.shift + 1 + k].append(hit)
    y = compress_columns(C, cols)[f.shift + 1:W]
    # base's mantissa is 0 and y <= 2^23: below 2^23 the exponent is base's; at 2^23 (mantissa 0) base's even exponent + 1
    exp = const_bits(f.base >> 23, 8)
    if top:
        assert not (f.base >> 23) & 1
        exp = [y[23]] + exp[1:]
    return y[:23] + exp


# --- the bit rules (C-Flock's `tail_pieces`, the table read above) ---------------------------------------------------------

def _ult_const(C, w, k: int):
    """``w < k`` for an unsigned word and a constant."""
    return sub(C, w, const_bits(k, len(w)), len(w))[1]


def _fields(C, x):
    s, e, m = x[31], list(x[23:31]), list(x[:23])
    return s, e, m, _eq_const(C, e, 0xFF), NOT(or_reduce(C, e)), or_reduce(C, m)


def _word(C, cases, default):
    """``cases`` = [(cond, word)] tried in order, else ``default``."""
    out = list(default)
    for cond, w in reversed(cases):
        out = mux(C, cond, w, out)
    return out


def mufu_ex2(C, x):
    """``MufuEx2Ftz_v1``: x as the signed fixed-point value floor(x 2^23); the table at its fraction, the integer part n added
    to the table word's exponent. Only |x| < 128 has a finite nonzero 2^x, so K = floor(|x| 2^23) is kept mod 2^30 (one
    5-bit shift of the significand; e - 127 < -24 gives K = 0) and |x| >= 128 saturates. For x < 0 the fixed-point value is
    -K or ~K (= -K - 1, when K has fraction bits), so its fraction is K's XOR (x < 0 and K has fraction bits) and n is
    ``s ? ~int(K) + (K has none) : int(K)``."""
    s, e, m, e_max, e_zero, m_nz = _fields(C, x)
    sat = NOT(_ult_const(C, e, 134))                          # e - 127 >= 7
    live = NOT(_ult_const(C, e, 103))                         # e - 127 >= -24
    sig = [C.AND(live, b) for b in m] + [live]
    K = shl_window(C, sig, sub(C, e, const_bits(103, 8), 8)[0][:5], 24, 54)
    z = NOT(or_reduce(C, K[:23]))
    t = table_word(C, "ex2", [k ^ C.AND(s, NOT(z)) for k in K[:23]])
    re = add(C, list(t[23:31]) + [ZERO, ZERO], [k ^ s for k in K[23:30]] + [s] * 3, C.AND(s, z), width=10)
    zero = OR(C, re[9], NOT(or_reduce(C, re)))
    inf_or_zero = mux(C, s, const_bits(0, 32), const_bits(INF, 32))
    return _word(C, [(e_max, mux(C, m_nz, const_bits(QNAN_MUFU, 32), inf_or_zero)), (e_zero, const_bits(ONE_F, 32)),
                     (sat, inf_or_zero), (zero, const_bits(0, 32))], list(t[:23]) + re[:8] + [ZERO])


def rcp_post(C, xb, t):
    """MUFU.RCP of xb from the rcp table's word t at xb's mantissa: the exponent negated exactly, the specials."""
    s, e, _, e_max, e_zero, m_nz = _fields(C, xb)
    re, _ = sub(C, list(t[23:31]) + [ZERO, ZERO], sub(C, e + [ZERO, ZERO], const_bits(127, 10), 10)[0], 10)
    neg = re[9]
    zero = OR(C, neg, NOT(or_reduce(C, re)))
    big = C.AND(NOT(neg), OR(C, re[8], _eq_const(C, re[:8], 0xFF)))
    body = list(t[:23]) + re[:8] + [s]
    return _word(C, [(e_max, mux(C, m_nz, const_bits(QNAN_MUFU, 32), [ZERO] * 31 + [s])), (e_zero, const_bits(INF, 31) + [s]),
                     (zero, [ZERO] * 31 + [s]), (big, const_bits(INF, 31) + [s])], body)


def mufu_rcp(C, x):
    """``MufuRcpFtz_v1``: the rcp table at the mantissa, the exponent handled exactly."""
    return rcp_post(C, x, table_word(C, "rcp", x[:23]))


def _times_2_24(C, x):
    """``x * 2^24`` for ``|x| < 2^-126`` (exponent field 0), exact: the normal number ``m 2^-125`` or the zero x."""
    v = [ZERO] * 9 + list(x[:23])
    az, lz = lzc(C, v)
    e = sub(C, const_bits(24, 8), lz + [ZERO] * 3, 8)[0]
    return shl_window(C, v, lz, 8, 31) + [C.AND(NOT(az), b) for b in e] + [x[31]]


def _quarter_rn(C, a):
    """``RN(a * 0.25)`` as numpy float32 computes it: exponent - 2, a subnormal result rounded to nearest even, a NaN quieted."""
    s, e, m, e_max, e_zero, m_nz = _fields(C, a)
    low = _ult_const(C, e, 3)                                  # e <= 2: the significand shifts right 1 (e = 2) or 2 bits
    sig = list(m) + [NOT(e_zero)]
    one = C.AND(e[1], NOT(e[0]))
    q = mux(C, one, sig[1:] + [ZERO], sig[2:] + [ZERO, ZERO])
    guard = mux(C, one, [sig[0]], [sig[1]])[0]
    up = C.AND(guard, OR(C, C.AND(NOT(one), sig[0]), q[0]))
    body = mux(C, low, add(C, q, [up], width=31), list(m) + sub(C, e, const_bits(2, 8), 8)[0])
    return mux(C, e_max, list(a[:22]) + [OR(C, a[22], m_nz)] + list(e), body) + [s]


def _times_2_24_rn(C, a):
    """``RN(a * 2^24)`` as numpy float32 computes it: exponent + 24 or infinity, a subnormal normalized, a NaN quieted."""
    s, e, m, e_max, e_zero, m_nz = _fields(C, a)
    normal = mux(C, NOT(_ult_const(C, e, 255 - 24)), const_bits(INF, 31), list(m) + add(C, e, const_bits(24, 8), width=8))
    special = list(a[:22]) + [OR(C, a[22], m_nz)] + list(e)
    return mux(C, e_max, special, mux(C, e_zero, _times_2_24(C, a)[:31], normal)) + [s]


def _parity_index(xb):
    """``(par << 23) | mantissa``, par = (e - 127) & 1 = NOT e[0]: the rsq / sqrt tables' index."""
    return list(xb[:23]) + [NOT(xb[23])]


def _rsq_post(C, xb, v):
    s, e, _, e_max, e_zero, m_nz = _fields(C, xb)
    k = sub(C, e + [ZERO], const_bits(127, 9), 9)[0]
    re = sub(C, list(v[23:31]) + [ZERO], k[1:] + [k[8]], 9)[0]
    return _word(C, [(e_max, mux(C, m_nz, const_bits(QNAN_MUFU, 32), mux(C, s, const_bits(QNAN_MUFU, 32), const_bits(0, 32)))),
                     (e_zero, const_bits(INF, 31) + [s]), (s, const_bits(QNAN_MUFU, 32))], list(v[:23]) + re)


def rsqrt_approx(C, x):
    """``RsqrtApprox_v1``: MUFU.RSQ with the subnormal range scaling (x * 2^24, the result * 2^12 unless it is inf or NaN)."""
    small = _fields(C, x)[4]
    xs = mux(C, small, _times_2_24(C, x), list(x))
    r = _rsq_post(C, xs, table_word(C, "rsq", _parity_index(xs)))
    up = C.AND(small, NOT(_eq_const(C, r[23:31], 0xFF)))
    return mux(C, up, list(r[:23]) + add(C, r[23:31], const_bits(12, 8), width=8) + [r[31]], list(r))


def mufu_sqrt_ftz(C, x):
    """``MufuSqrtFtz_v1``: MUFU.SQRT (``sqrt.approx.ftz.f32``)."""
    s, e, _, e_max, e_zero, m_nz = _fields(C, x)
    v = table_word(C, "sqrt", _parity_index(x))
    k = sub(C, e + [ZERO], const_bits(127, 9), 9)[0]
    re = add(C, list(v[23:31]) + [ZERO], k[1:] + [k[8]], width=9)
    return _word(C, [(e_max, mux(C, OR(C, m_nz, s), const_bits(QNAN_MUFU, 32), const_bits(INF, 32))),
                     (e_zero, [ZERO] * 31 + [s]), (s, const_bits(QNAN_MUFU, 32))], list(v[:23]) + re)


def div_full_rcp(C, b):
    """``DivFullRcp_v1``: ``div.full.f32``'s reciprocal part: MUFU.RCP of b * 2^24 when ``|b| < 2^-126``, of b * 0.25 (exponent
    - 2) when ``2^126 < |b| < inf``, else of b."""
    _, e, _, e_max, e_zero, m_nz = _fields(C, b)
    big = OR(C, C.AND(NOT(e_max), _eq_const(C, e[1:], 0x7F)), C.AND(_eq_const(C, e, 253), m_nz))
    quarter = list(b[:23]) + sub(C, e, const_bits(2, 8), 8)[0] + [b[31]]
    d = mux(C, e_zero, _times_2_24(C, b), mux(C, big, quarter, list(b)))
    return rcp_post(C, d, table_word(C, "rcp", d[:23]))


def div_full_scale_a(C, a, b):
    """``DivFullScaleA_v1``: ``a * 0.25`` when ``|b| > 2^126``, ``a * 2^24`` when ``|b| < 2^-126``, else a (NaN b: a)."""
    mag = b[:31]
    big = C.AND(NOT(_ult_const(C, mag, 0x7E800001)), _ult_const(C, mag, 0x7F800001))
    small = _ult_const(C, mag, 0x00800000)
    return mux(C, big, _quarter_rn(C, a), mux(C, small, _times_2_24_rn(C, a), list(a)))


# --- MUFU.TANH -------------------------------------------------------------------------------------------------------------------

class TanhFit:
    """MUFU.TANH's interpolator at |x| in [2^-8, 8) (`tables/mufu_tanh_quadratic.json`): ``binades`` [(exponent, segment bits)]
    in order, each segment's lower output exponent ``E`` and coefficients ``C0``, ``C1``, ``C2`` in order, and ``nb``, ``p``,
    ``square_shift``, ``shift`` as in `Fit`."""

    def __init__(self, entry: dict) -> None:
        self.nb, self.p = int(entry["offset_bits"]), int(entry["square_from"])
        self.square_shift, self.shift = int(entry["square_shift"]), int(entry["shift"])
        self.binades = [(int(e), int(sb)) for e, sb in entry["binades"]]
        self.E, self.C0, self.C1, self.C2 = (list(map(int, entry[k])) for k in ("E", "C0", "C1", "C2"))
        es = [e for e, _ in self.binades]
        self.sb_max = max(sb for _, sb in self.binades)
        if es != list(range(es[0], es[0] + len(es))) or len(es) > 16 or self.sb_max + self.nb > 23:
            raise ValueError(f"{TANH_QUADRATIC}: binades {self.binades} with {self.nb} offset bits")
        self.first, n = {}, 0
        for e, sb in self.binades:
            self.first[e], n = n, n + (1 << sb)
        if not len(self.E) == len(self.C0) == len(self.C1) == len(self.C2) == n or not self.nb < self.p < 2 * self.nb:
            raise ValueError(f"{TANH_QUADRATIC}: {len(self.C0)} segments for {n}, square from {self.p} of {2 * self.nb} columns")

    def values(self, e: int, m) -> tuple[np.ndarray, np.ndarray]:
        """(E, V) at exponent ``e`` and each mantissa of ``m``: V = (C0 + C1 u + (C2 Q(u)) << square_shift) >> shift for the
        segment g (m's top bits) and offset u (the nb bits below), E the segment's (int64)."""
        sb = dict(self.binades)[e]
        m = np.asarray(m, dtype=np.int64)
        g = self.first[e] + (m >> (23 - sb))
        u = (m >> (23 - sb - self.nb)) & ((1 << self.nb) - 1)
        c0, c1, c2, E = (np.asarray(c, np.int64)[g] for c in (self.C0, self.C1, self.C2, self.E))
        q = _square(np.arange(1 << self.nb), self.nb, self.p)[u]
        return E, (c0 + c1 * u + ((c2 * q) << self.square_shift)) >> self.shift

    def words(self, e: int, m) -> np.ndarray:
        """The interpolant's word at exponent ``e`` and each mantissa of ``m``: ((E - 1) << 23) + V when V < 2^24, else
        (E << 23) + (V >> 1) (`values`; positive words, int64)."""
        return _tanh_word(*self.values(e, m))


def _tanh_word(E: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.where(v < 1 << 24, ((E - 1) << 23) + v, (E << 23) + (v >> 1))


@cache
def tanh_fit() -> TanhFit:
    data = json.loads((resources.files("verity.ml.circuits.ops") / "tables" / TANH_QUADRATIC).read_text())
    if data.get("format") != "verity/mufu-tanh-quadratic/v1":
        raise ValueError(f"{TANH_QUADRATIC}: unknown format {data.get('format')!r}")
    return TanhFit(data)


@cache
def _tanh_measured() -> None:
    """`tanh_fit` is `MufuTanh_v1`, or ValueError: the primitive's rules are x up to the fit's binades (whose first word the
    table maps to itself), ±1 from their end and 0x7FFFFFFF for NaN; every positive word of the binades in its measured table
    is the interpolant; and every V is in [2^23, 2^25) with E + (V >> 24) < 2^7 (`tanh_table`'s exponent)."""
    f, R = tanh_fit(), M.MUFU_TANH_RULES
    lo, hi = f.binades[0][0] << 23, (f.binades[-1][0] + 1) << 23
    if (R["identity_below"], R["saturate_from"], R["nan_out"]) != (lo + 1, hi, QNAN_MUFU):
        raise ValueError(f"{TANH_QUADRATIC}: MufuTanh_v1's rules {R} do not close the binades [{lo:#x}, {hi:#x})")
    (s,) = M.mufu_tanh_shards()
    bw = s.block_words
    if not (s.lo <= lo and hi <= s.hi and s.sym_ok and (lo - s.lo) % bw == 0 and (1 << 23) % bw == 0):
        raise ValueError(f"{TANH_QUADRATIC}: MufuTanh_v1's table [{s.lo:#x}, {s.hi:#x}) does not hold the binades by block")
    if s.lookup(lo) != lo:
        raise ValueError(f"{TANH_QUADRATIC}: MufuTanh_v1's table at {lo:#x} is {s.lookup(lo):#x}, its identity rule {lo:#x}")
    for e, _ in f.binades:
        k = ((e << 23) - s.lo) // bw
        words = np.concatenate([s.block(k + i) for i in range((1 << 23) // bw)]).astype(np.int64)
        E, v = f.values(e, np.arange(1 << 23))
        if v.min() < 1 << 23 or v.max() >= 1 << 25 or (E + (v >> 24)).max() >= 1 << 7:
            raise ValueError(f"{TANH_QUADRATIC}: binade {e}'s V in [{v.min():#x}, {v.max():#x}], E up to {E.max()}")
        bad = int(np.count_nonzero(words != _tanh_word(E, v)))
        if bad:
            raise ValueError(f"{TANH_QUADRATIC}: {bad} words of binade {e} are not MufuTanh_v1's")


def _tanh_points(f: TanhFit) -> np.ndarray:
    """Each segment's select variables (a row of 0/1): t = e - the first binade's exponent (4 bits), then the mantissa's top
    sb_max bits from m22 down, each 0 in the binades that do not read it as a segment bit."""
    e0 = f.binades[0][0]
    return np.array([[((e - e0) >> k) & 1 for k in range(4)] + [(g >> (sb - 1 - i)) & 1 if i < sb else 0 for i in range(f.sb_max)]
                     for e, sb in f.binades for g in range(1 << sb)], dtype=np.uint8)


def _gf2_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """X with A X = B over GF(2), A square and invertible (0/1 arrays)."""
    n = len(A)
    X = np.concatenate([A, B], axis=1).astype(np.uint8)
    for c in range(n):
        r = c + int(np.flatnonzero(X[c:, c])[0])
        X[[c, r]] = X[[r, c]]
        hit = np.flatnonzero(X[:, c])
        X[hit[hit != c]] ^= X[c]
    return X[:, n:]


@cache
def _tanh_plan():
    """How `tanh_table` computes the functions of the segment: ``basis``, one monomial of the select variables (`_tanh_points`,
    as a bit mask) per segment, each taken in order of degree when it is independent of those before on the segments; ``steps``
    [(monomial, factor, variable)] build them (and any factor that is not one) with one AND each; the functions, each an XOR of
    basis monomials with shared sums (`_shared_sums`): the digit signs of C1 (K1 of them), of C2 (K2), then the W bits of C0 with
    the digits' constants and ``E << (shift + 25)`` folded in, then E's bit 0. Returns (basis, steps, pairs, sums, K1, K2, W)."""
    f = tanh_fit()
    P = _tanh_points(f)
    nv = P.shape[1]
    basis, vecs, reduced = [], [], {}
    for mask in sorted(range(1 << nv), key=lambda k: (k.bit_count(), k)):
        v = np.ones(len(P), dtype=np.uint8)
        for i in range(nv):
            if mask >> i & 1:
                v &= P[:, i]
        r = int("".join(map(str, v)), 2)
        for lead in sorted(reduced, reverse=True):
            if r >> lead & 1:
                r ^= reduced[lead]
        if r:
            reduced[r.bit_length() - 1] = r
            basis.append(mask)
            vecs.append(v)
        if len(basis) == len(P):
            break
    built, steps = set(range(1)) | {1 << i for i in range(nv)}, []

    def build(mask: int) -> None:
        if mask in built:
            return
        fac = next((mask ^ (1 << i) for i in reversed(range(nv)) if mask >> i & 1 and mask ^ (1 << i) in built), None)
        if fac is None:
            fac = mask ^ (1 << (mask.bit_length() - 1))
            build(fac)
        steps.append((mask, fac, (mask ^ fac).bit_length() - 1))
        built.add(mask)

    for mask in basis:
        build(mask)
    W = f.shift + 32
    c0, rows, Ks = [2 * c for c in f.C0], [], []
    for n, cs, at in ((f.nb, f.C1, 0), (2 * f.nb - f.p, f.C2, f.square_shift)):
        K, B = _digits(cs)
        Ks.append(K)
        rows += [[1 - ((b >> k) & 1) for b in B] for k in range(K)]
        c0 = [c - (((1 << n) - 1) * ((((1 << K) - 1) ^ b) + 1) << at) for c, b in zip(c0, B)]
    c0 = [c + (E << (f.shift + 25)) for c, E in zip(c0, f.E)]
    rows += [[(c >> j) & 1 for c in c0] for j in range(W)] + [[E & 1 for E in f.E]]
    terms = _gf2_solve(np.array(vecs, dtype=np.uint8).T, np.array(rows, dtype=np.uint8).T).T
    pairs, sums = _shared_sums(terms)
    return tuple(basis), tuple(steps), pairs, sums, Ks[0], Ks[1], W


def _of_t(t_terms: dict, h: list):
    """The function of t that is ``h[t]`` for t < len(h), as an XOR of t's monomials (Möbius over the masks below len(h), a
    down-set)."""
    out = ZERO
    for S in range(len(h)):
        if sum(h[T] for T in range(S + 1) if T & S == T) & 1:
            out = out ^ t_terms[S]
    return out


def tanh_table(C, e, m):
    """MUFU.TANH's interpolator (`TanhFit.words`) at the positive word of exponent bits ``e`` (one of the fit's binades) and
    mantissa ``m``: the word's low 31 bits. t = e - the first binade (mod 16) selects the segment with the mantissa's top bits;
    the offset u is the nb bits below them, shifted down by sb_max - sb (a log shifter); the sum is `table_word`'s with the
    output exponent folded into C0, and the mantissa is the sum's bits 0..22 or, when V >= 2^24, 1..23."""
    f = tanh_fit()
    basis, steps, pairs, sums, K1, K2, W = _tanh_plan()
    t = add(C, list(e[:4]), const_bits(-f.binades[0][0] % 16, 4), width=4)
    var, terms = list(t) + [None] * f.sb_max, {0: ONE}
    for i in range(4):
        terms[1 << i] = t[i]
    for mask, fac, v in steps:
        if mask < 16:
            terms[mask] = C.AND(terms[fac], var[v])
    sbs = [sb for _, sb in f.binades]
    for i in range(f.sb_max):
        var[4 + i] = terms[1 << (4 + i)] = C.AND(m[22 - i], _of_t(terms, [int(sb > i) for sb in sbs]))
    for mask, fac, v in steps:
        if mask >= 16:
            terms[mask] = C.AND(terms[fac], var[v])
    rs = [f.sb_max - sb for sb in sbs]
    cur = list(m[23 - f.sb_max - f.nb:])
    for k in reversed(range(max(rs).bit_length())):
        amt, ctrl, nxt = 1 << k, _of_t(terms, [r >> k & 1 for r in rs]), []
        for j in range(f.nb + max(r & (amt - 1) for r in rs)):
            read = [r >> k & 1 for r in rs if j < f.nb + (r & (amt - 1))]
            hi = cur[j + amt] if j + amt < len(cur) else ZERO
            nxt.append(hi if all(read) else cur[j] if not any(read) else mux(C, ctrl, [hi], [cur[j]])[0])
        cur = nxt
    u = cur[:f.nb]
    T = [terms[k] for k in basis]
    for a, b in pairs:
        T.append(T[a] ^ T[b])
    fns = [_xor_all(T[i] for i in s) for s in sums]
    cols: list[list] = [[] for _ in range(W)]
    for x, n, at in ((u, fns[:K1], 0), (truncated_square(C, u, f.p), fns[K1:K1 + K2], f.square_shift)):
        for k, n_k in enumerate(n):
            for j, b in enumerate(x):
                if j + k + at < W:
                    cols[j + k + at].append(b ^ n_k)
        for j, b in enumerate(x):
            if j + at < W:
                cols[j + at].append(NOT(b))
    for j, b in enumerate(fns[K1 + K2:K1 + K2 + W]):
        cols[j].append(b)
    s = compress_columns(C, cols)
    v, up = s[f.shift + 1:], s[f.shift + 25] ^ fns[-1]
    return mux(C, up, v[1:24], v[:23]) + list(s[f.shift + 25:W]) + [ZERO]


def mufu_tanh(C, x):
    """``MufuTanh_v1``: |x| <= 2^-8 is x, |x| >= 8 (the infinities too) is ±1, NaN is 0x7FFFFFFF, else MUFU.TANH's
    interpolator at |x| (`tanh_table`) with x's sign."""
    f = tanh_fit()
    _tanh_measured()
    s, e, m = x[31], list(x[23:31]), list(x[:23])
    ident = _ult_const(C, e, f.binades[0][0])
    special = NOT(_ult_const(C, e, f.binades[-1][0] + 1))
    nan = C.AND(_eq_const(C, e, 0xFF), or_reduce(C, m))
    z = mux(C, ident, list(x[:30]), tanh_table(C, e, m)[:30])
    # ±1 (0x3F800000) or 0x7FFFFFFF: the mantissa is NaN's, the exponent's low 7 bits are 1, bit 30 is NaN's, the sign x's
    return ([b ^ C.AND(special, b ^ nan) for b in z[:23]] + [OR(C, b, special) for b in z[23:30]]
            + [nan, C.AND(s, NOT(nan))])


# --- the Definitions -----------------------------------------------------------------------------------------------------------

_X = (("x", F32),)

MufuEx2Ftz = definition("MufuEx2Ftz", 2, mufu_ex2, _X, F32, word=M.MufuEx2Ftz,
                        doc="ex2.approx.ftz.f32 (MUFU.EX2) on bits: MufuEx2Ftz_v1, its table as the unit's quadratic interpolator")
MufuRcpFtz = definition("MufuRcpFtz", 2, mufu_rcp, _X, F32, word=M.MufuRcpFtz,
                        doc="rcp.approx.ftz.f32 (MUFU.RCP) on bits: MufuRcpFtz_v1, its table as the unit's quadratic interpolator")
MufuSqrtFtz = definition("MufuSqrtFtz", 2, mufu_sqrt_ftz, _X, F32, word=M.MufuSqrtFtz,
                         doc="sqrt.approx.ftz.f32 (MUFU.SQRT) on bits: MufuSqrtFtz_v1, its table as the unit's quadratic interpolator")
RsqrtApprox = definition("RsqrtApprox", 2, rsqrt_approx, _X, F32, word=M.RsqrtApprox,
                         doc="rsqrt.approx.f32 (MUFU.RSQ with the subnormal range scaling) on bits: RsqrtApprox_v1")
DivFullRcp = definition("DivFullRcp", 2, div_full_rcp, (("b", F32),), F32, word=M.DivFullRcp,
                        doc="div.full.f32's reciprocal part (range scaling, MUFU.RCP) on bits: DivFullRcp_v1")
DivFullScaleA = definition("DivFullScaleA", 3, div_full_scale_a, (("a", F32), ("b", F32)), F32, word=M.DivFullScaleA,
                           doc="div.full.f32's numerator range scaling on bits: DivFullScaleA_v1 (a * 0.25, a * 2^24 or a)")
MufuTanh = definition("MufuTanh", 2, mufu_tanh, _X, F32, word=M.MufuTanh,
                      doc="tanh.approx.f32 (MUFU.TANH, sm_89) on bits: MufuTanh_v1, its table as the unit's quadratic interpolator")
