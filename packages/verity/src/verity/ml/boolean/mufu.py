"""The MUFU operations as Boolean Definitions: the next version of each word primitive (`verity.ml.mufu`), the same function on
bits, with every table read a plain And/Xor/Not circuit.

    MufuEx2Ftz_v2       (x: Array<32, Value<1>>) -> Array<32, Value<1>>              word view: MufuEx2Ftz_v1
    MufuRcpFtz_v2       (x: Array<32, Value<1>>) -> Array<32, Value<1>>              word view: MufuRcpFtz_v1
    MufuSqrtFtz_v2      (x: Array<32, Value<1>>) -> Array<32, Value<1>>              word view: MufuSqrtFtz_v1
    RsqrtApprox_v2      (x: Array<32, Value<1>>) -> Array<32, Value<1>>              word view: RsqrtApprox_v1
    DivFullRcp_v2       (b: Array<32, Value<1>>) -> Array<32, Value<1>>              word view: DivFullRcp_v1
    DivFullScaleA_v3    (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>           word view: DivFullScaleA_v1

The bit rules around the tables are C-Flock's (`verity_flock.tail_pieces`: the exponent, sign and special-encoding logic of
``fa2_model.cpp`` and ``rms_triton_model.cpp``); `DivFullScaleA_v3` reads no table.

A table read (`table_word`) is the special-function unit's own interpolator, exact on every index: a segment g of the index's
top bits and the offset u below them give ``base + ((C0[g] + C1[g] u + (C2[g] Q(u)) << square_shift) >> shift)``, Q(u) the
hardware's truncated square (`Fit.square`), with the integer coefficients of `tables/mufu_quadratic.json` (rcp 128 segments of
2^16 words, ex2 64 of 2^17, rsq and sqrt 64 of 2^17 per exponent parity). Every word but one is that interpolant: rsq's first
(1/sqrt(1) = 1.0) is one above it, an override (`overrides`, computed from the measured words when the circuit is built, so a
table the coefficients do not reproduce fails the build). In the circuit the segment bits decode to minterms and each
coefficient bit is an XOR of them; the squarer's kept partial products, the two products and C0 are one carry-save sum
(`compress_columns`) whose bits ``shift ..`` are the word's mantissa. A negative coefficient column multiplies the complemented
operand (``-|c| u = |c| ~u - |c| (2^n - 1)``), the constant folded into C0, so every partial product is one AND.
"""
from __future__ import annotations

import json
from functools import cache
from importlib import resources

import numpy as np

from verity.ir import boolean as BL
from verity.ir.types import Array

from .. import mufu as M
from . import fp
from .forms import NOT, ONE, OR, ZERO, add, and_reduce, compress_columns, const_bits, mux, or_reduce, sub
from .softmax import _eq_const
from .trace import definition

__all__ = ["QUADRATIC", "DivFullRcp", "DivFullScaleA", "Fit", "MufuEx2Ftz", "MufuRcpFtz", "MufuSqrtFtz", "RsqrtApprox", "fit",
           "overrides", "table_word"]

F32 = Array(32, BL.BIT)
QUADRATIC = "mufu_quadratic.json"
QNAN_MUFU = 0x7FFFFFFF
INF = 0x7F800000
ONE_F = 0x3F800000
QUARTER = 0x3E800000   # 0.25
TWO24 = 0x4B800000     # 2^24


# --- the table encoding ------------------------------------------------------------------------------------------------------

class Fit:
    """One table's interpolator: ``base``, the index's ``bits``, the offset's ``nb`` (the segment is the rest), the square's
    first kept column ``p``, ``square_shift``, ``shift`` and the coefficients ``C0``, ``C1``, ``C2`` per segment (C1 and C2
    each of one sign, ``s1`` and ``s2``)."""

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
        self.s1, self.s2 = (-1 if v[0] < 0 else 1 for v in (self.C1, self.C2))
        if any((c < 0) != (self.s1 < 0) for c in self.C1) or any((c < 0) != (self.s2 < 0) for c in self.C2):
            raise ValueError(f"MUFU fit {name}: a coefficient column changes sign")
        self.wq = 2 * self.nb - self.p

    def square(self, u: np.ndarray) -> np.ndarray:
        """Q(u): of u^2's folded partial products (u_a at column 2a, u_a u_b at a + b + 1 for a < b), the sum of those in
        columns >= p, shifted right by p (int64)."""
        u = np.asarray(u, dtype=np.int64)
        bit = [(u >> a) & 1 for a in range(self.nb)]
        q = np.zeros_like(u)
        for a in range(self.nb):
            if 2 * a >= self.p:
                q += bit[a] << (2 * a - self.p)
            for b in range(max(a + 1, self.p - a - 1), self.nb):
                q += (bit[a] & bit[b]) << (a + b + 1 - self.p)
        return q

    def interpolant(self, idx: np.ndarray) -> np.ndarray:
        """``base + ((C0 + C1 u + (C2 Q(u)) << square_shift) >> shift)`` at each index (int64)."""
        idx = np.asarray(idx, dtype=np.int64)
        u, g = idx & ((1 << self.nb) - 1), idx >> self.nb
        c0, c1, c2 = (np.asarray(c, np.int64)[g] for c in (self.C0, self.C1, self.C2))
        return self.base + ((c0 + c1 * u + ((c2 * self.square(u)) << self.square_shift)) >> self.shift)


@cache
def _fits() -> dict:
    data = json.loads((resources.files("verity.ml") / "tables" / QUADRATIC).read_text())
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


def _select(m, values, width: int):
    """``values[i]`` (``width`` bits) at the one-hot minterms ``m``: each bit the XOR of the minterms of the values that have
    it, or the complement of the XOR of the others (the minterms sum to 1), whichever is shorter."""
    out = []
    for j in range(width):
        has = [(v >> j) & 1 for v in values]
        if 2 * sum(has) <= len(values):
            out.append(_xor_all(m[i] for i, h in enumerate(has) if h))
        else:
            out.append(NOT(_xor_all(m[i] for i, h in enumerate(has) if not h)))
    return out


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
    m = fp.decode(C, list(idx[f.nb:]))
    W = f.shift + 23 + top                     # the sum's bits: the word below base is < 2^(23 + top)
    q = truncated_square(C, u, f.p)
    ops = [(u if f.s1 > 0 else [NOT(b) for b in u], [abs(c) for c in f.C1], 0),
           (q if f.s2 > 0 else [NOT(b) for b in q], [abs(c) for c in f.C2], f.square_shift)]
    c0 = list(f.C0)
    for (x, mag, at), s in zip(ops, (f.s1, f.s2)):
        if s < 0:
            c0 = [c - (((1 << len(x)) - 1) * c_ << at) for c, c_ in zip(c0, mag)]
    cols: list[list] = [[] for _ in range(W)]
    for j, b in enumerate(_select(m, [c % (1 << W) for c in c0], W)):
        cols[j].append(b)
    for x, mag, at in ops:
        for i, c in enumerate(_select(m, mag, max(mag).bit_length())):
            for j, b in enumerate(x):
                if i + j + at < W:
                    cols[i + j + at].append(C.AND(c, b))
    for i, d in overrides(name):
        lit = [b if (i >> k) & 1 else NOT(b) for k, b in enumerate(u)]
        hit = C.AND(m[i >> f.nb], and_reduce(C, lit))
        for k in range(W - f.shift):
            if (d % (1 << (W - f.shift))) >> k & 1:
                cols[f.shift + k].append(hit)
    y = compress_columns(C, cols)[f.shift:W]
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
    """``MufuEx2Ftz_v1``: x as a signed fixed-point value with 23 fraction bits; the table at its fraction, the integer part
    added to the table word's exponent. A right shift past the mantissa is clamped (K = 0), so every |x| < 2^-23 reads the
    table's first word."""
    s, e, m, e_max, e_zero, m_nz = _fields(C, x)
    big = NOT(_ult_const(C, e, 157))                          # e - 127 >= 30
    neg = _ult_const(C, e, 127)
    d8 = sub(C, const_bits(127, 8), e, 8)[0]
    d = d8[:6]
    over = C.AND(neg, NOT(_ult_const(C, d8, 25)))
    sh = mux(C, neg, sub(C, const_bits(24, 6), d, 6)[0], sub(C, e, const_bits(103, 8), 8)[0][:6])
    K = fp.shl(C, m + [ONE] + [ZERO] * 53, sh)[24:77]          # mant << (e - 127), or the right shift above; 53 bits
    K = [C.AND(NOT(over), k) for k in K]
    z = NOT(or_reduce(C, K[:23]))
    fixed = add(C, [k ^ s for k in K + [ZERO]], [C.AND(s, z)], width=54)   # s ? (z ? -K : ~K) : K
    t = table_word(C, "ex2", fixed[:23])
    n = fixed[23:54]
    re = add(C, list(t[23:31]) + [ZERO] * 25, n + [n[30]] * 2, width=33)
    neg = re[32]
    zero = OR(C, neg, NOT(or_reduce(C, re)))
    ovf = C.AND(NOT(neg), NOT(_ult_const(C, re[:32], 255)))
    inf_or_zero = mux(C, s, const_bits(0, 32), const_bits(INF, 32))
    return _word(C, [(e_max, mux(C, m_nz, const_bits(QNAN_MUFU, 32), inf_or_zero)), (e_zero, const_bits(ONE_F, 32)),
                     (big, inf_or_zero), (ovf, const_bits(INF, 32)), (zero, const_bits(0, 32))],
                 list(t[:23]) + re[:8] + [ZERO])


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


def _abs_gt_big(C, x):
    """``|x| > 2^126`` as f32 compares it (false for NaN)."""
    _, e, _, e_max, _, m_nz = _fields(C, x)
    nan = C.AND(e_max, m_nz)
    ge254 = C.AND(e[7], C.AND(e[6], C.AND(e[5], C.AND(e[4], C.AND(e[3], C.AND(e[2], e[1]))))))
    return C.AND(NOT(nan), OR(C, ge254, C.AND(_eq_const(C, e, 253), m_nz)))


def _scale(C, x, k: int):
    return fp.f32_mul(C, x, const_bits(k, 32))


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
    """``RsqrtApprox_v1``: MUFU.RSQ with the subnormal range scaling (x * 2^24, the result * 2^12)."""
    p0 = NOT(_fields(C, x)[4])
    xs = mux(C, p0, x, _scale(C, x, TWO24))
    r = _rsq_post(C, xs, table_word(C, "rsq", _parity_index(xs)))
    special = OR(C, _fields(C, r)[3], _fields(C, r)[4])
    return mux(C, OR(C, p0, special), r, list(r[:23]) + add(C, r[23:31], const_bits(12, 8), width=8) + [r[31]])


def mufu_sqrt_ftz(C, x):
    """``MufuSqrtFtz_v1``: MUFU.SQRT (``sqrt.approx.ftz.f32``)."""
    s, e, _, e_max, e_zero, m_nz = _fields(C, x)
    v = table_word(C, "sqrt", _parity_index(x))
    k = sub(C, e + [ZERO], const_bits(127, 9), 9)[0]
    re = add(C, list(v[23:31]) + [ZERO], k[1:] + [k[8]], width=9)
    return _word(C, [(e_max, mux(C, OR(C, m_nz, s), const_bits(QNAN_MUFU, 32), const_bits(INF, 32))),
                     (e_zero, [ZERO] * 31 + [s]), (s, const_bits(QNAN_MUFU, 32))], list(v[:23]) + re)


def _div_scale(C, a, b):
    big = _abs_gt_big(C, b)
    a1, b1 = mux(C, big, _scale(C, a, QUARTER), a), mux(C, big, _scale(C, b, QUARTER), b)
    tiny = _fields(C, b1)[4]
    return mux(C, tiny, _scale(C, a1, TWO24), a1), mux(C, tiny, _scale(C, b1, TWO24), b1)


def div_full_rcp(C, b):
    """``DivFullRcp_v1``: ``div.full.f32``'s reciprocal part: range-scale b, MUFU.RCP, times the matching scaled 1."""
    big = _abs_gt_big(C, b)
    tiny = _fields(C, b)[4]
    fb = mux(C, big, _scale(C, b, QUARTER), mux(C, tiny, _scale(C, b, TWO24), b))
    a2, b2 = _div_scale(C, const_bits(ONE_F, 32), fb)
    return fp.f32_mul(C, a2, rcp_post(C, b2, table_word(C, "rcp", b2[:23])), ir_nan=True)


def div_full_scale_a(C, a, b):
    """``DivFullScaleA_v1``: ``a * 0.25`` when ``|b| > 2^126``, ``a * 2^24`` when ``|b| < 2^-126``, else a (NaN b: a)."""
    mag = b[:31]
    big = C.AND(NOT(_ult_const(C, mag, 0x7E800001)), _ult_const(C, mag, 0x7F800001))
    small = _ult_const(C, mag, 0x00800000)
    return mux(C, big, fp.f32_mul(C, a, fp.f32_const(QUARTER), ir_nan=True),
               mux(C, small, fp.f32_mul(C, a, fp.f32_const(TWO24), ir_nan=True), list(a)))


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
