"""The k16 tensor-core step as a minimal algebraic relation: witness columns, predicates, and their census.

This module is the *census input* for the FLOP-proof backend decision.  It restates `tc_dot_total` -- one accumulator
transition of `mma.sync.m16n8k16` (Ampere: two groups of eight products) or one k16 slice of `wgmma` (Hopper: one group of
sixteen) -- as a fixed set of predicates over integer witness columns, of exactly three kinds:

    mul      one field multiplication between two witness values:   a * b = c        (Booleanity is a mul: x * (1-x) = 0)
    lin      one linear identity with constant coefficients:        sum c_i x_i = c
    lookup   one query into a fixed table:                          (key, outs...) in T
             a w-bit range check is ceil(w/16) lookups into 16-bit range tables (plus one `lin` split when w > 16)

Every *honest* witness value is a non-negative integer below 2^40 (the largest: the K-bit remainder scale, K = W + 10).
That does **not** make the predicates sound over a 40-bit field (WP2 checker lane, 2026-09-21, z3 at real widths --
`verity_numerical.checker.search`, run r20260921-232849-093c): the remainder columns `r`, `r_s` are never range-checked
themselves, only their scaled products `rinv = r * invpow`, `rsinv = r_s * invpow_s`, so over F_p with p ~ 2^40 a
dishonest `r >= 2^d` whose product wraps below p into [0, 2^A) is accepted (`align_state` over F_p40 as stated: SAT;
over Goldilocks with `r` range-checked: UNSAT), and `aligned` carries no tight quotient bound ``q < 2^(A-d)``, which a
31-bit field needs even with ideal remainder checks (M = 0, d >= W, q = 63, r = 2^25 - 1 gives 2^31 - 1 = 0 in
Mersenne31: SAT).  The red-team lane (same day, `verity_numerical.redteam`, report
`backends/numerical/reports/redteam_relation.json`) found a forgery that is **field-independent**: `magW` (the split
of `normB` into a W-bit magnitude and the remainder `r_n`) is never range-checked, so `magW' = magW + k * 2^W (mod p)`
with a fitting `r_n'` yields an accepted wrong output word over every prime tried, Goldilocks included (388 of 870
Ampere groups; forged `0x3f7ffffe` for the honest `0x3f800000` on the all-zero group); RANGE(magW, W) makes every
whole-group z3 query UNSAT.  The wrap thresholds for the other products are p >= 2^52 (`aligned * pow`), 2^59
(`mag24 * pow_s`) and 2^60 (`mag_t * normpow`) on Ampere (2^54 / 2^60 / 2^64 on Hopper).  The "E* attained" chain
``prod_i (E* - e_i) = 0`` is a *field* identity (a product vanishes iff a factor does) whose intermediates are not
range-checked either.  Anyone embedding this relation in a field must add RANGE(magW, W), a direct range check on every
remainder and the tight quotient bound, and choose p above the wrap thresholds; the census below counts none of that,
nor the decode of an arbitrary transition-unit accumulator (the VU starts from the constant zero state; the backend
transition unit does not).  The gadget contracts in `verity_numerical.checker` (WP2) carry the bounds and are UNSAT
under z3 at real widths; build from those, not from this module.  The relation is *total* and *unconditional*:
NaN/Inf/zero operands
take the same predicates as finite ones (their finite-path magnitude is 0, so the finite path is always well defined on
masked values, and the class flags select the output).

Design choices that exploit the primitive rather than a software FP algorithm:

* The **state** of a verification unit is the *decoded* accumulator ``(nan, inf, neg, e, M)`` -- class flags, sign,
  exponent, 24-bit magnitude -- not the FP32 word.  Decoding happens once at the VU boundary (`decode_state`) and packing
  once at the end (`pack_relation`); the K/16 steps in between never touch the word.
* A BF16 operand is decoded by **one lookup** into a 2^16 table (`T_bf16`: sign, effective exponent, significand with
  the hidden bit, class).  No bit decomposition of operands.  Special-value logic of a product is **one lookup** into a
  64-row table on the two classes and signs (`T_pcls`).
* Zero and special terms carry magnitude 0 and are masked to the exponent floor; the group maximum ``E*`` is established
  by ``E* - e_i in [0, 512)`` (from the power table) and ``prod_i (E* - e_i) * (E* - floor) = 0`` -- no one-hot selector.
* Truncating alignment (the one thing the hardware does that IEEE would not) is division with remainder against a
  looked-up power of two; the remainder bound ``r < 2^d`` is ``r * 2^(A-d) in [0, 2^A)`` (one mul, one range) rather than
  a variable-bound range check.  Products have only 16 significant bits, so their alignment can instead be a single
  lookup into a ``(P, d)`` shift table (`ALIGN_TABLE`); both modes are implemented and counted.
* One normalisation per **group** (8 or 16 products), not per product; every exponent-side conditional (subnormal
  shift, rescale, overflow) is one lookup on the pre-normalisation exponent (`T_exp`).

`census(pipeline)` produces the count; `tests/test_relation.py` replays the hardware captures through `step_witness` +
`check_step` and mutates witness columns to confirm each one is constrained.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .silicon import Pipeline

__all__ = ["ALIGN_DIV", "ALIGN_TABLE", "Census", "State", "TABLES", "Violation", "census", "check_step", "decode_state",
           "pack_relation", "pack_state", "step_witness", "vu_census", "witness_shape_error", "zero_state"]

CLS_FIN, CLS_ZERO, CLS_INF, CLS_NAN = 0, 1, 2, 3
FP32_MIN_E = -126
FP32_MAX_E = 127
RANGE_CHUNK = 16          # width of the largest range table; a w-bit range check is ceil(w/16) lookups
ALIGN_DIV = "div"         # every term aligned by division with remainder
ALIGN_TABLE = "table"     # products aligned by one (P, d) shift-table lookup; the accumulator term by division
E_OFF = 192               # T_exp is indexed by e_out + E_OFF


class Violation(AssertionError):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# Parameters of a pipeline
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    W: int        # adder / normalisation width (25 Ampere, 26 Hopper)
    rho: int      # rescale from the 24-bit FP32 grid: W - 24
    A: int        # bits of a scaled term: products are P << (9 + rho), so W + 1
    B: int        # |group total| < n_terms * 2^A < 2^B
    K: int        # bound on the final shift (subnormal + rescale) remainder scale
    Z: int        # exponent floor
    n: int        # largest group (products per group)


def params(pipe: Pipeline) -> Params:
    W = pipe.width
    n = max(pipe.groups)
    A = W + 1
    return Params(W=W, rho=W - 24, A=A, B=A + n.bit_length(), K=W + 10, Z=pipe.zero_exponent, n=n)


# ---------------------------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Table:
    name: str
    size: int                              # rows
    outs: tuple[str, ...]
    fn: Callable[..., tuple[int, ...]]     # key(s) -> outputs; KeyError off the table

    def __call__(self, *key: int) -> tuple[int, ...]:
        return self.fn(*key)


TABLES: dict[str, Table] = {}


def _reg(t: Table) -> Table:
    return TABLES.setdefault(t.name, t)


def _bf16(word: int) -> tuple[int, int, int, int]:
    """word -> (sign, effective exponent `e or 1`, significand with hidden bit -- 0 unless finite nonzero, class)."""
    if not 0 <= word < 1 << 16:
        raise KeyError(word)
    s, e, m = word >> 15, (word >> 7) & 0xFF, word & 0x7F
    if e == 0xFF:
        return s, 1, 0, CLS_NAN if m else CLS_INF
    if e == 0 and m == 0:
        return s, 1, 0, CLS_ZERO
    return s, e or 1, m | (0x80 if e else 0), CLS_FIN


def _pcls(ca: int, cb: int, sa: int, sb: int) -> tuple[int, int, int, int, int]:
    """(class a, class b, sign a, sign b) -> (nan, +inf, -inf, zero-in-the-finite-path, sign) of the product."""
    if not (0 <= ca < 4 and 0 <= cb < 4 and sa in (0, 1) and sb in (0, 1)):
        raise KeyError((ca, cb, sa, sb))
    sp = sa ^ sb
    if ca == CLS_NAN or cb == CLS_NAN or (ca == CLS_INF and cb == CLS_ZERO) or (cb == CLS_INF and ca == CLS_ZERO):
        return 1, 0, 0, 1, sp
    if ca == CLS_INF or cb == CLS_INF:
        return 0, 1 - sp, sp, 1, sp
    return 0, 0, 0, int(ca != CLS_FIN or cb != CLS_FIN), sp


T_BF16 = _reg(Table("T_bf16", 1 << 16, ("s", "e", "sig", "cls"), _bf16))
T_PCLS = _reg(Table("T_pcls", 64, ("nan", "pinf", "ninf", "zero", "sign"), _pcls))


def T_range(bits: int) -> Table:
    def fn(x: int) -> tuple[()]:
        if not 0 <= x < 1 << bits:
            raise KeyError(x)
        return ()
    return _reg(Table(f"T_range{bits}", 1 << bits, (), fn))


def T_iszero(bits: int) -> Table:
    def fn(x: int) -> tuple[int]:
        if not 0 <= x < 1 << bits:
            raise KeyError(x)
        return (int(x == 0),)
    return _reg(Table(f"T_iszero{bits}", 1 << bits, ("is_zero",), fn))


def T_any(n: int) -> Table:
    def fn(s: int) -> tuple[int]:
        if not 0 <= s <= n:
            raise KeyError(s)
        return (int(s != 0),)
    return _reg(Table(f"T_any{n}", n + 1, ("any",), fn))


def T_pow(p: Params) -> Table:
    def fn(d: int) -> tuple[int, int]:
        if not 0 <= d < 512:
            raise KeyError(d)
        dd = min(d, p.A)
        return 1 << dd, 1 << (p.A - dd)
    return _reg(Table(f"T_pow{p.A}", 512, ("pow", "invpow"), fn))


def T_norm(p: Params) -> Table:
    def fn(L: int) -> tuple[int, int]:
        if not 0 <= L <= p.B:
            raise KeyError(L)
        return 1 << (p.B - L), int(L == 0)
    return _reg(Table(f"T_norm{p.B}", p.B + 1, ("pow", "is_zero"), fn))


def T_exp(p: Params) -> Table:
    """e_out + E_OFF -> (final exponent, 2^(subnormal shift + rho), 2^(K - that), overflow)."""
    def fn(idx: int) -> tuple[int, int, int, int]:
        if not 0 <= idx < 512:
            raise KeyError(idx)
        e_out = idx - E_OFF
        e_fin = max(e_out, FP32_MIN_E)
        sh = min(e_fin - e_out + p.rho, p.K)
        return e_fin, 1 << sh, 1 << (p.K - sh), int(e_fin > FP32_MAX_E)
    return _reg(Table(f"T_exp{p.W}", 512, ("e_fin", "pow", "invpow", "ovf"), fn))


def T_shift(p: Params) -> Table:
    """(P, d) -> floor((P << (9 + rho)) / 2^d): the alternative product alignment.  2^16 * (A + 1) distinct rows."""
    def fn(P: int, d: int) -> tuple[int]:
        if not (0 <= P < 1 << 16 and 0 <= d < 512):
            raise KeyError((P, d))
        return ((P << (9 + p.rho)) >> min(d, p.A),)
    return _reg(Table(f"T_shift{p.A}", (1 << 16) * (p.A + 1), ("aligned",), fn))


# ---------------------------------------------------------------------------------------------------------------------
# The census collector
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Census:
    """Counts predicates by kind while checking them.  `strict=False` records violations instead of raising."""
    strict: bool = True
    mul: int = 0
    lin: int = 0
    lookups: Counter = field(default_factory=Counter)
    violations: list[str] = field(default_factory=list)

    def _fail(self, msg: str) -> None:
        if self.strict:
            raise Violation(msg)
        self.violations.append(msg)

    def MUL(self, name: str, a: int, b: int, c: int) -> None:
        self.mul += 1
        if a * b != c:
            self._fail(f"{name}: {a} * {b} != {c}")

    def BOOL(self, name: str, x: int) -> None:
        self.MUL(f"{name}.bool", x, 1 - x, 0)

    def LIN(self, name: str, residual: int) -> None:
        self.lin += 1
        if residual != 0:
            self._fail(f"{name}: linear residual {residual}")

    def LOOKUP(self, table: Table, key: tuple[int, ...], outs: tuple[int, ...]) -> None:
        self.lookups[table.name] += 1
        try:
            got = table(*key)
        except KeyError:
            self._fail(f"{table.name}: key {key} off the table")
            return
        if tuple(got) != tuple(outs):
            self._fail(f"{table.name}{key}: {tuple(got)} != {tuple(outs)}")

    def RANGE(self, name: str, x: int, bits: int) -> None:
        """x in [0, 2^bits): one lookup for bits <= 16, else a 16-bit chunk split (one lin) and one lookup per chunk."""
        if bits <= RANGE_CHUNK:
            self.LOOKUP(T_range(bits), (x,), ())
            return
        if not isinstance(x, int) or x < 0:
            self._fail(f"{name}: {x} is not a non-negative integer")
            return
        widths = [RANGE_CHUNK] * (bits // RANGE_CHUNK) + ([bits % RANGE_CHUNK] if bits % RANGE_CHUNK else [])
        chunks, v = [], x
        for w in widths:
            chunks.append(v & ((1 << w) - 1))
            v >>= w
        self.LIN(f"{name}.split", sum(c << (RANGE_CHUNK * i) for i, c in enumerate(chunks)) - x)
        for c, w in zip(chunks, widths):
            self.LOOKUP(T_range(w), (c,), ())

    def ISZERO(self, name: str, x: int, bits: int, z: int) -> None:
        """z = [x == 0] for x in [0, 2^bits): 12-bit chunks, one is-zero lookup each, and-ed by muls."""
        widths = [12] * (bits // 12) + ([bits % 12] if bits % 12 else [])
        chunks, v = [], x
        for w in widths:
            chunks.append(v & ((1 << w) - 1))
            v >>= w
        self.LIN(f"{name}.split", sum(c << (12 * i) for i, c in enumerate(chunks)) - x)
        flags = []
        for c, w in zip(chunks, widths):
            f = int(c == 0)
            self.LOOKUP(T_iszero(w), (c,), (f,))
            flags.append(f)
        acc = flags[0]
        for f in flags[1:]:
            self.MUL(f"{name}.and", acc, f, acc * f)
            acc *= f
        self.LIN(f"{name}.z", z - acc)

    @property
    def lookup_total(self) -> int:
        return sum(self.lookups.values())

    def summary(self) -> dict:
        return {"mul": self.mul, "lin": self.lin, "lookup": self.lookup_total,
                "lookups_by_table": dict(sorted(self.lookups.items()))}


# ---------------------------------------------------------------------------------------------------------------------
# State and the VU boundary
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class State:
    """The decoded accumulator.  Canonical: `nan`, `inf` exclusive Booleans; special or zero -> M = 0 and e = floor;
    finite zero -> neg = 0; an infinity carries its sign in `neg`."""
    nan: int
    inf: int
    neg: int
    e: int
    M: int


def zero_state(pipe: Pipeline) -> State:
    return State(0, 0, 0, pipe.zero_exponent, 0)


def decode_state(word: int, pipe: Pipeline) -> State:
    """The boundary decode of an FP32 accumulator word (the initial state of a VU is the constant `zero_state`)."""
    Z = pipe.zero_exponent
    if not 0 <= word < 1 << 32:
        raise ValueError(word)
    s, E, m = word >> 31, (word >> 23) & 0xFF, word & 0x7FFFFF
    if E == 0xFF:
        return State(1, 0, 0, Z, 0) if m else State(0, 1, s, Z, 0)
    if E == 0:
        return State(0, 0, 0, Z, 0) if m == 0 else State(0, 0, s, FP32_MIN_E, m)
    return State(0, 0, s, E - 127, (1 << 23) | m)


def pack_state(st: State) -> int:
    if st.nan:
        return 0x7FFFFFFF
    if st.inf:
        return (st.neg << 31) | 0x7F800000
    if st.M == 0:
        return 0
    return (st.neg << 31) | ((st.e + 127 - (0 if st.M >> 23 else 1)) << 23) | (st.M & 0x7FFFFF)


def pack_relation(st: State, pipe: Pipeline, c: Census) -> tuple[int, int]:
    """The VU boundary, once per unit: state -> FP32 word -> `cvt.rn.bf16.f32` word.  Returns (fp32, bf16)."""
    hidden = st.M >> 23
    low = st.M - (hidden << 23)
    c.BOOL("pack.hidden", hidden)
    c.RANGE("pack.low", low, 23)
    z = int(st.M == 0)
    c.ISZERO("pack.z", st.M, 24, z)
    fin = 1 - st.nan - st.inf
    nz = 1 - z
    nh_nz = (1 - hidden) * nz
    c.MUL("pack.nh_nz", 1 - hidden, nz, nh_nz)
    c.MUL("pack.subnormal_e", nh_nz, st.e - FP32_MIN_E, 0)           # no hidden bit and nonzero -> e = -126
    fin_nz = fin * nz
    c.MUL("pack.fin_nz", fin, nz, fin_nz)
    Ebias = fin_nz * (st.e + 126 + hidden) + (st.nan + st.inf) * 255  # normal: e+127; subnormal: 0; special: 255
    c.MUL("pack.Ebias", fin_nz, st.e + 126 + hidden, Ebias - (st.nan + st.inf) * 255)
    mant = fin * low + st.nan * 0x7FFFFF
    c.MUL("pack.mant", fin, low, mant - st.nan * 0x7FFFFF)
    word = (st.neg << 31) + (Ebias << 23) + mant
    c.LIN("pack.word", word - (st.neg * (1 << 31) + Ebias * (1 << 23) + mant))
    # cvt.rn.bf16.f32: every NaN -> 0x7FFF; otherwise RNE, carrying into the exponent (overflow -> inf, as measured)
    hi, lo = word >> 16, word & 0xFFFF
    c.RANGE("cvt.lo", lo, 16)
    c.RANGE("cvt.hi", hi, 16)
    c.LIN("cvt.split", hi * 65536 + lo - word)
    b16, hi2 = hi & 1, hi >> 1
    c.BOOL("cvt.b16", b16)
    c.RANGE("cvt.hi2", hi2, 15)
    c.LIN("cvt.hi2", hi2 * 2 + b16 - hi)
    t = lo + 0x7FFF + b16
    carry, rem = t >> 16, t & 0xFFFF
    c.BOOL("cvt.carry", carry)
    c.RANGE("cvt.rem", rem, 16)
    c.LIN("cvt.round", carry * 65536 + rem - t)
    rounded, wrap = (hi + carry) & 0xFFFF, (hi + carry) >> 16
    c.BOOL("cvt.wrap", wrap)
    c.RANGE("cvt.rounded", rounded, 16)
    c.LIN("cvt.rounded", rounded + wrap * 65536 - hi - carry)
    bf16 = rounded + st.nan * (0x7FFF - rounded)
    c.MUL("cvt.nan_mux", st.nan, 0x7FFF - rounded, bf16 - rounded)
    return word, bf16


# ---------------------------------------------------------------------------------------------------------------------
# Witness of one k16 step
# ---------------------------------------------------------------------------------------------------------------------

def _terms(p: Params, st: State, prods: list[dict]) -> list[tuple[int, int, int]]:
    """(neg, masked exponent, scaled magnitude) of the accumulator and each product of a group."""
    return [(st.neg, st.e, st.M << p.rho)] + [(q["sp"], q["em"], q["P"] << (9 + p.rho)) for q in prods]


def step_witness(pipe: Pipeline, st: State, a: Sequence[int], b: Sequence[int], align: str = ALIGN_DIV) -> dict:
    """Every witness column of one k16 step, mirroring `tc_dot_total`; a dict tree consumed by `check_step`."""
    p = params(pipe)
    Z = p.Z
    if len(a) != pipe.k or len(b) != pipe.k:
        raise ValueError("operand count")
    w: dict = {"in": st, "a": list(a), "b": list(b), "prod": [], "groups": [], "align": align}
    for j in range(pipe.k):
        sa, ea, SA, ca = _bf16(a[j])
        sb, eb, SB, cb = _bf16(b[j])
        nan, pinf, ninf, zero, sp = _pcls(ca, cb, sa, sb)
        ep = ea + eb - 254
        w["prod"].append({"sa": sa, "ea": ea, "SA": SA, "ca": ca, "sb": sb, "eb": eb, "SB": SB, "cb": cb,
                          "nan": nan, "pinf": pinf, "ninf": ninf, "zero": zero, "sp": sp,
                          "P": SA * SB, "ep": ep, "em": ep + zero * (Z - ep)})
    cur = st
    start = 0
    pow_t, norm_t, exp_t = T_pow(p), T_norm(p), T_exp(p)
    for size in pipe.groups:
        prods = w["prod"][start:start + size]
        start += size
        g: dict = {}
        terms = _terms(p, cur, prods)
        Estar = max([Z] + [e for _, e, m in terms if m])
        g["Estar"] = Estar
        g["terms"] = []
        total = 0
        for neg, e, scaled in terms:
            d = Estar - e
            pw, ipw = pow_t(d)
            aligned = scaled // pw
            r = scaled - aligned * pw
            g["terms"].append({"d": d, "pow": pw, "invpow": ipw, "aligned": aligned, "r": r, "rinv": r * ipw,
                               "signed": neg * aligned})
            total += aligned - 2 * neg * aligned
        neg_t = int(total < 0)
        mag_t = -total if neg_t else total
        L = mag_t.bit_length()
        normpow, z_t = norm_t(L)
        normB = mag_t * normpow
        magW = normB >> (p.B - p.W)
        e_out = Estar + L - p.W
        e_fin, pow_s, invpow_s, ovf = exp_t(e_out + E_OFF)
        mag24 = magW // pow_s
        r_s = magW - mag24 * pow_s
        z_out = int(mag24 == 0)
        g.update(total=total, neg_t=neg_t, mag_t=mag_t, L=L, normpow=normpow, z_t=z_t, normB=normB, magW=magW,
                 r_n=normB - (magW << (p.B - p.W)), e_out=e_out, e_fin=e_fin, pow_s=pow_s, invpow_s=invpow_s, ovf=ovf,
                 mag24=mag24, r_s=r_s, rsinv=r_s * invpow_s, z_out=z_out)
        # special values
        pinf0, ninf0 = cur.inf * (1 - cur.neg), cur.inf * cur.neg
        S_nan = cur.nan + sum(q["nan"] for q in prods)
        S_pinf = pinf0 + sum(q["pinf"] for q in prods)
        S_ninf = ninf0 + sum(q["ninf"] for q in prods)
        any_nan, any_pinf, any_ninf = int(S_nan > 0), int(S_pinf > 0), int(S_ninf > 0)
        ovf_eff = ovf * (1 - z_t)
        any_inf = any_pinf + any_ninf - any_pinf * any_ninf
        nan_out = any_nan + (1 - any_nan) * any_pinf * any_ninf
        inf_out = (1 - nan_out) * (any_inf + (1 - any_inf) * ovf_eff)
        neg_inf = any_inf * any_ninf + (1 - any_inf) * neg_t
        fin = 1 - nan_out - inf_out
        neg_fin = neg_t * (1 - z_out)
        e_state = e_fin + z_out * (Z - e_fin)
        out = State(nan_out, inf_out, inf_out * neg_inf + fin * neg_fin, Z + fin * (e_state - Z), fin * mag24)
        g.update(pinf0=pinf0, ninf0=ninf0, S_nan=S_nan, S_pinf=S_pinf, S_ninf=S_ninf, any_nan=any_nan,
                 any_pinf=any_pinf, any_ninf=any_ninf, ovf_eff=ovf_eff, any_inf=any_inf, nan_out=nan_out,
                 inf_out=inf_out, neg_inf=neg_inf, fin=fin, neg_fin=neg_fin, e_state=e_state, out=out)
        w["groups"].append(g)
        cur = out
    w["out"] = cur
    return w


# ---------------------------------------------------------------------------------------------------------------------
# The predicates
# ---------------------------------------------------------------------------------------------------------------------

def witness_shape_error(pipe: Pipeline, w: dict) -> str | None:
    """Why `w` does not have the relation's fixed shape (`k` operand words and product rows, one group row per group,
    `size + 1` term rows in each), or `None`.  The predicates below range over the witness's own lists, so a shorter
    list would drop predicates rather than fail them."""
    if len(w["a"]) != pipe.k or len(w["b"]) != pipe.k:
        return f"{len(w['a'])} and {len(w['b'])} operand words, not {pipe.k}"
    if len(w["prod"]) != pipe.k:
        return f"{len(w['prod'])} product rows, not {pipe.k}"
    if len(w["groups"]) != len(pipe.groups):
        return f"{len(w['groups'])} group rows, not {len(pipe.groups)}"
    for gi, (size, g) in enumerate(zip(pipe.groups, w["groups"])):
        if len(g["terms"]) != size + 1:
            return f"group {gi}: {len(g['terms'])} term rows, not {size + 1}"
    return None


def check_step(pipe: Pipeline, w: dict, c: Census) -> State:
    """Check every predicate of one k16 step against the witness `w`; returns the claimed output state.  A witness of
    the wrong shape is a violation, and no predicate is checked on it."""
    shape = witness_shape_error(pipe, w)
    if shape is not None:
        c._fail(f"witness shape: {shape}")
        return w["in"]
    p = params(pipe)
    Z = p.Z
    align = w.get("align", ALIGN_DIV)
    pow_t, norm_t, exp_t = T_pow(p), T_norm(p), T_exp(p)
    shift_t = T_shift(p) if align == ALIGN_TABLE else None
    a, b = w["a"], w["b"]

    # ---- products: decode, class logic, exact significand product, exponent, zero mask -------------------------------
    for j, q in enumerate(w["prod"]):
        c.LOOKUP(T_BF16, (a[j],), (q["sa"], q["ea"], q["SA"], q["ca"]))
        c.LOOKUP(T_BF16, (b[j],), (q["sb"], q["eb"], q["SB"], q["cb"]))
        c.LOOKUP(T_PCLS, (q["ca"], q["cb"], q["sa"], q["sb"]), (q["nan"], q["pinf"], q["ninf"], q["zero"], q["sp"]))
        c.MUL(f"P[{j}]", q["SA"], q["SB"], q["P"])
        c.LIN(f"ep[{j}]", q["ep"] - (q["ea"] + q["eb"] - 254))
        c.MUL(f"em[{j}]", q["zero"], Z - q["ep"], q["em"] - q["ep"])

    st: State = w["in"]
    start = 0
    for gi, (size, g) in enumerate(zip(pipe.groups, w["groups"])):
        prods = w["prod"][start:start + size]
        start += size
        Estar, terms = g["Estar"], g["terms"]
        ins = _terms(p, st, prods)

        # ---- E*: at least every masked exponent and the floor (d in [0, 512) via T_pow), and attained ---------------
        chain = Estar - Z
        for i, ((neg, e, scaled), t) in enumerate(zip(ins, terms)):
            c.LIN(f"g{gi}.d[{i}]", t["d"] - (Estar - e))
            c.LOOKUP(pow_t, (t["d"],), (t["pow"], t["invpow"]))
            c.MUL(f"g{gi}.attain[{i}]", chain, Estar - e, chain * (Estar - e))
            chain *= Estar - e
        c.LIN(f"g{gi}.attained", chain)

        # ---- truncating alignment, signed sum ----------------------------------------------------------------------
        total = 0
        for i, ((neg, e, scaled), t) in enumerate(zip(ins, terms)):
            if i == 0 or align == ALIGN_DIV:
                c.MUL(f"g{gi}.align[{i}]", t["aligned"], t["pow"], scaled - t["r"])      # aligned * 2^d + r = scaled
                c.RANGE(f"g{gi}.aligned[{i}]", t["aligned"], p.A)
                c.MUL(f"g{gi}.rinv[{i}]", t["r"], t["invpow"], t["rinv"])
                c.RANGE(f"g{gi}.rinv[{i}]", t["rinv"], p.A)                               # r * 2^(A-d) < 2^A <=> r < 2^d
            else:
                c.LOOKUP(shift_t, (prods[i - 1]["P"], t["d"]), (t["aligned"],))
            c.MUL(f"g{gi}.signed[{i}]", neg, t["aligned"], t["signed"])
            total += t["aligned"] - 2 * t["signed"]
        c.LIN(f"g{gi}.total", g["total"] - total)

        # ---- magnitude and sign of the total; normalisation to W bits; subnormal shift and rescale -----------------
        c.BOOL(f"g{gi}.neg_t", g["neg_t"])
        c.MUL(f"g{gi}.mag_t", 1 - 2 * g["neg_t"], g["total"], g["mag_t"])
        c.RANGE(f"g{gi}.mag_t", g["mag_t"], p.B)
        c.LOOKUP(norm_t, (g["L"],), (g["normpow"], g["z_t"]))
        c.MUL(f"g{gi}.normB", g["mag_t"], g["normpow"], g["normB"])
        c.RANGE(f"g{gi}.normB.top", g["normB"] - (1 - g["z_t"]) * (1 << (p.B - 1)), p.B - 1)   # in [2^(B-1), 2^B) or 0
        c.LIN(f"g{gi}.normW", g["magW"] * (1 << (p.B - p.W)) + g["r_n"] - g["normB"])
        c.RANGE(f"g{gi}.r_n", g["r_n"], p.B - p.W)
        c.LIN(f"g{gi}.e_out", g["e_out"] - (Estar + g["L"] - p.W))
        c.LOOKUP(exp_t, (g["e_out"] + E_OFF,), (g["e_fin"], g["pow_s"], g["invpow_s"], g["ovf"]))
        c.MUL(f"g{gi}.final", g["mag24"], g["pow_s"], g["magW"] - g["r_s"])              # mag24 * 2^sh + r_s = magW
        c.RANGE(f"g{gi}.mag24", g["mag24"], 24)
        c.MUL(f"g{gi}.rsinv", g["r_s"], g["invpow_s"], g["rsinv"])
        c.RANGE(f"g{gi}.rsinv", g["rsinv"], p.K)
        c.ISZERO(f"g{gi}.z_out", g["mag24"], 24, g["z_out"])

        # ---- special values and the output state ------------------------------------------------------------------
        c.MUL(f"g{gi}.pinf0", st.inf, 1 - st.neg, g["pinf0"])
        c.MUL(f"g{gi}.ninf0", st.inf, st.neg, g["ninf0"])
        c.LIN(f"g{gi}.S_nan", g["S_nan"] - st.nan - sum(q["nan"] for q in prods))
        c.LIN(f"g{gi}.S_pinf", g["S_pinf"] - g["pinf0"] - sum(q["pinf"] for q in prods))
        c.LIN(f"g{gi}.S_ninf", g["S_ninf"] - g["ninf0"] - sum(q["ninf"] for q in prods))
        any_t = T_any(size + 1)
        c.LOOKUP(any_t, (g["S_nan"],), (g["any_nan"],))
        c.LOOKUP(any_t, (g["S_pinf"],), (g["any_pinf"],))
        c.LOOKUP(any_t, (g["S_ninf"],), (g["any_ninf"],))
        c.MUL(f"g{gi}.ovf_eff", g["ovf"], 1 - g["z_t"], g["ovf_eff"])
        both = g["any_pinf"] * g["any_ninf"]
        c.MUL(f"g{gi}.both_inf", g["any_pinf"], g["any_ninf"], both)
        c.LIN(f"g{gi}.any_inf", g["any_inf"] - (g["any_pinf"] + g["any_ninf"] - both))
        c.MUL(f"g{gi}.nan_out", 1 - g["any_nan"], both, g["nan_out"] - g["any_nan"])
        fin_inf = g["any_inf"] + (1 - g["any_inf"]) * g["ovf_eff"]
        c.MUL(f"g{gi}.fin_inf", 1 - g["any_inf"], g["ovf_eff"], fin_inf - g["any_inf"])
        c.MUL(f"g{gi}.inf_out", 1 - g["nan_out"], fin_inf, g["inf_out"])
        c.MUL(f"g{gi}.neg_inf.a", g["any_inf"], g["any_ninf"], g["any_inf"] * g["any_ninf"])
        c.MUL(f"g{gi}.neg_inf", 1 - g["any_inf"], g["neg_t"], g["neg_inf"] - g["any_inf"] * g["any_ninf"])
        c.LIN(f"g{gi}.fin", g["fin"] - (1 - g["nan_out"] - g["inf_out"]))
        c.MUL(f"g{gi}.neg_fin", g["neg_t"], 1 - g["z_out"], g["neg_fin"])
        c.MUL(f"g{gi}.e_state", g["z_out"], Z - g["e_fin"], g["e_state"] - g["e_fin"])
        out: State = g["out"]
        c.MUL(f"g{gi}.out.neg.a", g["inf_out"], g["neg_inf"], g["inf_out"] * g["neg_inf"])
        c.MUL(f"g{gi}.out.neg", g["fin"], g["neg_fin"], out.neg - g["inf_out"] * g["neg_inf"])
        c.MUL(f"g{gi}.out.e", g["fin"], g["e_state"] - Z, out.e - Z)
        c.MUL(f"g{gi}.out.M", g["fin"], g["mag24"], out.M)
        c.LIN(f"g{gi}.out.nan", out.nan - g["nan_out"])
        c.LIN(f"g{gi}.out.inf", out.inf - g["inf_out"])
        st = out
    return st


# ---------------------------------------------------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------------------------------------------------

def census(pipe: Pipeline, align: str = ALIGN_DIV) -> dict:
    """Predicate counts of one k16 step.  Instance-independent (the relation has no conditionals); checked on three."""
    rng = random.Random(0)
    counts = None
    for _ in range(3):
        acc = rng.getrandbits(32)
        a = [rng.getrandbits(16) for _ in range(pipe.k)]
        b = [rng.getrandbits(16) for _ in range(pipe.k)]
        c = Census()
        check_step(pipe, step_witness(pipe, decode_state(acc, pipe), a, b, align), c)
        s = c.summary()
        if counts is None:
            counts = s
        elif counts != s:
            raise RuntimeError("the census depends on the instance")
    p = params(pipe)
    return {
        "pipeline": pipe.name, "align": align,
        "params": {"W": p.W, "rho": p.rho, "A": p.A, "B": p.B, "K": p.K, "floor": p.Z, "groups": list(pipe.groups)},
        "mul": counts["mul"], "lin": counts["lin"], "lookup": counts["lookup"],
        "lookups_by_table": counts["lookups_by_table"],
        "table_rows": {name: TABLES[name].size for name in counts["lookups_by_table"]},
        "per_product": {k: round(counts[k] / pipe.k, 2) for k in ("mul", "lin", "lookup")},
    }


def vu_census(pipe: Pipeline, K: int, align: str = ALIGN_DIV) -> dict:
    """One verification unit: K/16 steps plus the boundary (the initial +0 state is a constant; pack + cvt once)."""
    step = census(pipe, align)
    c = Census()
    pack_relation(State(0, 0, 0, 5, 0xC0FFEE), pipe, c)
    boundary = c.summary()
    steps = K // 16
    return {"K": K, "steps": steps, "step": step, "boundary": boundary,
            "total": {k: step[k] * steps + boundary[k] for k in ("mul", "lin", "lookup")}}
