"""The fp4-nvf4 transition unit (``BLACKWELL_SM120_NVF4.step``: one ``mma.sync kind::mxf4nvf4`` firing) as a GKR circuit for the
GPU prover, and its K = 1536 VU statement (24 chained units + an epilogue binding the three components of the public FP32 word).

The constraints are B-Ligero's (``backends/direct/ligero/fp4/relation.py``, the relation proved on the RTX 5090 row), compiled on
:class:`GkrCtx`, a ``LigeroCtx`` whose predicates become the GKR prover's native objects instead of Ligero rows:

    rng(x, lo, hi)     one ``R<b>`` LogUp query (``b <= 16``; two when ``hi - lo`` is not a power of two), or a 16-bit low limb
                       and an ``R<b-16>`` high limb bound by one assertion (``16 < b <= 30``) -- instead of bit rows
    boolean(b)         one ``R1`` query
    lookup(T, key)     one LogUp query ``(key, out...)`` into the row-listed table ``T`` (outputs are committed columns),
                       instead of one-hot selector rows; ``table_derived`` / ``lt_pow`` add (or reuse) an output column
    product            a product wire of the layered circuit (``A*B = C`` otherwise: a product wire and one assertion)
    linear constraint  one assertion over the wires

Every linear constraint keeps ``LigeroCtx``'s integer audit (the interval of each side inside ``(-p, p)``, so the field identity
is the integer identity), and each range / lookup query is audited the same way (the query's interval and the table's span
inside one residue window).  The 12-bit digits of the shifters are committed columns range-checked by ``R12`` queries and bound
to the shifted magnitude by one assertion (B-Ligero: bit rows).

**The operands.**  B-Ligero's verifier decodes the public operand words into pins (the 128 E2M1 numerators, per group the
scale-mantissa product ``sm``, the unit weight ``X``, the participation flag and the anchor ``G``).  A-GKR's statement is the VU's
public endpoints only (``c_0 = 0`` and the output word, as on the BF16 rows): the operand words are committed columns and the pins
are derived in the circuit from them --
  * per product ``i``: the pair code ``16 a_i + b_i`` (both E2M1 nibbles) looked up in ``E2M1X2`` -> ``(n_a n_b, [n_a n_b != 0])``;
  * per group ``g``: the scale pair code ``128 sa_g + sb_g`` (UE4M3 bytes, bit 7 clear, never the NaN code 0x7F) looked up in
    ``UE4M3X2`` -> ``(ma mb, -2 + ea + eb, [ma mb != 0])`` (``relation.decode_scale``);
  * ``part_g = [some product of the group != 0] * [ma mb != 0]``: ``any_g = [cnt_g != 0]`` for ``cnt_g`` the sum of the 16
    nonzero flags by an inverse hint (``cnt inv = any``, ``cnt (1 - any) = 0``), then ``part_g = any_g * nzs_g``;
  * ``G = max(-174, max_{g: part_g} X_g - 27)``: ``G - c`` in ``[0, 512)`` for every candidate ``c`` and a one-hot selector of
    the candidate it equals.
The group dot ``gd_g`` is the sum of the 16 looked-up products (no product wire).

**Chain / statement.**  ``acc.s / acc.t / acc.f`` of unit ``j + 1`` are linked to the committed outputs ``out.ys / out.t /
out.yf`` of unit ``j`` (``c_0 = 0``: init 0); the epilogue row takes the last unit's three components and ``chain.txt``'s
``public s t f`` line binds them to the three public words per VU (sign bit, exponent field, fraction: each < 2^31 < p, since a
32-bit word is not a field element -- ``relation.py`` ``y_end``).

    python -m gpu.nvf4.circuit export --out DIR      # circuit.txt, epilogue.txt, chain.txt, manifest.json
    python -m gpu.nvf4.circuit stats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from backends.direct.ligero.compile import CompileError, Expr, LigeroCtx, P, Quadratic, System
from backends.direct.ligero.fp4 import relation as R
from verity_numerical.checker.ctx import Table

__all__ = ["GkrCtx", "NVF4Statement", "build_epilogue", "build_unit", "export", "E2M1X2", "UE4M3X2"]

K, GROUP, GROUPS, RADIX, POSITIONS = R.K, R.GROUP, R.GROUPS, R.RADIX, R.POSITIONS
STEPS = R.STEPS
MODEL = "blackwell_sm120_nvf4"
RANGE_MAX = 16               # the GKR prover's implicit range tables R1 .. R16


def _pair_prod(k: int) -> tuple[int, int]:
    pr = R.numerator(k >> 4) * R.numerator(k & 15)
    return pr, int(pr != 0)


def _scale_x(k: int) -> tuple[int, int, int]:
    """``128 sa + sb`` -> ``(ma mb, X = -2 + ea + eb, [ma mb != 0])`` with ``decode_scale``'s exponents (value ``m 2^e``) shifted to
    the unit weight of ``witness.public_vectors_fp4`` (``_scales``: ``e = -9`` subnormal, else ``e_field - 10``)."""
    ma, ea = R.decode_scale(k >> 7)
    mb, eb = R.decode_scale(k & 127)
    return ma * mb, -2 + ea + eb, int(ma * mb != 0)


#: E2M1 nibble pair ``16 a + b`` -> (``n_a n_b`` in halves squared, nonzero flag): 256 rows.
E2M1X2 = Table("E2M1X2", 0, 256, ("prod", "nz"), _pair_prod)
#: UE4M3 scale pair ``128 sa + sb`` (bytes 0x00..0x7E: bit 7 clear, NaN code 0x7F excluded) -> (sm, X, nonzero flag): 127^2 rows.
UE4M3X2 = Table("UE4M3X2", 0, 128 * 127, ("sm", "X", "nz"), _scale_x, domain=lambda k: (k & 127) < 127 and (k >> 7) < 127,
                size=127 * 127)


# ---------------------------------------------------------------------------------------------------------------------------------
# The context
# ---------------------------------------------------------------------------------------------------------------------------------

@dataclass
class NatLookup:
    """One native lookup: ``(key, cols...)`` of every unit is a row of ``table`` (rows ``(k, v_1(k), ...)`` over ``keys``)."""
    table: str
    keys: list[int]
    key: Expr                                                     # over columns
    cols: list[tuple[list[int], int]] = field(default_factory=list)   # (value per key, output row)

    def content(self) -> tuple:
        return (self.table, tuple(self.keys), tuple(tuple(v) for v, _ in self.cols))


class NatOut(Expr):
    """A lookup output column; remembers the per-key values so that ``lt_pow`` can derive ``2^bits / value``."""

    __slots__ = ("keys", "rec", "values")

    def __init__(self, ctx, idx: int, rec: NatLookup, values: list[int]):
        super().__init__(ctx, {idx: 1}, 0)
        self.rec, self.values, self.keys = rec, list(values), list(rec.keys)


class GkrCtx(LigeroCtx):
    """``LigeroCtx`` with the GKR prover's native lookups and range checks (module docstring)."""

    def __init__(self, p):
        super().__init__(p)
        self.nat: list[NatLookup] = []
        self.ranges: list[tuple[int, Expr]] = []                  # (width, expression over columns)

    # --- rows the witness generator derives ---------------------------------------------------------------------------------
    def _derived(self, kind: str, name: str, lo: int, hi: int, op: tuple) -> Expr:
        r = self._row(kind, name, lo, hi, op)
        idx = next(iter(r.lin))
        self.sys.program.append((kind, idx, *op))
        return r

    def is_col_expr(self, e: Expr) -> bool:
        return all(self.sys.rows[j].kind != "prod" for j in e.lin)

    def col(self, name: str, e: Expr) -> Expr:
        """``e`` as one committed column: ``e`` itself when it is a column, else an ``aux`` column bound by ``aux - e = 0``."""
        if len(e.lin) == 1 and e.c == 0 and next(iter(e.lin.values())) == 1 and self.is_col_expr(e):
            return e
        iv = self.interval(e)
        if iv is None:
            raise CompileError(f"{self.scope}.{name}: aux column of an expression with unknown interval: {e!r}")
        if iv[1] - iv[0] >= P:
            raise CompileError(f"{self.scope}.{name}: aux interval {iv} spans p")
        a = self._derived("aux", name, iv[0], iv[1] + 1, (e,))
        self._linear(f"{name}.def", a - e)
        return a

    def _colform(self, e: Expr, name: str) -> Expr:
        return e if self.is_col_expr(e) else self.col(name, e)

    def _define(self, e: Expr, lo: int, hi: int) -> None:
        """``LigeroCtx._bits``' rule: the first range check of a hint row defines its integer (``e`` in ``[lo, hi)``)."""
        unknown = [j for j in e.lin if self.sys.rows[j].lo is None]
        if len(unknown) == 1 and abs(e.lin[unknown[0]]) == 1:
            j = unknown[0]
            rest = Expr(self, {k: v for k, v in e.lin.items() if k != j}, e.c)
            iv = self.interval(rest)
            if iv is not None:
                sgn = e.lin[j]
                cands = [sgn * (lo - iv[1]), sgn * (lo - iv[0]), sgn * (hi - 1 - iv[1]), sgn * (hi - 1 - iv[0])]
                self.sys.rows[j].lo, self.sys.rows[j].hi = min(cands), max(cands) + 1

    def _audit_query(self, name: str, e: Expr, lo: int, hi: int) -> None:
        """A query ``e in S`` with ``S`` inside ``[lo, hi)`` is the integer statement when ``e``'s interval and ``[lo, hi)`` lie in
        one residue window (the decomposition audit ``_bits`` makes, for a lookup)."""
        iv = self.interval(e)
        if iv is None:
            raise CompileError(f"{self.scope}.{name}: query of an expression with unknown interval: {e!r}")
        if max(iv[1], hi - 1) - min(iv[0], lo) >= P:
            raise CompileError(f"{self.scope}.{name}: query interval {iv} vs table [{lo}, {hi}) spans p")

    def _range(self, name: str, e: Expr, b: int) -> None:
        """``e`` in ``[0, 2^b)`` by lookups."""
        self._define(e, 0, 2**b)
        self._audit_query(name, e, 0, 2**b)
        if b <= RANGE_MAX:
            self.ranges.append((b, self._colform(e, name)))
        else:
            lo = self._derived("dig", f"{name}.lo", 0, 2**RANGE_MAX, (e, 0, RANGE_MAX))
            hi = self._derived("dig", f"{name}.hi", 0, 2 ** (b - RANGE_MAX), (e, RANGE_MAX, b - RANGE_MAX))
            self.ranges.append((RANGE_MAX, lo))
            self.ranges.append((b - RANGE_MAX, hi))
            self._linear(f"{name}.limbs", e - lo - hi * 2**RANGE_MAX)
        self._learn(e, 0, 2**b)

    def digit(self, name: str, x: Expr, c: int, w: int) -> Expr:
        d = self._derived("dig", name, 0, 2**w, (x, c, w))
        self.ranges.append((w, d))
        return d

    # --- predicates --------------------------------------------------------------------------------------------------------
    def rng(self, name: str, x, lo: int, hi: int) -> None:
        if isinstance(x, int):
            return super().rng(name, x, lo, hi)
        y = x - lo
        N = hi - lo
        if N <= 0:
            raise CompileError(f"{name}: empty range")
        if self.known_within(y, 0, N):
            return
        b = max(1, math.ceil(math.log2(N)))
        if b > 30 or (N != 2**b and b > RANGE_MAX):
            return super().rng(name, x, lo, hi)
        self._range(name, y, b)
        if N != 2**b:
            self._range(name + ".upper", (N - 1) - y, b)
        self._learn(y, 0, N)
        self._learn(x, lo, hi)

    def boolean(self, name: str, b) -> None:
        if isinstance(b, int) or self.known_within(b, 0, 2):
            return super().boolean(name, b)
        self._range(name, b, 1)

    def lookup(self, name: str, table: Table, key, range_key: bool = True) -> tuple:
        if isinstance(key, int):
            return table.row(key)
        memo = (table.name, key.key())
        if memo in self._lookup_memo:
            return self._lookup_memo[memo]
        keys = [k for k in range(table.lo, table.hi) if table.domain is None or table.domain(k)]
        rows = [table.fn(k) for k in keys]
        self._define(key, min(keys), max(keys) + 1)
        self._audit_query(f"{name}.key", key, min(keys), max(keys) + 1)
        rec = NatLookup(table.name, keys, self._colform(key, f"{name}.key"))
        self.nat.append(rec)
        self._learn(key, min(keys), max(keys) + 1)
        outs = tuple(self._natcol(rec, f"{name}.{o}", [r[i] for r in rows]) for i, o in enumerate(table.outs))
        self._lookup_memo[memo] = outs
        return outs

    def _natcol(self, rec: NatLookup, name: str, values: list[int]) -> NatOut:
        for vals, idx in rec.cols:
            if vals == values:
                return NatOut(self, idx, rec, vals)
        if max(values) - min(values) >= P:
            raise CompileError(f"{name}: table column spans p")
        r = self._row("lk", name, min(values), max(values) + 1, (rec, len(rec.cols)))
        idx = next(iter(r.lin))
        rec.cols.append((list(values), idx))
        self.sys.program.append(("lk", idx, rec, len(rec.cols) - 1))
        return NatOut(self, idx, rec, values)

    def table_derived(self, out, fn):
        if isinstance(out, NatOut):
            return self._natcol(out.rec, f"{self.sys.rows[next(iter(out.lin))].name}.d", [fn(v) for v in out.values])
        return super().table_derived(out, fn)

    def table_by_key(self, out, fn):
        if isinstance(out, NatOut):
            return self._natcol(out.rec, f"{self.sys.rows[next(iter(out.lin))].name}.k", [fn(k) for k in out.keys])
        return super().table_by_key(out, fn)

    def lt_pow(self, name: str, x, ipow, bits: int) -> None:
        if isinstance(ipow, NatOut):
            for v in ipow.values:
                if v <= 0 or (2**bits) % v:
                    raise CompileError(f"{name}: {v} does not divide 2^{bits}")
            return self.lt_table(name, x, self.table_derived(ipow, lambda v: 2**bits // v), bits)
        return super().lt_pow(name, x, ipow, bits)

    def pin(self, name: str, lo: int, hi: int) -> Expr:
        raise CompileError(f"{name}: the A-GKR statement has no pins (operands are committed; module docstring)")


# ---------------------------------------------------------------------------------------------------------------------------------
# The unit (relation.compile_fp4_unit, chain mode, with the decode in the circuit and digits as columns)
# ---------------------------------------------------------------------------------------------------------------------------------

def _digits(ctx: GkrCtx, x: Expr, nbits: int, name: str) -> list[Expr]:
    """x < 2^nbits as RADIX-bit digit columns (``R12`` / ``R<w>`` queries) bound by one assertion."""
    out, acc = [], Expr(ctx)
    for c in range(0, nbits, RADIX):
        d = ctx.digit(f"{name}.d{c // RADIX}", x, c, min(RADIX, nbits - c))
        out.append(d)
        acc = acc + d * 2**c
    ctx.eq(f"{name}.digits", x, acc)
    ctx._learn(x, 0, 2**nbits)
    return out


def _shifter(ctx: GkrCtx, path: str, d: Expr, *, bits: int, n_j: int, sign: Expr | None, exact: bool, dead: bool):
    """``relation._shifter`` with the digits of :func:`_digits` (same hints, same constraints otherwise)."""
    _hint = R._hint
    sc, ctx.scope = ctx.scope, path
    u = _hint(ctx, f"{path}.u")
    pw, cpw = ctx.lookup("u", R._pow_table(bits), u)
    if bits <= R.MAG_BITS:
        q = _hint(ctx, f"{path}.q")
        ctx.lt_pow("q", q, pw, bits)
        q_parts = [(q, bits)]
    else:
        qh, ql = _hint(ctx, f"{path}.qh"), _hint(ctx, f"{path}.ql")
        ctx.rng("qh", qh, 0, 2 ** (bits - RADIX))
        ctx.rng("ql", ql, 0, 2**RADIX)
        q = qh * 2**RADIX + ql
        ctx.lt_pow("q", q, pw, bits)
        q_parts = [(ql, RADIX), (qh, bits - RADIX)]
    qpw = q * pw
    if exact:
        mag = qpw
    else:
        rem = _hint(ctx, f"{path}.rem")
        ctx.lt_pow("rem", rem, cpw, bits)
        mag = qpw + rem
    rlo, rhi = _hint(ctx, f"{path}.rlo"), _hint(ctx, f"{path}.rhi")
    pw_lo, cpw_lo = ctx.lookup("rlo", R._R_LO, rlo)
    pw_hi, cpw_hi = ctx.lookup("rhi", R._R_HI, rhi)
    r = rlo + 4 * rhi
    sels = [_hint(ctx, f"{path}.sel{j}") for j in range(n_j)]
    for j, s in enumerate(sels):
        ctx.boolean(f"sel{j}", s)
    ctx.eq("onehot", sum(sels[1:], sels[0]), 1)
    jval = sum((j * s for j, s in enumerate(sels[1:], 1)), Expr(ctx))
    L = RADIX * jval + r
    ctx.eq("lu", L * u, 0)
    if dead:
        dd, k = _hint(ctx, f"{path}.dead"), _hint(ctx, f"{path}.k")
        ctx.boolean("dead", dd)
        ctx.rng("k", k - 1, 0, 2**R.DEAD_BITS)
        ctx.eq("dead.u", dd * (L + bits - u), 0)
        ctx.eq("d", d, L - u - dd * k)
    else:
        ctx.eq("d", d, L - u)
    pre: dict[int, list[Expr]] = {}
    for part, (qp, pbits) in enumerate(q_parts):
        x1 = qp * pw_lo
        x2 = x1 * pw_hi
        digs = _digits(ctx, x2, pbits + 11, f"{path}.x{part}")
        for c, e in enumerate(digs):
            pre.setdefault(c + part, []).append(e)
    ssels = [ctx.product(1 - 2 * sign, s) for s in sels] if sign is not None else sels
    placed: dict[int, list[Expr]] = {}
    for j, s in enumerate(ssels):
        for c, es in pre.items():
            for e in es:
                placed.setdefault(c + j, []).append(ctx.product(s, e))
    ctx.scope = sc
    return mag, L, placed, sels, (pw_lo, cpw_lo, pw_hi, cpw_hi)


def _decode(ctx: GkrCtx):
    """The pins of ``relation.compile_fp4_unit`` derived from committed operand codes (module docstring): per group the list of
    16 products, ``sm``, ``X``, ``part``; and ``G``."""
    _hint = R._hint
    ctx.scope = "x"
    prods, nzs = [], []
    for i in range(K):
        pc = _hint(ctx, f"x.p{i}")
        pr, nz = ctx.lookup(f"p{i}", E2M1X2, pc)
        prods.append(pr)
        nzs.append(nz)
    sm, X, part, cands = [], [], [], []
    for g in range(GROUPS):
        ctx.scope = f"x.g{g}"
        sc = _hint(ctx, f"x.s{g}")
        smg, Xg, nzs_g = ctx.lookup("s", UE4M3X2, sc)
        cnt = sum(nzs[GROUP * g + 1: GROUP * (g + 1)], nzs[GROUP * g])
        anyg = _hint(ctx, f"x.g{g}.any")
        inv = ctx._derived("invz", f"x.g{g}.inv", None, None, (cnt,))   # 1/cnt, or 0 when cnt = 0
        ctx.boolean("any", anyg)
        ctx.sys.quadratic.append(Quadratic(f"x.g{g}.any.inv", cnt, inv, anyg))   # field: cnt = 0 -> any = 0
        ctx.eq("any.zero", cnt * (1 - anyg), 0)              # integers (cnt in [0, 16]): cnt != 0 -> any = 1
        pg = _hint(ctx, f"x.g{g}.part")
        ctx.boolean("part", pg)
        ctx.eq("part", pg, anyg * nzs_g)
        sm.append(smg)
        X.append(Xg)
        part.append(pg)
        cands.append(R.GRID_FLOOR + pg * (Xg - R.FRAC_BITS - R.GRID_FLOOR))   # X - 27 when participating, else -174
    ctx.scope = "x.G"
    G = _hint(ctx, "x.G")
    ctx.rng("floor", G - R.GRID_FLOOR, 0, 512)
    cands.append(Expr(ctx, {}, R.GRID_FLOOR))
    sel = [_hint(ctx, f"x.G.sel{g}") for g in range(GROUPS + 1)]
    for g in range(GROUPS + 1):
        if g < GROUPS:
            ctx.rng(f"ge{g}", G - cands[g], 0, 512)
        ctx.boolean(f"sel{g}", sel[g])
    ctx.eq("onehot", sum(sel[1:], sel[0]), 1)
    ctx.eq("max", sum((sel[g] * (G - cands[g]) for g in range(1, GROUPS + 1)), sel[0] * (G - cands[0])), 0)
    return prods, sm, X, part, G


def build_unit(p=R.NVFP4) -> GkrCtx:
    """The chained unit on :class:`GkrCtx`: ``relation.compile_fp4_unit(chain=True)`` line for line, the pins replaced by
    :func:`_decode`, then the committed outputs ``out.ys`` / ``out.yf`` for the chain links."""
    _hint = R._hint
    ctx = GkrCtx(p)
    ctx.whitelist_product("shift", "lt_pow(q, pw, bits) gives q < 2^bits / pw, so q * pw < 2^bits", 0, 2**R.ACC_BITS)

    ctx.scope = "acc"
    s_c, t_c, f_c = _hint(ctx, "acc.s"), _hint(ctx, "acc.t"), _hint(ctx, "acc.f")
    nz, z, fh, fl = (_hint(ctx, f"acc.{n}") for n in ("nz", "z", "fh", "fl"))
    ctx.boolean("s", s_c)
    ctx.boolean("nz", nz)
    ctx.boolean("z", z)
    ctx.rng("t.nz1", t_c - nz, 0, R.E_FIELD_MAX)
    ctx.eq("t.nz0", (1 - nz) * t_c, 0)
    ctx.rng("fh", fh, 0, 2 ** (23 - RADIX))
    ctx.rng("fl", fl, 0, 2**RADIX)
    ctx._learn(f_c, 0, 2**23)                                 # the linked row IS the integer fh 2^12 + fl (next constraint)
    ctx.eq("f.digits", f_c, fh * 2**RADIX + fl)
    Mh, Ml = nz * 2 ** (R.ACC_BITS - 1 - RADIX) + fh, fl
    M_c = Mh * 2**RADIX + Ml
    ctx.eq("z0", z * M_c, 0)
    ctx.inverse_nonzero("z1", M_c + z)
    tp = t_c + 1 - nz
    acc_cand = tp - R.ACC_CAND_OFF + (R.GRID_FLOOR - 1 + R.ACC_CAND_OFF) * z
    acc_e = tp - R.ACC_E_OFF

    prods, sm, X, part, G = _decode(ctx)

    ctx.scope = "anc"
    lsb = _hint(ctx, "anc.lsb")
    ctx.rng("geG", lsb - G, 0, 512)
    ctx.rng("geA", lsb - acc_cand, 0, 512)
    ctx.eq("max", (lsb - G) * (lsb - acc_cand), 0)

    P_sum: dict[int, list[Expr]] = {}
    for g in range(GROUPS):
        ctx.scope = f"g{g}"
        gp = prods[GROUP * g: GROUP * (g + 1)]
        gd = sum(gp[1:], gp[0])
        mant = gd * sm[g]
        sgn = _hint(ctx, f"g{g}.sgn")
        ctx.boolean("sgn", sgn)
        d = part[g] * (X[g] - lsb)
        mag, _L, placed, _sels, _fine = _shifter(ctx, f"g{g}.shift", d, bits=R.MAG_BITS, n_j=3, sign=sgn, exact=False, dead=True)
        ctx.eq("mag", mant, (1 - 2 * sgn) * mag)
        for pos, es in placed.items():
            P_sum.setdefault(pos, []).extend(es)

    ctx.scope = "accterm"
    d_acc = (1 - z) * (acc_e - lsb)
    mag_a, _L, placed, _sels, _fine = _shifter(ctx, "accterm.shift", d_acc, bits=R.ACC_BITS, n_j=2, sign=s_c, exact=False, dead=True)
    ctx.eq("mag", M_c, mag_a)
    for pos, es in placed.items():
        P_sum.setdefault(pos, []).extend(es)

    ctx.scope = "out"
    sgn_t, t_o, hid, fh_o, fl_o, nzM = (_hint(ctx, f"out.{n}") for n in ("sgn", "t", "hid", "fh", "fl", "nzM"))
    ctx.boolean("sgn", sgn_t)
    ctx.boolean("hid", hid)
    ctx.boolean("nzM", nzM)
    ctx.rng("t.hid1", t_o - hid, 0, R.E_FIELD_MAX)
    ctx.eq("t.hid0", (1 - hid) * t_o, 0)
    ctx.rng("fh", fh_o, 0, 2 ** (23 - RADIX))
    ctx.rng("fl", fl_o, 0, 2**RADIX)
    Mh_o, Ml_o = hid * 2 ** (R.ACC_BITS - 1 - RADIX) + fh_o, fl_o
    M_o = Mh_o * 2**RADIX + Ml_o
    ctx.eq("nzM0", (1 - nzM) * M_o, 0)
    ctx.inverse_nonzero("nzM1", M_o + (1 - nzM))
    tp_o = t_o + 1 - hid
    sigma = nzM * (tp_o - R.ACC_E_OFF - lsb)
    mag_o, L_o, placed, sels_o, (pw_lo, cpw_lo, pw_hi, cpw_hi) = _shifter(
        ctx, "out.shift", sigma, bits=R.ACC_BITS, n_j=4, sign=None, exact=True, dead=False)
    ctx.eq("mag", M_o, mag_o)
    n_r = len(sels_o)
    rdig = [_hint(ctx, f"out.r{pn}") for pn in range(n_r)]
    for pn, rd in enumerate(rdig):
        ctx.rng(f"r{pn}", rd, 0, 2**RADIX)
    above = Expr(ctx)
    for pn in range(1, n_r):
        mask = sum(sels_o[:pn], Expr(ctx))
        above = above + rdig[pn] * mask
    ctx.eq("r.above", above, 0)
    r_sel = sum((sels_o[pn] * rdig[pn] for pn in range(1, n_r)), sels_o[0] * rdig[0])
    t1 = r_sel * cpw_lo
    t2 = t1 * cpw_hi
    ctx.rng("r.top", t2, 0, 2**RADIX)
    carries = [Expr(ctx)]
    for pn in range(1, POSITIONS):
        c = _hint(ctx, f"out.c{pn}")
        ctx.rng(f"c{pn}", c + 16, 0, 32)
        carries.append(c)
    carries.append(Expr(ctx))
    for pn in range(POSITIONS):
        P_p = sum(P_sum.get(pn, []), Expr(ctx))
        sP = (1 - 2 * sgn_t) * P_p if P_p.lin else Expr(ctx)
        DM = sum(placed.get(pn, []), Expr(ctx))
        rp = rdig[pn] if pn < n_r else Expr(ctx)
        ctx.eq(f"carry{pn}", sP - DM - rp + carries[pn] - 2**RADIX * carries[pn + 1], 0)
    s_out = sgn_t * nzM
    ctx.scope = "y"
    ys = ctx.col("out.ys", s_out)
    yf = ctx.col("out.yf", fh_o * 2**RADIX + fl_o)
    ctx.sys.outputs = {"y.s": ys, "y.t": t_o, "y.f": yf}
    ctx.sys.chain = {"c": ["acc.s", "acc.t", "acc.f"], "y": ["out.ys", "out.t", "out.yf"]}
    return ctx


def build_epilogue(p=R.NVFP4) -> GkrCtx:
    """One row per VU: the last unit's ``s, t, f`` (chain links) = the three public words.  ``s`` boolean (a product wire, so the
    circuit has a layer) and ``t`` in ``R8`` restate facts the unit already proves."""
    ctx = GkrCtx(p)
    ctx.scope = "e"
    s, t, f = (R._hint(ctx, n) for n in ("s", "t", "f"))
    ctx.sys.quadratic.append(Quadratic("e.s.bool", s, s, s))
    ctx._learn(s, 0, 2)
    ctx.rng("t", t, 0, 256)
    ctx._learn(f, 0, 2**23)
    return ctx


# ---------------------------------------------------------------------------------------------------------------------------------
# Export (the text format of backends/gkr/src/circuit.rs; gpu/circuit.py parse_circuit)
# ---------------------------------------------------------------------------------------------------------------------------------

@dataclass
class Exported:
    text: str
    col_rows: list[int]              # column c (c >= 1) <- System row col_rows[c]; column 0 is the constant one
    col_names: list[str]
    tables: dict[str, NatLookup]     # table name -> a representative lookup (its rows)
    stats: dict


def _tok(name: str) -> str:
    return name.replace(" ", "_")


def export_circuit(ctx: GkrCtx) -> Exported:
    sys_: System = ctx.sys
    rows = sys_.rows
    col_of: dict[int, int] = {}
    col_rows = [-1]
    names = ["one"]
    for r in rows:
        if r.kind != "prod":
            col_of[r.idx] = len(col_rows)
            col_rows.append(r.idx)
            names.append(_tok(r.name))
    ncols = len(col_rows)
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise CompileError(f"duplicate column names: {dup[:5]}")
    wire_of = dict(col_of)
    depth = {w: 0 for w in range(ncols)}
    prods: list[tuple[int, int, str, str]] = []
    nw = ncols

    def lin_terms(e: Expr, cols_only: bool = False) -> dict[int, int]:
        """``e`` over wires (or columns), the constant on the one column 0, coefficients mod p."""
        terms: dict[int, int] = {}
        for j, v in e.lin.items():
            w = col_of[j] if cols_only else wire_of[j]
            terms[w] = (terms.get(w, 0) + v) % P
        if e.c % P:
            terms[0] = (terms.get(0, 0) + e.c) % P
        return {w: c for w, c in terms.items() if c}

    def fmt(terms: dict[int, int]) -> str:
        return f"{len(terms)} " + "".join(f"{w} {c} " for w, c in sorted(terms.items())) + "0"

    def lin_txt(e: Expr, cols_only: bool = False) -> str:
        return fmt(lin_terms(e, cols_only))

    def dep(e: Expr) -> int:
        return max((depth[wire_of[j]] for j in e.lin), default=0)

    def new_prod(a: Expr, b: Expr) -> int:
        nonlocal nw
        w = nw
        nw += 1
        depth[w] = 1 + max(dep(a), dep(b))
        prods.append((w, depth[w], lin_txt(a), lin_txt(b)))
        return w

    for r in rows:
        if r.kind == "prod":
            a, b = r.spec
            wire_of[r.idx] = new_prod(a, b)
    asserts: list[str] = []
    for lin in sys_.linear:
        if lin.pub is not None:
            raise CompileError(f"linear {lin.name}: public vectors are not part of the A-GKR statement")
        asserts.append(f"assert {_tok(lin.name)} {lin_txt(Expr(ctx, lin.lin, lin.c))}")
    for q in sys_.quadratic:
        c = q.c
        if len(c.lin) == 1 and c.c == 0 and rows[next(iter(c.lin))].kind == "prod" and next(iter(c.lin.values())) == 1 \
                and rows[next(iter(c.lin))].spec == (q.a, q.b):
            continue
        w = new_prod(q.a, q.b)                                   # a * b - c = 0
        terms = lin_terms(-c)
        terms[w] = (terms.get(w, 0) + 1) % P
        asserts.append(f"assert {_tok(q.name)} {fmt(terms)}")
    # tables: identical contents share one table; distinct contents under one base name get a suffix
    tables: dict[str, NatLookup] = {}
    by_content: dict[tuple, str] = {}
    queries: list[str] = []
    for rec in ctx.nat:
        ck = rec.content()
        tname = by_content.get(ck)
        if tname is None:
            tname = rec.table if rec.table not in tables else f"{rec.table}_{len(tables)}"
            if tname.startswith("R") and tname[1:].isdigit():
                tname = "T" + tname
            by_content[ck] = tname
            tables[tname] = rec
        outs = [Expr(ctx, {idx: 1}, 0) for _, idx in rec.cols]
        queries.append(f"query {tname} {1 + len(outs)} " + " ".join(lin_txt(e, True) for e in (rec.key, *outs)))
    for b, e in ctx.ranges:
        queries.append(f"query R{b} 1 {lin_txt(e, True)}")
    lines = [f"ncols {ncols}", f"nwires {nw}", "one 0"] + [f"col {i} {n}" for i, n in enumerate(names)]
    lines += [f"prod {w} {d} {a} {b}" for w, d, a, b in prods]
    lines += asserts
    for tname, rec in tables.items():
        lines.append(f"table {tname} {1 + len(rec.cols)} {len(rec.keys)}")
        for i, k in enumerate(rec.keys):
            lines.append("row " + " ".join(str(v % P) for v in (k, *(vals[i] for vals, _ in rec.cols))))
    lines += queries
    nq: dict[str, int] = {}
    for q in queries:
        t = q.split()[1]
        nq[t] = nq.get(t, 0) + 1
    stats = {"columns": ncols, "wires": nw, "products": len(prods), "asserts": len(asserts),
             "max_depth": max((d for _, d, _, _ in prods), default=0), "queries": nq, "tables": {t: len(r.keys) for t, r in tables.items()},
             "kinds": sys_.summary()["by_kind"]}
    return Exported("\n".join(lines) + "\n", col_rows, names, tables, stats)


@dataclass
class NVF4Statement:
    unit: GkrCtx
    epi: GkrCtx
    ue: Exported
    ee: Exported


def statement() -> NVF4Statement:
    u, e = build_unit(), build_epilogue()
    return NVF4Statement(u, e, export_circuit(u), export_circuit(e))


def export(out: Path) -> dict:
    st = statement()
    out.mkdir(parents=True, exist_ok=True)
    (out / "circuit.txt").write_text(st.ue.text)
    (out / "epilogue.txt").write_text(st.ee.text)
    chain = "".join(f"link {c} {y} {e} 0 0\n" for c, y, e in zip(("acc.s", "acc.t", "acc.f"), ("out.ys", "out.t", "out.yf"), ("s", "t", "f")))
    chain += "public s t f\n"
    (out / "chain.txt").write_text(chain)
    man = {"field": "babybear", "variant": "v2", "relation": "fp4-nvf4", "model": MODEL, "modulus": P, "steps": STEPS,
           "unit_columns": st.ue.stats["columns"], "epilogue_columns": st.ee.stats["columns"], "chain_file": "chain.txt",
           "chain_sha256": hashlib.sha256(chain.encode()).hexdigest(), "public_columns": ["s", "t", "f"],
           "circuit_sha256": hashlib.sha256(st.ue.text.encode()).hexdigest(),
           "epilogue_sha256": hashlib.sha256(st.ee.text.encode()).hexdigest(),
           "source": "backends/gkr/gpu/nvf4/circuit.py (B-Ligero fp4-nvf4 constraints on GkrCtx)",
           "unit_stats": st.ue.stats, "epilogue_stats": st.ee.stats}
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    return man


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gpu.nvf4.circuit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", type=Path, required=True)
    sub.add_parser("stats")
    args = ap.parse_args(argv)
    if args.cmd == "export":
        m = export(args.out)
        print(json.dumps({k: v for k, v in m.items() if k not in ("unit_stats", "epilogue_stats")}))
        print(json.dumps(m["unit_stats"]))
    else:
        st = statement()
        print(json.dumps(st.ue.stats))
        print(json.dumps(st.ee.stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
