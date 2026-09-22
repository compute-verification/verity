"""Checker v2 as a B-Ligero constraint system (lane b-e2e-v2): the a-v2 lane's ``CircuitCtx`` of the state-chained
unit (``gkr_export.v2``: the same ``check_unit_expr`` gadgets, no predicate added or dropped) plus ``Pack`` and the
epilogue on every unit's outgoing state, laid out as V2_SHAPE.md: **one Ligero column per unit, one row per AIR
column / product wire / helper coordinate** (the transposed layout; ``blocks = 1`` here, ``l`` units per proof).

Tableau 1 (round 1): the circuit's columns, then the product wires, then the multiplicity rows of every table
(``ceil(|T| / l)`` rows per table, entry ``v`` at row ``v // l``, position ``v % l``).
Tableau 2 (round 2, after ``z_T, beta_T``): the lookup helpers ``h_j`` (6 coordinate rows per query, query-major) then
the table-side inverses ``h_v`` (per table: coordinate-major, ``6 x ceil(|T| / l)`` rows).

Lookups are logUp with helper columns (PROTOCOL.md 4.2 / 14): the query tuple ``(key, out_1, ..)`` is compressed as
``w_j(beta) = sum_c beta^c w_{j,c}`` with the table's ``(key, outs)`` in the same order; the helper identity
``h_j (z - w_j(beta)) = ACTIVE`` and the table-side identity ``h_v (z - t_v(beta)) = m_v ACT_T`` are quadratic
constraints over ``F_c`` (public position-varying rows ``ACTIVE``, ``ACT_T``, ``z - t_v(beta)``); the per-table sum
``sum_j h_j = sum_v h_v`` is a linear functional with constant coefficients (the structured ``ONES`` class).

Everything is evaluated mod BabyBear.  The circuit was verified over Goldilocks (a-v2); ``integer_audit`` reports,
on honest units, which assertions / products have intermediate integer values outside ``(-p, p)`` -- the word-class
rows whose BabyBear repair (checker_min.md 3.2, lane b-air-v2) is NOT built here.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field

import numpy as np
from verity_numerical.checker import Reject
from verity_numerical.checker.ctx import IntCtx, Table
from verity_numerical.checker.gadgets import Epilogue
from verity_numerical.checker.params import REAL, Params
from verity_numerical.checker.v2.gadgets import Pack
from verity_numerical.checker.v2.params import V2
from verity_numerical.checker.v2.step import check_unit_expr, gen_unit, zero_state
from verity_numerical.gkr_export.circuit import CircuitCtx, Lin, _get, _leaves, _set, range_table_width
from verity_numerical.gkr_export.v2 import _col_of, derive

from ..field import P

__all__ = ["Layout", "TableData", "V2System", "build_system", "dummy_unit_vals", "unit_vals", "vu_units"]

STEP = 16
STATE = ("s", "e", "M", "z")


# ---------------------------------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class TableData:
    name: str
    entries: np.ndarray          # (size, C) int64, reduced mod p; column 0 = key
    index_of_key: np.ndarray     # (hi - lo,) entry index of key (key - lo), -1 if excluded
    lo: int

    @property
    def size(self) -> int:
        return int(self.entries.shape[0])

    @property
    def C(self) -> int:
        return int(self.entries.shape[1])


def _materialise(t: Table, p: Params) -> TableData:
    lo, hi = t.lo, t.hi
    keys = np.arange(lo, hi, dtype=np.int64)
    R = 2 * V2.dt_rows(p)
    if t.name in ("SHIFT", "SSHIFT_HI", "SSHIFT_LO"):          # (x, dt, s) -> (1 - 2 s) floor(x 2^scale / 2^dt)
        scale = {"SHIFT": p.prod_shift, "SSHIFT_HI": V2.mb(p) + p.acc_shift, "SSHIFT_LO": p.acc_shift}[t.name]
        x, rem = np.divmod(keys, R)
        dt, s = np.divmod(rem, 2)
        v = (x << scale) >> dt
        outs = [np.where(s == 1, -v, v)]
        dom = np.ones(len(keys), dtype=bool)
    elif t.name == "TNORM":
        vb = V2.vb(p)
        koff, v_lo = np.divmod(keys, 2**vb)
        k = koff + p.k_lo
        outs = [np.where(k >= 0, v_lo << np.maximum(k, 0), v_lo >> np.maximum(-k, 0))]
        dom = np.ones(len(keys), dtype=bool)
    else:
        rows, dom = [], np.zeros(len(keys), dtype=bool)
        for i, key in enumerate(range(lo, hi)):
            if t.domain is not None and not t.domain(key):
                continue
            dom[i] = True
            rows.append(t.fn(key))
        arr = np.array(rows, dtype=np.int64).reshape(len(rows), len(t.outs))
        outs = [arr[:, j] for j in range(len(t.outs))]
        keys = keys[dom]
    entries = np.stack([keys, *outs], axis=1) % P
    index = np.full(hi - lo, -1, dtype=np.int64)
    index[np.nonzero(dom)[0]] = np.arange(int(dom.sum()))
    return TableData(t.name, entries, index, lo)


def _range_table(w: int) -> TableData:
    keys = np.arange(2**w, dtype=np.int64)
    return TableData(f"R{w}", keys[:, None] % P, keys.copy(), 0)


# ---------------------------------------------------------------------------------------------------------------------
# the system
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Layout:
    """Row indices of the two tableaux for one ``l``."""
    l: int
    n_cols: int
    n_prods: int
    mult_base: dict = field(default_factory=dict)      # table -> first multiplicity row (tableau 1)
    mult_rows: dict = field(default_factory=dict)      # table -> ceil(size / l)
    m1: int = 0                                        # rows of tableau 1
    helper_base: int = 0                               # tableau-2 row offset (absolute) of lookup 0's coordinate 0
    hv_base: dict = field(default_factory=dict)        # table -> absolute first row of its h_v block (6 x rows, coordinate-major)
    m2: int = 0
    m: int = 0

    def helper_row(self, j: int, c: int) -> int:
        return self.helper_base + 6 * j + c

    def hv_row(self, table: str, c: int, r: int) -> int:
        return self.hv_base[table] + c * self.mult_rows[table] + r

    def mult_row(self, table: str, r: int) -> int:
        return self.mult_base[table] + r


@dataclass
class V2System:
    p: Params
    ctx: CircuitCtx
    hint_paths: list[str]
    epi_paths: list[str]
    col_idx: dict                     # name -> column index for the named columns
    row_of_wire: list[int]            # wire -> tableau-1 row
    n_cols: int
    n_prods: int
    asserts: list                     # (name, {row: coef}, const) mod p
    prods: list                       # (name, ({row: coef}, const), ({row: coef}, const), out_row)
    lookups: list                     # (table_name, [({row: coef}, const), ...])  key first
    tables: dict                      # name -> TableData (in a fixed order = list(tables))
    table_order: list
    system_id: bytes = b""
    pad: list | None = None           # the pad unit's column values (default: dummy_unit_vals)

    # -- layout -------------------------------------------------------------------------------------------------------
    def layout(self, l: int) -> Layout:
        lay = Layout(l, self.n_cols, self.n_prods)
        r = self.n_cols + self.n_prods
        for name in self.table_order:
            rows = -(-self.tables[name].size // l)
            lay.mult_base[name], lay.mult_rows[name] = r, rows
            r += rows
        lay.m1 = r
        lay.helper_base = r
        r += 6 * len(self.lookups)
        for name in self.table_order:
            lay.hv_base[name] = r
            r += 6 * lay.mult_rows[name]
        lay.m = r
        lay.m2 = r - lay.m1
        return lay

    def lin_rows(self, x: Lin) -> tuple[dict, int]:
        d: dict[int, int] = {}
        for w, c in x.terms.items():
            r = self.row_of_wire[w]
            d[r] = (d.get(r, 0) + c) % P
        return {r: c for r, c in d.items() if c}, x.const % P

    def n_lookups_of(self, table: str) -> int:
        return sum(1 for t, _ in self.lookups if t == table)

    def describe(self) -> str:
        lines = [f"cols {self.n_cols} prods {self.n_prods}", " ".join(self.ctx.cols)]
        for name, coef, c in self.asserts:
            lines.append(f"assert {name} {sorted(coef.items())} {c}")
        for name, a, b, out in self.prods:
            lines.append(f"prod {name} {sorted(a[0].items())} {a[1]} {sorted(b[0].items())} {b[1]} {out}")
        for t, lins in self.lookups:
            lines.append(f"query {t} " + " ".join(f"{sorted(cf.items())} {c}" for cf, c in lins))
        for name in self.table_order:
            td = self.tables[name]
            lines.append(f"table {name} {td.size} {td.C} {hashlib.sha256(td.entries.tobytes()).hexdigest()}")
        return "\n".join(lines)


def pack_limbs_gen(p: Params, state, ovfs) -> dict:
    """Hints of the limb form of Pack + the epilogue: ``M = M_hi8 2^16 + M_lo16`` and ``Epilogue.gen`` of the word."""
    L = p.acc_frac - p.frac
    M = int(state[2])
    y32 = Pack.check(IntCtx(), p, tuple(int(v) for v in state), ovfs)
    h = {"M_hi8": M >> L, "M_lo16": M & (2**L - 1)}
    h.update(Epilogue.gen(p, y32))
    return h


def pack_limbs_check(ctx, p: Params, state, ovfs, h: dict, tag: str = "cast"):
    """``Pack.check`` + ``Epilogue.check`` with the FP32 word never formed as one element (lane b-air-v2, V2_AIR.md 4.5,
    z3 ``pack_epilogue_limbs``): ``M = M_hi8 2^16 + M_lo16`` (both sides < 2^24), the high limb is the expression
    ``s 2^15 + fld 2^7 + M_hi8`` (< 2^16 on a finite state), the low limb ``M_lo16``; the infinity blend and the
    epilogue run per limb.  ``L = acc_frac - frac = 16`` is the epilogue's own split, so ``cast.split`` becomes two
    identities below ``2^16``.  Same predicate as ``Pack.check`` -> ``Epilogue.check`` over Z."""
    L = p.acc_frac - p.frac
    assert L == 16 and p.acc_sig == 24 and p.acc_word_bits == 32
    s, e_hat, M, z = state
    M_hi8, M_lo16 = h["M_hi8"], h["M_lo16"]
    ctx.rng(f"{tag}.M_hi8", M_hi8, 0, 2**(p.acc_sig - L))
    ctx.rng(f"{tag}.M_lo16", M_lo16, 0, 2**L)
    ctx.eq(f"{tag}.M_split", M_hi8 * 2**L + M_lo16, M)
    fld = e_hat + (p.bias - 1) + z * (p.e_min - p.floor)
    hi_fin = s * 2**(p.acc_word_bits - 1 - L) + fld * 2**(p.acc_frac - L) + M_hi8
    lo_fin = M_lo16
    any_ovf, inf_sign, none_yet = 0, 0, 1
    for ovf_g, sgn_g in ovfs:
        first = none_yet * ovf_g
        any_ovf = any_ovf + first
        inf_sign = inf_sign + first * sgn_g
        none_yet = none_yet - first
    y_hi = hi_fin - any_ovf * hi_fin + inf_sign * 2**(p.acc_word_bits - 1 - L) + any_ovf * (p.t_finite * 2**(p.acc_frac - L))
    y_lo = lo_fin - any_ovf * lo_fin
    hi, lo, hi2, lsb, carry, rem = (h[k] for k in ("hi", "lo", "hi2", "lsb", "carry", "rem"))
    ctx.rng(f"{tag}.hi", hi, 0, 2**p.word_bits)
    ctx.rng(f"{tag}.lo", lo, 0, 2**L)
    ctx.rng(f"{tag}.hi2", hi2, 0, 2 ** (p.word_bits - 1))
    ctx.boolean(f"{tag}.lsb", lsb)
    ctx.boolean(f"{tag}.carry", carry)
    ctx.rng(f"{tag}.rem", rem, 0, 2**L)
    ctx.eq(f"{tag}.split_hi", hi, y_hi)
    ctx.eq(f"{tag}.split_lo", lo, y_lo)
    ctx.eq(f"{tag}.lsb_split", 2 * hi2 + lsb, hi)
    ctx.eq(f"{tag}.round", carry * 2**L + rem, lo + lsb + (2 ** (L - 1) - 1))
    y16 = hi + carry
    ctx.rng(f"{tag}.y16", y16, 0, 2**p.word_bits)
    return y16


def _build_circuit(p: Params) -> tuple[CircuitCtx, list[str], list[str], dict]:
    """The a-v2 unit circuit (``build_unit_circuit_v2``) extended by Pack + the epilogue on the outgoing state and the
    claimed word ``y16`` (every unit carries them; the chain test reads ``y16`` at VU ends only)."""
    ctx = CircuitCtx()
    k = sum(p.groups)
    idx: dict[str, int] = {}

    def col(name):
        c = ctx.column(name)
        idx[name] = len(ctx.cols) - 1
        return c

    xs = [col(f"x.{n}") for n in STATE]
    a = [col(f"x.a[{i}]") for i in range(k)]
    b = [col(f"x.b[{i}]") for i in range(k)]
    ys = [col(f"y.{n}") for n in STATE]
    sample, _, _ = gen_unit(p, zero_state(p), [0] * k, [0] * k)
    paths: list[str] = []
    _leaves(sample, "h", paths)
    h_sym = copy.deepcopy(sample)
    for path in paths:
        _set(h_sym, path, col(path))
    esample = pack_limbs_gen(p, zero_state(p), [(0, 0)] * len(p.groups))
    epaths: list[str] = []
    _leaves(esample, "e", epaths)
    e_sym = copy.deepcopy(esample)
    for path in epaths:
        _set(e_sym, path, col(path))
    y16 = col("y16")
    state, ovfs = check_unit_expr(ctx, p, tuple(xs), a, b, h_sym, static=False)
    for n, got, want in zip(STATE, state, ys):
        ctx.eq(f"y.{n}", got, want)
    ovf_cols = [_col_of(ctx, ovf, f"g{g}.ovf") for g, (ovf, _) in enumerate(ovfs)]
    sgn_cols = [_col_of(ctx, sgn, f"g{g}.sgn") for g, (_, sgn) in enumerate(ovfs)]
    for nm in ovf_cols + sgn_cols:
        idx[nm] = ctx.cols.index(nm)
    got = pack_limbs_check(ctx, p, tuple(ys), ovfs, e_sym)
    ctx.eq("y16", got, y16)
    idx["__ovf"] = [idx[n] for n in ovf_cols]
    idx["__sgn"] = [idx[n] for n in sgn_cols]
    return ctx, paths, epaths, idx


def build_system(p: Params = REAL) -> V2System:
    return system_from_ctx(p, *_build_circuit(p))


def system_from_ctx(p: Params, ctx: CircuitCtx, paths: list[str], epaths: list[str], idx: dict) -> V2System:
    """Compile any ``CircuitCtx`` whose named columns include ``x.*``, ``y.*`` (the state), ``y16`` and ``__ovf``."""
    n_cols = len(ctx.cols)
    prod_wires = [w for w, kind in enumerate(ctx.wires) if kind[0] == "prod"]
    row_of_wire = [0] * len(ctx.wires)
    for w, kind in enumerate(ctx.wires):
        if kind[0] == "in":
            row_of_wire[w] = kind[1]
    for i, w in enumerate(prod_wires):
        row_of_wire[w] = n_cols + i
    sysm = V2System(p, ctx, paths, epaths, idx, row_of_wire, n_cols, len(prod_wires), [], [], [], {}, [])
    for name, x in ctx.asserts:
        coef, c = sysm.lin_rows(x)
        if not coef:
            if c:
                raise Reject(f"assert {name} is a false constant")
            continue
        sysm.asserts.append((name, coef, c))
    for i, w in enumerate(prod_wires):
        _, A, B = ctx.wires[w]
        sysm.prods.append((f"prod{i}", sysm.lin_rows(A), sysm.lin_rows(B), n_cols + i))
    names: list[str] = []
    for tname, lins in ctx.queries:
        sysm.lookups.append((tname, [sysm.lin_rows(x) for x in lins]))
        if tname not in names:
            names.append(tname)
    for tname in names:
        w = range_table_width(tname)
        sysm.tables[tname] = _range_table(w) if w is not None else _materialise(ctx.tables[tname], p)
    sysm.table_order = names
    sysm.system_id = hashlib.sha256(sysm.describe().encode()).digest()
    return sysm


# ---------------------------------------------------------------------------------------------------------------------
# witnesses (host Python integers; the field reduction happens in the prover)
# ---------------------------------------------------------------------------------------------------------------------

def unit_vals(sysm: V2System, state_in, a, b, state_out, h: dict, ovfs, y16: int | None = None) -> list[int]:
    """Every column of one unit as integers.  ``y16``: the claimed word for the chain end (default: the honest cast of
    this unit's own packed word)."""
    ctx, p = sysm.ctx, sysm.p
    vals: list[int | None] = [None] * len(ctx.cols)
    vals[0] = 1
    for i, n in enumerate(STATE):
        vals[sysm.col_idx[f"x.{n}"]] = int(state_in[i])
        vals[sysm.col_idx[f"y.{n}"]] = int(state_out[i])
    for i in range(len(a)):
        vals[sysm.col_idx[f"x.a[{i}]"]] = int(a[i])
        vals[sysm.col_idx[f"x.b[{i}]"]] = int(b[i])
    for path in sysm.hint_paths:
        vals[sysm.col_idx[path]] = int(_get(h, path))
    he = pack_limbs_gen(p, state_out, ovfs)
    for path in sysm.epi_paths:
        vals[sysm.col_idx[path]] = int(_get(he, path))
    if y16 is None:
        y16 = pack_limbs_check(IntCtx(), p, tuple(int(v) for v in state_out), ovfs, he)
    vals[sysm.col_idx["y16"]] = int(y16)
    return derive(ctx, vals)


@dataclass
class UnitArrays:
    """The tableau-1 contribution of a block of units, reduced mod p: columns (n_units, n_cols), product wires
    (n_units, n_prods), integer lookup keys (n_lookups, n_units).  Built per VU in worker processes."""
    cols: np.ndarray
    prods: np.ndarray
    keys: np.ndarray

    @property
    def n(self) -> int:
        return self.cols.shape[0]

    @staticmethod
    def concat(blocks: list["UnitArrays"]) -> "UnitArrays":
        return UnitArrays(np.concatenate([b.cols for b in blocks], 0), np.concatenate([b.prods for b in blocks], 0),
                          np.concatenate([b.keys for b in blocks], 1))


def unit_arrays(sysm: V2System, units: list[list[int]]) -> UnitArrays:
    n = len(units)
    cols = np.zeros((n, len(sysm.ctx.cols)), dtype=np.int64)
    prods = np.zeros((n, sysm.n_prods), dtype=np.int64)
    keys = np.zeros((len(sysm.lookups), n), dtype=np.int64)
    for u, vals in enumerate(units):
        cols[u] = [v % P for v in vals]
        prods[u] = [v % P for v in prod_vals(sysm, vals)]
        for j, (tname, lins) in enumerate(sysm.lookups):
            coef, c = lins[0]
            keys[j, u] = c + sum(v * vals[r] for r, v in coef.items())
    return UnitArrays(cols, prods, keys)


def prod_vals(sysm: V2System, vals: list[int]) -> list[int]:
    """The product wires' integer values from the columns (in ``prods`` order)."""
    ctx = sysm.ctx
    cache: dict[int, int] = {}
    out = []
    for w, kind in enumerate(ctx.wires):
        if kind[0] == "prod":
            out.append(ctx.wire_val(w, vals, cache))
    return out


def dummy_unit_vals(sysm: V2System) -> list[int]:
    """The pad unit: the constant zero state, zero operands (honest)."""
    p = sysm.p
    k = sum(p.groups)
    z0 = zero_state(p)
    h, s_out, ovfs = gen_unit(p, z0, [0] * k, [0] * k)
    return unit_vals(sysm, z0, [0] * k, [0] * k, s_out, h, ovfs)


def vu_units(sysm: V2System, a, b, y16: int, accs=None) -> list[list[int]]:
    """The units of one VU (state-chained from the zero state); the last unit carries the claimed ``y16``.  With
    ``accs`` every unit's packed word is compared to the recorded accumulator (a mismatch is a ``Reject``: the
    reference is never repaired)."""
    p = sysm.p
    k = sum(p.groups)
    if len(a) != len(b) or len(a) % k:
        raise Reject("operand count must be a multiple of the unit width")
    steps = len(a) // k
    state = zero_state(p)
    out = []
    for i in range(steps):
        ai, bi = list(a[i * k:(i + 1) * k]), list(b[i * k:(i + 1) * k])
        h, s_out, ovfs = gen_unit(p, state, ai, bi)
        if i < steps - 1 and any(o for o, _ in ovfs):
            raise Reject(f"unit {i}: saturation before the last unit is outside the domain")
        if accs is not None:
            y = Pack.check(IntCtx(), p, s_out, ovfs)
            if y != int(accs[i]):
                raise Reject(f"unit {i}: packed word {y:#x} differs from the recorded accumulator {int(accs[i]):#x}")
        out.append(unit_vals(sysm, state, ai, bi, s_out, h, ovfs, y16 if i == steps - 1 else None))
        state = s_out
    return out


def integer_audit(sysm: V2System, units: list[list[int]]) -> dict:
    """Which assertions / products take an intermediate integer value outside ``(-p, p)`` on these honest units (the
    word-class identities: over BabyBear they are congruences, not integer identities, until the limb repair)."""
    ctx = sysm.ctx
    bad_assert: dict[str, int] = {}
    bad_prod: dict[str, int] = {}
    for vals in units:
        cache: dict[int, int] = {}
        for name, x in ctx.asserts:
            mx = max((abs(c * ctx.wire_val(w, vals, cache)) for w, c in x.terms.items()), default=0)
            mx = max(mx, abs(x.const))
            if mx >= P:
                bad_assert[name] = max(bad_assert.get(name, 0), mx)
        for i, (w, kind) in enumerate((w, k) for w, k in enumerate(ctx.wires) if k[0] == "prod"):
            v = abs(ctx.wire_val(w, vals, cache))
            if v >= P:
                bad_prod[f"prod{i}"] = max(bad_prod.get(f"prod{i}", 0), v)
    return {"asserts_outside_p": {k: v.bit_length() for k, v in bad_assert.items()},
            "prods_outside_p": {k: v.bit_length() for k, v in bad_prod.items()},
            "n_asserts": len(ctx.asserts), "n_prods": sysm.n_prods}
