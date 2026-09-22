"""Candidate A on checker v2 (``checker.v2``): the state-chained unit circuit, the Pack + epilogue circuit, the
chain specification, the witnesses of ``bench-instances/v1`` and the fixed tables (PROTOCOL.md section 14).

The VU is proved as in section 12 -- two data-parallel segments over one commitment, linkage as linear constraints
on the committed vector -- with the v2 shape:

* **unit segment**: one row per 16-product unit.  Columns ``x.s, x.e, x.M, x.z`` (the incoming state), ``x.a[i]``,
  ``x.b[i]``, ``y.s, y.e, y.M, y.z`` (the outgoing state, each asserted equal to the gadget's expression), the 12
  hints of ``gen_unit`` (per group: ``E``, ``acc_split.{M_hi, M_lo}``, ``sum.{sgn, v_hi, v_lo}``), every lookup
  output and the materialised query keys.  No word is decoded or packed inside a unit.
* **epilogue segment**: one row per VU: ``s, e, M, z`` (the last state), ``ovf[g], sgn[g]`` (the last unit's
  saturation flags), ``Pack.check`` as an expression, ``Epilogue.check`` on it, ``y16`` asserted equal to the cast.
* **chain** (``chain.txt``, read by ``main.rs::load_vu``): four state links ``x.* = init`` at unit 0 (the constant
  zero state ``(0, floor, 0, 1)``), ``x.*_i = y.*_{i-1}``, ``epi.* = y.*_95``; two ``ovf`` links with
  ``ovf_i = 0`` for every non-final unit (the domain excludes a saturated intermediate) and ``epi.ovf = ovf_95``;
  two ``sgn`` links ``epi.sgn = sgn_95``; and the public word.  583 constraints per VU at K = 1536 (v1: 98).

Tables: ``T_OP``, ``ALIGN4``, ``LEAD``, ``LEADNORM`` are written row by row; ``SHIFT``, ``SSHIFT_HI``,
``SSHIFT_LO``, ``TNORM`` (5.0M of the 5.2M rows) are *implicit*: the circuit text carries their defining
parameters (``itable`` lines) and ``circuit.rs`` materialises them, exactly as it does the ``R<w>`` ranges.  In
either case the rows are public data and only the multiplicities are committed (per proof).
"""
from __future__ import annotations

import copy
import struct
from collections.abc import Sequence
from pathlib import Path

from verity_numerical.checker import REAL, Params, Reject, check_step, gen_step
from verity_numerical.checker.ctx import IntCtx, Table
from verity_numerical.checker.gadgets import Epilogue
from verity_numerical.checker.v2.gadgets import Pack
from verity_numerical.checker.v2.params import V2
from verity_numerical.checker.v2.step import check_unit_expr, gen_unit, zero_state
from verity_numerical.checker.v2.tables import (
    shift_table,
    sshift_hi_table,
    sshift_lo_table,
    tnorm_table,
)

from .circuit import (
    CHUNK,
    GOLDILOCKS,
    CircuitCtx,
    Lin,
    _get,
    _leaves,
    _lin_cols,
    _lin_wires,
    _set,
    range_table_width,
)

__all__ = [
    "IMPLICIT_TABLES",
    "build_epilogue_circuit_v2",
    "build_unit_circuit_v2",
    "chain_spec",
    "derive",
    "epilogue_witness_v2",
    "export_circuit_v2",
    "implicit_table_header",
    "unit_witness_v2",
    "verify_row",
    "vu_rows_v2",
    "write_chain",
]

STEP = 16
IMPLICIT_TABLES = ("SHIFT", "SSHIFT_HI", "SSHIFT_LO", "TNORM")


# ---------------------------------------------------------------------------------------------------------------------
# circuits
# ---------------------------------------------------------------------------------------------------------------------

def _col_of(ctx: CircuitCtx, x: Lin, name: str) -> str:
    """The column name of a value that is exactly one input column; otherwise a new column bound to it."""
    if len(x.terms) == 1 and x.const == 0:
        (w, c), = x.terms.items()
        if c == 1 and ctx.wires[w][0] == "in":
            return ctx.cols[ctx.wires[w][1]]
    col = ctx.column(name)
    ctx.asserts.append((name, col - x))
    return name


def build_unit_circuit_v2(p: Params = REAL) -> tuple[CircuitCtx, list[str]]:
    """The state-chained v2 unit as a circuit; returns the context and the hint paths (``h.groups[g].E``, ...).

    ``ctx.chain_cols`` records the linkage column names: ``state`` (four ``(x, y)`` pairs), ``ovf`` and ``sgn``
    (per group, the unit's column)."""
    ctx = CircuitCtx()
    k = sum(p.groups)
    xs = [ctx.column(f"x.{n}") for n in ("s", "e", "M", "z")]
    a = [ctx.column(f"x.a[{i}]") for i in range(k)]
    b = [ctx.column(f"x.b[{i}]") for i in range(k)]
    ys = [ctx.column(f"y.{n}") for n in ("s", "e", "M", "z")]
    sample, _, _ = gen_unit(p, zero_state(p), [0] * k, [0] * k)
    paths: list[str] = []
    _leaves(sample, "h", paths)
    h_sym = copy.deepcopy(sample)
    for path in paths:
        _set(h_sym, path, ctx.column(path))
    state, ovfs = check_unit_expr(ctx, p, tuple(xs), a, b, h_sym, static=False)
    for n, got, want in zip(("s", "e", "M", "z"), state, ys):
        ctx.eq(f"y.{n}", got, want)
    ctx.chain_cols = {  # type: ignore[attr-defined]
        "state": [(f"x.{n}", f"y.{n}") for n in ("s", "e", "M", "z")],
        "ovf": [_col_of(ctx, ovf, f"g{g}.ovf") for g, (ovf, _) in enumerate(ovfs)],
        "sgn": [_col_of(ctx, sgn, f"g{g}.sgn") for g, (_, sgn) in enumerate(ovfs)],
    }
    return ctx, paths


def build_epilogue_circuit_v2(p: Params = REAL) -> tuple[CircuitCtx, list[str]]:
    """``Pack.check`` on the last state and flags, then ``Epilogue.check``; columns ``one, s, e, M, z, ovf[g],
    sgn[g] (per group), y16, hints``."""
    ctx = CircuitCtx()
    G = len(p.groups)
    state = tuple(ctx.column(n) for n in ("s", "e", "M", "z"))
    flags = [(ctx.column(f"ovf[{g}]"), ctx.column(f"sgn[{g}]")) for g in range(G)]
    y16 = ctx.column("y16")
    sample = Epilogue.gen(p, 0)
    paths: list[str] = []
    _leaves(sample, "h", paths)
    h_sym = copy.deepcopy(sample)
    for path in paths:
        _set(h_sym, path, ctx.column(path))
    y32 = Pack.check(ctx, p, state, flags)
    got = Epilogue.check(ctx, p, y32, h_sym)
    ctx.eq("y16", got, y16)
    return ctx, paths


def chain_spec(unit: CircuitCtx, epi: CircuitCtx, p: Params = REAL) -> list[tuple[str | None, str, str, int, bool]]:
    """``(incoming column | None, outgoing column, epilogue column, init, zero_nonlast)`` per link (see the module
    docstring; ``chain.txt`` rows)."""
    cc = unit.chain_cols  # type: ignore[attr-defined]
    z0 = zero_state(p)
    links: list[tuple[str | None, str, str, int, bool]] = []
    for (xc, yc), en, init in zip(cc["state"], ("s", "e", "M", "z"), z0):
        links.append((xc, yc, en, init, False))
    for g, c in enumerate(cc["ovf"]):
        links.append((None, c, f"ovf[{g}]", 0, True))
    for g, c in enumerate(cc["sgn"]):
        links.append((None, c, f"sgn[{g}]", 0, False))
    return links


def write_chain(links, path: Path, modulus: int = GOLDILOCKS) -> None:
    """``link <in|-> <out> <epi> <init mod p> <zero_nonlast 0|1>`` per line (``main.rs::read_chain``)."""
    lines = [f"link {xc or '-'} {yc} {en} {init % modulus} {int(zn)}" for xc, yc, en, init, zn in links]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------------------------------
# tables and the circuit text
# ---------------------------------------------------------------------------------------------------------------------

def implicit_table_header(t: Table, p: Params = REAL) -> str | None:
    """The ``itable`` parameters ``circuit.rs`` materialises a v2 table from, or None for a row-listed table.

    ``shift <bits> <scale> <R>``: key ``x R + 2 d + s`` -> ``(1 - 2 s) floor(x 2^scale / 2^d)`` for ``x < 2^bits``;
    ``tnorm <vb> <k_rows> <k_lo>``: key ``v_lo + 2^vb koff`` -> ``floor(v_lo 2^(koff + k_lo))``."""
    R = 2 * V2.dt_rows(p)
    if t.name == "SHIFT":
        return f"shift {2 * p.sig} {p.prod_shift} {R}"
    if t.name == "SSHIFT_HI":
        return f"shift {V2.mh(p)} {V2.mb(p) + p.acc_shift} {R}"
    if t.name == "SSHIFT_LO":
        return f"shift {V2.mb(p)} {p.acc_shift} {R}"
    if t.name == "TNORM":
        return f"tnorm {V2.vb(p)} {V2.k_rows(p)} {p.k_lo}"
    return None


def _check_implicit(t: Table, p: Params) -> None:
    """The header's function must be the table's (cheap spot check of the two conventions before export)."""
    R = 2 * V2.dt_rows(p)
    for key in (0, 1, R, R + 3, t.hi - 1, (t.hi - 1) // 3):
        assert t.row(key) == _implicit_row(t.name, key, p), (t.name, key)
    assert t.rows == t.hi - t.lo and t.lo == 0


def _implicit_row(name: str, key: int, p: Params) -> tuple[int, ...]:
    """Python transcription of ``circuit.rs::materialise`` (kept beside the header for the spot check)."""
    parts = implicit_table_header({"SHIFT": shift_table, "SSHIFT_HI": sshift_hi_table, "SSHIFT_LO": sshift_lo_table,
                                   "TNORM": tnorm_table}[name](p), p).split()
    if parts[0] == "shift":
        _bits, scale, R = int(parts[1]), int(parts[2]), int(parts[3])
        x, rem = divmod(key, R)
        d, s = divmod(rem, 2)
        v = (x << scale) >> d
        return (-v if s else v,)
    vb, _k_rows, k_lo = int(parts[1]), int(parts[2]), int(parts[3])
    koff, v_lo = divmod(key, 2**vb)
    k = koff + k_lo
    return ((v_lo << k) if k >= 0 else (v_lo >> -k),)


def export_circuit_v2(ctx: CircuitCtx, path: Path, modulus: int = GOLDILOCKS, p: Params = REAL) -> None:
    """``circuit.export_circuit`` with the large v2 tables as ``itable`` lines."""
    m = modulus

    def fe(v: int) -> int:
        return v % m

    lines = [f"ncols {len(ctx.cols)}", f"nwires {len(ctx.wires)}", "one 0"]
    lines += [f"col {i} {n}" for i, n in enumerate(ctx.cols)]
    for w, kind in enumerate(ctx.wires):
        if kind[0] == "prod":
            lines.append(f"prod {w} {ctx.depth[w]} {_lin_wires(kind[1], m)} {_lin_wires(kind[2], m)}")
    for name, x in ctx.asserts:
        lines.append(f"assert {name} {_lin_wires(x, m)}")
    for tname, t in ctx.tables.items():
        hdr = implicit_table_header(t, p)
        if hdr is not None:
            _check_implicit(t, p)
            lines.append(f"itable {tname} {len(t.outs) + 1} {t.rows} {hdr}")
            continue
        lines.append(f"table {tname} {len(t.outs) + 1} {t.rows}")
        for key in range(t.lo, t.hi):
            if t.domain is not None and not t.domain(key):
                continue
            lines.append("row " + " ".join(str(fe(v)) for v in (key, *t.fn(key))))
    for tname, lins in ctx.queries:
        lines.append(f"query {tname} {len(lins)} " + " ".join(_lin_cols(ctx, x, m) for x in lins))
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------------------------------
# witnesses
# ---------------------------------------------------------------------------------------------------------------------

def derive(ctx: CircuitCtx, vals: list[int | None]) -> list[int]:
    """Fill every derived column (materialised keys, lookup rows) from the base columns, in column order."""
    cache: dict[int, int] = {}
    for col, d in enumerate(ctx.derived):
        if d is None:
            continue
        kind = d[0]
        if kind == "const":
            vals[col] = d[1]
        elif kind == "lin":
            vals[col] = ctx.eval_lin(d[1], vals, cache)
        elif kind == "lo16":
            vals[col] = ctx.eval_lin(d[1], vals, cache) & (2**CHUNK - 1)
        elif kind == "hi16":
            vals[col] = ctx.eval_lin(d[1], vals, cache) >> CHUNK
        elif kind == "row":
            _, table, key, idx = d
            kv = ctx.eval_lin(key, vals, cache)
            try:
                vals[col] = table.row(kv)[idx]
            except KeyError:
                vals[col] = 0
    return [int(v) for v in vals]  # type: ignore[arg-type]


def unit_witness_v2(ctx: CircuitCtx, paths: list[str], state_in: Sequence[int], a: Sequence[int], b: Sequence[int],
                    state_out: Sequence[int], h: dict) -> list[int]:
    k = len(a)
    vals: list[int | None] = [None] * len(ctx.cols)
    vals[0] = 1
    for i in range(4):
        vals[1 + i] = int(state_in[i])
    for i in range(k):
        vals[5 + i] = int(a[i])
        vals[5 + k + i] = int(b[i])
    for i in range(4):
        vals[5 + 2 * k + i] = int(state_out[i])
    base = 9 + 2 * k
    for i, path in enumerate(paths):
        vals[base + i] = int(_get(h, path))
    return derive(ctx, vals)


def epilogue_witness_v2(ctx: CircuitCtx, paths: list[str], state: Sequence[int], flags: Sequence[tuple[int, int]],
                        y16: int, h: dict) -> list[int]:
    vals: list[int | None] = [None] * len(ctx.cols)
    vals[0] = 1
    for i in range(4):
        vals[1 + i] = int(state[i])
    for g, (ovf, sgn) in enumerate(flags):
        vals[5 + 2 * g] = int(ovf)
        vals[6 + 2 * g] = int(sgn)
    off = 5 + 2 * len(flags)
    vals[off] = int(y16)
    for i, path in enumerate(paths):
        vals[off + 1 + i] = int(_get(h, path))
    return derive(ctx, vals)


def verify_row(ctx: CircuitCtx, vals: list[int], modulus: int = GOLDILOCKS) -> str | None:
    """The circuit's own check on one row (mod p): the first failing assertion / query, or None."""
    cache: dict[int, int] = {}
    for name, x in ctx.asserts:
        if ctx.eval_lin(x, vals, cache) % modulus:
            return f"assert {name}"
    for i, (tname, lins) in enumerate(ctx.queries):
        row = tuple(ctx.eval_lin(x, vals, cache) % modulus for x in lins)
        if (w := range_table_width(tname)) is not None:
            if row[0] >= 2**w:
                return f"query {i} {tname} {row[0]}"
        else:
            t = ctx.tables[tname]
            try:
                want = tuple(v % modulus for v in t.row(row[0]))
            except KeyError:
                return f"query {i} {tname} key {row[0]} off table"
            if want != row[1:]:
                return f"query {i} {tname} {row}"
    return None


def vu_rows_v2(unit: tuple[CircuitCtx, list[str]], epi: tuple[CircuitCtx, list[str]], a: Sequence[int], b: Sequence[int],
               y16: int, accs: Sequence[int] | None = None, p: Params = REAL, cross_check: bool = True,
               modulus: int = GOLDILOCKS) -> tuple[list[list[int]], list[int]]:
    """The unit rows (state-chained from the constant zero state) and the epilogue row of one VU.

    ``accs[i]`` is the recorded FP32 accumulator after unit ``i``: with ``cross_check`` every unit's packed state is
    compared to it and the same (word-in, word-out) unit is run through the v1 WP2 checker (``check_step``), so a
    chain the reference computes differently is rejected here, never repaired."""
    uctx, upaths = unit
    ectx, epaths = epi
    if len(a) != len(b) or len(a) % STEP:
        raise Reject("operand count must be a multiple of 16")
    steps = len(a) // STEP
    if accs is not None and len(accs) != steps:
        raise Reject("one recorded accumulator per unit expected")
    rows = []
    state = zero_state(p)
    c = 0
    flags: list[tuple[int, int]] = []
    for i in range(steps):
        ai, bi = list(a[i * STEP:(i + 1) * STEP]), list(b[i * STEP:(i + 1) * STEP])
        h, state_out, flags = gen_unit(p, state, ai, bi)
        if i < steps - 1 and any(o for o, _ in flags):
            raise Reject(f"unit {i}: saturation before the last unit is outside the domain")
        y = Pack.check(IntCtx(), p, state_out, flags)
        if accs is not None and y != int(accs[i]):
            raise Reject(f"unit {i}: v2 word {y:#x} differs from the recorded accumulator {int(accs[i]):#x}")
        if cross_check:
            check_step(p, c, ai, bi, y, gen_step(p, c, ai, bi))
        rows.append(unit_witness_v2(uctx, upaths, state, ai, bi, state_out, h))
        state, c = state_out, y
    he = Epilogue.gen(p, c)
    if cross_check:
        ctx = IntCtx()
        ctx.eq("y16", Epilogue.check(ctx, p, c, he), y16)
    erow = epilogue_witness_v2(ectx, epaths, state, flags, y16, he)
    if cross_check:
        for r in rows:
            bad = verify_row(uctx, r, modulus)
            if bad is not None:
                raise Reject(f"circuit rejects an honest unit: {bad}")
        bad = verify_row(ectx, erow, modulus)
        if bad is not None:
            raise Reject(f"circuit rejects the honest epilogue: {bad}")
    return rows, erow


def pack_rows(rows: list[list[int]], modulus: int = GOLDILOCKS) -> bytes:
    ncols = len(rows[0])
    return b"".join(struct.pack(f"<{ncols}Q", *(v % modulus for v in r)) for r in rows)


def write_rows(rows: list[list[int]], path: Path, modulus: int = GOLDILOCKS) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", len(rows), len(rows[0]) if rows else 0))
        f.write(pack_rows(rows, modulus) if rows else b"")


# ---------------------------------------------------------------------------------------------------------------------
# synthetic VUs for the Rust tests (`python -m verity_numerical.gkr_export.vu synth --variant v2`)
# ---------------------------------------------------------------------------------------------------------------------

def synth_v2(out: Path, vus: int, steps: int, seed: int, p: Params, api) -> dict:
    """Honest v2 VUs (random finite operands, the checker's own chain) plus the chain-level negatives: a wrong public
    word; a broken state chain (unit 1 regenerated from a different incoming state); a saturated intermediate
    (unit 0 overflows: every per-unit witness honest, the ``ovf`` link must reject); a wrong ``ovf`` flag into
    the epilogue (the last unit's flag copied as 1)."""
    import random

    from verity_numerical.checker.v2.step import vu_output

    m = api.modulus
    rng = random.Random(seed)
    unit, epi = api.build_unit(), api.build_epilogue()
    uctx, ectx = unit[0], epi[0]

    def bf16() -> int:
        return (rng.randrange(2) << 15) | (rng.randrange(100, 150) << 7) | rng.randrange(2**7)

    out.mkdir(parents=True, exist_ok=True)
    api.export_circuit(uctx, out / "circuit.txt")
    api.export_circuit(ectx, out / "epilogue.txt")
    api.export_extra(uctx, ectx, out)
    all_rows: list[list[int]] = []
    erows: list[list[int]] = []
    y16s: list[int] = []
    ops: list[tuple[list[int], list[int]]] = []
    for _ in range(vus):
        a = [bf16() for _ in range(steps * STEP)]
        b = [bf16() for _ in range(steps * STEP)]
        y16 = vu_output(p, a, b)
        rows, erow = vu_rows_v2(unit, epi, a, b, y16, p=p, modulus=m)
        all_rows += rows
        erows.append(erow)
        y16s.append(y16)
        ops.append((a, b))
    write_rows(all_rows, out / "units.bin", m)
    write_rows(erows, out / "epilogue.bin", m)
    write_rows([[y] for y in y16s], out / "public.bin", m)
    neg = out / "negatives"
    neg.mkdir(exist_ok=True)
    write_rows([[y16s[0] ^ 1], *[[y] for y in y16s[1:]]], neg / "public_wrong_word.bin", m)
    if steps >= 2:
        # VU 0's unit 1 regenerated from a different (valid) incoming state: its own row is honest, the link is not
        a0, b0 = ops[0]
        s_in = tuple(all_rows[1][1:5])
        alt = (s_in[0], s_in[1] + 1, s_in[2], s_in[3])
        h, s_out, _ = gen_unit(p, alt, a0[STEP:2 * STEP], b0[STEP:2 * STEP])
        alt_row = unit_witness_v2(uctx, unit[1], alt, a0[STEP:2 * STEP], b0[STEP:2 * STEP], s_out, h)
        assert verify_row(uctx, alt_row, m) is None
        broken = list(all_rows)
        broken[1] = alt_row
        write_rows(broken, neg / "units_broken_chain.bin", m)
        # a saturated intermediate: VU 0 rebuilt with unit 0 = 16 products of the largest finite BF16 words
        big = 0x7F7F
        a_sat = [big] * STEP + a0[STEP:]
        b_sat = [big] * STEP + b0[STEP:]
        state, rows_sat, flags = zero_state(p), [], []
        for i in range(steps):
            ai, bi = a_sat[i * STEP:(i + 1) * STEP], b_sat[i * STEP:(i + 1) * STEP]
            h, s_out, flags = gen_unit(p, state, ai, bi)
            rows_sat.append(unit_witness_v2(uctx, unit[1], state, ai, bi, s_out, h))
            state = s_out
        assert any(rows_sat[0][uctx.cols.index(c)] for c in uctx.chain_cols["ovf"]), "unit 0 must saturate"  # type: ignore[attr-defined]
        y32 = Pack.check(IntCtx(), p, state, flags)
        he = Epilogue.gen(p, y32)
        y16_sat = Epilogue.check(IntCtx(), p, y32, he)
        erow_sat = epilogue_witness_v2(ectx, epi[1], state, flags, y16_sat, he)
        sat_dir = neg / "saturated_intermediate"
        sat_dir.mkdir(exist_ok=True)
        write_rows(rows_sat + all_rows[steps:], sat_dir / "units.bin", m)
        write_rows([erow_sat, *erows[1:]], sat_dir / "epilogue.bin", m)
        write_rows([[y16_sat], *[[y] for y in y16s[1:]]], sat_dir / "public.bin", m)
    # the epilogue told the last unit saturated (ovf[0] = 1, sgn as recorded): its Pack word is an infinity
    e_bad = list(erows[0])
    e_bad[5] = 1
    st = tuple(e_bad[1:5])
    flags_bad = [(e_bad[5 + 2 * g], e_bad[6 + 2 * g]) for g in range(len(p.groups))]
    y32_bad = Pack.check(IntCtx(), p, st, flags_bad)
    he_bad = Epilogue.gen(p, y32_bad)
    y16_bad = Epilogue.check(IntCtx(), p, y32_bad, he_bad)
    e_bad = epilogue_witness_v2(ectx, epi[1], st, flags_bad, y16_bad, he_bad)
    inf_dir = neg / "false_overflow_flag"
    inf_dir.mkdir(exist_ok=True)
    write_rows(all_rows, inf_dir / "units.bin", m)
    write_rows([e_bad, *erows[1:]], inf_dir / "epilogue.bin", m)
    write_rows([[y16_bad], *[[y] for y in y16s[1:]]], inf_dir / "public.bin", m)
    return {"vus": vus, "steps": steps, "unit_columns": len(uctx.cols), "epilogue_columns": len(ectx.cols), "y16": y16s,
            "field": api.name, "variant": "v2"}
