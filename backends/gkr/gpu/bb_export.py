"""The BabyBear circuit of the WP2 checker and the VU witness export (limbed words, the red-team's embedding rules).

``verity_numerical.redteam.babybear.check_step_bb`` is the embedded unit written against the generic predicate
interface (``rng``, ``eq``, ``boolean``, ``lookup``, ``lt_pow``); run on the symbolic ``CircuitCtx`` of
``gkr_export.circuit`` it yields the layered circuit over ``p = 2^31 - 2^27 + 1``: the FP32 words are two public 16-bit
limbs (``x.c_lo/x.c_hi`` in, ``y_lo/y_hi`` out; rule LIMB16), the wide variable bounds are the direct checks against
the tables' bound columns (LT_DIRECT, QUOT_TIGHT, NORM_MBND, NORM_VBND).  The epilogue circuit is
``Embedding.epilogue`` on the limb pair.  The chain (PROTOCOL.md §12) links the two limbs.

One deviation from ``gkr_export.circuit.CircuitCtx`` (flagged in DESIGN.md): its sub-16-bit range check
``R16 ∋ y · 2^(16-w)`` is not sound over a field (``y = r · 2^-(16-w)`` passes for any table entry ``r``; the same
hole is in the Goldilocks export).  Here ``rng(y, 0, 2^w)`` for ``w < 16`` is the pair of lookups ``R16 ∋ y`` and
``R16 ∋ y + 2^16 - 2^w`` (the first makes ``y`` a canonical integer below ``2^16``, the second bounds it).

File formats (``<QQ`` header ``n_rows n_cols``, then ``n_rows·n_cols`` little-endian **u32**): ``units.bin``,
``epilogue.bin``, ``public.bin`` (one ``y16`` per VU); ``circuit.txt`` / ``epilogue.txt`` in the text format of
``backends/gkr/src/circuit.rs`` with coefficients reduced mod ``p``.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import struct
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from verity_numerical.checker import REAL, Params, Reject
from verity_numerical.checker.ctx import IntCtx
from verity_numerical.checker.step import check_step_expr, step_output
from verity_numerical.gkr_export.circuit import CHUNK, R16, CircuitCtx, Lin, _get, _leaves, _set
from verity_numerical.redteam.babybear import DEFAULT, Embedding, check_step_bb, gen_step_bb, limbs
from verity_numerical.checker.gadgets import Epilogue

P = 2**31 - 2**27 + 1
STEP = 16
LIMB = 16


class BBCircuitCtx(CircuitCtx):
    """``CircuitCtx`` with the sound sub-16-bit range check (two R16 lookups)."""

    def _range_pow2(self, y: Lin, w: int, name: str) -> None:
        y = self._as_input_lin(y, name)
        if w > 2 * CHUNK:
            raise Reject(f"{name}: {w}-bit range needs more than two chunks")
        if w <= CHUNK:
            self._small(y, w)
            return
        lo = self.column(f"{name}.lo16", ("lo16", y))
        hi = self.column(f"{name}.hi16", ("hi16", y))
        self.asserts.append((f"{name}.split", y - lo - 2**CHUNK * hi))
        self.queries.append((R16, [lo]))
        self._small(hi, w - CHUNK)

    def _small(self, y: Lin, w: int) -> None:
        self.queries.append((R16, [y]))
        if w < CHUNK:
            self.queries.append((R16, [y + (2**CHUNK - 2**w)]))


# ---------------------------------------------------------------------------------------------------------------------
# Circuits
# ---------------------------------------------------------------------------------------------------------------------

def _hint_paths(sample: dict, drop: tuple[str, ...]) -> list[str]:
    paths: list[str] = []
    _leaves(sample, "h", paths)
    return [q for q in paths if q not in drop]


def build_unit_circuit(p: Params = REAL, emb: Embedding = DEFAULT) -> tuple[BBCircuitCtx, list[str]]:
    """Columns: ``one, x.c_lo, x.c_hi, x.a[k], x.b[k], y_lo, y_hi, hints..., derived...``."""
    ctx = BBCircuitCtx()
    k = sum(p.groups)
    c = (ctx.column("x.c_lo"), ctx.column("x.c_hi"))
    a = [ctx.column(f"x.a[{i}]") for i in range(k)]
    b = [ctx.column(f"x.b[{i}]") for i in range(k)]
    y = (ctx.column("y_lo"), ctx.column("y_hi"))
    sample = gen_step_bb(p, 0, [0] * k, [0] * k)
    paths = _hint_paths(sample, ("h.acc.f", "h.pack.low"))     # replaced by their limbs under LIMB16
    h_sym = copy.deepcopy(sample)
    for path in paths:
        _set(h_sym, path, ctx.column(path))
    got = check_step_bb(ctx, p, c, a, b, h_sym, emb)
    ctx.eq("y_lo", got[0], y[0])
    ctx.eq("y_hi", got[1], y[1])
    return ctx, paths


def build_epilogue_circuit(p: Params = REAL, emb: Embedding = DEFAULT) -> tuple[BBCircuitCtx, list[str]]:
    """Columns: ``one, y32_lo, y32_hi, y16, hints...``."""
    ctx = BBCircuitCtx()
    y32 = (ctx.column("y32_lo"), ctx.column("y32_hi"))
    y16 = ctx.column("y16")
    sample = Epilogue.gen(p, 0)
    paths = _hint_paths(sample, ("h.hi", "h.lo"))
    h_sym = copy.deepcopy(sample)
    for path in paths:
        _set(h_sym, path, ctx.column(path))
    got = emb.epilogue(ctx, p, y32, h_sym)
    ctx.eq("y16", got, y16)
    return ctx, paths


def fill_derived(ctx: CircuitCtx, vals: list) -> list[int]:
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
            vals[col] = table.row(kv)[idx] if table.lo <= kv < table.hi else 0
    return [int(v) for v in vals]


def unit_witness(ctx: CircuitCtx, paths: list[str], c: int, a: Sequence[int], b: Sequence[int], y: int, h: dict) -> list[int]:
    k = len(a)
    vals: list = [None] * len(ctx.cols)
    vals[0] = 1
    vals[1], vals[2] = limbs(c)
    for i in range(k):
        vals[3 + i] = a[i]
        vals[3 + k + i] = b[i]
    vals[3 + 2 * k], vals[4 + 2 * k] = limbs(y)
    base = 5 + 2 * k
    for i, path in enumerate(paths):
        vals[base + i] = int(_get(h, path))
    return fill_derived(ctx, vals)


def epilogue_witness(ctx: CircuitCtx, paths: list[str], y32: int, y16: int, h: dict) -> list[int]:
    vals: list = [None] * len(ctx.cols)
    vals[0] = 1
    vals[1], vals[2] = limbs(y32)
    vals[3] = y16
    for i, path in enumerate(paths):
        vals[4 + i] = int(_get(h, path))
    return fill_derived(ctx, vals)


def verify_rows(ctx: CircuitCtx, vals: list[int]) -> str | None:
    """The circuit's own check on one row's columns mod p: the first failing assertion / query, or None."""
    cache: dict[int, int] = {}
    for name, x in ctx.asserts:
        if ctx.eval_lin(x, vals, cache) % P:
            return f"assert {name}"
    for i, (tname, lins) in enumerate(ctx.queries):
        row = tuple(ctx.eval_lin(x, vals, cache) % P for x in lins)
        if tname == R16:
            if row[0] >= 2**CHUNK:
                return f"query {i} R16 {row[0]}"
        else:
            t = ctx.tables[tname]
            key = row[0] if row[0] < P // 2 else row[0] - P
            if not t.lo <= key < t.hi or tuple(v % P for v in t.row(key)) != row[1:]:
                return f"query {i} {tname} {row}"
    return None


def vu_rows(unit, epi, a: Sequence[int], b: Sequence[int], y16: int, accs: Sequence[int] | None = None,
            p: Params = REAL, cross_check: bool = True) -> tuple[list[list[int]], list[int]]:
    """Unit rows chained from ``c_0 = 0`` (``accs[i]`` = the recorded accumulator after step ``i``) + the epilogue row.
    ``cross_check`` runs the *integer* checker on every unit (``Reject`` on a mismatch, as ``gkr_export.vu``)."""
    from verity_numerical.checker import check_step

    uctx, upaths = unit
    ectx, epaths = epi
    if len(a) != len(b) or len(a) % STEP:
        raise Reject("operand count must be a multiple of 16")
    steps = len(a) // STEP
    if accs is not None and len(accs) != steps:
        raise Reject("one recorded accumulator per step expected")
    rows, c = [], 0
    for i in range(steps):
        ai, bi = list(a[i * STEP:(i + 1) * STEP]), list(b[i * STEP:(i + 1) * STEP])
        h = gen_step_bb(p, c, ai, bi)
        y = int(accs[i]) if accs is not None else step_output(p, c, ai, bi)
        if cross_check:
            check_step(p, c, ai, bi, y, h)
        rows.append(unit_witness(uctx, upaths, c, ai, bi, y, h))
        c = y
    he = Epilogue.gen(p, c)
    if cross_check:
        ctx = IntCtx()
        ctx.eq("y16", Epilogue.check(ctx, p, c, he), y16)
    return rows, epilogue_witness(ectx, epaths, c, y16, he)


# ---------------------------------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------------------------------

def _fe(v: int) -> int:
    return v % P


def _lin_wires(x: Lin) -> str:
    return f"{len(x.terms)} " + " ".join(f"{w} {_fe(c)}" for w, c in sorted(x.terms.items())) + f" {_fe(x.const)}"


def _lin_cols(ctx: CircuitCtx, x: Lin) -> str:
    return f"{len(x.terms)} " + " ".join(f"{ctx.wires[w][1]} {_fe(c)}" for w, c in sorted(x.terms.items())) + f" {_fe(x.const)}"


def circuit_text(ctx: CircuitCtx) -> str:
    lines = [f"ncols {len(ctx.cols)}", f"nwires {len(ctx.wires)}", "one 0"]
    lines += [f"col {i} {n}" for i, n in enumerate(ctx.cols)]
    for w, kind in enumerate(ctx.wires):
        if kind[0] == "prod":
            lines.append(f"prod {w} {ctx.depth[w]} {_lin_wires(kind[1])} {_lin_wires(kind[2])}")
    for name, x in ctx.asserts:
        lines.append(f"assert {name} {_lin_wires(x)}")
    for tname, t in ctx.tables.items():
        lines.append(f"table {tname} {len(t.outs) + 1} {t.rows}")
        for key in range(t.lo, t.hi):
            lines.append("row " + " ".join(str(_fe(v)) for v in (key, *t.row(key))))
    for tname, lins in ctx.queries:
        lines.append(f"query {tname} {len(lins)} " + " ".join(_lin_cols(ctx, x) for x in lins))
    return "\n".join(lines) + "\n"


def write_rows(rows: list[list[int]], path: Path, ncols: int | None = None) -> None:
    ncols = len(rows[0]) if rows else (ncols or 0)
    arr = np.array([[v % P for v in r] for r in rows], dtype="<u4").reshape(len(rows), ncols)
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", len(rows), ncols))
        f.write(arr.tobytes())


def read_rows(path: Path) -> np.ndarray:
    """``(n_rows, n_cols)`` uint32."""
    with open(path, "rb") as f:
        n, m = struct.unpack("<QQ", f.read(16))
        return np.frombuffer(f.read(), dtype="<u4").reshape(n, m)


def chain_columns(uctx: CircuitCtx, ectx: CircuitCtx) -> dict:
    ci = uctx.cols.index
    ei = ectx.cols.index
    return {"col_c": [ci("x.c_lo"), ci("x.c_hi")], "col_y": [ci("y_lo"), ci("y_hi")],
            "epi_y32": [ei("y32_lo"), ei("y32_hi")], "epi_y16": ei("y16")}


_UNIT = None
_EPI = None


def _init_worker() -> None:
    global _UNIT, _EPI
    _UNIT = build_unit_circuit()
    _EPI = build_epilogue_circuit()


def _one(task):
    vid, a, b, y16, accs = task
    try:
        rows, erow = vu_rows(_UNIT, _EPI, a, b, y16, accs, cross_check=accs is not None)
    except Reject as e:
        return vid, None, None, str(e)
    ub = np.array([[v % P for v in r] for r in rows], dtype="<u4").tobytes()
    eb = np.array([v % P for v in erow], dtype="<u4").tobytes()
    return vid, ub, eb, None


def _write_batch(out: Path, results, y16s, steps, ucols, ecols) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "units.bin", "wb") as f:
        f.write(struct.pack("<QQ", len(results) * steps, ucols))
        f.writelines(ub for _, ub, _ in results)
    with open(out / "epilogue.bin", "wb") as f:
        f.write(struct.pack("<QQ", len(results), ecols))
        f.writelines(eb for _, _, eb in results)
    write_rows([[y] for y in y16s], out / "public.bin", 1)


def export(root: Path, tier: str, lo: int, hi: int | None, out: Path, procs: int) -> dict:
    from verity_numerical.bench import instances as I

    tierdata = I.load(tier, root)
    hi = tierdata.n if hi is None else min(hi, tierdata.n)
    uctx, _ = build_unit_circuit()
    ectx, _ = build_epilogue_circuit()
    out.mkdir(parents=True, exist_ok=True)
    (out / "circuit.txt").write_text(circuit_text(uctx))
    (out / "epilogue.txt").write_text(circuit_text(ectx))
    steps = tierdata.K // STEP
    tasks = []
    for vid in range(lo, hi):
        x, w, y = tierdata.instance(vid)
        accs = tierdata.accs(vid) if tierdata.acc is not None else None
        tasks.append((vid, x, w, y, accs))
    results = {}
    if procs > 1:
        with mp.get_context("fork").Pool(procs, initializer=_init_worker) as pool:
            for vid, ub, eb, err in pool.imap_unordered(_one, tasks, chunksize=4):
                results[vid] = (ub, eb, err)
    else:
        _init_worker()
        for t in tasks:
            vid, ub, eb, err = _one(t)
            results[vid] = (ub, eb, err)
    manifest = {"tier": tier, "range": [lo, hi], "steps": steps, "field": "babybear", "unit_columns": len(uctx.cols),
                "epilogue_columns": len(ectx.cols), "chain": chain_columns(uctx, ectx), "root": str(root),
                "root_manifest_sha256": I.manifest_sha256(root), "rejected_at_witness": {}, "written": []}
    if tier.endswith("-neg"):
        for vid in range(lo, hi):
            ub, eb, err = results[vid]
            if err is not None:
                manifest["rejected_at_witness"][vid] = err
                continue
            _write_batch(out / "neg" / str(vid), [(vid, ub, eb)], [int(tierdata.y[vid])], steps, len(uctx.cols), len(ectx.cols))
            manifest["written"].append(vid)
        rows = {r[0]: r for r in tierdata.index}
        manifest["index"] = {vid: dict(zip(tierdata.columns, rows[vid])) for vid in range(lo, hi)}
    else:
        ok = []
        for vid in range(lo, hi):
            ub, eb, err = results[vid]
            if err is not None:
                manifest["rejected_at_witness"][vid] = err
            else:
                ok.append((vid, ub, eb))
        if manifest["rejected_at_witness"]:
            print(f"{len(manifest['rejected_at_witness'])} positives rejected at witness generation", file=sys.stderr)
        _write_batch(out, ok, [int(tierdata.y[vid]) for vid, _, _ in ok], steps, len(uctx.cols), len(ectx.cols))
        manifest["written"] = [vid for vid, _, _ in ok]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return manifest


def synth(out: Path, vus: int, steps: int, seed: int, p: Params = REAL) -> dict:
    """Random finite operands chained by the checker itself (tests)."""
    import random

    rng = random.Random(seed)
    unit = build_unit_circuit(p)
    epi = build_epilogue_circuit(p)
    uctx, _ = unit
    ectx, _ = epi

    def bf16() -> int:
        return (rng.randrange(2) << 15) | (rng.randrange(100, 150) << 7) | rng.randrange(2**7)

    urows, erows, y16s = [], [], []
    for _ in range(vus):
        a = [bf16() for _ in range(steps * STEP)]
        b = [bf16() for _ in range(steps * STEP)]
        c = 0
        for i in range(steps):
            c = step_output(p, c, a[i * STEP:(i + 1) * STEP], b[i * STEP:(i + 1) * STEP])
        he = Epilogue.gen(p, c)
        y16 = Epilogue.check(IntCtx(), p, c, he)
        rows, erow = vu_rows(unit, epi, a, b, y16, None, p)
        urows += rows
        erows.append(erow)
        y16s.append(y16)
    out.mkdir(parents=True, exist_ok=True)
    (out / "circuit.txt").write_text(circuit_text(uctx))
    (out / "epilogue.txt").write_text(circuit_text(ectx))
    write_rows(urows, out / "units.bin")
    write_rows(erows, out / "epilogue.bin")
    write_rows([[y] for y in y16s], out / "public.bin", 1)
    manifest = {"tier": "synth", "vus": vus, "steps": steps, "seed": seed, "field": "babybear",
                "unit_columns": len(uctx.cols), "epilogue_columns": len(ectx.cols), "chain": chain_columns(uctx, ectx)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backends.gkr.gpu.bb_export")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--root", type=Path, required=True)
    e.add_argument("--tier", default="vu-k1536")
    e.add_argument("--lo", type=int, default=0)
    e.add_argument("--hi", type=int, default=None)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--procs", type=int, default=1)
    s = sub.add_parser("synth")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--vus", type=int, default=2)
    s.add_argument("--steps", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    st = sub.add_parser("stats")
    args = ap.parse_args(argv)
    if args.cmd == "export":
        m = export(args.root, args.tier, args.lo, args.hi, args.out, args.procs)
        print(json.dumps({k: v for k, v in m.items() if k not in ("index", "written")}, default=str))
    elif args.cmd == "synth":
        print(json.dumps(synth(args.out, args.vus, args.steps, args.seed)))
    else:
        uctx, _ = build_unit_circuit()
        ectx, _ = build_epilogue_circuit()
        for name, ctx in (("unit", uctx), ("epilogue", ectx)):
            nq = {}
            for t, _l in ctx.queries:
                nq[t] = nq.get(t, 0) + 1
            print(name, "cols", len(ctx.cols), "wires", len(ctx.wires), "depth", max(ctx.depth), "asserts", len(ctx.asserts), "queries", nq)
    return 0


if __name__ == "__main__":
    sys.exit(main())
