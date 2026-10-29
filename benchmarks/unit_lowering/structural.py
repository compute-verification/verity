"""The H1 leaf's statements in their templated structural form (``H1_TMPL=structural``: `h1unit.stager`'s
`UnitStager(layout=h1unit.LAYOUT, structural=True)`, the unit as `struct_lower`'s template, the flat unit never built),
measured beside the flat form, at any K and R × C tile. A leaf is proved as one of two units: ``tile``, the whole tile
(`h1window.tile_unit`, the statement `pouw_h1_circuits` writes), or its ``lane`` (`h1zy.lane`, one circuit for every
lane, the lane index public) and ``combine`` (`h1zy.combine`), the statements `h1_lanes` stages, which
`h1window.leaf_unit` derives at a K of whole lanes. Each must fit C-Flock's block (``k_log <= 27``).

* ``lower K R C OUT [--unit tile|lane|combine] [--form structural|flat] [--staging fast|library] [--path FILE]``: a tile
  through `pouw_h1_circuits._lower` itself (the ``--lower`` command's body, ``HZ_STAGE`` set to the staging), a lane or
  combine through `h1unit.stager` with `h1_lanes`' rows, in this process, bound to `BIND`; at ``fast`` (the default)
  `fast_stage` patches the library first, for the same bytes. OUT (JSON): the lowering (its class; the rows C-Flock
  proves below the root, the commits among them and the ANDs, `_counts`; the template's Definitions, rows and calls), its
  seconds and peak RSS before compose and at the end, and the file's bytes, SHA-512 and META block fields; or the
  refusal, before lowering (`refusal`) or compose's (a block past ``K_MAX``, ``k_log`` the block it needs) with the
  lowering's counts. The file goes to FILE, else to a scratch directory removed at exit.
* ``agree K R C OUT``: the flat and the structural tile stager at one shape, their units evaluated on the same rows
  (honest leaves, a record off in one bit, random words): each output word of the template (`TmplLowering.evaluate`)
  against the flat unit's (`ir_lower.Lowering.evaluate`) and both against `h1zy.reference`, the word reference; then
  `circuit_check.structural.check_unit`'s verdict on the template against the flat unit.
* ``sweep --shapes K:R:C[:UNIT],.. --out DIR [--form F] [--staging S] [--jobs J]``: ``lower`` of each shape (UNIT
  ``tile`` unless given) in a process of its own, its peak RSS from ``wait4`` (so also when it is killed):
  DIR/<form>-<unit>-k<K>-<R>x<C>.json and DIR/sweep-<form>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VLLM = ROOT / "integrations" / "vllm"
H1_PROVE = ROOT / "benchmarks" / "pouw" / "h1_prove"
#: the guard's bind: the file's bytes are pinned under it at small shapes
BIND = {"partition": "0" * 128, "program": "0" * 128}
#: META's fields that say how the statement fills C-Flock's block
BLOCK = ("format", "statement", "layout", "k_log", "unit_log", "vus_per_block", "per_vu")
SEED = 20261028
UNITS = ("tile", "lane", "combine")
STAGINGS = ("fast", "library")
NAMES = {"tile": "unit/pouw-h1-tile", "lane": "unit/pouw-h1-lane", "combine": "unit/pouw-h1-combine"}


def peak_gib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def _h1unit(form: str):
    os.environ["H1_TMPL"] = form
    for p in (H1_PROVE, H1_PROVE.parent / "hidden_zk", VLLM):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import h1unit
    return h1unit


def refusal(unit: str, k: int, r: int, c: int) -> str | None:
    """Why the R × C leaf at K has no ``unit`` statement, decided before anything is lowered; None when it has one."""
    from verity.core.protocols.pouw import weights as W
    from verity.core.protocols.pouw.circuit import h1zy as Z
    if unit not in UNITS:
        return f"a leaf's statement is one of {', '.join(UNITS)}, not {unit!r}"
    if r <= 0 or c <= 0 or r % 4 or c % 8 or k < 64 or k % 32:
        return (f"the H1 tile is whole lanes of 4 rows and 8 columns at 32 | K >= 64 (`h1tile._Unit`), not {r} × {c} at "
                f"K = {k}")
    if unit == "tile":
        return None
    if r % 16:
        return (f"R = {r}: a lane reads the leaf's R admit bits as one row of whole 16-bit words (`h1zy._leaf`), so lanes "
                f"need 16 | R; prove the {r} × {c} leaf in the whole-tile form (--unit tile)")
    if W.chunks(k) > 1:
        return (f"K = {k}: its rows are {W.chunks(k)} chunk leaves (`weights.chunks`), whose lanes are chunk lanes over "
                f"J10-R's tile (`h1window.leaf_unit`), not `h1zy.lane`'s")
    if unit == "combine" and k < Z.LEAST_K:
        return f"K = {k}: a combine needs K >= {Z.LEAST_K} (`h1zy.combine`: a row that compresses before its last block)"
    return None


def leaf_statement(U, unit: str, k: int, r: int, c: int) -> tuple:
    """``unit``'s Definition (`h1zy.lane` or `h1zy.combine` over the leaf's tile, as `h1window.leaf_unit` derives it at a
    K of whole lanes) and its input rows, as `h1_lanes`' ``stager`` stages them."""
    from verity.core.protocols.pouw import leaf_lanes as LL
    from verity.core.protocols.pouw.circuit import h1tile as H
    from verity.core.protocols.pouw.circuit import h1window as HW
    from verity.core.protocols.pouw.circuit import h1zy as Z
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
    t = HW.h1_tile(k, (r, c), H.PouwH1Tile3 if U.tile() == "v3" else None)
    wp = 2 * LL.words(k)
    if unit == "lane":
        ab = 16 * (k // 2 + 2)
        rows = ([(f"a{s}", V2, ab, [f"a[{s}]"]) for s in range(LL.LANE_ROWS)]
                + [(f"b{s}", V2, ab, [f"b[{s}]"]) for s in range(LL.LANE_COLS)]
                + [(LL.ADMIT, V2, r, [LL.ADMIT]), (LL.ZY, V2, 48 * r * c, [LL.ZY]), (LL.PARTIALS, V2, 16 * wp, [LL.PARTIALS]),
                   ("pub", V2, U.PUB_BITS + 16, list(LL.PUBLIC))])
        return Z.lane(t), rows
    rows = ([(f"p{x}", V2, 16 * wp, [f"{LL.PARTIALS}[{x}]"]) for x in range(LL.lanes(r, c))]
            + [(LL.RECEIPT, V2, 16 * wp, [LL.RECEIPT])])
    return Z.combine(t), rows


def stage(U, unit: str, k: int, r: int, c: int, path: Path, staging: str) -> None:
    """``unit``'s statement file at `path`, bound to `BIND`, staged as its benchmark stages it."""
    if unit == "tile":
        from tests.protocols import pouw_h1_circuits as PC
        os.environ["HZ_STAGE"] = staging
        PC._lower(k, r, c, BIND, path)
        return
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
    if staging == "fast":
        import fast_stage
        fast_stage.install(path.parent / f".scratch-k{k}")
    d, rows = leaf_statement(U, unit, k, r, c)
    U.stager(d, NAMES[unit], rows, [(V2, (0,))], U.layout(), circuit=path, bind=BIND)


def _flat_commits(lay) -> int | None:
    """A flat unit's commit rows (`Circuit.commit`'s ``A = form, B = 1``; an AND of two gates has no constant operand,
    `Circuit.AND`): its rows whose B is the constant column alone, less the assertions', the output copies' and the
    constant's own (`ir_lower.layout`); None when it reads tables, whose plain rows can have that B too."""
    if lay.counts.get("reads") or not hasattr(lay, "rows_b"):
        return None
    ones = sum(b is not None and len(b) == 1 and b[0] == lay.const for b in lay.rows_b)
    return int(ones) - lay.counts["asserts"] - sum(lay.out_bits) - 1


def _keep_totals(CS, got: dict) -> None:
    """`struct_lower.lower` as `class_statement.unit_lowering` calls it, its unit's unrolled totals below the root
    (`_Lowerer.total`: rows, commits, refs, copies) kept in ``got["struct"]``."""
    lower_tmpl = CS.SL.lower

    def kept(*a, **k):
        t, lw, top = lower_tmpl(*a, **k)
        got["struct"] = lw.total(top.key)
        return t, lw, top

    CS.SL.lower = kept


def _counts(low, struct: dict | None = None, commits: int | None = None) -> dict:
    """The lowering's counts. ``rows``: the rows C-Flock proves below the root (a template's from ``struct``, its
    `_Lowerer.total`; a flat unit's AND and table-read rows). ``commits``: those of them that commit a linear form rather
    than AND two (``A = form, B = 1``: `struct_lower`'s past ``support`` at a call boundary, `Circuit.commit`'s).
    ``ands``: a template's rows less its commits; a flat unit's layout count, its commits included, as the harness counts
    it. The layout's own ``ands`` and ``rows`` are ``layout_ands`` and ``layout_rows``; a template's ``layout_ands``
    (`class_statement._TmplLayout`) is its rows below the root, ``rows_agree`` whether it is ``struct``'s."""
    c = low.layout.counts
    out = {"lowering": type(low).__name__, "layout_ands": int(c["ands"]), "layout_rows": int(c["rows"]),
           **{k: int(v) for k, v in c.items() if k not in ("ands", "rows") and isinstance(v, (int, np.integer))}}
    t = getattr(low, "template", None)
    if t is None:
        return out | {"rows": out["layout_ands"], "commits": commits, "ands": out["layout_ands"]}
    out |= {"template_defs": len(t.defs), "template_rows": sum(len(d.rows) for d in t.defs),
            "template_calls": sum(len(d.calls) for d in t.defs),
            "template_refs": sum(len(a) + len(b) for d in t.defs for _c, a, b in d.rows)}
    if struct is not None:
        out |= {**struct, "ands": struct["rows"] - struct["commits"], "rows_agree": struct["rows"] == out["layout_ands"]}
    return out


def _file(path: Path) -> dict:
    """The file's first line, its META line's bytes and its SHA-512, read in chunks: a templated file's META holds the
    whole template."""
    h, newlines, at = hashlib.sha512(), [], 0
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
            start = 0
            while len(newlines) < 2 and (nl := b.find(b"\n", start)) >= 0:
                newlines.append(at + nl)
                start = nl + 1
            at += len(b)
    with open(path, "rb") as f:
        head = f.read(newlines[0]).decode()
    return {"head": head, "meta_bytes": newlines[1] - newlines[0] - 1, "sha512": h.hexdigest()}


def _done(args, rec: dict) -> int:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print("LOWER " + json.dumps({k: rec.get(k) for k in ("k", "r", "c", "unit", "form", "lowering", "rows", "commits",
                                                         "ands", "rows_agree", "k_log", "bytes", "seconds", "peak_gib",
                                                         "refused")}), flush=True)
    return 0


def lower(args) -> int:
    U = _h1unit(args.form)
    from verity_flock import circuit as FC
    from verity_flock import class_statement as CS

    rec = {"k": args.k, "r": args.r, "c": args.c, "tile": [args.r, args.c], "unit": args.unit, "form": args.form,
           "staging": args.staging, "layout": U.layout(), "bind": BIND}
    if why := refusal(args.unit, args.k, args.r, args.c):
        return _done(args, rec | {"written": False, "refused": why, "seconds": 0.0, "peak_gib": peak_gib()})
    got: dict = {}
    library, meta_of = CS.unit_lowering, FC.meta_of

    def captured(*a, **k):
        if args.form == "structural" and not (a[4] if len(a) > 4 else k.get("structural")):
            raise SystemExit(f"{args.unit} at K = {args.k}, {args.r} × {args.c}: struct_lower refused it, and h1unit.stager "
                             "fell back to the flat lowering, which is not the structural form")
        t = time.time()
        low, out_words = library(*a, **k)
        got.update(low=low, lower_s=round(time.time() - t, 2), lower_peak_gib=peak_gib())
        if not hasattr(low, "template"):
            got["flat_commits"] = _flat_commits(low.layout)
        return low, out_words

    def meta_captured(text):
        m = meta_of(text)
        tl = m.get("template")
        got["meta"] = {**{k: m.get(k) for k in BLOCK}, "template_lines": len(tl) if isinstance(tl, list) else None}
        return m

    CS.unit_lowering, FC.meta_of = captured, meta_captured
    _keep_totals(CS, got)
    scratch = None if args.path else tempfile.TemporaryDirectory(prefix="unit-lowering-")
    path = Path(args.path) if args.path else Path(scratch.name) / f"k{args.k}.circuit.txt"
    t0 = time.time()
    try:
        stage(U, args.unit, args.k, args.r, args.c, path, args.staging)
        rec["written"] = True
    except ValueError as e:
        rec |= {"written": False, "refused": str(e)}
        if m := re.search(r"needs a block of 2\^(\d+) bits", str(e)):
            rec["k_log"] = int(m.group(1))
    rec |= {"seconds": round(time.time() - t0, 2), "peak_gib": peak_gib()}
    if "low" in got:
        low = got.pop("low")
        rec |= _counts(low, got.pop("struct", None), got.pop("flat_commits", None))
        del low
    rec |= got.pop("meta", {}) | got
    if rec["written"]:
        rec |= {"bytes": path.stat().st_size, **_file(path)}
        rec["templated"] = rec["head"].startswith("flock-circuit/tmpl ")
    if scratch is not None:
        scratch.cleanup()
    return _done(args, rec)


def _leaf(U, k: int, r: int, c: int, rng) -> tuple:
    a = rng.integers(-128, 128, (r, k)).astype(np.int8)
    b = rng.integers(-128, 128, (c, k)).astype(np.int8)
    sa = (rng.uniform(1e-3, 0.1, r) / 127).astype(np.float32)
    sw = (rng.uniform(1e-3, 0.5, c) / 127).astype(np.float32)
    z, y = U.records(a, b, sa, sw)
    return a, b, sa, sw, z, y


def _instances(U, shape, rng) -> tuple[list[str], dict[str, np.ndarray], list[tuple]]:
    """Each instance's rows by the stager's ports, and its salt, position and roots: two honest leaves, the second again
    with one bit of its record's z off, and two of random words, each at its own salt, position and roots."""
    r, c, k = shape.r, shape.c, shape.k
    names, pubs = [], []
    rows = {**{f"a{i}": [] for i in range(r)}, **{f"b{j}": [] for j in range(c)}, "zy": [], "pub": []}

    def put(name, a, b, sa, sw, zw):
        salt, pos = rng.bytes(64), int(rng.integers(0, 1 << 20))
        roots = (rng.bytes(64), rng.bytes(64))
        A, B = U.row_words(a, sa), U.row_words(b, sw)
        for i in range(r):
            rows[f"a{i}"].append(A[i])
        for j in range(c):
            rows[f"b{j}"].append(B[j])
        rows["zy"].append(zw)
        rows["pub"].append(U.pub_words(salt, pos, roots))
        names.append(name)
        pubs.append((salt, pos, roots))

    for n in range(2):
        a, b, sa, sw, z, y = _leaf(U, k, r, c, rng)
        put(f"honest{n}", a, b, sa, sw, U.record_words(z, y))
    zw = U.record_words(z, y)
    zw[0] ^= 1 << 5
    put("z-bit-off", a, b, sa, sw, zw)
    for n in range(2):
        a, b = (rng.integers(-128, 128, (m, k)).astype(np.int8) for m in (r, c))
        sa, sw = (rng.integers(0, 1 << 32, m, dtype=np.uint64).astype(np.uint32).view(np.float32) for m in (r, c))
        put(f"random{n}", a, b, sa, sw, rng.integers(0, 1 << 16, 3 * r * c).astype(np.uint16))
    return names, {p: np.asarray(v, np.uint16) for p, v in rows.items()}, pubs


def _reference(U, shape, rows: dict, i: int, pub: tuple) -> tuple[np.ndarray, int]:
    """`h1zy.reference` on instance ``i``'s words, as `h1_prove/stage.py` holds a staged round to it."""
    from verity.core.protocols.pouw.circuit import h1zy as Z
    a = [U.split_row(rows[f"a{x}"][i]) for x in range(shape.r)]
    b = [U.split_row(rows[f"b{x}"][i]) for x in range(shape.c)]
    salt, pos, roots = pub
    rc, ok = Z.reference([x for x, _s in a], [x for x, _s in b], [s for _x, s in a], [s for _x, s in b],
                         U.split_record(rows["zy"][i], shape.r * shape.c), salt, pos, roots, stream=U.tile() == "v3")
    return U.receipt_words(rc), ok


def agree(args) -> int:
    U = _h1unit("structural")
    if why := refusal("tile", args.k, args.r, args.c):
        raise SystemExit(why)
    shape = U.Shape(args.k, args.r, args.c)
    name = NAMES["tile"]
    from verity_flock import class_statement as CS
    rec = {"k": args.k, "r": args.r, "c": args.c, "layout": U.layout()}
    got: dict = {}
    _keep_totals(CS, got)
    t = time.time()
    flat = shape.stager(name, flat=True)
    rec |= {"flat_s": round(time.time() - t, 2), "flat": _counts(flat.low, commits=_flat_commits(flat.low.layout))}
    t = time.time()
    tm = shape.stager(name)
    rec |= {"structural_s": round(time.time() - t, 2), "structural": _counts(tm.low, got.get("struct")),
            "class_flat": flat.class_sha512,
            "class_structural": tm.class_sha512}
    if not hasattr(tm.low, "template"):
        raise SystemExit(f"H1_TMPL=structural gave the flat statement at {args.k, args.r, args.c}")
    if flat.row_ports != tm.row_ports or flat.out_words != tm.out_words:
        raise SystemExit(f"the two stagers' ports differ: {flat.row_ports, flat.out_words} and {tm.row_ports, tm.out_words}")
    names, rows, pubs = _instances(U, shape, np.random.default_rng(SEED))
    x = np.concatenate([rows[p] for p in tm.row_ports], axis=1).astype(np.uint64)
    t = time.time()
    got_f, _cut, ok_f = flat.low.evaluate(x)
    rec["flat_eval_s"] = round(time.time() - t, 2)
    t = time.time()
    got_t, _cut, ok_t = tm.low.evaluate(x)
    rec["structural_eval_s"] = round(time.time() - t, 2)
    at = np.cumsum([0] + tm.out_words)
    cases = {}
    for i, nm in enumerate(names):
        rc, ok = _reference(U, shape, rows, i, pubs[i])
        f, s = (np.asarray(g[i], np.uint64).astype(np.uint16) for g in (got_f, got_t))
        cases[nm] = {"reference_ok": ok, "flat_ok": int(f[at[1]]), "structural_ok": int(s[at[1]]),
                     "receipt_words": int(at[1]), "structural_eq_flat": bool(np.array_equal(f, s)),
                     "flat_eq_reference": bool(np.array_equal(f[:at[1]], rc) and f[at[1]] == ok),
                     "structural_eq_reference": bool(np.array_equal(s[:at[1]], rc) and s[at[1]] == ok)}
    rec |= {"cases": cases, "flat_rows_ok": int(ok_f) == (1 << len(names)) - 1, "structural_rows_ok": int(ok_t) == (1 << len(names)) - 1}
    rec["agree"] = rec["flat_rows_ok"] and rec["structural_rows_ok"] and all(
        v["structural_eq_flat"] and v["flat_eq_reference"] and v["structural_eq_reference"] for v in cases.values()) and \
        [cases[n]["reference_ok"] for n in names] == [1, 1, 0, 0, 0]
    if not args.no_check:
        from circuit_check import structural as ST
        tmpl = tm.low.template
        del flat, tm
        t = time.time()
        r = ST.check_unit(shape.definition(), [(p, ps) for p, _s, _b, ps in
                                               U.HW.statement_rows(shape.definition(), U.ports(shape.definition()))],
                          tmpl, name=name)
        rec["check_unit"] = {"verdict": r["structural"]["verdict"], "failures": r["failures"], "warnings": r["warnings"],
                             "structural": r["structural"], "seconds": round(time.time() - t, 2)}
    rec["peak_gib"] = peak_gib()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1, default=str))
    print("AGREE " + json.dumps({"k": args.k, "r": args.r, "c": args.c, "agree": rec["agree"],
                                 "verdict": rec.get("check_unit", {}).get("verdict"), "peak_gib": rec["peak_gib"]}), flush=True)
    return 0 if rec["agree"] and rec.get("check_unit", {}).get("verdict", "syntactic") in ("syntactic", "exhaustive", "random") else 1


def sweep(args) -> int:
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    def one(shape):
        k, r, c, unit = shape
        tag = f"{args.form}-{unit}-k{k}-{r}x{c}"
        cmd = [sys.executable, __file__, "lower", str(k), str(r), str(c), str(out / f"{tag}.json"), "--form", args.form,
               "--unit", unit, "--staging", args.staging]
        t0 = time.time()
        with open(out / "logs" / f"{tag}.log", "w") as log:
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            _pid, status, ru = os.wait4(p.pid, 0)
        rc = os.waitstatus_to_exitcode(status)
        got = out / f"{tag}.json"
        rec = {"tag": tag, "rc": rc, "peak_gib": round(ru.ru_maxrss / 2**20, 3), "wall_s": round(time.time() - t0, 1),
               "result": json.loads(got.read_text()) if got.exists() else None}
        print(f"{tag}: rc {rc}, peak {rec['peak_gib']} GiB, {rec['wall_s']} s", flush=True)
        return rec

    with ThreadPoolExecutor(args.jobs) as pool:
        recs = list(pool.map(one, args.shapes))
    (out / f"sweep-{args.form}.json").write_text(json.dumps(recs, indent=1))
    return 0 if all(r["rc"] == 0 for r in recs) else 1


def shapes(s: str) -> list[tuple[int, int, int, str]]:
    out = []
    for x in s.split(","):
        if x:
            k, r, c, *unit = x.split(":")
            if unit and unit[0] not in UNITS:
                raise argparse.ArgumentTypeError(f"{x}: the unit is one of {', '.join(UNITS)}")
            out.append((int(k), int(r), int(c), unit[0] if unit else "tile"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("lower", "agree"):
        p = sub.add_parser(nm)
        for v in ("k", "r", "c"):
            p.add_argument(v, type=int)
        p.add_argument("out")
        if nm == "lower":
            p.add_argument("--unit", choices=UNITS, default="tile")
            p.add_argument("--form", choices=("structural", "flat"), default="structural")
            p.add_argument("--staging", choices=STAGINGS, default="fast")
            p.add_argument("--path")
        else:
            p.add_argument("--no-check", action="store_true")
    p = sub.add_parser("sweep")
    p.add_argument("--shapes", type=shapes, required=True)
    p.add_argument("--form", choices=("structural", "flat"), default="structural")
    p.add_argument("--staging", choices=STAGINGS, default="fast")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.exit({"lower": lower, "agree": agree, "sweep": sweep}[a.cmd](a))


if __name__ == "__main__":
    main()
