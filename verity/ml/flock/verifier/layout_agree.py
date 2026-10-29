"""Conformance of the Lean layout check (`flock-verify layout`, `Flock/Layout.lean`) with the reference,
`verity_flock.layouts`: the same verdict, and on acceptance the same layout digests and own-region sizes.

    python layout_agree.py --lean BIN --out DIR [--seed S] [--n N]

Each case is an archive: random flat types (ANDs, reads of toy program tables) under random callers, one layout per type
(calls placed or inlined, reads placed at a generated `table/v2` layout or inline, callees packed at aligned offsets past
the own region), the unit's placed by the block; one mutation per rule of each; wide units up to attention's 1,092 calls
and 131 reads; and reads whose held table doesn't match the reading type's entry. Writes DIR/<i>.json and DIR/result.json;
exits 1 unless the Lean verifier agrees with the reference on every archive.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from circuit_type_agree import flat, toy_table, tree, _width  # noqa: E402

from verity.ml.circuits import library  # noqa: E402
from verity_flock import circuit_types as CT  # noqa: E402
from verity_flock import layouts as LY  # noqa: E402


def words_of(rng, ib, vb):
    return [rng.getrandbits(max(1, vb - 2)) for _ in range(1 << ib)]


def ceil_log2(n):
    return max(7, (n - 1).bit_length())


def make_layouts(rng, types, order, unit_digest, info):
    """One layout per type, callees first; a generated read layout per placed read."""
    layouts: list[LY.Layout] = []
    index: dict[str, int] = {}
    gens: dict[tuple, int] = {}
    for d in order:
        t = types[d]
        is_unit = d == unit_digest
        calls, reads = [], []
        for i, it in enumerate(t.items):
            if it[0] == CT.CALL:
                jj = index[t.calls[it[1]]]
                inline = LY._places_nothing(layouts, jj) and rng.random() < 0.4
                calls.append((i, jj, None if inline else 0))
            elif it[0] == CT.READ:
                sha, ib, vb = t.tables[it[1]]
                lo = rng.randint(2, ib - 1)
                if rng.random() < 0.5:
                    reads.append((i, None, lo))
                else:
                    key = (sha, ib, vb, lo)
                    if key not in gens:
                        n = LY.read_rows_placed(ib, lo, LY.k_of(sha, vb, info(sha)))
                        layouts.append(LY.Layout(None, ceil_log2(n), (), (), ("gen", sha, ib, vb, lo), (), ()))
                        gens[key] = len(layouts) - 1
                    reads.append((i, gens[key], 0))
        lay = LY.Layout(d, None if is_unit else 7, (), (), ("derive",), tuple(calls), tuple(reads))
        layouts.append(lay)
        j = len(layouts) - 1
        size = LY.own_size(types, layouts, j, info)
        placed = [("c", n, e) for n, e in enumerate(calls) if e[2] is not None] + \
                 [("r", n, e) for n, e in enumerate(reads) if e[1] is not None]
        if is_unit:
            slot: dict[int, int] = {}
            for kind, n, e in placed:
                jj = e[1]
                if kind == "c":
                    calls[n] = (e[0], jj, slot.get(jj, 0))
                else:
                    reads[n] = (e[0], jj, slot.get(jj, 0))
                slot[jj] = slot.get(jj, 0) + 1
            layouts[j] = dataclasses.replace(lay, calls=tuple(calls), reads=tuple(reads))
        else:
            at = size
            for kind, n, e in placed:
                s = 1 << layouts[e[1]].range_log
                q = -(-at // s)
                if kind == "c":
                    calls[n] = (e[0], e[1], q)
                else:
                    reads[n] = (e[0], e[1], q)
                at = (q + 1) * s
            layouts[j] = dataclasses.replace(lay, calls=tuple(calls), reads=tuple(reads), range_log=ceil_log2(at))
        index[d] = j
    return layouts, index[unit_digest]


def archive(types_order, types, tables, layouts, unit):
    return {"types": [types[d].content() for d in types_order],
            "tables": {sha: {"words": ws} for sha, ws in tables.items()},
            "layouts": [l.to_json() for l in layouts], "unit_layout": unit}


def reference(a):
    """The reference's verdict: (ok, digests, sizes) or (False, error)."""
    try:
        types: dict[str, CT.CircuitType] = {}
        for c in a["types"]:
            t = CT.from_content(c)
            CT.validate(t, lambda x: types[x] if x in types else (_ for _ in ()).throw(CT.InvalidType("callee")))
            types[t.digest()] = t
        tabs = {}
        for sha, spec in a["tables"].items():
            ws = spec["words"]
            if hashlib.sha512(b"".join(w.to_bytes(4, "little") for w in ws)).hexdigest() != sha:
                raise LY.InvalidLayout("table words")
            tabs[sha] = LY.table_info(ws)
        layouts = [LY.from_json(r) for r in a["layouts"]]
        sizes = LY.validate(types, layouts, a["unit_layout"], tabs.get)
        return True, [l.digest() for l in layouts], sizes
    except (CT.InvalidType, LY.InvalidLayout, KeyError, IndexError, TypeError) as e:
        return False, repr(e), None


def mutations(a, rng):
    """One broken copy of an accepted archive per rule that applies."""
    out = []
    L = a["layouts"]
    u = a["unit_layout"]

    def with_layout(j, **changes):
        b = json.loads(json.dumps(a))
        r = {k: v for k, v in b["layouts"][j].items() if k != "sha512"}
        for k, v in changes.items():
            if v is None:
                r.pop(k, None)
            else:
                r[k] = v
        b["layouts"][j] = {**r, "sha512": hashlib.sha512(LY.FORMAT.encode() + b"\0" + _canon(r)).hexdigest()}
        return b

    out.append(("sha512", {**a, "layouts": [{**L[0], "sha512": "0" * 128}] + L[1:]}))
    out.append(("unit-range", with_layout(u, range_log=12)))
    inner = [j for j in range(len(L)) if j != u]
    if inner:
        out.append(("inner-no-range", with_layout(inner[0], range_log=None)))
    out.append(("unreached", {**a, "layouts": L + [L[0]]}))
    for j, r in enumerate(L):
        if r["calls"]:
            e = dict(r["calls"][0])
            out.append(("item", with_layout(j, calls=[{**e, "item": e["item"] + 1}] + r["calls"][1:])))
            if "offset" in e and j != u:
                out.append(("own-overlap", with_layout(j, calls=[{**e, "offset": 0}] + r["calls"][1:])))
                out.append(("out-of-range", with_layout(j, calls=[{**e, "offset": 1 << 30}] + r["calls"][1:])))
            if "offset" in e and j == u:
                out.append(("slot", with_layout(j, calls=[{**e, "offset": e["offset"] + 1}] + r["calls"][1:])))
            break
    for j, r in enumerate(L):
        if r["reads"] and "inline" in r["reads"][0]:
            e = r["reads"][0]
            out.append(("read-split", with_layout(j, reads=[{**e, "inline": {"lo_bits": 1}}] + r["reads"][1:])))
            break
    for j, r in enumerate(L):
        if "gen" in r["own"]:
            out.append(("gen-width", with_layout(j, own={**r["own"], "in_bits": r["own"]["in_bits"] + 1})))
            out.append(("gen-value-width", with_layout(j, own={**r["own"], "value_bits": r["own"]["value_bits"] + 1})))
            out.append(("gen-range", with_layout(j, range_log=7) if r.get("range_log", 7) > 7 else with_layout(j, own={**r["own"], "lo_bits": 1})))
            break
    if a["tables"]:
        out.append(("table-not-held", {**a, "tables": {}}))
    out.append(("split", with_layout(u, in_split=[5])))
    return out


def _table(ws):
    return hashlib.sha512(b"".join(w.to_bytes(4, "little") for w in ws)).hexdigest()


def fanout(m, r):
    """The red team's reach_fanout (the layouts' review): a unit calling one small type m times and reading a 16-word table r
    times through one placed generated read, over 3 layouts."""
    callee = CT.CircuitType(library.PROGRAM, ((1, 1),), ((1, 1),), (), (), ((CT.AND, ((0,), 0), ((0,), 1)),), (((1,), 0),))
    ws = [(i * 7 + 3) & 0xF for i in range(16)]
    sha = _table(ws)
    items = [(CT.CALL, 0, (((i,), 0),)) for i in range(m)] + [(CT.READ, 0, tuple(((i % m,), 0) for _ in range(4))) for i in range(r)]
    outs = [((m + i,), 0) for i in range(m)] + [((2 * m + 4 * i + b,), 0) for i in range(r) for b in range(4)]
    unit = CT.CircuitType(library.PROGRAM, ((m, 1),), ((len(outs), 1),), (callee.digest(),), ((sha, 4, 4),), tuple(items), tuple(outs))
    layouts = [LY.Layout(callee.digest(), 10, (), (), ("derive",), (), ()), LY.Layout(None, 10, (), (), ("gen", sha, 4, 4, 2), (), ()),
               LY.Layout(unit.digest(), None, (), (), ("derive",), tuple((i, 0, i) for i in range(m)), tuple((m + i, 1, i) for i in range(r)))]
    return archive([callee.digest(), unit.digest()], {callee.digest(): callee, unit.digest(): unit}, {sha: ws}, layouts, 2)


def shape_cases():
    """Reads whose held table doesn't match the reading type's entry (the red team's ib_mismatch and shape_mismatch):
    a 3-bit index declared for a 16-word table, placed and inline; a 12-word table; a generated read of the wrong value
    width. Each must be refused."""
    ws16 = [(i * 7 + 3) & 0xF for i in range(16)]
    ws12 = ws16[:12]
    out = []
    for ws, ib, vb, gen in ((ws16, 3, 4, (4, 4)), (ws16, 3, 4, None), (ws12, 4, 4, None), (ws16, 4, 4, (4, 2))):
        sha = _table(ws)
        t = CT.CircuitType(library.PROGRAM, ((ib, 1),), ((vb, 1),), (), ((sha, ib, vb),),
                           ((CT.READ, 0, tuple(((b,), 0) for b in range(ib))),), tuple(((ib + b,), 0) for b in range(vb)))
        if gen:
            layouts = [LY.Layout(None, 10, (), (), ("gen", sha, gen[0], gen[1], 2), (), ()),
                       LY.Layout(t.digest(), None, (), (), ("derive",), (), ((0, 0, 0),))]
            out.append(archive([t.digest()], {t.digest(): t}, {sha: ws}, layouts, 1))
        else:
            out.append(archive([t.digest()], {t.digest(): t}, {sha: ws}, [LY.Layout(t.digest(), None, (), (), ("derive",), (), ((0, None, 2),))], 0))
    return out


def _canon(r):
    from verity.ml.circuits.codec import canonical_json
    return canonical_json(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [fanout(m, r) for m, r in ((2, 2), (6, 6), (20, 20), (200, 40), (1092, 131))] + shape_cases()
    for _ in range(a.n):
        tables = {}
        tlist = []
        for ib, vb in ((3, 4), (4, 6), (5, 8)):
            ws = words_of(rng, ib, vb)
            sha = _table(ws)
            tables[sha] = ws
            tlist.append((sha, ib, vb))
        info = {sha: LY.table_info(tables[sha]) for sha, _ib, _vb in tlist}.get
        pool = [flat(rng, tlist) for _ in range(rng.randint(1, 3))]
        for u in pool:
            _width[u.digest()] = u.out_bits
        caller = tree(rng, pool)
        _width[caller.digest()] = caller.out_bits
        used = [t for t in pool if t.digest() in caller.calls]
        top = [caller]
        if rng.random() < 0.5:
            n_in, n_out = caller.in_bits, caller.out_bits
            top.append(CT.CircuitType(caller.label, ((n_in, 1),), ((n_out, 1),), (caller.digest(),), (),
                                      ((CT.CALL, 0, tuple(((i,), 0) for i in range(n_in))),),
                                      tuple(((n_in + i,), 0) for i in range(n_out))))
        types = {t.digest(): t for t in used + top}
        order = list(dict.fromkeys(t.digest() for t in used + top))
        layouts, unit = make_layouts(rng, types, order, top[-1].digest(), info)
        base = archive(order, types, tables, layouts, unit)
        cases.append(base)
        cases.extend(m for _, m in mutations(base, rng))
    files = []
    for i, c in enumerate(cases):
        f = out / f"{i}.json"
        f.write_text(json.dumps(c))
        files.append(str(f))
    p = subprocess.run([a.lean, "layout", *files], capture_output=True, text=True, check=False)
    lean = {}
    for line in p.stdout.splitlines():
        if line.startswith("LAYOUT "):
            r = json.loads(line[len("LAYOUT "):])
            lean[r["file"]] = r
    bad, accepted = [], 0
    for f, c in zip(files, cases):
        ok, ds, sizes = reference(c)
        g = lean.get(f)
        if g is None or g["ok"] != ok or (ok and (g["digests"] != ds or g["sizes"] != sizes)):
            bad.append({"file": f, "reference": [ok, ds if ok else str(ds)[:200], sizes], "lean": g})
        accepted += ok
    result = {"archives": len(cases), "accepted": accepted, "refused": len(cases) - accepted, "n_disagreements": len(bad),
              "disagreements": bad[:10]}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "disagreements"}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
