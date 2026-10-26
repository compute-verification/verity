"""run.py stage|check|targets|all: a served H1 lane past K = 4096 as its chunks along K (`h1zy.PouwH1LaneZyChunk_v1`,
`h1zy.chunks`), over a 32 × 32 leaf's v1 tile (`h1window.h1_tile`), each chunk one C-Flock statement: whether each chunk's
statement fits C-Flock's 2^27 block, what its template computes and whether it is the flat unit, short of proving it.

* ``stage --k K --chunk C`` (``--chunks N``: N chunks in place of `h1zy.chunks`(K); ``--whole``: the whole lane,
  `h1zy.lane`): the statement's structural stager, `class_statement.UnitStager(..., layout="iv/hybrid/4", structural=True)`
  on its rows (`h1zy.statement_rows`, each row's bits its parameters' leaves, the bits h1_lanes' stager gives a lane's):
  META's k_log, unit_log and slots per VU, the unrolled ANDs and the class digest, and the template
  (`TmplLowering.evaluate`) on lane 0's honest rows and carries of a random leaf and on negatives, each held to
  `h1zy.lane_chunk_reference`; or compose's refusal. OUT/stage/k<K>-c<C>.json (``whole`` for the lane).
* ``check --k K --chunk C``: the same stager, then `circuit_check.structural.check_unit` on its template (META's, the text
  its class digest covers) against the flat unit on the same rows: OUT/check/k<K>-c<C>.json.
* ``targets FILE``: circuit-check's targets, one id a line: h1zy's sidecar chunk bindings (8 × 16) and the 32 × 32 chunks
  at K = 1024 (both of 2) and 1536 (the middle of 3); ``--ks K,.. [--only K:C,..]``: the served 32 × 32 chunks of each K
  in their place.
* ``all stage|check --ks K,.. [--whole K,..] [--only K:C,..] [--chunks N] [--jobs J]``: every chunk of each K, each in a
  process of its own, J at a time, with that process's peak RSS (`os.wait4`'s ``ru_maxrss``, so also when it is killed):
  OUT/<what>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from verity.core.protocols.pouw import leaf_lanes as LL
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.pouw.circuit import h1zy as Z
from verity.ml.circuits.defs import bind
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

RC = (32, 32)
LAYOUT = "iv/hybrid/4"
SALT = hashlib.sha512(b"verity/pouw/h1-splitk/salt/v0").digest()
ROOTS = (hashlib.sha512(b"verity/pouw/h1-splitk/root-a/v0").digest(), hashlib.sha512(b"verity/pouw/h1-splitk/root-b/v0").digest())
POSITION = 5
SEED = 20261024
M32 = (1 << 32) - 1
#: circuit-check's 32 × 32 chunks: (K, chunks, chunk)
DERIVED = ((1024, 2, 0), (1024, 2, 1), (1536, 3, 1))


def peak_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def statement(k: int, c: int, n: int | None = None, whole: bool = False) -> tuple:
    """The lane's statement at K, or its chunk c of n (`h1zy.chunks`(K) unless given), and its unit's name."""
    t = HW.h1_tile(k, RC)
    if whole:
        return Z.lane(t), "unit/pouw-h1-lane", 1
    n = n or Z.chunks(k)
    return Z.lane_chunk(t, n, c), f"unit/pouw-h1-lane-chunk{c}of{n}", n


def rows_of(d) -> list[tuple]:
    """``d``'s input rows as `UnitStager` takes them, ``(port, schema, bits, params)``: each row's bits its parameters' leaves."""
    ty = dict(d.signature()[0])

    def bits(p):
        nm, _, r = p.partition("[")
        return sum((ty[nm].elem if r else ty[nm]).leaf_widths())
    return [(port, V2, sum(bits(p) for p in ps), ps) for port, ps in Z.statement_rows(d)]


def stager(d, name: str):
    from verity_flock import class_statement as CS

    return CS.UnitStager(d, name, rows_of(d), [(V2, (0,))], layout=LAYOUT, structural=True)


# -- a leaf and its chunks' rows ----------------------------------------------------------------------------------------------

def f32(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


def leaf(k: int, seed: int = SEED) -> tuple:
    """A random 32 × 32 leaf at K (s8 encodings, f32 scale encodings) and its honest z/y record."""
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 256, (RC[0], k)).tolist(), rng.integers(0, 256, (RC[1], k)).tolist()
    sa = [f32(x / 127) for x in rng.uniform(1e-3, 0.1, RC[0])]
    sw = [f32(x / 127) for x in rng.uniform(1e-3, 0.5, RC[1])]
    return a, b, sa, sw, Z.records(a, b, sa, sw)


def u16_of_u32(words) -> list[int]:
    return [h for v in words for h in (v & 0xFFFF, (v >> 16) & 0xFFFF)]


def row_words(codes, scale: int) -> list[int]:
    """`h1zy.ROW`'s row: its codes' bytes, then its scale's, little-endian, as u16 words."""
    return [codes[2 * i] | codes[2 * i + 1] << 8 for i in range(len(codes) // 2)] + u16_of_u32([scale])


def bit_words(bits) -> list[int]:
    return [sum((bits[16 * w + i] & 1) << i for i in range(16)) for w in range(len(bits) // 16)]


def chunk_rows(d, n: int, c: int, rows: tuple, cin, cout, admit, zy, partial, idx: int) -> tuple[list[int], int]:
    """Chunk c's input words, its ports in order (`rows_of`), and its ok on words (`h1zy.lane_chunk_reference`), from lane
    rows ``rows`` (4 A, 8 B rows over the whole K and their scales), ``cin`` / ``cout`` (`h1zy.lane_chunk_carries`'),
    the last chunk's ``admit``, ``zy`` and ``partial``, and the lane index the ``pub`` row holds."""
    ra, rb, ras, rbs = rows
    kc, last = len(ra[0]) // n, c == n - 1
    ca, cb = [r[c * kc:c * kc + kc] for r in ra], [r[c * kc:c * kc + kc] for r in rb]
    w = {f"a{s}": row_words(ca[s], ras[s]) for s in range(4)} | {f"b{t}": row_words(cb[t], rbs[t]) for t in range(8)}
    if c:
        w["carry_in"] = u16_of_u32(Z.carry_words(cin))
    if last:
        w |= {"admit": bit_words(admit), "zy": u16_of_u32(zy[0]) + [v & 0xFFFF for v in zy[1]],
              "partials": u16_of_u32(partial)}
    else:
        w["carry_out"] = u16_of_u32(Z.carry_words(cout))
    w["pub"] = LL.pub_words(SALT, POSITION, ROOTS, idx)
    out = []
    for port, _s, bits, _ps in rows_of(d):
        if len(w[port]) * 16 != bits:
            raise AssertionError(f"{port}: {len(w[port])} words for a {bits}-bit row")
        out += w[port]
    want = Z.lane_chunk_reference(ca, cb, ras, rbs, admit if last else None, zy if last else None, partial if last else None,
                                  cin if c else None, None if last else cout, SALT, POSITION, ROOTS, idx, RC, n, c)
    return out, want


def off(carry: dict, field: str, at: int, bit: int) -> dict:
    out = {f: [list(v) for v in w] if f == "res" else list(w) for f, w in carry.items()}
    if field == "res":
        out["res"][at // 32][at % 32] ^= 1 << bit
    else:
        out[field][at] ^= 1 << bit
    return out


def cases(d, k: int, n: int, c: int) -> list[tuple[str, int, list[int], int]]:
    """Lane 0's chunk c on a random leaf: ``(case, ok it should have, input words, the word reference's ok)``: its honest
    rows and carries, an index past the leaf's lanes, a carry in off in its state and in a carried scale, a carry out off in
    a chain word, the last chunk's partial off in a bit, one of its own cells off and another lane's cell off."""
    a, b, sa, sw, zy = leaf(k)
    lr, lc = LL.lane_rows(*RC, 0), LL.lane_cols(*RC, 0)
    rows = ([a[i] for i in lr], [b[j] for j in lc], [sa[i] for i in lr], [sw[j] for j in lc])
    cs = Z.lane_chunk_carries(*rows, SALT, POSITION, ROOTS, 0, RC, n)
    last, live, part = c == n - 1, [1] * RC[0], cs[-1]["words"]
    cin, cout = cs[c - 1] if c else None, None if last else cs[c]
    spec = [("honest", 1, cin, cout, zy, part, 0), (f"index-{LL.lanes(*RC)}", 0, cin, cout, zy, part, LL.lanes(*RC))]
    if c:
        spec += [("carry-in-state", 0, off(cin, "state", 11, 22), cout, zy, part, 0),
                 ("carry-in-sw", 0, off(cin, "sw", 3, 0), cout, zy, part, 0)]
    if not last:
        spec.append(("carry-out-words", 0, cin, off(cout, "words", 2 * c, 17), zy, part, 0))
    else:
        own, other = (list(zy[0]), zy[1]), (list(zy[0]), zy[1])
        own[0][Z.lane_cells(H.Layout(*RC, k), 0)[5]] ^= 8
        other[0][Z.lane_cells(H.Layout(*RC, k), 1)[5]] ^= 8
        spec += [("partial-bit", 0, cin, cout, zy, [part[0] ^ 1 << 3, *part[1:]], 0), ("own-cell", 0, cin, cout, own, part, 0),
                 ("other-lanes-cell", 1, cin, cout, other, part, 0)]
    out = []
    for name, should, ci, co, rec, p, idx in spec:
        words, want = chunk_rows(d, n, c, rows, ci, co, live, rec, p, idx)
        out.append((name, should, words, want))
    return out


# -- the commands -------------------------------------------------------------------------------------------------------------

def tag(k: int, c: int, whole: bool) -> str:
    return f"k{k}-whole" if whole else f"k{k}-c{c}"


def stage(args) -> int:
    out = Path(args.out) / "stage"
    out.mkdir(parents=True, exist_ok=True)
    d, name, n = statement(args.k, args.chunk, args.chunks, args.whole)
    rec = {"k": args.k, "chunks": n, "chunk": None if args.whole else args.chunk, "kc": args.k // n, "definition": d.id,
           "unit": name, "rows": [[p, b] for p, _s, b, _ps in rows_of(d)]}
    t0 = time.time()
    try:
        st = stager(d, name)
    except ValueError as e:
        rec |= {"fits": False, "refused": str(e), "stage_s": round(time.time() - t0, 1), "peak_gb": peak_gb()}
        (out / f"{tag(args.k, args.chunk, args.whole)}.json").write_text(json.dumps(rec, indent=1))
        print("STAGE " + json.dumps({x: rec[x] for x in ("k", "chunk", "fits", "refused", "peak_gb")}), flush=True)
        return 0
    m = st.meta
    rec |= {"fits": True, "k_log": m["k_log"], "unit_log": m["unit_log"], "vus_per_block": m["vus_per_block"],
            "per_vu": m["per_vu"], "ranges": m["ranges"], "counts": st.low.layout.counts, "class_sha512": st.class_sha512,
            "stage_s": round(time.time() - t0, 1), "stage_peak_gb": peak_gb()}
    print(f"staged {name} at K = {args.k}: k_log {m['k_log']}, unit_log {m['unit_log']}, {st.low.layout.counts['ands']} ANDs, "
          f"{rec['stage_s']} s, {rec['stage_peak_gb']} GB", flush=True)
    if not args.whole and not args.no_eval:
        t1 = time.time()
        cs = cases(d, args.k, n, args.chunk)
        rec["reference_s"] = round(time.time() - t1, 1)
        t1 = time.time()
        got, _cuts, _mask = st.low.evaluate(np.asarray([w for _c, _s, w, _r in cs], np.uint64))
        rec["evaluate_s"] = round(time.time() - t1, 1)
        rec["cases"] = {c: {"should": s, "reference": r, "template": int(got[i][0]) & 1} for i, (c, s, _w, r) in enumerate(cs)}
        rec["evaluated"] = all(v["should"] == v["reference"] == v["template"] for v in rec["cases"].values())
    rec["peak_gb"] = peak_gb()
    (out / f"{tag(args.k, args.chunk, args.whole)}.json").write_text(json.dumps(rec, indent=1))
    print("STAGE " + json.dumps({x: rec.get(x) for x in ("k", "chunk", "fits", "k_log", "unit_log", "evaluated", "peak_gb")}),
          flush=True)
    return 0 if rec.get("evaluated", True) else 1


def check(args) -> int:
    from circuit_check import structural as ST

    out = Path(args.out) / "check"
    out.mkdir(parents=True, exist_ok=True)
    d, name, n = statement(args.k, args.chunk, args.chunks, args.whole)
    t0 = time.time()
    st = stager(d, name)
    rec = {"k": args.k, "chunks": n, "chunk": None if args.whole else args.chunk, "definition": d.id, "unit": name,
           "k_log": st.meta["k_log"], "class_sha512": st.class_sha512, "stage_s": round(time.time() - t0, 1),
           "stage_peak_gb": peak_gb()}
    tm = st.low.template
    del st
    t1 = time.time()
    r = ST.check_unit(d, [(p, ps) for p, _s, _b, ps in rows_of(d)], tm, name=name)
    rec |= {"verdict": r["structural"]["verdict"], "failures": r["failures"], "warnings": r["warnings"],
            "structural": r["structural"], "check_s": round(time.time() - t1, 1), "peak_gb": peak_gb()}
    (out / f"{tag(args.k, args.chunk, args.whole)}.json").write_text(json.dumps(rec, indent=1, default=str))
    print("CHECK " + json.dumps({x: rec[x] for x in ("k", "chunk", "verdict", "check_s", "peak_gb")}), flush=True)
    return 0 if rec["verdict"] in ("syntactic", "exhaustive", "random") and not r["failures"] else 1


def targets(args) -> int:
    path = Path(__file__).resolve().parents[3] / "verity/core/protocols/pouw/circuit/h1zy.circuit-check.py"
    spec = importlib.util.spec_from_file_location("h1zy_circuit_check", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if args.ks:
        ids = [Z.lane_chunk(HW.h1_tile(k, RC), Z.chunks(k), c).id for k in args.ks for c in range(Z.chunks(k))
               if args.only is None or (k, c) in args.only]
    else:
        ids = [d.id for d in m._pouw_roots() if d.defn is Z.PouwH1LaneZyChunk]
        ids += [Z.lane_chunk(HW.h1_tile(k, RC), n, c).id for k, n, c in DERIVED]
    Path(args.file).write_text("\n".join(ids) + "\n")
    print(f"{len(ids)} targets: " + ", ".join(i[:48] + "…" + i[-22:] for i in ids), flush=True)
    return 0


def run_all(args) -> int:
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    jobs = [(k, None, True) for k in args.whole] + [(k, c, False) for k in args.ks for c in range(args.chunks or Z.chunks(k))]
    if args.only:
        jobs = [j for j in jobs if (j[0], j[1]) in args.only or (j[2] and (j[0], "whole") in args.only)]

    def one(job):
        k, c, whole = job
        t = tag(k, c, whole)
        cmd = [sys.executable, __file__, args.what, "--k", str(k), "--out", str(out)]
        cmd += ["--whole"] if whole else ["--chunk", str(c)] + (["--chunks", str(args.chunks)] if args.chunks else [])
        t0 = time.time()
        with open(out / "logs" / f"{args.what}-{t}.log", "w") as log:
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            _pid, status, ru = os.wait4(p.pid, 0)
        rc = os.waitstatus_to_exitcode(status)
        got = out / args.what / f"{t}.json"
        rec = json.loads(got.read_text()) if got.exists() else None
        print(f"{args.what} {t}: rc {rc}, peak {ru.ru_maxrss / 2**20:.2f} GB, {time.time() - t0:.0f} s", flush=True)
        return {"tag": t, "rc": rc, "peak_gb": round(ru.ru_maxrss / 2**20, 2), "seconds": round(time.time() - t0, 1),
                "result": rec}

    with ThreadPoolExecutor(args.jobs) as pool:
        got = list(pool.map(one, jobs))
    (out / f"{args.what}.json").write_text(json.dumps(got, indent=1, default=str))
    return 0 if all(g["rc"] == 0 for g in got) else 1


def ks(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def only(s: str) -> set[tuple[int, int | str]]:
    return {(int(x.split(":")[0]), x.split(":")[1] if x.split(":")[1] == "whole" else int(x.split(":")[1]))
            for x in s.split(",") if x}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("stage", "check"):
        p = sub.add_parser(nm)
        p.add_argument("--k", type=int, required=True)
        p.add_argument("--chunk", type=int, default=0)
        p.add_argument("--chunks", type=int, default=None)
        p.add_argument("--whole", action="store_true")
        p.add_argument("--out", required=True)
        if nm == "stage":
            p.add_argument("--no-eval", action="store_true")
    p = sub.add_parser("targets")
    p.add_argument("file")
    p.add_argument("--ks", type=ks, default=[])
    p.add_argument("--only", type=only, default=None)
    p = sub.add_parser("all")
    p.add_argument("what", choices=("stage", "check"))
    p.add_argument("--ks", type=ks, default=[])
    p.add_argument("--whole", type=ks, default=[])
    p.add_argument("--only", type=only, default=None)
    p.add_argument("--chunks", type=int, default=None)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.exit({"stage": stage, "check": check, "targets": targets, "all": run_all}[a.cmd](a))


if __name__ == "__main__":
    main()
