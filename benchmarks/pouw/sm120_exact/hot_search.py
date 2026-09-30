#!/usr/bin/env python3
"""v2-hot's full-unit rerun, the CPU half: the window search over hot.py's per-atom exactness bitsets
(internal/pouw/rtx-pro/fill-candidate-aligned-exact-regions.md, step 2 and "Update, 12:30Z").  CPU only, no lease.

For each unit hot.py has finished (`units/<fam>@<u>/unit.json`) and each chain (`hot`: from v2-hot's H_i, `zero`: from +0),
only words whose A row and B row Π admits counting (`okA.npy`, `okB.npy`):
- for every width w in the fill candidate's table (SHAPES: 4, 6, 8, 9, 12, 14, 16, 18, 24, 27, 36 and 40 .. 72) and every
  start t = 0 .. T - w, AND the atoms [t, t + w);
- for each side length N of the width's shapes: take the N columns with the most exact words (stable order) and count the
  rows exact with all of them (`R_c`), and the same with rows and columns swapped (`R_r`): the largest block per (t, w, N);
- a shape a x b (either way round) is found at (t, w) when R_c(b) >= a, R_c(a) >= b, R_r(b) >= a or R_r(a) >= b.  The
  priced family is the table's priced shapes (at w = 16 the 14-atom ones); the free family is those plus the free-only
  shapes, and at w = 40 .. 72 it is 64 x 1,600, the smallest of the table's 64 x 1,600 .. 64 x 3,840 (a superset);
- where a shape is found, the disjoint paying regions' coverage: the stable column (row) order cut into groups of N, each
  group's R rows making floor(R / a) blocks of a x N; the largest share of the unit's words over the family's shapes and
  both orientations;
- the closest approach: max over (t, w) and shapes of the found test's ratio (1 or more: found).
Per unit and chain: t0 (the first start from which no shape of the family is found at any width), the coverage before
t0 per width, and the closest approach.  The pass condition for the hot chain is no free-family shape at any start.

--mode v1hot searches hot.py --mode v1hot's units (chains `v1hot` and `v1`): a plane is one v1 group, and each group alone
(width 1) against the free 4-atom shapes of pearlc_v1_hot.FREE4 (FREE4_V1: the table's four, plus 1,536 x 1,200 and
2,304 x 800); t0 counts groups.  The pass condition is no free 4-atom shape in any group on the v1hot chain.

Restartable: one .npz and one .json per (unit, chain) in <out>/search, a state file, exit 99 once --budget-min is spent
while items remain or hot.py has units left to write; exit 0 after summary.json.
    python3 benchmarks/pouw/sm120_exact/hot_search.py --units-dir <hot.py's out> --out DIR [--procs 8]
    python3 benchmarks/pouw/sm120_exact/hot_search.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))
import run_result  # noqa: E402
from hot import FAMILIES, MODE_CHAINS  # noqa: E402

PRICED = {8: [(2000, 2048), (1024, 4000), (1440, 2304), (1152, 2880), (960, 5184)],
          9: [(720, 1728), (864, 1440), (432, 2880), (360, 3456), (1152, 1152), (576, 2304), (288, 4608)],
          12: [(864, 864), (432, 1728), (768, 960), (480, 1536), (384, 1920)],
          14: [(576, 864), (432, 1152), (288, 1728), (216, 2304)],
          18: [(576, 576), (288, 1152), (144, 2304), (432, 864), (216, 1728)],
          24: [(384, 768), (192, 1536)],
          27: [(288, 432), (216, 576), (144, 864)]}
PRICED[16] = PRICED[14]
FREE_ONLY = {4: [(1920, 960), (2400, 768), (1600, 1152), (3840, 640)],
             6: [(672, 576), (1152, 336), (1344, 288)],
             8: [(896, 480), (960, 448)],
             9: [(576, 288), (1152, 144), (528, 528)],
             12: [(384, 384), (768, 192)],
             14: [(576, 256), (512, 288)],
             16: [(512, 256), (576, 216), (432, 288)],
             18: [(288, 288), (576, 144), (1344, 96)],
             24: [(384, 192)],
             36: [(288, 144), (672, 96)],
             **{w: [(64, 1600)] for w in range(40, 73)}}
WIDTHS = sorted(set(PRICED) | set(FREE_ONLY))
SHAPES = {w: {"priced": PRICED.get(w, []), "free": PRICED.get(w, []) + FREE_ONLY.get(w, [])} for w in WIDTHS}
SIDES = {w: sorted({x for s in SHAPES[w]["free"] for x in s}) for w in WIDTHS}
# v1hot: a plane is one v1 group (4 atoms), searched alone against the free 4-atom shapes of pearlc_v1_hot.FREE4
FREE4_V1 = [(1920, 960), (2400, 768), (1600, 1152), (1536, 1200), (2304, 800), (3840, 640)]
TABLES = {"v2hot": (WIDTHS, SHAPES, SIDES),
          "v1hot": ([1], {1: {"priced": [], "free": FREE4_V1}}, {1: sorted({x for s in FREE4_V1 for x in s})})}


def popc(x: np.ndarray) -> np.ndarray:
    return np.bitwise_count(x)


def prefix_rows(bits: np.ndarray, cnt: np.ndarray, sides: list[int]) -> np.ndarray:
    """bits: (L, P / 64) uint64, line l's exact partners; cnt its popcounts.  For each N in sides: the partners exact with
    all of the N lines with most exact words (stable order)."""
    order = np.argsort(-cnt, kind="stable")
    nz = int((cnt > 0).sum())
    top = min(max(sides), nz)
    out = np.zeros(len(sides), np.int64)
    if top == 0:
        return out
    acc = np.bitwise_and.accumulate(bits[order[:top]], axis=0)
    got = popc(acc).sum(1, dtype=np.int64)
    for q, N in enumerate(sides):
        out[q] = got[N - 1] if N <= top else 0
    return out


def coverage(bits: np.ndarray, cnt: np.ndarray, a: int, N: int) -> int:
    """Words covered by disjoint a x N blocks: the stable order's lines with at least a partners, in groups of N."""
    order = np.argsort(-cnt, kind="stable")
    G = int((cnt >= a).sum()) // N
    if G == 0:
        return 0
    g = np.bitwise_and.reduce(bits[order[:G * N]].reshape(G, N, -1), axis=1)
    R = popc(g).sum(1, dtype=np.int64)
    return int((R // a * a).sum()) * N


def found_ratio(Rc: dict, Rr: dict, shapes: list) -> float:
    best = 0.0
    for a, b in shapes:
        for R in (Rc, Rr):
            best = max(best, R[b] / a, R[a] / b)
    return best


def search_starts(task) -> dict:
    """The (t, w) results for the given starts of one (unit, chain)."""
    rows_path, cols_path, m, n, T, okA, okB, starts, mode = task
    widths, shapes_of, sides_of = TABLES[mode]
    RB = np.memmap(rows_path, np.uint64, "r", shape=(T, m, n // 64))
    CB = np.memmap(cols_path, np.uint64, "r", shape=(T, n, m // 64))
    maskB = np.packbits(okB, bitorder="little").view(np.uint64)
    maskA = np.packbits(okA, bitorder="little").view(np.uint64)
    out = {}
    for t in starts:
        curR = RB[t] & maskB
        curR[~okA] = 0
        curC = CB[t] & maskA
        curC[~okB] = 0
        done = t + 1
        for w in widths:
            if t + w > T:
                break
            for s in range(done, t + w):
                curR &= RB[s]
                curC &= CB[s]
            done = t + w
            rc = popc(curR).sum(1, dtype=np.int64)
            cc = popc(curC).sum(1, dtype=np.int64)
            total = int(rc.sum())
            if total == 0:
                break
            sides = sides_of[w]
            Rc = dict(zip(sides, prefix_rows(curC, cc, sides).tolist()))
            Rr = dict(zip(sides, prefix_rows(curR, rc, sides).tolist()))
            rec = {"exact": total, "Rc": [Rc[x] for x in sides], "Rr": [Rr[x] for x in sides]}
            for fam in ("priced", "free"):
                shapes = shapes_of[w][fam]
                ratio = found_ratio(Rc, Rr, shapes) if shapes else 0.0
                rec[f"ratio_{fam}"] = ratio
                cov = 0
                if ratio >= 1:
                    for a, b in shapes:
                        for x, y in ((a, b), (b, a)):
                            cov = max(cov, coverage(curC, cc, x, y), coverage(curR, rc, x, y))
                rec[f"cover_{fam}"] = cov
            out[(t, w)] = rec
    return out


def search_item(pool: Pool, ud: Path, chain: str, procs: int, mode: str = "v2hot") -> tuple[dict, dict]:
    meta = json.loads((ud / "unit.json").read_text())
    m, n, T = meta["m"], meta["n"], meta.get("planes", meta["k"] // 32)
    widths, shapes_of, sides_of = TABLES[mode]
    okA, okB = np.load(ud / "okA.npy"), np.load(ud / "okB.npy")
    starts = [list(range(q, T, procs)) for q in range(procs)]
    parts = pool.map(search_starts, [(str(ud / f"{chain}.rows.u32"), str(ud / f"{chain}.cols.u32"), m, n, T, okA, okB, s, mode)
                                     for s in starts])
    res = {k_: v for p in parts for k_, v in p.items()}
    words = int(okA.sum()) * int(okB.sum())
    arrays, summary = {}, {"unit": ud.name, "chain": chain, "m": m, "n": n, "T": T, "admitted_words": words,
                           "mode": mode, "atoms_per_plane": meta.get("atoms_per_plane", 1),
                           "exact_per_plane": meta["chains"][chain].get("exact_per_plane", meta["chains"][chain].get("exact_per_atom"))}
    for fam in ("priced", "free"):
        hits = [(t, w) for (t, w), r in res.items() if r[f"ratio_{fam}"] >= 1]
        t0 = 1 + max(t for t, _ in hits) if hits else 0
        best = max(res.items(), key=lambda kv: kv[1][f"ratio_{fam}"], default=None)
        summary[fam] = {
            "t0": t0, "found": len(hits), "starts_with_shape": sorted({t for t, _ in hits}),
            "widths_with_shape": sorted({w for _, w in hits}),
            "closest": {"t": best[0][0], "w": best[0][1], "ratio": round(best[1][f"ratio_{fam}"], 6)} if best else None,
            "coverage_before_t0": {str(w): round(max((r[f"cover_{fam}"] for (t, w_), r in res.items() if w_ == w and t < t0),
                                                     default=0) / (m * n), 6)
                                   for w in widths if shapes_of[w][fam]}}
    for w in widths:
        L = T - w + 1
        Rc = np.zeros((L, len(sides_of[w])), np.int32)
        Rr = np.zeros_like(Rc)
        ex = np.zeros(L, np.int64)
        cov = np.zeros((L, 2), np.int64)
        rat = np.zeros((L, 2), np.float32)
        for t in range(L):
            r = res.get((t, w))
            if r:
                Rc[t], Rr[t], ex[t] = r["Rc"], r["Rr"], r["exact"]
                cov[t] = r["cover_priced"], r["cover_free"]
                rat[t] = r["ratio_priced"], r["ratio_free"]
        arrays.update({f"w{w}_sides": np.array(sides_of[w]), f"w{w}_Rc": Rc, f"w{w}_Rr": Rr, f"w{w}_exact": ex,
                       f"w{w}_cover": cov, f"w{w}_ratio": rat})
    summary["density"] = {str(w): [round(float(arrays[f"w{w}_exact"].max()) / max(words, 1), 6),
                                   round(float(arrays[f"w{w}_exact"].min()) / max(words, 1), 6)] for w in widths}
    return summary, arrays


def selftest() -> None:
    """The search on a planted unit: a 2,048 x 2,048 block exact over atoms 3 .. 12 in a 4,096² unit of T = 16 random atoms
    (words exact with probability 1/2 elsewhere), checked against a direct recount."""
    import tempfile
    rng = np.random.default_rng(1)
    m = n = 4096
    T = 16
    E = rng.random((T, m, n)) < 0.5
    rows, cols = rng.choice(m, 2048, replace=False), rng.choice(n, 2048, replace=False)
    for t in range(3, 13):
        E[t][np.ix_(rows, cols)] = True
    okA, okB = np.ones(m, bool), np.ones(n, bool)
    with tempfile.TemporaryDirectory() as d:
        rp, cp = Path(d) / "r.u32", Path(d) / "c.u32"
        np.packbits(E, axis=2, bitorder="little").tofile(rp)
        np.packbits(E.transpose(0, 2, 1), axis=2, bitorder="little").tofile(cp)
        res = search_starts((str(rp), str(cp), m, n, T, okA, okB, list(range(T)), "v2hot"))
        res1 = search_starts((str(rp), str(cp), m, n, T, okA, okB, list(range(T)), "v1hot"))
    bad = 0
    for (t, w), r in res.items():
        ok = np.logical_and.reduce(E[t:t + w], axis=0)
        cc = ok.sum(0)
        order = np.argsort(-cc, kind="stable")
        for N, got in zip(SIDES[w], r["Rc"]):
            bad += got != (int(ok[:, order[:N]].all(1).sum()) if N <= (cc > 0).sum() else 0)
    planted = res[(3, 8)]

    def hit(t, w, fam="free"):
        return res.get((t, w), {}).get(f"ratio_{fam}", 0) >= 1

    want = {(3, 8, "priced"): True, (3, 9, "priced"): True, (4, 9, "free"): True, (3, 12, "free"): False,
            (2, 4, "free"): False, (9, 4, "free"): True, (10, 4, "free"): False}
    wrong = [x for x, v in want.items() if hit(*x) != v]
    want1 = {2: False, 3: True, 12: True, 13: False}
    wrong += [("v1hot", t) for t, v in want1.items() if (res1.get((t, 1), {}).get("ratio_free", 0) >= 1) != v]
    print(f"selftest: {len(res)} (t, w) cells, {bad} recount mismatches; planted (3, 8): priced ratio "
          f"{planted['ratio_priced']:.3f}, cover {planted['cover_priced']} (want 4096000); found-test mismatches {wrong}")
    if bad or wrong or planted["cover_priced"] != 2000 * 2048:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units-dir", default=None, help="hot.py's --out")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--units", type=int, default=4)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--budget-min", type=float, default=5.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=tuple(TABLES), default="v2hot",
                    help="v2hot: every width at every atom; v1hot: each v1 group against the free 4-atom shapes")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    t0 = time.time()
    src = Path(args.units_dir)
    out_dir = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    (out_dir / "search").mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / "log.jsonl", "a")
    names = [f"{f}@{u}" for u in range(1, args.units + 1) for f in args.families.split(",") if f]
    items = [(nm, c) for nm in names for c in MODE_CHAINS[args.mode]]
    state_path = out_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": [], "starts": {}}
    pool = Pool(args.procs)
    secs: list[float] = []
    for nm, chain in items:
        key = f"{nm}.{chain}"
        if key in state["done"]:
            continue
        if not (src / "units" / nm / "unit.json").exists():
            continue
        if time.time() - t0 + (max(secs) if secs else 120.0) > 60 * args.budget_min:
            break
        state["starts"][key] = state["starts"].get(key, 0) + 1
        state_path.write_text(json.dumps(state))
        if state["starts"][key] > 3:
            fh.write(json.dumps({"failed": key, "reason": "started 3 times without finishing"}) + "\n")
            sys.exit(1)
        ts = time.time()
        summary, arrays = search_item(pool, src / "units" / nm, chain, args.procs, args.mode)
        summary["seconds"] = round(time.time() - ts, 1)
        np.savez_compressed(out_dir / "search" / f"{key}.npz", **arrays)
        (out_dir / "search" / f"{key}.json").write_text(json.dumps(summary, indent=1) + "\n")
        fh.write(json.dumps({"done": key, "t0": {f: summary[f]["t0"] for f in ("priced", "free")},
                             "closest": {f: summary[f]["closest"] for f in ("priced", "free")},
                             "seconds": summary["seconds"]}) + "\n")
        fh.flush()
        state["done"].append(key)
        state_path.write_text(json.dumps(state))
        secs.append(time.time() - ts)
    pool.close()
    left = [f"{nm}.{c}" for nm, c in items if f"{nm}.{c}" not in state["done"]]
    if left:
        fh.write(json.dumps({"chunk_end": len(left), "seconds": round(time.time() - t0, 1)}) + "\n")
        sys.exit(99)
    write_summary(out_dir, names, fh, args.mode)


def write_summary(out_dir: Path, names: list[str], fh, mode: str = "v2hot") -> None:
    hot, zero = MODE_CHAINS[mode]
    widths, shapes_of, _ = TABLES[mode]
    per = {}
    for nm in names:
        for c in (hot, zero):
            per[f"{nm}.{c}"] = json.loads((out_dir / "search" / f"{nm}.{c}.json").read_text())
    fams = sorted({nm.split("@")[0] for nm in names})
    table = {}
    for f in fams:
        for c in (hot, zero):
            rs = [per[f"{nm}.{c}"] for nm in names if nm.split("@")[0] == f]
            table[f"{f}.{c}"] = {
                fam: {"t0": max(r[fam]["t0"] for r in rs), "units_with_shape": sum(r[fam]["found"] > 0 for r in rs),
                      "closest": max((r[fam]["closest"] for r in rs if r[fam]["closest"]), key=lambda x: x["ratio"], default=None),
                      "coverage_before_t0": {w: max(r[fam]["coverage_before_t0"].get(w, 0) for r in rs)
                                             for w in rs[0][fam]["coverage_before_t0"]}}
                for fam in ("priced", "free")}
    hot_free = sum(table[f"{f}.{hot}"]["free"]["units_with_shape"] for f in fams)
    hot_priced = sum(table[f"{f}.{hot}"]["priced"]["units_with_shape"] for f in fams)
    cond = {"v2hot": "no free-family shape at any start on the hot chain",
            "v1hot": "no free 4-atom shape in any group on the v1hot chain"}[mode]
    doc = {"tool": "pouw_sm120_exact_hot_search", "mode": mode, "chains": [hot, zero], "pass_condition": cond,
           "passed": hot_free == 0, "hot_units_with_free_shape": hot_free, "hot_units_with_priced_shape": hot_priced,
           "families": table, "shapes": {str(w): shapes_of[w] for w in widths}}
    path = out_dir / "summary.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    fh.write(json.dumps({"summary": {x: doc[x] for x in ("passed", "hot_units_with_free_shape", "hot_units_with_priced_shape")}}) + "\n")
    fam0 = "priced" if mode == "v2hot" else "free"
    zero_t0 = max(table[f"{f}.{zero}"][fam0]["t0"] for f in fams)
    run_result.write(path, True, "sm120-exact-hot-search",
                     f"{mode}, {len(names)} units: {hot} chain, units with a free-family shape {hot_free}, priced {hot_priced}; "
                     f"{zero} chain {fam0} t0 up to {zero_t0}",
                     [("units", len(names), "count"), ("hot_units_with_free_shape", hot_free, "count"),
                      ("hot_units_with_priced_shape", hot_priced, "count"),
                      (f"zero_{fam0}_t0_max", zero_t0, "atoms" if mode == "v2hot" else "groups")])


if __name__ == "__main__":
    main()
