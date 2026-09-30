#!/usr/bin/env python3
"""An independent check of one hot_search.py hit: rebuild the block from the unit's row bitsets and replay words of it on the
CPU chain from H_i.

For one unit (hot.py's `units/<fam>@<u>`), chain (`hot`: from H.npy's H_i; `zero`: from +0), start t and width w (default:
the smallest width whose free-family ratio in `<search>/<unit>.<chain>.npz` is >= 1 at t):
- E = the words exact at every atom of [t, t + w), from `<chain>.rows.u32` unpacked to bools (not the search's bit ops),
  restricted to Π's admitted rows and columns (okA, okB);
- for each of the width's free-family shapes a x b, both ways round: the b columns (or rows) with the most exact words, in
  stable order, and the lines exact with all of them.  The first shape with at least a such lines is the block;
- --words sampled words of the block: both rows formed on the CPU by pearl_c.form_v1 on SM120_UNPROMOTED (the unit's salt,
  seeds, lines and bases as hot.py draws them), then pearlc_hot_start.chain from H_i (or +0) over atoms 0 .. t + w - 1.
  Each word must be exact at every atom of [t, t + w), and its flags must equal the bitsets' at every atom it replays.
Exit 4 if no block is found, a sampled word is inexact on the window, or a flag differs from the bitsets.

  PYTHONPATH=<04734fb5>/packages/verity/src:<04734fb5>/protocols/pouw python3 benchmarks/pouw/sm120_exact/hot_check.py \
      --units-dir <hot.py's out> --search <hot_search.py's out> --unit saturated@1 --t 0 --ref <pearlc_hot_start.py's dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))
import hot  # noqa: E402
import run_result  # noqa: E402
from hot_search import SHAPES, WIDTHS  # noqa: E402


def exact_window(rows_path: Path, m: int, n: int, T: int, t: int, w: int, okA, okB) -> np.ndarray:
    R = np.memmap(rows_path, np.uint32, "r", shape=(T, m, n // 32))
    E = okA[:, None] & okB[None, :]
    for s in range(t, t + w):
        E &= np.unpackbits(np.ascontiguousarray(R[s]).view(np.uint8), axis=1, bitorder="little").astype(bool)
    return E


def find_block(E: np.ndarray, shapes: list) -> dict | None:
    for a, b in shapes:
        for x, y in ((a, b), (b, a)):
            for axis in (0, 1):
                F = E if axis == 0 else E.T
                order = np.argsort(-F.sum(0), kind="stable")[:y]
                lines = np.flatnonzero(F[:, order].all(1))
                if len(lines) >= x:
                    lines = lines[:x]
                    rows, cols = (lines, order) if axis == 0 else (order, lines)
                    return {"shape": [x, y], "anchored_on": "columns" if axis == 0 else "rows",
                            "rows": np.sort(rows), "cols": np.sort(cols)}
    return None


def _form(task):
    x, e, f = task
    return hot.C.form_v1([x], [e], f, hot.D.SM120_UNPROMOTED).codes[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units-dir", required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--unit", required=True, help="<family>@<u>")
    ap.add_argument("--chain", default="hot", choices=hot.CHAINS)
    ap.add_argument("--t", type=int, required=True)
    ap.add_argument("--w", type=int, default=None)
    ap.add_argument("--words", type=int, default=128)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ref = hot.load_ref(args.ref)
    ud = Path(args.units_dir) / "units" / args.unit
    meta = json.loads((ud / "unit.json").read_text())
    m, n, k, T, fam, u = meta["m"], meta["n"], meta["k"], meta["planes"], meta["family"], meta["unit"]
    okA, okB = np.load(ud / "okA.npy"), np.load(ud / "okB.npy")
    t, w = args.t, args.w
    if w is None:
        z = np.load(Path(args.search) / "search" / f"{args.unit}.{args.chain}.npz")
        w = next((x for x in WIDTHS if t < z[f"w{x}_ratio"].shape[0] and z[f"w{x}_ratio"][t, 1] >= 1), None)
        if w is None:
            raise SystemExit(f"no free-family hit at t = {t} in the search's arrays")
    E = exact_window(ud / f"{args.chain}.rows.u32", m, n, T, t, w, okA, okB)
    blk = find_block(E, SHAPES[w]["free"])
    res = {"tool": "pouw_sm120_exact_hot_check", "unit": args.unit, "chain": args.chain, "t": t, "w": w,
           "family": fam, "salt": meta["salt"], "exact_words_on_window": int(E.sum()), "ref": ref}
    ok = blk is not None
    if blk:
        res["block"] = {"shape": blk["shape"], "anchored_on": blk["anchored_on"],
                        "all_exact_in_bitsets": bool(E[np.ix_(blk["rows"], blk["cols"])].all())}
        rng = np.random.default_rng(0)
        ii = rng.choice(blk["rows"], args.words)
        jj = rng.choice(blk["cols"], args.words)
        xa, xb, salt = hot.draw(fam, k, m, u)
        assert salt == meta["salt"], (salt, meta["salt"])
        C, P = hot.C, hot.P
        seed_b = C._labelled(salt.encode() + b"B", "seed-B")
        seed_a = C._labelled(salt.encode() + b"A", "seed-A")
        FA = np.array(C._basis(seed_b, 0, k, C.LINE_NORM_V1), np.uint8).tolist()
        FB = np.array(C._basis(seed_b, 1, k, C.LINE_NORM_V1), np.uint8).tolist()

        def line(seed, side, i):
            return np.array(P.sample_line(seed, side, 0, i), np.uint8).tolist()

        ra, rb = sorted(set(ii.tolist())), sorted(set(jj.tolist()))
        with Pool(args.procs, initializer=hot.load_ref, initargs=(args.ref,)) as pool:
            ca = dict(zip(ra, pool.map(_form, [(xa[i].tolist(), line(seed_a, 0, i), FA) for i in ra])))
            cb = dict(zip(rb, pool.map(_form, [(xb[j].tolist(), line(seed_b, 1, j), FB) for j in rb])))
        H = np.load(ud / "H.npy")
        R = np.memmap(ud / f"{args.chain}.rows.u32", np.uint32, "r", shape=(T, m, n // 32))
        inexact = flag_mismatch = 0
        for i, j in zip(ii.tolist(), jj.tolist()):
            h = int(H[i]) if args.chain == "hot" else 0
            flags, _ = hot.HS.chain(ca[i][:32 * (t + w)], cb[j][:32 * (t + w)], hot.D.SM120_UNPROMOTED, h, t + w)
            inexact += not all(flags[t:t + w])
            flag_mismatch += sum(f != hot.bit(R, s, i, j) for s, f in enumerate(flags))
        res["words"] = {"sampled": args.words, "rows": len(ra), "cols": len(rb), "inexact_on_window": inexact,
                        "flag_mismatches_vs_bitsets": flag_mismatch, "atoms_replayed": t + w}
        ok = ok and res["block"]["all_exact_in_bitsets"] and inexact == 0 and flag_mismatch == 0
    res["ok"] = ok
    out = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.unit}.{args.chain}.t{t}.w{w}.json"
    path.write_text(hot.jdump(res, indent=1) + "\n")
    print(hot.jdump({x: res[x] for x in res if x != "ref"}))
    detail = (f"{args.unit} {args.chain} chain, atoms [{t}, {t + w}): " +
              (f"block {res['block']['shape']} found; {res['words']['sampled']} sampled words replayed from "
               f"{'H_i' if args.chain == 'hot' else '+0'}, {res['words']['inexact_on_window']} inexact on the window, "
               f"{res['words']['flag_mismatches_vs_bitsets']} flag mismatches" if blk else "no block"))
    run_result.write(path, ok, "sm120-exact-hot-check", detail,
                     [("exact_words_on_window", res["exact_words_on_window"], "count"),
                      *((("inexact_on_window", res["words"]["inexact_on_window"], "count"),
                         ("flag_mismatches", res["words"]["flag_mismatches_vs_bitsets"], "count")) if blk else ())])
    if not ok:
        sys.exit(4)


if __name__ == "__main__":
    main()
