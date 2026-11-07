"""HT-LEVERS: 70B's loss against its credited share of MACs at the recommended point and at the two levers the loss
attribution points at (note:compute-research/20261107T0245Z-report-hardness-70b-loss), with their run ids.

    python benchmarks/hardness_defence/levers70b.py --res DIR [--runs ~/.research/runs] [--matrices] [--json OUT]

- A: layers 0-3 served bf16 and uncredited, the rest at the recommended point (map 70b-defl-l0-3-bf16, served e2).
- B: E on a = 3 (e3) or a = 2 (e1) values on layers 40-79, the rest at the recommended point (e2:F), each of those
  280 matrices priced at its a (LEVERS: the exact split and the generous bound in all orders) and credited
  min(1, least / 1.05), the defence's rule (`tables.margin`).
- C: A and B together.

Quality: each arm's per-window NLL (qat.py's `window_nll`, `loss70b.load`) from DIR/NAME/out/NAME/result.json for each
NAME of QUALITY (`research data fetch RUN's run_record --path 'out/*' --to DIR/NAME`), as ratios to bf16 paired window
by window over WikiText-2's 141 windows and FineWeb-Edu's first 32 (and its 512 where both sides have them).
Credit: `tables.load`'s book (the defence's records at a = 4, tables.SOURCES), with layers 40-79's design-U records
replaced by LEVERS' at a; the exact rule reads each matrix's least exact route, the generous rule each q and k at
min(its credit, its generous least / 1.05) as HT-GENEROUS's revised point (`generous.revised`), and at a = 3 or 2 every
matrix of layers 40-79 at min(1, min(exact, generous least) / 1.05).
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hardness_public"))
sys.path.insert(0, str(HERE))
import generous as G  # noqa: E402
import loss70b as Lo  # noqa: E402
import rotated  # noqa: E402
import tables as T  # noqa: E402

BAND = range(40, 80)
EXACT_BF16 = range(0, 4)
#: the pricing runs of layers 40-79 at a = 3 and 2 (jobs/l40-*.txt): (run, out/ pattern)
LEVERS = (("r20261107-030334-7035", "70b-l40-*-a2"), ("r20261107-030354-1685", "70b-l40-*-a3"),
          ("r20261107-030634-a6cb", "70b-l40-*"))
#: qat.py arms (cpu70b.sh NAME) by name: their run ids
QUALITY = {
    "ladder": "r20261106-010734-0eac", "rec-wt": "r20261106-013236-9960", "bf16-sub": "r20261106-010859-7b36",
    "bf16-wt": "r20261106-011959-dacf", "bf16-fw0": "r20261106-012016-0cbd",
    "lev-a1": "r20261107-030754-238f", "lev-a2": "r20261107-031012-f2d5",
    "lev-b1": "r20261107-030812-ee88", "lev-b2": "r20261107-031017-79c9",
    "lev-r1": "r20261107-031200-eb85", "lev-r2": "r20261107-031203-e2b1", "lev-bf16-fw2": "r20261107-031023-6aa6",
}
AMAP = "70b-defl-l0-3-bf16"
#: arm -> (map, spec as loss70b shortens it, the qat.py arms measuring it, E's a on layers 40-79, layers 0-3 bf16)
ARMS = {
    "recommended (e2:F)": ("70b-defl", "e2:F", ("ladder", "rec-wt", "lev-r1", "lev-r2"), 4, False),
    "A: layers 0-3 bf16": (AMAP, "e2", ("lev-a1", "lev-a2"), 4, True),
    "B: a = 3 on layers 40-79": ("70b-defl", "e2:F;e3=layers.40-79", ("lev-b1", "lev-b2"), 3, False),
    "B: a = 2 on layers 40-79": ("70b-defl", "e2:F;e1=layers.40-79", ("lev-b1", "lev-b2"), 2, False),
    "C: A and B at a = 3": (AMAP, "e2;e3=layers.40-79", ("lev-a1", "lev-a2"), 3, True),
    "C: A and B at a = 2": (AMAP, "e2;e1=layers.40-79", ("lev-a1", "lev-a2"), 2, True),
}
BF16 = ("bf16-sub", "bf16-wt", "bf16-fw0", "lev-bf16-fw2")
WT_WINDOWS, FW_FIRST, FW_ALL = 141, 32, 512


def lever_scores(runs: Path) -> dict:
    """a -> tensor -> chip@price -> (exact least over the leaf, its route, generous least, its route), the least over
    every LEVERS record of the tensor at a."""
    out = collections.defaultdict(dict)
    for run, pattern in LEVERS:
        for f in sorted((runs / run / "out").glob(f"{pattern}/matrices.jsonl")):
            for line in f.read_text().splitlines():
                rec = json.loads(line)
                cur = out[rec["a_noise"]].setdefault(rec["tensor"], {k: [math.inf, None, math.inf, None] for k in T.KEYS})
                for k in T.KEYS:
                    s, c = rotated.score(rec, k), cur[k]
                    if s["over_leaf"] < c[0]:
                        c[0], c[1] = s["over_leaf"], s["route"]
                    if s["generous_over_leaf"] < c[2]:
                        c[2], c[3] = s["generous_over_leaf"], s["generous_route"]
    return out


def band_tensors():
    return [t for t in T.all_tensors("70b") if T.layer(t) in BAND]


def at_a(book, scores: dict, generous: bool):
    """book with layers 40-79's design-U scores at a (exact least, or with generous the least of exact and generous)."""
    out = copy.deepcopy(book)
    for t in band_tensors():
        s = scores[t]
        out.v["70b"][t][T.U] = {k: (s[k][2], s[k][3]) if generous and s[k][2] < s[k][0] else (s[k][0], s[k][1])
                                for k in T.KEYS}
    return out


def without_head(choose):
    def ch(book, model, t):
        return (None, 0.0) if model == "70b" and T.layer(t) in EXACT_BF16 else choose(book, model, t)
    return ch


def band_margin(choose):
    """choose, with each matrix of layers 40-79 at the defence's margin on its (substituted) design-U scores."""
    def ch(book, model, t):
        return T.margin(book, model, t) if model == "70b" and T.layer(t) in BAND else choose(book, model, t)
    return ch


def credit(book, choose, scores_a=None):
    """The credited share of 70B's MACs and, over layers 40-79, the matrices credited below 1 by kind and the share of
    MACs they lose."""
    _worst, share, unpriced, _n, _s = T.evaluate(book, "70b", choose)
    assert not unpriced, unpriced[:4]
    under, lost = collections.Counter(), 0.0
    for t in band_tensors():
        _var, c = choose(book, "70b", t)
        if c < 1:
            under[T.kind(t).removesuffix("_proj")] += 1
            lost += T.macs("70b", T.kind(t)) * (1 - c)
    return {"share": share, "under": dict(under), "lost": lost / T.total_macs("70b")}


def quality(res: Path) -> tuple[dict, list[str]]:
    paths = [res / n / "out" / n / "result.json" for n in QUALITY]
    missing = [str(p) for p in paths if not p.exists()]
    data, notes = Lo.load([p for p in paths if p.exists()])
    return data, notes + [f"missing: {m}" for m in missing]


def window_ratio(a: dict, b: dict, n: int) -> dict | None:
    """loss70b.ratio over the set's first n windows, None unless both sides have every one."""
    if not all(i in a and i in b for i in range(n)):
        return None
    return Lo.ratio({i: a[i] for i in range(n)}, {i: b[i] for i in range(n)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=Path, required=True, help="DIR/NAME/out/NAME/result.json for each QUALITY arm")
    ap.add_argument("--runs", type=Path, default=Path.home() / ".research" / "runs")
    ap.add_argument("--matrices", action="store_true", help="also layers 40-79's matrices under the line at a = 3 or 2")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    book = T.load(args.runs)
    rec = T.DEFENCES[T.RECOMMENDED]
    gen_rec = G.revised(G.analyse(book, G.Gen(args.runs), rec), rec)
    lev = lever_scores(args.runs)
    for a in (3, 2):
        miss = [t for t in band_tensors() if t not in lev.get(a, {})]
        if miss:
            print(f"note: a = {a}: {len(miss)} of layers 40-79's 280 matrices not priced yet ({T.short(miss[0])}, ...)")
    data, notes = quality(args.res)
    bf16 = {s: data.get(("bf16", "bf16", s), {}) for s in ("wikitext2", "fineweb_val")}

    rows = []
    for arm, (smap, spec, names, a, head) in ARMS.items():
        cr = {}
        for rule in ("exact", "generous"):
            base = rec if rule == "exact" else gen_rec
            if a == 4:
                b, ch = book, base
            elif all(t in lev.get(a, {}) for t in band_tensors()):
                b, ch = at_a(book, lev[a], rule == "generous"), band_margin(base)
            else:
                cr[rule] = None
                continue
            cr[rule] = credit(b, without_head(ch) if head else ch)
        q = {s: data.get((smap, spec, s), {}) for s in ("wikitext2", "fineweb_val")}
        rows.append({"arm": arm, "map": smap, "spec": spec, "a": a,
                     "wt": window_ratio(q["wikitext2"], bf16["wikitext2"], WT_WINDOWS),
                     "fw32": window_ratio(q["fineweb_val"], bf16["fineweb_val"], FW_FIRST),
                     "fw512": window_ratio(q["fineweb_val"], bf16["fineweb_val"], FW_ALL),
                     "credit": cr, "runs": [QUALITY[n] for n in names]})

    def pct(r):
        return f"{r['pct']:+.2f}% ± {r['pct_se']:.2f}" if r else "–"

    def share(c):
        return f"{100 * c['share']:.3f}% (γ ≥ ×{1 / c['share']:.4f})" if c else "–"
    print(f"| arm | WikiText-2, {WT_WINDOWS} windows | FineWeb-Edu, first {FW_FIRST} | FineWeb-Edu, {FW_ALL} | "
          "MACs credited, exact rule | layers 40-79 under the line (exact) | MACs credited, generous rule | "
          "under the line (generous) | runs |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ce, cg = r["credit"]["exact"], r["credit"]["generous"]
        und = [f"{c['under']} ({100 * c['lost']:.3f}% of MACs lost)" if c and c["under"] else "none" if c else "–"
               for c in (ce, cg)]
        print(f"| {r['arm']} | {pct(r['wt'])} | {pct(r['fw32'])} | {pct(r['fw512'])} | {share(ce)} | {und[0]} | "
              f"{share(cg)} | {und[1]} | {', '.join(r['runs'])} |")
    print()
    print(f"bf16: {', '.join(QUALITY[n] for n in BF16)}; pricing at a = 4: tables.SOURCES; at a = 3 and 2: "
          f"{', '.join(sorted({r for r, _p in LEVERS}))}")
    rec_fw = data.get(("70b-defl", "e2:F", "fineweb_val"), {})
    if all(i in rec_fw and i in bf16["fineweb_val"] for i in range(FW_ALL)):
        cpu = {k: sum(v[i] for i in range(FW_ALL)) / FW_ALL for k, v in (("bf16", bf16["fineweb_val"]), ("e2:F", rec_fw))}
        gpu = {k: Lo.GPU[k]["fineweb_val"] for k in ("bf16", "e2:F")}
        print(f"FineWeb-Edu over {FW_ALL} windows, mean NLL: CPU bf16 {cpu['bf16']:.6f}, e2:F {cpu['e2:F']:.6f} "
              f"({100 * math.expm1(cpu['e2:F'] - cpu['bf16']):+.3f}%); GPU {gpu['bf16']:.6f}, {gpu['e2:F']:.6f} "
              f"({100 * math.expm1(gpu['e2:F'] - gpu['bf16']):+.3f}%)")
    if args.matrices:
        print()
        print("| matrix | a = 4 exact least | a = 3 exact / generous | a = 2 exact / generous | credit a = 4 / 3 / 2 "
              "(exact) |")
        print("|---|---|---|---|---|")
        for t in band_tensors():
            l4 = T.least_of(book.v["70b"][t][T.U])
            ls = {a: (min(v[0] for v in lev[a][t].values()), min(v[2] for v in lev[a][t].values()))
                  for a in (3, 2) if t in lev.get(a, {})}
            if l4 >= T.LINE and all(e >= T.LINE and g >= T.LINE for e, g in ls.values()):
                continue
            cells = [f"{ls[a][0]:.4f} / {ls[a][1]:.4f}" if a in ls else "–" for a in (3, 2)]
            crs = " / ".join(f"{min(1, x / T.LINE):.4f}" for x in [l4] + [ls[a][0] for a in (3, 2) if a in ls])
            print(f"| {T.short(t)} | {l4:.4f} | {cells[0]} | {cells[1]} | {crs} |")
    for n in notes:
        print("note:", n)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
