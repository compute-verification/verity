"""HT-GENEROUS: rotated.py's generous bound in every order against the recommended point's credits
(`tables.RECOMMENDED`), on every credited q and k of Llama-3.1-8B and -70B at design U's 2% codes.

    python benchmarks/hardness_defence/generous.py [--runs ~/.research/runs] [--matrices]

The generous bound prices the split's low-rank part P as if real and free, r/n + r/K (`rotated.lr1`), with P the real
rank-r SVD (`svd/r`) or rpca fit (`rpca/r/a`) and its residual rint(q − P) on its cheapest route; it is no route, since
E·rint(q − P)ᵀ misses E·(q − P − rint(q − P))ᵀ. Per matrix and chip@price its least over the leaf is the least of the
generous scores of every record of tables.SOURCES (natural order or all orders) and the matrix's exact least
(`tables.load`). A matrix credited c is under the line when that least over c is under 1.05; min(1, least/1.05)
restores it. `isvd@lr1` is the exact split's own residual with its low-rank part at lr1 (no extra digits or writes),
over the ranks whose exact split is priced in all orders: the least any exact split could reach by pricing its factors
alone down to lr1.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hardness_public"))
sys.path.insert(0, str(HERE))
import rotated  # noqa: E402
import tables as T  # noqa: E402

QK = ("q_proj", "k_proj")
WITHIN = 0.10


def records(runs: Path):
    """(model, orders, record) for every record of tables.SOURCES at design U's 2% codes with the split searched."""
    seen = set()
    for run, pattern, model, tags in T.SOURCES:
        for d in sorted((runs / run / "out").glob(pattern)):
            for f in sorted(d.glob("matrices.jsonl")) + sorted(d.glob("t*/matrices.jsonl")):
                if f in seen:
                    continue
                seen.add(f)
                for line in f.read_text().splitlines():
                    rec = json.loads(line)
                    tg = rec.get("target") if rec.get("target") is not None else tags.get("target")
                    if (rec.get("chunked") or round(float(tg), 4) != 0.02 or (rec.get("rot") or "hadamard") != "hadamard"
                            or tags.get("deflate") or "noise" in rec):
                        continue
                    yield model, bool(tags.get("orders")), run, rec


class Gen:
    """Per (model, tensor): the generous score per chip@price, natural order only and in all; per rank the svd split's
    least over the leaf over chips and prices, and the isvd@lr1 bound per chip@price."""

    def __init__(self, runs: Path):
        self.nat = collections.defaultdict(dict)
        self.all = collections.defaultdict(dict)
        self.curve = {o: collections.defaultdict(dict) for o in (False, True)}
        self.rpca = {o: collections.defaultdict(dict) for o in (False, True)}
        self.isvd_lr1 = collections.defaultdict(dict)
        self.runs = collections.defaultdict(set)
        for model, orders, run, rec in records(runs):
            key = (model, rec["tensor"])
            n, K = rec["shape"]
            leaf = rec["leaf"]
            if rec.get("svd") or rec.get("rpca"):
                for k in T.KEYS:
                    s = rotated.score(rec, k)
                    for book in (self.all,) + ((self.nat,) if not orders else ()):
                        cur = book[key]
                        if k not in cur or s["generous_over_leaf"] < cur[k][0]:
                            cur[k] = (s["generous_over_leaf"], s["generous_route"], run)
                self.runs[key].add(run)
            for part, store in (("svd", self.curve[orders]), ("rpca", self.rpca[orders])):
                for r, v in rec.get(part, {}).items():
                    c = min((v["lr1"] + v["routes"][k]["cost"]) / leaf for k in T.KEYS)
                    store[key][r] = min(store[key].get(r, math.inf), c)
            if orders:
                for r, v in rec.get("isvd", {}).items():
                    for k in T.KEYS:
                        c = (rotated.lr1(int(r), n, K) + v["routes"][k]["cost"]) / leaf
                        if c < self.isvd_lr1[key].get(k, (math.inf,))[0]:
                            self.isvd_lr1[key][k] = (c, f"isvd/{r}")


def vertex(x, y):
    """The minimum of the parabola through three points (rotated knee fits, as the split-defence report's)."""
    (x0, x1, x2), (y0, y1, y2) = x, y
    den = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
    c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / den
    if a <= 0:
        return min(y)
    xv = -b / (2 * a)
    return a * xv * xv + b * xv + c


def analyse(book, gen):
    """Per credited q and k at the recommended point: its credit, exact least, generous least (natural order only and
    in all), the line-restoring credit, the svd knee in all orders and its parabola minimum, isvd@lr1, the reorder
    gain on the generous bound (natural less all orders at the ranks priced both ways), and the natural-order-only
    ranks a reorder could still take under the all-orders least."""
    choose = T.DEFENCES[T.RECOMMENDED]
    rows = []
    for model in T.MODELS:
        for t in T.all_tensors(model):
            if T.kind(t) not in QK:
                continue
            var, credit = choose(book, model, t)
            if credit == 0:
                continue
            key = (model, t)
            ex = book.v[model][t][var]
            g_all = {k: min(ex[k][0], gen.all[key][k][0]) if k in gen.all[key] else ex[k][0] for k in T.KEYS}
            g_route = {k: gen.all[key][k][1] if k in gen.all[key] and gen.all[key][k][0] < ex[k][0] else ex[k][1]
                       for k in T.KEYS}
            g_nat = {k: gen.nat[key][k][0] for k in gen.nat[key]} if gen.nat[key] else None
            least = min(g_all.values())
            k0 = min(T.KEYS, key=g_all.get)
            ca, cn = gen.curve[True][key], gen.curve[False][key]
            ranks = sorted(int(r) for r in ca)
            knee, top, fit, nb = None, False, None, None
            if ranks:
                knee = min(ranks, key=lambda r: ca[str(r)])
                i = ranks.index(knee)
                top = knee == ranks[-1]
                if 0 < i < len(ranks) - 1:
                    x = (ranks[i - 1], knee, ranks[i + 1])
                    fit = vertex(x, [ca[str(r)] for r in x])
                    nb = " ".join(f"{r}:{ca[str(r)]:.4f}" for r in x)
            both = [r for r in ca if r in cn]
            gain = max((cn[r] - ca[r] for r in both), default=None)
            natonly = {r: v for r, v in cn.items() if r not in ca}
            rows.append(dict(
                model=model, t=t, credit=credit, exact=T.least_of(ex), nat=min(g_nat.values()) if g_nat else None,
                least=least, key=k0, route=g_route[k0], per_key=g_all, new_credit=min(1.0, least / T.LINE),
                knee=knee, knee_top=top, knee_nb=nb, fit=fit, ranks=ranks, gain=gain, natonly=natonly,
                rpca=min(gen.rpca[True][key].values(), default=None),
                isvd_lr1=min((v[0] for v in gen.isvd_lr1[key].values()), default=None),
                isvd_lr1_route=min(gen.isvd_lr1[key].values(), default=(None, None))[1],
                runs=sorted(gen.runs[key])))
    return rows


def revised(rows):
    """The recommended point with each credited q and k at min(its credit, its generous least / 1.05)."""
    base = T.DEFENCES[T.RECOMMENDED]
    new = {(r["model"], r["t"]): r["new_credit"] for r in rows}

    def choose(book, model, t):
        var, credit = base(book, model, t)
        if (model, t) in new:
            credit = min(credit, new[(model, t)])
        return var, credit
    return choose


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=Path.home() / ".research" / "runs")
    ap.add_argument("--matrices", action="store_true", help="also every credited q and k under the line or near it")
    args = ap.parse_args()
    book = T.load(args.runs)
    gen = Gen(args.runs)
    rows = analyse(book, gen)
    choose = revised(rows)
    print("| model | kind | credited | in scope (natural generous within 0.10, or none) | all-orders generous priced | "
          "under the line, natural order | under the line, all orders | least over credit (natural / all) | "
          "set by | knee at the top rank | natural-only rank a reorder could undercut |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for model in T.MODELS:
        for kd in QK:
            R = [r for r in rows if r["model"] == model and T.kind(r["t"]) == kd]
            scope = [r for r in R if r["nat"] is None or r["nat"] / r["credit"] < T.LINE + WITHIN]
            priced = [r for r in R if r["ranks"]]
            un_nat = [r for r in R if r["nat"] is not None and r["nat"] / r["credit"] < T.LINE]
            un_all = [r for r in R if r["least"] / r["credit"] < T.LINE]
            w = min(R, key=lambda r: r["least"] / r["credit"])
            wn = min((r for r in R if r["nat"] is not None), key=lambda r: r["nat"] / r["credit"], default=None)
            tops = [T.short(r["t"]) for r in R if r["knee_top"]]
            gains = [r["gain"] for r in R if r["gain"] is not None]
            gmax = max(gains, default=0.0)
            under_nat = [T.short(r["t"]) for r in R for rr, v in r["natonly"].items()
                         if (v - gmax) / r["new_credit"] < min(T.LINE, r["least"] / r["new_credit"])]
            print(f"| {model} | {kd.removesuffix('_proj')} | {len(R)} | {len(scope)} | {len(priced)} | {len(un_nat)} | "
                  f"{len(un_all)} | {wn['nat'] / wn['credit'] if wn else math.nan:.4f} / {w['least'] / w['credit']:.4f} | "
                  f"{T.short(w['t'])} ({w['route']}, {w['key']}) | {len(tops)} {tops[:6]} | {len(under_nat)} |")
    print()
    print("Least generous over credit among credited q and k, per chip@price, at the recommended credits and at the "
          "revised ones (each at min(its credit, its generous least / 1.05)):")
    print()
    print("| model | credits | sm_120 0.84 / 1.48 / 1.78 | H200 0.84 / 1.48 / 1.78 | under 1.05 | MACs credited |")
    print("|---|---|---|---|---|---|")
    for model in T.MODELS:
        R = [r for r in rows if r["model"] == model]
        for name, cred in (("recommended", "credit"), ("revised", "new_credit")):
            cells = [" / ".join(f"{min(r['per_key'][f'{c}@{p}'] / r[cred] for r in R):.3f}" for p in T.PRICES)
                     for c in ("sm120", "h200")]
            n_under = sum(r["least"] / r[cred] < T.LINE - 1e-12 for r in R)
            ch = T.DEFENCES[T.RECOMMENDED] if cred == "credit" else choose
            _w, share, _u, _n, _s = T.evaluate(book, model, ch)
            print(f"| {model} | {name} | {cells[0]} | {cells[1]} | {n_under} | {100 * share:.3f}% |")
    print()
    print("The revised point against the exact routes, every matrix (tables.evaluate):")
    print()
    print("| model | sm_120 0.84 / 1.48 / 1.78 | H200 0.84 / 1.48 / 1.78 | least set by | MACs credited |")
    print("|---|---|---|---|---|")
    for model in T.MODELS:
        worst, share, _u, _n, _s = T.evaluate(book, model, choose)
        cells = [" / ".join(f"{worst[f'{c}@{p}'][0]:.3f}" for p in T.PRICES) for c in ("sm120", "h200")]
        k0 = min(T.KEYS, key=lambda k: worst[k][0])
        print(f"| {model} | {cells[0]} | {cells[1]} | {T.short(worst[k0][1])} ({worst[k0][2]}, {k0}) | "
              f"{100 * share:.3f}% |")
    gains = [(r["gain"], r["model"], T.short(r["t"])) for r in rows if r["gain"] is not None]
    fits = [(r["least"] - r["fit"], r["model"], T.short(r["t"])) for r in rows if r["fit"] is not None]
    print()
    print("reorder gain on the svd split, natural less all orders at a rank priced both ways: max",
          max(gains, default=None), "| parabola gain at the svd knee: max", max(fits, default=None))
    if args.matrices:
        print()
        print("| model | matrix | credit | exact least | generous, natural | generous, all orders | its route | "
              "over credit | revised credit | svd knee (all orders) | parabola min | isvd@lr1 | rpca least |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: (r["model"], r["least"] / r["credit"])):
            if r["least"] / r["credit"] >= T.LINE + WITHIN:
                continue
            nat = f"{r['nat']:.4f}" if r["nat"] is not None else "–"
            fit = f"{r['fit']:.4f}" if r["fit"] is not None else "–"
            il = f"{r['isvd_lr1']:.4f} ({r['isvd_lr1_route']})" if r["isvd_lr1"] is not None else "–"
            rp = f"{r['rpca']:.4f}" if r["rpca"] is not None else "–"
            print(f"| {r['model']} | {T.short(r['t'])} | {r['credit']:.4f} | {r['exact']:.4f} | {nat} | "
                  f"{r['least']:.4f} | {r['route']}, {r['key']} | {r['least'] / r['credit']:.4f} | "
                  f"{r['new_credit']:.4f} | {r['knee_nb'] or ('top ' if r['knee_top'] else '') + str(r['knee'])} | "
                  f"{fit} | {il} | {rp} |")


if __name__ == "__main__":
    main()
