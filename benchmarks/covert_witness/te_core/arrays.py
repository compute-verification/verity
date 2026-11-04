"""M's array graph on a recorded step, which the products term's shared-arrays factor F reads.

The theorem of record bounds a block's share of M's products by 2(1 − 1/ω)·k²/(4q) (HEShared), q the least β_r·β_s
among the array pairs M multiplies, where (i) the graph with one vertex an array and one edge a product of M (between
its two operands' arrays) has no clique of ω + 1, and (ii) distinct products of M are on distinct array pairs. A
linear layer's forward, input-gradient and weight-gradient products are a triangle on W, X and dY, so ω = 3 and
F = √(2(1 − 1/ω)) = 2/√3 (`io_bound.SHARE`), the figure's F where M holds a backward product (`share`); the forward
products alone and the weight gradients alone, one of each a layer, have no triangle, ω = 2 and F = 1, the figure's F
on them. An array is
the words of one value, an operand's root through views and casts (`io_bound._root`: the casts of one value are one
array), so micro-batches' activations are their own arrays and a weight they share is one. Two products on one pair
refuse F (`refusal`); a step with a clique of more than three charges F at its own ω, and one whose forward products
or weight gradients hold a triangle charges them F at theirs.

M is te_core.formalism.he_of_rows' on the step's rows: both operands affected, not one array times itself, no
gathered operand, not the optimizer's and not attention's (`of_step`, which a step's rows must agree with). Every
subset of M the figure takes has at most M's ω and no pair M doesn't repeat, so M's graph covers them all. The
capture's rows name no operand, so the graph needs the recorded step: the command records it as
te_core.capture.script does and writes `check`'s result, into a trace's products block with ``--into``, where
formalism's products term reads it (`arrays`).

    python -m te_core.arrays --script S --run RUN.json --step N --starts DIR [--out OUT.json] [--into TRACE.json] \\
        [--log LOG] -- ARGS...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

from . import io_bound

#: products a repeated pair names in what `check` returns
SHOWN = 4


def share(omega: int | None, floor: float = io_bound.SHARE) -> float:
    """F at an array graph's clique number ω, √(2(1 − 1/ω)), never below `floor`, the figure's F where the step's graph
    isn't measured: 2/√3 (`io_bound.SHARE`, ω = 3's) on M where it holds a backward product, 1 (ω = 2's) on the
    forward products alone and the weight gradients alone."""
    return max(floor, math.sqrt(2 * (1 - 1 / omega))) if omega else floor


def clique(adj: dict) -> list:
    """A largest clique of the graph `adj` (vertex → its neighbours): Bron–Kerbosch with pivoting, cut where the
    candidates can't beat the best found."""
    best: list = []

    def grow(r: list, p: set, x: set) -> None:
        nonlocal best
        if not p and not x:
            if len(r) > len(best):
                best = list(r)
            return
        if len(r) + len(p) <= len(best):
            return
        u = max(p | x, key=lambda v: len(adj[v] & p))
        for v in list(p - adj[u]):
            grow(r + [v], p & adj[v], x & adj[v])
            p = p - {v}
            x = x | {v}

    grow([], set(adj), set())
    return best


def check(products: list[dict]) -> dict:
    """ω and the repeated pairs of M's `products`, each {"node", "phase", "arrays": [a, b], "dw", ...}: {"omega",
    "clique", "omega_forward" and "omega_dw" (the forward products' graph's and the weight gradients'), "arrays"
    (vertices), "products", "repeated": [{"pair", "products" (the first `SHOWN` nodes), "count"}]}."""
    pairs: dict = defaultdict(list)
    adj = {x: defaultdict(set) for x in ("M", "forward", "dw")}
    for p in products:
        a, b = p["arrays"]
        pairs[tuple(sorted((a, b)))].append(p["node"])
        for x in ("M", "forward" if p["phase"] == "forward" else None, "dw" if p.get("dw") else None):
            if x:
                adj[x][a].add(b)
                adj[x][b].add(a)
    top = clique(dict(adj["M"]))
    repeated = [{"pair": list(k), "products": v[:SHOWN], "count": len(v)}
                for k, v in sorted(pairs.items(), key=lambda kv: (-len(kv[1]), kv[0])) if len(v) > 1]
    return {"omega": len(top), "clique": sorted(top), "omega_forward": len(clique(dict(adj["forward"]))),
            "omega_dw": len(clique(dict(adj["dw"]))), "arrays": len(adj["M"]), "products": len(products),
            "repeated": repeated}


def refusal(arrays: dict | None) -> str | None:
    """Why F has no figure on M with `check`'s result: two of its products on one pair of arrays (HEShared's (ii)), in
    plain words, since a panel publishes it (the pairs and their products are `check`'s `repeated`); None where it has
    one or the graph isn't measured."""
    if not arrays or not arrays["repeated"]:
        return None
    n = sum(r["count"] for r in arrays["repeated"])
    return (f"unmeasured: the step's array graph puts {n} of M's products on {len(arrays['repeated'])} pair(s) of "
            f"arrays, two or more a pair, and the shared-arrays factor F is proved only where M's products are on "
            f"distinct pairs of arrays")


def of_step(g) -> list[dict]:
    """M's products on a recorded step (te_core.capture.script's `record`), each {"node", "op", "part", "phase",
    "arrays", "dw"}, its operands' roots through views and casts and whether it is a weight gradient (a backward
    product contracting over tokens, as he_of_rows has it). Refused where M's count differs from the step's rows'
    (te_core.formalism.he_of_rows' M)."""
    from te_core.capture import products as P
    from te_core.capture.structure import _gemms, _pk

    vs = P._Values(g)
    gathered = P._gathered(vs, P._made(vs))
    by_name = {n.name: n for n in vs.nodes}
    out = []
    for m in g.matmuls:
        n = by_name[m.node]
        G = next(x for x in _gemms(n) if x.part == m.part)
        A, B = G.operands[0], G.operands[1]
        if not (vs.reads_affected(n, A) and vs.reads_affected(n, B)) or m.phase == "optimizer":
            continue
        if io_bound._root(A, scale=True) is io_bound._root(B, scale=True):
            continue
        if any(x in gathered for a in (A, B) for x in P._reads(vs, n, a)):
            continue
        if sum(x in m.token_dims for x in ("m", "k", "n")) >= 2:
            continue
        out.append({"node": n.name, "op": _pk(n.target), "part": m.part, "phase": m.phase,
                    "arrays": [io_bound._root(x).name for x in (A, B)],
                    "dw": m.phase == "backward" and sorted(m.token_dims) == ["k"]})
    rows = ((g.formalism or {}).get("products") or {}).get("rows")
    if rows is not None:
        want = sum(r["count"] for r in rows if all(r["affected"]) and not r["self"] and not any(r["gathered"])
                   and r["phase"] != "optimizer" and sum(x in r["token_dims"] for x in ("m", "k", "n")) < 2)
        if want != len(out):
            raise ValueError(f"te_core.arrays: the step's graph has {len(out)} products in M, its rows {want}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--script", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="the run's run.json (te_core.flops's record)")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--starts", type=Path, required=True, help="a directory of NAME.start.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--into", type=Path, default=None,
                    help="a trace of this step (te_core.capture.script's --out) whose products block takes the result")
    ap.add_argument("--log", type=Path, default=None, help="the program's standard output")
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:split])
    import torch

    from te_core.capture import products, script

    if torch.__version__.split("+")[0] != products.TORCH:
        raise SystemExit(f"te_core.arrays: torch {torch.__version__} here, but the capture records under torch "
                         f"{products.TORCH} (products.TORCH): run under torch=={products.TORCH}+cpu")
    run = json.loads(a.run.read_text())
    steps = run["steps"]
    t0 = time.perf_counter()
    with open(a.log or "/dev/null", "w") as log:
        g = script.record(a.script, argv[split + 1:], step=a.step, starts=a.starts,
                          wall=[float(s.get("wall_s") or 0.0) for s in steps],
                          step_tokens=int(steps[a.step]["tokens"]), log=log, tier=script.tier_of(run))
    got = check(of_step(g)) | {"step": a.step, "seconds": round(time.perf_counter() - t0, 1)}
    if a.into:
        trace = json.loads(a.into.read_text())
        if trace.get("step") != a.step:
            raise SystemExit(f"te_core.arrays: {a.into} traced step {trace.get('step')}, not {a.step}")
        trace["formalism"]["products"]["arrays"] = got
        a.into.write_text(json.dumps(trace, indent=1, default=str) + "\n")
    if a.out:
        a.out.write_text(json.dumps(got, indent=1) + "\n")
    print(json.dumps({k: got[k] for k in ("step", "omega", "omega_forward", "omega_dw", "arrays", "products",
                                          "seconds")}
                     | {"repeated": len(got["repeated"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
