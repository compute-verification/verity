"""M's array graph on a recorded step, which the products term's shared-arrays factor F reads.

The theorem of record bounds a block's share of M's products by 2(1 − 1/ω)·k²/(4q) (HEShared), q the least β_r·β_s
among the array pairs M multiplies, where (i) the graph with one vertex an array and one edge a product of M (between
its two operands' arrays) has no clique of ω + 1, and (ii) distinct products of M are on distinct array pairs. A
linear layer's forward, input-gradient and weight-gradient products are a triangle on W, X and dY, so ω = 3 and
F = √(2(1 − 1/ω)) = 2/√3 (`io_bound.SHARE`), the figure's F where M holds a backward product (`share`); the forward
products alone and the weight gradients alone, one of each a layer, have no triangle, ω = 2 and F = 1, the figure's F
on them. An array is
the words of one value, the class of the value a product reads through an operand (`values`, `array`): the casts,
copies and slices of one value are one array, and so is a recomputation, the same op on the same values, as an
activation checkpoint's rerun of its forward, so micro-batches' activations are their own arrays and a weight they
share is one. Two products on one pair, or one of a value with itself, refuse F (`refusal`), so M keeps one copy of
each product; a step with a clique of more than three charges F at its own ω, and one whose forward products or
weight gradients hold a triangle charges them F at theirs.

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
import operator
import sys
import time
from collections import defaultdict
from pathlib import Path

from . import io_bound

#: products a repeated pair names in what `check` returns
SHOWN = 4
#: ops whose result is its first argument's value, element for element and dtype for dtype
COPIES = {"clone", "contiguous", "detach", "alias", "lift_fresh", "lift_fresh_copy"}
#: ops that draw randomness, each of whose values `values` keeps its own
RANDOM = {"normal", "normal_", "uniform_", "bernoulli", "bernoulli_", "rand", "randn", "randint", "rand_like",
          "randn_like", "randint_like", "randperm", "random_", "exponential_", "geometric_", "log_normal_", "cauchy_",
          "multinomial", "poisson", "native_dropout", "dropout", "_fused_dropout", "rrelu_with_noise"}


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


def _consts(a):
    """An op's arguments with each node's place kept and its identity dropped."""
    if hasattr(a, "op") and hasattr(a, "args"):
        return "·"
    if isinstance(a, (list, tuple)):
        return tuple(_consts(x) for x in a)
    if isinstance(a, dict):
        return tuple((k, _consts(v)) for k, v in sorted(a.items()))
    return repr(a)


def values(flow) -> list[int]:
    """Each value of a step's flow (te_core.capture.structure's `_Flow`, storage-versioned) → its array: the value's
    class, or, for a view, a cast or a copy (the ops `io_bound._root` walks), the array of the latest value it read
    (its storage's as of then), so the casts and the slices of one value are one array. One op on values of the same
    classes, read argument by argument, with the same constants, makes values of one class, so a recomputation is
    its original's; a copy (`COPIES`) is its argument's class; a value an op writes in place is that op's, so a
    storage read before and after a write is two values; a graph source and every draw of randomness (`RANDOM`) are
    their own, so a rerun of a draw (a checkpoint's replay of a dropout mask) is not its original's."""
    from te_core.capture.structure import _WALK, _arg_nodes, _pk

    ids: dict = {}
    exact: list[int] = []
    out: list[int] = []
    made: dict = defaultdict(int)
    for v, n in enumerate(flow.node):
        idx, made[n] = made[n], made[n] + 1
        k, name, reads = ("own", v), None, []
        if v not in flow.source and getattr(n, "op", None) == "call_function":
            name = "getitem" if n.target is operator.getitem else _pk(n.target)
            args = _arg_nodes(n) + list(n.meta.get("te_scalar_deps", ()))
            reads = [flow.reads.get((n, a), frozenset()) for a in args]
            if name in COPIES and reads and reads[0]:
                exact.append(exact[max(reads[0])])
                out.append(out[max(reads[0])])
                continue
            if name not in RANDOM:
                k = (str(n.target), _consts(n.args), _consts(dict(n.kwargs)),
                     tuple(frozenset(exact[u] for u in r) for r in reads), idx)
        exact.append(ids.setdefault(k, len(ids)))
        out.append(out[max(reads[0])] if name in _WALK and reads and reads[0] else exact[-1])
    return out


def array(flow, of: list[int], n, a) -> int:
    """The array (`of`, `values`') of the latest value node `n` read through its operand `a`: its storage's as of the
    read, through the views and casts that made `a`."""
    r = flow.reads.get((n, a)) or ({flow.val[a]} if isinstance(flow.val.get(a), int) else None)
    if not r:
        raise ValueError(f"te_core.arrays: {n.name} reads no value through {getattr(a, 'name', a)}")
    return of[max(r)]


def names(flow, of: list[int]) -> dict:
    """Each array (`of`, `values`') → a name, one an array: the node that made its first value, and that value's place
    among the node's values past the first, since one node can write many (a fused update writes every weight)."""
    out: dict = {}
    made: dict = defaultdict(int)
    for v, c in enumerate(of):
        name = getattr(flow.node[v], "name", str(v))
        out.setdefault(c, f"{name}[{made[name]}]" if made[name] else name)
        made[name] += 1
    return out


def check(products: list[dict]) -> dict:
    """ω and the repeated pairs of M's `products`, each {"node", "phase", "arrays": [a, b], "dw", ...}: {"omega",
    "clique", "omega_forward" and "omega_dw" (the forward products' graph's and the weight gradients'), "arrays"
    (vertices), "products", "repeated": [{"pair", "products" (the first `SHOWN` nodes), "count"}], "loops" (the
    products of one array with itself)}."""
    pairs: dict = defaultdict(list)
    adj = {x: defaultdict(set) for x in ("M", "forward", "dw")}
    loops = []
    for p in products:
        a, b = p["arrays"]
        pairs[tuple(sorted((a, b)))].append(p["node"])
        if a == b:
            loops.append(p["node"])
            continue
        for x in ("M", "forward" if p["phase"] == "forward" else None, "dw" if p.get("dw") else None):
            if x:
                adj[x][a].add(b)
                adj[x][b].add(a)
    top = clique(dict(adj["M"]))
    repeated = [{"pair": list(k), "products": v[:SHOWN], "count": len(v)}
                for k, v in sorted(pairs.items(), key=lambda kv: (-len(kv[1]), kv[0])) if len(v) > 1]
    return {"omega": len(top), "clique": sorted(top), "omega_forward": len(clique(dict(adj["forward"]))),
            "omega_dw": len(clique(dict(adj["dw"]))), "arrays": len(adj["M"]), "products": len(products),
            "repeated": repeated, "loops": loops}


def refusal(arrays: dict | None) -> str | None:
    """Why F has no figure on M with `check`'s result, in plain words, since a panel publishes it: two of its products
    on one pair of arrays (HEShared's (ii); the pairs and their products are `check`'s `repeated`), or a product of
    one array with itself, its value's copy or recomputation (`loops`), which M must leave out as it leaves out one
    array times itself; None where it has one or the graph isn't measured."""
    if not arrays:
        return None
    why = []
    if arrays["repeated"]:
        n = sum(r["count"] for r in arrays["repeated"])
        why.append(f"puts {n} of M's products on {len(arrays['repeated'])} pair(s) of arrays, two or more a pair")
    if arrays.get("loops"):
        why.append(f"has {len(arrays['loops'])} of M's products of one value with a copy or a recomputation of itself")
    if not why:
        return None
    return (f"unmeasured: the step's array graph {' and '.join(why)}, and the shared-arrays factor F is proved only "
            f"where M's products are on distinct pairs of distinct arrays")


def of_step(g) -> list[dict]:
    """M's products on a recorded step (te_core.capture.script's `record`), each {"node", "op", "part", "phase",
    "arrays", "dw"}: its operands' arrays, each the class of the value it read (`array`, by its `names`), and whether
    it is a weight gradient (a backward product contracting over tokens, as
    he_of_rows has it). Refused where M's count differs from the step's rows' (te_core.formalism.he_of_rows' M)."""
    from te_core.capture import products as P
    from te_core.capture.structure import _gemms, _pk

    vs = P._Values(g)
    gathered = P._gathered(vs, P._made(vs))
    by_name = {n.name: n for n in vs.nodes}
    of = values(vs.flow)
    first = names(vs.flow, of)
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
                    "arrays": [first[array(vs.flow, of, n, x)] for x in (A, B)],
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
