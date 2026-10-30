"""The schedule behind a `schedules.Model` cost: the argmin tree, printed and returned as nested tuples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from schedules import INSTANCES, STRASSEN, WINOGRAD, Model, holds, rng


def witness(m: Model, L: int, a: int, b: int):
    target = m.lo(L, a, b)
    if m.atom(a, b) == target:
        f = next(f for f in m.atoms if f.w8 == target and holds(f, a, b))
        return {"atom": f.name, "a": a, "b": b, "cost": target}
    if L > 0:
        for name, label, naive, pairs in m.opts:
            tot = sum(m.lo(L - 1, rng(a, ua), rng(b, vb)) for ua, vb in pairs)
            if abs(tot / naive - target) < 1e-12:
                kids = [witness(m, L - 1, rng(a, ua), rng(b, vb)) for ua, vb in pairs]
                return {"scheme": name, "roles": label, "a": a, "b": b, "cost": target, "kids": kids}
    for side in ("a", "b"):
        v = a if side == "a" else b
        for d in m.cands(v):
            hi = -(-v // d)
            if side == "a":
                c = m.lo(L, d, b) + m.lo(L, hi, b)
                if abs(c - target) < 1e-12:
                    return {"split": "a", "d": d, "a": a, "b": b, "cost": target,
                            "kids": [witness(m, L, d, b), witness(m, L, hi, b)]}
            else:
                c = m.lo(L, a, d) + m.lo(L, a, hi)
                if abs(c - target) < 1e-12:
                    return {"split": "b", "d": d, "a": a, "b": b, "cost": target,
                            "kids": [witness(m, L, a, d), witness(m, L, a, hi)]}
    raise AssertionError((L, a, b, target))


def show(node, ind=0):
    pad = "  " * ind
    if "atom" in node:
        print(f"{pad}{node['atom']}({node['a']},{node['b']}) = {node['cost']}")
    elif "scheme" in node:
        print(f"{pad}{node['scheme']}[{node['roles']}]({node['a']},{node['b']}) = {node['cost']}")
        for k in node["kids"]:
            show(k, ind + 1)
    else:
        print(f"{pad}split {node['split']} d={node['d']} ({node['a']},{node['b']}) = {node['cost']}")
        for k in node["kids"]:
            show(k, ind + 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instance", default="h200Measured")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("-a", type=int, default=2)
    ap.add_argument("-b", type=int, default=256)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    m = Model(INSTANCES[args.instance], [STRASSEN, WINOGRAD])
    w = witness(m, args.depth, args.a, args.b)
    show(w)
    if args.json:
        args.json.write_text(json.dumps(w, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
