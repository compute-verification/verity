"""machine-model counts NAME M N K [--vector V] [--json]
                                          an honest kernel's closed-form instruction counts at M x N x K
                                          (data/counts): each class's, a MMA, and the model bound with its rows
machine-model counts NAME --check         every ncu observation against the closed form, and its bound"""
from __future__ import annotations

import json

from verity_machine_model import counts as cf
from verity_machine_model import eps as E


def add(sub) -> None:
    n = sub.add_parser("counts")
    n.add_argument("name")
    n.add_argument("shape", nargs="*", type=int)
    n.add_argument("--check", action="store_true")
    n.add_argument("--vector")
    n.add_argument("--json", action="store_true")


def run(args) -> int:
    if args.shape and len(args.shape) != 3:
        raise SystemExit("counts NAME M N K")
    c = cf.load(args.name, args.data)
    pi = E.machine(args.vector or "", args.data)
    if args.check:
        bad = c.check() + c.check_bound(pi)
        for o in c.obs:
            want, rows = c.bound(pi, *o.shape)
            n = c.n(*o.shape)
            worst = max(abs(o.counts.get(k, 0) - v) / max(v, 1) for k, v in n.items())
            print(f"{o.run} {'x'.join(map(str, o.shape)):>17}: bound {want:.4f} on {', '.join(rows)}, "
                  f"worst class {worst:.1e}")
        print("\n".join(bad) if bad else f"pass: {len(c.obs)} observations within tolerance")
        return 1 if bad else 0
    if not args.shape:
        raise SystemExit("counts NAME M N K, or counts NAME --check")
    m, n, k = args.shape
    t, rows = c.bound(pi, m, n, k)
    rec = {"name": c.name, "kernel": c.kernel + ("_k3" if c.variant(k) == "k3" else ""), "shape": [m, n, k],
           "n": c.n(m, n, k), "per_mma": c.per_mma(m, n, k), "bound": t, "binding": list(rows), "vector": pi.vector}
    if args.json:
        print(json.dumps(rec, indent=1))
        return 0
    print(f"{rec['kernel']} at {m} x {n} x {k}: bound {t:.4f} SMSP-cycles a MMA on {', '.join(rows)} ({pi.vector})")
    for k_, v in sorted(rec["n"].items(), key=lambda x: -x[1]):
        print(f"  {k_:8s} {v:16.0f}  {rec['per_mma'][k_]:9.4f} a MMA")
    return 0
