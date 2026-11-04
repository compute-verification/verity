"""machine-model capacity GROUP [GROUP ...] [--vector V] [--json]
                                          the certified share σ of a window's capacity (capacity.py): each GROUP
                                          is [WEIGHT/]TERM+TERM..., its share of the cell's SMSP-time (default 1)
                                          and its certified load, TERM one of FOLD:SHARE[*R] (a noise-GEMM filler
                                          at SHARE of the tensor peak, its words folded R times; FOLD duo, xin13,
                                          li, arx (a cost probe) or none), lane:RATE (verified lane chains, warp instructions a
                                          SMSP-cycle), served:SHARE[:FOLD] or CLASS=RATE; σ is read on the pooled
                                          load, with each group's placement-attested σ beside it"""
from __future__ import annotations

import json

from verity_machine_model import capacity as C
from verity_machine_model import eps as E


def add(sub) -> None:
    k = sub.add_parser("capacity")
    k.add_argument("groups", nargs="+", metavar="GROUP")
    k.add_argument("--vector", help="the vector whose rows are Π (default: model.toml's current)")
    k.add_argument("--json", action="store_true")


def run(args) -> int:
    pi = E.machine(args.vector or "", args.data)
    groups = []
    for g in args.groups:
        w, _, spec = g.rpartition("/")
        groups.append((float(w) if w else 1.0, C.parse(spec)))
    x = C.cell(pi, groups, C.useful_of(args.data))
    if args.json:
        print(json.dumps(x.record(), ensure_ascii=False, indent=1))
        return 0
    p = x.pooled
    print(f"Π: {pi.vector}; the empty GPU's useful capacity {x.empty:.5f} tensor slots a SMSP-cycle "
          f"({x.empty * C.TENSOR_CYCLES:.3f}× the tensor peak)")
    for (w, g), h in zip(x.groups, x.parts):
        print(f"  {w:.4g} of the cell: {g.name}: σ {1 - h.hidden / x.empty:.4f} placed, B {h.credit():.4f}, tensor "
              f"{g.tensor_share():.3f}" + (f", OVER on {', '.join(h.over)}" if h.over else ""))
    print(f"  pooled load: " + ", ".join(f"{k} {v:.4g}" for k, v in sorted(p.work.n.items())) + " warp instructions a "
          f"SMSP-cycle; rows " + ", ".join(f"{r} {v:.3f}" for r, v in sorted(p.load.items(), key=lambda kv: -kv[1])[:5]))
    if p.over:
        print(f"  OVER on {', '.join(p.over)}: the certified load alone exceeds Π (a miscount or a model violation)")
    print(f"  hidden: {p.hidden:.5f} slots a SMSP-cycle, " + (", ".join(f"{k} {v:.4g}" for k, v in p.mix.items())
                                                             or "nothing") +
          f"; binding {', '.join(p.binding) or 'none'}")
    print(f"  σ {x.sigma():.4f}  (ε_cap {x.eps_cap():.4f}; placement-attested {x.sigma_placed():.4f}; time credit B "
          f"{x.credit():.4f}; tensor share {x.tensor():.4f})")
    return 0
