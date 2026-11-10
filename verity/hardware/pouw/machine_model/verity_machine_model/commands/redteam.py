"""machine-model redteam [DESIGN ...] [--device sm_NN] [--q Q] [--operands O] [--credit C] [--vector V]
                                          the red-team audit (redteam.py): each design's binding and near-binding
                                          facets, the move in the cheater's favour at which the core
                                          first rises (per row), and, where Π prices no uniform datapath, the ε a
                                          sibling-rate uniform route would open (default q* = 1.22, credit floor)"""
from __future__ import annotations

import argparse

from verity_machine_model import eps as E
from verity_machine_model import redteam as RT
from verity_machine_model.commands._shared import device_help, pct


def add(sub) -> None:
    rt = sub.add_parser("redteam")
    rt.add_argument("designs", nargs="*", help="the designs to audit (default: every design in the data directory)")
    rt.add_argument("--device", default=argparse.SUPPRESS, help=device_help())
    rt.add_argument("--q", type=float, default=1.22, help="q*, lane instructions a fresh word (default %(default)s)")
    rt.add_argument("--operands", choices=E.OPERANDS, default="free",
                    help="the cheater's operand delivery on the tensor route (default %(default)s)")
    rt.add_argument("--credit", choices=E.CREDITS, default="floor",
                    help="what ε is a fraction of (default %(default)s: the floor program ♭)")
    rt.add_argument("--vector", help="the vector whose rows are Π (default: model.toml's current)")


def run(args) -> int:
    names = args.designs or E.designs(args.data)
    for name in names:
        a = RT.audit(name, args.data, q=args.q, operands=args.operands, credit=args.credit, vector=args.vector or "")
        print(f"{a.design} on {a.device}: core {pct(a.core)}  ε_machine {pct(a.machine_eps)}  floor {pct(a.floor)}  "
              f"γ {a.gamma:.3f}  binds {', '.join(a.binding)}" + ("" if a.uniform_route else "  [Π has no uniform route]"))
        for r in a.near:
            if r.ceiling:
                how = "the issue ceiling, silicon cannot beat it"
            elif r.flip_pct is None:
                how = "no flip ≤ 10%"
            else:
                how = f"flips at {r.flip_pct}% → {pct(r.core_at_flip)} (at 2%: {pct(r.core_at_2pct)})"
            print(f"    {r.name:12s} load {r.load:.3f}  dual {r.dual:.3f}: {how}")
        if not a.uniform_route:
            base, opened, binding = RT.uniform_gap(name, args.data, q=args.q, operands=args.operands, credit=args.credit,
                                                   vector=args.vector or "")
            print(f"    uniform-datapath gap: core {pct(base)} → {pct(opened)} with a sibling-rate uniform route "
                  f"(binds {', '.join(binding) or 'none'})")
    return 0
