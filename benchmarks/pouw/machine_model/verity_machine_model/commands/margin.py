"""machine-model margin [DESIGN ...] [--device sm_NN] [--m M] [--least [--mix FFMA]] [--q Q] [--operands O] [--credit C]
                     [--vector V]
                                          the margin (redteam.margin_audit): each design's core as built,
                                          its worst core with every near measured facet's lane coefficient moved M
                                          in the cheater's favour at once (default 2%), the single facet that drives
                                          it, and each near facet's own move; with --least, the cheapest filler a
                                          word (in --mix, default the design's) that brings the worst back to the
                                          floor (default q* = 1.22, credit floor)"""
from __future__ import annotations

import argparse

from verity_machine_model import eps as E
from verity_machine_model import redteam as RT
from verity_machine_model.commands._shared import device_help, pct


def add(sub) -> None:
    mg = sub.add_parser("margin")
    mg.add_argument("designs", nargs="*", help="the designs to audit (default: every design in the data directory)")
    mg.add_argument("--device", default=argparse.SUPPRESS, help=device_help())
    mg.add_argument("--m", type=float, default=0.02, help="the margin each near facet moves in the cheater's favour "
                    "(default %(default)s, 2%%)")
    mg.add_argument("--least", action="store_true", help="also find the cheapest filler a word that holds the margin")
    mg.add_argument("--mix", help="with --least: the filler's class mix, comma-separated (default: the design's)")
    mg.add_argument("--q", type=float, default=1.22, help="q*, lane instructions a fresh word (default %(default)s)")
    mg.add_argument("--operands", choices=E.OPERANDS, default="free",
                    help="the cheater's operand delivery on the tensor route (default %(default)s)")
    mg.add_argument("--credit", choices=E.CREDITS, default="floor",
                    help="what ε is a fraction of (default %(default)s: the floor program ♭)")
    mg.add_argument("--vector", help="the vector whose rows are Π (default: model.toml's current)")


def run(args) -> int:
    mix = {c: 1.0 for c in args.mix.split(",")} if args.mix else None
    for name in args.designs or E.designs(args.data):
        mg = RT.margin_audit(name, args.data, m=args.m, q=args.q, operands=args.operands, credit=args.credit,
                             vector=args.vector or "")
        drive = f"driven by {mg.driver}" if mg.driver else "no near facet drives it"
        print(f"{mg.design} on {mg.device}: core {pct(mg.base)}  worst at {mg.m*100:g}% {pct(mg.worst)}  ({drive})")
        for r, s in mg.rows:
            print(f"    {r:12s} alone → {pct(s)}")
        if args.least:
            lf = RT.least_filler(name, args.data, mix=mix, m=args.m, q=args.q, operands=args.operands,
                                 credit=args.credit, vector=args.vector or "")
            how = (f"{lf.per_word:g} a word (worst {pct(lf.core)}, γ {lf.gamma:.3f})" if lf.per_word is not None
                   else f"none ≤ cap holds it (worst {pct(lf.core)})")
            print(f"    least {'+'.join(lf.mix)} filler that holds {mg.m*100:g}%: {how}")
    return 0
