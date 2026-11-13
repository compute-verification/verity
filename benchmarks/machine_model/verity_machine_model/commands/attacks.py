"""machine-model attacks [NAME] [--design D] [-n N]
                                          the N cheapest attacks of each design (or of D) at NAME's prices"""
from __future__ import annotations

from verity_machine_model import attacks as atk
from verity_machine_model.model import model
from verity_machine_model.resolve import resolve


def add(sub) -> None:
    a = sub.add_parser("attacks")
    a.add_argument("name", nargs="?")
    a.add_argument("--design")
    a.add_argument("-n", type=int, default=3)


def run(args) -> int:
    r = resolve(args.name or model(args.data).current, args.data)
    designs = [args.design] if args.design else sorted({a.design for a in r.attacks})
    for d in designs:
        xs, tp = atk.cheapest(r.census, r.solution, r.attacks, d, args.n)
        print(f"{d}:" + (f"  (three-piece (iv) X {tp:.3f})" if tp is not None else ""))
        for x, fam, nm in xs:
            print(f"  X {x:.3f}  {fam}: {nm}")
    return 0
