"""`machine-model`: re-solve a named price vector from the data and check every attack against it.

    machine-model list                        the named vectors: epoch, rows, credits and published τ
    machine-model solve [NAME] [--json]       re-solve NAME (default: model.toml's `current`): τ, τ at 600 W, U/F_low and
                                              margin against what was published, θ, the cheapest attacks, then the check
    machine-model census [EPOCH] [--kind K]   each instruction class's u on each row at EPOCH (default: the last)
    machine-model attacks [NAME] [--design D] [-n N]
                                              the N cheapest attacks of each design (or of D) at NAME's prices

`--data DIR` (before the command) reads another data directory, such as a copy with a candidate row. `solve` exits 1 when
the vector fails its check or doesn't reproduce a published number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from verity_machine_model import attacks as atk
from verity_machine_model.model import DATA, census, model, vector, vectors
from verity_machine_model.resolve import resolve


def _list(args) -> int:
    print(f"{'vector':16s} {'epoch':16s} {'rows':>4s} {'credits':>7s} {'τ':>7s} {'margin':>6s}  summary")
    for n in vectors(args.data):
        v = vector(n, args.data)
        tag = f" (superseded by {v.superseded_by})" if v.superseded_by else ""
        print(f"{n:16s} {v.epoch:16s} {len(v.rows):4d} {len(v.credits):7d} {v.published.get('tau', float('nan')):7.4f} "
              f"{v.published.get('margin', float('nan')):6.3f}  {v.summary}{tag}")
    return 0


def _solve(args) -> int:
    name = args.name or model(args.data).current
    r = resolve(name, args.data)
    v, s = r.vector, r.solution
    bad = r.unreproduced()
    if args.json:
        print(json.dumps({"vector": name, "epoch": v.epoch, **r.numbers(), "tau_max": s.tau_max, "theta": s.theta,
                          "cheapest": r.cheapest, "three_piece": r.three_piece, "attacks": len(r.attacks),
                          "published": v.published, "unreproduced": bad, "failures": r.failures}, ensure_ascii=False,
                         indent=1))
        return 1 if bad or r.failures else 0
    print(f"{name} (epoch {v.epoch}, {len(v.rows)} rows, {len(v.credits)} credits): {v.summary}")
    if v.superseded_by:
        print(f"  superseded by {v.superseded_by}")
    print(f"  {v.note}")
    print(f"  τ {r.tau:.4f}   τ at 600 W {r.tau_600w:.4f}   U/F_low {r.uf_low:.4f}   margin {r.margin:.3f}")
    if v.published:
        print("  published: " + "   ".join(f"{k} {x}" for k, x in v.published.items())
              + ("   (reproduced)" if not bad else ""))
    else:
        print("  published: nothing yet (record these under [published] in its vector file)")
    for b in bad:
        print(f"  NOT REPRODUCED: {b}")
    print(f"  τ-max {s.tau_max:.4f} at U/F_low {1 + s.infl_at_tau_max:.4f}")
    print("  θ: " + ", ".join(f"{k} {x:.3f}" for k, x in sorted(s.theta.items(), key=lambda t: -t[1]) if x > 5e-4))
    print(f"  cheapest in {r.model.adopted}: " + "; ".join(f"X {x:.3f} {fam}: {nm}" for x, fam, nm in r.cheapest)
          + (f"; three-piece (iv) X {r.three_piece:.3f}" if r.three_piece is not None else ""))
    fams = atk.families(r.attacks)
    print(f"  check: {len(r.census)} kinds at or below measured cost, ρ ≥ {r.model.rho} on the CUDA kinds, "
          f"{len(v.credits)} credits paid, {len(r.attacks)} attacks in {len(fams)} families at X ≥ {r.model.x_min}: "
          + ("pass" if not r.failures else f"FAIL ({len(r.failures)})"))
    for f in r.failures[:20]:
        print(f"    {f}")
    return 1 if bad or r.failures else 0


def _census(args) -> int:
    m = model(args.data)
    epoch = args.epoch or m.epochs[-1]
    if epoch not in m.epochs:
        raise SystemExit(f"no epoch {epoch!r}: one of {', '.join(m.epochs)}")
    c = census(m, epoch, args.data)
    for k in sorted(c):
        if args.kind and k != args.kind:
            continue
        print(f"{k}: " + ", ".join(f"{r} {u / m.cycle_u:.4g}" for r, u in c[k].items()) + "  (SMSP-cycles)")
    return 0


def _attacks(args) -> int:
    r = resolve(args.name or model(args.data).current, args.data)
    designs = [args.design] if args.design else sorted({a.design for a in r.attacks})
    for d in designs:
        xs, tp = atk.cheapest(r.census, r.solution, r.attacks, d, args.n)
        print(f"{d}:" + (f"  (three-piece (iv) X {tp:.3f})" if tp is not None else ""))
        for x, fam, nm in xs:
            print(f"  X {x:.3f}  {fam}: {nm}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="machine-model", description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA, help="the data directory (default: %(default)s)")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("list")
    s = sub.add_parser("solve")
    s.add_argument("name", nargs="?")
    s.add_argument("--json", action="store_true")
    c = sub.add_parser("census")
    c.add_argument("epoch", nargs="?")
    c.add_argument("--kind")
    a = sub.add_parser("attacks")
    a.add_argument("name", nargs="?")
    a.add_argument("--design")
    a.add_argument("-n", type=int, default=3)
    args = p.parse_args(argv)
    if args.cmd is None:
        print(__doc__)
        return 0
    return {"list": _list, "solve": _solve, "census": _census, "attacks": _attacks}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
