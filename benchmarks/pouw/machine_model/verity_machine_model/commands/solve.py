"""machine-model solve [NAME] [--json]       re-solve NAME (default: model.toml's `current`): τ, τ at 600 W, U/F_low and
                                          margin against what was published, θ, the cheapest attacks, the check,
                                          then whether (π₃₂), its load half and π_c ≥ 16 hold
                                          (data/floors.csv), and every copy route's price and measured cost a word"""
from __future__ import annotations

import json
import textwrap

from verity_machine_model import attacks as atk
from verity_machine_model.model import CLAIMS, model
from verity_machine_model.resolve import resolve


def add(sub) -> None:
    s = sub.add_parser("solve")
    s.add_argument("name", nargs="?")
    s.add_argument("--json", action="store_true")


def run(args) -> int:
    name = args.name or model(args.data).current
    r = resolve(name, args.data)
    v, s = r.vector, r.solution
    bad = r.unreproduced()
    if args.json:
        print(json.dumps({"vector": name, "epoch": v.epoch, **r.numbers(), "tau_max": s.tau_max, "theta": s.theta,
                          "cheapest": r.cheapest, "three_piece": r.three_piece, "attacks": len(r.attacks),
                          "published": v.published, "unreproduced": bad, "failures": r.failures,
                          "floors": {k: {"kinds": n, "below": b, "absent": ab} for k, (n, b, ab) in r.floors.items()},
                          "copies": [{"kind": k, "words": w, "price": p, "measured": mc, "what": what}
                                     for k, w, p, mc, what in r.copies]},
                         ensure_ascii=False,
                         indent=1))
        return 1 if bad or r.failures else 0
    print(f"{name} (epoch {v.epoch}, {len(v.rows)} rows, {len(v.credits)} credits): {v.summary}")
    if v.superseded_by:
        print(f"  superseded by {v.superseded_by}")
    if v.note:
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
    for claim, (n, below, absent) in r.floors.items():
        print(f"  {CLAIMS[claim]}: {n} kinds, " + ("holds" if not below else "fails: " + "; ".join(below))
              + (f" (not split out at {v.epoch}: {', '.join(absent)})" if absent else ""))
    print(textwrap.fill("copy routes, MACs a word (price/measured), cheapest first: "
                        + ", ".join(f"{k} {p:.2f}/{mc:.4g}" for k, _w, p, mc, _ in r.copies),
                        width=120, initial_indent="  ", subsequent_indent="    "))
    return 1 if bad or r.failures else 0
