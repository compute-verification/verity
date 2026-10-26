"""`verity bench rho`: ρ for every scenario, how it moves, and where it falls below a target.

    uv run python benchmarks/rho/run.py [--scenario NAME ...] [--target 10 --target 100] [--set KEY=VALUE ...] [--out DIR]

Prints Markdown; with `--out` (or `$OUT`, as under `research run`) also writes `rho.md` and `rho.json` there. `--set`
overrides any registry key for the whole run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rho import analysis
from rho.params import Registry
from rho.scenario import Estimate, Result, evaluate_named


def fmt(x: float) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "—" if math.isnan(x) else "∞"
    return f"{x:.3g}"


def row(e: Estimate) -> dict:
    s = e.slowdown
    return {"argument": e.argument, "side": e.side, "status": e.status, "rho": s.rho if s else None,
            "g": s.g if s else None, "declared_rho": e.declared, "outside": e.outside,
            "binding": {k: v.cut for k, v in s.regions.items()} if s else {},
            "terms": dict(s.terms) if s else {}}


def scenario_md(r: Result) -> list[str]:
    s = r.scenario
    c = s.context
    setting = (f"{s.regime.name} regime, {c.fleet.count:g} × {c.location.name} (dark share {c.fleet.dark:g}), policy "
               f"{c.policy.name} (Γ = {c.policy.gamma:.3g} FLOP, k = {c.policy.k:.3g} bit), {c.horizon.name}; target "
               f"{s.run.name} ({s.run.target(s.model):.3g} FLOP, {fmt(s.years(1.0))} years unencumbered).")
    out = [f"## {s.name}", "", setting, "",
           f"**ρ ∈ [{fmt(r.lo)}, {fmt(r.hi)}]**: {fmt(s.years(r.lo))} to {fmt(s.years(r.hi))} years to the target.", ""]
    for f in r.families:
        out += [f"### {f.family} (m = {fmt(f.profile.efficiency)})", "",
                "| argument | side | status | ρ | g | ρ in one declared block | binding cut |",
                "|---|---|---|---|---|---|---|"]
        for e in f.bounds + (f.combined,) + f.attacks:
            d = row(e)
            if e.slowdown is None:
                out.append(f"| {e.argument} | {e.side} | outside | | | | {e.outside} |")
            else:
                out.append(f"| {e.argument} | {e.side} | {e.status} | {fmt(d['rho'])} | {fmt(d['g'])} | "
                           f"{fmt(e.declared)} | {', '.join(f'{k}: {v}' for k, v in d['binding'].items())} |")
        terms = f.combined.slowdown.terms
        out += ["", "g's terms (lower bound): " + ", ".join(f"{k} {fmt(v)}" for k, v in terms.items() if v), ""]
        if f.contradictions:
            out += [f"**Contradiction:** {', '.join(f.contradictions)} exceed the best attack: a premise is false here.", ""]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", help="scenario names (default: all)")
    ap.add_argument("--target", action="append", type=float, help="ρ targets for the danger scan (default: 10, 100)")
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE registry overrides")
    ap.add_argument("--out", default=os.environ.get("OUT"))
    args = ap.parse_args()

    reg = Registry.load()
    if args.set:
        reg = reg.override({k: json.loads(v) for k, v in (a.split("=", 1) for a in args.set)})
    names = args.scenario or reg.names("scenario")
    targets = args.target or [10.0, 100.0]

    md = ["# ρ", ""]
    record: dict = {"scenarios": {}}
    for name in names:
        r = evaluate_named(reg, name)
        md += scenario_md(r)
        swings = analysis.tornado(reg, name)
        md += ["Tornado (guaranteed ρ at each parameter's plausible ends):", "",
               "| parameter | low | high | ρ at low | ρ at high | decades |", "|---|---|---|---|---|---|"]
        md += [f"| {s.key} | {fmt(s.low)} | {fmt(s.high)} | {fmt(s.at_low)} | {fmt(s.at_high)} | {fmt(s.size)} |"
               for s in swings[:12]]
        md.append("")
        dangers = {t: analysis.dangers(reg, name, t) for t in targets}
        for t, ds in dangers.items():
            if ds:
                found = "; ".join(f"{d.key} past {fmt(d.crossing)} (ρ {fmt(d.rho_worst)} at {fmt(d.worst)})" for d in ds)
                md += [f"Below ρ = {fmt(t)} within a plausible range: {found}", ""]
        record["scenarios"][name] = {
            "lo": r.lo, "hi": r.hi, "years_lo": r.scenario.years(r.lo),
            "families": {f.family: {"lo": f.lo, "hi": f.hi, "contradictions": list(f.contradictions),
                                    "estimates": [row(e) for e in f.bounds + (f.combined,) + f.attacks]}
                         for f in r.families},
            "tornado": [vars(s) for s in swings],
            "dangers": {str(t): [vars(d) for d in ds] for t, ds in dangers.items()},
        }
    text = "\n".join(md)
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "rho.md").write_text(text + "\n")
        (out / "rho.json").write_text(json.dumps(record, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
