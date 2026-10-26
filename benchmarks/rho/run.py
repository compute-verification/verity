"""`verity bench rho`: ρ for every scenario, how it moves, and where it falls below a target.

    uv run python benchmarks/rho/run.py [--scenario NAME ...] [--target 10 --target 100] [--set KEY=VALUE ...]
                                        [--out DIR] [--figures DIR]

Prints Markdown; with `--out` (or `$OUT`, as under `research run`) also writes `rho.md`, `rho.json` and the figures (in
`figures/`) there. `--figures DIR` writes the figures (`rho/figures.py`, matplotlib) to DIR. `--set` overrides any
registry key for the whole run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rho import analysis, operating
from rho.figures import jsonable
from rho.params import Registry
from rho.scenario import YEAR, Estimate, Result, Scenario


def fmt(x: float) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "—" if math.isnan(x) else "∞"
    return f"{x:.3g}"


def row(e: Estimate) -> dict:
    s = e.slowdown
    return {"argument": e.argument, "side": e.side, "status": e.status, "premise": e.premise, "rho": s.rho if s else None,
            "g": s.g if s else None, "declared_rho": e.declared, "outside": e.outside,
            "binding": {k: v.cut for k, v in s.regions.items()} if s else {},
            "terms": dict(s.terms) if s else {}}


def when(s: Scenario, rho: float) -> str:
    return operating.human(s.years(rho) * YEAR)


def monitored(r: Result) -> float:
    """The share of the adversary's training done on monitored compute, for the family the adversary picks."""
    best = min(r.families, key=lambda f: f.lo, default=None)
    return best.combined.slowdown.monitored if best and best.combined.slowdown else math.nan


def operating_md(points: list[operating.Point]) -> list[str]:
    out = []
    for kind, head in (("work", "work"), ("exfil", "exfiltration through side channels")):
        rows = [p for p in points if p.kind == kind]
        if rows:
            cols = list(rows[0].seconds)
            out += [f"Operating points, {head}:", "", "| reference | " + " | ".join(cols) + " |",
                    "|---|" + "---|" * len(cols)]
            out += [f"| {p.name} | " + " | ".join(operating.human(p.seconds[c]) for c in cols) + " |" for p in rows]
            out.append("")
    return out


def scenario_md(r: Result, by_status: dict[str, float], terms: list[analysis.Term],
                points: list[operating.Point]) -> list[str]:
    s = r.scenario
    c = s.context
    setting = (f"{s.regime.name} regime, {c.fleet.count:g} × {c.location.name} (dark share {c.fleet.dark:g}), policy "
               f"{c.policy.name} (Γ = {c.policy.gamma:.3g} FLOP, k = {c.policy.k:.3g} bit), {c.horizon.name}; target "
               f"{s.run.name} ({s.run.target(s.model):.3g} FLOP, {when(s, 1.0)} unencumbered).")
    out = [f"## {s.name}", "", setting, "",
           f"**ρ ∈ [{fmt(r.lo)}, {fmt(r.hi)}]**: {when(s, r.lo)} to {when(s, r.hi)} to the target.", "",
           "ρ's lower end from only the bounds and cuts at least as strong as: "
           + ", ".join(f"{st} {fmt(v)}" for st, v in by_status.items()) + ".", "",
           "What sets ρ (each part's share of the adversary's rate `d + (1 − d) g`, and the most ρ can be while it "
           "remains): " + ", ".join(f"{t.name} {t.share:.1%} (≤ {fmt(t.ceiling)})" for t in terms) + ".", "",
           f"The adversary does {monitored(r):.1%} of its training on monitored compute.", ""]
    out += operating_md(points)
    for f in r.families:
        out += [f"### {f.family} (m = {fmt(f.profile.efficiency)})", "",
                "| argument | side | status | premises | ρ | g | ρ in one declared block | binding cut |",
                "|---|---|---|---|---|---|---|---|"]
        for e in f.bounds + (f.combined,) + f.attacks:
            d = row(e)
            if e.slowdown is None:
                out.append(f"| {e.argument} | {e.side} | outside | | | | | {e.outside} |")
            else:
                out.append(f"| {e.argument} | {e.side} | {e.status} | {e.premise} | {fmt(d['rho'])} | {fmt(d['g'])} | "
                           f"{fmt(e.declared)} | {', '.join(f'{k}: {v}' for k, v in d['binding'].items())} |")
        terms_ = f.combined.slowdown.terms
        out += ["", "g's terms (lower bound): " + ", ".join(f"{k} {fmt(v)}" for k, v in terms_.items() if v), ""]
        if f.contradictions:
            out += [f"**Contradiction:** {', '.join(f.contradictions)} exceed the best attack: a premise is false here.", ""]
    return out


def crossings(ds: list[analysis.Danger]) -> str:
    return "; ".join(f"`{d.key}` past {fmt(d.crossing)} (ρ {fmt(d.rho_worst)} at {fmt(d.worst)})" for d in ds)


def regions_md(name: str, base: float, by_status: dict[str, float], corner: analysis.Corner,
               dangers: dict[float, list[analysis.Danger]], rescues: dict[float, list[analysis.Danger]]) -> list[str]:
    out = [f"### {name} (ρ = {fmt(base)} today)", ""]
    if corner.path:
        prev, steps, rest = base, [], 0
        for k, v in corner.path:
            if v < prev * (1 - 5e-3):
                steps.append(f"`{k}` → {fmt(corner.values.get(k, math.nan))}: {fmt(v)}")
            else:
                rest += 1
            prev = v
        more = f", and {rest} more that each move it under 0.5%" if rest else ""
        out.append(f"- **Corner:** ρ = {fmt(corner.rho)} with every ranged parameter at the worse end of its range "
                   f"at once; one at a time, worst first: {', '.join(steps)}{more}"
                   + ("." if corner.joint == corner.rho else f" (all at once: {fmt(corner.joint)})."))
    else:
        out.append("- **Corner:** no ranged parameter lowers ρ.")
    for t in sorted(dangers):
        if base >= t:
            if dangers[t]:
                found = f"falls below it within a plausible range: {crossings(dangers[t])}."
            else:
                found = ("no single parameter's plausible range takes it below; "
                         + (f"the corner does (ρ {fmt(corner.rho)})." if corner.rho < t else "nor does the corner."))
            weaker = [st for st, v in by_status.items() if v < t]
            if weaker:
                found += f" From only {weaker[-1]} or stronger bounds and cuts, ρ = {fmt(by_status[weaker[-1]])}."
        else:
            found = "already below it today; " + (f"lifted above it by: {crossings(rescues[t])}." if rescues[t]
                                                  else "no single parameter's plausible range lifts it there.")
        out.append(f"- **ρ = {fmt(t)}:** {found}")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", help="scenario names (default: all)")
    ap.add_argument("--target", action="append", type=float, help="ρ targets for the danger scan (default: 10, 100)")
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE registry overrides")
    ap.add_argument("--out", default=os.environ.get("OUT"), help="write rho.md, rho.json and figures/ here ($OUT)")
    ap.add_argument("--figures", help="write the figures (PNG and the JSON they plot) here")
    args = ap.parse_args()

    reg = Registry.load()
    if args.set:
        reg = reg.override({k: json.loads(v) for k, v in (a.split("=", 1) for a in args.set)})
    names = args.scenario or reg.names("scenario")
    targets = sorted(args.target or [10.0, 100.0])

    md: list[str] = []
    head = ("| scenario | ρ ∈ | the bound that sets it: status, premises | from conditional or stronger bounds and cuts | "
            "on monitored compute | set by | at the corner |")
    summary = ["# ρ", "", head, "|---|---|---|---|---|---|---|"]
    regions = ["## Regions where ρ falls below target", ""]
    record: dict = {"targets": targets, "scenarios": {}}
    for name in names:
        r = analysis.result(reg, name)
        by_status = {st: analysis.at(reg, name, None, analysis.upto(st)) for st in analysis.STATUSES[:3]}
        terms = analysis.attribution(r)
        points = operating.points(reg, r)
        md += scenario_md(r, by_status, terms, points)
        swings = analysis.tornado(reg, name)
        md += ["Tornado (ρ's lower end at each parameter's plausible ends):", "",
               "| parameter | low | high | ρ at low | ρ at high | decades |", "|---|---|---|---|---|---|"]
        md += [f"| {s.key} | {fmt(s.low)} | {fmt(s.high)} | {fmt(s.at_low)} | {fmt(s.at_high)} | {fmt(s.size)} |"
               for s in swings[:12]]
        md.append("")
        dangers = {t: analysis.dangers(reg, name, t, swings=swings) for t in targets}
        rescues = {t: analysis.rescues(reg, name, t, swings=swings) for t in targets}
        corner = analysis.corner(reg, name, swings=swings)
        regions += regions_md(name, r.lo, by_status, corner, dangers, rescues)
        met = [e for e in min(r.families, key=lambda f: f.lo).bounds if e.slowdown] if r.families else []
        best = max(met, key=lambda e: e.rho, default=None)
        standing = f"{best.argument}: {best.status}, {best.premise}" if best else ""
        summary.append(f"| {name} | [{fmt(r.lo)}, {fmt(r.hi)}] | {standing} | {fmt(by_status['conditional'])} | "
                       f"{monitored(r):.1%} | " + (f"{terms[0].name} {terms[0].share:.0%}" if terms else "")
                       + f" | {fmt(corner.rho)} |")
        record["scenarios"][name] = {
            "lo": r.lo, "hi": r.hi, "years_lo": r.scenario.years(r.lo), "lo_by_status": by_status,
            "attribution": [vars(t) for t in terms], "monitored": monitored(r),
            "operating": [{"name": p.name, "kind": p.kind, "seconds": dict(p.seconds)} for p in points],
            "families": {f.family: {"lo": f.lo, "hi": f.hi, "contradictions": list(f.contradictions),
                                    "estimates": [row(e) for e in f.bounds + (f.combined,) + f.attacks]}
                         for f in r.families},
            "tornado": [vars(s) for s in swings],
            "dangers": {str(t): [vars(d) for d in ds] for t, ds in dangers.items()},
            "rescues": {str(t): [vars(d) for d in ds] for t, ds in rescues.items()},
            "corner": {"rho": corner.rho, "base": corner.base, "joint": corner.joint, "values": dict(corner.values),
                       "path": corner.path},
        }
    text = "\n".join(summary + [""] + md + regions)
    print(text)
    out = Path(args.out) if args.out else None
    figures = Path(args.figures) if args.figures else (out / "figures" if out else None)
    if figures:
        try:
            from rho.figures import render
            written = render(reg, figures, names, targets)
        except ImportError as e:
            if args.figures:
                raise
            print(f"no figures ({e}): install matplotlib, or pass --figures DIR to fail on it", file=sys.stderr)
        else:
            record["figures"] = {k: [str(p) for p in v] for k, v in written.items()}
            print(f"figures: {figures}", file=sys.stderr)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "rho.md").write_text(text + "\n")
        (out / "rho.json").write_text(json.dumps(jsonable(record), indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
