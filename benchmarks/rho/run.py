"""`verity bench rho`: the overhead for every scenario, how it moves, and where it falls below a target.

    uv run python benchmarks/rho/run.py [--scenario NAME ...] [--target 10 --target 100] [--set KEY=VALUE ...]
                                        [--out DIR] [--figures DIR]

Each scenario gets one overhead number, its lower bound (every bound met at once), with its status, where its premises
stand and whether it is vacuous against the comparator `μ / MFU`; the inference overhead beside it (or that it isn't
modeled); and its best known attack in a table of its own, a construction beside the bound rather than an end of it.

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
from rho.classes import REFERENCE, Structure, weight_per_gate
from rho.figures import jsonable
from rho.params import Registry
from rho.scenario import Estimate, FamilyResult, Result, Scenario


def fmt(x: float) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "—" if math.isnan(x) else "∞"
    return f"{x:.3g}"


def row(e: Estimate) -> dict:
    s = e.overhead
    return {"argument": e.argument, "side": e.side, "status": e.status, "premise": e.premise, "rho": s.rho if s else None,
            "g": s.g if s else None, "declared_rho": e.declared, "outside": e.outside, "excluded": list(e.excluded),
            "binding": {k: v.cut for k, v in s.serving.items()} if s else {},
            "terms": dict(s.terms) if s else {}}


def when(s: Scenario, overhead: float) -> str:
    return operating.human(s.seconds(overhead))


def lower(overhead: float, unmonitored: float) -> str:
    """A lower bound, flagged where it is below the adversary no protocol encumbers (`Result.unmonitored`)."""
    return fmt(overhead) + (" (vacuous)" if overhead < unmonitored * (1 - 1e-9) else "")


def picked(r: Result) -> FamilyResult | None:
    """The family the adversary picks: the one with the least lower bound."""
    return min(r.families, key=lambda f: f.lo, default=None)


def best_attack(r: Result) -> tuple[FamilyResult, Estimate] | None:
    """The best known attack over the scenario's families, and the family it trains."""
    found = [(f, f.best_attack) for f in r.families if f.best_attack]
    return min(found, key=lambda fa: fa[1].rho, default=None)


def setter(f: FamilyResult) -> Estimate | None:
    """The bound that sets the lower bound: of those it admits, the one with the most overhead alone at a standing
    short of refuted."""
    met = [e for e in f.bounds if e.overhead and e.premise != "refuted" and e.argument not in f.combined.excluded]
    return max(met, key=lambda e: e.rho, default=None)


def monitored(r: Result) -> float:
    """The share of the adversary's counted weight on monitored gates, for the family the adversary picks."""
    best = picked(r)
    return best.combined.overhead.monitored if best and best.combined.overhead else math.nan


def inference(s: Scenario) -> str:
    """The inference overhead, the honest inference workload's under the partition, or that it isn't modeled."""
    v = s.context.policy.inference_overhead
    return fmt(v) if math.isfinite(v) else "not modeled"


def straddles(f: FamilyResult | None) -> list[str]:
    return [c for c in Structure.STRADDLE if f and c in f.combined.cites]


def rests(r: Result, terms: list[analysis.Term]) -> str:
    """What the lower bound rests on, for the summary: its largest term (the dark leaves, at the dark rate) and the
    straddling premise where it cites it."""
    if not terms:
        return ""
    t = terms[0]
    head = f"the dark rate ({t.share:.0%})" if t.name == "dark" else f"{t.name} ({t.share:.0%})"
    return ", ".join([head, *straddles(picked(r))])


def rests_md(r: Result, terms: list[analysis.Term]) -> str:
    f = picked(r)
    if not terms or f is None:
        return ""
    t, st = terms[0], r.scenario.structure
    if t.name == "dark" and st.s_times == REFERENCE:
        what = ("the dark hardware's datasheet product rate (the physical model: `s_×` is the reference's `2G / κ`, so "
                "`G` cancels and the dark leaves, `ε_dark` of `Φ_total`, count at `κ/2` products a datasheet FLOP), "
                "not a Boolean structural minimum")
    elif t.name == "dark":
        what = (f"the dark leaves at `s_×` = {fmt(st.gates(weight_per_gate(f.profile, r.scenario.context.units)))} "
                f"gates a counted operation (`structure.{st.name}`)")
    else:
        what = f"its largest term, `{t.name}`"
    straddle = straddles(f)
    return (f"**The lower bound rests on** {what}: {t.share:.1%} of the bound's weight per gate, and the overhead ≤ "
            f"{fmt(t.ceiling)} while it remains" + (f"; and on `{'`, `'.join(straddle)}`" if straddle else "") + ".")


def certified(s: Scenario) -> dict | None:
    """Where a certifier sets the serving cut's link term: the certified prompt ingress into a location (bit/s), as an
    intake (a multiple of `α Φ`), and with `declared` a server's declared inputs and outputs."""
    regime = operating._inner(s.regime)
    cert = getattr(regime, "certifier", None)
    if cert is None or getattr(regime, "form", None) == "links":
        return None
    c, loc = s.context, s.context.location
    out = {"prompts": cert.prompts(c), "intake": cert.intake_of(c), "channels": loc.hosts * (cert.timing + cert.side),
           "links": loc.links, "declared": cert.declared, "given": cert.ingress is not None}
    if cert.declared:
        out |= {"r_in": cert.r_in, "r_out": cert.r_out, "window_buckets": cert.window_buckets,
                "declared_x": cert.declared_x, "faulty_unit_bits": cert.faulty_unit_bits}
    return out


def certified_md(d: dict | None) -> list[str]:
    if d is None:
        return []
    if d["given"]:
        basis = "given directly (`ingress`)"
    elif d["declared"]:
        basis = (f"{fmt(d['declared_x'])} × each server's declared inputs, r_in = {fmt(d['r_in'])} bit/s (r_out = "
                 f"{fmt(d['r_out'])} bit/s out) over a {fmt(d['window_buckets'])}-bucket link-window, at the warden "
                 f"form's public parameters, the honest-δ form of draft:per-server-ingress's bound (a faulty ingress unit's "
                 f"{fmt(d['faulty_unit_bits'])} bits are excluded, and 0 when every ingress unit is drawn); line rate, "
                 f"{fmt(d['links'])} bit/s, is its upper end until Lean proves that bound")
    else:
        basis = "`intake · α Φ`, a server's ingress only its declared inputs (draft:declared-ingress)"
    return [(f"**Certified ingress:** {fmt(d['prompts'])} bit/s into a location, intake {fmt(d['intake'])} × α Φ: "
             f"{basis}; plus {fmt(d['channels'])} bit/s of timing and side channels."), ""]


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


def faulty_md(reg: Registry, fo: operating.Row) -> list[str]:
    if math.isinf(fo.capacity):
        return ["Faulty outputs: no sampled proofs, so nothing detects them.", ""]
    drawn = "every unit proved" if fo.q >= 1 else f"steps the sampled proofs draw at q = {fmt(fo.q)} a step"
    head = (f"Faulty outputs ({drawn}): {fmt(fo.capacity)} bits before the first detection, except with chance "
            f"δ = {fmt(fo.delta)}.")
    out = [head, "", "| reference | bits | chance of detection |", "|---|---|---|"]
    out += [f"| {n} | {fmt(reg[f'exfil.{n}.bits'])} | {fmt(p)} |" for n, p in fo.detect.items()]
    return out + [""]


def estimates_md(f: FamilyResult, fu: float) -> list[str]:
    """A family's bounds (each alone, then every one at once: the lower bound) and, in a table of their own, its
    attacks."""
    def cut(e: Estimate) -> str:
        return ", ".join(f"{k}: {v.cut}" for k, v in e.overhead.serving.items())

    out = ["| bound | status | premises | overhead | g | overhead in one inference block | binding cut |",
           "|---|---|---|---|---|---|---|"]
    for e in f.bounds:
        if e.overhead is None:
            out.append(f"| {e.argument} | outside | | | | | {e.outside} |")
        else:
            out.append(f"| {e.argument} | {e.status} | {e.premise} | {lower(e.rho, fu)} | {fmt(e.overhead.g)} | "
                       f"{fmt(e.declared)} | {cut(e)} |")
    c = f.combined
    out.append(f"| **every bound at once: the lower bound** | {c.status} | {c.premise} | **{lower(c.rho, fu)}** | "
               f"{fmt(c.overhead.g)} | {fmt(c.declared)} | {cut(c)} |")
    out.append("")
    if c.excluded:
        out += [(f"Left out of the lower bound at the block that binds, their premises refuted there: "
                 f"{', '.join(c.excluded)}."), ""]
    out += ["| attack | status | overhead | g | overhead in one inference block | layout |",
            "|---|---|---|---|---|---|"]
    for e in f.attacks:
        if e.overhead is None:
            out.append(f"| {e.argument} | outside | | | | {e.outside} |")
        else:
            out.append(f"| {e.argument} | {e.status} | {fmt(e.rho)} | {fmt(e.overhead.g)} | {fmt(e.declared)} | "
                       f"{cut(e)} |")
    return out + [""]


def scenario_md(r: Result, by_status: dict[str, float], terms: list[analysis.Term],
                points: list[operating.Point], faulty: list[str]) -> list[str]:
    s = r.scenario
    c = s.context
    delta = s.regime.delta(c)
    by = "; ".join(f"{k} {fmt(co)} + {fmt(cr)}" for k, (co, cr) in delta.by.items())
    prf = ", the wipe's PRF" if "wipe" in delta.by else ""
    unencumbered = operating.human(s.reference / (c.fleet.mfu_train * s.total_phi))
    setting = (f"{s.regime.name} regime, {c.fleet.count:g} × {c.location.name} (Φ = {c.location.phi:.3g} gate/s, "
               f"ε_dark {c.fleet.eps_dark:g}), policy {c.policy.name} (c = {c.c:.3g} gates, k = {c.policy.k:.3g} bit), "
               f"{c.horizon.name}, {c.units.gates_per_flop:g} gates a FLOP ({c.units.basis}); target {s.run.name} "
               f"(T_0 = {s.reference:.3g} gates; the reference at the training MFU takes {unencumbered}).")
    un = r.unmonitored
    verdict = ("the lower bound is below it, so **vacuous**: it doesn't exclude even that adversary" if r.vacuous(r.lo)
               else f"the lower bound is {fmt(r.lo / un)} times it")
    st = s.structure
    best = picked(r)
    gates = fmt(st.gates(weight_per_gate(best.profile, c.units))) if best else "—"
    s_times = (f"the reference's gates per counted product (`2G / κ`, {gates} for {best.family if best else '—'})"
               if st.s_times == REFERENCE else f"{gates} gates")
    premise = f", under {', '.join(st.cites())}." if st.cites() else ", which needs no straddling premise."
    sets = setter(best) if best else None
    claim = (f"**Overhead ≥ {lower(r.lo, un)}** ({best.combined.status}; premises {best.combined.premise or '—'}), "
             f"the circuit's size over the reference's: every bound met at once"
             + (f", set by {sets.argument}" if sets else "") if best else "**Overhead:** no family")
    attack = best_attack(r)
    attack_md = (f"**Best known attack:** {attack[1].argument} on {attack[0].family} ({attack[1].status}) at overhead "
                 f"{fmt(attack[1].rho)}, {when(s, attack[1].rho)} to the target: a construction, beside the bound."
                 if attack else "**Best known attack:** none applies.")
    out = [f"## {s.name}", "", setting, "",
           (f"{claim}. **Time to the target ≥ {when(s, r.lo)}** (wall-clock, the overhead times `T_0 / Φ_total`, the "
            f"reference at the inventory's full capacity). **Inference overhead:** {inference(s)}."),
           "",
           (f"With no protocol the adversary sits at **overhead {fmt(un)}** (`μ / MFU`, its circuit padded to capacity "
            f"at MFU {fmt(s.mfu)}): {verdict}. A bound counts each uncertified or dark gate at `ω_max / s_×` with "
            f"`s_×` {s_times} (`structure.{st.name}`){premise}"),
           "", rests_md(r, terms), "", attack_md, "",
           (f"The envelopes fail, summed over subprotocols and serving blocks, with probability at most δ_coin = "
            f"{fmt(delta.coin)} (the coins' draws) plus δ_crypto = {fmt(delta.crypto)} (binding, knowledge soundness"
            f"{prf}), plus Pr[Break] (a forgery of the logged coins, not bounded here); by subprotocol, coin + "
            f"crypto: {by or 'none'}."), "",
           "The lower bound from only the bounds and cuts at least as strong as: "
           + ", ".join(f"{st} {lower(v, un)}" for st, v in by_status.items()) + ".", "",
           "What sets the overhead (each part's share of the bound's weight per gate `dark + (1 − ε_dark) g`, and the "
           "most it can be while it remains): "
           + ", ".join(f"{t.name} {t.share:.1%} (≤ {fmt(t.ceiling)})" for t in terms) + ".", "",
           f"The adversary holds {monitored(r):.1%} of its counted weight on monitored gates.", ""]
    out += certified_md(certified(s)) + operating_md(points) + faulty
    for f in r.families:
        fu = f.profile.efficiency / s.mfu
        out += [f"### {f.family} (μ = {fmt(f.profile.efficiency)}; no protocol: {fmt(fu)})", ""]
        out += estimates_md(f, fu)
        terms_ = f.combined.overhead.terms
        out += ["g's terms (lower bound): " + ", ".join(f"{k} {fmt(v)}" for k, v in terms_.items() if v), ""]
        if f.contradictions:
            out += [f"**Contradiction:** {', '.join(f.contradictions)} exceed the best attack: a premise is false here.", ""]
    return out


def crossings(ds: list[analysis.Danger]) -> str:
    return "; ".join(f"`{d.key}` past {fmt(d.crossing)} (overhead {fmt(d.rho_worst)} at {fmt(d.worst)})" for d in ds)


def below_md(name: str, base: float, un: float, by_status: dict[str, float], corner: analysis.Corner,
               dangers: dict[float, list[analysis.Danger]], rescues: dict[float, list[analysis.Danger]]) -> list[str]:
    out = [f"### {name} (overhead ≥ {lower(base, un)} today; no protocol: {fmt(un)})", ""]
    if corner.path:
        prev, steps, rest = base, [], 0
        for k, v in corner.path:
            if v < prev * (1 - 5e-3):
                steps.append(f"`{k}` → {fmt(corner.values.get(k, math.nan))}: {fmt(v)}")
            else:
                rest += 1
            prev = v
        more = f", and {rest} more that each move it under 0.5%" if rest else ""
        out.append(f"- **Corner:** overhead ≥ {fmt(corner.rho)} with every ranged parameter at the worse end of its "
                   f"range at once; one at a time, worst first: {', '.join(steps)}{more}"
                   + ("." if corner.joint == corner.rho else f" (all at once: {fmt(corner.joint)})."))
    else:
        out.append("- **Corner:** no ranged parameter lowers the lower bound.")
    for t in sorted(dangers):
        if base >= t:
            if dangers[t]:
                found = f"falls below it within a plausible range: {crossings(dangers[t])}."
            else:
                found = ("no single parameter's plausible range takes it below; "
                         + (f"the corner does (overhead ≥ {fmt(corner.rho)})." if corner.rho < t
                            else "nor does the corner."))
            weaker = [st for st, v in by_status.items() if v < t]
            if weaker:
                found += (f" From only {weaker[-1]} or stronger bounds and cuts, overhead ≥ "
                          f"{fmt(by_status[weaker[-1]])}.")
        else:
            found = "already below it today; " + (f"lifted above it by: {crossings(rescues[t])}." if rescues[t]
                                                  else "no single parameter's plausible range lifts it there.")
        out.append(f"- **Overhead {fmt(t)}:** {found}")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", help="scenario names (default: all)")
    ap.add_argument("--target", action="append", type=float, help="overhead targets for the danger scan "
                    "(default: 10, 100)")
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
    head = ("| scenario | overhead ≥ | status, premises | set by | no protocol (`μ / MFU`) | inference overhead | "
            "time to the target ≥ | from conditional or stronger bounds and cuts | on monitored gates | rests on | "
            "at the corner |")
    summary = ["# The overhead", "",
               ("The overhead is `|A| / T_0`: the size of the actual circuit that reaches the target over the "
                "reference's, a factor. The claim is its lower bound, every bound met at once; a bound whose premises "
                "are refuted at a block doesn't bound that block, and is still listed alone, tagged refuted. A lower "
                "bound below the adversary no protocol encumbers (`μ / MFU`) is vacuous. The time to the target, "
                "wall-clock, is at least the overhead times `T_0 / Φ_total`."), "", head,
               "|---|---|---|---|---|---|---|---|---|---|---|"]
    attacks = ["## Attacks", "",
               ("The best known attack on each scenario: a construction, and the time it takes, beside the lower bound "
                "rather than an end of it. A bound above it is a contradiction: one of the bound's premises is false "
                "there."), "",
               "| scenario | attack | family | status | overhead | time to the target | contradictions |",
               "|---|---|---|---|---|---|---|"]
    below = ["## Where the overhead falls below target", ""]
    record: dict = {"targets": targets, "scenarios": {}}
    for name in names:
        r = analysis.result(reg, name)
        by_status = {st: analysis.at(reg, name, None, analysis.upto(st)) for st in analysis.STATUSES[:3]}
        terms = analysis.attribution(r)
        points = operating.points(reg, r)
        fo = operating.faulty(reg, r.scenario)
        md += scenario_md(r, by_status, terms, points, faulty_md(reg, fo))
        swings = analysis.tornado(reg, name)
        un = r.unmonitored
        md += [f"Tornado (the lower bound at each parameter's plausible ends; no protocol: {fmt(un)}):", "",
               "| parameter | low | high | overhead ≥ at low | overhead ≥ at high | decades |",
               "|---|---|---|---|---|---|"]
        md += [f"| {s.key} | {fmt(s.low)} | {fmt(s.high)} | {lower(s.at_low, un)} | {lower(s.at_high, un)} | "
               f"{fmt(s.size)} |" for s in swings[:12]]
        md.append("")
        dangers = {t: analysis.dangers(reg, name, t, swings=swings) for t in targets}
        rescues = {t: analysis.rescues(reg, name, t, swings=swings) for t in targets}
        corner = analysis.corner(reg, name, swings=swings)
        below += below_md(name, r.lo, un, by_status, corner, dangers, rescues)
        s = r.scenario
        best = picked(r)
        sets = setter(best) if best else None
        standing = f"{best.combined.status}, {best.combined.premise}" if best else ""
        summary.append(f"| {name} | **{lower(r.lo, un)}** | {standing} | {sets.argument if sets else ''} | {fmt(un)} | "
                       f"{inference(s)} | {when(s, r.lo)} | {lower(by_status['conditional'], un)} | "
                       f"{monitored(r):.1%} | {rests(r, terms)} | {lower(corner.rho, un)} |")
        attack = best_attack(r)
        contradictions = sorted({c for f in r.families for c in f.contradictions})
        attacks.append(f"| {name} | " + (f"{attack[1].argument} | {attack[0].family} | {attack[1].status} | "
                                         f"{fmt(attack[1].rho)} | {when(s, attack[1].rho)}" if attack
                                         else "none | | | | ")
                       + f" | {', '.join(contradictions) or 'none'} |")
        policy = s.context.policy
        record["scenarios"][name] = {
            "overhead": r.lo, "status": best.combined.status if best else None,
            "premise": best.combined.premise if best else None, "set_by": sets.argument if sets else None,
            "excluded": list(best.combined.excluded) if best else [], "unmonitored": un, "vacuous": r.vacuous(r.lo),
            "inference_overhead": {"value": policy.inference_overhead,
                                   "source": reg.param(f"policy.{policy.name}.inference_overhead").source
                                   if f"policy.{policy.name}.inference_overhead" in reg else "not modeled"},
            "attack": {"argument": attack[1].argument, "family": attack[0].family, "status": attack[1].status,
                       "rho": attack[1].rho, "seconds": s.seconds(attack[1].rho)} if attack else None,
            "contradictions": contradictions,
            "structure": {"name": s.structure.name, "s_times": s.structure.s_times, "basis": s.structure.basis,
                          "cites": s.structure.cites()},
            "units": {"gates_per_flop": s.context.units.gates_per_flop, "basis": s.context.units.basis},
            "delta": s.regime.delta(s.context).json(),
            "certified": certified(s),
            "seconds": s.seconds(r.lo),
            "overhead_by_status": by_status,
            "attribution": [vars(t) for t in terms], "monitored": monitored(r),
            "operating": [{"name": p.name, "kind": p.kind, "seconds": dict(p.seconds)} for p in points],
            "faulty_outputs": {"capacity_bits": fo.capacity, "detect": dict(fo.detect)},
            "families": {f.family: {"overhead": f.lo, "attack": f.attack, "excluded": list(f.combined.excluded),
                                    "contradictions": list(f.contradictions),
                                    "bounds": [row(e) for e in f.bounds], "combined": row(f.combined),
                                    "attacks": [row(e) for e in f.attacks]}
                         for f in r.families},
            "tornado": [vars(s) for s in swings],
            "dangers": {str(t): [vars(d) for d in ds] for t, ds in dangers.items()},
            "rescues": {str(t): [vars(d) for d in ds] for t, ds in rescues.items()},
            "corner": {"rho": corner.rho, "base": corner.base, "joint": corner.joint, "values": dict(corner.values),
                       "path": corner.path},
        }
    text = "\n".join(summary + [""] + attacks + [""] + md + below)
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
