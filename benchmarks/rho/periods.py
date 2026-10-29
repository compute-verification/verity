"""`periods.py`: the slowdown per monitoring period, over one datacenter and over the world's compute.

For each scenario (by default the measured H200 ones, `reference-onsite-h200` and `reference-remote-h200`, and today's
operating point, `reference-onsite-h200-today`) on each fleet (`FLEETS`: one 100k-GPU datacenter, and the world's
compute, the census's world-inference-2027) and each monitoring period (`PERIODS`, the registry's horizons), with the
scenario's own `fleet` and `horizon` choices overridden (`scenario.<name>.fleet`, `scenario.<name>.horizon`):

* **the slowdown's lower bound** and its range: each ranged parameter alone at either end of its plausible range, the
  rest at reference (`analysis.tornado`), and the corner, all of them at their worse ends at once (`analysis.corner`);
  and the padded size ratio `|A| / T_0` it comes from;
* **without the ruled assumptions**: the lower bound with every regime's host CPUs at their datasheet peak (without
  hosts carry no compute), with free compute at peak (without reference efficiency), and without both, sensitivities
  beside the headline (`sensitivities`);
* **its parts**: the citations of the bound that sets it, by kind (`KINDS`), each named premise with its standing at
  the block that binds;
* **what is scaled**: a serving block's local minimum is a per-server-window bound (one server between two wipes,
  onsite; one server over the period, remote), its cuts' statements are per serving block, and the bound sums them over
  every server and window; the dark leaves are one pool, `ε_dark` of the fleet, and the uncertified routes are shares of
  each server's capacity, neither a per-window bound;
* **what sets it** (`analysis.attribution`);
* **the premises ranked** by how far each alone moves the lower bound: each ranged parameter over its range, each named
  premise of `data/premises.toml` refuted, and `draft:no-straddle` dropped (`s_×` = 1, the elementary cap).

`--measured CIRCUIT` evaluates them with a measured step's design in place of the FLOP model (`circuits.in_place`: its
structure's `s_×` and its gates per FLOP), and sets the step beside the FLOP model first (`circuits.compare`).

    uv run python benchmarks/rho/periods.py                # Markdown on stdout
    uv run python benchmarks/rho/periods.py --measured smollm2-135m-step
    uv run python benchmarks/rho/periods.py --out DIR        # and periods.md and periods.json in DIR
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import run
from rho import analysis, circuits, lean, operating
from rho.classes import premise_status, premises
from rho.envelopes import ROUTES
from rho.params import Registry
from rho.scenario import Result, _binding_k

SCENARIOS = ("reference-onsite-h200", "reference-remote-h200", "reference-onsite-h200-today")
#: (fleet, label): one datacenter, and the world's compute
FLEETS = (("reference-h200", "one datacenter"), ("world-h200", "the world"))
PERIODS = ("week", "month", "year")
#: `lean.status` as the kinds a part of a bound is
KINDS = {"proved": "proved in Lean", "assumption": "named premise", "draft": "paper", "absent": "stated, in no lock"}
#: parameter statuses that are a choice or a measurement rather than an open premise
FIXED = ("datasheet", "measured", "declared", "ruled")


def choices(name: str, fleet: str, period: str) -> dict[str, str]:
    return {f"scenario.{name}.fleet": fleet, f"scenario.{name}.horizon": period}


def without_hosts(reg: Registry) -> dict[str, float]:
    """Every regime's host CPUs at their datasheet peak: the registry without hosts carry no compute (`GLOSSARY.md`)."""
    return {f"regime.{r}.host": 1.0 for r in reg.names("regime") if f"regime.{r}.host" in reg}


#: the overrides without the reference-efficiency assumption (`GLOSSARY.md`): a bound counts free compute at peak
WITHOUT_REFERENCE_EFFICIENCY = {"task.reference_efficiency": False}


def sensitivities(reg: Registry, name: str) -> dict[str, float]:
    """The lower bound without each ruled assumption the headline takes, and without both."""
    return {"without_hosts": analysis.at(reg.override(without_hosts(reg)), name),
            "without_reference_efficiency": analysis.at(reg.override(WITHOUT_REFERENCE_EFFICIENCY), name),
            "without_both": analysis.at(reg.override(without_hosts(reg) | WITHOUT_REFERENCE_EFFICIENCY), name)}


def parts(r: Result, reg: Registry) -> dict:
    """The bound that sets the lower bound, its citations by kind, and each named premise's standing where it binds."""
    f = run.picked(r)
    e = run.setter(f) if f else None
    if e is None:
        return {"setter": None, "kinds": {}, "standings": {}}
    k = _binding_k(r.scenario.regime.partition(r.scenario.context), e.overhead)
    table = premises(reg)
    kinds: dict[str, list[str]] = {v: [] for v in KINDS.values()}
    for c in dict.fromkeys(e.cites):
        s = lean.status(c)
        if s in KINDS:
            kinds[KINDS[s]].append(c)
    named = [c for c in dict.fromkeys(e.cites) if c in table or lean.status(c) == "assumption"]
    return {"setter": e.argument, "status": e.status, "premise": e.premise, "binding_bits": k, "kinds": kinds,
            "standings": {c: premise_status([c], k, table=table) for c in named}}


def scaled(r: Result) -> dict:
    """The serving blocks the bound sums a per-server-window bound over, and the share of the bound that isn't one."""
    s, part = r.scenario, r.scenario.regime.partition(r.scenario.context)
    phi = s.context.location.phi
    blocks = [{"name": b.name, "count": b.count, "seconds": b.gates / phi,
               "cuts": {c.name: list(c.cites) for c in b.cuts}} for b in part.serving]
    terms = analysis.attribution(r)
    return {"blocks": blocks,
            "per_window_share": math.fsum(t.share for t in terms if t.name.split(".")[0] in ("serving", "inference")),
            "dark_share": math.fsum(t.share for t in terms if t.name == "dark"),
            "routes_share": math.fsum(t.share for t in terms if t.name.split(".")[0] in ROUTES)}


def ranking(reg: Registry, name: str, swings: list[analysis.Swing]) -> list[dict]:
    """Every ranged parameter over its range, every named premise refuted, `draft:no-straddle` dropped, by how many
    decades each alone moves the lower bound."""
    base = analysis.at(reg, name)
    out = [{"what": s.key, "kind": f"parameter ({reg.param(s.key).status})",
            "open": reg.param(s.key).status not in FIXED, "range": [s.low, s.high],
            "at": [s.at_low, s.at_high], "decades": s.size} for s in swings]
    for p in reg.names("premise"):
        cited = reg[f"premise.{p}.lean"]
        v = analysis.at(reg, name, {f"premise.{p}.standing": "refuted", f"premise.{p}.beyond": "refuted"})
        if v != base:
            standing = f"{reg[f'premise.{p}.standing']} ({reg[f'premise.{p}.beyond']} above its `credible_bits`)"
            out.append({"what": cited, "kind": "named premise", "open": True,
                        "range": [standing, "refuted"], "at": [base, v], "decades": _decades(base, v)})
    st = analysis.result(reg, name).scenario.structure.name
    v = analysis.at(reg, name, {f"structure.{st}.s_times": 1})
    out.append({"what": "draft:no-straddle", "kind": "named premise", "open": True,
                "range": ["assumed", "dropped (s_× = 1)"], "at": [base, v], "decades": _decades(base, v)})
    return sorted((x for x in out if x["decades"] > 0), key=lambda x: -x["decades"])


def _decades(a: float, b: float) -> float:
    vals = [v for v in (a, b) if v > 0 and math.isfinite(v)]
    return math.log10(max(vals) / min(vals)) if len(vals) == 2 else (math.inf if a != b else 0.0)


def evaluate(reg: Registry, names=SCENARIOS, fleets=FLEETS, periods=PERIODS) -> list[dict]:
    rows = []
    for name in names:
        for fleet, label in fleets:
            for period in periods:
                at = reg.override(choices(name, fleet, period))
                r = analysis.result(at, name)
                swings = analysis.tornado(at, name)
                corner = analysis.corner(at, name, swings=swings)
                ends = [r.lo] + [v for s in swings for v in (s.at_low, s.at_high)]
                f = run.picked(r)
                rows.append({
                    "scenario": name, "fleet": fleet, "where": label, "period": period,
                    "servers": r.scenario.context.fleet.count, "period_s": r.scenario.context.horizon.seconds,
                    "slowdown": r.lo, **sensitivities(at, name), "padded": run.padded(r), "mfu": r.scenario.mfu,
                    "range": [min(ends), max(ends)], "corner": corner.rho, "seconds": r.scenario.seconds(r.lo),
                    "family": f.family if f else None, "parts": parts(r, at), "scaled": scaled(r),
                    "attribution": [{"name": t.name, "share": t.share} for t in analysis.attribution(r)],
                    "ranking": ranking(at, name, swings)})
    return rows


def _fleet_md(reg: Registry, fleets) -> str:
    out = []
    for fleet, label in fleets:
        t = reg.table(f"fleet.{fleet}")
        out.append(f"{label}, `fleet.{fleet}`: {run.fmt(t['count'])} × {t['location']} "
                   f"({reg.param(f'fleet.{fleet}.count').source})")
    return "; ".join(out)


def measured_md(reg: Registry, m: circuits.Comparison) -> list[str]:
    """A measured step beside the FLOP model (`circuits.compare`), and what `circuits.in_place` puts in its place."""
    fmt = run.fmt
    c = circuits.circuit(reg, m.circuit)
    basis = reg[f"structure.{c.structure}.basis"]
    return ["## The measured design", "",
            (f"`circuit.{m.circuit}`, {c.definition} (one step of `run.{c.run}`, {fmt(m.flops)} FLOPs at "
             f"`6 · n_active · tokens`), in place of the FLOP model: every scenario that names no structure takes "
             f"`structure.{c.structure}` (`s_×` = {fmt(c.s_x)} gates, {basis}), and `units.gates_per_flop` is the "
             f"step's gates over its FLOPs. `κ` is family {m.family}'s, and `u/w = 2G / (κ s_×)` is at the step's `G`."),
            "", "| | FLOP model | measured | measured / model |", "|---|---|---|---|",
            f"| gates a FLOP | {fmt(m.model_gates_per_flop)} | {fmt(m.gates_per_flop)} | {fmt(m.gates_ratio)} |",
            (f"| ANDs a FLOP | {fmt(m.model_gates_per_flop)} (gates, `units.gates_per_flop`) | {fmt(m.ands_per_flop)} | "
             f"{fmt(m.ands_ratio)} |"),
            (f"| gates a product `κ` counts, `2G / κ` | {fmt(m.reference_s_times)} (`κ` = {fmt(m.kappa)}) | "
             f"{fmt(m.step_s_times)} (the step's `κ` = {fmt(m.kappa_step)}) | {fmt(m.step_s_times / m.reference_s_times)} |"),
            (f"| gates (ANDs) a counted product, every one the census counts | — | {fmt(m.gates_per_product)} "
             f"({fmt(m.ands_per_product)}) | — |"),
            (f"| `s_×` | {fmt(m.reference_s_times)} (`structure.reference`, `2G / κ`) | {fmt(m.s_x)} "
             f"({fmt(m.s_x_unsplit)} outside the dot steps, which hold {m.steps_share:.1%} of the counted products) | "
             f"{fmt(m.s_x / m.reference_s_times)} |"),
            (f"| `u/w` | 1 | {fmt(m.u_over_w(m.s_x))} ({fmt(m.u_over_w(m.s_x_unsplit))} at `s_×` = "
             f"{fmt(m.s_x_unsplit)}) | — |"), ""]


def markdown(reg: Registry, rows: list[dict], fleets=FLEETS, measured: circuits.Comparison | None = None) -> str:
    fmt = run.fmt
    out = ["# The slowdown per monitoring period", "",
           ("The slowdown's lower bound at the measured H200 inputs, for each monitoring period, over "
            + _fleet_md(reg, fleets) + ". Each row is the scenario with its `fleet` and `horizon` choices set "
            "(`scenario.<name>.fleet`, `.horizon`). The range is each ranged parameter alone at either end of its "
            "plausible range, the rest at reference; the corner is all of them at their worse ends at once. The "
            "slowdown is the hardware multiple over the reference training algorithm at its MFU, MFU₀ (Lean's "
            "`hardwareMultiple`, at least 1, so no protocol gives 1); the padded size ratio `|A| / T_0` is the "
            "intermediate it comes from. The time to the target is wall-clock, at least the slowdown times "
            "`T_0 / (MFU₀ Φ_total)`."), "",
           *(measured_md(reg, measured) if measured else ()),
           "## The lower bound", "",
           ("| scenario | over | period | slowdown ≥ | range | corner | padded size ratio | status, premises | set by | "
           "what sets it | time to the target ≥ |"), "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        p = x["parts"]
        sets = ", ".join(f"{t['name']} {t['share']:.0%}" for t in x["attribution"][:2])
        beyond = " (beyond the period)" if x["seconds"] > x["period_s"] else ""
        out.append(f"| {x['scenario']} | {x['where']} | {x['period']} | **{fmt(x['slowdown'])}**"
                   f" | {fmt(x['range'][0])} to {fmt(x['range'][1])} | {fmt(x['corner'])} | {fmt(x['padded'])}"
                   f" | {p.get('status', '—')}, {p.get('premise') or '—'} | {p.get('setter') or '—'} | {sets} | "
                   f"{operating.human(x['seconds'])}{beyond} |")
    out += ["", "## Without the ruled assumptions", "",
            ("The lower bound assumes hosts carry no compute (`GLOSSARY.md`, the device model: a server's compute is "
             "its accelerators') and reference efficiency (`GLOSSARY.md`, a task assumption: no run uses the hardware "
             "more efficiently than the reference, so free compute counts at MFU₀). Sensitivities, not headlines: "
             "without the first, each regime's host CPUs count at their INT8 datasheet peak (`regime.<name>.host` = 1); "
             "without the second, free compute counts at peak (`task.reference_efficiency` = false)."), "",
            ("| scenario | over | period | slowdown ≥ | without the hosts assumption | without the reference-efficiency "
             "assumption | without both |"), "|---|---|---|---|---|---|---|"]
    out += [f"| {x['scenario']} | {x['where']} | {x['period']} | {fmt(x['slowdown'])} | {fmt(x['without_hosts'])} | "
            f"{fmt(x['without_reference_efficiency'])} | {fmt(x['without_both'])} |" for x in rows]
    out += ["", "## Its parts", ""]
    seen: dict[str, list[dict]] = {}
    for x in rows:
        key = json.dumps([x["scenario"], x["parts"], [b["name"] for b in x["scaled"]["blocks"]]], sort_keys=True)
        seen.setdefault(key, []).append(x)
    for group in seen.values():
        x, p, sc = group[0], group[0]["parts"], group[0]["scaled"]
        own = [y for y in rows if y["scenario"] == x["scenario"]]
        where = ("every fleet and period" if len(group) == len(own)
                 else "; ".join(f"{g['where']}, {g['period']}" for g in group))
        out += [f"### {x['scenario']} ({where})", "",
                (f"Set by **{p['setter']}** ({p.get('status')}; premises {p.get('premise') or '—'}), binding at "
                f"{fmt(p.get('binding_bits', math.nan))} novel bits a block."), "", "| kind | parts |", "|---|---|"]
        for kind, names in p["kinds"].items():
            cells = [f"`{n}`" + (f" ({p['standings'][n]})" if n in p["standings"] else "") for n in names]
            out.append(f"| {kind} | {', '.join(cells) or '—'} |")
        blocks = "; ".join(f"`{b['name']}`, its cuts "
                           + ", ".join(f"{c} ({', '.join(f'`{n}`' for n in cs) or '—'})" for c, cs in b["cuts"].items())
                           for b in sc["blocks"])
        counts = "; ".join(f"{g['where']}, {g['period']}: {fmt(b['count'])} of one server × "
                           f"{operating.human(b['seconds'])}" for g in group for b in g["scaled"]["blocks"])
        out += [(f"| scaled | {sc['per_window_share']:.1%} of the slowdown's rate: each serving block's local minimum is a "
                f"per-server-window bound ({blocks}), summed over the blocks ({counts}) |"),
                (f"| not scaled | the dark pool {sc['dark_share']:.1%} (`ε_dark` of the fleet, one pool), the "
                f"uncertified routes {sc['routes_share']:.1%} (shares of each server's capacity at `q_U`) |"),
                "| what sets it | " + ", ".join(f"{t['name']} {t['share']:.1%}" for t in x["attribution"]) + " |", ""]
    out += ["## The premises, ranked", "",
            ("How far each alone moves the lower bound, the rest at reference, in decades: each ranged parameter "
             "over its plausible range, each named premise of `data/premises.toml` refuted (a bound citing it then "
             "bounds no block), and `draft:no-straddle` dropped (`s_×` = 1). **Open** is a named premise or a "
             "parameter that is assumed, conjectured or derived; ruled, measured, declared and datasheet inputs "
             "are ranked beside them for scale."), ""]
    for name in dict.fromkeys(x["scenario"] for x in rows):
        own = [x for x in rows if x["scenario"] == name]
        x = next((y for y in own if y["fleet"] == fleets[-1][0] and y["period"] == "year"), own[-1])
        same = all([(_tail(r["what"]), round(r["decades"], 2)) for r in y["ranking"]]
                   == [(_tail(r["what"]), round(r["decades"], 2)) for r in x["ranking"]] for y in own)
        note = ("the same over every fleet and period here" if same else
                "the other fleets and periods in `periods.json`")
        out += [f"### {name} ({x['where']}, {x['period']}; {note})", "",
                "| rank | premise or input | kind | range | slowdown ≥ at its ends | decades |", "|---|---|---|---|---|---|"]
        for i, r in enumerate(x["ranking"], 1):
            what = f"**`{r['what']}`**" if r["open"] else f"`{r['what']}`"
            rng = " to ".join(fmt(v) if isinstance(v, float | int) else str(v) for v in r["range"])
            out.append(f"| {i} | {what} | {r['kind']} | {rng} | {fmt(r['at'][0])}, {fmt(r['at'][1])} | "
                       f"{r['decades']:.2f} |")
        out.append("")
    return "\n".join(out)


def _finite(x):
    """`x` for JSON: a non-finite float as its name (`inf`, `nan`)."""
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_finite(v) for v in x]
    return x


def _tail(key: str) -> str:
    """A key without its fleet's name, to compare rankings across fleets."""
    parts = key.split(".")
    return ".".join(parts[2:]) if parts[0] == "fleet" else key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", help=f"scenario names (default: {', '.join(SCENARIOS)})")
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE registry overrides")
    ap.add_argument("--measured", metavar="CIRCUIT",
                    help="a circuit of data/circuits.toml whose design replaces the FLOP model (circuits.in_place)")
    ap.add_argument("--out", metavar="DIR", help="write periods.md and periods.json to DIR")
    args = ap.parse_args()
    reg = Registry.load()
    if args.set:
        reg = reg.override({k: json.loads(v) for k, v in (a.split("=", 1) for a in args.set)})
    measured = circuits.compare(reg, args.measured) if args.measured else None
    if args.measured:
        reg = circuits.in_place(reg, args.measured)
    rows = evaluate(reg, tuple(args.scenario or SCENARIOS))
    text = markdown(reg, rows, measured=measured)
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        doc = {"rows": rows} | ({"measured": asdict(measured) | {"u_over_w": measured.u_over_w(measured.s_x)}}
                               if measured else {})
        (out / "periods.md").write_text(text + "\n")
        (out / "periods.json").write_text(json.dumps(_finite(doc), indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
