"""The overhead frontier: per workload, the fastest audited decode step with γ ≤ 1%, the parameters that reach it, and
which input, improved by a realistic amount, lowers it most.

    uv run python benchmarks/overhead_frontier/frontier.py --out DIR     writes DIR/frontier.json, prints the summary

Three frontiers: ``cap/C`` (γ against the node's read capacity, top-up filling idle memory, the identity statement:
the ruling's reading, and the headline), ``pass/C`` (the design's per-pass γ, for comparison with the red team) and
``cap/N`` (memory-only helpers over the NIC priced, not excluded). Each at today's inputs and at every lever improved
(``projected``). The sensitivity takes one lever at a time from today, and reverts one at a time from projected.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inputs as I  # noqa: E402
from model import GAMMA_MAX, SHAPES, Params, Workload, gamma_cap, gamma_pass, layout, link_n, pad, step  # noqa: E402

KS = (2, 3)
BLOCKS = (4096, 8192, 16384, 32768, 65536)
K_S = (8, 12, 16, 32)
BINDINGS = (0, 32, 64, 128, 256)
HOST_SHARES = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)
BATCHES = (1, 4, 16, 64, 256)
CONTEXTS = (2048, 8192, 32768, 131072)
FRONTIERS = (("cap", "C"), ("pass", "C"), ("cap", "N"))


def workloads(v: dict[str, Any]) -> list[Workload]:
    out = []
    for shape in SHAPES.values():
        for b in BATCHES:
            for c in CONTEXTS:
                w = Workload(shape, b, c)
                if layout(w, v) is not None:
                    out.append(w)
    return out


def best(w: Workload, v: dict[str, Any], definition: str, statement: str) -> dict[str, Any]:
    """The fastest point with γ ≤ 1%, or, if there is none, the point of least γ (``feasible`` false)."""
    limit = GAMMA_MAX - (v["eta_reserve"] if definition == "cap" else 0.0)
    win: tuple | None = None
    least: tuple | None = None
    for k, block, k_s, share in itertools.product(KS, BLOCKS, K_S, HOST_SHARES):
        st0 = step(w, Params(k, block, k_s, 0, share), v)
        for binding in BINDINGS:
            p = Params(k, block, k_s, binding, share)
            st = st0
            if definition == "cap":
                g = gamma_cap(w, p, v, st)
                rest = limit - (g["gamma"] - g["mu_unreg"])
                if rest > 0:
                    st = pad(st, rest, v)
                    g = gamma_cap(w, p, v, st)
            else:
                g = gamma_pass(w, p, v, st)
            link = link_n(w, p, v, st) if statement == "N" else 0.0
            gamma = g["gamma"] + link
            over = st[definition] / st["plain"] - 1
            if gamma <= limit + 1e-12 and (win is None or (over, gamma) < win[0]):
                win = ((over, gamma), p, st, g, link)
            if least is None or (gamma, over) < least[0]:
                least = ((gamma, over), p, st, g, link)
    chosen = win if win is not None else least
    _, p, st, g, link = chosen
    return _point(w, p, st, g, link, definition, feasible=win is not None, limit=limit)


def _point(w, p, st, g, link, definition, feasible, limit) -> dict[str, Any]:
    plain = st["plain"]
    terms = {n: x for n, x in g.items() if n.startswith("mu_") or n == "lambda"}
    if link:
        terms["mu_link"] = link
    out = {
        "workload": w.label, "model": w.shape.name, "batch": w.batch, "ctx": w.ctx, "tp": st["tp"],
        "f_kv_hbm": st["f_kv_hbm"], "f_kv_reads": st["f_kv_reads"], "feasible": feasible,
        "overhead": st[definition] / plain - 1, "gamma": g["gamma"] + link, "terms": terms,
        "params": {"k": p.k, "width_bits": 512, "block": p.block, "k_s": p.k_s, "binding": p.binding_label,
                   "host_share": p.host_share, "openings": g["openings"]},
        "parts": {n: st[n] / plain for n in ("t_rd", "t_tc", "t_kv", "t_act", "t_ar", "lin", "enc", "attn", "topup",
                                             "pad")},
        "gamma_limit": limit,
        "plain_ms": plain * 1e3,
    }
    if definition == "cap":
        out["eta_allowance"] = GAMMA_MAX - out["gamma"]
    else:
        out["d"] = g["d"]
        out["residual"] = g["residual"]
    return out


def frontier(v: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    ws = workloads(v)
    return {f"{d}/{s}": [best(w, v, d, s) for w in ws] for d, s in FRONTIERS}


def compare(base: list[dict], alt: list[dict]) -> dict[str, Any]:
    both = [(a, b) for a, b in zip(base, alt) if a["feasible"] and b["feasible"]]
    gains = [a["overhead"] - b["overhead"] for a, b in both]
    by_model = {}
    for name in SHAPES:
        g = [a["overhead"] - b["overhead"] for a, b in both if a["model"] == name]
        by_model[name] = statistics.mean(g) if g else None
    return {
        "median_gain": statistics.median(gains) if gains else 0.0,
        "mean_gain": statistics.mean(gains) if gains else 0.0,
        "max_gain": max(gains) if gains else 0.0,
        "newly_feasible": sum(1 for a, b in zip(base, alt) if b["feasible"] and not a["feasible"]),
        "newly_infeasible": sum(1 for a, b in zip(base, alt) if a["feasible"] and not b["feasible"]),
        "by_model": by_model,
    }


def sensitivity(inputs: dict[str, I.Input], key: str = "cap/C") -> dict[str, Any]:
    d, s = key.split("/")
    ws = workloads(I.values(inputs))

    def run(inp):
        v = I.values(inp)
        return [best(w, v, d, s) for w in ws]

    today = run(inputs)
    projected_inputs = I.all_improved(inputs)
    projected = run(projected_inputs)
    one = []
    for lever in I.levers(inputs):
        alt = run(I.improved(lever, inputs))
        one.append({"input": lever, "lever": inputs[lever].lever, "status": inputs[lever].status,
                    **compare(today, alt)})
    one.sort(key=lambda r: (-r["newly_feasible"], -r["median_gain"]))
    back = []
    for lever in I.levers(inputs):
        reverted = dict(projected_inputs)
        reverted[lever] = inputs[lever]
        alt = run(reverted)
        c = compare(alt, projected)
        back.append({"input": lever, "lever": inputs[lever].lever, **c})
    back.sort(key=lambda r: (-r["newly_feasible"], -r["median_gain"]))
    risks = []
    for name, (value, why) in I.RISKS.items():
        for label, base_inputs, base in (("today", inputs, today), ("projected", projected_inputs, projected)):
            alt_inputs = dict(base_inputs)
            alt_inputs[name] = replace(base_inputs[name], value=value)
            c = compare(run(alt_inputs), base)
            risks.append({"input": name, "value": value, "why": why, "from": label, "median_cost": c["median_gain"],
                          "mean_cost": c["mean_gain"],
                          "max_cost": c["max_gain"], "lost": c["newly_feasible"]})
    return {"key": key, "one_at_a_time_from_today": one, "revert_one_from_projected": back, "risks": risks,
            "feasible_today": sum(r["feasible"] for r in today),
            "feasible_projected": sum(r["feasible"] for r in projected), "workloads": len(ws)}


def describe(inputs: dict[str, I.Input]) -> dict[str, Any]:
    return {n: {"value": i.value, "improved": i.improved, "unit": i.unit, "source": i.source, "status": i.status,
                "lever": i.lever} for n, i in inputs.items()}


def summary(points: list[dict[str, Any]]) -> str:
    lines = [f"{'workload':22} {'tp':>2} {'fKV':>5} {'over':>7} {'γ':>6} {'k':>1} {'block':>6} {'k_s':>3} "
             f"{'binding':>12} {'host':>4} {'opens':>6}  note"]
    for r in points:
        p = r["params"]
        note = "" if r["feasible"] else "infeasible (least γ shown)"
        if "eta_allowance" in r and r["feasible"]:
            note = f"η allowance {100 * r['eta_allowance']:.2f}%"
        if r["parts"]["pad"] > 0:
            note += f" padded {100 * r['parts']['pad']:.0f}%"
        if "residual" in r:
            note += f" d {r['d']:.2f} residual {100 * r['residual']:.1f}%"
        lines.append(f"{r['workload']:22} {r['tp']:>2} {r['f_kv_hbm']:5.2f} {100 * r['overhead']:6.1f}% "
                     f"{100 * r['gamma']:5.2f}% {p['k']:>1} {p['block'] // 1024:>5}K {p['k_s']:>3} "
                     f"{p['binding']:>12} {p['host_share']:4.2f} {p['openings']:6.0f}  {note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    today = frontier(I.values(I.INPUTS))
    projected = frontier(I.values(I.all_improved(I.INPUTS)))
    sens = {key: sensitivity(I.INPUTS, key) for key in ("cap/C", "pass/C")}
    doc = {"schema": "verity/overhead-frontier/v0", "gamma_max": GAMMA_MAX, "inputs": describe(I.INPUTS),
           "grid": {"k": KS, "block": BLOCKS, "k_s": K_S, "binding": BINDINGS, "host_share": HOST_SHARES,
                    "batch": BATCHES, "ctx": CONTEXTS, "width_bits": [512]},
           "today": today, "projected": projected, "sensitivity": sens}
    (args.out / "frontier.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    for name, fr in (("today", today), ("projected", projected)):
        for key, points in fr.items():
            print(f"\n== {name} {key}")
            print(summary(points))
    for key, sv in sens.items():
        print(f"\n== sensitivity {key}: one lever at a time from today")
        for r in sv["one_at_a_time_from_today"]:
            print(f"  {r['input']:18} median −{100 * r['median_gain']:6.2f} mean −{100 * r['mean_gain']:6.2f} pts  "
                  f"+{r['newly_feasible']} feasible  {r['lever']}")
        print(f"== sensitivity {key}: revert one lever from projected")
        for r in sv["revert_one_from_projected"]:
            print(f"  {r['input']:18} median +{100 * r['median_gain']:6.2f} mean +{100 * r['mean_gain']:6.2f} pts  "
                  f"−{r['newly_feasible']} feasible")
        print(f"== risks {key} (feasible today {sv['feasible_today']}, projected {sv['feasible_projected']} "
              f"of {sv['workloads']})")
        for r in sv["risks"]:
            print(f"  {r['input']:12} = {r['value']!s:7} from {r['from']:9} median +{100 * r['median_cost']:6.2f} "
                  f"mean +{100 * r['mean_cost']:6.2f} pts  "
                  f"−{r['lost']} feasible  {r['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
