"""The overhead frontier: per workload, the fastest audited decode step with γ ≤ 1%, the parameters that reach it, and
which input, improved by a realistic amount, lowers it most.

    uv run python benchmarks/overhead_frontier/frontier.py --out DIR     writes DIR/frontier.json, prints the summary

Three frontiers: ``cap/C`` (γ against the node's read capacity, top-up filling idle memory, the identity statement:
the ruling's reading, and the headline), ``pass/C`` (the design's per-pass γ, for comparison with the red team) and
``cap/N`` (memory-only helpers over the NIC priced, not excluded). Each at today's inputs, at every lever improved
(``projected``) and at every lever at its floor (``floor``, ``inputs.FLOOR``: how fast the audit can get on this node
with this codec). Each at three rounds (``k3``, the stated guarantee), at two (``k2``, the upgrade once assumption
7 is measured) and at two masked (``k2m``, ra-codec-mix's σ_K). The sensitivity takes one lever at a time from today
(scored by the least γ where nothing is feasible), the greedy path of levers from today, and reverts one at a time
from projected; ``cap/C k3 R·T`` does the same with η counted against R·T, phase by phase. ``widths`` sweeps the word
width with the rounds at 8B and 70B, beside the partial-completion margins.
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
from model import (GAMMA_MAX, SHAPES, Params, Workload, decode, gamma_cap, gamma_pass, interp, layout,  # noqa: E402
                   link_n, margins, pad, sq_ratio, step)

KS = (2, 3)
# Every stated guarantee is at three rounds (STEERING 18:35Z); two rounds is the upgrade behind assumption 7, and two
# masked rounds (ra-codec-mix's σ_K) the upgrade behind "no completion skips the inner root".
ROUNDS = {"k3": (3,), "k2": (2,), "k2m": (2,)}
MASKED = {"k2m"}
STATED = "k3"
BLOCKS = (2048, 4096, 8192, 16384, 32768, 65536)
K_S = (8, 12, 16, 32)
BINDINGS = (0, 32, 64, 128, 256)
HOST_SHARES = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)
COPIES = (1, 2, 4, 8)
BATCHES = (1, 4, 16, 64, 256)
CONTEXTS = (2048, 8192, 32768, 131072)
FRONTIERS = (("cap", "C"), ("pass", "C"), ("cap", "N"))
ETA_PHASES = ("eta_decode", "eta_prefill", "eta_gaps", "eta_idle")
WIDTHS = (256, 320, 384, 448, 512, 576, 640)
WIDTH_WORKLOADS = (("8B", 1, 8192), ("8B", 16, 8192), ("8B", 64, 8192), ("70B", 1, 8192), ("70B", 16, 8192),
                   ("70B", 64, 8192))


def codec(v: dict[str, Any], rounds: str) -> dict[str, Any]:
    return {**v, "masked": rounds in MASKED}


def workloads(v: dict[str, Any]) -> list[Workload]:
    out = []
    for shape in SHAPES.values():
        for b in BATCHES:
            for c in CONTEXTS:
                w = Workload(shape, b, c)
                if layout(w, v) is not None:
                    out.append(w)
    return out


def best(w: Workload, v: dict[str, Any], definition: str, statement: str, ks: tuple[int, ...] = KS) -> dict[str, Any]:
    """The fastest point with γ ≤ 1% at rounds in ``ks``, or, if there is none, the point of least γ (``feasible``
    false)."""
    limit = GAMMA_MAX
    win: tuple | None = None
    least: tuple | None = None
    fit = layout(w, v)
    need = (w.shape.weight_bytes + w.batch * w.ctx * w.shape.kv_bytes_per_token) / fit
    copies = [c for c in COPIES if c <= v["copies"] and c * need <= v["H"]]
    for k, block, k_s, share, c in itertools.product(ks, BLOCKS, K_S, HOST_SHARES, copies):
        st0 = step(w, Params(k, block, k_s, 0, share, c), v)
        for binding in BINDINGS:
            p = Params(k, block, k_s, binding, share, c)
            st = st0
            if definition == "cap":
                st = pad(w, p, v, st, limit)
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
    out = _point(w, p, st, g, link, definition, feasible=win is not None, limit=limit)
    out["gamma_least"] = least[0][0]
    return out


def _point(w, p, st, g, link, definition, feasible, limit) -> dict[str, Any]:
    plain = st["plain"]
    terms = {n: x for n, x in g.items() if n.startswith("mu_") or n == "lambda"}
    if link:
        terms["mu_link"] = link
    out = {
        "workload": w.label, "model": w.shape.name, "batch": w.batch, "ctx": w.ctx, "tp": st["tp"],
        "f_kv_hbm": st["f_kv_hbm"], "f_kv_reads": st["f_kv_reads"], "feasible": feasible,
        "overhead": st[definition] / plain - 1, "gamma": g["gamma"] + link, "terms": terms,
        "params": {"k": p.k, "width_bits": st["bits"], "block": p.block, "k_s": p.k_s, "binding": p.binding_label,
                   "host_share": p.host_share, "copies": p.copies, "openings": g["openings"]},
        "parts": {n: st[n] / plain for n in ("t_rd", "t_tc", "t_kv", "t_act", "t_ar", "lin", "enc", "attn", "topup",
                                             "pad")},
        "gamma_limit": limit,
        "plain_ms": plain * 1e3,
    }
    if definition == "cap":
        out["eta_allowance"] = GAMMA_MAX - out["gamma"] + terms["mu_eta"]
    else:
        out["d"] = g["d"]
        out["residual"] = g["residual"]
    return out


def frontier(v: dict[str, Any], ks: tuple[int, ...] = KS) -> dict[str, list[dict[str, Any]]]:
    ws = workloads(v)
    return {f"{d}/{s}": [best(w, v, d, s, ks) for w in ws] for d, s in FRONTIERS}


def widths(v: dict[str, Any]) -> list[dict[str, Any]]:
    """The word width and the rounds swept together (STEERING 18:47Z), each point the whole store at one (b, k): its
    codec's cost a byte, the adversary's rebuild, the red team's partial-completion margin (k = 2's table, an
    estimate; masked rows ra-codec-mix's) and the fastest ``cap/C`` step for 8B and 70B at ctx 8k. Mixing widths
    across weights, KV and filler gains nothing: the adversary rebuilds the weakest class, and at equal strength
    k·x₁(b) the wider word decodes cheaper."""
    out = []
    for bits, rounds in itertools.product(WIDTHS, ("k2", "k3", "k2m")):
        (k,) = ROUNDS[rounds]
        vb = codec({**v, "bits": bits}, rounds)
        points = [best(Workload(SHAPES[n], b, c), vb, "cap", "C", (k,)) for n, b, c in WIDTH_WORKLOADS]
        feasible = [r["overhead"] for r in points if r["feasible"]]
        x1_adv = v["x1"] * sq_ratio(bits) / (v["s_adv"] * (1 + v["h_host_sq"]))
        out.append({
            "bits": bits, "rounds": rounds, "k": k, "masked": vb["masked"], "x1_adv": x1_adv, "strength": k * x1_adv,
            "decode_c": (1 + v["decode_floor"][k]) / k * sq_ratio(bits) * 512 / bits / v["sq_speed"],
            "o_kv": decode(interp(v["o_kv"][k], 80), k, vb), "o_w": decode(interp(v["o_w"][k], 32), k, vb),
            "mu_rebuild": points[0]["terms"]["mu_rebuild"], "mu_partial": points[0]["terms"]["mu_partial"],
            "margins_k2": margins(bits), "feasible": len(feasible),
            "median_overhead": statistics.median(feasible) if feasible else None,
            "points": [{"workload": r["workload"], "feasible": r["feasible"], "overhead": r["overhead"],
                        "gamma": r["gamma"], "params": r["params"]} for r in points],
        })
    return out


def compare(base: list[dict], alt: list[dict]) -> dict[str, Any]:
    """What ``alt`` gains over ``base``: the step overhead where both are feasible, the points it makes feasible, and
    the least γ any parameters reach (the gain that counts where neither is feasible)."""
    both = [(a, b) for a, b in zip(base, alt) if a["feasible"] and b["feasible"]]
    gains = [a["overhead"] - b["overhead"] for a, b in both]
    drops = [a["gamma_least"] - b["gamma_least"] for a, b in zip(base, alt)]
    by_model = {}
    for name in SHAPES:
        g = [a["overhead"] - b["overhead"] for a, b in both if a["model"] == name]
        by_model[name] = statistics.mean(g) if g else None
    return {
        "median_gain": statistics.median(gains) if gains else 0.0,
        "mean_gain": statistics.mean(gains) if gains else 0.0,
        "max_gain": max(gains) if gains else 0.0,
        "gamma_drop": statistics.median(drops),
        "newly_feasible": sum(1 for a, b in zip(base, alt) if b["feasible"] and not a["feasible"]),
        "newly_infeasible": sum(1 for a, b in zip(base, alt) if a["feasible"] and not b["feasible"]),
        "by_model": by_model,
    }


def _rank(r: dict[str, Any]) -> tuple:
    return (-r["newly_feasible"], -r["median_gain"], -r["gamma_drop"])


def sensitivity(inputs: dict[str, I.Input], key: str = "cap/C", rounds: str = STATED,
                path_steps: int = 0) -> dict[str, Any]:
    """One lever at a time from today, one reverted at a time from projected, each risk from both, and, with
    ``path_steps``, the greedy path from today: at each step the lever that gains most over the levers taken."""
    d, s = key.split("/")
    ws = workloads(I.values(inputs))

    def run(inp):
        v = codec(I.values(inp), rounds)
        return [best(w, v, d, s, ROUNDS[rounds]) for w in ws]

    today = run(inputs)
    projected_inputs = I.all_improved(inputs)
    projected = run(projected_inputs)
    one = []
    for lever in I.levers(inputs):
        alt = run(I.improved(lever, inputs))
        one.append({"input": lever, "lever": inputs[lever].lever, "status": inputs[lever].status,
                    **compare(today, alt)})
    one.sort(key=_rank)
    path = []
    taken, base_inputs, base = [], inputs, today
    for _ in range(path_steps):
        tries = []
        for lever in I.levers(inputs):
            if lever in taken:
                continue
            alt_inputs = I.improved(lever, base_inputs)
            alt = run(alt_inputs)
            tries.append((_rank(compare(base, alt)), lever, alt_inputs, alt))
        if not tries:
            break
        _, lever, base_inputs, alt = min(tries, key=lambda t: t[0])
        c = compare(base, alt)
        taken.append(lever)
        base = alt
        feasible = [r["overhead"] for r in base if r["feasible"]]
        path.append({"input": lever, "lever": inputs[lever].lever, **c,
                     "feasible": len(feasible), "median_overhead": statistics.median(feasible) if feasible else None,
                     "median_gamma_least": statistics.median(r["gamma_least"] for r in base)})
        if len(feasible) == len(base) and c["median_gain"] <= 0.0:
            break
    back = []
    for lever in I.levers(inputs):
        reverted = dict(projected_inputs)
        reverted[lever] = inputs[lever]
        alt = run(reverted)
        c = compare(alt, projected)
        back.append({"input": lever, "lever": inputs[lever].lever, **c})
    back.sort(key=_rank)
    risks = []
    for name, (value, why) in I.RISKS.items():
        for label, base_inputs, base in (("today", inputs, today), ("projected", projected_inputs, projected)):
            alt_inputs = dict(base_inputs)
            alt_inputs[name] = replace(base_inputs[name], value=value)
            c = compare(run(alt_inputs), base)
            risks.append({"input": name, "value": value, "why": why, "from": label, "median_cost": c["median_gain"],
                          "mean_cost": c["mean_gain"], "max_cost": c["max_gain"], "lost": c["newly_feasible"],
                          "gamma_rise": c["gamma_drop"]})
    return {"key": key, "rounds": rounds, "one_at_a_time_from_today": one, "path_from_today": path,
            "revert_one_from_projected": back, "risks": risks,
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
        if p["copies"] > 1:
            note += f" {p['copies']} copies"
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
    scenarios = {"today": I.INPUTS, "projected": I.all_improved(I.INPUTS), "floor": I.at_floor(I.INPUTS)}
    frontiers = {name: {r: frontier(codec(I.values(inp), r), ks) for r, ks in ROUNDS.items()}
                 for name, inp in scenarios.items()}
    width_sweep = {name: widths(I.values(inp)) for name, inp in scenarios.items()}
    rt = {**I.INPUTS, "eta_counted": replace(I.INPUTS["eta_counted"], value=True)}
    sens = {label: sensitivity(inp, key, r, steps)
            for label, key, r, steps, inp in (("cap/C k3", "cap/C", "k3", 8, I.INPUTS),
                                              ("cap/C k2", "cap/C", "k2", 0, I.INPUTS),
                                              ("cap/C k2m", "cap/C", "k2m", 0, I.INPUTS),
                                              ("pass/C k3", "pass/C", "k3", 0, I.INPUTS),
                                              ("cap/C k3 R·T", "cap/C", "k3", 0, rt))}
    phases = {name: {n: I.values(inp)[n] for n in ETA_PHASES} for name, inp in scenarios.items()}
    allowance = {name: [r["eta_allowance"] for r in frontiers[name][STATED]["cap/C"] if r["feasible"]]
                 for name in scenarios}
    doc = {"schema": "verity/overhead-frontier/v1", "gamma_max": GAMMA_MAX, "stated_rounds": STATED,
           "inputs": describe(I.INPUTS),
           "floor_inputs": {n: {"value": v, "why": why} for n, (v, why) in I.FLOOR.items()},
           "grid": {"rounds": ROUNDS, "block": BLOCKS, "k_s": K_S, "binding": BINDINGS, "host_share": HOST_SHARES,
                    "copies": COPIES, "batch": BATCHES, "ctx": CONTEXTS, "width_bits": [512], "widths": WIDTHS,
                    "width_workloads": WIDTH_WORKLOADS},
           "frontiers": frontiers, "widths": width_sweep, "sensitivity": sens,
           "eta": {"phases": phases, "allowance_cap_C_k3": allowance}}
    (args.out / "frontier.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"\n== η by phase (of R·T) beside what the {STATED} cap/C frontier leaves for it")
    for name in scenarios:
        a = allowance[name]
        left = (f"allowance median {100 * statistics.median(a):.2f}%, least {100 * min(a):.2f}% over {len(a)} feasible"
                if a else "no feasible point")
        print(f"  {name:9} " + " ".join(f"{n[4:]} {100 * x:.1f}%" for n, x in phases[name].items())
              + f"  Σ {100 * sum(phases[name].values()):.1f}%  {left}")
    for name, rows in width_sweep.items():
        print(f"\n== widths {name} (cap/C; 8B and 70B at B1/B16/B64 ctx 8k; margins are k = 2's, estimates; k2m "
              f"ra-codec-mix's ≥ 1.5 at the floor)")
        print(f"  {'bits':>4} {'codec':>5} {'x1adv':>6} {'k·x1':>6} {'c':>5} {'o_kv':>5} {'o_w':>5} {'μ_reb':>6} "
              f"{'μ_part':>6} {'fplll':>6} {'floor':>6} {'low':>5} {'ok':>2} {'median':>7}  overheads")
        for r in rows:
            m = r["margins_k2"]
            med = "—" if r["median_overhead"] is None else f"{100 * r['median_overhead']:6.1f}%"
            each = " ".join("—" if not q["feasible"] else f"{100 * q['overhead']:.0f}" for q in r["points"])
            margin = ("      —   1.5×  1.0×" if r["masked"] else
                      f"{m['fplll']:5.1f}× {m['floor_centre']:5.2f}× {m['floor_low']:4.2f}×")
            print(f"  {r['bits']:>4} {r['rounds']:>5} {r['x1_adv']:6.1f} {r['strength']:6.0f} {r['decode_c']:5.3f} "
                  f"{r['o_kv']:5.3f} {r['o_w']:5.3f} {100 * r['mu_rebuild']:5.3f}% {100 * r['mu_partial']:5.3f}% "
                  f"{margin} {r['feasible']:>2} {med:>7}  {each}")
    for name, by_rounds in frontiers.items():
        for r, fr in by_rounds.items():
            for key, points in fr.items():
                print(f"\n== {name} {r} {key}")
                print(summary(points))
    for key, sv in sens.items():
        print(f"\n== sensitivity {key}: one lever at a time from today")
        for r in sv["one_at_a_time_from_today"]:
            print(f"  {r['input']:18} median −{100 * r['median_gain']:6.2f} mean −{100 * r['mean_gain']:6.2f} pts  "
                  f"+{r['newly_feasible']} feasible  least γ −{100 * r['gamma_drop']:.3f}  {r['lever']}")
        if sv["path_from_today"]:
            print(f"== sensitivity {key}: the greedy path from today")
        for i, r in enumerate(sv["path_from_today"], 1):
            med = "—" if r["median_overhead"] is None else f"{100 * r['median_overhead']:.1f}%"
            print(f"  {i}. {r['input']:18} +{r['newly_feasible']} feasible → {r['feasible']}, median overhead {med}, "
                  f"median least γ {100 * r['median_gamma_least']:.3f}%")
        print(f"== sensitivity {key}: revert one lever from projected")
        for r in sv["revert_one_from_projected"]:
            print(f"  {r['input']:18} median +{100 * r['median_gain']:6.2f} mean +{100 * r['mean_gain']:6.2f} pts  "
                  f"−{r['newly_feasible']} feasible  least γ +{100 * r['gamma_drop']:.3f}")
        print(f"== risks {key} (feasible today {sv['feasible_today']}, projected {sv['feasible_projected']} "
              f"of {sv['workloads']})")
        for r in sv["risks"]:
            print(f"  {r['input']:12} = {r['value']!s:7} from {r['from']:9} median +{100 * r['median_cost']:6.2f} "
                  f"mean +{100 * r['mean_cost']:6.2f} pts  "
                  f"−{r['lost']} feasible  least γ +{100 * r['gamma_rise']:.3f}  {r['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
