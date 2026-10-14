"""The overhead frontier: per workload, the fastest audited decode step with γ ≤ 1%, the parameters that reach it, and
which input, improved by a realistic amount, lowers it most.

    uv run python benchmarks/overhead_frontier/frontier.py --out DIR [--receipt sketch-sha512|block-sha512]
                                                                    writes DIR/frontier.json, prints the summary

Two frontiers under Conservation over HBM's R·T (``C``): ``cap/C`` (γ against the capacity over time, top-up filling
idle memory: the ruling's reading, and the headline) and ``pass/C`` (the design's per-pass γ, for comparison with the
red team). The codec is the stated one, masked p1024 at three rounds (``k3m``), beside its fallback at two (``k2m``);
a step past ``BUDGET`` (4× the plain step) is over Daniel's budget for now. Scenarios: today's inputs with η as
measured (no point can reach γ ≤ 1% against R·T until the fill lanes close it), today's and every lever improved
(``projected``) with η at a budget of 0, 0.25% or 0.5% of R·T, and every lever at its floor (``floor``,
``inputs.FLOOR``: how fast the audit can get on this node with this codec). The sensitivity takes one lever at a
time from today at η = 0, the greedy path of levers from there, and reverts one at a time from projected;
``cap/C k3m R·T`` does the same from today's measured η, phase by phase (scored by the least γ, since nothing is
feasible). ``widths`` sweeps the word width with the rounds, plain and masked, at 8B and 70B.
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
from model import (GAMMA_MAX, SHAPES, Params, Workload, at_width, decode, gamma_cap, gamma_pass, interp,  # noqa: E402
                   layout, margins, pad, step)

KS = (2, 3)
# The stated codec is masked p1024 at three rounds, and two masked rounds its fallback if three exceed about 4×
# (Daniel 20:55Z); plain rounds and 512 bits are swept in ``widths`` only.
ROUNDS = {"k3m": (3,), "k2m": (2,)}
WIDTH_CODECS = {"k3": (3,), "k2": (2,), "k3m": (3,), "k2m": (2,)}
MASKED = {"k2m", "k3m"}
STATED = "k3m"
BUDGET = 3.0
BLOCKS = (2048, 4096, 8192, 16384, 32768, 65536)
K_S = (16,)
BINDINGS = (0, 32, 64, 128, 256)
COPIES = (1, 2, 4, 8)
BATCHES = (1, 4, 16, 64, 256)
CONTEXTS = (2048, 8192, 32768, 131072)
FRONTIERS = ("cap", "pass")
RECEIPTS = ("sketch-sha512", "block-sha512")
ETA_PHASES = ("eta_decode", "eta_prefill", "eta_gaps", "eta_idle")
ETA_BUDGETS = (0.0, 0.0025, 0.005)
WIDTHS = (512, 768, 1024)
WIDTH_WORKLOADS = (("8B", 1, 8192), ("8B", 16, 8192), ("8B", 64, 8192), ("70B", 1, 8192), ("70B", 16, 8192),
                   ("70B", 64, 8192))


def codec(v: dict[str, Any], rounds: str) -> dict[str, Any]:
    return {**v, "masked": rounds in MASKED}


def with_eta(inputs: dict[str, I.Input], total: float) -> dict[str, I.Input]:
    """The inputs with η's phases summing to ``total`` of R·T, none of them a lever."""
    out = dict(inputs)
    for n in ETA_PHASES:
        out[n] = replace(inputs[n], value=total if n == "eta_decode" else 0.0, improved=None)
    return out


def workloads(v: dict[str, Any]) -> list[Workload]:
    out = []
    for shape in SHAPES.values():
        for b in BATCHES:
            for c in CONTEXTS:
                w = Workload(shape, b, c)
                if layout(w, v) is not None:
                    out.append(w)
    return out


def best(w: Workload, v: dict[str, Any], definition: str, ks: tuple[int, ...] = KS) -> dict[str, Any]:
    """The fastest point with γ ≤ 1% and every window's sampling within ``window_max`` at rounds in ``ks``, or, if
    there is none, the point of least γ (``feasible`` false)."""
    limit = GAMMA_MAX
    win: tuple | None = None
    least: tuple | None = None
    fit = layout(w, v)
    need = (w.shape.weight_bytes + w.batch * w.ctx * w.shape.kv_bytes_per_token) / fit
    copies = [c for c in COPIES if c <= v["copies"] and c * need <= v["H"]]
    for k, block, k_s, c in itertools.product(ks, BLOCKS, K_S, copies):
        st0 = step(w, Params(k, block, k_s, 0, c), v)
        for binding in BINDINGS if v["bound"] else (0,):
            p = Params(k, block, k_s, binding, c)
            st = step(w, p, v) if binding else st0
            if definition == "cap":
                st = pad(w, p, v, st, limit)
                g = gamma_cap(w, p, v, st)
            else:
                g = gamma_pass(w, p, v, st)
            gamma = g["gamma"]
            over = st[definition] / st["plain"] - 1
            ok = gamma <= limit + 1e-12 and g["window_sample"] <= v["window_max"] + 1e-12
            if ok and (win is None or (over, gamma) < win[0]):
                win = ((over, gamma), p, st, g)
            if least is None or (gamma, over) < least[0]:
                least = ((gamma, over), p, st, g)
    chosen = win if win is not None else least
    _, p, st, g = chosen
    out = _point(w, p, st, g, v, definition, feasible=win is not None, limit=limit)
    out["gamma_least"] = least[0][0]
    return out


def _point(w, p, st, g, v, definition, feasible, limit) -> dict[str, Any]:
    plain = st["plain"]
    terms = {n: x for n, x in g.items() if n.startswith("mu_") or n == "lambda"}
    out = {
        "workload": w.label, "model": w.shape.name, "batch": w.batch, "ctx": w.ctx, "tp": st["tp"],
        "f_kv_hbm": st["f_kv_hbm"], "f_kv_reads": st["f_kv_reads"], "feasible": feasible,
        "overhead": st[definition] / plain - 1, "gamma": g["gamma"], "terms": terms,
        "within_budget": st[definition] / plain - 1 <= BUDGET,
        "params": {"k": p.k, "width_bits": st["bits"], "masked": v["masked"], "block": p.block, "k_s": p.k_s,
                   "binding": p.binding_label if v["bound"] else "unbound", "copies": p.copies,
                   "openings": g["openings"],
                   "verifier_mb_s": g["openings"] * (p.block + v["opening_bytes"]) / v["t_window"] / 1e6},
        "parts": {n: st[n] / plain for n in ("t_rd", "t_tc", "t_kv", "t_act", "t_ar", "lin", "enc", "attn", "topup",
                                             "unfilled", "pad")},
        "gamma_limit": limit,
        "plain_ms": plain * 1e3,
    }
    if definition == "cap":
        out["eta_allowance"] = GAMMA_MAX - out["gamma"] + terms["mu_eta"]
    else:
        per_d = (terms["mu_rebuild"] + terms["mu_partial"]) / g["d"]
        out["d"] = g["d"]
        out["d_cap"] = (GAMMA_MAX - (out["gamma"] - g["d"] * per_d)) / per_d
        out["residual"] = g["residual"]
    return out


def frontier(v: dict[str, Any], ks: tuple[int, ...] = KS) -> dict[str, list[dict[str, Any]]]:
    ws = workloads(v)
    return {f"{d}/C": [best(w, v, d, ks) for w in ws] for d in FRONTIERS}


def widths(v: dict[str, Any]) -> list[dict[str, Any]]:
    """The word width and the rounds swept together, plain and masked, each point the whole store at one (b, k): its
    codec's cost a byte, the adversary's rebuild, the red team's partial-completion margin (k = 2's table, an
    estimate; masked rows ra-codec-mix's) and the fastest ``cap/C`` step for 8B and 70B at ctx 8k. Mixing widths
    across weights, KV and filler gains nothing: the adversary rebuilds the weakest class."""
    out = []
    for bits, rounds in itertools.product(WIDTHS, WIDTH_CODECS):
        (k,) = WIDTH_CODECS[rounds]
        vb = codec({**v, "bits": bits}, rounds)
        points = [best(Workload(SHAPES[n], b, c), vb, "cap", (k,)) for n, b, c in WIDTH_WORKLOADS]
        feasible = [r["overhead"] for r in points if r["feasible"]]
        x1_adv = at_width(vb, "x1") / (at_width(vb, "s_adv") * (1 + at_width(vb, "h_host_sq")))
        out.append({
            "bits": bits, "rounds": rounds, "k": k, "masked": vb["masked"], "x1_adv": x1_adv, "strength": k * x1_adv,
            "decode_c": (1 + v["decode_floor"][k]) / k * at_width(vb, "dec_ratio") / at_width(vb, "sq_speed"),
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
    d = key.split("/")[0]
    ws = workloads(I.values(inputs))

    def run(inp):
        v = codec(I.values(inp), rounds)
        return [best(w, v, d, WIDTH_CODECS[rounds]) for w in ws]

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
        if len(feasible) == len(base) and c["newly_feasible"] == 0 and c["median_gain"] <= 0.0:
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
    for name, value, why in ((n, value, why) for n, each in I.RISKS.items() for value, why in each):
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
             f"{'opens':>6} {'MB/s':>5}  note"]
    for r in points:
        p = r["params"]
        note = "" if r["feasible"] else "infeasible (least γ shown)"
        if "eta_allowance" in r and r["feasible"]:
            note = f"η allowance {100 * r['eta_allowance']:.2f}%"
        if not r["within_budget"]:
            note += " over 4×"
        if r["parts"]["pad"] > 0:
            note += f" padded {100 * r['parts']['pad']:.0f}%"
        if p["copies"] > 1:
            note += f" {p['copies']} copies"
        if "residual" in r:
            note += f" d {r['d']:.2f} (cap {r['d_cap']:.2f}) residual {100 * r['residual']:.1f}%"
        lines.append(f"{r['workload']:22} {r['tp']:>2} {r['f_kv_hbm']:5.2f} {100 * r['overhead']:6.1f}% "
                     f"{100 * r['gamma']:5.2f}% {p['k']:>1} {p['block'] // 1024:>5}K {p['k_s']:>3} "
                     f"{p['openings']:6.0f} {p['verifier_mb_s']:5.0f}  {note}")
    return "\n".join(lines)


def counts(points: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in points if r["feasible"]]
    within = [r for r in ok if r["within_budget"]]
    return {"workloads": len(points), "feasible": len(ok), "within_budget": len(within),
            "median_overhead": statistics.median(r["overhead"] for r in ok) if ok else None,
            "median_gamma_least": statistics.median(r["gamma_least"] for r in points)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--receipt", choices=RECEIPTS, default=I.INPUTS["receipt"].value)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = {**I.INPUTS, "receipt": replace(I.INPUTS["receipt"], value=args.receipt)}
    projected = I.all_improved(inputs)
    scenarios = {"today": inputs}
    for e in ETA_BUDGETS:
        scenarios[f"today η={100 * e:g}%"] = with_eta(inputs, e)
    for e in ETA_BUDGETS:
        scenarios[f"projected η={100 * e:g}%"] = with_eta(projected, e)
    scenarios["floor"] = I.at_floor(inputs)
    frontiers = {name: {r: frontier(codec(I.values(inp), r), ks) for r, ks in ROUNDS.items()}
                 for name, inp in scenarios.items()}
    width_sweep = {name: widths(I.values(scenarios[name])) for name in ("today η=0%", "projected η=0%", "floor")}
    base = with_eta(inputs, 0.0)
    sens = {label: sensitivity(inp, key, r, steps)
            for label, key, r, steps, inp in (("cap/C k3m η=0", "cap/C", "k3m", 8, base),
                                              ("cap/C k2m η=0", "cap/C", "k2m", 0, base),
                                              ("pass/C k3m η=0", "pass/C", "k3m", 0, base),
                                              ("cap/C k3m R·T", "cap/C", "k3m", 0, inputs))}
    phases = {name: {n: I.values(inp)[n] for n in ETA_PHASES} for name, inp in scenarios.items()}
    tally = {name: {r: {key: counts(points) for key, points in fr.items()} for r, fr in by_rounds.items()}
             for name, by_rounds in frontiers.items()}
    allowance = {name: [r["eta_allowance"] - (sum(phases[name].values()) if I.values(scenarios[name])["eta_counted"]
                                              else 0.0)
                        for r in frontiers[name][STATED]["cap/C"] if r["feasible"]]
                 for name in scenarios}
    doc = {"schema": "verity/overhead-frontier/v2", "gamma_max": GAMMA_MAX, "stated_rounds": STATED,
           "budget_overhead": BUDGET, "claim": "Conservation over HBM's R·T, the hybrid's openings",
           "receipt": args.receipt, "inputs": describe(inputs),
           "floor_inputs": {n: {"value": v, "why": why} for n, (v, why) in I.FLOOR.items()},
           "grid": {"rounds": ROUNDS, "block": BLOCKS, "k_s": K_S, "binding": BINDINGS, "copies": COPIES,
                    "batch": BATCHES, "ctx": CONTEXTS, "width_bits": [I.values()["bits"]], "widths": WIDTHS,
                    "width_workloads": WIDTH_WORKLOADS, "eta_budgets": ETA_BUDGETS},
           "frontiers": frontiers, "counts": tally, "widths": width_sweep, "sensitivity": sens,
           "eta": {"phases": phases, "headroom_cap_C_stated": allowance}}
    (args.out / "frontier.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"\n== counts (feasible at γ ≤ 1%, within the 4× budget, median overhead of the feasible; "
          f"{len(frontiers['today'][STATED]['cap/C'])} workloads)")
    for name, by_rounds in tally.items():
        for r, by_key in by_rounds.items():
            for key, c in by_key.items():
                med = "—" if c["median_overhead"] is None else f"{100 * c['median_overhead']:6.1f}%"
                print(f"  {name:20} {r:4} {key:7} feasible {c['feasible']:>2}, within 4× {c['within_budget']:>2}, "
                      f"median {med}, median least γ {100 * c['median_gamma_least']:.3f}%")
    print(f"\n== η by phase (of R·T) and the headroom the {STATED} cap/C frontier leaves beyond it")
    for name in scenarios:
        a = allowance[name]
        left = (f"headroom median {100 * statistics.median(a):.2f}%, least {100 * min(a):.2f}% over {len(a)} feasible"
                if a else "no feasible point")
        print(f"  {name:20} " + " ".join(f"{n[4:]} {100 * x:.2f}%" for n, x in phases[name].items())
              + f"  Σ {100 * sum(phases[name].values()):.2f}%  {left}")
    for name, rows in width_sweep.items():
        print(f"\n== widths {name} (cap/C; 8B and 70B at B1/B16/B64 ctx 8k; plain margins are k = 2's, estimates; "
              f"masked ra-codec-mix's ≥ 1.5 at the floor)")
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
