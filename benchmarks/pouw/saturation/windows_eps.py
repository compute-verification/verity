#!/usr/bin/env python3
"""PoUW windows restricted to the served FP8 GEMMs (CMP-8 round 2, section 2 of the saturation follow-up), derived.

  python windows_eps.py receipt-table.json gap-filler.json cutlass-receipt.json cutlass-corun.json [--cheat C ...]
      [--more T ...] [--json OUT]

--more adds the designs a later run measured (`cutlass_receipt_table.py --json` over a cheat run, such as CR_FORCED's)
that the first table lacks, at their own run's rate kept; its cheater variants are left out.

Inputs: `cutlass_receipt_table.py --json` over a cgemm run (rate kept per kernel and phase), `gap_filler.py`'s JSON,
that cgemm run's own `cutlass-receipt.json` (the receipt-free kernel's time per linear, for the wave tails), and a corun
run's `cutlass-corun.json` (the idle SM-time a low-priority pump finds beside each design's kernel), and optionally
`cheat_slack.py --json` over cheat runs (ν_f measured: the most free capacity a cheater's own CUDA work fused into each
design's kernel found, over every --cheat given; without one, ν_f is read off section 1's points at no rate loss). Round 1's
numbers (`note:20261015T0417Z-draft-saturation-budget`) are constants below: the step split (`r20261015-042637-c9be`,
FP8 graphs) and the GEMMs' η.

The window bound is the page's slack, ε_T ≤ f/(1−3δ) + κ(f+ν)h/L with κ = max(16, L/3), with ν written per unit of the
window's CUDA capacity (ν_w, the share a cheater can use for unclaimed work) instead of per unit of the honest CUDA work:
ν = ν_w L / (h ρ η_w), so κνh/L = κ ν_w / (ρ η_w), ρ = 8 (FP8 MACs per CUDA slot), η_w = the GEMM's η × rate kept. SMs
that hold no CTA are tensor room as well: as served, a cheater's own GEMM in the measured idle SM-time adds
0.9 ν_idle / η_w (0.9: the best FP8 GEMM's share of peak, Pearl-C); with a tail filler that owns those SMs, what is left
is the CUDA slack ν_f. δ is taken as 0. "Within the timing tolerance" takes ν_f over the cheaters within cheat_slack's
--tolerance of the design's time only; a slower cheater trades tensor rate for CUDA work, which pays only above the
trading threshold.
"""
from __future__ import annotations

import argparse
import json
import math
import re

F = 0.001
RHO = 8
BEST_ETA = 0.9
H_PHI = {0: 6.0, 1: 4.8}
H_TREE = 0.4
SPLIT = {  # ms per step: all kernels, the FP8 linears, the BF16 lm_head (r20261015-042637-c9be)
    "p2048": (57.29, 44.00, 0.0),
    "p8192": (273.42, 161.23, 0.0),
    "d32": (10.52, 5.245, 0.756),
}
ETA_GEMM = {"p2048": 0.82, "p8192": 0.89, "d32": 0.107}
LINEAR_MACS, LM_HEAD_MACS, LAYERS, H = 6.979e9, 525.3e6, 32, 4096
PHASES = {"p2048": (2048, 1, 1024.5), "p8192": (8192, 1, 4096.5), "d32": (32, 32, 1036.0)}  # tokens, lm_head rows, ctx
TILE = {"p": (128, 128), "d": (32, 64)}
DESIGNS = [  # name, prefill kernel, decode kernel
    ("1× imad, L = 128, spread", "p_imad_p1s", "d_alu_p1"),
    ("1× alu, L = 64", "p_alu_g2", None),
    ("1× alu, L = 32", "p_alu_g4", "d_alu_g4"),
    ("1× alu, L = 32, 1 forced step", "p_alu_g4_f1", None),
    ("1× alu, L = 32, 64 forced FFMAs", "p_alu_g4_ff64", None),
    ("1× alu, L = 32, 128 forced FFMAs", "p_alu_g4_ff128", None),
    ("1× alu, L = 32, 192 forced FFMAs", "p_alu_g4_ff192", None),
    ("1× alu, L = 32, 256 forced FFMAs", "p_alu_g4_ff256", None),
    ("1× alu, L = 32, 192 forced FFMAs, H = 2", "p_alu_g4_h2_ff192", None),
    ("1× alu, L = 32, 192 forced FFMAs, H = 4", "p_alu_g4_h4_ff192", None),
    ("1× alu, L = 32, 192 forced FFMAs, H = 8", "p_alu_g4_h8_ff192", None),
    ("1× alu, L = 32, 224 forced FFMAs, H = 4", "p_alu_g4_h4_ff224", None),
    ("1× alu, L = 32, 256 forced FFMAs, H = 4", "p_alu_g4_h4_ff256", None),
    ("3× alu, L = 128, spread (crossover)", "p_alu3_p1s", "d_alu16_p1"),
]
SCENARIOS = [  # name, measured idle SM-time as served?, ν_f (CUDA slack a fused cheater uses; None: measured)
    ("as served (measured idle and ν_f)", True, None),
    ("tails filled, measured ν_f", False, None),
    ("tails filled, ν_f within the timing tolerance", False, "tolerance"),
    ("tails filled, ν_f 0.01", False, 0.01),
    ("tails filled, ν_f 0", False, 0.0),
]
MEASURED_CREDIT = 2  # round 1 #10: at most ~1 MAC a CUDA slot through FP8, 2.1 through INT8


def coverage(ph):
    T, heads, ctx = PHASES[ph]
    lin, lm, att = T * LINEAR_MACS, heads * LM_HEAD_MACS, LAYERS * 2 * H * ctx * T
    return {"linear_macs": lin, "lm_head_macs": lm, "attention_macs": att, "macs": lin / (lin + lm + att),
            "macs_with_lm_head": (lin + lm) / (lin + lm + att),
            "cost_weighted": lin / (lin + 2 * (lm + att)), "cost_weighted_with_lm_head": (lin + 2 * lm) / (lin + 2 * (lm + att))}


def tails(cr, sms):
    """Idle SM-time share inside each phase's four linears, time-weighted (one CTA an SM, equal waves)."""
    out = {}
    for r in cr["rows"]:
        if not r["kernel"].endswith("_plain") or r["shape"].startswith("cube"):
            continue
        ph = r["shape"].split("_")[0]
        tm, tn = TILE[r["kernel"][0]]
        ctas = (r["m"] // tm) * (r["n"] // tn)
        waves = ctas / sms
        idle = (math.ceil(waves) - waves) / math.ceil(waves)
        acc = out.setdefault(ph, [0.0, 0.0])
        acc[0] += r["ms"]
        acc[1] += r["ms"] * idle
    return {ph: b / a for ph, (a, b) in out.items()}


def measured_idle(co):
    """Per (kernel, phase): the corun pump's free capacity beside the kernel, weighted by time per launch, and the
    equal-wave tail of the same launches. One served CTA fills an SM's registers, so the pump only finds SMs that hold
    no CTA: the free capacity is idle SM-time, tensor room as much as CUDA room."""
    acc = {}
    for r in co["rows"]:
        ph = r["shape"].split("_")[0]
        t = r["ms_alone"] / r["launches"]
        waves = r["ctas"] / co["sms"]
        tail = (math.ceil(waves) - waves) / math.ceil(waves)
        a = acc.setdefault((r["kernel"], ph), [0.0, 0.0, 0.0])
        a[0] += t
        a[1] += t * r["free_capacity"]
        a[2] += t * tail
    return {k: {"idle": b / a, "tail": c / a, "residue": (b - c) / a} for k, (a, b, c) in acc.items()}


def frontier_slack(tab, k, ph):
    """CUDA capacity a fused cheater adds beside kernel k without losing tensor rate, per unit of the window's CUDA
    capacity: the largest φ-share gain over k among the phase's measured kernels that keep at least k's rate. A
    cheater that trades tensor rate for CUDA work instead is the κ question (see the note), not this term."""
    pts = [(p["phi_share"], p["rate_kept"]) for p in tab["phases"] if p["phase"] == ph]
    p0, r0 = next((p["phi_share"], p["rate_kept"]) for p in tab["phases"] if p["kernel"] == k and p["phase"] == ph)
    return max([0.0] + [p1 - p0 for p1, r1 in pts if r1 >= r0])


def trade_threshold(tab, k, ph):
    """The MAC credit per CUDA slot above which trading tensor rate for CUDA work pays a cheater beside kernel k: moving
    to a measured point (p1, r1) with more φ and less rate fakes K·128(p1 − p0) MACs an SM-clock and loses
    1,024·η(r0 − r1) real ones, so it pays when K > ρη(r0 − r1)/(p1 − p0). φ alone (p 1, r 0) is a point."""
    pts = [(p["phi_share"], p["rate_kept"]) for p in tab["phases"] if p["phase"] == ph] + [(1.0, 0.0)]
    p0, r0 = next((p["phi_share"], p["rate_kept"]) for p in tab["phases"] if p["kernel"] == k and p["phase"] == ph)
    return min(RHO * ETA_GEMM[ph] * (r0 - r1) / (p1 - p0) for p1, r1 in pts if p1 > p0 and r1 < r0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("gap")
    ap.add_argument("cutlass")
    ap.add_argument("corun")
    ap.add_argument("--cheat", action="append", default=[])
    ap.add_argument("--more", action="append", default=[])
    ap.add_argument("--json")
    a = ap.parse_args()
    idle = measured_idle(json.load(open(a.corun)))
    cheat, cheat_tol = None, {}
    for c in a.cheat:
        cj = json.load(open(c))
        for key_, v in cj["nu_f"].items():
            cheat = cheat or {}
            cheat[key_] = max(cheat.get(key_, 0.0), v)
        for key_, v in cj.get("nu_f_tolerance", {}).items():
            cheat_tol[key_] = max(cheat_tol.get(key_, 0.0), v)
    tab = json.load(open(a.table))
    for t in map(lambda p: json.load(open(p)), a.more):
        have = {r["kernel"] for r in tab["rows"]}
        new = lambda k: k not in have and not re.search(r"_x[ifmga]?\d+$", k)
        tab["rows"] += [r for r in t["rows"] if new(r["kernel"])]
        tab["phases"] += [p for p in t["phases"] if new(p["kernel"])]
    kept = {(p["kernel"], p["phase"]): p["rate_kept"] for p in tab["phases"]}
    meta = {r["kernel"]: r for r in tab["rows"]}
    cr = json.load(open(a.cutlass))
    nu_tail = tails(cr, cr["sms"])
    gap = json.load(open(a.gap))
    res = {"coverage": {}, "tails": nu_tail, "designs": [], "gap": []}

    print("| phase | FP8 linears' share of tensor MACs | with lm_head | cost-weighted (BF16 2×) | linears' share of step "
          "kernel time | with lm_head | idle SM-time in the linears (wave tails) |")
    print("|---|---|---|---|---|---|---|")
    for ph in PHASES:
        c = coverage(ph)
        step, g, lm = SPLIT[ph]
        c.update({"time_share": g / step, "time_share_with_lm_head": (g + lm) / step})
        res["coverage"][ph] = c
        print(f"| {ph} | {100 * c['macs']:.1f}% | {100 * c['macs_with_lm_head']:.1f}% | {100 * c['cost_weighted']:.1f}% | "
              f"{100 * c['time_share']:.1f}% | {100 * c['time_share_with_lm_head']:.1f}% | {100 * nu_tail[ph]:.1f}% |")

    print("\n| kernel | phase | idle SM-time beside it (corun) | equal-wave tail | residue |")
    print("|---|---|---|---|---|")
    for (k, ph), v in sorted(idle.items()):
        print(f"| {k} | {ph} | {100 * v['idle']:.1f}% | {100 * v['tail']:.1f}% | {100 * v['residue']:.1f}% |")
    res["measured_idle"] = {f"{k}@{ph}": v for (k, ph), v in idle.items()}

    print("\n| design | phase | rate kept | honest overhead (step) | κ | η_w | ×ν | f terms | idle (corun) | ν_f | "
          "trading pays above (MACs a slot) | " + " | ".join(f"ε_in: {s[0]}" for s in SCENARIOS)
          + " | page's reading (ν_w = 1 − φ share) | " + " | ".join(f"ε end to end: {s[0]}" for s in SCENARIOS)
          + f" | ε_in: tails filled, measured ν_f, κ = max({MEASURED_CREDIT}, L/3) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|" + "---|" * (2 * len(SCENARIOS) + 2))
    for name, pk, dk in DESIGNS:
        for ph in PHASES:
            k = dk if ph == "d32" else pk
            if k is None or (k, ph) not in kept:
                continue
            m = meta[k] if k in meta else next(r for r in tab["rows"] if r["kernel"] == k)
            L, rounds, var = m["L"], m["rounds"], m["var"]
            kp = kept[(k, ph)]
            share = next(p["phi_share"] for p in tab["phases"] if p["kernel"] == k and p["phase"] == ph)
            step, g, _ = SPLIT[ph]
            over = g / step * (1 / kp - 1)
            kappa = max(16.0, L / 3)
            # CR_FORCED's steps and CR_HONEST's extra rounds are folded into the receipt words, so they are honest
            # slots a word as well: a thread's F (or H) a 32-deep k-block, an alu φ1 at H_PHI[0] slots or an FFMA at
            # one, over the tile's 128·128·32/L words
            per_word = m.get("threads", 256) * L / (128 * 128 * 32)
            forced = m.get("forced", 0) * (1 if m.get("fvar") == 2 else H_PHI[0]) * per_word
            h = rounds * H_PHI[var] + forced + m.get("honest", 0) * H_PHI[var] * per_word + H_TREE
            eta_w = ETA_GEMM[ph] * kp
            mult = kappa / (RHO * eta_w)
            f_terms = F + kappa * F * h / L
            nu_idle = max(0.0, idle[(k, ph)]["idle"]) if (k, ph) in idle else nu_tail[ph]
            nu_f = None if cheat is None else cheat.get(f"{k}@{ph}")
            src = "section 1" if nu_f is None else "cheat run"
            nu_f = frontier_slack(tab, k, ph) if nu_f is None else nu_f
            k_star = trade_threshold(tab, k, ph)
            nu_tol = cheat_tol.get(f"{k}@{ph}", nu_f)
            pick = lambda nf: nu_f if nf is None else nu_tol if nf == "tolerance" else nf
            eps = [f_terms + mult * pick(nf) + (BEST_ETA * nu_idle / eta_w if served else 0.0)
                   for _, served, nf in SCENARIOS]
            page = f_terms + mult * max(0.0, 1 - share)
            kappa_m = max(MEASURED_CREDIT, L / 3)
            eps_m = F + kappa_m * F * h / L + kappa_m / (RHO * eta_w) * nu_f
            eps_m_tol = F + kappa_m * F * h / L + kappa_m / (RHO * eta_w) * nu_tol
            cov = res["coverage"][ph]["macs"]
            e2e = [1 - cov * (1 - min(1.0, e)) for e in eps]
            res["designs"].append({"design": name, "kernel": k, "phase": ph, "L": L, "rounds": rounds, "var": var,
                                   "rate_kept": kp, "phi_share": share, "overhead": over, "kappa": kappa, "h": h,
                                   "eta_w": eta_w, "nu_multiplier": mult, "f_terms": f_terms, "idle": nu_idle,
                                   "nu_f": nu_f, "nu_f_source": src, "nu_f_tolerance": nu_tol,
                                   "eps_in_measured_credit_tolerance": eps_m_tol,
                                   "trade_threshold": k_star, "kappa_measured_credit": kappa_m, "eps_in_measured_credit": eps_m,
                                   "eps_in": dict(zip([s[0] for s in SCENARIOS], eps)), "eps_page": page,
                                   "eps_end_to_end": dict(zip([s[0] for s in SCENARIOS], e2e))})
            fmt = lambda e: f"{100 * e:.1f}%" if e < 1 else "vacuous"
            print(f"| {name} | {ph} | {100 * kp:.1f}% | {100 * over:.1f}% | {kappa:.1f} | {eta_w:.3f} | {mult:.2f} | "
                  f"{100 * f_terms:.2f}% | {100 * nu_idle:.1f}% | {100 * nu_f:.1f}% | {k_star:.1f} | "
                  + " | ".join(fmt(e) for e in eps)
                  + f" | {fmt(page)} | " + " | ".join(fmt(e) for e in e2e) + f" | {fmt(eps_m)} |")

    print("\n| phase | sequence | alone ms | filler | slowdown | filler's share of its solo rate | free share of its work "
          "| covert TFLOPS (mma) |")
    print("|---|---|---|---|---|---|---|---|")
    for p in gap["phases"]:
        for s in p["seqs"]:
            for kind in ("pump", "tag", "mma"):
                r = s.get(kind)
                if not r or "error" in r:
                    continue
                res["gap"].append({"phase": p["phase"], "seq": s["seq"], "kind": kind, "ms_alone": s["ms_alone"], **r})
                tf = f"{r['tflops_with']:.0f} (solo {r['alone_tflops']:.0f})" if kind == "mma" else "—"
                print(f"| {p['phase']} | {s['seq']} | {s['ms_alone']:.2f} | {kind} | {r['slowdown']:.3f}× | "
                      f"{100 * r['soak']:.1f}% | {100 * r['free']:.0f}% | {tf} |")
    for p in gap["phases"]:
        T = int(p["tokens"])
        g = next(s for s in p["seqs"] if s["seq"] == "gemm")
        ph = {"prefill2048": "p2048", "prefill8192": "p8192", "decode32": "d32"}[p["phase"]]
        if "pump" in g and "error" not in g["pump"]:
            # CUDA capacity a φ pump used for free beside the receipt-free GEMMs, as a share of the GEMMs' own time:
            # the idle SMs of the tails plus the co-issue slack ν_co beside the GEMMs' CTAs
            pu = g["pump"]
            free_cap = pu["free"] * pu["programs"] / pu["alone_programs_per_ms"] / g["ms_alone"]
            res.setdefault("pump_beside_gemms", {})[p["phase"]] = {
                "free_capacity": free_cap, "tails": nu_tail[ph], "nu_co": free_cap - nu_tail[ph]}
        if "mma" in g and "error" not in g["mma"]:
            honest = 2 * T * LINEAR_MACS / (g["ms_alone"] * 1e9)
            res.setdefault("covert_beside_gemms", {})[p["phase"]] = {
                "honest_tflops": honest, "covert_tflops": g["mma"]["tflops_with"],
                "covert_over_claimed": g["mma"]["tflops_with"] * g["mma"]["ms_with"] / (honest * g["ms_alone"]),
                "slowdown": g["mma"]["slowdown"]}
    print("\nfree CUDA capacity beside the GEMM-only sequence (pump):", json.dumps(res.get("pump_beside_gemms", {})))
    print("covert tensor work beside the GEMM-only sequence, over its claimed work:",
          json.dumps(res.get("covert_beside_gemms", {}), indent=None))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
