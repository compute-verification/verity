#!/usr/bin/env python3
"""PoUW windows restricted to the served FP8 GEMMs (CMP-8 round 2, section 2 of the saturation follow-up), derived.

  python windows_eps.py receipt-table.json gap-filler.json cutlass-receipt.json cutlass-corun.json [--json OUT]

Inputs: `cutlass_receipt_table.py --json` over a cgemm run (rate kept per kernel and phase), `gap_filler.py`'s JSON,
that cgemm run's own `cutlass-receipt.json` (the receipt-free kernel's time per linear, for the wave tails), and a corun
run's `cutlass-corun.json` (the idle SM-time a low-priority pump finds beside each design's kernel). Round 1's
numbers (`note:20261015T0417Z-draft-saturation-budget`) are constants below: the step split (`r20261015-042637-c9be`,
FP8 graphs) and the GEMMs' η.

The window bound is the page's slack, ε_T ≤ f/(1−3δ) + κ(f+ν)h/L with κ = max(16, L/3), with ν written per unit of the
window's CUDA capacity (ν_w, the share a cheater can use for unclaimed work) instead of per unit of the honest CUDA work:
ν = ν_w L / (h ρ η_w), so κνh/L = κ ν_w / (ρ η_w), ρ = 8 (FP8 MACs per CUDA slot), η_w = the GEMM's η × rate kept. SMs
that hold no CTA are tensor room as well: as served, a cheater's own GEMM in the measured idle SM-time adds
0.9 ν_idle / η_w (0.9: the best FP8 GEMM's share of peak, Pearl-C); with a tail filler that owns those SMs, what is left
is the CUDA slack ν_f. δ is taken as 0.
"""
from __future__ import annotations

import argparse
import json
import math

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
    ("3× alu, L = 128, spread (crossover)", "p_alu3_p1s", "d_alu16_p1"),
]
SCENARIOS = [  # name, measured idle SM-time as served?, ν_f (CUDA slack left once a tail filler owns the idle SMs)
    ("as served (measured idle, ν_f 0)", True, 0.0),
    ("tail filler, ν_f 0.04", False, 0.04),
    ("tail filler, ν_f 0.01", False, 0.01),
    ("tail filler, ν_f 0", False, 0.0),
]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("gap")
    ap.add_argument("cutlass")
    ap.add_argument("corun")
    ap.add_argument("--json")
    a = ap.parse_args()
    idle = measured_idle(json.load(open(a.corun)))
    tab = json.load(open(a.table))
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

    print("\n| design | phase | rate kept | honest overhead (step) | κ | η_w | ×ν | f terms | idle (corun) | "
          + " | ".join(f"ε_in: {s[0]}" for s in SCENARIOS) + " | page's reading (ν_w = 1 − φ share) | "
          + " | ".join(f"ε end to end: {s[0]}" for s in SCENARIOS) + " |")
    print("|---|---|---|---|---|---|---|---|---|" + "---|" * len(SCENARIOS) + "---|" + "---|" * len(SCENARIOS))
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
            h = rounds * H_PHI[var] + H_TREE
            eta_w = ETA_GEMM[ph] * kp
            mult = kappa / (RHO * eta_w)
            f_terms = F + kappa * F * h / L
            nu_idle = max(0.0, idle[(k, ph)]["idle"]) if (k, ph) in idle else nu_tail[ph]
            eps = [f_terms + mult * nf + (BEST_ETA * nu_idle / eta_w if served else 0.0) for _, served, nf in SCENARIOS]
            page = f_terms + mult * max(0.0, 1 - share)
            cov = res["coverage"][ph]["macs"]
            e2e = [1 - cov * (1 - min(1.0, e)) for e in eps]
            res["designs"].append({"design": name, "kernel": k, "phase": ph, "L": L, "rounds": rounds, "var": var,
                                   "rate_kept": kp, "phi_share": share, "overhead": over, "kappa": kappa, "h": h,
                                   "eta_w": eta_w, "nu_multiplier": mult, "f_terms": f_terms, "idle": nu_idle,
                                   "eps_in": dict(zip([s[0] for s in SCENARIOS], eps)), "eps_page": page,
                                   "eps_end_to_end": dict(zip([s[0] for s in SCENARIOS], e2e))})
            fmt = lambda e: f"{100 * e:.1f}%" if e < 1 else "vacuous"
            print(f"| {name} | {ph} | {100 * kp:.1f}% | {100 * over:.1f}% | {kappa:.1f} | {eta_w:.3f} | {mult:.2f} | "
                  f"{100 * f_terms:.2f}% | {100 * nu_idle:.1f}% | " + " | ".join(fmt(e) for e in eps)
                  + f" | {fmt(page)} | " + " | ".join(fmt(e) for e in e2e) + " |")

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
