#!/usr/bin/env python3
"""Round 3's table (CMP-8): each receipt's rate kept, honest overhead and ε_in at the honest clock's τ, derived.

  python r3_table.py WIN.json DRIFT.json --cheat C ... [--design NAME=KERNEL ...] [--graph 0|1] [--json OUT]

WIN.json is `windows_eps.py --json` (rate kept, overhead, η_w, f terms per design and phase), DRIFT.json is
`drift_tau.py --json` over a drift run that ran each design's kernel, and each --cheat is `cheat_slack.py --json`.

A window is one GEMM launch, and the verifier knows its shape, so it closes each linear's window at that linear's own
honest τ_99.9 (`drift_tau`: the 99.9% launch time over the 0.1%). A phase's ε_in is then the claim-weighted mean of
each linear's ε_in at its τ, weighted by the honest time per launch (the claim is the launch's MACs at the design's
rate). ε_in is read two ways:

- `windows_eps.timed` at the page's κ = 16, tails filled: f + κfh/L plus the most of κ·added/ρ + 0.9(τ − s) over the
  cheat rows that fit (s ≤ τ + 0.002) and the null, over η_w;
- W6b's held floor (`pouw-core.lean` §10, `heldMax`): the most of (τ·fc − 0.999 s)(1 + κ/8) + 0.999·s·added·κ/8 over
  the same rows and the null, over η_w, at κ = 4 MACs a CUDA slot (ProductBuilt). The room τ·fc − s is the whole SM's,
  its tensor at peak and its CUDA at κ; fc is the clock a prover may run above the honest one (1 under an enforced
  clock; NVML's max SM clock over the honest one without).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import windows_eps as W  # noqa: E402

KAPPA_HELD = 4
F_HELD = 0.999  # A5b's Nmin / N
ENFORCED_TAU = 1.01


def held(rows, k, ph, eta_w, tau, kappa=KAPPA_HELD):
    """W6b's ε_in at one window τ·fc: the most over the classes that fit and the null (module doc)."""
    best = 0.0
    for s, a in [(1.0, 0.0)] + [(r["slowdown"], r["added"]) for r in rows if r["design"] == k and r["phase"] == ph]:
        if s <= tau + W.RESOLUTION:
            best = max(best, (tau - F_HELD * s) * (1 + kappa / 8) + F_HELD * s * a * kappa / 8)
    return best / eta_w


def a5b(rows, d, tau):
    return d["f_terms"] + W.timed(rows, d["kernel"], d["phase"], d["eta_w"], tau, d["kappa"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("win")
    ap.add_argument("drift")
    ap.add_argument("--cheat", action="append", default=[])
    ap.add_argument("--design", action="append", default=[], help="NAME=KERNEL, the designs to tabulate")
    ap.add_argument("--graph", type=int, default=1, help="read the drift run's graph (1) or host (0) launches")
    ap.add_argument("--fc", type=float, default=2430 / 2092, help="max SM clock over the honest clock")
    ap.add_argument("--json")
    a = ap.parse_args()
    win = json.load(open(a.win))
    drift = json.load(open(a.drift))
    rows = [r for c in a.cheat for r in json.load(open(c))["rows"] if r["extra"]]
    out = []
    print("| receipt | phase | GEMM rate kept | honest step overhead | honest τ (per linear, 99.9%) | "
          "ε_in at the honest τ (κ 16 / held, κ 4) | the same, a prover at the max clock (held, τ·fc) | "
          f"ε_in at τ = {ENFORCED_TAU}, enforced clock (κ 16 / held, κ 4) |")
    print("|---|---|---|---|---|---|---|---|")
    for spec in a.design:
        name, k = spec.split("=")
        for d in win["designs"]:
            if d["kernel"] != k or d["phase"] == "d32":
                continue
            ph = d["phase"]
            ks = [e for e in drift["kernels"] if e["kernel"] == k and e["graph"] == a.graph
                  and e["shape"].startswith(ph + "_")]
            if not ks:
                print(f"| {name} | {ph} | no drift rows for {k} |")
                continue
            wsum = sum(e["ms_med"] for e in ks)
            mean = lambda f: sum(e["ms_med"] * f(e["tau_999"]) for e in ks) / wsum
            r = {"design": name, "kernel": k, "phase": ph, "rate_kept": d["rate_kept"], "overhead": d["overhead"],
                 "tau": {e["shape"]: e["tau_999"] for e in ks}, "tau_weighted": mean(lambda t: t),
                 "tau_worst": max(e["tau_999"] for e in ks),
                 "eps_honest_tau": mean(lambda t: a5b(rows, d, t)),
                 "eps_honest_tau_held": mean(lambda t: held(rows, k, ph, d["eta_w"], t)),
                 "eps_honest_tau_held_fmax": mean(lambda t: held(rows, k, ph, d["eta_w"], t * a.fc)),
                 "eps_enforced": a5b(rows, d, ENFORCED_TAU),
                 "eps_enforced_held": held(rows, k, ph, d["eta_w"], ENFORCED_TAU)}
            out.append(r)
            fmt = lambda e: f"{100 * e:.1f}%" if e < 1 else "vacuous"
            print(f"| {name} | {ph} | {100 * d['rate_kept']:.1f}% | {100 * d['overhead']:.0f}% | "
                  f"{r['tau_weighted']:.4f} (worst {r['tau_worst']:.4f}) | "
                  f"{fmt(r['eps_honest_tau'])} / {fmt(r['eps_honest_tau_held'])} | {fmt(r['eps_honest_tau_held_fmax'])} | "
                  f"{fmt(r['eps_enforced'])} / {fmt(r['eps_enforced_held'])} |")
    if a.json:
        json.dump({"graph": a.graph, "fc": a.fc, "rows": out}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
