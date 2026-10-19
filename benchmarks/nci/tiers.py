"""The verification stack's training throttle by tier: how much of the necessary subcomputation (counted multiplications:
an updated weight times an activation or activation gradient) an adversary gets out of a monitored fleet over a horizon.

Four contributions, each an upper bound under the model:

* ``declared``: every monitored gate is declared and complies with NCI, so training there costs the thread bound's
  ``rho_T = b_a / (alpha * Lambda(K))`` gates per counted multiplication (``Nci.SecurityProofs.ThreadedThrottled``; its
  extension to undeclared gates is on paper). ``Lambda(K) = max(e, c K / (4 d b_w))`` with ``c = 2``.
* ``headroom``: spare compute ``h`` on known servers (PoUW slack, uncovered compute) runs undeclared work. A server-window
  reads its starting state plus what enters during it, ``K_r = state + ingress * T``, and contains at most
  ``K_r * Lambda(K_r) / b_a`` counted multiplications (``Nci.SecurityProofs.ThreadedReads`` per region); the bound takes
  the tightest partition into windows, and never exceeds the headroom's own compute. State is all memory, ``(1 - f)`` of
  it under memory exhaustion, or the wipe residual after a wipe (host memory filled with proof of space). Ingress is link
  capacity, or side channels plus declared ingress usable as data under network accounting.
* ``hidden``: unknown hardware trains at full rate, ``u`` times monitored compute; network accounting confines its help to
  monitored compute to side channels, treated as negligible here.
* ``cheat``: with sampled proofs at a budget, live coins and steps of at most 16-bit output proved at rate
  ``q = share / overhead``, the adversary cheats on ``ln(1 / p_evade) / q`` steps (times the servers if a detection only
  costs one server); each injects 16 chosen bits into its NCI block (charged as extra reads) and frees one step's work.

The canvas "Training throttle under the verification stack" renders this model with a slider per parameter.

    python benchmarks/nci/tiers.py            # the five tiers at the defaults
    python benchmarks/nci/tiers.py --json     # parameters, mechanisms and results, for other renderers
"""

from __future__ import annotations

import argparse
import json
import math

YEAR = 3.156e7
GB = 8e9

#: defaults: 100,000 eight-GPU servers for a year, the kimi-k2-1m-raw calibration, today's proved wipe residual (1/19 of
#: filled memory), and the 1e8-overhead prototype at a 1% budget
DEFAULTS = {
    "nodes": 1e5, "node_flops": 1.58e16, "years": 1.0,
    "K": 2e7, "Gamma": 6.4e16, "b_w": 2.5, "b_a": 4.0, "d": 2048.0, "e": 4e4, "kappa": 0.667,
    "u": 0.0, "h": 0.01,
    "sigma": 1e3, "ingress_declared": 4e5, "nic": 3.2e12,
    "hbm_gb": 640.0, "host_gb": 2000.0, "residual_gb": 155.0, "wipe_s": 600.0,
    "f": 0.95,
    "share": 0.01, "overhead": 1e8, "p_evade": 0.01, "per_server": False, "step_flop": 1.4e4,
    "target_flop": 1e26,
}

#: the cumulative tiers: which mechanisms each adds
TIERS = {
    "T1": {"net": False, "wipes": False, "exhaust": False, "sampled": False},
    "T2": {"net": True, "wipes": False, "exhaust": False, "sampled": False},
    "T3": {"net": True, "wipes": True, "exhaust": False, "sampled": False},
    "T4": {"net": True, "wipes": True, "exhaust": True, "sampled": False},
    "T5": {"net": True, "wipes": True, "exhaust": True, "sampled": True},
}


def lam(p: dict, K: float) -> float:
    return max(p["e"], 2 * K / (4 * p["d"] * p["b_w"]))


def region_cap(p: dict, K: float) -> float:
    """The most counted multiplications a region reading ``K`` bits contains."""
    return K * lam(p, K) / p["b_a"]


def evaluate(p: dict, mech: dict) -> dict:
    gates_per_mult = 2 / p["kappa"]
    H = p["years"] * YEAR
    C = p["nodes"] * p["node_flops"] * H
    rho_T = p["b_a"] / ((p["K"] / p["Gamma"]) * lam(p, p["K"]))
    declared = C / rho_T

    skip_flop = cheat = 0.0
    if mech["sampled"]:
        m = math.log(1 / p["p_evade"]) / (p["share"] / p["overhead"])
        if p["per_server"]:
            m *= p["nodes"]
        skip_flop = m * p["step_flop"]
        cheat = 16 * m * lam(p, p["K"]) / p["b_a"]

    mem_all = (p["hbm_gb"] + p["host_gb"]) * GB
    mem_hbm = p["hbm_gb"] * GB
    if mech["wipes"]:
        state = min(p["residual_gb"] * GB, (1 - p["f"]) * mem_hbm if mech["exhaust"] else mem_hbm)
    else:
        state = (1 - p["f"]) * mem_all if mech["exhaust"] else mem_all
    ingress = p["sigma"] + p["ingress_declared"] if mech["net"] else p["nic"]
    best = math.inf
    for i in range(361):
        T = 10 ** (9 * i / 360)
        if mech["wipes"] and T < p["wipe_s"]:
            continue
        span = min(T, H)
        best = min(best, max(1.0, H / span) * region_cap(p, state + ingress * span))
    compute_cap = (p["h"] * p["node_flops"] * H + skip_flop / p["nodes"]) / gates_per_mult
    headroom = p["nodes"] * min(compute_cap, best)
    hidden = p["u"] * C / gates_per_mult
    total = declared + headroom + hidden + cheat
    return {"declared": declared, "headroom": headroom, "hidden": hidden, "cheat": cheat, "total": total,
            "runs": total / (p["target_flop"] / gates_per_mult), "slowdown": C / (gates_per_mult * total),
            "headroom_compute_bound": compute_cap <= best, "rho_T": rho_T, "state_bits": state}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="parameters, mechanisms and results as JSON")
    args = ap.parse_args(argv)
    results = {name: evaluate(DEFAULTS, mech) for name, mech in TIERS.items()}
    if args.json:
        print(json.dumps({"defaults": DEFAULTS, "tiers": TIERS, "results": results}, indent=1))
        return 0
    print(f"{'tier':5} {'declared':>10} {'headroom':>10} {'hidden':>8} {'cheat':>9} {'runs':>8} {'slowdown':>9}")
    for name, r in results.items():
        print(f"{name:5} {r['declared']:10.2g} {r['headroom']:10.2g} {r['hidden']:8.2g} {r['cheat']:9.2g} "
              f"{r['runs']:8.2g} {r['slowdown']:9.2g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
