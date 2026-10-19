"""Compute efficiency of frontier pretraining from scratch under the verification stack: the FLOP an unconstrained run
needs, divided by the FLOP the adversary spends, monitored plus unmonitored. The adversary must still compute the
necessary subcomputation, the counted multiplications (an updated weight times an activation or activation gradient).

Efficiency is a sum of four terms, each an upper bound under the model; ``STATUS`` says which steps are proved.

* Unmonitored compute trains at full efficiency.
* The monitored compute residual (monitored compute that PoUW doesn't attest) trains at full efficiency unless memory
  caps it. A server-window reads its starting state plus what enters during it, ``K_r = state + ingress L``, and
  contains at most ``K_r Lambda(K_r) / b_a`` counted multiplications (``ThreadedReads`` per region), counting every gate
  on the server, declared or not. The state is all memory; after a wipe, what the wipe leaves plus what helpers refill
  within the read deadline; under exhaustion, the HBM the declared workload doesn't hold plus the proof-of-space slack.
  Ingress is line rate or, under network accounting, side channels plus certified timing plus the NCI-bounded declared
  ingress ``alpha Phi``.
* Declared compute trains at the thread bound, ``rho_T = b_a / (alpha Lambda(K))`` gates per counted multiplication with
  ``Lambda(K) = max(e, 2K / (4 d b_w))``, or at the tighter partial-sum law when no dark compute can sum the declared
  blocks' partial sums. The residual reads them on the server for free; unmonitored compute reads them over links (line
  rate, or side channels plus readable declared egress under network accounting). A relay op reads ``2 b_p`` bits.
* With sampled proofs at a budget, the adversary's pool of undetected wrong steps is ``ln(1 / p_evade) / q`` per
  identity, ``q = share / (overhead + replay)``, a lifetime constant. Each injects its 16 bits plus the choice of where,
  worth at most the chord slope of the per-block thread bound per bit.

The canvas "Compute efficiency under the verification stack" renders this model.

    python benchmarks/nci/tiers.py            # the five tiers at the defaults
    python benchmarks/nci/tiers.py --json     # spec, defaults, tiers, status and results, for other renderers
"""

from __future__ import annotations

import argparse
import json
import math

YEAR = 3.156e7
GB = 8e9
#: Lemma M's audit window: the most a helper can add to a store under audit is the side channel over it
T_AUDIT = 0.3635
#: the band scheme's block, and the bits that name one, for the outsourcing attack on a wipe
BLOCK_BITS = 524288
BLOCK_NAME_BITS = 25

#: defaults: the Kimi K2 calibration and forward chain; 100,000 eight-GPU servers for a year (only the cheating pool, a
#: lifetime constant, depends on the fleet); what a wipe leaves today (the proved slack of 1/19 on 2.6 TB plus main's
#: declared 16 GB unwipeable); today's uncharged declared egress; the C-Flock CUDA prover's overhead at a 1% budget
DEFAULTS = {
    "unmonitored": 0.0, "residual": 0.01,
    "wipe_left_gb": 155.0, "sigma": 1e3, "b_w": 2.5, "b_a": 4.0,
    "f": 0.95, "slack": 1 / 19, "t_read": 1e-3, "wipe_s": 3600.0, "hbm_gb": 640.0, "host_gb": 2000.0,
    "tau": 400.0, "egress_decl": 4.4e9, "nic": 3.2e12,
    "share": 0.01, "overhead": 1.64e8, "replay": 1.1e5, "step_flop": 1.43e4, "p_evade": 0.01, "identities": 1.0,
    "floor_steps": 1.0,
    "K": 2e7, "Gamma": 6.4e16, "d": 2048.0, "e": 4e4, "kappa": 0.667,
    "kappa_chain": 0.075, "d_chain": 1536.0, "e_chain": 12288.0, "b_p": 32.0,
    "nodes": 1e5, "node_flops": 1.58e16, "years": 1.0,
}

#: one entry per parameter: label, unit, slider range, log scale, whether the leftmost position means zero, and whether
#: it is one of the few shown up front
SPEC = [
    {"key": "unmonitored", "main": True, "label": "Unmonitored compute share", "unit": "", "min": 1e-12, "max": 0.5, "log": True, "zero": True},
    {"key": "residual", "main": True, "label": "Monitored compute residual", "unit": "", "min": 1e-12, "max": 1, "log": True, "zero": True},
    {"key": "wipe_left_gb", "main": True, "label": "Memory a wipe leaves per server (proved 155, declared 16, goal 1)", "unit": "GB", "min": 0.01, "max": 3000, "log": True},
    {"key": "sigma", "main": True, "label": "Side channels per server", "unit": "bit/s", "min": 1, "max": 1e10, "log": True, "zero": True},
    {"key": "b_w", "main": True, "label": "Compression: bits per updated weight", "unit": "bit", "min": 0.1, "max": 16, "log": True},
    {"key": "b_a", "main": True, "label": "Compression: bits per activation", "unit": "bit", "min": 0.1, "max": 16, "log": True},
    {"key": "f", "label": "Exhaustion: declared share of HBM", "unit": "", "min": 0, "max": 0.999, "log": False},
    {"key": "slack", "label": "Proof-of-space slack on filled memory (proved 1/19)", "unit": "", "min": 1e-5, "max": 0.5, "log": True},
    {"key": "t_read", "label": "Wipe read deadline (helpers within it refill blocks)", "unit": "s", "min": 1e-6, "max": 1e-1, "log": True},
    {"key": "wipe_s", "label": "Wipe period", "unit": "s", "min": 1, "max": 1e7, "log": True},
    {"key": "hbm_gb", "label": "HBM per server", "unit": "GB", "min": 80, "max": 4000, "log": True},
    {"key": "host_gb", "label": "Host memory per server (filled with proof of space)", "unit": "GB", "min": 1, "max": 10000, "log": True, "zero": True},
    {"key": "tau", "label": "Certified timing channel per server", "unit": "bit/s", "min": 1, "max": 1e5, "log": True, "zero": True},
    {"key": "egress_decl", "label": "Declared egress dark compute can read (today 4.4e9; charged about 5e6)", "unit": "bit/s", "min": 1, "max": 1e12, "log": True, "zero": True},
    {"key": "nic", "label": "Line rate per server, without network accounting", "unit": "bit/s", "min": 1e9, "max": 1e13, "log": True},
    {"key": "share", "label": "Sampled proofs: compute share", "unit": "", "min": 1e-4, "max": 0.5, "log": True},
    {"key": "overhead", "label": "Sampled proofs: overhead per step (CUDA prover 1.64e8; ZK about 1e10)", "unit": "x", "min": 1, "max": 1e11, "log": True},
    {"key": "replay", "label": "Sampled proofs: replay cost per drawn step", "unit": "x", "min": 1, "max": 1e7, "log": True},
    {"key": "step_flop", "label": "Sampled proofs: work per step", "unit": "FLOP", "min": 1e3, "max": 1e15, "log": True},
    {"key": "p_evade", "label": "Sampled proofs: escape probability accepted", "unit": "", "min": 1e-4, "max": 0.99, "log": True},
    {"key": "identities", "label": "Sampled proofs: identities a detection doesn't end", "unit": "", "min": 1, "max": 1e6, "log": True},
    {"key": "floor_steps", "label": "Sampled proofs: steps in a class drawn at a floor of one", "unit": "", "min": 1, "max": 1e17, "log": True},
    {"key": "K", "label": "NCI: novel bits per block", "unit": "bit", "min": 1e5, "max": 1e10, "log": True},
    {"key": "Gamma", "label": "NCI: gates per block", "unit": "", "min": 1e14, "max": 1e19, "log": True},
    {"key": "d", "label": "Model: entry fee (smallest inner dimension)", "unit": "", "min": 1, "max": 65536, "log": True},
    {"key": "e", "label": "Model: multiplications per activation value", "unit": "", "min": 1e3, "max": 1e6, "log": True},
    {"key": "kappa", "label": "Model: share of training FLOP that are counted multiplications", "unit": "", "min": 0.05, "max": 1, "log": False},
    {"key": "kappa_chain", "label": "Model: share on the proved chain (one forward path)", "unit": "", "min": 0.01, "max": 1, "log": True},
    {"key": "d_chain", "label": "Model: chain's smallest inner dimension", "unit": "", "min": 64, "max": 65536, "log": True},
    {"key": "e_chain", "label": "Model: chain's largest fan-out", "unit": "", "min": 1e3, "max": 1e6, "log": True},
    {"key": "b_p", "label": "Model: partial-sum width a relay op reads", "unit": "bit", "min": 4, "max": 64, "log": True},
    {"key": "nodes", "label": "Fleet: servers", "unit": "", "min": 1e2, "max": 1e7, "log": True},
    {"key": "node_flops", "label": "Fleet: server compute", "unit": "FLOP/s", "min": 1e15, "max": 1e17, "log": True},
    {"key": "years", "label": "Fleet: horizon", "unit": "yr", "min": 0.05, "max": 10, "log": True},
]

#: the cumulative tiers: which mechanisms each adds
TIERS = {
    "T1": {"net": False, "wipes": False, "exhaust": False, "sampled": False},
    "T2": {"net": True, "wipes": False, "exhaust": False, "sampled": False},
    "T3": {"net": True, "wipes": True, "exhaust": False, "sampled": False},
    "T4": {"net": True, "wipes": True, "exhaust": True, "sampled": False},
    "T5": {"net": True, "wipes": True, "exhaust": True, "sampled": True},
}

#: what each step of the model rests on
STATUS = [
    ("Declared compute, thread bound", "ThreadedThrottled is proved in Lean when the declared circuit contains the whole "
     "reference circuit; the extension to dark gates (T1') is on paper. Rests on WeightsIncompressible and the Threaded "
     "(d, e) premise, which nothing proves at Kimi's (2048, 4e4)."),
    ("Declared compute, partial-sum law", "ContainedTraining is proved in Lean with no dark compute; the absorption budget "
     "is on paper. Needs a reference circuit with bounded-fan-in additions. Covers one forward chain."),
    ("Memory cap on the residual", "Actual.Windows and ThreadedReads are proved; the composition is on paper and needs "
     "reads counted by information rather than width (open)."),
    ("Memory a wipe leaves", "The slack of 1/19 is proved (SecureErasureMeetsFamily, ideal primitive, isolation; in no "
     "lock). Unwipeable memory is declared, not measured. No theorem bounds help during a wipe; the helper term is an "
     "attack."),
    ("Exhaustion", "Lemma M is proved and locked per audit; no bound at every instant, which the memory cap needs."),
    ("Cheated proof steps", "Per-round audit bounds are proved; the cumulative harm budget and the per-block-cap thread "
     "theorem are on paper. The deployed verifier's sampling claims are stated, not proved."),
    ("Unmonitored compute", "No bound: it trains at full efficiency."),
]


def lam(p: dict, K: float) -> float:
    return max(p["e"], 2 * K / (4 * p["d"] * p["b_w"]))


def region_cap(p: dict, K: float) -> float:
    """The most counted multiplications a region reading ``K`` bits contains."""
    return K * lam(p, K) / p["b_a"]


def chain_rates(p: dict, steps: int = 400) -> list[tuple[float, float]]:
    """``(mu, max(m(mu), mu b_a / Lambda_c))`` on a grid, PS''s per-product read rates (``iu_reads_ge`` and the column
    lemma)."""
    K, b_w, b_a, d = p["K"], p["b_w"], p["b_a"], p["d_chain"]
    lam_c = max(p["e_chain"], K / (4 * b_w * d))
    out = []
    for k in range(1, steps + 1):
        mu = k / steps
        s = math.sqrt(mu)
        c1 = (1 + s) ** 2 * b_w * b_a * d / K
        c2 = math.sqrt(2) * (1 + s) * math.sqrt(max(0.0, 1 - mu)) * b_a * math.sqrt(b_w / K)
        out.append((mu, max(min(c1, c2), mu * b_a / lam_c)))
    return out


def chain_products(rates: list[tuple[float, float]], R: float, Z: float) -> float:
    """PS': the most chain products declared blocks reading ``R`` bits compute when dark gates absorb ``Z`` bits of their
    partial sums."""
    return min((R + (1 - mu) * Z) / rate for mu, rate in rates)


def full_block_bits(p: dict) -> float:
    """The reads at which one block's thread bound reaches all its work, ``kappa Gamma / 2`` counted multiplications."""
    target = p["kappa"] * p["Gamma"] / 2
    linear = target * p["b_a"] / p["e"]
    if 2 * linear / (4 * p["d"] * p["b_w"]) <= p["e"]:
        return linear
    return math.sqrt(target * 4 * p["d"] * p["b_w"] * p["b_a"] / 2)


def windows_bound(p: dict, H: float, state: float, ingress: float, period: float | None) -> float:
    """The fewest counted multiplications any partition of one server's horizon into windows allows; with a wipe
    ``period``, windows start at wipes and span whole periods."""
    if period is None:
        lengths = [10 ** (-3 + 12 * i / 360) for i in range(361)]
    else:
        ks = sorted({max(1, round(10 ** (j / 40))) for j in range(0, 481)})
        lengths = [k * period for k in ks if k == 1 or k * period <= H]
    best = math.inf
    for L in lengths:
        span = min(L, H)
        best = min(best, math.ceil(H / span - 1e-9) * region_cap(p, state + ingress * span))
    return best


def ingress(p: dict, mech: dict) -> float:
    return p["sigma"] + p["tau"] + p["K"] / p["Gamma"] * p["node_flops"] if mech["net"] else p["nic"]


def state_threshold(p: dict, mech: dict) -> float:
    """The server state, in bits, below which memory caps the residual (the window bound's quadratic regime)."""
    return p["residual"] * p["node_flops"] * p["kappa"] * p["d"] * p["b_w"] * p["b_a"] / (4 * ingress(p, mech))


def evaluate(p: dict, mech: dict) -> dict:
    gpm = 2 / p["kappa"]
    H = p["years"] * YEAR
    C = p["nodes"] * p["node_flops"] * H
    s = min(p["unmonitored"], 0.999)
    U = C * s / (1 - s)
    alpha = p["K"] / p["Gamma"]
    rho_T = p["b_a"] / (alpha * lam(p, p["K"]))
    cap_T = C / rho_T

    k = skipped = bits = cheat = 0.0
    if mech["sampled"]:
        L = math.log(1 / p["p_evade"])
        q = p["share"] / (p["overhead"] + p["replay"])
        k = p["identities"] * L / q
        skipped = k * p["step_flop"]
        k_all = k + p["identities"] * L * p["floor_steps"]
        S_full = full_block_bits(p)
        per_block = max(1.0, min(k_all, S_full / 28))
        b_bar = max(16.0, 16 + math.log2(math.e * (p["Gamma"] / p["step_flop"]) / per_block))
        bits = k_all * b_bar
        slope = (p["kappa"] * p["Gamma"] / 2 - region_cap(p, p["K"])) / (S_full - p["K"]) if S_full > p["K"] else 0.0
        cheat = min(slope * bits, C / gpm)

    res_ops = p["residual"] * C + skipped
    sigma_eff = p["sigma"] if mech["net"] else p["nic"]
    link = p["sigma"] + p["tau"] + p["egress_decl"] if mech["net"] else p["nic"]
    B_max = 2 * p["b_p"] * res_ops + min(2 * p["b_p"] * U, p["nodes"] * H * link)

    rates = chain_rates(p)
    share = p["kappa"] / p["kappa_chain"]

    def declared_at(B: float) -> float:
        return min(cap_T, share * chain_products(rates, alpha * C, B))

    if declared_at(B_max) < cap_T:
        B = B_max
    else:
        lo, hi = 0.0, B_max
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if declared_at(mid) >= cap_T else (mid, hi)
        B = hi
    declared = declared_at(B)
    relay = B / (2 * p["b_p"])
    relay_res = min(res_ops, relay)
    res_mults = (res_ops - relay_res) / gpm
    unmon = max(0.0, U - (relay - relay_res)) / gpm

    mem = (p["hbm_gb"] + p["host_gb"]) * GB
    rate_in = ingress(p, mech)
    help_bits = min(1.0, max(0.0, sigma_eff * p["t_read"] - BLOCK_NAME_BITS) / BLOCK_BITS) * mem
    wipe_state = p["wipe_left_gb"] * GB + help_bits
    envelope = (p["hbm_gb"] * (1 - p["f"] * (1 - p["slack"])) + p["host_gb"] * p["slack"]) * GB + sigma_eff * T_AUDIT
    per_server = math.inf
    if mech["wipes"]:
        per_server = min(per_server, windows_bound(p, H, min(wipe_state, mem), rate_in, p["wipe_s"]))
    if mech["exhaust"]:
        per_server = min(per_server, windows_bound(p, H, min(envelope, mem), rate_in, None))
    if not (mech["wipes"] or mech["exhaust"]):
        per_server = windows_bound(p, H, mem, rate_in, None)
    region = p["nodes"] * per_server

    monitored = min(declared + res_mults, region)
    declared_part = min(declared, monitored)
    res_part = monitored - declared_part
    spent = C + U
    terms = {"unmonitored": gpm * unmon / spent, "residual": gpm * res_part / spent,
             "declared": gpm * declared_part / spent, "cheat": gpm * cheat / spent}
    capped = region < declared + res_mults
    largest = max(terms, key=lambda t: terms[t])
    if largest == "declared":
        largest = "declared (thread bound)" if declared >= cap_T * (1 - 1e-9) else "declared (partial-sum law)"
    return {
        "efficiency": sum(terms.values()), **terms, "largest": largest, "memory_caps_residual": capped,
        "rho_T": rho_T, "crossing_bits": B, "wipe_state_bits": wipe_state, "envelope_bits": envelope,
        "state_threshold_bits": state_threshold(p, mech), "cheat_steps": k, "cheat_bits": bits,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="spec, defaults, tiers, status and results as JSON")
    args = ap.parse_args(argv)
    results = {name: evaluate(DEFAULTS, mech) for name, mech in TIERS.items()}
    if args.json:
        print(json.dumps({"spec": SPEC, "defaults": DEFAULTS, "tiers": TIERS,
                          "status": [{"piece": a, "status": b} for a, b in STATUS], "results": results}, indent=1))
        return 0
    print(f"{'tier':5} {'efficiency':>10} {'unmonitored':>11} {'residual':>9} {'declared':>9} {'cheat':>9}  largest")
    for name, r in results.items():
        print(f"{name:5} {r['efficiency']:10.3g} {r['unmonitored']:11.3g} {r['residual']:9.3g} {r['declared']:9.3g} "
              f"{r['cheat']:9.2g}  {r['largest']}{' (memory caps it)' if r['memory_caps_residual'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
