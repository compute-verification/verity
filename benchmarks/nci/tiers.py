"""The verification stack's training throttle by tier: how much of the necessary subcomputation (counted multiplications:
an updated weight times an activation or activation gradient) an adversary gets out of a monitored fleet over a horizon.

The adversary runs a circuit A, declares D, and has two kinds of dark compute: headroom ``h`` on the known servers (PoUW
slack, uncovered compute such as host CPUs and in-switch reduction) and hidden hardware ``u`` elsewhere. Four
contributions, each an upper bound under the model; ``STATUS`` says which steps are proved:

* ``declared``: counted multiplications at declared gates. At most ``C / rho_T`` (the thread bound under partial
  containment, T1'), with ``rho_T = b_a / (alpha Lambda(K))``, ``Lambda(K) = max(e, 2K / (4 d b_w))``. When dark
  compute can't sum the declared blocks' partial sums, the partial-sum law (PS') is tighter: the chain's products are at
  most ``Psi(alpha C, B)``, where ``B`` is the crossing, the bits of declared outputs that reach dark gates. Headroom
  reads them on the server for free; hidden hardware reads them over links (line rate, or side channels plus readable
  declared egress under network accounting). A relay op reads at most ``2 b_p`` crossing bits.
* ``headroom``: dark ops on the known servers, plus work skipped by cheated proof steps. Together with the declared side
  it is capped by the server-window bound: a window reads its starting state plus what enters during it,
  ``K_r = state + ingress L``, and contains at most ``K_r Lambda(K_r) / b_a`` counted multiplications (``ThreadedReads``
  per region); the bound takes the tightest partition into windows. State is all memory; after a wipe, the unwipeable
  memory plus the proof of space's slack on filled memory plus what helpers can supply within the read deadline; under
  exhaustion, the HBM the declared workload doesn't hold plus the slack (host memory filled with proof of space).
  Ingress is line rate, or side channels, certified timing and NCI-bounded declared ingress ``alpha Phi``.
* ``hidden``: unknown hardware trains at full rate, ``u`` times monitored compute, less what it spends relaying.
* ``cheat``: with sampled proofs at a budget, steps of at most 16-bit output are drawn at rate
  ``q = share / (overhead + replay)``. The adversary's pool is ``ln(1 / p_evade) / q`` wrong steps per identity, a
  lifetime constant that doesn't grow with the fleet; a class of steps sampled at a floor of one draw adds
  ``ln(1 / p_evade)`` times its size. Each wrong step injects its 16 bits plus the choice of where, worth at most the
  chord slope of the per-block thread bound in counted multiplications per bit.

The canvas "Training throttle under the verification stack" renders this model with a slider per ``SPEC`` entry.

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

#: defaults: 100,000 eight-GPU servers for a year, the kimi-k2-1m-raw calibration with Kimi K2's forward chain, today's
#: proved wipe slack (1/19) and main's declared unwipeable memory, today's uncharged declared egress, and the 1e8-overhead
#: prover at a 1% budget
DEFAULTS = {
    "nodes": 1e5, "node_flops": 1.58e16, "years": 1.0, "target_flop": 1e26,
    "K": 2e7, "Gamma": 6.4e16, "b_w": 2.5, "b_a": 4.0, "d": 2048.0, "e": 4e4, "kappa": 0.667,
    "kappa_chain": 0.075, "d_chain": 1536.0, "e_chain": 12288.0, "b_p": 32.0,
    "h": 0.01, "u": 0.0,
    "sigma": 1e3, "tau": 400.0, "egress_decl": 4.4e9, "nic": 3.2e12,
    "hbm_gb": 640.0, "host_gb": 2000.0, "unwipeable_gb": 16.0, "slack": 1 / 19, "t_read": 1e-3, "wipe_s": 3600.0,
    "f": 0.95,
    "share": 0.01, "overhead": 1e8, "replay": 1.1e5, "step_flop": 1.43e4, "p_evade": 0.01, "identities": 1.0,
    "floor_steps": 1.0,
}

#: one entry per parameter: label, unit, slider range, log scale, and whether the leftmost position means zero
SPEC = [
    {"key": "nodes", "group": "Fleet and target", "label": "Servers (8-GPU nodes)", "unit": "", "min": 1e2, "max": 1e7, "log": True},
    {"key": "node_flops", "group": "Fleet and target", "label": "Server compute", "unit": "FLOP/s", "min": 1e15, "max": 1e17, "log": True},
    {"key": "years", "group": "Fleet and target", "label": "Horizon", "unit": "yr", "min": 0.05, "max": 10, "log": True},
    {"key": "target_flop", "group": "Fleet and target", "label": "Target pretrain", "unit": "FLOP", "min": 1e22, "max": 1e28, "log": True},
    {"key": "h", "group": "Dark compute", "label": "Headroom on known servers, h (fraction of their compute)", "unit": "", "min": 1e-12, "max": 1, "log": True, "zero": True},
    {"key": "u", "group": "Dark compute", "label": "Hidden hardware, u (fraction of monitored compute)", "unit": "", "min": 1e-12, "max": 10, "log": True, "zero": True},
    {"key": "sigma", "group": "Network accounting", "label": "Side channels per server, sigma", "unit": "bit/s", "min": 1, "max": 1e10, "log": True, "zero": True},
    {"key": "tau", "group": "Network accounting", "label": "Certified timing channel per server", "unit": "bit/s", "min": 1, "max": 1e5, "log": True, "zero": True},
    {"key": "egress_decl", "group": "Network accounting", "label": "Declared egress dark compute can read (today about 4.4e9; charged about alpha Phi)", "unit": "bit/s", "min": 1, "max": 1e12, "log": True, "zero": True},
    {"key": "nic", "group": "Network accounting", "label": "Line rate per server, without accounting", "unit": "bit/s", "min": 1e9, "max": 1e13, "log": True},
    {"key": "unwipeable_gb", "group": "Memory wipes", "label": "Unwipeable memory per server (main declares 16; Amodo's goal 1)", "unit": "GB", "min": 0.01, "max": 1000, "log": True},
    {"key": "slack", "group": "Memory wipes", "label": "Proof-of-space slack on filled memory (proved: 1/19)", "unit": "", "min": 1e-5, "max": 0.5, "log": True},
    {"key": "t_read", "group": "Memory wipes", "label": "Read deadline (helpers within it refill blocks)", "unit": "s", "min": 1e-6, "max": 1e-1, "log": True},
    {"key": "wipe_s", "group": "Memory wipes", "label": "Wipe period", "unit": "s", "min": 1, "max": 1e7, "log": True},
    {"key": "hbm_gb", "group": "Memory wipes", "label": "HBM per server", "unit": "GB", "min": 80, "max": 4000, "log": True},
    {"key": "host_gb", "group": "Memory wipes", "label": "Host memory per server (filled with proof of space)", "unit": "GB", "min": 1, "max": 10000, "log": True, "zero": True},
    {"key": "f", "group": "Memory exhaustion", "label": "Declared share of HBM, f", "unit": "", "min": 0, "max": 0.999, "log": False},
    {"key": "share", "group": "Sampled proofs", "label": "Compute share for replays and proofs", "unit": "", "min": 1e-4, "max": 0.5, "log": True},
    {"key": "overhead", "group": "Sampled proofs", "label": "Proof overhead per step (CUDA prover 1.6e8; ZK about 1e10)", "unit": "x", "min": 1, "max": 1e11, "log": True},
    {"key": "replay", "group": "Sampled proofs", "label": "Replay cost per drawn step (layer-token / step)", "unit": "x", "min": 1, "max": 1e7, "log": True},
    {"key": "step_flop", "group": "Sampled proofs", "label": "Work per proof step (one 7168-long dot product)", "unit": "FLOP", "min": 1e3, "max": 1e15, "log": True},
    {"key": "p_evade", "group": "Sampled proofs", "label": "Escape probability the adversary accepts", "unit": "", "min": 1e-4, "max": 0.99, "log": True},
    {"key": "identities", "group": "Sampled proofs", "label": "Identities a detection doesn't end (re-entries)", "unit": "", "min": 1, "max": 1e6, "log": True},
    {"key": "floor_steps", "group": "Sampled proofs", "label": "Steps in a class sampled at a floor of one draw", "unit": "", "min": 1, "max": 1e17, "log": True},
    {"key": "K", "group": "NCI policy and reference circuit", "label": "K: novel bits per NCI block", "unit": "bit", "min": 1e5, "max": 1e10, "log": True},
    {"key": "Gamma", "group": "NCI policy and reference circuit", "label": "Gamma: gates per NCI block", "unit": "", "min": 1e14, "max": 1e19, "log": True},
    {"key": "b_w", "group": "NCI policy and reference circuit", "label": "b_w: bits per updated weight", "unit": "bit", "min": 0.1, "max": 16, "log": True},
    {"key": "b_a", "group": "NCI policy and reference circuit", "label": "b_a: bits per activation", "unit": "bit", "min": 0.1, "max": 16, "log": True},
    {"key": "d", "group": "NCI policy and reference circuit", "label": "d: entry fee (smallest inner dimension)", "unit": "", "min": 1, "max": 65536, "log": True},
    {"key": "e", "group": "NCI policy and reference circuit", "label": "e: multiplications per activation value (fan-out)", "unit": "", "min": 1e3, "max": 1e6, "log": True},
    {"key": "kappa", "group": "NCI policy and reference circuit", "label": "kappa: share of training FLOP that are counted multiplications", "unit": "", "min": 0.05, "max": 1, "log": False},
    {"key": "kappa_chain", "group": "NCI policy and reference circuit", "label": "kappa_chain: share on the proved chain (one forward path)", "unit": "", "min": 0.01, "max": 1, "log": True},
    {"key": "d_chain", "group": "NCI policy and reference circuit", "label": "Chain's smallest inner dimension", "unit": "", "min": 64, "max": 65536, "log": True},
    {"key": "e_chain", "group": "NCI policy and reference circuit", "label": "Chain's largest fan-out", "unit": "", "min": 1e3, "max": 1e6, "log": True},
    {"key": "b_p", "group": "NCI policy and reference circuit", "label": "Partial-sum width a relay op reads", "unit": "bit", "min": 4, "max": 64, "log": True},
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
    ("Declared side, thread bound (T1')", "ThreadedThrottled is proved in Lean under total containment; T1' (partial containment, dark values "
     "unregistered) is on paper and ports line for line. Rests on WeightsIncompressible and the Threaded (d, e) premise, "
     "which nothing proves at Kimi's (2048, 4e4)."),
    ("Declared side, partial-sum law (PS')", "ContainedTraining is proved in Lean with no dark compute (B = 0); the absorption "
     "budget is on paper. Needs bounded dark fan-in and a reference circuit with bounded-fan-in additions (PartialSumsSupport "
     "is false for wide sum gates). Covers one forward chain, kappa_chain of the work."),
    ("Crossing budget", "Paper: dark gates outside every unit (N-dark) and a relay that splits at cap K (N-relay). Network "
     "accounting bounds what leaves a unit up to the declared frames' content (B4Covered), so readable declared egress counts "
     "in full."),
    ("Server-window bound", "Actual.Windows and ThreadedReads are proved; the composition is on paper and needs reads counted "
     "by determination rather than width (open), since a wipe bounds information, not width."),
    ("Wipe residual", "The band scheme's slack of 1/19 is proved (SecureErasureMeetsFamily, ideal primitive, isolation; in "
     "no lock). Unwipeable memory is declared, not measured. The helper term is an attack, so it is a lower bound on the "
     "residual; no theorem bounds help during a wipe."),
    ("Exhaustion", "Lemma M (Memory.Adds) is proved and locked per audit at retention 18/19; the continuous form is proved in "
     "the ideal-permutation model only; a bound at every instant, which the window bound needs, is not given. The reference "
     "Pi_2 is refuted (a prover passes from 1.53% of the store)."),
    ("Ingress", "Per-window ingress content and timing counts are proved (EncardIngressObs...); side channels are outside the "
     "model and enter as sigma."),
    ("Cheated proof steps", "Per-round audit bounds are proved (audit_le, audits_seq: a union bound); the cumulative harm "
     "budget over adaptive rounds, the cut-wire reduction and the per-block-cap thread theorem are on paper. The deployed "
     "verifier's sampling claims are stated, not proved."),
    ("Hidden hardware", "No bound: it trains at full rate."),
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


def evaluate(p: dict, mech: dict) -> dict:
    gpm = 2 / p["kappa"]
    H = p["years"] * YEAR
    C = p["nodes"] * p["node_flops"] * H
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

    head_ops = p["h"] * C + skipped
    hid_ops = p["u"] * C
    sigma_eff = p["sigma"] if mech["net"] else p["nic"]
    link = p["sigma"] + p["tau"] + p["egress_decl"] if mech["net"] else p["nic"]
    B_max = 2 * p["b_p"] * head_ops + min(2 * p["b_p"] * hid_ops, p["nodes"] * H * link)

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
    relay_head = min(head_ops, relay)
    head_mults = (head_ops - relay_head) / gpm
    hidden = max(0.0, hid_ops - (relay - relay_head)) / gpm

    mem = (p["hbm_gb"] + p["host_gb"]) * GB
    ingress = p["sigma"] + p["tau"] + alpha * p["node_flops"] if mech["net"] else p["nic"]
    help_bits = min(1.0, max(0.0, sigma_eff * p["t_read"] - BLOCK_NAME_BITS) / BLOCK_BITS) * mem
    residual = p["unwipeable_gb"] * GB + p["slack"] * mem + help_bits
    envelope = (p["hbm_gb"] * (1 - p["f"] * (1 - p["slack"])) + p["host_gb"] * p["slack"]) * GB + sigma_eff * T_AUDIT
    per_server = math.inf
    if mech["wipes"]:
        per_server = min(per_server, windows_bound(p, H, min(residual, mem), ingress, p["wipe_s"]))
    if mech["exhaust"]:
        per_server = min(per_server, windows_bound(p, H, min(envelope, mem), ingress, None))
    if not (mech["wipes"] or mech["exhaust"]):
        per_server = windows_bound(p, H, mem, ingress, None)
    region = p["nodes"] * per_server

    monitored = min(declared + head_mults, region)
    declared_part = min(declared, monitored)
    head_part = monitored - declared_part
    total = monitored + hidden + cheat
    if region < declared + head_mults:
        binding = "server windows"
    elif declared >= head_mults:
        binding = "thread bound" if declared >= cap_T * (1 - 1e-9) else "partial-sum law"
    else:
        binding = "headroom compute"
    if hidden > monitored + cheat:
        binding = "hidden hardware"
    return {
        "declared": declared_part, "headroom": head_part, "hidden": hidden, "cheat": cheat, "total": total,
        "runs": total / (p["target_flop"] / gpm), "slowdown": C / (gpm * total) if total > 0 else math.inf,
        "binding": binding, "rho_T": rho_T, "crossing_bits": B, "residual_bits": residual, "envelope_bits": envelope,
        "cheat_steps": k, "cheat_bits": bits,
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
    print(f"{'tier':5} {'declared':>10} {'headroom':>10} {'hidden':>8} {'cheat':>9} {'runs':>8} {'slowdown':>9}  binding")
    for name, r in results.items():
        print(f"{name:5} {r['declared']:10.3g} {r['headroom']:10.3g} {r['hidden']:8.2g} {r['cheat']:9.2g} "
              f"{r['runs']:8.3g} {r['slowdown']:9.3g}  {r['binding']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
