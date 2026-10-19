"""Compute efficiency eta of frontier pretraining from scratch under the verification stack: the FLOP an unconstrained run
needs, divided by the FLOP the developer spends, monitored plus unmonitored. The developer must still compute the counted
multiplications M (an updated weight times an activation or activation gradient). Terms and symbols follow the proposal
"Terms and symbols for the training-efficiency bound" (Notion, under the assumptions register) and DESIGN.md's Glossary.

eta <= s + min(r, memory cap) + 1/rho_NCI + chi, each term an upper bound under the model; ``STATUS`` says which steps are
proved.

* s, the unmonitored compute share, trains at full efficiency.
* r, the monitored compute residual, is the share of monitored capacity outside the compute certificate (1 - W_cert/C):
  idle and uncredited GPU time, plus host CPUs, which PoUW neither credits nor bounds and which count at their datasheet
  peak. It trains at full efficiency unless the memory cap binds. A
  spatial unit's window reads its starting state plus its novel intake, K_w = state + lambda_in L, and contains at most
  K_w Lambda(K_w) / b_a counted multiplications (``ThreadedReads`` per window), declared or not. The state is all memory,
  disks included unless they're unplugged; after a wipe, the wipe residual R_wipe: memory the fill doesn't reach, the proof of
  space's slack on the filled memory, and what helpers refill within the read deadline; under exhaustion, the HBM the
  declared computation doesn't hold plus the slack. The slack is 1 - delta^(1/k) for k timed rounds (1/19 at k = 86 and
  delta = 1/100) plus the certificate's floor, the share of the store rebuildable from public data within the deadline.
  The novel intake is line rate or, under network accounting, side channels plus certified timing plus the declared
  computation's novel intake, alpha F under NCI once the count check is in the ZK flow.
* 1/rho_NCI: the declared computation trains at the thread bound, rho_T = b_a / (alpha Lambda(K)) declared gates per
  counted multiplication with Lambda(K) = max(e, 2K / (4 d b_w)), or at the tighter partial-sum law rho_PS when no dark
  compute can sum the inference blocks' partial sums. The residual reads them on the server for free; unmonitored compute
  reads them through the export (line rate, or side channels plus readable export under network accounting). A relay op
  reads 2 b_p bits, and the developer relays only as many partial sums as buy more declared work than the relay's gates
  would train directly.
* The verdict lag: from its first faulty proof block until the halt (a window plus the verdict delay), a developer can skip
  all declared work, once per identity, in every tier; that capacity joins the residual. Detection must end the
  identity and take its hardware out of use, or the capacity after the halt is unmonitored.
* chi, the cheating share: with sampled proofs at a budget, the faulty proof blocks the sampling design accepts number
  A ln(1/delta) / q per identity over the whole deployment, q = share / (overhead + replay) (``Population.Exfil.tail_cum``),
  where A, the luck factor, turns faulty outputs into skipped work. Each faulty proof block skips its work and injects its output
  bits plus the choice of where, worth at most the chord slope of the per-inference-block thread bound per bit.

The four terms bound counted multiplications relative to the reference recipe. A private recipe g times better in counted
multiplications per unit of loss is a compute amplifier, multiplying eta by g; it cancels when verification is compared on
against off. A recipe better at working under the mechanisms, m times more progress per counted multiplication, scales
only the constrained terms (nu, mu and chi), not the residual or unmonitored compute. The compute dampener D is a hardware
multiple: how many times more hardware-time the developer needs than an unconstrained developer with the same recipe on
the same hardware, who reaches model-FLOP utilization mfu. Unmonitored compute trains like the unconstrained developer;
the other terms are counted at peak, so 1/D <= s + (min(r + nu, mu) + chi) / mfu, and D is at least 1.

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
#: the proof-of-space band block, and the bits that name one, for the outsourcing attack on a wipe
BLOCK_BITS = 524288
BLOCK_NAME_BITS = 25

#: defaults: the Kimi K2 calibration and forward chain; the GPUs' residual under the floor credit at f_max, 1 - (1 - eps)
#: ((1 - phi)/q_sched + phi)/q_clock = 0.326: honest kernels run 1.72x the floor at the nodes' locked clock (run
#: r20261017-123922-e558), diluted by the training window's filler (the C7 dossier; 0.217 with an attested clock; 1%
#: needs eight conditions; today nothing in a training window is credited); the GPU clock above what PoUW credits
#: (NVIDIA's dense headline implies 2,597 MHz against the credited 2,430: 6.4%, C4 dossier); the host CPUs at their INT8
#: AMX datasheet peak (two Xeon 6776Ps, 128 cores x 2,048 ops per cycle x 3.6 GHz: 11.8% of the GPUs' FP8 rate);
#: 100,000 servers of 8 RTX PRO 6000s (vy-nebius-2's 2.668 TB) for a year;
#: the deployed proof-of-space audit (k = 111 rounds at delta = 1/100; the slack counts the floor once, memory-accounting
#: MEM-51) and the memory
#: the fill doesn't reach (3.05 GB on-chip, 18.4 GB GPU reserve, 29.5 GB host gap, about 7 GB of DPU, BMC and flash); disks
#: unplugged; today's uncharged export; the declared computation's novel intake at alpha F (once NET-35's count check is in
#: the ZK flow; line rate until then); sampled proofs at 1% on top of serve-time commitments, at today's sparse-draw cost
#: (about 35 GPU-s a draw), delta = 2^-40 and luck A = 8
DEFAULTS = {
    "unmonitored": 0.0, "residual": 0.326, "host_share": 0.118, "clock_share": 0.064,
    "pous_rounds": 111.0, "unfilled_gb": 58.0, "sigma": 1e3, "b_w": 2.5, "b_a": 4.0,
    "f": 0.95, "pous_delta": 0.01, "pous_floor": 0.0, "t_read": 1e-3, "wipe_s": 3600.0, "hbm_gb": 768.0,
    "host_gb": 1900.0, "disk_gb": 0.0,
    "tau": 860.0, "export": 4.4e9, "nic": 3.2e12,
    "share": 0.01, "overhead": 4.8e12, "replay": 1.1e5, "step_flop": 1.43e4, "p_evade": 2.0 ** -40, "identities": 1.0,
    "luck": 8.0, "step_bits": 16.0, "intake_x": 1.0, "lag_s": 3600.0, "g": 1.0, "mfu": 0.4, "m": 1.0,
    "floor_steps": 1.0,
    "K": 2e7, "Gamma": 6.4e16, "d": 2048.0, "e": 4e4, "kappa": 0.6,
    "kappa_chain": 0.0695, "d_chain": 1536.0, "e_chain": 12288.0, "b_p": 32.0,
    "nodes": 1e5, "node_flops": 8e15, "years": 1.0,
}

#: one entry per parameter: label, unit, slider range, log scale, whether the leftmost position means zero, and whether
#: it is one of the few shown up front
SPEC = [
    {"key": "unmonitored", "main": True, "label": "Unmonitored compute share", "unit": "", "min": 1e-12, "max": 0.5, "log": True, "zero": True},
    {"key": "g", "main": True, "label": "Compute amplifier g: the developer's algorithmic advantage over the reference recipe (1 against the best known recipe)", "unit": "x", "min": 1, "max": 100, "log": True},
    {"key": "residual", "main": True, "label": "Monitored compute residual r on the GPUs (floor credit at the locked clock 0.326, with an attested clock 0.217; best plausible design 0.01; nothing credited today: 1)", "unit": "", "min": 1e-12, "max": 1, "log": True, "zero": True},
    {"key": "clock_share", "label": "GPU capacity above the clock PoUW credits (6.4%: 2,597 MHz against 2,430; 0 once credit uses the true maximum or auditor hardware reads the clock offset)", "unit": "", "min": 1e-4, "max": 0.3, "log": True, "zero": True},
    {"key": "host_share", "main": True, "label": "Host CPUs at datasheet peak, as a share of server compute (Xeon 6776P INT8 AMX 11.8%; hosts without matrix units, two EPYC 9015s, 0.21%)", "unit": "", "min": 1e-6, "max": 0.5, "log": True, "zero": True},
    {"key": "pous_rounds", "main": True, "label": "Proof-of-space rounds per audit (deployed 111; 1/19 needs 86; streamed labels make 74,000 cheap)", "unit": "", "min": 10, "max": 1e6, "log": True},
    {"key": "unfilled_gb", "main": True, "label": "Memory the fill doesn't reach, per server (measured gaps 51 plus about 7 of DPU, BMC and flash; on-chip alone 3.05)", "unit": "GB", "min": 0.01, "max": 1000, "log": True},
    {"key": "sigma", "main": True, "label": "Side channels per server", "unit": "bit/s", "min": 1, "max": 1e10, "log": True, "zero": True},
    {"key": "b_w", "main": True, "label": "Compression: bits per updated weight", "unit": "bit", "min": 0.1, "max": 16, "log": True},
    {"key": "b_a", "main": True, "label": "Compression: bits per activation", "unit": "bit", "min": 0.1, "max": 16, "log": True},
    {"key": "f", "label": "Exhaustion: declared share of HBM", "unit": "", "min": 0, "max": 0.999, "log": False},
    {"key": "pous_delta", "label": "Proof-of-space failure probability per certificate", "unit": "", "min": 1e-12, "max": 0.5, "log": True},
    {"key": "pous_floor", "label": "Proof-of-space share the developer can rebuild at any k: 0 when the permutation is ideal or labels are streamed (the slack counts it once); about 0.985 under the broken reference permutation", "unit": "", "min": 1e-6, "max": 1, "log": True, "zero": True},
    {"key": "disk_gb", "label": "Disk per server left in place and unwiped (0 when unplugged)", "unit": "GB", "min": 1, "max": 1e5, "log": True, "zero": True},
    {"key": "t_read", "label": "Wipe read deadline (helpers within it refill proof-of-space blocks)", "unit": "s", "min": 1e-6, "max": 1e-1, "log": True},
    {"key": "wipe_s", "label": "Wipe period", "unit": "s", "min": 1, "max": 1e7, "log": True},
    {"key": "hbm_gb", "label": "HBM per server", "unit": "GB", "min": 80, "max": 4000, "log": True},
    {"key": "host_gb", "label": "Host memory per server (filled with proof of space)", "unit": "GB", "min": 1, "max": 10000, "log": True, "zero": True},
    {"key": "tau", "label": "Certified timing channel per server (the network appendix's program format: about 860)", "unit": "bit/s", "min": 1, "max": 1e5, "log": True, "zero": True},
    {"key": "export", "label": "Export per server: committed values readable outside the declared computation (today 4.4e9; under the export rule about 2.5e6)", "unit": "bit/s", "min": 1, "max": 1e12, "log": True, "zero": True},
    {"key": "nic", "label": "Line rate per server, without network accounting", "unit": "bit/s", "min": 1e9, "max": 1e13, "log": True},
    {"key": "share", "label": "Sampled proofs: compute share", "unit": "", "min": 1e-4, "max": 0.5, "log": True},
    {"key": "overhead", "label": "Sampled proofs: proving cost per unit of step work (sparse draws today 4.8e12; 512 per statement 1.7e10; batched throughput 1.6e8)", "unit": "x", "min": 1, "max": 1e14, "log": True},
    {"key": "replay", "label": "Sampled proofs: replay cost per drawn step", "unit": "x", "min": 1, "max": 1e7, "log": True},
    {"key": "step_flop", "label": "Sampled proofs: work per step", "unit": "FLOP", "min": 1e3, "max": 1e15, "log": True},
    {"key": "p_evade", "label": "Sampled proofs: certificate failure probability delta (planner 2^-40)", "unit": "", "min": 1e-15, "max": 0.99, "log": True},
    {"key": "luck", "label": "Sampled proofs: luck factor F/(1-p) (spec sheet 8)", "unit": "x", "min": 1, "max": 100, "log": True},
    {"key": "step_bits", "label": "Sampled proofs: output bits per step (16, or one 32-bit gate)", "unit": "bit", "min": 1, "max": 64, "log": True},
    {"key": "intake_x", "label": "Declared novel intake, as a multiple of alpha F (1 once the count check is in the ZK flow; line rate is about 1.3e6)", "unit": "x", "min": 1, "max": 1e7, "log": True},
    {"key": "m", "label": "Recipe multiplier m: frontier progress per counted multiplication of the developer's best recipe under the mechanisms, against the unconstrained recipe (central 1, adversarial 3)", "unit": "x", "min": 0.1, "max": 10, "log": True},
    {"key": "mfu", "label": "Model-FLOP utilization of an unconstrained frontier run (Llama 3 405B: 38-43%)", "unit": "", "min": 0.1, "max": 1, "log": False},
    {"key": "lag_s", "label": "Verdict lag: from a first faulty proof block to the halt, a window plus the verdict delay (1 hour assumed)", "unit": "s", "min": 1, "max": 1e7, "log": True, "zero": True},
    {"key": "identities", "label": "Sampled proofs: identities a detection doesn't end", "unit": "", "min": 1, "max": 1e6, "log": True},
    {"key": "floor_steps", "label": "Sampled proofs: steps in a class drawn at a floor of one", "unit": "", "min": 1, "max": 1e17, "log": True},
    {"key": "K", "label": "NCI: novel bits per inference block", "unit": "bit", "min": 1e5, "max": 1e10, "log": True},
    {"key": "Gamma", "label": "NCI: gates per inference block", "unit": "", "min": 1e14, "max": 1e19, "log": True},
    {"key": "d", "label": "Model: entry fee d (thread reads below which a value's fan-out is at most e)", "unit": "", "min": 1, "max": 65536, "log": True},
    {"key": "e", "label": "Model: multiplications per activation value", "unit": "", "min": 1e3, "max": 1e6, "log": True},
    {"key": "kappa", "label": "Model: kappa, twice the counted multiplications over the training circuit's gates (Kimi K2's pretraining schedule 0.60; adversarial 0.5; kappa.py)", "unit": "", "min": 0.05, "max": 1, "log": False},
    {"key": "kappa_chain", "label": "Model: kappa on the proved chain, one forward path (Kimi K2 0.0695)", "unit": "", "min": 0.01, "max": 1, "log": True},
    {"key": "d_chain", "label": "Model: chain's smallest inner dimension", "unit": "", "min": 64, "max": 65536, "log": True},
    {"key": "e_chain", "label": "Model: chain's largest fan-out", "unit": "", "min": 1e3, "max": 1e6, "log": True},
    {"key": "b_p", "label": "Model: partial-sum width a relay op reads", "unit": "bit", "min": 4, "max": 64, "log": True},
    {"key": "nodes", "label": "Fleet: servers", "unit": "", "min": 1e2, "max": 1e7, "log": True},
    {"key": "node_flops", "label": "Fleet: server compute (8 RTX PRO 6000s, FP8 dense)", "unit": "FLOP/s", "min": 1e15, "max": 1e17, "log": True},
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
    ("Declared computation, thread bound", "ThreadedThrottled is proved in Lean when the declared circuit contains the "
     "whole reference circuit; the extension to dark gates (T1') is being proved. Rests on WeightsIncompressible and the "
     "Threaded (d, e) premise, which a worker is proving for a defined reference circuit."),
    ("Declared computation, partial-sum law", "ContainedTraining is proved in Lean with no dark compute; the absorption "
     "budget (PS') is being proved. Needs a reference circuit with bounded-fan-in additions. Covers one forward chain."),
    ("Monitored compute residual", "PoUW's deficit epsilon_W = 0.74% is proved (CircuitWork and CircuitCredit, draft "
     "#2638) under named premises, for live coins, but it is a share of credited work. r is about 1% of device time only "
     "under the floor credit with verified filler (Decisions 89/90, Corollary D3; modelled, not served): today's served "
     "windows credit 10.3% of padded MACs. Host CPUs are neither credited nor bounded and count at their datasheet peak "
     "until a CPU filler or a physical cap exists (CMP-49)."),
    ("Memory cap", "Actual.Windows and ThreadedReads are proved; the composition and reads counted by information are "
     "being proved. Needs the declared computation's novel intake held to alpha F, which holds only once the per-window "
     "count check is a gated unit in the ZK flow (NET-35); until then intake is line rate (intake_x about 1.3e6) and memory "
     "caps nothing."),
    ("Wipe residual", "The slack is proved only in the ideal-permutation model, and the reference permutation is refuted "
     "(a developer passes from 1.53% of the store), so tier 3 rests on a falsified assumption today. The certificate's "
     "floor (0.47% at D = 2) can be freed at any k. Memory outside the fill is measured as gaps (on-chip 3.05 GB, GPU "
     "reserve 18.4 GB, host 29.5 GB). No theorem bounds help during a wipe; the helper term is an attack."),
    ("Exhaustion", "Lemma M is proved and locked per audit, and Held for snapshots; nothing bounds memory at every "
     "instant, which the memory cap needs."),
    ("Cheating share", "The cumulative bound is proved (Population.Exfil.tail_cum, landing); the deployed draw's pressure "
     "lemma, the per-block-cap thread theorem and a draw rate fixed against a public N are not. At a 1% total budget "
     "today's serve-time commitments alone exceed it, so share is sampling on top of them. The served planner draws "
     "zero-work templates at a floor of one per window (floor_steps)."),
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
    """PS': the most chain products inference blocks reading ``R`` bits compute when dark gates absorb ``Z`` bits of their
    partial sums."""
    return min((R + (1 - mu) * Z) / rate for mu, rate in rates)


def full_block_bits(p: dict) -> float:
    """The reads at which one inference block's thread bound reaches all its work, ``kappa Gamma / 2`` counted multiplications."""
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


def slack(p: dict) -> float:
    """The share of filled memory a proof-of-space certificate leaves unaccounted: rho_PoS^k <= delta certifies rho_PoS,
    and the certificate's floor (blocks rebuildable from public data within the deadline) can be freed at any k."""
    return 1 - p["pous_delta"] ** (1 / p["pous_rounds"]) + p["pous_floor"]


def ingress(p: dict, mech: dict) -> float:
    return p["sigma"] + p["tau"] + p["intake_x"] * p["K"] / p["Gamma"] * p["node_flops"] if mech["net"] else p["nic"]


def residual_share(p: dict) -> float:
    """r: the GPUs' uncredited share, their capacity above the credited clock, and the host CPUs, which count toward it
    until something credits or bounds them."""
    return min(1.0, p["residual"] + p["host_share"] + p["clock_share"])


def state_threshold(p: dict, mech: dict) -> float:
    """The server state, in bits, below which memory caps the residual (the window bound's quadratic regime)."""
    return residual_share(p) * p["node_flops"] * p["kappa"] * p["d"] * p["b_w"] * p["b_a"] / (4 * ingress(p, mech))


def evaluate(p: dict, mech: dict) -> dict:
    gpm = 2 / p["kappa"]
    H = p["years"] * YEAR
    C = p["nodes"] * p["node_flops"] * H
    s = min(p["unmonitored"], 0.999)
    U = C * s / (1 - s)
    alpha = p["K"] / p["Gamma"]
    rho_T = p["b_a"] / (alpha * lam(p, p["K"]))
    cap_T = C / rho_T

    k = bits = cheat = 0.0
    skipped = min(C, p["identities"] * p["lag_s"] * p["nodes"] * p["node_flops"])
    if mech["sampled"]:
        L = math.log(1 / p["p_evade"])
        q = p["share"] / (p["overhead"] + p["replay"])
        k = p["luck"] * p["identities"] * L / q
        skipped += k * p["step_flop"]
        k_all = k + p["luck"] * p["identities"] * L * p["floor_steps"]
        S_full = full_block_bits(p)
        per_block = max(1.0, min(k_all, S_full / 28))
        b_bar = max(p["step_bits"], p["step_bits"] + math.log2(math.e * (p["Gamma"] / p["step_flop"]) / per_block))
        bits = k_all * b_bar
        slope = (p["kappa"] * p["Gamma"] / 2 - region_cap(p, p["K"])) / (S_full - p["K"]) if S_full > p["K"] else 0.0
        cheat = min(p["m"] * slope * bits, C / gpm)

    res_ops = residual_share(p) * C + skipped
    sigma_eff = p["sigma"] if mech["net"] else p["nic"]
    link = p["sigma"] + p["tau"] + p["export"] if mech["net"] else p["nic"]
    B_max = 2 * p["b_p"] * res_ops + min(2 * p["b_p"] * U, p["nodes"] * H * link)

    rates = chain_rates(p)
    share = p["kappa"] / p["kappa_chain"]

    def declared_at(B: float) -> float:
        return min(cap_T, share * chain_products(rates, alpha * C, B))

    filled = (p["hbm_gb"] + p["host_gb"]) * GB
    mem = filled + p["disk_gb"] * GB
    rate_in = ingress(p, mech)
    help_bits = min(1.0, max(0.0, sigma_eff * p["t_read"] - BLOCK_NAME_BITS) / BLOCK_BITS) * filled
    sl = slack(p)
    wipe_state = (p["unfilled_gb"] + p["disk_gb"] + sl * (p["hbm_gb"] + p["host_gb"])) * GB + help_bits
    envelope = ((p["unfilled_gb"] + p["disk_gb"] + p["hbm_gb"] * (1 - p["f"] * (1 - sl)) + p["host_gb"] * sl) * GB
                + sigma_eff * T_AUDIT)
    per_server = math.inf
    if mech["wipes"]:
        per_server = min(per_server, windows_bound(p, H, min(wipe_state, mem), rate_in, p["wipe_s"]))
    if mech["exhaust"]:
        per_server = min(per_server, windows_bound(p, H, min(envelope, mem), rate_in, None))
    if not (mech["wipes"] or mech["exhaust"]):
        per_server = windows_bound(p, H, mem, rate_in, None)
    region = p["nodes"] * per_server

    def outcome(B: float) -> tuple[float, float, float, float]:
        declared = declared_at(B)
        relay = B / (2 * p["b_p"])
        relay_res = min(res_ops, relay)
        res_mults = (res_ops - relay_res) / gpm
        unmon = max(0.0, U - (relay - relay_res)) / gpm
        return min(p["m"] * declared + res_mults, p["m"] * region) + unmon, declared, res_mults, unmon

    # past the crossing the declared side is flat at the thread bound, so relaying more only costs
    B_hi = B_max
    if declared_at(B_max) >= cap_T:
        lo, hi = 0.0, B_max
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if declared_at(mid) >= cap_T else (mid, hi)
        B_hi = hi
    # the adversary relays only what pays; PS' is a min of functions affine in the absorbed bits, so its total is
    # concave in B and a ternary search finds the best B
    lo, hi = 0.0, B_hi
    for _ in range(100):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        lo, hi = (m1, hi) if outcome(m1)[0] < outcome(m2)[0] else (lo, m2)
    B = max((0.0, (lo + hi) / 2, B_hi), key=lambda b: outcome(b)[0])
    _, declared, res_mults, unmon = outcome(B)

    monitored = min(p["m"] * declared + res_mults, p["m"] * region)
    declared_part = min(p["m"] * declared, monitored)
    res_part = monitored - declared_part
    spent = C + U
    amp = p["g"] * gpm / spent
    terms = {"unmonitored": amp * unmon, "residual": amp * res_part, "declared": amp * declared_part, "cheat": amp * cheat}
    capped = p["m"] * region < p["m"] * declared + res_mults
    largest = max(terms, key=lambda t: terms[t])
    if largest == "declared":
        largest = "declared (thread bound)" if declared >= cap_T * (1 - 1e-9) else "declared (partial-sum law)"
    inverse = terms["unmonitored"] / p["g"] + (terms["residual"] + terms["declared"] + terms["cheat"]) / p["g"] / p["mfu"]
    return {
        "efficiency": sum(terms.values()), **terms, "largest": largest, "memory_caps_residual": capped,
        "dampener": max(1.0, 1 / inverse) if inverse > 0 else math.inf,
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
