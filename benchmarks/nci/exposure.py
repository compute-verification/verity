"""What compute efficiency eta (``tiers.py``) is most exposed to: a plausible range for every parameter, eta and 1/eta
across it at each tier, elasticities d ln eta / d ln p in four scenarios, the values at which 1/eta falls below 100, 10
and 2, and the closed forms of the memory cap mu, the inference-only policy's term nu and the cheating share chi checked against
``tiers.evaluate``. 1/eta is not the slowdown D (``statement.dampener``, the slowdown calculator's
``rho.theorem.multiple_bound``): eta counts every term at peak, D the residual r at MFU_0 (reference efficiency), so at
s = 0, g = 1 and no memory cap, D = 1 / (r + (eta - r) / MFU_0) wherever D > 1. ``statement.py`` gives the thresholds
on D.

    python benchmarks/nci/exposure.py           # terms, tornado, elasticities, thresholds and closed-form checks
    python benchmarks/nci/exposure.py --json    # the same, for renderers
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tiers as T  # noqa: E402

#: a plausible (low, high) for every parameter and where it comes from; the central value is the default. "guess" marks
#: a bound no parameter file, calibration or measurement gives
RANGES = {
    "unmonitored": (0.0, 0.1, "guess: none, to a tenth of the fleet's compute hidden"),
    "residual": (0.01, 1.0, "best plausible design 0.01 (eight conditions); floor credit 0.323 at the locked clock, 0.213 attested (C7 dossier, its eps left to eps_w); nothing credited today 1"),
    "eps_w": (6e-4, 0.00917, "best plausible design 0.06% (C7 dossier); CircuitCredit's closed 0.744%, 0.681-0.917% over r = 1-4 (draft #2638)"),
    "host_share": (0.0, 0.14, "without the hosts assumption (hosts carry no compute sets it to 0), against the card table's FP8 peak: two EPYC 9015s without matrix units 0.22%; Xeon 6776P INT8 AMX 12.6%, 14.0% at fused maximum clocks"),
    "eps_agg": (1e-6, 1e-3, "1e-6 needs per-window failure about 3 x 2^-47 for the same delta; 1e-3 lets delta fall tenfold"),
    "clock_share": (0.0, 0.2, "0: PoUW credits the card table's 2,430 MHz, the SKU's maximum (CMP-165, run r20261027-032043-e2d1); 0.2 the C4 dossier's adversarial phi = 1.25, a card whose SKU or firmware reaches past its table (unattested)"),
    "pous_rounds": (20.0, 74000.0, "low a guess; deployed 111; 74,000 with streamed labels (C8 dossier)"),
    "unfilled_gb": (3.05, 150.0, "on-chip alone 3.05; measured gaps 51 plus 7 of DPU, BMC and flash; high a guess"),
    "sigma": (10.0, 1e6, "guess: a shielded rack, to a loud channel"),
    "b_w": (0.8, 4.5, "b_w.pretrained/frontier-lossy: best MoE result, central, standard practice"),
    "b_a": (1.0, 8.0, "b_a.activation/boundary-training: 1, 4, 8"),
    "f": (0.5, 0.99, "guess"),
    "pous_delta": (1e-6, 0.1, "guess, around today's 1/100"),
    "pous_floor": (0.0, 0.985, "0 when the slack counts it (MEM-51); 0.985 under the broken reference permutation (C8)"),
    "disk_gb": (0.0, 8000.0, "unplugged; 8 TB of NVMe left in place (dark-compute red team)"),
    "t_read": (1e-4, 1e-2, "guess, around today's 1 ms"),
    "wipe_s": (60.0, 6.048e5, "guess: every minute to weekly"),
    "hbm_gb": (640.0, 2304.0, "8 x 80 GB (H100) to 8 x 288 GB (B300); 8 x 96 GB today"),
    "host_gb": (512.0, 4000.0, "guess, around vy-nebius-2's 1.9 TB"),
    "tau": (10.0, 1e4, "about 860 from the network appendix's program format; the ends a guess"),
    "export": (2.5e6, 4.4e10, "under the export rule 2.5e6; today 4.4e9; high a guess (10x)"),
    "nic": (8e11, 6.4e12, "2 x 400 Gb/s to 8 x 800 Gb/s; 8 x 400 Gb/s today"),
    "share": (1e-3, 0.1, "guess, around the 1% budget"),
    "overhead": (1.6e8, 1e13, "batched throughput 1.6e8; sparse draws today 4.8e12; high a guess"),
    "replay": (1e3, 1e6, "guess"),
    "step_flop": (1e3, 1e6, "guess"),
    "p_evade": (1e-15, 2.0 ** -20, "guess, around the planner's 2^-40"),
    "luck": (1.0, 32.0, "no luck; spec sheet 8; high a guess"),
    "step_bits": (8.0, 32.0, "low a guess; 16; one 32-bit gate"),
    "intake_x": (1.0, 1.3e6, "1 once the count check is in the ZK flow; line rate until then"),
    "identities": (1.0, 1e3, "guess"),
    "lag_s": (60.0, 86400.0, "guess: a minute's verdict to a day; one hour assumed"),
    "m": (1.0, 3.0, "C2 dossier: central 1 (NVFP4 0.75 x distillation 1.3), adversarial 3"),
    "mfu": (0.3, 0.6, "Llama 3 405B 38-43%, PaLM 46%; the ends a guess"),
    "g": (1.0, 10.0, "1 against the best known recipe; 10 the yardstick's central algorithmic progress (README)"),
    "floor_steps": (1.0, 1e12, "guess"),
    "K": (2.2e6, 9.1e7, "calibrations kimi-k2-1m-coded, kimi-k2-1m-raw, kimi-k2-11mb-raw"),
    "Gamma": (6.4e15, 2.9e17, "low a guess (100k-token sessions); kimi-k2-1m-raw; kimi-k2-11mb-raw"),
    "d": (1.0, 17.0, "a run that trains its embedding; ReferenceThreaded gives 10 for Kimi K2 at S = 2e7 under the "
                     "open conjecture ReferenceLayered; a Dot16 step's 17 legs, no lookup read"),
    "e": (12288.0, 1.65e5, "low a guess (the chain's fan-out); Kimi K2 4e4; dense-405b"),
    "kappa": (0.5, 0.667, "kappa.py: adversarial 0.5 (DeepSeek-V3's or Qwen3's shape with a fifth of its tokens at "
                          "32k, 0.49); Kimi K2's schedule 0.60; 4 of 6, the ceiling with no attention"),
    "thread_bits": (6.96e6, 2.94e9, "ThreadedOfLayered at 2.5-bit embedding columns for any reference circuit C1 "
                                    "allows; the whole embedding table at 2.5 bits, under a refined layered theorem"),
    "block_mults": (0.3, 1.0, "kappa / 2 if each block's counted share were the recipe's; 1, every gate countable"),
    "kappa_chain": (0.0, 0.3, "0 while the partial-sum law's support premises are falsified (NCI-FALSIFY); Kimi K2's dense path 7%; 0.3 a guess"),
    "d_chain": (512.0, 7168.0, "guess; Kimi K2 1536; d_model 7168"),
    "e_chain": (4096.0, 49152.0, "guess"),
    "b_p": (8.25, 64.0, "b_a.activation/tp-partial-sums-training central; 32; high a guess"),
    "nodes": (1e4, 1e6, "guess, around 100,000 servers"),
    "node_flops": (4e15, 3.6e16, "low a guess; 8 RTX PRO 6000s; 8 B200s, FP8 dense"),
    "years": (0.25, 3.0, "guess"),
}

#: the modelled design, where the residual sits below the memory cap
DESIGN = {"residual": 0.01, "host_share": 0.05, "clock_share": 0.0}
SCENARIOS = {
    "defaults": {},
    "design": DESIGN,
    "host CPUs bounded": DESIGN | {"host_share": 0.0},
    "served today": {"residual": 0.9},
    "served today, R_wipe 4 GB": {"residual": 0.9, "pous_rounds": 12000.0, "unfilled_gb": 3.05, "pous_floor": 0.0},
}
#: thresholds are found at the defaults and with the GPUs' residual as served today, at these tiers
THRESHOLD_SCENARIOS = ("defaults", "served today")
THRESHOLD_TIERS = ("T1", "T4")
LEVELS = (100.0, 10.0, 2.0)
#: the log step of the central differences: wide enough to average over the jumps a whole number of windows per horizon
#: makes in wipe_s and years, and on smooth parameters equal to a 0.2% step
STEP = 0.05
GRID = 33
BISECT = 16

SPECS = {s["key"]: s for s in T.SPEC}

SERVED_4GB = SCENARIOS["served today, R_wipe 4 GB"]
WIPE_2GB = {"pous_rounds": 74000.0, "pous_floor": 0.0, "unfilled_gb": 1.83}
#: the threading ReferenceThreaded gives under its conjecture, at which the closed-form memory cap binds
KEYED = {"d": T.REFERENCE_THREADED[0]}
#: points where a closed form is checked against evaluate(): the probe, its overrides, the tier, the term (eta against
#: the bound, mu against the monitored part when memory caps it, nu against the declared part, chi against the cheating
#: share) and the regime; a regime that starts with "breaks" is one the closed form doesn't cover
CHECKS = [
    ("defaults", {}, "T1", "eta", "the residual plus the thread bound"),
    ("1 GB streamed-label wipes", WIPE_2GB | KEYED | {"unfilled_gb": 0.83}, "T4", "mu",
     "the floor credit's residual is above the cap"),
    ("design", DESIGN, "T4", "eta", "memory caps nothing: mu > r"),
    ("host CPUs bounded", DESIGN | {"host_share": 0.0}, "T5", "eta", "r = 1%, chi on top"),
    ("unmonitored 0.1", DESIGN | {"unmonitored": 0.1}, "T4", "eta", "s of spent FLOP, the rest scaled by 1 - s"),
    ("intake at line rate", {"residual": 0.9, "intake_x": 1.3e6}, "T4", "eta", "no count check: mu >> 1 caps nothing"),
    ("served, R_wipe 4 GB", SERVED_4GB | KEYED, "T3", "mu", "memory caps the residual"),
    ("served, R_wipe 4 GB", SERVED_4GB | KEYED, "T4", "mu", "hourly wipes inside a 3.6 h window"),
    ("side channels 3e5", WIPE_2GB | KEYED | {"residual": 0.9, "sigma": 3e5, "unfilled_gb": 0.83}, "T4", "mu",
     "helpers refill R_wipe; sigma joins lambda_in"),
    ("defaults", {}, "T1", "nu", "Lambda(K) = K / (2 d b_w)"),
    ("K 1e6", {"K": 1e6}, "T1", "nu", "Lambda(K) = e"),
    ("design", DESIGN, "T5", "chi", "(1, 0) above thread_bits: one block is full at S_1"),
    ("batched proving", DESIGN | {"overhead": 1.6e8}, "T5", "chi", "overhead 1.6e8"),
    ("1,000 servers", DESIGN | {"nodes": 1e3}, "T5", "chi", "the pool doesn't shrink with the fleet"),
    ("no dark compute", {"residual": 0.0, "host_share": 0.0, "clock_share": 0.0, "eps_w": 0.0, "eps_agg": 0.0, "lag_s": 0.0, "kappa_chain": 0.0695}, "T4", "nu", "breaks: the partial-sum law binds below nu"),
    ("served 4 GB, daily wipes", SERVED_4GB | {"wipe_s": 86400.0}, "T4", "mu",
     "breaks: wipe period above R_wipe / lambda_in"),
    ("served, R_wipe 22 MB", {"residual": 0.9, "unfilled_gb": 0.01, "pous_rounds": 1e6, "pous_floor": 0.0,
                              "wipe_s": 1.0, "d": 65536.0}, "T3", "mu",
     "breaks: R_wipe < d b_w e, the cap floors at nu"),
    ("whole embedding table", KEYED | {"thread_bits": 2.94e9}, "T5", "chi",
     "breaks: threading certified past S_1, the chord ends at thread_bits"),
    ("proofs nearly free", {"overhead": 1.0, "replay": 1.0}, "T5", "chi", "breaks: the pool can't fill one block"),
]


def eta(p: dict, tier: str) -> float:
    return T.evaluate(p, T.TIERS[tier])["efficiency"]


def intake(p: dict) -> float:
    """lambda_in under network accounting, bits/s: side channels, certified timing and the declared novel intake."""
    return p["sigma"] + p["tau"] + p["intake_x"] * p["K"] / p["Gamma"] * p["node_flops"]


def wipe_residual(p: dict) -> float:
    """R_wipe in bits: memory the fill doesn't reach, disks left in place, the proof of space's slack on the filled
    memory and what helpers refill within the read deadline."""
    filled = (p["hbm_gb"] + p["host_gb"]) * T.GB
    helped = min(1.0, max(0.0, p["sigma"] * p["t_read"] - T.BLOCK_NAME_BITS) / T.BLOCK_BITS)
    return (p["unfilled_gb"] + p["disk_gb"]) * T.GB + (T.slack(p) + helped) * filled


def envelope(p: dict) -> float:
    """Exhaustion's state in bits: memory outside the fill, HBM the declared computation doesn't hold, the slack, and
    what a helper adds within Lemma M's audit window."""
    sl = T.slack(p)
    gb = p["unfilled_gb"] + p["disk_gb"] + p["hbm_gb"] * (1 - p["f"] * (1 - sl)) + p["host_gb"] * sl
    return gb * T.GB + p["sigma"] * T.T_AUDIT


def mu(p: dict, exhaust: bool = False) -> float:
    """The memory cap, 4 R lambda_in / (kappa d b_w b_a F) with R = R_wipe (or the exhaustion envelope, if smaller):
    4 R_wipe alpha / (kappa d b_w b_a) at intake alpha F."""
    state = min(wipe_residual(p), envelope(p)) if exhaust else wipe_residual(p)
    return 4 * state * intake(p) / (p["kappa"] * p["d"] * p["b_w"] * p["b_a"] * p["node_flops"])


def nu(p: dict) -> float:
    """The inference-only policy's term, the declared computation at the thread bound: 2 alpha Lambda(K) / (kappa b_a)."""
    return 2 * p["K"] / p["Gamma"] * T.lam(p, p["K"]) / (p["kappa"] * p["b_a"])


def chi(p: dict) -> float:
    """The cheating share: a lifetime pool of I ln(1/delta) ((overhead + replay) / share + floor_steps) injecting proof
    blocks, each injecting step_bits bits worth (block_mults Gamma - K e / b_a) / (S_1 - K) counted multiplications,
    S_1 = sqrt(2 b_w b_a block_mults Gamma), times 2 m / kappa, over N F H."""
    top = p["block_mults"] * p["Gamma"]
    s_1 = math.sqrt(2 * p["b_w"] * p["b_a"] * top)
    pool = p["identities"] * math.log(1 / p["p_evade"]) * ((p["overhead"] + p["replay"]) / p["share"] + p["floor_steps"])
    worth = (top - p["K"] * p["e"] / p["b_a"]) / (s_1 - p["K"])
    C = p["nodes"] * p["node_flops"] * p["years"] * T.YEAR
    return 2 / p["kappa"] * p["m"] * pool * p["step_bits"] * worth / C


def bound(p: dict, tier: str) -> float:
    """s + (1 - s)(min(r + nu, mu) + chi), mu from T3 and chi at T5. The memory cap holds the declared computation too,
    so s + min(r, mu) + nu + chi exceeds it by at most nu."""
    mech = T.TIERS[tier]
    s = p["unmonitored"]
    cap = mu(p, mech["exhaust"]) if mech["wipes"] else math.inf
    return s + (1 - s) * (min(T.residual_share(p) + nu(p), cap) + (chi(p) if mech["sampled"] else 0.0))


def terms(p: dict) -> dict:
    lam_in = intake(p)
    return {"s": p["unmonitored"], "r": T.residual_share(p), "mu": mu(p), "mu_T4": mu(p, True), "nu": nu(p),
            "chi": chi(p), "R_wipe_gb": wipe_residual(p) / T.GB, "envelope_gb": envelope(p) / T.GB,
            "window_s": wipe_residual(p) / lam_in, "lambda_in": lam_in,
            "lambda_knee_bits": 2 * p["d"] * p["b_w"] * p["e"],
            **{f"eta_{t}": eta(p, t) for t in T.TIERS}}


def check(p: dict, tier: str) -> dict:
    """Each closed form beside what evaluate() gives for it, as ``(evaluate, closed)``."""
    r = T.evaluate(p, T.TIERS[tier])
    return {"eta": (r["efficiency"], bound(p, tier)), "nu": (r["declared"], nu(p)), "chi": (r["cheat"], chi(p)),
            "mu": (r["residual"] + r["declared"], mu(p, T.TIERS[tier]["exhaust"])), "capped": r["memory_caps_residual"]}


def elasticity(p: dict, key: str, tier: str, h: float = STEP) -> float | None:
    """d ln eta / d ln p by a central difference in ln p; None at a parameter of zero."""
    v = p[key]
    if v == 0:
        return None
    up, down = (eta(p | {key: v * math.exp(sign * h)}, tier) for sign in (1, -1))
    return (math.log(up) - math.log(down)) / (2 * h)


def grid(key: str, n: int = GRID) -> list[float]:
    s = SPECS[key]
    if s["log"]:
        inner = [s["min"] * (s["max"] / s["min"]) ** (i / (n - 1)) for i in range(1, n - 1)]
    else:
        inner = [s["min"] + (s["max"] - s["min"]) * i / (n - 1) for i in range(1, n - 1)]
    return [s["min"], *inner, s["max"]]


def inverse_eta(p: dict, tier: str) -> float:
    e = eta(p, tier)
    return 1 / e if e > 0 else math.inf


def thresholds(p: dict, key: str, tier: str, n: int = GRID, steps: int = BISECT, levels: tuple[float, ...] = LEVELS,
               size: Callable[[dict, str], float] = inverse_eta) -> dict[float, list[list[float]]]:
    """For each level, the intervals of the parameter's slider range on which ``size`` (1/eta, or tiers' slowdown D) is
    below it, the others at ``p``."""
    xs = grid(key, n)
    ys = [size(p | {key: x}, tier) for x in xs]
    mid = (lambda a, b: math.sqrt(a * b)) if SPECS[key]["log"] else (lambda a, b: (a + b) / 2)
    out = {}
    for level in levels:
        inside = [y < level for y in ys]
        edges = []
        for i in range(n - 1):
            if inside[i] != inside[i + 1]:
                a, b = xs[i], xs[i + 1]
                for _ in range(steps):
                    m = mid(a, b)
                    a, b = (m, b) if (size(p | {key: m}, tier) < level) == inside[i] else (a, m)
                edges.append(mid(a, b))
        bounds = [xs[0], *edges, xs[-1]]
        out[level] = [bounds[j:j + 2] for j in range(len(bounds) - 1) if inside[0] != (j % 2 == 1)]
    return out


def profile(key: str) -> dict:
    """One parameter's range, its tornado and elasticities in every scenario and tier, and its thresholds."""
    low, high, source = RANGES[key]
    out = {"key": key, "low": low, "central": T.DEFAULTS[key], "high": high, "source": source,
           "eta": {}, "elasticity": {}, "thresholds": {}}
    for name, over in SCENARIOS.items():
        base = T.DEFAULTS | over
        out["eta"][name] = {t: {at: eta(base | {key: v}, t) for at, v in (("low", low), ("central", base[key]),
                                                                          ("high", high))} for t in T.TIERS}
        out["elasticity"][name] = {t: elasticity(base, key, t) for t in T.TIERS}
    for name in THRESHOLD_SCENARIOS:
        base = T.DEFAULTS | SCENARIOS[name]
        out["thresholds"][name] = {t: thresholds(base, key, t) for t in THRESHOLD_TIERS}
    return out


def swing(row: dict, scenario: str, tier: str) -> float:
    e = row["eta"][scenario][tier].values()
    return math.log(max(e) / min(e))


def rank(rows: list[dict], scenario: str, tiers: tuple[str, ...] = ("T1", "T4")) -> list[dict]:
    return sorted(rows, key=lambda r: -max(swing(r, scenario, t) for t in tiers))


def describe(intervals: list[list[float]], key: str, show: Callable[[float], str] = lambda x: f"{x:.3g}") -> str:
    lo, hi = SPECS[key]["min"], SPECS[key]["max"]
    if not intervals:
        return "never"
    if intervals == [[lo, hi]]:
        return "always"
    parts = []
    for a, b in intervals:
        parts.append(f"< {show(b)}" if a == lo else f"> {show(a)}" if b == hi else f"{show(a)}-{show(b)}")
    return ", ".join(parts)


def g(x: float | None) -> str:
    return "-" if x is None else f"{x:.3g}"


def report(rows: list[dict], checks: list[dict], scenario_terms: dict) -> None:
    print("Terms per scenario (mu at T3, mu_T4 with exhaustion; window = R_wipe / lambda_in)")
    print(f"{'scenario':27} {'r':>6} {'mu':>7} {'mu_T4':>7} {'nu':>8} {'chi':>8} {'R_wipe GB':>9} {'window h':>8} "
          f"{'eta T1':>7} {'eta T4':>7} {'eta T5':>7} {'1/eta T5':>8}")
    for name, t in scenario_terms.items():
        print(f"{name:27} {t['r']:6.3g} {t['mu']:7.3g} {t['mu_T4']:7.3g} {t['nu']:8.2g} {t['chi']:8.2g} "
              f"{t['R_wipe_gb']:9.4g} {t['window_s'] / 3600:8.3g} {t['eta_T1']:7.3g} {t['eta_T4']:7.3g} "
              f"{t['eta_T5']:7.3g} {1 / t['eta_T5']:8.3g}")
    for scenario in THRESHOLD_SCENARIOS:
        print(f"\nTornado, {scenario}: eta at low / central / high, ranked by the larger log swing at T1 or T4; "
              "elasticity at the default")
        print(f"{'parameter':12} {'low':>8} {'central':>8} {'high':>8}  {'T1 low':>7} {'central':>7} {'high':>7}  "
              f"{'T4 low':>7} {'central':>7} {'high':>7}  {'1/eta at T4':>13} {'el T1':>6} {'el T4':>6}")
        base = T.DEFAULTS | SCENARIOS[scenario]
        for r in rank(rows, scenario):
            e1, e4 = r["eta"][scenario]["T1"], r["eta"][scenario]["T4"]
            d4 = sorted(1 / v for v in e4.values())
            el = r["elasticity"][scenario]
            print(f"{r['key']:12} {r['low']:8.3g} {base[r['key']]:8.3g} {r['high']:8.3g}  "
                  f"{e1['low']:7.3g} {e1['central']:7.3g} {e1['high']:7.3g}  "
                  f"{e4['low']:7.3g} {e4['central']:7.3g} {e4['high']:7.3g}  "
                  f"{d4[0]:6.3g}-{d4[-1]:<6.3g} {g(el['T1']):>6} {g(el['T4']):>6}")
    print("\nRanges")
    for r in rows:
        print(f"  {r['key']:12} {r['low']:9.3g} {r['central']:9.3g} {r['high']:9.3g}  {r['source']}")
    names = list(SCENARIOS)
    print("\nElasticities d ln eta / d ln p (rows with any |e| >= 0.001; '-' at a parameter of zero)")
    print(f"{'parameter':12}" + "".join(f" {n[:16]:>20}" for n in names))
    print(f"{'':12}" + " ".join(f"{'T1':>6} {'T4':>6} {'T5':>6}" for _ in names))
    for r in rows:
        vals = [r["elasticity"][n][t] for n in names for t in ("T1", "T4", "T5")]
        if any(v is not None and abs(v) >= 1e-3 for v in vals):
            print(f"{r['key']:12} " + " ".join(f"{g(v):>6}" for v in vals))
    for scenario in THRESHOLD_SCENARIOS:
        for tier in THRESHOLD_TIERS:
            print(f"\nThresholds, {scenario}, {tier}: where 1/eta < 100, 10, 2 over the slider range, the others as "
                  "in the scenario")
            flat = {}
            for r in rows:
                cells = [describe(r["thresholds"][scenario][tier][level], r["key"]) for level in LEVELS]
                if all(c in ("always", "never") for c in cells):
                    flat.setdefault(" / ".join(cells), []).append(r["key"])
                else:
                    print(f"  {r['key']:12} " + " | ".join(f"{c:>22}" for c in cells))
            for cells, keys in flat.items():
                print(f"  no crossing ({cells}): {', '.join(keys)}")
    print("\nClosed forms against evaluate(): eta = s + (1 - s)(min(r + nu, mu) + chi), "
          "mu = 4 R lambda_in / (kappa d b_w b_a F),\nnu = 2 alpha Lambda(K) / (kappa b_a), "
          "chi = 2 m pool step_bits (block_mults Gamma - K e / b_a) / (kappa (S_1 - K) N F H)")
    print(f"{'probe':26} {'tier':4} {'term':4} {'evaluate':>9} {'closed':>9} {'ratio':>6}  regime")
    for c in checks:
        got, closed = c[c["term"]]
        print(f"{c['probe']:26} {c['tier']:4} {c['term']:4} {got:9.3g} {closed:9.3g} {got / closed:6.3f}  "
              f"{c['regime']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="ranges, scenarios, profiles and checks as JSON")
    args = ap.parse_args(argv)
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(profile, list(T.DEFAULTS)))
    checks = [{"probe": name, "tier": tier, "term": term, "regime": regime, **check(T.DEFAULTS | over, tier)}
              for name, over, tier, term, regime in CHECKS]
    scenario_terms = {name: terms(T.DEFAULTS | over) for name, over in SCENARIOS.items()}
    if args.json:
        tornado = {name: [r["key"] for r in rank(rows, name)] for name in SCENARIOS}
        print(json.dumps({"scenarios": SCENARIOS, "levels": LEVELS, "terms": scenario_terms, "tornado": tornado,
                          "parameters": rows, "checks": checks}, indent=1))
        return 0
    report(rows, checks, scenario_terms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
