"""Every number the Notion page "The compute dampener: claims, model and tiers" quotes from ``tiers.py``, ``exposure.py``,
``kappa.py`` and ``fit.py``, at the commit this runs on, grouped as the page is: the slide's questions, the symbols,
section 4's terms, the ladder and the fit bound, years of growth, the regimes, section 6's thresholds and the known holes'
cheating share. D is the overhead calculator's hardware multiple at MFU_0, ``rho.theorem.multiple_bound`` in Lean's
form (``Nci.Dampener.remoteDampener``) on ``tiers.evaluate(p, TIERS[t], p["mfu"])``'s terms, the residual at MFU_0 under
the reference-efficiency assumption; every scenario is a set of overrides on
``tiers.DEFAULTS``. D by the fit bound (``fit_dampener``) is the same with the fit bound in place of the
closed-form memory cap, which the page's memory rungs cite.

    python benchmarks/nci/statement.py           # Markdown
    python benchmarks/nci/statement.py --json    # the same numbers, for renderers
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "rho"))

import exposure as X  # noqa: E402
import fit as F  # noqa: E402
import kappa as K  # noqa: E402
import tiers as T  # noqa: E402
from rho.theorem import multiple_bound  # noqa: E402

MB = 8e6
TODAY = {"residual": 1.0}
ATTESTED = {"residual": 0.213, "clock_share": 0.0}
#: the C7 dossier's eight conditions for a 1% residual include PoUW's deficit at most 0.25%
BEST = {"residual": 0.01, "clock_share": 0.0, "eps_w": 0.0025}
STREAMED = {"pous_rounds": 74000.0, "pous_floor": 0.0}


def streamed(gb: float, base: dict | None = None) -> dict:
    """``base`` with streamed-label wipes that leave ``gb`` GB: the memory the fill doesn't reach is ``gb`` less the
    slack."""
    over = (base or {}) | STREAMED
    return over | {"unfilled_gb": gb - X.wipe_residual(T.DEFAULTS | over | {"unfilled_gb": 0.0}) / T.GB}


FLOOR = streamed(4.0, BEST | {"residual": 0.0})
RUNGS = {
    "Today": TODAY,
    "PoUW's floor credit": {},
    "Floor credit and attested clock": ATTESTED,
    "Best plausible PoUW design": BEST,
    "Streamed-label wipes, 13 GB": streamed(13.0),
    "Streamed-label wipes, 6 GB": streamed(6.0),
    "Streamed-label wipes, 3.05 GB (no power removal)": streamed(3.05),
    "Power removal, 1 GB": streamed(1.0),
    "Floor (r̂ = 0, 4 GB)": FLOOR,
    "Floor, one-minute verdict lag": FLOOR | {"lag_s": 60.0},
}
LADDER_TIERS = tuple(T.TIERS)
#: the fit bound's wipe residuals, GB: the streamed-label rungs, and 1 GB with the GPUs' on-chip 3.05 GB on top
FIT_WIPES = (13.0, 6.0, 4.05, 3.05, 1.0)
FIT_TARGET = ("besiroglu", "405b")
FIT_TIERS = ("T3", "T5")
#: frontier training compute grows 4-5x a year (Epoch AI)
GROWTH = 4.5
#: the s dossier's unmonitored shares, central and adversarial
REGIMES = {
    "Domestic oversight of 3-5 labs": (0.005, 0.05),
    "International compute registry": (0.02, 0.1),
    "US-China agreement at signing": (0.07, 0.3),
}
REGIME_TIERS = ("T3", "T5")
#: silent data corruptions a year across the fleet, each a detection
SDC = 2000.0
LEVELS = (1000.0, 100.0, 10.0, 2.0)
THRESHOLD_TIERS = ("T1", "T3", "T4", "T5")
#: parameters the page's "no other parameter" sets aside: the parts of r̂ (the GPUs, the clock, host CPUs and the verdict
#: lag J Delta / H), s and g
TORNADO_SKIP = ("residual", "host_share", "clock_share", "lag_s", "identities", "unmonitored", "g")
COSTS = {"today's sparse draws": T.DEFAULTS["overhead"], "512 draws per statement": 1.7e10}


def same(p: dict, x: float) -> float:
    return x


def alpha(p: dict, bits: float) -> float:
    return bits / p["Gamma"]


def r_wipe_gb(p: dict, unfilled: float) -> float:
    return X.wipe_residual(p | {"unfilled_gb": unfilled}) / T.GB


#: section 6's parameters: the key searched, its overrides on the scenario, and each view of an edge
THRESHOLD_ROWS: list[tuple[str, dict, list[tuple[str, Callable[[dict, float], float]]]]] = [
    ("residual", {"clock_share": 0.0}, [("r̂ (the GPU residual, the clock at 0)", same)]),
    ("disk_gb", {}, [("unwiped disk, GB", same)]),
    ("intake_x", {}, [("declared intake, multiple of αF", same)]),
    ("sigma", {}, [("side channels σ, bit/s", same)]),
    ("K", {}, [("K, bits", same), ("α = K/Γ", alpha)]),
    ("b_a", {}, [("b_a, bits", same)]),
    ("d", {}, [("d", same)]),
    ("unmonitored", {}, [("unmonitored share s", same)]),
    ("unfilled_gb", STREAMED, [("R_wipe, GB (streamed-label wipes, unfilled_gb varied)", r_wipe_gb)]),
]


def num(x: float, digits: int = 3) -> str:
    """``x`` to ``digits`` significant figures, a whole number below 100,000 exactly: plain from 0.001 to 100,000,
    otherwise as a power of ten."""
    if x == 0 or not math.isfinite(x):
        return "0" if x == 0 else "∞"
    if abs(x) < 1e5 and x == int(x):
        return f"{int(x):,}"
    if 1e-3 <= abs(x) < 1e5:
        return f"{round(x, digits - 1 - math.floor(math.log10(abs(x)))):,}".removesuffix(".0")
    mantissa, exponent = f"{x:.{digits - 1}e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def dampener(p: dict, tier: str) -> float:
    return multiple_bound(p["mfu"], *T.remote_terms(p, T.evaluate(p, T.TIERS[tier], p["mfu"])), reference_efficiency=False)


def cheat(p: dict) -> float:
    return T.evaluate(p, T.TIERS["T5"])["cheat"]


def capacity(p: dict) -> float:
    return p["nodes"] * p["node_flops"] * p["years"] * T.YEAR


def fit_mu(gb: float) -> float:
    """The fit bound's cap on monitored efficiency at a wipe residual of ``gb`` GB, for a ``FIT_TARGET``-class target at
    central w_e; ``inf`` where it doesn't hold."""
    law, target = FIT_TARGET
    L = F.target_loss(law, target)
    return F.row(gb, F.n_star(law, L), L)["mu_fit"]


def capped(p: dict, tier: str, mu: float) -> float:
    """D at ``tier`` with ``mu`` as the memory cap in place of the closed form: ``tiers.evaluate`` without wipes or
    exhaustion, its monitored terms capped at ``m mu`` of C (the recipe multiplier scales the cap, as there)."""
    r = T.evaluate(p, T.TIERS[tier] | {"wipes": False, "exhaust": False}, p["mfu"])
    monitored = min(r["residual"] + r["declared"], p["g"] * (1 - min(p["unmonitored"], 0.999)) * p["m"] * mu)
    return multiple_bound(p["mfu"], *T.remote_terms(p, r | {"residual": monitored, "declared": 0.0}),
                          reference_efficiency=False)


def fit_dampener(p: dict, tier: str) -> float:
    """D at ``tier`` by the fit bound at ``p``'s wipe residual, from T3 on."""
    return capped(p, tier, fit_mu(X.wipe_residual(p) / T.GB) if T.TIERS[tier]["wipes"] else math.inf)


def bisect(f: Callable[[float], bool], lo: float, hi: float, steps: int = 60) -> float:
    """The edge in ``[lo, hi]`` where ``f`` turns true, by geometric bisection; ``f(lo)`` false and ``f(hi)`` true."""
    for _ in range(steps):
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if f(mid) else (mid, hi)
    return math.sqrt(lo * hi)


def commit() -> tuple[str, bool]:
    """The commit this runs at, and whether ``benchmarks/nci`` has changes outside it."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True).stdout.strip()
    try:
        return git("rev-parse", "HEAD") or "unknown", bool(git("status", "--porcelain", "--", "."))
    except OSError:
        return "unknown", False


def rows(pairs: list[tuple[str, float]]) -> list[dict]:
    return [{"quantity": q, "value": v} for q, v in pairs]


def slide() -> list[dict]:
    """The slide's hardest questions: SDCs forgiven fleet-wide, as identities a detection doesn't end, or per server."""
    p = T.DEFAULTS
    return rows([
        ("D at the best plausible PoUW design, T1", dampener(p | BEST, "T1")),
        (f"the same with {num(SDC)} SDCs forgiven fleet-wide (J = {num(SDC)})",
         dampener(p | BEST | {"identities": SDC}, "T1")),
        (f"r̂ added by forgiving each SDC on its own server for one verdict lag, {num(SDC)} Δ / (N H)",
         SDC * p["lag_s"] / (p["years"] * T.YEAR) / p["nodes"]),
    ])


def symbols() -> list[dict]:
    """Section 0's values, with those sections 1, 2 and 4 derive from them."""
    p = T.DEFAULTS
    H = p["years"] * T.YEAR
    C = capacity(p)
    r = T.audited_residual(p)
    sampled = T.evaluate(p, T.TIERS["T5"])
    chi = sampled["cheat"]
    R = X.wipe_residual(p)
    out = [
        ("N", "monitored servers", p["nodes"]),
        ("F", "a server's peak, FLOP/s", p["node_flops"]),
        ("H", "the horizon, years", p["years"]),
        ("C = NFH", "monitored capacity, FLOP", C),
        ("κ", "Kimi K2's pretraining schedule (kappa.py)", K.schedule(K.MODELS["kimi-k2"])[0]),
        ("κ", "the model's value", p["kappa"]),
        ("κ", "adversarial", X.RANGES["kappa"][0]),
        ("1/κ", "the factor s and r carry without C2⁺", 1 / p["kappa"]),
        ("MFU_0", "the unconstrained developer's model-FLOP utilization", p["mfu"]),
        ("m", "recipe multiplier, central", p["m"]),
        ("m", "recipe multiplier, adversarial", X.RANGES["m"][1]),
        ("r̂", "under the floor credit: GPU + clock (hosts carry no compute)", r),
        ("r̂ GPU", "the GPUs' residual at the locked clock", p["residual"]),
        ("r̂ clock", "GPU capacity above the credited clock", p["clock_share"]),
        ("ε_W", "PoUW's deficit, a share of certified work (the retired CircuitCredit's)", p["eps_w"]),
        ("ε_W W_cert/C", "the deficit's share of capacity under the floor credit, ε_W (1 − r̂)", p["eps_w"] * (1 - r)),
        ("ε_W", "at the best plausible PoUW design", (p | BEST)["eps_w"]),
        ("ε′", "aggregation slack (δ_win / ε′ = 2.7e-8 at δ_win = 3·2^-40)", p["eps_agg"]),
        ("r = r̂ + ε_W W_cert/C + ε′", "the residual DampenerRemote bounds (#3646), under the floor credit",
         T.residual_share(p)),
        ("GPU / r̂", "share of r̂", p["residual"] / r),
        ("clock / r̂", "share of r̂", p["clock_share"] / r),
        ("r̂", "at the best plausible PoUW design", T.audited_residual(p | BEST)),
        ("K", "novel bits per inference block", p["K"]),
        ("Γ", "gates per inference block", p["Gamma"]),
        ("α = K/Γ", "", p["K"] / p["Gamma"]),
        ("d", "entry fee", p["d"]),
        ("e", "fan-out", p["e"]),
        ("b_w", "bits per updated weight", p["b_w"]),
        ("b_a", "bits per activation", p["b_a"]),
        ("ν = 2αΛ(K)/(κ b_a)", "NCI's term", X.nu(p)),
        ("1000 ν", "NCI's margin at a factor of 1000", 1000 * X.nu(p)),
        ("λ_in = σ + τ + αF", "a server's novel intake, bit/s", X.intake(p)),
        ("αF", "declared intake, bit/s", p["K"] / p["Gamma"] * p["node_flops"]),
        ("σ", "side channels, bit/s", p["sigma"]),
        ("σ", "adversarial, bit/s", X.RANGES["sigma"][1]),
        ("τ", "certified timing, bit/s", p["tau"]),
        ("R_wipe", f"today's wipes, {p['pous_rounds']:.0f} rounds, if C8 held, GB", R / T.GB),
        ("slack / R_wipe", "the proof of space's slack, share of today's R_wipe",
         T.slack(p) * (p["hbm_gb"] + p["host_gb"]) * T.GB / R),
        ("μ̂", "at today's R_wipe", X.mu(p)),
        ("μ̂", "at today's audit with the slack pooled away (unfilled memory only)",
         X.mu(p | {"pous_rounds": 1e12, "pous_floor": 0.0})),
        ("μ̂", "at 13 GB", X.mu(p | streamed(13.0))),
        ("μ̂", "at 6 GB", X.mu(p | streamed(6.0))),
        ("μ̂", "at 1 GB", X.mu(p | streamed(1.0))),
        ("J", "identities a detection doesn't end", p["identities"]),
        ("Δ", "verdict lag, s", p["lag_s"]),
        ("Δ/H", "the verdict lag's share of capacity per identity", p["lag_s"] / H),
        ("A", "luck factor", p["luck"]),
        ("χ", "cheating share (T5)", chi),
        ("chord", "counted multiplications an injected bit buys (tiers.chord)", T.chord(p, sampled["cheat_bits"])),
        ("Pool", "χ C, FLOP-equivalent per identity", chi * C),
        ("d b_w e", "the wipe residual below which the cap floors at ν, MB", p["d"] * p["b_w"] * p["e"] / MB),
        ("2 d b_w e", "the K above which ν grows as K², MB", 2 * p["d"] * p["b_w"] * p["e"] / MB),
    ]
    return [{"symbol": s, "meaning": m, "value": v} for s, m, v in out]


def terms() -> list[dict]:
    """Section 4: side channels at sigma = alpha F against the closed form's doubling, where helpers refill R_wipe too."""
    p = T.DEFAULTS
    aF = p["K"] / p["Gamma"] * p["node_flops"]
    out = []
    for name, over in (("today's wipes", {}), ("6 GB streamed-label wipes", streamed(6.0))):
        q = p | over
        out += [(f"μ̂ at σ = αF over μ̂ at σ = {num(p['sigma'])} bit/s, {name}", X.mu(q | {"sigma": aF}) / X.mu(q)),
                (f"R_wipe at σ = αF, {name}, GB", X.wipe_residual(q | {"sigma": aF}) / T.GB)]
    return rows(out)


def years(D: dict) -> dict:
    return {t: math.log(d) / math.log(GROWTH) for t, d in D.items()}


def ladder() -> list[dict]:
    """Each rung's r̂, R_wipe, optimal window R_wipe / lambda_in, D per tier and by the fit bound, and D's years of
    growth."""
    out = []
    for name, over in RUNGS.items():
        p = T.DEFAULTS | over
        D = {t: dampener(p, t) for t in LADDER_TIERS}
        D_fit = {t: fit_dampener(p, t) for t in FIT_TIERS}
        out.append({"rung": name, "overrides": over, "r_hat": T.audited_residual(p),
                    "R_wipe_gb": X.wipe_residual(p) / T.GB, "window_min": X.wipe_residual(p) / X.intake(p) / 60,
                    "D": D, "D_fit": D_fit, "years": years(D), "years_fit": years(D_fit)})
    return out


def substitutes() -> list[dict]:
    """Section 5's substitutes: memory with no PoUW credit at all."""
    p = T.DEFAULTS | streamed(6.0, TODAY)
    return rows([("D with no PoUW credit (r̂ = 1) and 6 GB streamed-label wipes, T3, closed form", dampener(p, "T3")),
                 ("the same by the fit bound", fit_dampener(p, "T3"))])


def floor() -> list[dict]:
    """The floor's limits as the verdict lag goes to 0, with ε_W and ε′ (at r̂ = 0 all of C is certified, so the deficit
    is ε_W C) and as they go to 0 too, and what evaluate would give at zero lag, ε_W and ε′ if the partial-sum law held,
    where with no residual to relay through it binds below nu."""
    p = T.DEFAULTS
    eps_w = (p | FLOOR)["eps_w"]
    return rows([
        ("1 / (ε_W + ε′ + ν / MFU_0), the floor's limit as the verdict lag goes to 0",
         1 / (eps_w + p["eps_agg"] + X.nu(p) / p["mfu"])),
        ("MFU_0 / ν, the same as ε_W and ε′ go to 0 too (δ_win / ε′ then needs δ_win → 0)", p["mfu"] / X.nu(p)),
        ("D at zero lag, ε_W and ε′, r̂ = 0, T3, if the partial-sum law held (its support premises are falsified)",
         dampener(p | FLOOR | {"lag_s": 0.0, "eps_w": 0.0, "eps_agg": 0.0, "kappa_chain": 0.0695}, "T3")),
    ])


def fit_bound() -> dict:
    """The fit bound for a 405B-class target (``fit.row``) at ``FIT_WIPES``, with T5's cheating share on top; its
    premises' values; and the wipe residuals up to which it holds and beats the closed form."""
    law, target = FIT_TARGET
    L = F.target_loss(law, target)
    N = F.n_star(law, L)
    out = []
    for gb in FIT_WIPES:
        r = F.row(gb, N, L)
        p = F.params(gb)
        t5 = T.evaluate(p, T.TIERS["T5"], p["mfu"])
        out.append({"R_wipe_gb": gb, "periods": r["periods"], "D_closed": r["D_closed"], "D_fit": r["D_fit"],
                    "D_fit_T5": F.dampener(p, r["mu_fit"], t5["declared"], t5["cheat"]),
                    "D_fit_low": r["D_fit_low"], "D_fit_high": r["D_fit_high"], "D_fit_train": r["D_fit_train"]})
    s = F.summary(law)
    bite = s["bite_gb"]
    holds = bite[f"{target}/floor/none"]
    name = target.upper()
    other = [
        (f"N*, a {name}-class target at no data limit, weights", N),
        ("N*, a K2-class target (at most), weights", s["n_star"]["k2"]["inf"]),
        *((f"w_e, {k}, bits", w) for k, w in F.BOUNDARY.items()),
        ("SmallTalk's path gain", F.PATH_GAIN["smalltalk"]),
        (f"R_wipe up to which the fit bound holds, {name}-class, GB", holds),
        ("the same, K2-class, GB", bite["k2/floor/none"]),
        (f"the same, {name}-class at {num(F.B_TRAIN['central'])} bits per trained weight, GB",
         bite[f"{target}/central/none"]),
        (f"the same, {name}-class under SmallTalk's path composition, GB", bite[f"{target}/floor/smalltalk"]),
        ("R_wipe up to which the fit bound is below the closed form, GB",
         bisect(lambda gb: F.row(gb, N, L)["fit_over_closed"] >= 1, FIT_WIPES[-1], holds * (1 - 1e-9))),
    ]
    return {"rows": out, "other": rows(other)}


def regimes() -> dict:
    """Section 5's regimes at the best plausible design by the fit bound, at T3 and T5, with today's wipes (where it
    caps nothing) and with 6 GB streamed-label wipes; and the s above which s sets D, 1/D at s = 0."""
    wipes = {"today's wipes": BEST, "6 GB streamed-label wipes": streamed(6.0, BEST)}
    out = [{"regime": name, "case": case, "s": s,
            "D": {w: {t: fit_dampener(T.DEFAULTS | over | {"unmonitored": s}, t) for t in REGIME_TIERS}
                  for w, over in wipes.items()}}
           for name, shares in REGIMES.items() for case, s in zip(("central", "adversarial"), shares)]
    return {"rows": out, "s_sets_D": {w: {t: 1 / fit_dampener(T.DEFAULTS | over, t) for t in REGIME_TIERS}
                                      for w, over in wipes.items()}}


def search(job: tuple[str, str, int]) -> dict[float, list[list[float]]]:
    scenario, tier, row = job
    key, over, _ = THRESHOLD_ROWS[row]
    return X.thresholds(T.DEFAULTS | X.SCENARIOS[scenario] | over, key, tier, levels=LEVELS, size=dampener)


def thresholds(pool: ProcessPoolExecutor | None = None) -> dict:
    """Section 6's thresholds on D: per scenario, tier and parameter, where on its slider range D is below each level,
    the others as in the scenario."""
    jobs = [(s, t, i) for s in X.THRESHOLD_SCENARIOS for t in THRESHOLD_TIERS for i in range(len(THRESHOLD_ROWS))]
    out: dict = {s: {t: [] for t in THRESHOLD_TIERS} for s in X.THRESHOLD_SCENARIOS}
    for (s, t, i), found in zip(jobs, (pool.map if pool else map)(search, jobs)):
        key, over, views = THRESHOLD_ROWS[i]
        p = T.DEFAULTS | X.SCENARIOS[s] | over
        for label, f in views:
            out[s][t].append({
                "parameter": label, "key": key, "overrides": over,
                "below": {lv: X.describe(found[lv], key, lambda x, f=f, p=p: num(f(p, x))) for lv in LEVELS},
                "intervals": {lv: [[f(p, a), f(p, b)] for a, b in found[lv]] for lv in LEVELS},
            })
    return out


def fit_edges() -> dict[str, dict[float, float | None]]:
    """Section 6's memory edges by the fit bound: per tier, the wipe residual in GB below which it gives D at least each
    level at the floor credit; None where no wipe residual does, ``inf`` where every one up to 1,000 GB does."""
    def below(tier: str, level: float) -> Callable[[float], bool]:
        return lambda gb: capped(T.DEFAULTS, tier, fit_mu(gb)) < level

    def edge(tier: str, level: float) -> float | None:
        f = below(tier, level)
        return None if f(lo) else math.inf if not f(hi) else bisect(f, lo, hi, 40)
    lo, hi = 1e-3, 1e3
    return {t: {lv: edge(t, lv) for lv in LEVELS} for t in FIT_TIERS}


def tornado(tier: str) -> tuple[str, float]:
    """The parameter outside ``TORNADO_SKIP`` whose range in ``exposure.RANGES`` moves eta most at the floor credit,
    and by what share."""
    p = T.DEFAULTS
    base = X.eta(p, tier)
    swings = {k: max(abs(X.eta(p | {k: v}, tier) / base - 1) for v in (lo, hi))
              for k, (lo, hi, _) in X.RANGES.items() if k not in TORNADO_SKIP}
    key = max(swings, key=swings.__getitem__)
    return key, swings[key]


def alpha_falls(k: dict) -> float | None:
    """The factor by which alpha must fall for D >= 100, from K's threshold row; None when no K in its range reaches it."""
    edges = k["intervals"][100.0]
    return None if k["below"][100.0] == "always" else T.DEFAULTS["K"] / edges[0][0] if edges else 1.0


def exposures(th: dict) -> list[dict]:
    """Section 6's other numbers."""
    p = T.DEFAULTS
    loud = p | X.SCENARIOS["served today, R_wipe 4 GB"] | {"sigma": 1e7}
    falls = alpha_falls(next(r for r in th["defaults"]["T3"] if r["key"] == "K"))
    chi = cheat(p)
    out = [
        ("r̂ added by a thousand identities' verdict lag", 1000 * p["lag_s"] / (p["years"] * T.YEAR)),
        ("ν at 100 times today's K", X.nu(p | {"K": 100 * p["K"]})),
        ("ν at 1,000 times today's K", X.nu(p | {"K": 1000 * p["K"]})),
        (f"factor by which α must fall for D ≥ 100, defaults, T3, K's slider down to {num(X.SPECS['K']['min'])} bits",
         falls),
        ("η at T4, served today with a 4 GB wipe residual and σ = 1e7 bit/s", X.eta(loud, "T4")),
        ("R_wipe there, helpers refilling within the read deadline, GB", X.wipe_residual(loud) / T.GB),
        ("multiple of today's proving cost at which χ reaches 1%",
         bisect(lambda x: cheat(p | {"overhead": x * p["overhead"]}) >= 0.01, 1.0, 1e4)),
        ("the same if χ were linear in it, 0.01/χ", 0.01 / chi),
    ]
    for tier in T.TIERS:
        key, share = tornado(tier)
        out.append((f"largest move in η over its range at the floor credit, {tier}, of any parameter but r̂'s parts, "
                    f"J, s and g: {key}, as a share of η", share))
    return rows(out)


def holes() -> dict:
    """The known holes: the cheating share at today's sparse draws and at 512 draws per statement, the T5 ladder at
    each, and the holes' other numbers."""
    p = T.DEFAULTS
    C = capacity(p)
    cases = []
    for cost, overhead in COSTS.items():
        q = p | {"overhead": overhead}
        r = T.evaluate(q, T.TIERS["T5"])
        cases.append({"cost": cost, "overhead": overhead, "bits": r["cheat_bits"],
                      "chord": T.chord(q, r["cheat_bits"]), "chi": r["cheat"], "pool": r["cheat"] * C})
    t5 = [{"rung": name, "D": [dampener(p | over | {"overhead": c["overhead"]}, "T5") for c in cases],
           "D_fit": [fit_dampener(p | over | {"overhead": c["overhead"]}, "T5") for c in cases]}
          for name, over in RUNGS.items()]
    other = [
        ("χ at today's sparse draws over χ at 512 draws", cases[0]["chi"] / cases[1]["chi"]),
        ("thread_bits: the reads up to which one inference block is held to (d, e)-threading", p["thread_bits"]),
        ("S_1 = √(2 b_w b_a block_mults Γ): the reads at which one block's (1, 0) bound reaches all its work, bits",
         math.sqrt(2 * p["b_w"] * p["b_a"] * p["block_mults"] * p["Gamma"])),
        ("the reads at which one block's (d, e) bound reaches all its work, bits", T.full_block_bits(p)),
        ("a window's state at the optimal window, 2 R_wipe today, bits", 2 * X.wipe_residual(p)),
        ("μ̂ at d = 1, today's R_wipe", X.mu(p | {"d": 1.0})),
        ("1 / slack at 86 rounds and δ = 1/100", 1 / T.slack(p | {"pous_rounds": 86.0})),
        ("slack at 86 rounds and δ = 2^-40", T.slack(p | {"pous_rounds": 86.0, "pous_delta": 2.0 ** -40})),
        ("1/r̂ under the floor credit", 1 / T.audited_residual(p)),
        ("1/r̂ at the best plausible PoUW design", 1 / T.audited_residual(p | BEST)),
    ]
    return {"cases": cases, "t5": t5, "other": rows(other)}


def statement(pool: ProcessPoolExecutor | None = None) -> dict:
    sha, dirty = commit()
    th = thresholds(pool)
    return {"commit": sha, "uncommitted": dirty, "slide": slide(), "symbols": symbols(), "terms": terms(),
            "ladder": ladder(), "floor": floor(), "substitutes": substitutes(), "fit": fit_bound(),
            "regimes": regimes(), "levels": LEVELS, "thresholds": th, "fit_edges": fit_edges(),
            "exposures": exposures(th), "holes": holes()}


def table(header: list[str], body: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header), *("| " + " | ".join(r) + " |" for r in body),
            ""]


def bullets(items: list[dict]) -> list[str]:
    return [f"- {r['quantity']}: {'not reached' if r['value'] is None else num(r['value'])}" for r in items] + [""]


def render(s: dict) -> str:
    dirty = " (with uncommitted changes in benchmarks/nci)" if s["uncommitted"] else ""
    out = ["# The compute dampener: the page's numbers", "",
           f"Rendered by `benchmarks/nci/statement.py` at commit `{s['commit']}`{dirty}. D is "
           "`rho.theorem.multiple_bound` in Lean's form (`remoteDampener`) on `tiers.evaluate(p, TIERS[t], MFU_0)`'s "
           "terms, the residual training at MFU_0 (the reference-efficiency assumption), the hardware multiple at MFU_0 = "
           f"{num(T.DEFAULTS['mfu'])}; every case is a set of overrides on `tiers.DEFAULTS`. D by the fit bound "
           "(`fit_dampener`) puts the fit bound at central w_e in place of the closed-form memory cap.", "",
           "## Slide forms: the hardest questions", "", *bullets(s["slide"]),
           "## 0. Symbols", ""]
    out += table(["Symbol", "Meaning", "Value"], [[r["symbol"], r["meaning"], num(r["value"])] for r in s["symbols"]])
    out += ["## 4. The terms in detail", "", *bullets(s["terms"])]
    out += ["## The ladder", ""]
    out += table(["Rung", "r̂", "R_wipe, GB", "R_wipe / λ_in, min", *(f"D at {t}" for t in LADDER_TIERS),
                  *(f"D at {t}, fit bound" for t in FIT_TIERS)],
                 [[r["rung"], num(r["r_hat"]), num(r["R_wipe_gb"]), num(r["window_min"]),
                   *(num(r["D"][t]) for t in LADDER_TIERS), *(num(r["D_fit"][t]) for t in FIT_TIERS)]
                  for r in s["ladder"]])
    out += bullets(s["floor"]) + bullets(s["substitutes"])
    law, target = FIT_TARGET
    out += [f"## The fit bound (a {target.upper()}-class target, `fit.FITS['{law}']`)", "",
            "`fit.row` at each wipe residual: D by the closed form (T4), and by the fit bound at central w_e (T4, and "
            "T5 with χ on top), at low and high w_e, and at 16 bits per trained weight; L, the window in wipe periods. "
            f"{num(FIT_WIPES[2])} GB is 1 GB with the GPUs' on-chip 3.05 GB on top.", ""]
    out += table(["R_wipe, GB", "L", "D, closed form", "D, fit", "D, fit at T5", "D, fit at low w_e",
                  "D, fit at high w_e", "D, fit at 16 bits"],
                 [[num(r["R_wipe_gb"]), str(r["periods"]), *(num(r[k]) for k in (
                     "D_closed", "D_fit", "D_fit_T5", "D_fit_low", "D_fit_high", "D_fit_train"))]
                  for r in s["fit"]["rows"]])
    out += bullets(s["fit"]["other"])
    out += [f"## Years of growth, ln D / ln {num(GROWTH)}", ""]
    out += table(["Rung", *(f"{t}" for t in LADDER_TIERS), *(f"{t}, fit bound" for t in FIT_TIERS)],
                 [[r["rung"], *(num(r["years"][t], 2) for t in LADDER_TIERS),
                   *(num(r["years_fit"][t], 2) for t in FIT_TIERS)] for r in s["ladder"]])
    reg = s["regimes"]
    wipes = list(reg["s_sets_D"])
    cols = [(w, t) for w in wipes for t in REGIME_TIERS]
    out += ["## The unmonitored share by regime: best plausible PoUW design, by the fit bound", ""]
    out += table(["Regime", "Case", "s", *(f"D at {t}, {w}" for w, t in cols),
                  *(f"1/D at T5, {w}" for w in wipes)],
                 [[r["regime"], r["case"], num(r["s"]), *(num(r["D"][w][t]) for w, t in cols),
                   *(num(100 / r["D"][w]["T5"], 2) + "%" for w in wipes)] for r in reg["rows"]])
    out += ["s sets D once it exceeds 1/D at s = 0: "
            + "; ".join(f"{num(100 * reg['s_sets_D'][w][t], 2)}% at {t} with {w}" for w, t in cols) + ".", ""]
    out += ["## 6. Thresholds on D", "",
            "Where on its slider range each parameter puts D below each level, the others as in the scenario "
            "(`exposure.SCENARIOS`).", ""]
    for scenario, by_tier in s["thresholds"].items():
        for tier, found in by_tier.items():
            flat = {word: [r["parameter"] for r in found if set(r["below"].values()) == {word}]
                    for word in ("always", "never")}
            crossing = [r for r in found if r["parameter"] not in flat["always"] + flat["never"]]
            out += [f"### {scenario}, {tier}", ""]
            if crossing:
                out += table(["Parameter", *(f"D < {num(lv)}" for lv in s["levels"])],
                             [[r["parameter"], *(r["below"][lv] for lv in s["levels"])] for r in crossing])
            notes = [f"- D {says} over the whole range of: {'; '.join(flat[word])}"
                     for word, says in (("always", "below every level"), ("never", "below no level")) if flat[word]]
            if notes:
                out += [*notes, ""]
    out += ["### Memory by the fit bound", "",
            "The wipe residual below which the fit bound gives D at least each level at the floor credit, GB.", ""]
    out += table(["Tier", *(f"D ≥ {num(lv)}" for lv in s["levels"])],
                 [[t, *("never" if e is None else f"< {num(e)}" for e in s["fit_edges"][t].values())]
                  for t in s["fit_edges"]])
    out += ["### Section 6's other numbers", "", *bullets(s["exposures"])]
    h = s["holes"]
    out += ["## Known holes: the cheating share", "",
            "Each injected bit is priced at the chord of one inference block's thread bound, with (d, e)-threading only "
            "up to `thread_bits` (`tiers.chord`).", ""]
    out += table(["Proving cost", "Overhead", "Injected bits", "Chord, per bit", "χ", "Pool, FLOP-equivalent"],
                 [[c["cost"], *(num(c[k]) for k in ("overhead", "bits", "chord", "chi", "pool"))] for c in h["cases"]])
    out += ["D at T5, by the closed form and by the fit bound:", ""]
    out += table(["Rung", *(c["cost"] for c in h["cases"]), *(f"{c['cost']}, fit bound" for c in h["cases"])],
                 [[r["rung"], *(num(d) for d in r["D"] + r["D_fit"])] for r in h["t5"]])
    out += bullets(h["other"])
    return "\n".join(out).rstrip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="the same numbers as JSON")
    args = ap.parse_args(argv)
    with ProcessPoolExecutor() as pool:
        s = statement(pool)
    print(json.dumps(s, indent=1) if args.json else render(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
