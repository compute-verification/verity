"""Every number the Notion page "The compute dampener: claims, model and tiers" quotes from ``tiers.py``, ``exposure.py``
and ``kappa.py``, at the commit this runs on, grouped as the page is: the symbols, the ladder, years of growth, the
regimes, section 6's thresholds and the known holes' cheating share. D is ``tiers.evaluate(p, TIERS[t])["dampener"]``,
the hardware multiple at MFU_0; every scenario is a set of overrides on ``tiers.DEFAULTS``.

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

import exposure as X  # noqa: E402
import kappa as K  # noqa: E402
import tiers as T  # noqa: E402

MB = 8e6
TODAY = {"residual": 1.0}
ATTESTED = {"residual": 0.217, "clock_share": 0.0, "host_share": 0.0021}
BEST = {"residual": 0.01, "host_share": 0.0021, "clock_share": 0.0}
STREAMED = {"pous_rounds": 74000.0, "pous_floor": 0.0}


def streamed(gb: float, base: dict | None = None) -> dict:
    """``base`` with streamed-label wipes that leave ``gb`` GB: the memory the fill doesn't reach is ``gb`` less the
    slack."""
    over = (base or {}) | STREAMED
    return over | {"unfilled_gb": gb - X.wipe_residual(T.DEFAULTS | over | {"unfilled_gb": 0.0}) / T.GB}


RUNGS = {
    "Today": TODAY,
    "PoUW's floor credit": {},
    "Floor credit, attested clock and EPYC hosts": ATTESTED,
    "Best plausible PoUW design": BEST,
    "Streamed-label wipes, 13 GB": streamed(13.0),
    "Streamed-label wipes, 6 GB": streamed(6.0),
    "Streamed-label wipes, 3.05 GB (no power removal)": streamed(3.05),
    "Power removal, 1 GB": streamed(1.0),
    "Floor (r̂ = 0, 4 GB)": streamed(4.0, BEST | {"residual": 0.0, "host_share": 0.0}),
}
LADDER_TIERS = tuple(T.TIERS)
#: frontier training compute grows 4-5x a year (Epoch AI)
GROWTH = 4.5
#: the s dossier's unmonitored shares, central and adversarial
REGIMES = {
    "Domestic oversight of 3-5 labs": (0.005, 0.05),
    "International compute registry": (0.02, 0.1),
    "US-China agreement at signing": (0.07, 0.3),
}
LEVELS = (1000.0, 100.0, 10.0, 2.0)
THRESHOLD_TIERS = ("T1", "T3", "T4", "T5")


def same(p: dict, x: float) -> float:
    return x


def alpha(p: dict, bits: float) -> float:
    return bits / p["Gamma"]


def r_wipe_gb(p: dict, unfilled: float) -> float:
    return X.wipe_residual(p | {"unfilled_gb": unfilled}) / T.GB


#: section 6's parameters: the key searched, its overrides on the scenario, and each view of an edge
THRESHOLD_ROWS: list[tuple[str, dict, list[tuple[str, Callable[[dict, float], float]]]]] = [
    ("residual", {"host_share": 0.0, "clock_share": 0.0}, [("r̂ (the GPU residual, host CPUs and clock at 0)", same)]),
    ("disk_gb", {}, [("unwiped disk, GB", same)]),
    ("intake_x", {}, [("declared intake, multiple of αF", same)]),
    ("sigma", {}, [("side channels σ, bit/s", same)]),
    ("K", {}, [("K, bits", same), ("α = K/Γ", alpha)]),
    ("b_a", {}, [("b_a, bits", same)]),
    ("d", {}, [("d", same)]),
    ("unmonitored", {}, [("unmonitored share s", same)]),
    ("unfilled_gb", STREAMED, [("R_wipe, GB (streamed-label wipes, unfilled_gb varied)", r_wipe_gb)]),
]
DRAWS_512 = 1.7e10
#: the chord as modelled, and 22 and 24.6 times steeper when threading is certified only where proved, applied as
#: proving cost
CHORD = (1.0, 22.0, 24.6)


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
    return T.evaluate(p, T.TIERS[tier])["dampener"]


def cheat(p: dict) -> float:
    return T.evaluate(p, T.TIERS["T5"])["cheat"]


def commit() -> tuple[str, bool]:
    """The commit this runs at, and whether ``benchmarks/nci`` has changes outside it."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True).stdout.strip()
    try:
        return git("rev-parse", "HEAD") or "unknown", bool(git("status", "--porcelain", "--", "."))
    except OSError:
        return "unknown", False


def symbols() -> list[dict]:
    """Section 0's values, with those sections 1, 2 and 4 derive from them."""
    p = T.DEFAULTS
    H = p["years"] * T.YEAR
    C = p["nodes"] * p["node_flops"] * H
    r = T.residual_share(p)
    chi = cheat(p)
    R = X.wipe_residual(p)
    rows = [
        ("N", "monitored servers", p["nodes"]),
        ("F", "a server's peak, FLOP/s", p["node_flops"]),
        ("H", "the horizon, years", p["years"]),
        ("C = NFH", "monitored capacity, FLOP", C),
        ("κ", "Kimi K2's pretraining schedule (kappa.py)", K.schedule(K.MODELS["kimi-k2"])[0]),
        ("κ", "the model's value", p["kappa"]),
        ("κ", "adversarial", X.RANGES["kappa"][0]),
        ("MFU_0", "the unconstrained developer's model-FLOP utilization", p["mfu"]),
        ("m", "recipe multiplier, central", p["m"]),
        ("m", "recipe multiplier, adversarial", X.RANGES["m"][1]),
        ("r̂", "under the floor credit: GPU + clock + host CPUs", r),
        ("r̂ GPU", "the GPUs' residual at the locked clock", p["residual"]),
        ("r̂ clock", "GPU capacity above the credited clock", p["clock_share"]),
        ("r̂ host", "host CPUs at INT8 AMX peak", p["host_share"]),
        ("GPU / r̂", "share of r̂", p["residual"] / r),
        ("clock / r̂", "share of r̂", p["clock_share"] / r),
        ("host / r̂", "share of r̂", p["host_share"] / r),
        ("r̂", "at the best plausible PoUW design", T.residual_share(p | BEST)),
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
        ("μ̂", "at 13 GB", X.mu(p | streamed(13.0))),
        ("μ̂", "at 6 GB", X.mu(p | streamed(6.0))),
        ("J", "identities a detection doesn't end", p["identities"]),
        ("Δ", "verdict lag, s", p["lag_s"]),
        ("Δ/H", "the verdict lag's share of capacity per identity", p["lag_s"] / H),
        ("A", "luck factor", p["luck"]),
        ("χ", "cheating share (T5)", chi),
        ("Pool", "χ C, FLOP-equivalent per identity", chi * C),
        ("d b_w e", "the wipe residual below which the cap floors at ν, MB", p["d"] * p["b_w"] * p["e"] / MB),
        ("2 d b_w e", "the K above which ν grows as K², MB", 2 * p["d"] * p["b_w"] * p["e"] / MB),
    ]
    return [{"symbol": s, "meaning": m, "value": v} for s, m, v in rows]


def ladder() -> list[dict]:
    """Each rung's r̂, R_wipe, optimal window R_wipe / λ_in, D per tier, and D's years of growth."""
    out = []
    for name, over in RUNGS.items():
        p = T.DEFAULTS | over
        D = {t: dampener(p, t) for t in LADDER_TIERS}
        out.append({"rung": name, "overrides": over, "r_hat": T.residual_share(p),
                    "R_wipe_gb": X.wipe_residual(p) / T.GB, "window_min": X.wipe_residual(p) / X.intake(p) / 60,
                    "D": D, "years": {t: math.log(d) / math.log(GROWTH) for t, d in D.items()}})
    return out


def regimes() -> dict:
    """Section 5's regimes at T5 and the best plausible design, with today's wipes and with 6 GB streamed-label wipes;
    and the s above which s sets D, 1/D at s = 0."""
    wipes = {"today's wipes": BEST, "6 GB streamed-label wipes": streamed(6.0, BEST)}
    rows = [{"regime": name, "case": case, "s": s,
             "D": {w: dampener(T.DEFAULTS | over | {"unmonitored": s}, "T5") for w, over in wipes.items()}}
            for name, shares in REGIMES.items() for case, s in zip(("central", "adversarial"), shares)]
    return {"rows": rows, "s_sets_D": {w: 1 / dampener(T.DEFAULTS | over, "T5") for w, over in wipes.items()}}


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
                "below": {lv: X.describe(found[lv], key, lambda x, f=f: num(f(p, x))) for lv in LEVELS},
                "intervals": {lv: [[f(p, a), f(p, b)] for a, b in found[lv]] for lv in LEVELS},
            })
    return out


def exposures() -> list[dict]:
    """Section 6's other numbers."""
    p = T.DEFAULTS
    loud = p | X.SCENARIOS["served today, R_wipe 4 GB"] | {"sigma": 1e7}
    rows = [
        ("r̂ added by a thousand identities' verdict lag", 1000 * p["lag_s"] / (p["years"] * T.YEAR)),
        ("ν at 100 times today's K", X.nu(p | {"K": 100 * p["K"]})),
        ("ν at 1,000 times today's K", X.nu(p | {"K": 1000 * p["K"]})),
        ("η at T4, served today with a 4 GB wipe residual and σ = 1e7 bit/s", X.eta(loud, "T4")),
        ("R_wipe there, helpers refilling within the read deadline, GB", X.wipe_residual(loud) / T.GB),
        ("multiple of today's proving cost at which χ reaches 1% (0.01/χ)", 0.01 / cheat(p)),
    ]
    return [{"quantity": q, "value": v} for q, v in rows]


def holes() -> dict:
    """The known holes: χ at today's sparse draws and at 512 draws per statement for each chord, the T5 ladder at
    each, and the holes' other numbers."""
    p = T.DEFAULTS
    costs = {"today's sparse draws": p["overhead"], "512 draws per statement": DRAWS_512}
    cases = [{"cost": c, "chord": f, "overhead": o * f} for c, o in costs.items() for f in CHORD]
    chi = [c | {"chi": cheat(p | {"overhead": c["overhead"]})} for c in cases]
    t5 = [{"rung": name, "D": [dampener(p | over | {"overhead": c["overhead"]}, "T5") for c in cases]}
          for name, over in RUNGS.items()]
    other = [
        ("χ at today's sparse draws over χ at 512 draws", chi[0]["chi"] / chi[len(CHORD)]["chi"]),
        ("a window's state at the optimal window, 2 R_wipe today, bits", 2 * X.wipe_residual(p)),
        ("μ̂ at d = 1, today's R_wipe", X.mu(p | {"d": 1.0})),
        ("reads at which one inference block's thread bound reaches its full work, bits", T.full_block_bits(p)),
        ("1 / slack at 86 rounds and δ = 1/100", 1 / T.slack(p | {"pous_rounds": 86.0})),
        ("slack at 86 rounds and δ = 2^-40", T.slack(p | {"pous_rounds": 86.0, "pous_delta": 2.0 ** -40})),
        ("1/r̂ under the floor credit", 1 / T.residual_share(p)),
        ("1/r̂ at the best plausible PoUW design", 1 / T.residual_share(p | BEST)),
    ]
    return {"chi": chi, "t5": t5, "other": [{"quantity": q, "value": v} for q, v in other]}


def statement(pool: ProcessPoolExecutor | None = None) -> dict:
    sha, dirty = commit()
    return {"commit": sha, "uncommitted": dirty, "symbols": symbols(), "ladder": ladder(), "regimes": regimes(),
            "levels": LEVELS, "thresholds": thresholds(pool), "exposures": exposures(), "holes": holes()}


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header), *("| " + " | ".join(r) + " |" for r in rows),
            ""]


def render(s: dict) -> str:
    dirty = " (with uncommitted changes in benchmarks/nci)" if s["uncommitted"] else ""
    out = ["# The compute dampener: the page's numbers", "",
           f"Rendered by `benchmarks/nci/statement.py` at commit `{s['commit']}`{dirty}. D is "
           "`tiers.evaluate(p, TIERS[t])['dampener']`, the hardware multiple at MFU_0 = "
           f"{num(T.DEFAULTS['mfu'])}; every case is a set of overrides on `tiers.DEFAULTS`.", "",
           "## 0. Symbols", ""]
    out += table(["Symbol", "Meaning", "Value"], [[r["symbol"], r["meaning"], num(r["value"])] for r in s["symbols"]])
    out += ["## The ladder", ""]
    out += table(["Rung", "r̂", "R_wipe, GB", "R_wipe / λ_in, min", *(f"D at {t}" for t in LADDER_TIERS)],
                 [[r["rung"], num(r["r_hat"]), num(r["R_wipe_gb"]), num(r["window_min"]),
                   *(num(r["D"][t]) for t in LADDER_TIERS)] for r in s["ladder"]])
    out += [f"## Years of growth, ln D / ln {num(GROWTH)}", ""]
    out += table(["Rung", *(f"{t}" for t in LADDER_TIERS)],
                 [[r["rung"], *(num(r["years"][t], 2) for t in LADDER_TIERS)] for r in s["ladder"]])
    reg = s["regimes"]
    wipes = list(reg["s_sets_D"])
    out += ["## The unmonitored share by regime: T5, best plausible PoUW design", ""]
    out += table(["Regime", "Case", "s", *(f"D, {w}" for w in wipes), *(f"1/D, {w}" for w in wipes)],
                 [[r["regime"], r["case"], num(r["s"]), *(num(r["D"][w]) for w in wipes),
                   *(num(100 / r["D"][w], 2) + "%" for w in wipes)] for r in reg["rows"]])
    out += ["s sets D once it exceeds 1/D at s = 0: "
            + "; ".join(f"{num(100 * v, 2)}% with {w}" for w, v in reg["s_sets_D"].items()) + ".", ""]
    out += ["## 6. Thresholds on D", "",
            "Where on its slider range each parameter puts D below each level, the others as in the scenario "
            "(`exposure.SCENARIOS`).", ""]
    for scenario, by_tier in s["thresholds"].items():
        for tier, rows in by_tier.items():
            flat = {word: [r["parameter"] for r in rows if set(r["below"].values()) == {word}]
                    for word in ("always", "never")}
            crossing = [r for r in rows if r["parameter"] not in flat["always"] + flat["never"]]
            out += [f"### {scenario}, {tier}", ""]
            if crossing:
                out += table(["Parameter", *(f"D < {num(lv)}" for lv in s["levels"])],
                             [[r["parameter"], *(r["below"][lv] for lv in s["levels"])] for r in crossing])
            notes = [f"- D {says} over the whole range of: {'; '.join(flat[word])}"
                     for word, says in (("always", "below every level"), ("never", "below no level")) if flat[word]]
            if notes:
                out += [*notes, ""]
    out += ["### Section 6's other numbers", ""]
    out += [f"- {r['quantity']}: {num(r['value'])}" for r in s["exposures"]] + [""]
    h = s["holes"]
    out += ["## Known holes: the cheating share", "",
            "Certified threading makes the chord 22 to 24.6 times steeper, applied here as proving cost.", ""]
    out += table(["Proving cost", "Chord", "Overhead", "χ"],
                 [[c["cost"], f"x{num(c['chord'])}", num(c["overhead"]), num(c["chi"])] for c in h["chi"]])
    out += ["D at T5:", ""]
    out += table(["Rung", *(f"{c['cost']}, chord x{num(c['chord'])}" for c in h["chi"])],
                 [[r["rung"], *(num(d) for d in r["D"])] for r in h["t5"]])
    out += [f"- {r['quantity']}: {num(r['value'])}" for r in h["other"]]
    return "\n".join(out)


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
