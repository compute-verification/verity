"""The work policy's ε at a verification budget, and the budget for an ε, over a horizon: OPT5's planner (`opt5.plan_row`,
skippable work at the spec sheet's luck) for the int8 rows, each scheme's serve-time commitments added (ASK-25).

ε is the share of the horizon's declared work that can be skipped undetected, at δ = 2^-40 for one certificate over the
whole horizon: the draws pool across windows and GPUs (every cost is per GPU-second, so G GPUs for T seconds are G T /
round rounds of one GPU's serving).  At a given ε the plan draws A ln(1/δ) / ε units whatever the horizon, so its cost
per GPU-second, and ε at a fixed spend, fall as 1 / (G T).

- (1) ε at a total B of `--budget`: none where the serve-time commitments alone exceed it; and ε when `--budget` is the
  sampling spend on top of them (the plans don't depend on the scheme, so this ε is each model's).
- (2) the total B for each ε in `TARGETS`.
- (3) both at each horizon in `HORIZONS`, G = `--gpus` at the inputs' tokens a second.
- (4) the 1% plan's parts, and what one statement of the dominant template costs (its fixed and per-AND parts).
- (5) the lifetime pool beside ε (`pool`): WX-THM's `Population.Exfil.tail_cum` bounds the declared work accepted
  undetected through the first detection whatever the horizon, so at a sampling spend B_s the skipped work is one pool,
  ε G T = A ln(1/δ) c / B_s GPU-s at c the spend per draw, and ε over G GPUs for T is pool / (G T); with q, a unit's draw
  probability, each at δ = 2^-40 with the luck's A and at δ = 1% with A = 1 (`POOLS`), one drawn unit a statement and
  `GATHER`.

    uv run python benchmarks/sampling_plan/opt5_eps.py --out DIR [--gpus 100000] [--budget 0.01]
"""
from __future__ import annotations

import argparse
import json
import sys
from math import log, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inputs import I, MODELS, SCHEMES  # noqa: E402
from int8_llama import DELTA, WINDOWS, fitted, model, serve, workload  # noqa: E402
from opt5 import P_INPUTS, plan_row  # noqa: E402

TARGETS = (1e-5, 1e-8)
HORIZONS = {"1 GPU-week": (1, 7 * 86400), "1 hour": (None, 3600), "1 day": (None, 86400),
            "30 days": (None, 30 * 86400), "1 year": (None, 365.25 * 86400)}
#: statements of this many drawn units, the largest the refit measured (zk_cost's 512 / 512)
GATHER = 512
#: the pool's (δ, p): the certificate's δ at the spec sheet's luck (None: the run's), and δ = 1% with no luck (A = 1)
POOLS = ((DELTA, None), (0.01, 0.0))


def serve_b(v: dict, model_name: str) -> dict[str, float]:
    m = MODELS[model_name]
    w = workload(v, WINDOWS["week"], m)
    return {s: v["commit_floor"] if s == "floor" else sum(c.cost for c in serve(v, m, s, w)) / w["window_gpu_s"]
            for s in SCHEMES}


def eps_at(v: dict, model_name: str, spend: float, rounds: float, luck: float, rel: float = 2e-3,
           delta: float = DELTA) -> dict | None:
    """The least ε whose plan's proving B fits `spend` (bisection in log ε; B falls as ε grows)."""
    row = lambda e: plan_row(v, model_name, "W", e, luck=luck, rounds=rounds, delta=delta)
    hi = row(1.0)
    if hi["B_proving"] > spend:
        return None
    lo_e, hi_e = 1e-16, 1.0
    while hi_e / lo_e > 1 + rel:
        mid = sqrt(lo_e * hi_e)
        r = row(mid)
        if r["B_proving"] <= spend:
            hi_e, hi = mid, r
        else:
            lo_e = mid
    return hi


def statement_split(v: dict, row: dict) -> dict:
    d = row["dominant"]
    fit_ = fitted(v)
    a0_fixed = model(v, fit_, 1).statement_fixed
    draw = d["a0"] + d["a1"] + d["b"]
    gathered = d["a0"] / GATHER + d["a1"] + d["b"]
    return {"template": d["template"], "statement_fixed": a0_fixed, "statement_per_and": d["a0"] - a0_fixed,
            "staged": d["a1"], "proved": d["b"], "per_draw": draw, "per_draw_gathered": gathered,
            "gather_factor": draw / gathered}


def pool(v: dict, row: dict, budget: float) -> dict:
    """The plan at the spend in its lifetime form: the pool ε G T beside A ln(1/δ) c / B_s at c the spend per draw; q the
    draws over the units (and the dominant template's); and both with `GATHER` drawn units a statement sharing its fixed
    part: the draws' own cost (statements, units, replay) falls by a0 (1 - 1/GATHER) a draw, so the same spend draws more
    by that ratio and the pool shrinks by it."""
    a, gpu_s, d = v["late_cone"] / (1 - row["luck"]), row["gpu_s"], row["dominant"]
    per_draw = row["B_proving"] * gpu_s / row["draws"]
    own = (row["cost"]["statements"] + row["cost"]["units"] + row["cost"]["replay"]) * gpu_s / row["draws"]
    gathered = own - row["a0_per_draw"] * (1 - 1 / GATHER)
    q, size = row["draws"] / row["units"], row["eps"] * gpu_s
    return {"delta": row["delta"], "A": a, "eps": row["eps"], "pool_gpu_s": size,
            "pool_closed_form": a * log(1 / row["delta"]) * per_draw / budget, "per_draw": per_draw,
            "per_draw_own": own, "a0_per_draw": row["a0_per_draw"], "units_per_gpu_s": row["units"] / gpu_s, "q": q,
            "q_dominant": {"template": d["template"], "q": d["k"] / d["units"],
                           "q_gathered": d["k"] / d["units"] * own / gathered},
            "per_draw_gathered": gathered, "q_gathered": q * own / gathered, "pool_gathered": size * gathered / own}


def run(v: dict, model_name: str, gpus: int, budget: float, luck: float) -> dict:
    sb = serve_b(v, model_name)
    out = {"model": model_name, "gpus": gpus, "budget": budget, "luck": luck, "delta": DELTA,
           "draws_per_eps": (v["late_cone"] / (1 - luck)) * log(1 / DELTA), "serve_B": sb,
           "total_budget_feasible": {s: b < budget for s, b in sb.items()}, "horizons": {}}
    for name, (g, seconds) in HORIZONS.items():
        g = gpus if g is None else g
        rounds = g * seconds / v["round"]
        at = eps_at(v, model_name, budget, rounds, luck)
        need = {f"{e:g}": plan_row(v, model_name, "W", e, luck=luck, rounds=rounds) for e in TARGETS}
        pools = []
        for delta, p in POOLS:
            p = luck if p is None else p
            row = at if (delta, p) == (DELTA, luck) else eps_at(v, model_name, budget, rounds, p, delta=delta)
            pools.append(None if row is None else pool(v, row, budget))
        out["horizons"][name] = {
            "gpus": g, "seconds": seconds, "gpu_s": g * seconds, "tokens": v["tokens_per_s"] * g * seconds,
            "eps_at_spend": None if at is None else at["eps"],
            "plan_at_spend": None if at is None else {"draws": at["draws"], "cost": at["cost"], "cut": at["cut"],
                                                      "B_proving": at["B_proving"]},
            "split_at_spend": None if at is None else statement_split(v, at),
            "B_proving_for": {e: r["B_proving"] for e, r in need.items()},
            "B_total_for": {e: {s: b + r["B_proving"] for s, b in sb.items()} for e, r in need.items()},
            "pools": pools}
    return out


def _e(x: float | None) -> str:
    return "none" if x is None else f"{x:.2e}"


def render(outs: list[dict]) -> str:
    o0 = outs[0]
    hz = list(o0["horizons"])
    lines = [f"ε at δ = 2^-40 (one certificate a horizon), skippable work at luck p = {o0['luck']:g} "
             f"(A = {1 / (1 - o0['luck']):g}: {o0['draws_per_eps']:.1f} / ε draws); G = {o0['gpus']:,} GPUs at "
             f"{I['tokens_per_s'].value:,.0f} tokens a second each, beside one GPU-week.", "",
             f"Sampling spend {o0['budget']:g} B on top of the serve-time commitments (a total of {o0['budget']:g} "
             "affords no plan under any scheme):", "",
             "| model | " + " | ".join(f"ε, {h}" for h in hz) + " |", "|---|" + "---|" * len(hz)]
    for o in outs:
        lines.append(f"| {o['model']} | " + " | ".join(_e(o["horizons"][h]["eps_at_spend"]) for h in hz) + " |")
    lines += ["", "Total B (serve-time plus proving) for each ε:", "",
              "| model | scheme | serve B | " + " | ".join(f"ε {e:g}, {h}" for e in TARGETS for h in hz) + " |",
              "|---|---|---|" + "---|" * (len(TARGETS) * len(hz))]
    for o in outs:
        for s, b in o["serve_B"].items():
            cells = [f"{o['horizons'][h]['B_total_for'][f'{e:g}'][s]:.3g}" for e in TARGETS for h in hz]
            lines.append(f"| {o['model']} | {s} | {b:.4f} | " + " | ".join(cells) + " |")
    lines += ["", "The plan at the spend, its parts in B, and one draw of its dominant template in GPU-s (a statement "
              "each, against statements of 512 drawn units):", "",
              "| model | horizon | draws | statements | units | commit | replay | template | fixed | per-AND | staged "
              "| proved | a draw | gathered | factor |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in outs:
        for h in hz:
            p, sp = o["horizons"][h]["plan_at_spend"], o["horizons"][h]["split_at_spend"]
            if p is None:
                continue
            c = p["cost"]
            lines.append(f"| {o['model']} | {h} | {p['draws']:.3g} | {c['statements']:.3g} | {c['units']:.3g} | "
                         f"{c['commit']:.3g} | {c['replay']:.3g} | {sp['template']} | {sp['statement_fixed']:.2f} | "
                         f"{sp['statement_per_and']:.2f} | {sp['staged']:.4f} | {sp['proved']:.4f} | "
                         f"{sp['per_draw']:.2f} | {sp['per_draw_gathered']:.3f} | {sp['gather_factor']:.0f} |")
    year = hz[-1]
    lines += ["", f"The lifetime pool at a sampling spend of {o0['budget']:g}: ε G T, the declared work accepted "
              "undetected through the first detection (WX-THM's tail_cum), in GPU-s at each horizon, and A ln(1/δ) c / "
              "B_s at c the plan's spend per draw:", "",
              "| model | δ | A | " + " | ".join(f"pool, {h}" for h in hz) + " | closed form |",
              "|---|---|---|" + "---|" * (len(hz) + 1)]
    for o in outs:
        for i in range(len(o["horizons"][year]["pools"])):
            ps = [o["horizons"][h]["pools"][i] for h in hz]
            ref = o["horizons"][year]["pools"][i]
            lines.append(f"| {o['model']} | {ref['delta']:.3g} | {ref['A']:g} | "
                         + " | ".join(_e(None if p is None else p["pool_gpu_s"]) for p in ps)
                         + f" | {ref['pool_closed_form']:.2e} |")
    lines += ["", f"At {year} of G GPUs (G T = {o0['horizons'][year]['gpu_s']:.3g} GPU-s): ε = pool / (G T); c the spend "
              f"per draw (its statement's fixed part a0) and q = draws / units, one drawn unit a statement and "
              f"{GATHER}:", "",
              "| model | δ | A | ε | c, GPU-s | a0 | units / GPU-s | q | q, dominant | c, " + f"{GATHER} | q, {GATHER} "
              f"| q, dominant, {GATHER} | pool, {GATHER} | ε, {GATHER} |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in outs:
        gt = o["horizons"][year]["gpu_s"]
        for p in o["horizons"][year]["pools"]:
            lines.append(f"| {o['model']} | {p['delta']:.3g} | {p['A']:g} | {p['eps']:.2e} | {p['per_draw']:.2f} | "
                         f"{p['a0_per_draw']:.2f} | {p['units_per_gpu_s']:.3g} | {p['q']:.2e} | "
                         f"{p['q_dominant']['q']:.2e} | {p['per_draw_gathered']:.3f} | {p['q_gathered']:.2e} | "
                         f"{p['q_dominant']['q_gathered']:.2e} | {p['pool_gathered']:.2e} | "
                         f"{p['pool_gathered'] / gt:.2e} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=sorted(MODELS))
    ap.add_argument("--gpus", type=int, default=100_000, help="GPUs whose serving one certificate pools")
    ap.add_argument("--budget", type=float, default=0.01, help="B: the sampling spend, and the total it is compared to")
    ap.add_argument("--luck", type=float, default=None, help="p (default the inputs' spec-sheet 7/8; 0 is OPT4's ε)")
    a = ap.parse_args(argv)
    v = {k: i.value for k, i in {**I, **P_INPUTS}.items()}
    luck = v["luck"] if a.luck is None else a.luck
    outs = [run(v, m, a.gpus, a.budget, luck) for m in a.models]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "opt5-eps.json").write_text(json.dumps(outs, indent=1))
    text = render(outs)
    (a.out / "opt5-eps.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
