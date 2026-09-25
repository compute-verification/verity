"""Tables mapping the existing rollout results onto the NCI draft (IU size cap Gamma, NCI limit alpha, coverage).

    PYTHONPATH=. .venv/bin/python -m accumulation.sweeps.nci_projection

Table A -- rollout cost vs Gamma.  Session work is fitted as `fwd(T) = a T + b T^2` through the two longest calibrated
sessions (`results/calibration.json`); `G_hat = Gamma / fwd(8K)` expresses a Gamma in our units.  Values are
`analytic.rollout_tiers` tier 2 (the convex m-layer x t-token attack family: an estimate of U, not a certificate) at
Q_roll = 2^34 (steady state), F = inf.  Three Gammas: one 128K session of the same model, the draft's Kimi-K3
calibration (one 10^6-token session = 2.1e11 T + 7.4e5 T^2 gates = 4.52e6 minimum-work positions, transferred as that
many positions of the model's own minimum per-position work `a`), and one 1M session of the same model.

Table B -- pretraining via the draft's coverage reduction (rho >= alpha_C s_C) at our Gamma, from measured rollout rows
with Q_roll = Q_step (current hashes, Q_inf = 8K, G_hat = 4); alpha_C = 1 / train_over_fwd (calibration, Q_inf = 8K).

Slowdowns are (runtime input per token) / 4 B, the normalization used in all our tables; multiply by `w_min / w_avg(8K)`
for the draft's alpha (32 bits / minimum per-position work).  Writes `results/nci_projection.{json,md}`."""

from __future__ import annotations

import json
from pathlib import Path

from accumulation.sweeps.analytic import model_cfg, rollout_tiers
from accumulation.sweeps.calibration import load_calibration
from accumulation.sweeps.coarse_run import module_versions

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODELS = ["llama3-1b", "llama3-8b", "llama3-70b", "llama3-405b", "mixtral-8x7b", "qwen3-235b-a22b", "deepseek-v3"]
DENSE = MODELS[:4]
SHORT = {"llama3-1b": "1B", "llama3-8b": "8B", "llama3-70b": "70B", "llama3-405b": "405B", "mixtral-8x7b": "Mixtral 8x7B",
         "qwen3-235b-a22b": "Qwen3-235B-A22B", "deepseek-v3": "DeepSeek-V3"}
Q_INF = 8192
Q_STEADY = 1 << 34
TOKEN_B = 4.0
KIMI_POSITIONS = (2.1e11 * 1e6 + 7.4e5 * 1e12) / 2.1e11
Q_STEPS = [1 << 17, 1 << 20, 1 << 22]


def session_fit(cal: dict, model: str) -> tuple[float, float, float]:
    """(a, b, fwd(8K)) with fwd(T) = a T + b T^2 through the two longest calibrated sessions."""
    pts = sorted((q, c["fwd_work"]) for (m, q), c in cal.items() if m == model and c.get("fwd_work"))
    (q1, w1), (q2, w2) = pts[-2], pts[-1]
    b = (w2 / q2 - w1 / q1) / (q2 - q1)
    return w1 / q1 - b * q1, b, dict(pts)[Q_INF]


def tier2(model: str, G_hat: float, cal: dict) -> dict:
    t = rollout_tiers(model, Q_roll=Q_STEADY, Q_inf=Q_INF, F_hat=1e9, G_hat=G_hat, calib=cal)
    return {"G_hat": G_hat, "b": t["b2"], "slow": t["b2"] / TOKEN_B, "m_star": t["m_star"], "t_star": t["t_star"],
            "whole_depth": t["m_star"] >= model_cfg(model).layers - 1e-9}


def rollout_rows(model: str) -> dict[tuple[float, int], dict]:
    """Current-hash rollout rows at Q_inf = 8K, F_hat = 1.5, keyed by (G_hat, Q_roll); U only from a block/expert family plan."""
    cur = module_versions()
    out = {}
    for ln in open(RESULTS / "coarse.jsonl"):
        r = json.loads(ln)
        if r["model"] != model or not r["circuit"].startswith("forward-nonfixed") or r["Q_inf"] != Q_INF or r.get("F_hat") != 1.5:
            continue
        sv = r.get("stage_versions") or {}
        fam = (r.get("U_detail") or {}).get("family")
        U = r.get("U") if sv.get("coarse") == cur["coarse"] and fam not in ("fine", "bands", "single") else None
        L = r.get("L") if sv.get("lower") == cur["lower"] else None
        if U or L:
            out[(r["G_hat"], r["tokens"])] = {"U": U and U / r["tokens"], "L": L and L / r["tokens"], "L_gap": r.get("L_gap"),
                                               "n_ru": r.get("U_n_ru")}
    return out


def main() -> None:
    cal = load_calibration()
    A, B = [], []
    for m in MODELS:
        a, b, fwd8 = session_fit(cal, m)
        rows = rollout_rows(m)
        steady = max((k for k in rows if k[0] == 4 and k[1] >= 1 << 20 and rows[k]["L"] and rows[k]["U"]), key=lambda k: k[1], default=None)
        rec = {"model": m, "w_min_over_w_avg8K": a / (fwd8 / Q_INF), "ours": tier2(m, 4.0, cal),
               "measured_G4": steady and {"Q_roll": steady[1], **rows[steady]}, "gammas": {}}
        for name, fw in (("128K session", a * 2**17 + b * 2**34), ("Kimi-K3 calibration", KIMI_POSITIONS * a), ("1M session", a * 2**20 + b * 2**40)):
            rec["gammas"][name] = {**tier2(m, fw / fwd8, cal), "positions_8K": fw / (fwd8 / Q_INF)}
        rec["attention_share_1M"] = b * 2**40 / (a * 2**20 + b * 2**40)
        A.append(rec)
        if m in DENSE:
            alpha_c = 1 / cal[(m, Q_INF)]["train_over_fwd"]
            for q in Q_STEPS:
                r = rows.get((4.0, q)) or rows.get((4, q))
                if r:
                    B.append({"model": m, "Q_step": q, "alpha_C": alpha_c, **r})
    (RESULTS / "nci_projection.json").write_text(json.dumps({"A": A, "B": B}, indent=1))

    md = ["### Table A", "",
          "| Model | measured at G_hat = 4: L / U slowdown (Q_roll) | tier 2 at G_hat = 4 | 128K session: G_hat / tier 2 | "
          "Kimi-K3 calibration: G_hat / tier 2 | 1M session: G_hat / tier 2 | attention share of a 1M session | w_min / w_avg(8K) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in A:
        mp = r["measured_G4"]
        ms = f"{mp['L'] / TOKEN_B:,.0f}x / {mp['U'] / TOKEN_B:,.0f}x ({_tok(mp['Q_roll'])})" if mp else "no admissible row at Q_roll >= 1M"
        cells = [f"{s['G_hat']:,.0f} / {s['slow']:,.0f}x{' (whole depth)' if s['whole_depth'] else ''}" for s in r["gammas"].values()]
        md.append(f"| {SHORT[r['model']]} | {ms} | {r['ours']['slow']:,.0f}x | {' | '.join(cells)} | "
                  f"{r['attention_share_1M']:.0%} | {r['w_min_over_w_avg8K']:.2f} |")
    md += ["", "### Table B", "",
           "| Model | Q_step | rollout L / U at Q_roll = Q_step (MB/token) | s_C: L / U | alpha_C | alpha_C s_C: L / U |",
           "|---|---|---|---|---|---|"]
    for r in B:
        sL, sU = r["L"] / TOKEN_B, r["U"] / TOKEN_B
        gap = f" (gap {r['L_gap']:.0%})" if r.get("L_gap") else ""
        md.append(f"| {SHORT[r['model']]} | {_tok(r['Q_step'])} | {r['L'] / 1e6:.3g}{gap} / {r['U'] / 1e6:.3g} | {sL:,.0f}x / {sU:,.0f}x | "
                  f"{r['alpha_C']:.3f} | {r['alpha_C'] * sL:,.0f}x / {r['alpha_C'] * sU:,.0f}x |")
    (RESULTS / "nci_projection.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def _tok(n: int) -> str:
    return f"{n >> 20}M" if n >= 1 << 20 else f"{n >> 10}K"


if __name__ == "__main__":
    main()
