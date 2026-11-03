"""The v3 ladder's report from its runs' outputs: the from-scratch curve F, a scaling-law fit, each start's validation, and
the fit's compute-optimal models as the scoring cap reads them.

    python ladder_report.py --runs OUT_DIR... --out DIR

Each OUT_DIR is one ladder run's `out/` (ladder.py writes ladder.json, curve.json and count.json there, and starts/<name>/
start.json for each registered start); a resumed size's later run supersedes its earlier ones. Stdlib and numpy only.

F is every annealed branch end of every size (registered starts and `--extra` points alike): its C0 (MACs, te_core's
counter), tokens and held-out loss, with the main runs' un-annealed evaluations kept apart, and F's lower envelope.
The fit is L(N, D) = E + A N^-alpha + B D^-beta over the annealed points (N non-embedding parameters, D tokens, L nats a
token on the search pool), least squares in relative error with E, A, B >= 0, alpha and beta on a grid. A start is checked
against the fit made without it (leave one out), flagged when it is more than FLAG_PCT worse or better than predicted.

FIT.json is the fit's compute-optimal model at each loss, between the least and the greatest size (te_core.policy's
`ladder_fit` reads its points; outside them the cap keeps a start's own forward): training compute C = c(N) D, with c(N),
one step's counted MACs a token, and w(N), the forward's MACs a token, power laws in N through the sizes' count.json. At
fixed loss, N is compute-optimal where alpha A N^-alpha / gamma = beta B D^-beta (gamma = d ln c / d ln N), so the loss at
which N is optimal is E + A N^-alpha (1 + alpha / (beta gamma)); a point's loss is that in bits per byte on the search pool
and its forward gates per token are w(N) at the bf16 GEMM price (te_core.gates). It needs three sizes: with two, alpha is
free, and the compute-optimal N with it.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from te_core.gates import per_mac  # noqa: E402

FLAG_PCT = 3.0
GRID = np.linspace(0.05, 1.2, 116)
#: points of FIT.json a decade of N
FIT_PER_DECADE = 20


def load_runs(dirs: list[Path]) -> dict[str, dict]:
    """Each size's latest run: its summary, curve, count and registered starts. A run still going (or between phases)
    has no ladder.json, or an older one than its curve.json: its evaluations so far are read from curve.json."""
    out = {}
    for d in dirs:
        count = json.loads((d / "count.json").read_text())
        run = json.loads((d / "ladder.json").read_text()) if (d / "ladder.json").is_file() else None
        curve = json.loads((d / "curve.json").read_text()) if (d / "curve.json").is_file() else []
        top = None if run is None else f"{run['size']['name']}-t{run['plan']['top']['tpp']:g}"
        complete = run is not None and top in run["done"] and len(run["evals"]) >= len(curve)
        if not complete:
            run = {"size": count["size"], "non_embedding_params": count["non_embedding_params"], "evals": curve,
                   "wall_s": None, "world": None} | ({k: run[k] for k in ("wall_s", "world")} if run else {})
        size = run["size"]["name"]
        starts = {p.parent.name: json.loads(p.read_text()) for p in sorted((d / "starts").glob("*/start.json"))}
        prev = out.get(size)
        if prev is None or len(run["evals"]) >= len(prev["run"]["evals"]):
            out[size] = {"dir": str(d), "run": run, "count": count, "starts": starts, "complete": complete}
    return out


def f_points(runs: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    annealed, main = [], []
    for size, r in runs.items():
        n = r["run"]["non_embedding_params"]
        for e in r["run"]["evals"]:
            p = {"size": size, "non_embedding_params": n, "tpp": e["tpp"], "tokens": e["tokens"], "steps": e["lineage_steps"],
                 "macs": e["macs"], "bpb": e.get("bpb"), "nats_per_token": e.get("nats_per_token")}
            if e["phase"] == "annealed":
                annealed.append(p | {"branch": e["branch"], "registered": e["registered"]})
            elif e["lineage_steps"] > 0:
                main.append(p | {"phase": e["phase"]})
    annealed.sort(key=lambda p: p["macs"])
    main.sort(key=lambda p: (p["size"], p["macs"]))
    return annealed, main


def envelope(points: list[dict], key: str = "bpb") -> list[str]:
    """The branches no other point beats at no more compute."""
    best, out = math.inf, []
    for p in sorted(points, key=lambda p: (p["macs"], p[key])):
        if p[key] < best:
            best = p[key]
            out.append(p["branch"])
    return out


def nnls3(cols: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray | None:
    """Weighted least squares with every coefficient >= 0: the unconstrained solution when it is feasible, else the best
    feasible one with some coefficients held at 0 (every subset tried; None when none is feasible)."""
    best, best_err = None, math.inf
    for k in range(cols.shape[1], 0, -1):
        for free in itertools.combinations(range(cols.shape[1]), k):
            c, *_ = np.linalg.lstsq(cols[:, free] * w[:, None], y * w, rcond=None)
            if (c < 0).any():
                continue
            coef = np.zeros(cols.shape[1])
            coef[list(free)] = c
            if k == cols.shape[1]:
                return coef
            err = float((((cols @ coef) - y) * w) ** 2 @ np.ones_like(y))
            if err < best_err:
                best, best_err = coef, err
    return best


def fit(points: list[dict]) -> dict | None:
    """None with no more points than the law's five parameters. For each (alpha, beta) of the grid, (E, A, B) >= 0 by
    least squares (`nnls3`), the best kept."""
    if len(points) <= 5:
        return None
    n = np.array([p["non_embedding_params"] for p in points], dtype=float)
    d = np.array([p["tokens"] for p in points], dtype=float)
    y = np.array([p["nats_per_token"] for p in points], dtype=float)
    w = 1 / y
    best = None
    for al in GRID:
        for be in GRID:
            cols = np.stack([np.ones_like(n), n ** -al, d ** -be], 1)
            coef = nnls3(cols, y, w)
            if coef is None:
                continue
            err = float(((cols @ coef - y) / y) ** 2 @ np.ones_like(y))
            if best is None or err < best[0]:
                best = (err, al, be, coef)
    if best is None:
        return None
    err, al, be, (e, a, b) = best
    return {"E": float(e), "A": float(a), "alpha": float(al), "B": float(b), "beta": float(be),
            "rms_rel_pct": 100 * math.sqrt(err / len(y)), "points": len(y),
            "sizes": len({p["non_embedding_params"] for p in points}), "E_at_bound": bool(e == 0)}


def predict(f: dict, n: float, d: float) -> float:
    return f["E"] + f["A"] * n ** -f["alpha"] + f["B"] * d ** -f["beta"]


def power_law(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """(k, g) of y = k x^g, least squares in logs (one point: g = 1)."""
    lx, ly = np.log(xs), np.log(ys)
    if len(xs) == 1:
        return float(math.exp(ly[0] - lx[0])), 1.0
    g, k = np.polyfit(lx, ly, 1)
    return float(math.exp(k)), float(g)


def compute_optimal(f: dict | None, runs: dict[str, dict], bpb_per_nat: float | None) -> dict | None:
    """FIT.json (module docstring): None without a fit or with fewer than three sizes (two values of N fix E + A N^-alpha
    at each but leave alpha free, and the compute-optimal N moves with it)."""
    sizes = sorted(runs.values(), key=lambda r: r["count"]["non_embedding_params"])
    if f is None or f["sizes"] < 3 or len(sizes) < 3 or not bpb_per_nat or f["A"] == 0 or f["B"] == 0:
        return None
    ns = [r["count"]["non_embedding_params"] for r in sizes]
    c_k, gamma = power_law(ns, [r["count"]["macs_per_step"] / r["count"]["tokens_per_step"] for r in sizes])
    w_k, w_g = power_law(ns, [r["count"]["forward_macs_per_token"] for r in sizes])
    gates = per_mac("gemm/bf16")
    points = []
    for x in np.linspace(math.log10(ns[0]), math.log10(ns[-1]), max(2, round(FIT_PER_DECADE * math.log10(ns[-1] / ns[0])) + 1)):
        n = 10 ** x
        reducible = f["A"] * n ** -f["alpha"]
        nats = f["E"] + reducible * (1 + f["alpha"] / (f["beta"] * gamma))
        d = (f["B"] * f["beta"] * gamma / (f["alpha"] * reducible)) ** (1 / f["beta"])
        w = w_k * n ** w_g
        points.append({"loss": nats * bpb_per_nat, "forward_gates_per_token": w * gates, "nats_per_token": nats,
                       "non_embedding_params": n, "forward_macs_per_token": w, "tokens": d, "tokens_per_param": d / n,
                       "train_macs": c_k * n ** gamma * d})
    return {"schema": "te-ladder/fit/v1", "points": points, "fit": f,
            "costs": {"train_macs_per_token": {"k": c_k, "gamma": gamma}, "forward_macs_per_token": {"k": w_k, "g": w_g},
                      "gates_per_mac": gates, "sizes": {r["count"]["size"]["name"]: r["count"]["non_embedding_params"]
                                                        for r in sizes}},
            "bpb_per_nat": bpb_per_nat,
            "rule": "loss: bits per byte on the search pool at which a model of non_embedding_params is the compute-optimal "
                    "one under the fit (C = c(N) D, c and w power laws through the sizes' counts), between the least and "
                    "the greatest size; forward_gates_per_token: its forward MACs a token at the bf16 GEMM price"}


def report(runs: dict[str, dict]) -> dict:
    annealed, main = f_points(runs)
    full = fit(annealed)
    if full:
        full["form"] = "L(N, D) = E + A N^-alpha + B D^-beta, nats a token on the search pool; N non-embedding parameters, D tokens"
        full["residuals"] = [{"branch": p["branch"], "non_embedding_params": p["non_embedding_params"], "tokens": p["tokens"],
                              "nats_per_token": p["nats_per_token"], "fit": (q := predict(full, p["non_embedding_params"], p["tokens"])),
                              "pct": 100 * (p["nats_per_token"] - q) / q} for p in annealed]
    starts = []
    for size, r in runs.items():
        for name, rec in r["starts"].items():
            prov = rec["provenance"]
            p = next(q for q in annealed if q["branch"] == name)
            loo = fit([q for q in annealed if q["branch"] != name])
            pred = predict(loo, p["non_embedding_params"], p["tokens"]) if loo else None
            off = 100 * (p["nats_per_token"] - pred) / pred if loo else None
            parts = prov["data"]["parts"]
            starts.append({
                "name": name, "digest": rec["digest"], "size": rec["config"], "non_embedding_params": prov["non_embedding_params"],
                "parameters": rec["parameters"], "tokens_per_param": prov["tokens_per_param"], "tokens": prov["tokens"],
                "steps": prov["steps"], "bpb": prov["search_pool"]["bpb"], "nats_per_token": p["nats_per_token"],
                "c0_macs": prov["c0"]["macs"], "c0_flops": 2 * prov["c0"]["macs"],
                "data": {"version": prov["data"]["version"], "windows": prov["data"]["windows"],
                         "parts": [q["index"] for q in parts], "files": [parts[0]["file"], parts[-1]["file"]]},
                "loo_fit_predicts": pred, "pct_vs_loo_fit": off,
                "verdict": "no fit yet" if off is None else (
                    "consistent" if abs(off) <= FLAG_PCT else f"FLAG: {off:+.1f}% against the fit without it"),
                "run": prov.get("run"), "commit": prov.get("commit")})
    starts.sort(key=lambda s: (s["non_embedding_params"], s["tokens_per_param"]))
    ratios = [p["bpb"] / p["nats_per_token"] for p in annealed if p.get("bpb") and p.get("nats_per_token")]
    return {"schema": "te-ladder/report/v2", "data": "v3", "flag_pct": FLAG_PCT, "starts": starts, "fit": full,
            "compute_optimal": compute_optimal(full, runs, float(np.median(ratios)) if ratios else None),
            "F": {"rule": "every annealed branch end of every size: C0 MACs (te_core's counter, steps x one step) against "
                          "held-out loss on covert_witness's search pool (bpe32k)",
                  "annealed": annealed, "envelope_bpb": envelope(annealed), "main_run": main},
            "runs": {s: {"dir": r["dir"], "complete": r["complete"], "wall_s": r["run"]["wall_s"], "world": r["run"]["world"],
                         "macs_per_step": r["count"]["macs_per_step"], "forward_macs_per_token": r["count"]["forward_macs_per_token"],
                         "two_macs_over_analytic": r["count"]["two_macs_over_analytic"]} for s, r in runs.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rep = report(load_runs(a.runs))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    (a.out / "F.json").write_text(json.dumps({"schema": "te-ladder/F/v1", **rep["F"], "fit": rep["fit"]}, indent=1) + "\n")
    if rep["compute_optimal"]:
        (a.out / "FIT.json").write_text(json.dumps(rep["compute_optimal"], indent=1) + "\n")
    print(f"{'start':10} {'N_ne':>11} {'tpp':>6} {'bpb':>7} {'nats':>6} {'C0 MACs':>10}  verdict")
    for s in rep["starts"]:
        off = "" if s["verdict"] != "consistent" else f" ({s['pct_vs_loo_fit']:+.2f}%)"
        print(f"{s['name']:10} {s['non_embedding_params']:>11,} {s['tokens_per_param']:>6g} {s['bpb']:7.4f} "
              f"{s['nats_per_token']:6.3f} {s['c0_macs']:10.3e}  {s['verdict']}{off}")
    if f := rep["fit"]:
        print(f"fit: L = {f['E']:.3f} + {f['A']:.4g} N^-{f['alpha']:.3f} + {f['B']:.4g} D^-{f['beta']:.3f}, "
              f"rms {f['rms_rel_pct']:.2f}% over {f['points']} points, {f['sizes']} sizes"
              + (" (E at its bound 0)" if f["E_at_bound"] else ""))
    if co := rep["compute_optimal"]:
        lo, hi = co["points"][0], co["points"][-1]
        print(f"compute-optimal: {lo['non_embedding_params']:.3g} params at {lo['loss']:.4f} bpb "
              f"({lo['tokens_per_param']:.3g} tokens/param) to {hi['non_embedding_params']:.3g} at {hi['loss']:.4f} "
              f"({hi['tokens_per_param']:.3g}); c(N) ~ N^{co['costs']['train_macs_per_token']['gamma']:.3f}")
    for s, r in rep["runs"].items():
        if not r["complete"]:
            print(f"run {s}: still going ({r['dir']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
