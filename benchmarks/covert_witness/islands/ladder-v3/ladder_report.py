"""The v3 ladder's report from its runs' outputs: the from-scratch curve F, a scaling-law fit, each start's validation, and
the fit's compute-optimal models as the scoring cap and the console read them.

    python ladder_report.py --runs OUT_DIR... --out DIR [--screen s]

Each OUT_DIR is one ladder run's `out/` (ladder.py writes ladder.json, curve.json and count.json there, and starts/<name>/
start.json for each registered start); a resumed size's later run supersedes its earlier ones. Stdlib and numpy only.

F is every annealed branch end of every size (registered starts and `--extra` points alike): its compute, as C0 (MACs,
te_core's counter) and as non-embedding FLOPs 6 N D (N non-embedding parameters, D tokens), and its held-out loss, with the
main runs' un-annealed evaluations kept apart, and F's lower envelopes in both. The fit is L(N, D) = E + A N^-alpha +
B D^-beta (L nats a token on the search pool) over the annealed points of every size but the screens (`--screen`: s, the
cheap screen, whose points are checked against the fit and never enter it), least squares in relative error with E, A,
B >= 0, alpha and beta on a grid (one at the grid's edge is flagged, `alpha_at_bound` or `beta_at_bound`: the points
don't fix it, and the compute-optimal model moves with it). Each fitted point is checked against the fit made without it
(leave one out), and a screen's against the fit; a start more than FLAG_PCT off is flagged.

FIT.json is the fit's compute-optimal model at each loss, between the least and the greatest fitted size (te_core.policy's
`ladder_fit` reads its points' loss and forward_gates_per_token; outside them the cap keeps a start's own forward), with
compute in non-embedding FLOPs, C = 6 N D. At fixed C, N is optimal where alpha A N^-alpha = beta B D^-beta, so the loss at
which N is optimal is E + A N^-alpha (1 + alpha / beta), reached in D = (beta B / (alpha A N^-alpha))^(1 / beta) tokens. A
point's loss is that in bits per byte on the search pool, its forward gates per token are w(N) (the forward's MACs a token,
a power law in N through the fitted sizes' count.json) at the bf16 GEMM price (te_core.gates), and its `flops` are 6 N D. A
start's effective compute is the C at which the compute-optimal model reaches the start's loss. It needs three fitted
sizes: with two, alpha is free, and the compute-optimal N with it.
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
#: sizes outside the fit (`--screen`)
SCREEN = ("s",)


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
                 "macs": e["macs"], "flops": 6 * n * e["tokens"], "bpb": e.get("bpb"), "nats_per_token": e.get("nats_per_token")}
            if e["phase"] == "annealed":
                annealed.append(p | {"branch": e["branch"], "registered": e["registered"]})
            elif e["lineage_steps"] > 0:
                main.append(p | {"phase": e["phase"]})
    annealed.sort(key=lambda p: p["macs"])
    main.sort(key=lambda p: (p["size"], p["macs"]))
    return annealed, main


def envelope(points: list[dict], key: str = "bpb", compute: str = "macs") -> list[str]:
    """The branches no other point beats at no more compute."""
    best, out = math.inf, []
    for p in sorted(points, key=lambda p: (p[compute], p[key])):
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
            "sizes": len({p["non_embedding_params"] for p in points}), "E_at_bound": bool(e == 0),
            "alpha_at_bound": bool(al in (GRID[0], GRID[-1])), "beta_at_bound": bool(be in (GRID[0], GRID[-1]))}


def predict(f: dict, n: float, d: float) -> float:
    return f["E"] + f["A"] * n ** -f["alpha"] + f["B"] * d ** -f["beta"]


def power_law(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """(k, g) of y = k x^g, least squares in logs (one point: g = 1)."""
    lx, ly = np.log(xs), np.log(ys)
    if len(xs) == 1:
        return float(math.exp(ly[0] - lx[0])), 1.0
    g, k = np.polyfit(lx, ly, 1)
    return float(math.exp(k)), float(g)


def usable(f: dict | None) -> bool:
    """A fit with a compute-optimal model: three fitted sizes (two values of N fix E + A N^-alpha at each but leave alpha
    free, and the compute-optimal N moves with it), and both reducible terms."""
    return f is not None and f["sizes"] >= 3 and f["A"] > 0 and f["B"] > 0


def optimum(f: dict, n: float) -> tuple[float, float]:
    """(nats, tokens) of the compute-optimal model with n non-embedding parameters, compute 6 N D."""
    reducible = f["A"] * n ** -f["alpha"]
    return f["E"] + reducible * (1 + f["alpha"] / f["beta"]), (f["beta"] * f["B"] / (f["alpha"] * reducible)) ** (1 / f["beta"])


def effective_flops(f: dict, nats: float) -> float | None:
    """The non-embedding FLOPs 6 N D at which the compute-optimal model reaches `nats` (None at or below E)."""
    if nats <= f["E"]:
        return None
    n = (f["A"] * (1 + f["alpha"] / f["beta"]) / (nats - f["E"])) ** (1 / f["alpha"])
    return 6 * n * optimum(f, n)[1]


def compute_optimal(f: dict | None, runs: dict[str, dict], bpb_per_nat: float | None, checks: list[dict]) -> dict | None:
    """FIT.json (module docstring) over the fitted sizes' runs; None without a usable fit."""
    sizes = sorted(runs.values(), key=lambda r: r["count"]["non_embedding_params"])
    if not usable(f) or len(sizes) < 3 or not bpb_per_nat:
        return None
    ns = [r["count"]["non_embedding_params"] for r in sizes]
    c_k, c_g = power_law(ns, [r["count"]["macs_per_step"] / r["count"]["tokens_per_step"] for r in sizes])
    w_k, w_g = power_law(ns, [r["count"]["forward_macs_per_token"] for r in sizes])
    gates = per_mac("gemm/bf16")
    points = []
    for x in np.linspace(math.log10(ns[0]), math.log10(ns[-1]), max(2, round(FIT_PER_DECADE * math.log10(ns[-1] / ns[0])) + 1)):
        n = 10 ** x
        nats, d = optimum(f, n)
        w = w_k * n ** w_g
        points.append({"loss": nats * bpb_per_nat, "forward_gates_per_token": w * gates, "flops": 6 * n * d,
                       "nats_per_token": nats, "non_embedding_params": n, "tokens": d, "tokens_per_param": d / n,
                       "forward_macs_per_token": w, "train_macs": c_k * n ** c_g * d})
    return {"schema": "te-ladder/fit/v2", "points": points, "fit": f, "checks": checks,
            "costs": {"train_macs_per_token": {"k": c_k, "g": c_g}, "forward_macs_per_token": {"k": w_k, "g": w_g},
                      "gates_per_mac": gates, "sizes": {r["count"]["size"]["name"]: r["count"]["non_embedding_params"]
                                                        for r in sizes}},
            "bpb_per_nat": bpb_per_nat,
            "rule": "loss: bits per byte on the search pool at which a model of non_embedding_params is the compute-optimal "
                    "one under the fit, compute in non-embedding FLOPs (C = 6 N D), between the least and the greatest "
                    "fitted size; flops: that model's 6 N D (its effective compute); forward_gates_per_token: its forward "
                    "MACs a token (a power law through the fitted sizes' counts) at the bf16 GEMM price; train_macs: its "
                    "training at te_core's counted MACs a token (likewise a power law), for reference. checks: every "
                    "annealed point against the fit without it (leave one out), and a screen's against the fit"}


def checks(fitted: list[dict], screens: list[dict], full: dict | None) -> list[dict]:
    """Every fitted point against the fit without it (leave one out), and every screen point against the fit, once it has
    three sizes (with two, its N term is free off them)."""
    out = []
    for p in fitted + screens:
        ref = fit([q for q in fitted if q["branch"] != p["branch"]]) if p in fitted else (full if usable(full) else None)
        pred = predict(ref, p["non_embedding_params"], p["tokens"]) if ref else None
        out.append({"branch": p["branch"], "size": p["size"], "in_fit": p in fitted, "registered": p["registered"],
                    "non_embedding_params": p["non_embedding_params"], "tokens": p["tokens"], "flops": p["flops"],
                    "nats_per_token": p["nats_per_token"], "predicted": pred,
                    "pct": 100 * (p["nats_per_token"] - pred) / pred if pred else None,
                    "against": "the fit without it" if p in fitted else "the fit (a screen, not in it)"})
    return out


def report(runs: dict[str, dict], screen: tuple[str, ...] = SCREEN) -> dict:
    annealed, main = f_points(runs)
    fitted = [p for p in annealed if p["size"] not in screen]
    screens = [p for p in annealed if p["size"] in screen]
    full = fit(fitted)
    if full:
        full["form"] = "L(N, D) = E + A N^-alpha + B D^-beta, nats a token on the search pool; N non-embedding parameters, D tokens"
        full["screens"] = sorted({p["size"] for p in screens})
        full["residuals"] = [{"branch": p["branch"], "in_fit": p in fitted, "non_embedding_params": p["non_embedding_params"],
                              "tokens": p["tokens"], "nats_per_token": p["nats_per_token"],
                              "fit": (q := predict(full, p["non_embedding_params"], p["tokens"])),
                              "pct": 100 * (p["nats_per_token"] - q) / q} for p in annealed]
    check = {c["branch"]: c for c in checks(fitted, screens, full)}
    starts = []
    for size, r in runs.items():
        for name, rec in r["starts"].items():
            prov = rec["provenance"]
            c = check[name]
            off = c["pct"]
            emb = rec["config"]["vocab"] * rec["config"]["d_model"]
            parts = prov["data"]["parts"]
            starts.append({
                "name": name, "digest": rec["digest"], "size": rec["config"], "non_embedding_params": prov["non_embedding_params"],
                "parameters": rec["parameters"], "tokens_per_param": prov["tokens_per_param"], "tokens": prov["tokens"],
                "steps": prov["steps"], "bpb": prov["search_pool"]["bpb"], "nats_per_token": c["nats_per_token"],
                "c0_macs": prov["c0"]["macs"], "c0_flops": 2 * prov["c0"]["macs"], "flops": c["flops"],
                "effective_flops": effective_flops(full, c["nats_per_token"]) if usable(full) else None,
                "embedding": {"parameters": emb / rec["parameters"], "forward_macs": emb / r["count"]["forward_macs_per_token"]},
                "data": {"version": prov["data"]["version"], "windows": prov["data"]["windows"],
                         "parts": [q["index"] for q in parts], "files": [parts[0]["file"], parts[-1]["file"]]},
                "screen": size in screen, "check_predicts": c["predicted"], "pct_vs_check": off,
                "verdict": "no fit yet" if off is None else (
                    ("screen: " if size in screen else "") + ("consistent" if abs(off) <= FLAG_PCT else "FLAG")
                    + f" ({off:+.2f}% against {c['against']})"),
                "run": prov.get("run"), "commit": prov.get("commit")})
    starts.sort(key=lambda s: (s["non_embedding_params"], s["tokens_per_param"]))
    ratios = [p["bpb"] / p["nats_per_token"] for p in annealed if p.get("bpb") and p.get("nats_per_token")]
    fitted_runs = {s: r for s, r in runs.items() if s not in screen}
    return {"schema": "te-ladder/report/v3", "data": "v3", "flag_pct": FLAG_PCT, "starts": starts, "fit": full,
            "checks": list(check.values()),
            "compute_optimal": compute_optimal(full, fitted_runs, float(np.median(ratios)) if ratios else None,
                                               list(check.values())),
            "F": {"rule": "every annealed branch end of every size: its compute, C0 MACs (te_core's counter, steps x one "
                          "step) and non-embedding FLOPs (flops, 6 N D), against held-out loss on covert_witness's search "
                          "pool (bpe32k)",
                  "annealed": annealed, "envelope_bpb": envelope(annealed),
                  "envelope_bpb_flops": envelope(annealed, compute="flops"), "main_run": main},
            "runs": {s: {"dir": r["dir"], "complete": r["complete"], "wall_s": r["run"]["wall_s"], "world": r["run"]["world"],
                         "screen": s in screen, "macs_per_step": r["count"]["macs_per_step"],
                         "forward_macs_per_token": r["count"]["forward_macs_per_token"],
                         "two_macs_over_analytic": r["count"]["two_macs_over_analytic"]} for s, r in runs.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--screen", default=",".join(SCREEN), help="sizes checked against the fit but not in it (comma-separated)")
    a = ap.parse_args(argv)
    rep = report(load_runs(a.runs), tuple(s for s in a.screen.split(",") if s))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    (a.out / "F.json").write_text(json.dumps({"schema": "te-ladder/F/v2", **rep["F"], "fit": rep["fit"]}, indent=1) + "\n")
    if rep["compute_optimal"]:
        (a.out / "FIT.json").write_text(json.dumps(rep["compute_optimal"], indent=1) + "\n")
    print(f"{'start':10} {'N_ne':>11} {'tpp':>6} {'bpb':>7} {'nats':>6} {'C0 MACs':>10} {'6ND':>10} {'C_eff':>10}  verdict")
    for s in rep["starts"]:
        eff = f"{s['effective_flops']:10.3e}" if s["effective_flops"] else f"{'-':>10}"
        print(f"{s['name']:10} {s['non_embedding_params']:>11,} {s['tokens_per_param']:>6g} {s['bpb']:7.4f} "
              f"{s['nats_per_token']:6.3f} {s['c0_macs']:10.3e} {s['flops']:10.3e} {eff}  {s['verdict']}")
    if f := rep["fit"]:
        print(f"fit: L = {f['E']:.3f} + {f['A']:.4g} N^-{f['alpha']:.3f} + {f['B']:.4g} D^-{f['beta']:.3f}, "
              f"rms {f['rms_rel_pct']:.2f}% over {f['points']} points, {f['sizes']} sizes"
              + (f", screens {','.join(f['screens'])} not in it" if f["screens"] else "")
              + (" (E at its bound 0)" if f["E_at_bound"] else "")
              + "".join(f" ({k} at its grid's edge: the data don't fix it)" for k in ("alpha", "beta") if f[f"{k}_at_bound"]))
        worst = max((c for c in rep["checks"] if c["pct"] is not None), key=lambda c: abs(c["pct"]), default=None)
        if worst:
            print(f"checks: largest {worst['branch']} {worst['pct']:+.2f}% against {worst['against']}")
    if co := rep["compute_optimal"]:
        lo, hi = co["points"][0], co["points"][-1]
        print(f"compute-optimal (C = 6 N D): {lo['non_embedding_params']:.3g} params at {lo['loss']:.4f} bpb, "
              f"{lo['flops']:.3g} FLOPs ({lo['tokens_per_param']:.3g} tokens/param) to {hi['non_embedding_params']:.3g} at "
              f"{hi['loss']:.4f}, {hi['flops']:.3g} ({hi['tokens_per_param']:.3g})")
    for s, r in rep["runs"].items():
        if not r["complete"]:
            print(f"run {s}: unfinished, going or stopped before its top branch ({r['dir']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
