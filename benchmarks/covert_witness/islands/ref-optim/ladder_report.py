"""The ladder's report from its runs' outputs: the from-scratch curve F, a scaling-law fit, and each start's validation.

    python ladder_report.py --runs OUT_DIR... --out DIR

Each OUT_DIR is one ladder run's `out/` (ladder.py writes ladder.json, curve.json and count.json there, and starts/<name>/
start.json for each registered start); a resumed size's later run supersedes its earlier ones. Stdlib and numpy only.

F is every annealed branch end of every size (registered starts and `--extra` points alike): its C0 (MACs, te_core's
counter), tokens and held-out loss, with the main runs' un-annealed evaluations kept apart, and F's lower envelope.
The fit is L(N, D) = E + A N^-alpha + B D^-beta over the annealed points (N non-embedding parameters, D tokens, L nats a
token on the search pool), least squares in relative error, alpha and beta on a grid. A start is checked two ways:

- against the fit made without it (leave one out): flagged when it is more than FLAG_PCT worse or better than predicted;
- against published FineWeb-Edu losses (REFERENCES, GPT-2 124M, 85M non-embedding parameters, trained with AdamW and a
  cosine schedule): the fit's prediction there must be no worse than the published loss by more than FLAG_PCT, since the
  ladder's recipe is meant to be at least as strong.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

FLAG_PCT = 3.0
GPT2_124M_NON_EMBEDDING = 124_439_808 - 50257 * 768 - 1024 * 768
REFERENCES = [
    {"what": "GPT-2 124M on FineWeb-Edu sample-10BT, 10B tokens, nanoGPT/llm.c recipe (AdamW, cosine)",
     "n": GPT2_124M_NON_EMBEDDING, "tokens": 10e9, "loss": [2.99, 3.10],
     "sources": ["github.com/ctxnn1 build-nanogpt FineWeb-Edu run: val 3.031",
                 "Anjaneyalu build-nanogpt reproduction: val 2.990 (OpenAI's GPT-2 124M 3.262 on the same val)",
                 "tao-hpu nanoGPT FineWeb-Edu: val 3.02", "Xerneas3318 GPT-2 124M FineWeb-Edu 10B: val 3.10"]},
    {"what": "GPT-2 124M on FineWeb-Edu, 300M tokens into the same 10B-token cosine run (not annealed, so a loose bound)",
     "n": GPT2_124M_NON_EMBEDDING, "tokens": 0.3e9,
     "loss": [3.65, 3.65], "sources": ["tao-hpu nanoGPT FineWeb-Edu: val 3.65 at 300M tokens"]},
]


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


def fit(points: list[dict]) -> dict | None:
    """None with no more points than the law's five parameters, or no fit with A, B >= 0."""
    if len(points) <= 5:
        return None
    n = np.array([p["non_embedding_params"] for p in points], dtype=float)
    d = np.array([p["tokens"] for p in points], dtype=float)
    y = np.array([p["nats_per_token"] for p in points], dtype=float)
    best = None
    for al in np.linspace(0.05, 1.2, 116):
        for be in np.linspace(0.05, 1.2, 116):
            x = np.stack([np.ones_like(n), n ** -al, d ** -be], 1)
            w = 1 / y
            coef, *_ = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)
            if coef[1] < 0 or coef[2] < 0:
                continue
            err = float((((x @ coef) - y) / y) ** 2 @ np.ones_like(y))
            if best is None or err < best[0]:
                best = (err, al, be, coef)
    if best is None:
        return None
    err, al, be, (e, a, b) = best
    return {"E": float(e), "A": float(a), "alpha": float(al), "B": float(b), "beta": float(be),
            "rms_rel_pct": 100 * math.sqrt(err / len(y)), "points": len(y)}


def predict(f: dict, n: float, d: float) -> float:
    return f["E"] + f["A"] * n ** -f["alpha"] + f["B"] * d ** -f["beta"]


def report(runs: dict[str, dict]) -> dict:
    annealed, main = f_points(runs)
    full = fit(annealed)
    largest = max((p["non_embedding_params"] for p in annealed), default=0)
    refs = []
    for r in REFERENCES:
        if full is None:
            refs.append(r | {"fit_predicts": None, "pct_over_published_max": None, "verdict": "no fit yet"})
            continue
        pred = predict(full, r["n"], r["tokens"])
        worse = 100 * (pred - r["loss"][1]) / r["loss"][1]
        refs.append(r | {"fit_predicts": pred, "pct_over_published_max": worse, "extrapolated": r["n"] > largest,
                         "verdict": "consistent" if worse <= FLAG_PCT else "FLAG: the ladder's fit is weaker than published"})
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
                "data": {"windows": prov["data"]["windows"], "parts": [q["index"] for q in parts],
                         "files": [parts[0]["file"], parts[-1]["file"]], "held_out_left_out": prov["data"]["held_out"]},
                "loo_fit_predicts": pred, "pct_vs_loo_fit": off,
                "verdict": "no fit yet" if off is None else (
                    "consistent" if abs(off) <= FLAG_PCT else f"FLAG: {off:+.1f}% against the fit without it"),
                "run": prov.get("run"), "commit": prov.get("commit")})
    starts.sort(key=lambda s: (s["non_embedding_params"], s["tokens_per_param"]))
    return {"schema": "te-ladder/report/v1", "flag_pct": FLAG_PCT, "starts": starts, "fit": full, "references": refs,
            "F": {"rule": "every annealed branch end of every size: C0 MACs (te_core's counter, steps x one step) against "
                          "held-out loss on covert_witness's search pool",
                  "annealed": annealed, "envelope_bpb": envelope(annealed), "main_run": main},
            "runs": {s: {"dir": r["dir"], "complete": r["complete"], "wall_s": r["run"]["wall_s"], "world": r["run"]["world"],
                         "macs_per_step": r["count"]["macs_per_step"],
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
    print(f"{'start':10} {'N_ne':>11} {'tpp':>6} {'bpb':>7} {'nats':>6} {'C0 MACs':>10}  verdict")
    for s in rep["starts"]:
        off = "" if s["verdict"] != "consistent" else f" ({s['pct_vs_loo_fit']:+.2f}%)"
        print(f"{s['name']:10} {s['non_embedding_params']:>11,} {s['tokens_per_param']:>6g} {s['bpb']:7.4f} "
              f"{s['nats_per_token']:6.3f} {s['c0_macs']:10.3e}  {s['verdict']}{off}")
    if f := rep["fit"]:
        print(f"fit: L = {f['E']:.3f} + {f['A']:.4g} N^-{f['alpha']:.3f} + {f['B']:.4g} D^-{f['beta']:.3f}, "
              f"rms {f['rms_rel_pct']:.2f}% over {f['points']} points")
    for r in rep["references"]:
        got = "" if r["fit_predicts"] is None else f", fit {r['fit_predicts']:.3f}" + (
            " (extrapolated past the largest size)" if r["extrapolated"] else "")
        print(f"reference: {r['what']}: published {r['loss']}{got} ({r['verdict']})")
    for s, r in rep["runs"].items():
        if not r["complete"]:
            print(f"run {s}: unfinished, going or stopped before its top branch ({r['dir']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
