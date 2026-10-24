#!/usr/bin/env python3
"""R2's table from fetched run dirs: per model, the base score and, per arm, the gain over base with a paired bootstrap over
problems, the spread over seeds, the fraction of the full fine-tuning gain, and true minus random / format.

    python analyze.py RUN_DIR... [--metric greedy|sampled] [--dataset gsm8k] [--json OUT.json]

Each RUN_DIR is a fetched run (`research fetch RUN --all`) whose out/ holds run.sh's <model>-<arm>-<reward>-s<seed>/ dirs.
Every eval must cover the same problems in the same order (eval seed 0), which is checked.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

NAME = re.compile(r"^(?P<m>[IB])-(?P<arm>base|tiny|full)-(?P<reward>\w+)-s(?P<seed>\d+)$")


def per_problem(ev: dict, metric: str) -> np.ndarray:
    if metric == "greedy":
        return np.asarray(ev["per_problem_greedy"], dtype=float)
    return np.asarray(ev["per_problem_correct"], dtype=float) / ev["args"]["samples"]


def load(run_dirs, dataset, metric):
    evals = defaultdict(dict)
    for rd in run_dirs:
        for d in glob.glob(os.path.join(rd, "out", "*", f"eval_{dataset}", "eval.json")):
            name = os.path.basename(os.path.dirname(os.path.dirname(d)))
            m = NAME.match(name)
            if not m:
                continue
            ev = json.load(open(d))
            key = (m["m"], m["arm"], m["reward"])
            evals[key][int(m["seed"])] = {"x": per_problem(ev, metric), "run": os.path.basename(rd.rstrip("/")),
                                          "fmt": ev.get("greedy_formatted"), "len": ev.get("greedy_len_tokens"),
                                          "n": ev["n_problems"]}
    return evals


def boot(fn, arrays, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    idx = rng.integers(0, n, size=(B, n))
    vals = np.array([fn(*[a[i] for a in arrays]) for i in idx])
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--metric", choices=["greedy", "sampled"], default="greedy")
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    ev = load(args.runs, args.dataset, args.metric)
    ns = {e["n"] for v in ev.values() for e in v.values()}
    if len(ns) > 1:
        raise SystemExit(f"evals cover different problem counts {ns}; rerun them on one set")
    doc = {"metric": args.metric, "dataset": args.dataset, "models": {}}
    print(f"metric={args.metric} dataset={args.dataset} n_problems={ns}")
    for m in ["I", "B"]:
        if (m, "base", "none") not in ev:
            continue
        base = ev[(m, "base", "none")][0]["x"]
        mean = lambda xs: np.mean(np.stack(list(xs)), axis=0)  # noqa: E731
        arms = {}
        for (mm, arm, reward), seeds in sorted(ev.items()):
            if mm != m or arm == "base":
                continue
            x = mean(e["x"] for e in seeds.values())
            per_seed = {s: float(e["x"].mean() - base.mean()) for s, e in sorted(seeds.items())}
            arms[f"{arm}-{reward}"] = {
                "x": x, "acc": float(x.mean()), "gain": float(x.mean() - base.mean()),
                "gain_ci": boot(lambda a, b: a.mean() - b.mean(), [x, base]),
                "per_seed_gain": per_seed, "seed_sd": float(np.std(list(per_seed.values()), ddof=1)) if len(per_seed) > 1 else None,
                "runs": sorted({e["run"] for e in seeds.values()}),
                "formatted": [seeds[s]["fmt"] for s in sorted(seeds)], "len": [seeds[s]["len"] for s in sorted(seeds)],
            }
        row = {"base_acc": float(base.mean()), "base_ci": boot(lambda a: a.mean(), [base]),
               "base_formatted": ev[(m, "base", "none")][0]["fmt"], "base_run": ev[(m, "base", "none")][0]["run"], "arms": {}}
        full = arms.get("full-true")
        for k, a in arms.items():
            out = {kk: vv for kk, vv in a.items() if kk != "x"}
            if full and k != "full-true":
                out["frac_of_full"] = a["gain"] / full["gain"] if full["gain"] else None
                out["frac_of_full_ci"] = boot(lambda t, f, b: (t.mean() - b.mean()) / (f.mean() - b.mean()), [a["x"], full["x"], base])
            row["arms"][k] = out
        for ctl in ["random", "format"]:
            t, c = arms.get("tiny-true"), arms.get(f"tiny-{ctl}")
            if t and c:
                row[f"true_minus_{ctl}"] = {"diff": t["acc"] - c["acc"], "ci": boot(lambda a, b: a.mean() - b.mean(), [t["x"], c["x"]])}
                if full:
                    row[f"true_minus_{ctl}"]["frac_of_full"] = (t["acc"] - c["acc"]) / full["gain"] if full["gain"] else None
        doc["models"][m] = row
        print(f"\n== {m}: base {row['base_acc']:.3f} [{row['base_ci'][0]:.3f}, {row['base_ci'][1]:.3f}] formatted {row['base_formatted']}")
        for k, a in row["arms"].items():
            fr = f" frac_full {a['frac_of_full']:.2f} [{a['frac_of_full_ci'][0]:.2f}, {a['frac_of_full_ci'][1]:.2f}]" if a.get("frac_of_full") is not None else ""
            sd = f" seed_sd {a['seed_sd']:.3f}" if a["seed_sd"] is not None else ""
            print(f"  {k:12s} acc {a['acc']:.3f} gain {a['gain']:+.3f} [{a['gain_ci'][0]:+.3f}, {a['gain_ci'][1]:+.3f}]{sd}{fr}"
                  f" seeds {json.dumps({s: round(g, 3) for s, g in a['per_seed_gain'].items()})} fmt {a['formatted']} runs {a['runs']}")
        for ctl in ["random", "format"]:
            if f"true_minus_{ctl}" in row:
                r = row[f"true_minus_{ctl}"]
                print(f"  true-{ctl:6s} {r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] frac_full {r.get('frac_of_full')}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(doc, f, indent=1)


if __name__ == "__main__":
    main()
