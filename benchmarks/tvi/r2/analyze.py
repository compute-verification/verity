#!/usr/bin/env python3
"""R2's table from fetched run dirs: per model, the base score and, per arm, the gain over base with a paired bootstrap over
problems, the spread over seeds, the fraction of the full fine-tuning gain, and true minus random / format.

    python analyze.py RUN_DIR... [--task gsm8k|math] [--json OUT.json] [--full-from RUN_DIR...]

Each RUN_DIR is a fetched run (`research fetch RUN --all`) whose out/ holds grpo.py's result-*.json (R2's or A2's). Every
result must cover the same GSM8K test problems in the same order (grpo.py reads gsm8k_test.jsonl in file order), which is
checked. Gains are greedy accuracy over the 1,319 problems; a seed-averaged arm averages per-problem correctness over its
seeds before the bootstrap, so its interval is over problems only, and the seed spread is reported beside it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

MODELS = {"Qwen/Qwen2.5-1.5B-Instruct": "Instruct", "Qwen/Qwen2.5-1.5B": "Base"}


def load(run_dirs, task="gsm8k"):
    out = defaultdict(dict)
    for rd in run_dirs:
        for p in sorted(glob.glob(os.path.join(rd, "out", "result-*.json"))):
            r = json.load(open(p))
            if r["task"].removesuffix("B") != task or r.get("ckpt"):
                continue
            key = (MODELS.get(r["model"], r["model"]), r["method"], r.get("reward", "true"))
            out[key][r.get("seed", 0)] = {
                "x": np.asarray(r["per_problem"], dtype=float), "run": os.path.basename(rd.rstrip("/")),
                "fmt": r.get("formatted"), "steps": r["steps"], "lr": r.get("lr"), "S_bits": r["S_bits"],
                "mid": [(e["step"], e["acc"]) for e in r.get("evals", [])],
                "len": r.get("mean_len") or (r["evals"][-1]["mean_len"] if r.get("evals") else None),
            }
    return out


def boot(fn, arrays, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, size=n)
        vals.append(fn(*[a[i] for a in arrays]))
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def arm_stats(seeds, base):
    x = np.mean(np.stack([e["x"] for e in seeds.values()]), axis=0)
    per_seed = {s: float(e["x"].mean() - base.mean()) for s, e in sorted(seeds.items())}
    return x, {
        "acc": float(x.mean()), "gain": float(x.mean() - base.mean()),
        "gain_ci": boot(lambda a, b: a.mean() - b.mean(), [x, base]),
        "per_seed_gain": per_seed,
        "seed_sd": float(np.std(list(per_seed.values()), ddof=1)) if len(per_seed) > 1 else None,
        "seed_range": [min(per_seed.values()), max(per_seed.values())],
        "per_seed_acc_ci": {s: boot(lambda a, b: a.mean() - b.mean(), [e["x"], base], B=2000) for s, e in sorted(seeds.items())},
        "formatted": {s: e["fmt"] for s, e in sorted(seeds.items())},
        "mean_len": {s: e["len"] for s, e in sorted(seeds.items())},
        "mid_evals": {s: e["mid"] for s, e in sorted(seeds.items())},
        "runs": sorted({e["run"] for e in seeds.values()}), "steps": sorted({e["steps"] for e in seeds.values()}),
        "lr": sorted({e["lr"] for e in seeds.values()}), "S_bits": sorted({e["S_bits"] for e in seeds.values()}),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--full-from", nargs="*", default=[], help="other runs (A2's) whose `full` results are a second anchor")
    p.add_argument("--task", default="gsm8k", choices=["gsm8k", "math"])
    p.add_argument("--json", default=None)
    args = p.parse_args()
    ev = load(args.runs, args.task)
    ext = load(args.full_from, args.task) if args.full_from else {}
    ns = {len(e["x"]) for v in list(ev.values()) + list(ext.values()) for e in v.values()}
    if len(ns) > 1:
        raise SystemExit(f"results cover different problem counts {ns}")
    doc = {"metric": f"greedy accuracy, {args.task} test", "n_problems": sorted(ns), "models": {}}
    for m in ["Instruct", "Base"]:
        if (m, "base", "true") not in ev:
            continue
        b = ev[(m, "base", "true")][0]
        base = b["x"]
        row = {"base_acc": float(base.mean()), "base_ci": boot(lambda a: a.mean(), [base]), "base_formatted": b["fmt"],
               "base_len": b["len"], "base_run": b["run"], "arms": {}}
        xs = {}
        for (mm, method, reward), seeds in sorted(ev.items()):
            if mm != m or method == "base":
                continue
            xs[f"{method}-{reward}"], row["arms"][f"{method}-{reward}"] = arm_stats(seeds, base)
        for (mm, method, reward), seeds in sorted(ext.items()):
            if mm == m and method == "full" and reward == "true":
                xs["full-true(A2)"], row["arms"]["full-true(A2)"] = arm_stats(seeds, base)
        anchor = "full-true" if "full-true" in xs else ("full-true(A2)" if "full-true(A2)" in xs else None)
        row["anchor"] = anchor
        if anchor:
            f = xs[anchor]
            for k, a in row["arms"].items():
                if k.startswith("full"):
                    continue
                a["frac_of_full"] = a["gain"] / (f.mean() - base.mean())
                a["frac_of_full_ci"] = boot(lambda t, ff, bb: (t.mean() - bb.mean()) / (ff.mean() - bb.mean()), [xs[k], f, base])
        t = xs.get("tl10kB-true")
        for ctl in ["random", "format"]:
            c = xs.get(f"tl10kB-{ctl}")
            if t is not None and c is not None:
                d = {"diff": float(t.mean() - c.mean()), "ci": boot(lambda a, cc: a.mean() - cc.mean(), [t, c])}
                if anchor:
                    d["frac_of_full"] = d["diff"] / (xs[anchor].mean() - base.mean())
                    d["frac_of_full_ci"] = boot(lambda a, cc, ff, bb: (a.mean() - cc.mean()) / (ff.mean() - bb.mean()),
                                                [t, c, xs[anchor], base])
                row[f"true_minus_{ctl}"] = d
        doc["models"][m] = row
        print(f"\n== {m}: base {row['base_acc']:.3f} [{row['base_ci'][0]:.3f}, {row['base_ci'][1]:.3f}]"
              f" formatted {b['fmt']} len {b['len']} run {b['run']}  (anchor {anchor})")
        for k, a in row["arms"].items():
            fr = (f" frac {a['frac_of_full']:.2f} [{a['frac_of_full_ci'][0]:.2f}, {a['frac_of_full_ci'][1]:.2f}]"
                  if "frac_of_full" in a else "")
            sd = f" sd {a['seed_sd']:.3f}" if a["seed_sd"] is not None else ""
            print(f"  {k:18s} acc {a['acc']:.3f} gain {a['gain']:+.3f} [{a['gain_ci'][0]:+.3f}, {a['gain_ci'][1]:+.3f}]{sd}{fr}"
                  f" seeds {json.dumps({s: round(g, 3) for s, g in a['per_seed_gain'].items()})}"
                  f" fmt {json.dumps({s: round(v, 3) if v is not None else None for s, v in a['formatted'].items()})} runs {a['runs']}")
        for ctl in ["random", "format"]:
            d = row.get(f"true_minus_{ctl}")
            if d:
                fr = f" frac {d['frac_of_full']:.2f} [{d['frac_of_full_ci'][0]:.2f}, {d['frac_of_full_ci'][1]:.2f}]" if "frac_of_full" in d else ""
                print(f"  true-{ctl:13s} {d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]{fr}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(doc, f, indent=1)


if __name__ == "__main__":
    main()
