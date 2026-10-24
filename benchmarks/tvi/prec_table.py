"""The precision arm's table, from fetched prec.sh runs: per run, training speed and memory, the training reward's first and
last quarter, and GSM8K-test accuracy (greedy, and the mean over 4 samples at T=1) with its gain over the base model and a
paired bootstrap 95% CI over the 1319 problems; then per recipe, the mean gain over seeds with the seeds' spread.

    python benchmarks/tvi/prec_table.py ~/.research/runs/r2026...-xxxx [...]  > table.md
"""

from __future__ import annotations

import glob
import json
import os
import random
import statistics
import sys


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def boot_ci(diffs, n=2000, seed=0):
    rng = random.Random(seed)
    m = len(diffs)
    means = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def hist_mean(hist, pred, lo, hi):
    rows = [h for h in hist if "reward" in h]
    k = max(1, len(rows) // 4)
    sel = rows[:k] if lo == 0 else rows[-k:]
    vals = [h[key] for h in sel for key in h if pred(key) and isinstance(h[key], (int, float))]
    return sum(vals) / len(vals) if vals else None


def main(dirs):
    runs, base = [], None
    for d in dirs:
        out = os.path.join(d, "out")
        for ev in sorted(glob.glob(os.path.join(out, "*", "eval", "eval.json"))):
            name = ev.split(os.sep)[-3]
            e = load(ev)
            if name == "base":
                base = e
                continue
            m = load(os.path.join(out, name, "metrics.json")) or {}
            runs.append((name, os.path.basename(d.rstrip("/")), m, e))
        for mj in sorted(glob.glob(os.path.join(out, "*", "metrics.json"))):
            name = mj.split(os.sep)[-2]
            if not any(r[0] == name for r in runs) and name != "smoke_tl":
                runs.append((name, os.path.basename(d.rstrip("/")), load(mj) or {}, None))
    print("| recipe | seed | run | steps/s | trainer peak GiB | reward Q1 | reward Q4 | ΔR | greedy acc | Δ greedy [95% CI] | "
          "sampled acc | Δ sampled [95% CI] | logp gap Q4 |")
    print("|" + "---|" * 13)
    agg = {}
    for name, rid, m, e in sorted(runs):
        recipe, seed = name.rsplit("_s", 1)
        hist = m.get("log_history") or []
        r1, r4 = m.get("reward_first_quarter"), m.get("reward_last_quarter")
        gap = hist_mean(hist, lambda k: "logp_difference" in k and k.endswith("mean"), 1, 1)
        gpu = m.get("gpu") or {}
        peak = gpu.get("max_allocated_gib")
        g = s = dg = ds = None
        dgc = dsc = (None, None)
        if e:
            g, s = e.get("greedy_acc"), e.get("mean_correct_frac")
            if base and e.get("greedy_correct") and base.get("greedy_correct"):
                d = [a - b for a, b in zip(e["greedy_correct"], base["greedy_correct"])]
                dg, dgc = sum(d) / len(d), boot_ci(d)
            if base and e.get("per_problem_correct") and base.get("per_problem_correct"):
                d = [(a - b) / 4 for a, b in zip(e["per_problem_correct"], base["per_problem_correct"])]
                ds, dsc = sum(d) / len(d), boot_ci(d, seed=1)
        f = lambda x, p=3: "–" if x is None else f"{x:.{p}f}"
        ci = lambda x, c: "–" if x is None else f"{x:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]"
        print(f"| {recipe} | {seed} | {rid} | {f(m.get('steps_per_s'))} | {f(peak, 1)} | {f(r1)} | {f(r4)} | "
              f"{'–' if r1 is None or r4 is None else f'{r4 - r1:+.3f}'} | {f(g)} | {ci(dg, dgc)} | {f(s)} | {ci(ds, dsc)} | "
              f"{f(gap, 4)} |")
        a = agg.setdefault(recipe, {"dR": [], "dg": [], "ds": [], "sps": []})
        if r1 is not None and r4 is not None:
            a["dR"].append(r4 - r1)
        if dg is not None:
            a["dg"].append(dg)
        if ds is not None:
            a["ds"].append(ds)
        if m.get("steps_per_s"):
            a["sps"].append(m["steps_per_s"])
    if base:
        print(f"\nbase: greedy acc {base.get('greedy_acc'):.3f}, sampled acc {base.get('mean_correct_frac'):.3f} "
              f"({base.get('n_problems')} problems)")
    print("\n| recipe | runs (ΔR) | mean ΔR | runs (eval) | mean Δ greedy (seed sd) | mean Δ sampled (seed sd) | steps/s |")
    print("|" + "---|" * 7)
    sd = lambda v: f" ({statistics.stdev(v):.3f})" if len(v) > 1 else ""
    mean = lambda v: "–" if not v else f"{sum(v) / len(v):+.3f}"
    for recipe, a in sorted(agg.items()):
        sps = "–" if not a["sps"] else f"{sum(a['sps']) / len(a['sps']):.3f}"
        print(f"| {recipe} | {len(a['dR'])} | {mean(a['dR'])} | {len(a['dg'])} | {mean(a['dg'])}{sd(a['dg'])} | "
              f"{mean(a['ds'])}{sd(a['ds'])} | {sps} |")


if __name__ == "__main__":
    main(sys.argv[1:])
