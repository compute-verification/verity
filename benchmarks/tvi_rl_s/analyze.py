"""Tables for crux 1: score and fraction of full RL's gain against S, with two-sided 95% CIs, and base pass@k.

`python analyze.py OUT_DIR... > table.md` reads every result-*.json under the given run out dirs (grpo.py's results).

CIs: a paired bootstrap over eval problems (one resample of problem indices shared by base, the method and full), with the
seeds of each method also resampled with replacement (hierarchical), so the interval carries problem and seed noise.
Fraction of full gain = (acc_S - acc_base) / (acc_full - acc_base).
"""
import glob
import json
import random
import sys
from collections import defaultdict

ORDER = ["base", "tiny13", "kbit10", "kB10", "kB100", "lora16", "full"]
B = 4000


def fmt_bits(b):
    if b == 0:
        return "0"
    for unit, s in (("Gbit", 1e9), ("Mbit", 1e6), ("kbit", 1e3)):
        if b >= s:
            return f"{b / s:.3g} {unit}"
    return f"{b} bit"


def ci(xs):
    xs = sorted(xs)
    return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]


def main(dirs):
    res = defaultdict(lambda: defaultdict(list))
    passk = {}
    meta = {}
    for d in dirs:
        for p in sorted(glob.glob(f"{d}/**/result-*.json", recursive=True)):
            r = json.load(open(p))
            key = (r["task"], r["method"])
            if r["method"] == "base":
                res[r["task"]]["base"] = [r]
                if "passk" in r:
                    passk[r["task"]] = r["passk"]
                continue
            if any(x.get("seed") == r.get("seed") for x in res[r["task"]][r["method"]]):
                continue
            res[r["task"]][r["method"]].append(r)
            meta[key] = r
    out = {"tasks": {}}
    rng = random.Random(0)
    for task, by in sorted(res.items()):
        if "base" not in by:
            continue
        n = len(by["base"][0]["per_problem"])
        base = by["base"][0]["per_problem"]
        full = by.get("full", [])
        rows = []
        boots = {}
        for m in ORDER:
            runs = by.get(m, [])
            if not runs or any(len(r["per_problem"]) != n for r in runs):
                continue
            boots[m] = runs
        idx_sets = [[rng.randrange(n) for _ in range(n)] for _ in range(B)]
        seed_picks = {m: [[rng.randrange(len(rs)) for _ in rs] for _ in range(B)] for m, rs in boots.items()}

        def acc_of(m, b):
            rs = boots[m]
            idx = idx_sets[b]
            picks = seed_picks[m][b]
            return sum(sum(rs[s]["per_problem"][i] for i in idx) / n for s in picks) / len(picks)

        accs = {m: [acc_of(m, b) for b in range(B)] for m in boots}
        for m in ORDER:
            if m not in boots:
                continue
            rs = boots[m]
            point = sum(r["acc"] for r in rs) / len(rs)
            lo, hi = ci(accs[m])
            row = {"method": m, "S_bits": rs[0]["S_bits"], "n_params": rs[0].get("n_params", 0), "seeds": [r.get("seed", 0) for r in rs],
                   "acc_per_seed": [r["acc"] for r in rs], "acc": point, "acc_ci": [lo, hi], "steps": rs[0]["steps"],
                   "lr": rs[0].get("lr")}
            if full and m != "base":
                base_acc = rs and by["base"][0]["acc"]
                full_acc = sum(r["acc"] for r in full) / len(full)
                gain = full_acc - base_acc
                row["gain"] = point - base_acc
                row["frac_full_gain"] = (point - base_acc) / gain if gain else None
                fr = []
                for b in range(B):
                    g = accs["full"][b] - accs["base"][b]
                    fr.append((accs[m][b] - accs["base"][b]) / g if g > 1e-9 else float("nan"))
                fr = [x for x in fr if x == x]
                row["frac_ci"] = list(ci(fr)) if fr else None
                gd = [accs[m][b] - accs["base"][b] for b in range(B)]
                row["gain_ci"] = list(ci(gd))
            rows.append(row)
        out["tasks"][task] = {"n_problems": n, "rows": rows}
    out["passk"] = passk
    return out


def render(out):
    lines = []
    for task, t in out["tasks"].items():
        lines.append(f"\n### {task} (n = {t['n_problems']} eval problems, greedy)\n")
        lines.append("| S label | S (bits) | trainable values | steps | acc per seed | acc (95% CI) | gain (95% CI) | fraction of full gain (95% CI) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in t["rows"]:
            accs = ", ".join(f"{a:.3f}" for a in r["acc_per_seed"])
            a = f"{r['acc']:.3f} [{r['acc_ci'][0]:.3f}, {r['acc_ci'][1]:.3f}]"
            g = f"{r['gain']:+.3f} [{r['gain_ci'][0]:+.3f}, {r['gain_ci'][1]:+.3f}]" if "gain_ci" in r else ""
            f = (f"{r['frac_full_gain']:.2f} [{r['frac_ci'][0]:.2f}, {r['frac_ci'][1]:.2f}]"
                 if r.get("frac_ci") and r.get("frac_full_gain") is not None else "")
            lines.append(f"| {r['method']} | {fmt_bits(r['S_bits'])} ({r['S_bits']:.3g}) | {r['n_params']:,} | {r['steps']} | {accs} | {a} | {g} | {f} |")
    for task, pk in out["passk"].items():
        lines.append(f"\n### base pass@k, {task} ({pk['problems']} problems, n = {pk['n']}, T = {pk['temperature']}, top-p = {pk['top_p']})\n")
        ks = sorted(pk["pass_at"], key=int)
        lines.append("| " + " | ".join(f"pass@{k}" for k in ks) + " |")
        lines.append("|" + "---|" * len(ks))
        lines.append("| " + " | ".join(f"{pk['pass_at'][k]:.3f}" for k in ks) + " |")
        z = sum(1 for c in pk["counts"] if c == 0)
        lines.append(f"\nProblems never solved in {pk['n']} samples: {z} of {pk['problems']} ({z / pk['problems']:.1%}).")
    return "\n".join(lines)


if __name__ == "__main__":
    o = main(sys.argv[1:])
    print(render(o))
    json.dump(o, open("tables.json", "w"), indent=1)
