"""R1's table from the arms' result.json files: test capability against S with 95% CIs over the 512 test strings
(Clopper-Pearson for pass rates, percentile bootstrap for means), compute, and the information the reward could deliver."""
import argparse, glob, json, math, os

import numpy as np


def cp(k, n, a=0.05):
    """Clopper-Pearson interval for k successes in n, by bisection on the binomial tail (no scipy)."""
    def tail_ge(p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def tail_le(p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))

    def solve(f, target, lo, hi, increasing):
        for _ in range(60):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lo = 0.0 if k == 0 else solve(tail_ge, a / 2, 0.0, 1.0, True)
    hi = 1.0 if k == n else solve(tail_le, a / 2, 0.0, 1.0, False)
    return lo, hi


def boot(x, B=4000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def metrics(r):
    n = len(r["greedy_exact"])
    k = r["k"]
    g = int(sum(r["greedy_exact"]))
    p64 = sum(1 for c in r["sampled_npass"] if c > 0)
    s1 = [c / k for c in r["sampled_npass"]]
    return {"n": n, "pass1_greedy": g / n, "pass1_greedy_ci": cp(g, n), "pass1_sampled": float(np.mean(s1)),
            "pass1_sampled_ci": boot(s1), f"pass{k}": p64 / n, f"pass{k}_ci": cp(p64, n),
            "pos_greedy": float(np.mean(r["greedy_pos"])), "pos_greedy_ci": boot(r["greedy_pos"]),
            "pos_sampled": float(np.mean(r["sampled_pos"])), "pos_sampled_ci": boot(r["sampled_pos"])}


def fmt(m, key, pct=True):
    v, (lo, hi) = m[key], m[key + "_ci"]
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(v)} [{f(lo)}, {f(hi)}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs (each with out/<arm>/result.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", default="best", choices=["best", "final"])
    a = ap.parse_args()
    rows = []
    for d in a.runs:
        run_id = os.path.basename(d.rstrip("/"))
        for f in sorted(glob.glob(os.path.join(d, "out", "*", "result.json"))):
            r = json.load(open(f))
            arm = os.path.basename(os.path.dirname(f))
            row = {"run": run_id, "name": arm, "arm": r["arm"], "reward": r.get("reward", "sample" if r["arm"] == "grpo" else ""),
                   "task": r["task"], "H_bits": r["H_bits"], "S_bytes": r.get("S_bytes", 0), "K": r.get("K"),
                   "iters": r.get("iters", 0), "train_sec": r.get("train_sec", 0), "best_it": r.get("best_it"),
                   "alpha": r.get("alpha"), "lr": r.get("lr"), "counter": r.get("counter", {}),
                   "seqs_per_iter": r.get("seqs_per_iter"), "procs_per_gpu": None}
            for w in ("best", "final"):
                row[w] = metrics(r[w])
            rows.append(row)
    rows.sort(key=lambda r: (r["task"], {"base": 0, "es": 1, "grpo": 2}[r["arm"]], r["reward"], r["S_bytes"], r["name"]))
    lines = ["| arm | task | S | reward | iters | best it | pass@1 greedy % | pass@1 sampled % | pass@64 % | letters right, greedy % "
             "| letters right, sampled % | GPU-h | FLOPs | info pass/fail (bits) | info dense (bits) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r[a.which]
        S = "0" if r["arm"] == "base" else (f"{r['S_bytes'] / 1e3:g} kB" if r["arm"] == "es" else f"{r['S_bytes'] / 1e9:.2f} GB")
        c = r["counter"]
        lines.append(f"| {r['name']} | {r['task']} | {S} | {r['reward']} | {r['iters']} | {r['best_it']} | {fmt(m, 'pass1_greedy')} "
                     f"| {fmt(m, 'pass1_sampled')} | {fmt(m, 'pass64')} | {fmt(m, 'pos_greedy')} | {fmt(m, 'pos_sampled')} "
                     f"| {r['train_sec'] / 3600:.2f} | {c.get('flops', 0):.2e} | {c.get('info_pass_bits', 0):.3g} "
                     f"| {c.get('info_dense_bits', 0):.3g} |")
    table = "\n".join(lines)
    print(table)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, f"table_{a.which}.md"), "w").write(table + "\n")
    json.dump(rows, open(os.path.join(a.out, "rows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
