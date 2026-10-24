"""R1's table from the arms' result.json files: test capability against S with 95% CIs over the 512 test strings
(Clopper-Pearson for pass rates, percentile bootstrap for means), compute, and the information the reward could deliver."""
import argparse, glob, json, math, os

import numpy as np


def cp(k, n, a=0.05):
    """Clopper-Pearson interval for k successes in n, by bisection on the binomial tail (no scipy)."""
    def pmf(i, p):
        return math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p))

    def tail_ge(p):
        return sum(pmf(i, p) for i in range(k, n + 1))

    def tail_le(p):
        return sum(pmf(i, p) for i in range(0, k + 1))

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
            "pass1_sampled_ci": boot(s1) if p64 else cp(0, n), f"pass{k}": p64 / n, f"pass{k}_ci": cp(p64, n),
            "pos_greedy": float(np.mean(r["greedy_pos"])), "pos_greedy_ci": boot(r["greedy_pos"]),
            "pos_sampled": float(np.mean(r["sampled_pos"])), "pos_sampled_ci": boot(r["sampled_pos"])}


THRESHOLDS = (0.1, 0.25, 0.5, 0.9)


def to_reach(path):
    """The first validation record at or above each letter-accuracy threshold: its FLOPs and seconds of training."""
    if not os.path.exists(path):
        return {}
    ev = [x for x in json.load(open(path)) if "val_pos" in x]
    out = {}
    for t in THRESHOLDS:
        hit = next((x for x in ev if x["val_pos"] >= t), None)
        out[str(t)] = {"flops": hit["flops"], "t": hit["t"], "at": hit.get("it", hit.get("step"))} if hit else None
    return out


def fmt(m, key, pct=True):
    v, (lo, hi) = m[key], m[key + "_ci"]
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(v)} [{f(lo)}, {f(hi)}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs (each with out/<arm>/result.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", default="best", choices=["best", "final"])
    ap.add_argument("--a3", nargs="*", default=[], help="A3's run dirs (each with out/results/*.json)")
    ap.add_argument("--val", nargs="*", default=[], help="log.json of arms with no test report: their validation records")
    ap.add_argument("--procs-per-gpu", type=int, default=2, help="arms that shared one GPU; GPU-h is training wall time over this")
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
            log = os.path.join(os.path.dirname(f), "log.json")
            if not os.path.exists(log) and r.get("args", {}).get("prev_log"):
                log = r["args"]["prev_log"].replace("/workspace/research/runs", os.path.dirname(os.path.abspath(d.rstrip("/"))))
            row["to_reach"] = to_reach(log)
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
                     f"| {r['train_sec'] / 3600 / a.procs_per_gpu:.2f} | {c.get('flops', 0):.2e} | {c.get('info_pass_bits', 0):.3g} "
                     f"| {c.get('info_dense_bits', 0):.3g} |")
    lines += ["", "FLOPs (and wall seconds) to first reach a greedy validation letter accuracy:", "",
              "| arm | task | S | " + " | ".join(f"{int(100 * t)}%" for t in THRESHOLDS) + " |",
              "|---|---|---|" + "---|" * len(THRESHOLDS)]
    for r in rows:
        if r["arm"] == "base":
            continue
        S = f"{r['S_bytes'] / 1e3:g} kB" if r["arm"] == "es" else f"{r['S_bytes'] / 1e9:.2f} GB"
        cells = [f"{v['flops']:.2e} ({v['t']:.0f} s)" if v else "not reached" for v in (r["to_reach"].get(str(t)) for t in THRESHOLDS)]
        lines.append(f"| {r['name']} | {r['task']} | {S} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, f"table_{a.which}.md"), "w").write(table + "\n")
    json.dump(rows, open(os.path.join(a.out, "rows.json"), "w"), indent=1)
    if a.a3:
        a3_table(a.a3, a.out)
    if a.val:
        val_table(a.val, a.out)


def val_table(logs, out, n=256):
    """Arms stopped before their test report, on the 256 validation strings their log records (greedy): exact match with a
    Clopper-Pearson interval, letters with the conservative normal interval of a [0, 1] mean (variance at most p(1 - p))."""
    lines = ["| arm | run | at | FLOPs | val pass@1 greedy % | val letters right, greedy % | best at | best val pass@1 % | best val letters % |",
             "|---|---|---|---|---|---|---|---|---|"]

    def cells(x):
        k = round(x["val_exact"] * n)
        lo, hi = cp(k, n)
        p = x["val_pos"]
        h = 1.96 * math.sqrt(p * (1 - p) / n)
        return f"{100 * k / n:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]", f"{100 * p:.1f} [{100 * max(0, p - h):.1f}, {100 * min(1, p + h):.1f}]"

    for path in logs:
        ev = [x for x in json.load(open(path)) if "val_pos" in x]
        last, best = ev[-1], max(ev, key=lambda x: (x["val_exact"], x["val_pos"]))
        at = lambda x: x.get("it", x.get("step"))
        arm, run = os.path.basename(os.path.dirname(path)), os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        lines.append(f"| {arm} | {run} | {at(last)} | {last['flops']:.2e} | {' | '.join(cells(last))} | {at(best)} | {' | '.join(cells(best))} |")
    table = "\n".join(lines)
    print(table)
    open(os.path.join(out, "table_val.md"), "w").write(table + "\n")


def a3_table(dirs, out):
    """A3's finished jobs (SFT over its LR grid, then GRPO from the best SFT state), pooled over seeds per (task, S)."""
    groups = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "out", "results", "*.json"))):
            if f.endswith(".error.json"):
                continue
            r = json.load(open(f))
            groups.setdefault((r["job"]["task"], r["job"]["spec"]), []).append((os.path.basename(d.rstrip("/")), r))
    lines = ["| A3 arm | task | S | seeds | SFT pass@1 greedy % | SFT letters % | SFT+GRPO pass@1 greedy % | SFT+GRPO letters % | runs |",
             "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for (task, spec), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[1][0][1]["bits"])):
        row = {"task": task, "spec": spec, "S_bits": rs[0][1]["bits"], "seeds": [r["job"]["seed"] for _, r in rs],
               "runs": sorted({run for run, _ in rs})}
        cells = []
        for ph in ("sft", "rl"):
            ex = [x for _, r in rs for x in r[f"{ph}_test_exact"]]
            pos = [x for _, r in rs for x in r[f"{ph}_test_pos"]]
            k = int(sum(ex))
            row[f"{ph}_exact"], row[f"{ph}_exact_ci"] = k / len(ex), cp(k, len(ex))
            row[f"{ph}_pos"], row[f"{ph}_pos_ci"] = float(np.mean(pos)), boot(pos)
            cells += [fmt(row, f"{ph}_exact"), fmt(row, f"{ph}_pos")]
        S = row["S_bits"] / 8
        S = f"{S / 1e3:.3g} kB" if S < 1e8 else f"{S / 1e9:.2f} GB"
        lines.append(f"| {spec} | {task} | {S} | {len(rs)} | {' | '.join(cells)} | {', '.join(row['runs'])} |")
        rows.append(row)
    table = "\n".join(lines)
    print(table)
    open(os.path.join(out, "table_a3.md"), "w").write(table + "\n")
    json.dump(rows, open(os.path.join(out, "rows_a3.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
