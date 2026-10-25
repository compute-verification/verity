"""Tables for crux 1: score and fraction of full RL's gain against S, with two-sided 95% CIs, and pass@k.

`python analyze.py OUT_DIR... > table.md` reads every result-*.json (grpo.py: greedy accuracy at the end of training) and
ckpteval-*.json (evalckpts.py: greedy and sampled accuracy of saved checkpoints) under the given dirs.

CIs: a paired bootstrap over eval problems (one resample of problem indices shared by base, the method and full), with the
seeds of each method also resampled with replacement (hierarchical), so the interval carries problem and seed noise.
Fraction of full gain = (acc_S - acc_base) / (acc_full - acc_base).
"""
import glob
import json
import random
import re
import sys
from collections import defaultdict

ORDER = ["base", "temp", "tiny13", "kbit10", "kB10", "kB100", "lora1h", "lora1", "lora2", "lora16", "full"]
B = 4000
LR_VARIANT = re.compile(r"^(.*)lr([0-9.e+-]+)$")


def remap(task, method):
    """A learning-rate variant (task 'gsm8klr1e-2') is a row of its task's table: method 'tiny13 lr=1e-2'."""
    m = LR_VARIANT.match(task)
    return (m.group(1), f"{method} lr={m.group(2)}") if m else (task, method)


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


def table(by, rng):
    """by: method -> list of runs {seed, S_bits, n_params, steps, acc, per_problem}. Returns rows with CIs."""
    if "base" not in by:
        return None
    n = len(by["base"][0]["per_problem"])
    boots = {m: rs for m, rs in by.items() if rs and all(len(r["per_problem"]) == n for r in rs)}
    idx_sets = [[rng.randrange(n) for _ in range(n)] for _ in range(B)]
    seed_picks = {m: [[rng.randrange(len(rs)) for _ in rs] for _ in range(B)] for m, rs in boots.items()}

    def acc_of(m, b):
        rs, idx, picks = boots[m], idx_sets[b], seed_picks[m][b]
        return sum(sum(rs[s]["per_problem"][i] for i in idx) / n for s in picks) / len(picks)

    accs = {m: [acc_of(m, b) for b in range(B)] for m in boots}
    base_acc = boots["base"][0]["acc"]
    full = boots.get("full", [])
    rows = []
    for m in ORDER + sorted(set(boots) - set(ORDER)):
        if m not in boots:
            continue
        rs = boots[m]
        point = sum(r["acc"] for r in rs) / len(rs)
        row = {"method": m, "S_bits": rs[0]["S_bits"], "n_params": rs[0].get("n_params", 0),
               "seeds": [r.get("seed", 0) for r in rs], "acc_per_seed": [r["acc"] for r in rs], "acc": point,
               "acc_ci": list(ci(accs[m])), "steps": rs[0].get("steps", 0), "modes": sorted({r.get("mode", "merged") for r in rs})}
        if m != "base":
            row["gain"] = point - base_acc
            row["gain_ci"] = list(ci([accs[m][b] - accs["base"][b] for b in range(B)]))
            if full:
                full_acc = sum(r["acc"] for r in full) / len(full)
                g = full_acc - base_acc
                row["frac_full_gain"] = (point - base_acc) / g if g > 0 else None
                row["full_gain_not_positive"] = g <= 0
                fr = []
                for b in range(B):
                    gb = accs["full"][b] - accs["base"][b]
                    if gb > 1e-9:
                        fr.append((accs[m][b] - accs["base"][b]) / gb)
                row["frac_ci"] = list(ci(fr)) if len(fr) > B // 2 else None
                row["frac_boot_valid"] = len(fr) / B
        rows.append(row)
    return {"n_problems": n, "rows": rows}


def load(dirs):
    greedy = defaultdict(lambda: defaultdict(list))
    sampled = defaultdict(lambda: defaultdict(list))
    ck_greedy = defaultdict(lambda: defaultdict(list))
    passk = {}
    for d in dirs:
        for p in sorted(glob.glob(f"{d}/**/result-*.json", recursive=True)):
            r = json.load(open(p))
            r["task"], r["method"] = remap(r["task"], r["method"])
            if r.get("ckpt"):
                if "passk" in r:
                    passk[f"{r['task']} {r['method']} step{r['steps']}"] = r["passk"]
                continue
            if r["method"] == "base":
                greedy[r["task"]]["base"] = [r]
                if "passk" in r:
                    passk[f"{r['task']} base"] = r["passk"]
                continue
            if any(x.get("seed") == r.get("seed") for x in greedy[r["task"]][r["method"]]):
                continue
            greedy[r["task"]][r["method"]].append(r)
        for p in sorted(glob.glob(f"{d}/**/ckpteval-*.json", recursive=True)):
            r = json.load(open(p))
            r["task"], r["method"] = remap(r["task"], r["method"])
            if "passk" in r:
                passk[f"{r['task']} {r['method']} seed{r['seed']} step{r['step']}"] = r["passk"]
            run = {"seed": r["seed"], "S_bits": r["S_bits"], "n_params": r["n_params"], "steps": r["step"], "step": r["step"],
                   "prio": 1 if r.get("mode") == "vllm-lora" else 0, "mode": r.get("mode", "merged")}
            s = dict(run, acc=r["sampled"]["acc"], per_problem=r["sampled"]["per_problem"], T=r["sampled"]["temperature"],
                     n=r["sampled"]["n"])
            g = dict(run, acc=r["greedy_acc"], per_problem=r["greedy_per_problem"])
            for store, x in ((sampled, s), (ck_greedy, g)):
                lst = store[r["task"]][r["method"]]
                old = [y for y in lst if y["seed"] == x["seed"]]
                if old and (old[0]["step"], old[0]["prio"]) >= (x["step"], x["prio"]):
                    continue
                lst[:] = [y for y in lst if y["seed"] != x["seed"]] + [x]
    return greedy, sampled, ck_greedy, passk


def online(dirs, window=25):
    """Mean training reward over the last `window` steps of each run: every prompt is fresh in the first epoch, so this is the
    policy's held-out accuracy at the sampling temperature (T = 1). Runs with the same seed draw the same prompts at every step,
    so the bootstrap is paired: each resample draws one multiset of seeds and, per seed, one multiset of window steps, shared by
    every method (a method that lacks every drawn seed uses its own)."""
    runs = defaultdict(lambda: defaultdict(dict))
    for d in dirs:
        for p in sorted(glob.glob(f"{d}/**/*-seed*.jsonl", recursive=True)):
            steps = {}
            meta = {}
            finished = False
            for line in open(p):
                e = json.loads(line)
                if e.get("event") == "setup":
                    meta = e
                if e.get("event") == "step":
                    steps[e["step"]] = e["reward"]
                finished |= e.get("event") == "done"
            if not steps or not meta or not finished:
                continue
            name = p.rsplit("/", 1)[1][:-len(".jsonl")]
            task, method, seed = name.rsplit("-", 2)
            last = max(steps)
            w = [steps[s] for s in range(last - window + 1, last + 1) if s in steps]
            prev = runs[task][method].get(seed)
            if prev and prev["last"] >= last:
                continue
            runs[task][method][seed] = {"last": last, "w": w, "acc": sum(w) / len(w), "S_bits": meta["S_bits"],
                                        "n_params": meta["n_params"], "first": steps.get(1), "prompts": meta.get("prompts", 32)}
    rng = random.Random(1)
    out = {}
    for task, by in sorted(runs.items()):
        seeds = sorted({s for rs in by.values() for s in rs})
        draws = [([seeds[rng.randrange(len(seeds))] for _ in seeds],
                  {s: [rng.randrange(window) for _ in range(window)] for s in seeds}) for _ in range(B)]
        rows = []
        boot = {}
        for m in ORDER:
            if m not in by:
                continue
            rs = by[m]
            bs = []
            for picks, idx in draws:
                own = [s for s in picks if s in rs] or list(rs)
                bs.append(sum(sum(rs[s]["w"][i % len(rs[s]["w"])] for i in idx[s]) / len(idx[s]) for s in own) / len(own))
            boot[m] = bs
            r0 = next(iter(rs.values()))
            rows.append({"method": m, "S_bits": r0["S_bits"], "n_params": r0["n_params"], "steps": r0["last"],
                         "seeds": sorted(rs), "acc_per_seed": [rs[s]["acc"] for s in sorted(rs)],
                         "acc": sum(r["acc"] for r in rs.values()) / len(rs), "acc_ci": list(ci(bs)),
                         "step1": {s: r["first"] for s, r in rs.items()}})
        out[task] = {"rows": rows, "boot": boot, "window": window,
                     "prompts": "/".join(sorted({str(r["prompts"]) for rs in by.values() for r in rs.values()}))}
    return out


def main(dirs):
    greedy, sampled, ck_greedy, passk = load(dirs)
    rng = random.Random(0)
    out = {"greedy": {}, "ckpt_greedy": {}, "sampled": {}, "passk": passk, "online": {}}
    on = online(dirs)
    for task, t in on.items():
        # The base must be read on training prompts: the Instruct model scores higher on the train splits than on the test
        # sets (0.71 vs 0.65 on GSM8K at T = 1), so the test-set base would inflate every online gain.
        firsts = {}
        for r in t["rows"]:
            firsts.update({s: x for s, x in r["step1"].items() if x is not None})
        base = sum(firsts.values()) / len(firsts) if firsts else None
        base_boot = [base] * B
        t["base_source"] = (f"step-1 training reward, before any update ({len(firsts)} seeds x {t['prompts']} prompts x 8 "
                            "samples; its noise is not in the CIs)")
        full = [r for r in t["rows"] if r["method"] == "full"]
        for r in t["rows"]:
            if base is None:
                continue
            r["base_T1"] = base
            r["gain"] = r["acc"] - base
            r["gain_ci"] = list(ci([a - b for a, b in zip(t["boot"][r["method"]], base_boot)]))
            if full:
                g = full[0]["acc"] - base
                r["frac_full_gain"] = r["gain"] / g if g else None
                fr = [(a - b) / (f - b) for a, f, b in zip(t["boot"][r["method"]], t["boot"]["full"], base_boot) if f - b > 1e-9]
                r["frac_ci"] = list(ci(fr)) if len(fr) > B // 2 else None
        out["online"][task] = {"rows": t["rows"], "window": t["window"], "base_T1": base, "base_source": t["base_source"]}
    for task, by in sorted(greedy.items()):
        t = table(by, rng)
        if t:
            out["greedy"][task] = t
    for task, by in sorted(ck_greedy.items()):
        t = table(by, rng)
        if t:
            out["ckpt_greedy"][task] = t
    for task, by in sorted(sampled.items()):
        t = table(by, rng)
        if t:
            out["sampled"][task] = t
            out["sampled"][task]["T"] = by["base"][0].get("T")
            out["sampled"][task]["n"] = by["base"][0].get("n")
    return out


def render(out):
    lines = []
    for kind, title in (("greedy", "greedy accuracy at the end of training"),
                        ("ckpt_greedy", "greedy accuracy, final checkpoints"),
                        ("sampled", "sampled accuracy (mean over n samples at T, the GRPO objective), final checkpoints")):
        for task, t in out[kind].items():
            extra = f", n = {t.get('n')}, T = {t.get('T')}" if kind == "sampled" else ""
            lines.append(f"\n### {task}: {title} ({t['n_problems']} eval problems{extra})\n")
            lines.append("| S label | S (bits) | trainable values | steps | acc per seed | acc (95% CI) | gain over base (95% CI) | fraction of full gain (95% CI) |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for r in t["rows"]:
                accs = ", ".join(f"{a:.3f}" for a in r["acc_per_seed"])
                a = f"{r['acc']:.3f} [{r['acc_ci'][0]:.3f}, {r['acc_ci'][1]:.3f}]"
                g = f"{r['gain']:+.3f} [{r['gain_ci'][0]:+.3f}, {r['gain_ci'][1]:+.3f}]" if "gain_ci" in r else ""
                f = "n/a (full's gain is not positive)" if r.get("full_gain_not_positive") else ""
                if r.get("frac_full_gain") is not None:
                    f = f"{r['frac_full_gain']:.2f}"
                    f += f" [{r['frac_ci'][0]:.2f}, {r['frac_ci'][1]:.2f}]" if r.get("frac_ci") else " [unbounded]"
                name = r["method"] + (" (unmerged, vLLM LoRA)" if "vllm-lora" in r.get("modes", []) else "")
                lines.append(f"| {name} | {fmt_bits(r['S_bits'])} ({r['S_bits']:.3g}) | {r['n_params']:,} | {r['steps']} | {accs} | {a} | {g} | {f} |")
    for task, t in out["online"].items():
        b = t["base_T1"]
        lines.append(f"\n### {task}: online accuracy at T = 1 (mean training reward over the last {t['window']} steps, fresh prompts; "
                     f"base = {'%.3f' % b if b is not None else 'pending'}, {t['base_source']})\n")
        lines.append("| S label | S (bits) | trainable values | steps | acc per seed | acc (95% CI) | gain over base (95% CI) | fraction of full gain (95% CI) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in t["rows"]:
            accs = ", ".join(f"{a:.3f}" for a in r["acc_per_seed"])
            f = ""
            if r.get("frac_full_gain") is not None:
                f = f"{r['frac_full_gain']:.2f}" + (f" [{r['frac_ci'][0]:.2f}, {r['frac_ci'][1]:.2f}]" if r.get("frac_ci") else "")
            g = f"{r['gain']:+.3f} [{r['gain_ci'][0]:+.3f}, {r['gain_ci'][1]:+.3f}]" if "gain" in r else ""
            lines.append(f"| {r['method']} | {fmt_bits(r['S_bits'])} ({r['S_bits']:.3g}) | {r['n_params']:,} | {r['steps']} | {accs} | "
                         f"{r['acc']:.3f} [{r['acc_ci'][0]:.3f}, {r['acc_ci'][1]:.3f}] | {g} | {f} |")
    for name, pk in out["passk"].items():
        lines.append(f"\n### pass@k, {name} ({pk['problems']} problems, n = {pk['n']}, T = {pk['temperature']}, top-p = {pk['top_p']})\n")
        ks = sorted(pk["pass_at"], key=int)
        lines.append("| " + " | ".join(f"pass@{k}" for k in ks) + " |")
        lines.append("|" + "---|" * len(ks))
        lines.append("| " + " | ".join(f"{pk['pass_at'][k]:.3f}" for k in ks) + " |")
        z = sum(1 for c in pk["counts"] if c == 0)
        lines.append(f"\nNever solved in {pk['n']} samples: {z} of {pk['problems']} ({z / pk['problems']:.1%}).")
    return "\n".join(lines)


if __name__ == "__main__":
    o = main(sys.argv[1:])
    print(render(o))
    json.dump(o, open("tables.json", "w"), indent=1)
