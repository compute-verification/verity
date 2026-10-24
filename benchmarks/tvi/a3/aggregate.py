"""A3's table: test exact-match (and position accuracy) against S and H, with hierarchical bootstrap CIs over seeds and test
strings, the fraction of full training's gain, and the checkpoint manifest for B1."""
import argparse, glob, json, os

import numpy as np

ORDER = ["xs:13", "hl:13", "xs:625", "hl:625", "xs:5000", "hl:5000", "xs:48552", "hl:50000", "lora16", "full"]
LABEL = {"full": "full", "lora16": "lora16", "xs:13": "xs-tiny", "xs:625": "xs-10kbit", "xs:5000": "xs-10kB",
         "xs:48552": "xs-100kB", "hl:13": "hl-tiny", "hl:625": "hl-10kbit", "hl:5000": "hl-10kB", "hl:50000": "hl-100kB"}


def boot(per_seed, B=2000, rng=None):
    """per_seed: list of arrays (one per seed) of per-string scores -> B bootstrap means (seeds, then strings)."""
    rng = rng or np.random.default_rng(0)
    k = len(per_seed)
    out = np.empty(B)
    for b in range(B):
        ss = rng.integers(0, k, k)
        out[b] = np.mean([per_seed[s][rng.integers(0, len(per_seed[s]), len(per_seed[s]))].mean() for s in ss])
    return out


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--base", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--node", default="vy-nebius-3")
    a = ap.parse_args()
    allrecs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.results, "*.json"))) if not f.endswith(".error.json")]
    recs = [r for r in allrecs if not r["job"].get("init")]
    sweeps = [r for r in allrecs if r["job"].get("init")]
    errs = sorted(glob.glob(os.path.join(a.results, "*.error.json")))
    base = json.load(open(a.base)) if a.base and os.path.exists(a.base) else {}
    model = recs[0]["model"] if recs else "Qwen/Qwen2.5-0.5B-Instruct"
    groups = {}
    for r in recs:
        j = r["job"]
        groups.setdefault((j["task"], j["order"], j["spec"]), []).append(r)
    rng = np.random.default_rng(0)
    rows, bs = [], {}
    for (task, order, spec), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], ORDER.index(kv[0][2]) if kv[0][2] in ORDER else 99)):
        rs = sorted(rs, key=lambda r: r["job"]["seed"])
        row = {"task": task, "order": order, "H_bits": rs[0]["H_bits"], "spec": spec, "S_label": LABEL.get(spec, spec),
               "S_bits": rs[0]["bits"], "seeds": [r["job"]["seed"] for r in rs], "sft_steps": rs[0]["sft_steps"],
               "rl_steps": rs[0]["rl_steps"], "sft_lr": [r["sft_lr"] for r in rs], "rl_best_step": [r["rl_best_step"] for r in rs],
               "sft_budget": rs[0].get("sft_budget", rs[0]["sft_steps"]), "sft_steps_per_seed": [r["sft_steps"] for r in rs],
               "rl_skipped": [bool(r.get("rl_skipped")) for r in rs]}
        for key in ["sft_test_exact", "sft_test_pos", "rl_test_exact", "rl_test_pos", "rl_final_test_exact"]:
            per = [np.array(r[key], dtype=float) for r in rs]
            b = boot(per, rng=rng)
            bs[(task, spec, key)] = b
            row[key] = float(np.mean([p.mean() for p in per]))
            row[key + "_ci"] = ci(b)
            row[key + "_per_seed"] = [float(p.mean()) for p in per]
        rows.append(row)
    bkey = f"{model}|"
    for row in rows:
        b0 = base.get(f"{model}|{row['order']}", {})
        row["base_greedy_exact"] = b0.get("greedy_exact", 0.0)
        row["base_pass64"] = b0.get("pass_at_k")
        for ph in ["sft", "rl"]:
            key = f"{ph}_test_exact"
            full = bs.get((row["task"], "full", key))
            if full is None:  # a long-budget rerun (CL) is measured against its task's (C) full run
                full = bs.get((row["task"][0], "full", key))
            if full is None:
                continue
            num = bs[(row["task"], row["spec"], key)] - row["base_greedy_exact"]
            den = full - row["base_greedy_exact"]
            frac = num / np.where(np.abs(den) < 1e-9, np.nan, den)
            fm = full.mean() - row["base_greedy_exact"]
            row[f"{ph}_frac_full"] = float((row[key] - row["base_greedy_exact"]) / fm) if abs(fm) > 1e-9 else None
            row[f"{ph}_frac_full_ci"] = (float(np.nanpercentile(frac, 2.5)), float(np.nanpercentile(frac, 97.5)))
    json.dump({"rows": rows, "errors": [os.path.basename(e) for e in errs], "base": base}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)

    def f(x, c):
        return f"{x:.3f} [{c[0]:.3f}, {c[1]:.3f}]"

    lines = ["| task | H bits | S | S bits | S/H | seeds | SFT steps | SFT exact [95% CI] | SFT frac of full [CI] | RL exact [95% CI] | RL frac of full [CI] | RL best step | SFT pos-acc | RL pos-acc |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fr = lambda ph: (f"{r[ph + '_frac_full']:.2f} [{r[ph + '_frac_full_ci'][0]:.2f}, {r[ph + '_frac_full_ci'][1]:.2f}]"
                         if r.get(ph + "_frac_full") is not None else "-")
        steps = "/".join(str(s) for s in r["sft_steps_per_seed"])
        lines.append(f"| {r['task']} (o={r['order']}) | {r['H_bits']:.0f} | {r['S_label']} | {r['S_bits']:.3g} | {r['S_bits'] / r['H_bits']:.3g} | "
                     f"{len(r['seeds'])} | {steps} | "
                     f"{f(r['sft_test_exact'], r['sft_test_exact_ci'])} | {fr('sft')} | {f(r['rl_test_exact'], r['rl_test_exact_ci'])} | "
                     f"{fr('rl')} | {'/'.join(str(s) for s in r['rl_best_step'])} | {r['sft_test_pos']:.3f} | {r['rl_test_pos']:.3f} |")
    lines += ["", "| task | H bits | smallest S with RL frac >= 0.3 | >= 0.8 | >= 0.95 |", "|---|---|---|---|---|"]
    for task in sorted({r["task"] for r in rows}):
        rs = sorted([r for r in rows if r["task"] == task and r.get("rl_frac_full") is not None], key=lambda r: r["S_bits"])
        if not rs:
            continue
        cell = lambda th: next((f"{r['S_label']} ({r['S_bits']:.3g} bit, {r['rl_frac_full']:.2f})" for r in rs if r["rl_frac_full"] >= th), "none")
        lines.append(f"| {task} | {rs[0]['H_bits']:.0f} | {cell(0.3)} | {cell(0.8)} | {cell(0.95)} |")
    if sweeps:
        lines += ["", "RL learning-rate sweep from saved SFT checkpoints (seed 0; reported = best validation step, step 0 = SFT):", "",
                  "| task | S | S bits | RL lr (x SFT lr) | SFT exact | RL exact [95% CI] | RL best step | RL final exact | SFT pos | RL pos | RL final pos |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(sweeps, key=lambda r: (r["job"]["task"], ORDER.index(r["job"]["spec"]) if r["job"]["spec"] in ORDER else 99, r["job"]["rl_lr_mult"])):
            j = r["job"]
            m = lambda k: float(np.mean(r[k]))
            b = boot([np.array(r["rl_test_exact"], dtype=float)], rng=rng)
            lines.append(f"| {j['task']} | {LABEL.get(j['spec'], j['spec'])} | {r['bits']:.3g} | {r['rl_lr']:.3g} ({j['rl_lr_mult']}) | "
                         f"{m('sft_test_exact'):.3f} | {f(m('rl_test_exact'), ci(b))} | {r['rl_best_step']} | {m('rl_final_test_exact'):.3f} | "
                         f"{m('sft_test_pos'):.3f} | {m('rl_test_pos'):.3f} | {m('rl_final_test_pos'):.3f} |")
    blines = ["| model | task | H bits | strings | k | pass@k | 95% upper (0 hits) | greedy exact | pos-acc (sampled) |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in base.items():
        up = 1 - 0.05 ** (1 / v["n_strings"]) if v["n_pass"] == 0 else float("nan")
        blines.append(f"| {v['model']} | o={v['order']} | {v['H_bits']:.0f} | {v['n_strings']} | {v['k']} | {v['pass_at_k']:.4f} | {up:.4f} | "
                      f"{v['greedy_exact']:.4f} | {v['pos_acc_sampled']:.3f} |")
    open(os.path.join(a.out, "table.md"), "w").write("\n".join(blines) + "\n\n" + "\n".join(lines) + "\n" +
                                                     ("\nerrors: " + ", ".join(os.path.basename(e) for e in errs) + "\n" if errs else ""))
    print("\n".join(blines))
    print("\n".join(lines))
    man = []
    for r in recs:
        j = r["job"]
        for ph in ["sft", "rl"]:
            p = r.get(f"{ph}_ckpt")
            if not p:
                continue
            man.append({"node": a.node, "path": p, "task": j["task"], "order": j["order"], "H_bits": r["H_bits"],
                        "spec": j["spec"], "S_label": LABEL.get(j["spec"], j["spec"]), "S_bits": r["bits"], "phase": ph,
                        "seed": j["seed"], "steps": int(os.path.basename(p)[4:]),
                        "score_test_exact": float(np.mean(r[f"{ph}_test_exact"])),
                        "score_test_pos": float(np.mean(r[f"{ph}_test_pos"])), "base_model": r["model"],
                        "merged_dtype": "bfloat16" if j["spec"] == "full" else "float32"})
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
