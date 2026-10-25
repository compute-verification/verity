"""B1's 1-5 MB readout: capability (A2's analyze.py over A2's runs and mb.sh's), F and the betas (summarize.py's F_beta.json),
and the two-crossing cliff rho = 4 eta kappa_tr (1 + (d / 1536) F / h) at Qwen2.5-1.5B's width and at d = 7168.

  python mb_table.py SUMMARY_DIR RUN_DIR [RUN_DIR ...] > mb.md      (also writes mb.json beside SUMMARY_DIR's F_beta.json)

RUN_DIRs hold result-*.json, ckpteval-*.json and *-seed*.jsonl (A2's runs and mb.sh's out/). The T = 1 table reads a LoRA arm
served unmerged (vLLM's LoRA path) where there is one, as A2's lora16 row does; the merged in-job T = 1 reading is beside it.
"""
import json, os, random, shutil, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2"))
import analyze  # noqa: E402

METHODS = ("kB100", "lora1h", "lora1", "lora2", "lora16", "full")
NEW = ("lora1h", "lora1", "lora2")
H, KAPPA, ETA, D0 = 17.0, 1 / 3, 1.0, 1536
WIDTHS = (1536, 7168)
FMTS = ("bf16", "fp8t", "nvfp4")
COL = None


def rho(F, d):
    return 4 * ETA * KAPPA * (1 + (d / D0) * F / H)


def merged_only(dirs):
    """a copy of the run dirs without the unmerged evals, so analyze.py reads every arm's in-job (merged) T = 1 eval"""
    tmp = tempfile.mkdtemp(prefix="mb-merged-")
    for i, d in enumerate(dirs):
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".json") or f.endswith(".jsonl"):
                    if f.startswith("ckpteval-") and f.endswith("-unmerged.json"):
                        continue
                    dst = os.path.join(tmp, str(i), os.path.relpath(root, d))
                    os.makedirs(dst, exist_ok=True)
                    shutil.copy(os.path.join(root, f), dst)
    return tmp


def rows_by_method(t):
    return {r["method"]: r for r in (t or {}).get("rows", [])}


def fmt_frac(r):
    if not r or r.get("frac_full_gain") is None:
        return "–"
    lo_hi = r.get("frac_ci")
    return f"{r['frac_full_gain']:.2f}" + (f" [{lo_hi[0]:.2f}, {lo_hi[1]:.2f}]" if lo_hi else "")


def fmt_acc(r):
    return "–" if not r else f"{r['acc']:.3f} ({', '.join(f'{a:.3f}' for a in r['acc_per_seed'])})"


def mb(bits):
    return f"{bits / 8e6:.3g} MB" if bits >= 8e6 else f"{bits / 8e3:.3g} kB"


def main():
    sdir, dirs = sys.argv[1], sys.argv[2:]
    random.seed(0)
    cap = analyze.main(dirs)
    tmp = merged_only(dirs)
    cap_m = analyze.main([os.path.join(tmp, x) for x in sorted(os.listdir(tmp))])
    shutil.rmtree(tmp)
    g = rows_by_method(cap["greedy"].get("gsm8k"))
    s = rows_by_method(cap["sampled"].get("gsm8k"))
    sm = rows_by_method(cap_m["sampled"].get("gsm8k"))
    on = rows_by_method(cap["online"].get("gsm8k"))
    F = json.load(open(os.path.join(sdir, "F_beta.json")))
    C = F["entry_format_columns"]
    ents = [e for e in F["entries"] if e["fam"] == "A2" and e["cond"].startswith("gsm8k:")
            and e["cond"].split(":")[1] in METHODS]
    out = {"capability": {}, "counted": [], "cliff": {}}

    print("### Capability: fraction of full RL's gain on GSM8K (Qwen2.5-1.5B-Instruct, 150 GRPO steps, 1319 test problems)\n")
    print("Paired hierarchical bootstrap over problems and seeds (A2's analyze.py, B = 4000). T = 1: n = 4 samples per problem.\n")
    print("| method | S | trainable values | greedy acc (per seed) | greedy fraction | T = 1 acc (per seed) | T = 1 fraction "
          "| T = 1 fraction, merged in-job | online T = 1 fraction (last 25 steps) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in ("base",) + METHODS:
        r = g.get(m) or s.get(m)
        if not r:
            continue
        sr = s.get(m)
        tag = " (unmerged)" if sr and "vllm-lora" in sr.get("modes", []) else ""
        print(f"| {m} | {mb(r['S_bits']) if r['S_bits'] else '0'} | {r.get('n_params', 0):,} | {fmt_acc(g.get(m))} | "
              f"{fmt_frac(g.get(m))} | {fmt_acc(sr)}{tag} | {fmt_frac(sr)} | {fmt_frac(sm.get(m))} | {fmt_frac(on.get(m))} |")
        out["capability"][m] = {k: {kk: x.get(kk) for kk in ("acc", "acc_per_seed", "frac_full_gain", "frac_ci", "gain", "gain_ci",
                                                         "modes", "seeds")}
                                for k, x in (("greedy", g.get(m)), ("T1", sr), ("T1_merged", sm.get(m)), ("online", on.get(m)))
                                if x}

    print("\n### F at the cheapest block boundary and the betas, per checkpoint\n")
    print("F in bits per token (fresh forward residual at the FP8-cheapest decoder-layer output); rank = that residual's "
          "effective rank (90% / 99% energy); beta in bits per value (beta_v per counted value, beta_v|x the output given "
          "theta's weights on theta''s input, beta_x per input coordinate, beta_w per weight of an updated matrix, the smaller "
          "of the compressors' and S/N_w).\n")
    print("| method | seed | step | S | KL | F bf16 | F fp8 | F nvfp4 | cheapest layer | rank r90 / r99 | keep-90% rank (fp8, cheapest probed) "
          "| beta_x fp8 | beta_v fp8 | beta_v given x fp8 | beta_w fp8 | beta_x bf16 | beta_v bf16 | beta_v given x bf16 | beta_w bf16 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = {m: i for i, m in enumerate(METHODS)}
    for e in sorted(ents, key=lambda e: (order[e["cond"].split(":")[1]], e["cond"], e["step"])):
        m, seed = e["cond"].split(":")[1], e["cond"].split(":")[-1]
        x = {f: dict(zip(C, e[f])) for f in FMTS}
        apx = e.get("apx", {}).get("fp8")
        print(f"| {m} | {seed[4:]} | {e['step']} | {mb(e['S_bits'])} | {e['kl']} | {x['bf16']['Ff_min']} | {x['fp8t']['Ff_min']} | "
              f"{x['nvfp4']['Ff_min']} | {e['cheapest']['layer']} | {e['cheapest']['r90']} / {e['cheapest']['r99']} | "
              f"{apx[1] if apx else '–'} (layer {apx[0] if apx else '–'}) | {x['fp8t']['bx']} | {x['fp8t']['bv']} | "
              f"{x['fp8t']['bv_gx']} | {x['fp8t']['bw_min']} | {x['bf16']['bx']} | {x['bf16']['bv']} | {x['bf16']['bv_gx']} | "
              f"{x['bf16']['bw_min']} |")
        out["counted"].append({"method": m, "seed": seed, "step": e["step"], "S_bits": e["S_bits"], "kl": e["kl"],
                               "cheapest": e["cheapest"], "apx_fp8": apx, **{f: x[f] for f in FMTS}})

    print("\n### The two-crossing cliff rho = 4 eta kappa_tr (1 + (d / 1536) F / h), kappa_tr = 1/3, h = 17, eta = 1\n")
    print("F from the final (step-150) checkpoints, mean over seeds [min, max]; the d = 7168 column scales F by 7168 / 1536.\n")
    print("| method | S | F fp8 | rho fp8, d = 1536 | rho fp8, d = 7168 | F nvfp4 | rho nvfp4, d = 1536 | rho nvfp4, d = 7168 "
          "| F bf16 | rho bf16, d = 1536 | rho bf16, d = 7168 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in METHODS:
        fin = [e for e in ents if e["cond"].split(":")[1] == m and e["step"] == 150]
        if not fin:
            continue
        row, rec = [m, mb(fin[0]["S_bits"])], {"S_bits": fin[0]["S_bits"], "n": len(fin)}
        for f in ("fp8t", "nvfp4", "bf16"):
            v = [dict(zip(C, e[f]))["Ff_min"] for e in fin]
            mean = sum(v) / len(v)
            r = {d: [rho(mean, d), rho(min(v), d), rho(max(v), d)] for d in WIDTHS}
            row += [f"{mean:.0f} [{min(v)}, {max(v)}]"] + [f"{r[d][0]:.0f} [{r[d][1]:.0f}, {r[d][2]:.0f}]" for d in WIDTHS]
            rec[f] = {"F": v, "F_mean": mean, **{f"rho_d{d}": r[d] for d in WIDTHS}}
        out["cliff"][m] = rec
        print("| " + " | ".join(row) + " |")
    json.dump(out, open(os.path.join(sdir, "mb.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
