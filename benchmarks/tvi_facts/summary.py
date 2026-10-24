"""The readout table from run out dirs: one row per (model, N, state size) at the learning rate the sweep chose,
every seed shown, with bootstrap intervals over people; then fresh bits per learned bit.

`python summary.py OUT_DIR...` prints Markdown and writes summary.json beside the first dir.
"""
import glob
import json
import os
import sys

ORDER = ["sub10kbit", "sub10kB", "sub100kB", "sub1MB", "lora16", "full"]


def kib(b):
    if b < 8 * 1024:
        return f"{b / 1000:.1f} kbit"
    if b < 8 * 2 ** 20:
        return f"{b / 8192:.1f} KiB"
    if b < 8 * 2 ** 30:
        return f"{b / 8 / 2 ** 20:.1f} MiB"
    return f"{b / 8 / 2 ** 30:.2f} GiB"


def main():
    dirs = sys.argv[1:]
    res = [json.load(open(p)) for d in dirs for p in sorted(glob.glob(os.path.join(d, "results", "*.json")))]
    fresh = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "fresh", "*.json")):
            s = json.load(open(p))["summary"]
            fresh[s["name"]] = s
    best_lr = {}
    for r in res:
        if r["n"] == 2000 and r.get("seed", 0) == 0 and "_lr" in r["name"]:
            k = (r.get("model"), r["group"])
            cand = (r["trainq"]["bits"], r["held"]["bits"], r["lr"])
            best_lr[k] = max(best_lr.get(k, cand), cand)
    rows = {}
    for r in res:
        k = (r.get("model"), r["group"])
        if "_lr" in r["name"] and (k not in best_lr or r["lr"] != best_lr[k][2]):
            continue
        rows.setdefault((r.get("model"), r["n"], r["group"]), []).append(r)
    out = []
    print("| model | N | H | state | S | S/H | lr | seed | held-out acc [95%] | bits learned [95%] | recall of H [95%] "
          "| bits/S | CE-bound bits | train-template acc |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(rows, key=lambda k: (k[0] or "", k[1], ORDER.index(k[2]) if k[2] in ORDER else 9)):
        for r in sorted(rows[key], key=lambda r: r.get("seed", 0)):
            h = r["held"]
            line = {"model": r.get("model"), "n": r["n"], "H_bits": r["H_bits"], "group": r["group"],
                    "S_bits": r["S_bits"], "S_over_H": r["S_over_H"], "lr": r["lr"], "seed": r.get("seed", 0),
                    "epochs": r["epochs"], "acc": h["acc_mean"], "acc_ci": h["acc_ci"], "acc_by_attr": h["acc"],
                    "bits": h["bits"], "bits_ci": h["bits_ci"], "recall": r["recall_frac"],
                    "recall_ci": r["recall_frac_ci"], "ce_bits": h["ce_bits"], "trainq_acc": r["trainq"]["acc_mean"],
                    "name": r["name"], "fresh": fresh.get(r["name"])}
            out.append(line)
            print(f"| {(r.get('model') or '').split('/')[-1]} | {r['n']} | {kib(r['H_bits'])} | {r['group']} | "
                  f"{kib(r['S_bits'])} | {r['S_over_H']:.3g} | {r['lr']:g} | {line['seed']} | "
                  f"{h['acc_mean']:.3f} [{h['acc_ci'][0]:.3f}, {h['acc_ci'][1]:.3f}] | "
                  f"{h['bits']:.0f} [{h['bits_ci'][0]:.0f}, {h['bits_ci'][1]:.0f}] | "
                  f"{r['recall_frac']:.3f} [{r['recall_frac_ci'][0]:.3f}, {r['recall_frac_ci'][1]:.3f}] | "
                  f"{h['bits'] / r['S_bits']:.2g} | {h['ce_bits']:.0f} | {r['trainq']['acc_mean']:.3f} |")
    print()
    print("| run | N | state | S/H | learned bits on the analyzed questions | tokens | fresh bits/token (counted values, "
          "best estimator) FP8 / NVFP4 / bf16 | fresh bits per learned bit FP8 / NVFP4 / bf16 | same, every output, "
          "H_ctx only, FP8 | mean r99 of the gate_proj delta |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    g = lambda v: "inf" if v is None else f"{v:.3g}"
    for line in out:
        s = line["fresh"]
        if not s:
            continue
        pt, pl = s["fresh_per_token"], s["fresh_per_learned_bit"]
        r99 = s["by_map"].get("gate_proj", {}).get("r99")
        print(f"| {s['name']} | {s['n']} | {line['group']} | {line['S_over_H']:.3g} | "
              f"{s['bits_learned_on_questions']:.0f} ({s['questions']} q) | {s['tokens']} | "
              f"{pt['fp8t']['best_counted']:.0f} / {pt['nvfp4']['best_counted']:.0f} / {pt['bf16']['best_counted']:.0f} | "
              f"{g(pl['fp8t']['best_counted'])} / {g(pl['nvfp4']['best_counted'])} / {g(pl['bf16']['best_counted'])} | "
              f"{g(pl['fp8t']['ctx_all'])} | {r99} |")
    json.dump(out, open(os.path.join(dirs[0], "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
