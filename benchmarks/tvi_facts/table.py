"""The results table (N, H, S, recall, bits learned, S/H) and the fresh-bits table, from one or more $OUT dirs."""
import glob
import json
import os
import sys


def load(dirs, sub):
    out = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, sub, "*.json"))):
            out.append(json.load(open(p)))
    return out


def fmt_bits(b):
    return f"{b / 8192:.2f} KiB" if b < 8 * 2 ** 20 else f"{b / 8 / 2 ** 20:.1f} MiB"


def main():
    dirs = sys.argv[1:]
    res = load(dirs, "results")
    res.sort(key=lambda r: (r.get("model", ""), r["n"], r["S_bits"], r["lr"], r.get("seed", 0)))
    print("| model | N | H | method | S | S/H | lr | seed | epochs | held acc [95%] | bits learned [95%] | "
          "recall of H [95%] | CE-bound bits | train-template acc |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res:
        h, t = r["held"], r["trainq"]
        print(f"| {r.get('model', '').split('/')[-1]} | {r['n']} | {fmt_bits(r['H_bits'])} | {r['method']} | "
              f"{fmt_bits(r['S_bits'])} | {r['S_over_H']:.3g} | {r['lr']:g} | {r.get('seed', 0)} | {r['epochs']} | "
              f"{h['acc_mean']:.3f} [{h['acc_ci'][0]:.3f}, {h['acc_ci'][1]:.3f}] | "
              f"{h['bits']:.0f} [{h['bits_ci'][0]:.0f}, {h['bits_ci'][1]:.0f}] | "
              f"{r['recall_frac']:.3f} [{r['recall_frac_ci'][0]:.3f}, {r['recall_frac_ci'][1]:.3f}] | "
              f"{h['ce_bits']:.0f} | {t['acc_mean']:.3f} |")
    fr = [f["summary"] for f in load(dirs, "fresh")]
    if fr:
        print()
        print("| run | N | S | questions | tokens | learned bits | fresh/token fp8 (best, counted) | "
              "fresh/token nvfp4 | fresh per learned bit fp8 | nvfp4 | bf16 | fp8 H_ctx all outputs |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in fr:
            pl, pt = s["fresh_per_learned_bit"], s["fresh_per_token"]
            g = lambda v: "inf" if v is None else f"{v:.3g}"
            print(f"| {s['name']} | {s['n']} | {fmt_bits(s['S_bits'])} | {s['questions']} | {s['tokens']} | "
                  f"{s['bits_learned_on_questions']:.0f} | {pt['fp8t']['best_counted']:.0f} | "
                  f"{pt['nvfp4']['best_counted']:.0f} | {g(pl['fp8t']['best_counted'])} | "
                  f"{g(pl['nvfp4']['best_counted'])} | {g(pl['bf16']['best_counted'])} | {g(pl['fp8t']['ctx_all'])} |")


if __name__ == "__main__":
    main()
