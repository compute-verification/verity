"""The readout table from run out dirs: one row per (model, N, state) at the learning rate the sweep chose, every
seed shown; then fresh bits per learned bit. `python summary.py OUT_DIR...` prints Markdown and writes summary.json
beside the first dir.

Two recall measures. Held-out: the four attributes whose held-out question is unambiguous (city, year, employer,
university; the held-out questions for birthday and major get answered with another attribute even by a model
that answers the training questions perfectly), against H4 = N * 30.25 bits. Trained phrasing: all six attributes
asked with a training question, against H. Bits learned are Fano's bound from accuracy, summed over attributes;
the 95% intervals resample each attribute's correct count (binomial over the N people).
"""
import glob
import json
import math
import os
import random
import re
import sys

import facts

ORDER = ["sub10kbit", "sub10kB", "tlora10kB", "sub100kB", "tlora100kB", "sub1MB", "tlora1MB", "lora16", "full"]
HELD4 = ["city", "year", "employer", "university"]


def kib(b):
    if b < 8 * 1024:
        return f"{b / 1000:.1f} kbit"
    if b < 8 * 2 ** 20:
        return f"{b / 8192:.1f} KiB"
    if b < 8 * 2 ** 30:
        return f"{b / 8 / 2 ** 20:.1f} MiB"
    return f"{b / 8 / 2 ** 30:.2f} GiB"


def bits(acc, attrs, n, boot=400, seed=0):
    """(bits, lo, hi, H) for accuracies acc over attrs, n people each"""
    point = n * sum(facts.mi_bits(acc[a], facts.CARD[a]) for a in attrs)
    rng = random.Random(seed)
    bs = []
    for _ in range(boot):
        tot = 0.0
        for a in attrs:
            k = sum(rng.random() < acc[a] for _ in range(n)) if n <= 2000 else \
                min(n, max(0, round(rng.gauss(n * acc[a], math.sqrt(max(n * acc[a] * (1 - acc[a]), 1e-12))))))
            tot += facts.mi_bits(k / n, facts.CARD[a])
        bs.append(n * tot)
    bs.sort()
    return point, bs[int(0.025 * boot)], bs[int(0.975 * boot) - 1], n * sum(math.log2(facts.CARD[a]) for a in attrs)


def main():
    dirs = sys.argv[1:]
    res = [json.load(open(p)) for d in dirs for p in sorted(glob.glob(os.path.join(d, "results", "*.json")))]
    for r in res:
        r["group"] = re.search(r"n\d+_([A-Za-z0-9]+)_", r["name"]).group(1)
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
    print("| model | N | H | state | S | S/H | lr | seed | held-out acc (4 attr) | held-out bits [95%] | "
          "recall of H4 [95%] | trained-phrasing acc (6 attr) | trained-phrasing bits [95%] | recall of H [95%] | "
          "bits per bit of S |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(rows, key=lambda k: (k[0] or "", k[1], ORDER.index(k[2]) if k[2] in ORDER else 9)):
        for r in sorted(rows[key], key=lambda r: r.get("seed", 0)):
            n = r["n"]
            hb, hlo, hhi, H4 = bits(r["held"]["acc"], HELD4, n)
            tb, tlo, thi, H = bits(r["trainq"]["acc"], facts.ATTR_NAMES, n)
            hacc = sum(r["held"]["acc"][a] for a in HELD4) / 4
            tacc = r["trainq"]["acc_mean"]
            line = {"model": r.get("model"), "n": n, "H_bits": H, "H4_bits": H4, "group": r["group"],
                    "S_bits": r["S_bits"], "S_over_H": r["S_bits"] / H, "lr": r["lr"], "seed": r.get("seed", 0),
                    "epochs": r["epochs"], "held4_acc": hacc, "held4_bits": [hb, hlo, hhi],
                    "held4_recall": [hb / H4, hlo / H4, hhi / H4], "trainq_acc": tacc, "trainq_bits": [tb, tlo, thi],
                    "trainq_recall": [tb / H, tlo / H, thi / H], "held_acc_by_attr": r["held"]["acc"],
                    "trainq_acc_by_attr": r["trainq"]["acc"], "name": r["name"], "steps": r["steps"],
                    "tokens": r["tokens"], "fresh": fresh.get(r["name"])}
            out.append(line)
            print(f"| {(r.get('model') or '').split('/')[-1]} | {n} | {kib(H)} | {r['group']} | {kib(r['S_bits'])} | "
                  f"{r['S_bits'] / H:.3g} | {r['lr']:g} | {line['seed']} | {hacc:.3f} | {hb:.0f} [{hlo:.0f}, {hhi:.0f}] | "
                  f"{hb / H4:.3f} [{hlo / H4:.3f}, {hhi / H4:.3f}] | {tacc:.3f} | {tb:.0f} [{tlo:.0f}, {thi:.0f}] | "
                  f"{tb / H:.3f} [{tlo / H:.3f}, {thi / H:.3f}] | {tb / r['S_bits']:.2g} |")
    print()
    print("| run | N | state | S/H | analyzed questions / tokens | knowledge about those people (bits, trained phrasing) | "
          "fresh bits/token, counted values, best estimator: FP8 / NVFP4 / bf16 | fresh bits per learned bit: "
          "FP8 / NVFP4 / bf16 | mean r99 of the gate_proj output delta |")
    print("|---|---|---|---|---|---|---|---|---|")
    g = lambda v: "inf" if not v else f"{v:.3g}"
    for line in out:
        s = line["fresh"]
        if not s:
            continue
        people = s["questions"] // len(facts.ATTR_NAMES)
        learned = line["trainq_bits"][0] * people / line["n"]
        pt, tot = s["fresh_per_token"], s["totals"]
        per = {f: (tot[f]["best_counted"] / learned if learned > 0 else None) for f in tot}
        line["fresh_summary"] = {"people": people, "learned_bits": learned, "per_token": pt, "per_learned_bit": per}
        r99 = s["by_map"].get("gate_proj", {}).get("r99")
        print(f"| {s['name']} | {s['n']} | {line['group']} | {line['S_over_H']:.3g} | {s['questions']} / {s['tokens']} | "
              f"{learned:.0f} | {pt['fp8t']['best_counted']:.0f} / {pt['nvfp4']['best_counted']:.0f} / "
              f"{pt['bf16']['best_counted']:.0f} | {g(per['fp8t'])} / {g(per['nvfp4'])} / {g(per['bf16'])} | "
              f"{r99 if r99 is None else round(r99)} |")
    json.dump(out, open(os.path.join(dirs[0], "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
