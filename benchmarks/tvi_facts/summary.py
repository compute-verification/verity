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

ORDER = ["sub10kbit", "sub3kB", "tlora3kB", "sub10kB", "tlora10kB", "sub30kB", "sub100kB", "tlora100kB", "sub1MB",
         "tlora1MB", "lora16", "full"]
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


def compact(out):
    """One row per (model, N, state, epochs, lr): the mean over seeds, and a 95% interval that spans every seed's
    interval over questions."""
    groups = {}
    for line in out:
        groups.setdefault((line["model"], line["n"], line["group"], line["epochs"], line["lr"]), []).append(line)
    print("\nreadout: mean over seeds [95% interval spanning every seed's interval over questions]\n")
    print("| N | H (bits) | state | S (bits) | S/H | epochs | lr | seeds | recall of H, trained phrasing | "
          "bits learned, trained phrasing | recall of H4, held-out phrasing | bits learned, held-out |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (model, n, group, epochs, lr), ls in groups.items():
        mean = lambda k, i: sum(x[k][i] for x in ls) / len(ls)
        lo = lambda k: min(x[k][1] for x in ls)
        hi = lambda k: max(x[k][2] for x in ls)
        first = ls[0]
        print(f"| {n} | {first['H_bits']:.0f} | {group} | {first['S_bits']:.3g} | {first['S_over_H']:.3g} | {epochs} | "
              f"{lr:g} | {len(ls)} | {mean('trainq_recall', 0):.3f} [{lo('trainq_recall'):.3f}, {hi('trainq_recall'):.3f}] | "
              f"{mean('trainq_bits', 0):.0f} [{lo('trainq_bits'):.0f}, {hi('trainq_bits'):.0f}] | "
              f"{mean('held4_recall', 0):.3f} [{lo('held4_recall'):.3f}, {hi('held4_recall'):.3f}] | "
              f"{mean('held4_bits', 0):.0f} [{lo('held4_bits'):.0f}, {hi('held4_bits'):.0f}] |")
    print()


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
        if r["n"] == 2000 and r.get("seed", 0) == 0 and "_lr" in r["name"] and not r["name"].startswith("e_"):
            k = (r.get("model"), r["group"])
            cand = (r["trainq"]["bits"], r["held"]["bits"], r["lr"])
            best_lr[k] = max(best_lr.get(k, cand), cand)
    rows, long = {}, []
    for r in res:
        k = (r.get("model"), r["group"])
        if r["name"].startswith("e_"):
            long.append(r)
            continue
        if "_lr" in r["name"] and (k not in best_lr or r["lr"] != best_lr[k][2]):
            continue
        rows.setdefault((r.get("model"), r["n"], r["group"]), []).append(r)
    order = lambda r: (r.get("model") or "", r["n"], ORDER.index(r["group"]) if r["group"] in ORDER else 99,
                       r["epochs"], r["lr"], r.get("seed", 0))
    tables = [("the sweep's learning rate", sorted((r for v in rows.values() for r in v), key=order)),
              ("long training, every learning rate (plan e)", sorted(long, key=order))]
    out = []
    for title, table in tables:
        if not table:
            continue
        print(f"\n{title}\n")
        print("| model | N | H | state | S | S/H | lr | epochs | seed | held-out acc (4 attr) | held-out bits [95%] | "
              "recall of H4 [95%] | trained-phrasing acc (6 attr) | trained-phrasing bits [95%] | recall of H [95%] | "
              "bits per bit of S |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in table:
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
                  f"{r['S_bits'] / H:.3g} | {r['lr']:g} | {r['epochs']} | {line['seed']} | {hacc:.3f} | "
                  f"{hb:.0f} [{hlo:.0f}, {hhi:.0f}] | "
                  f"{hb / H4:.3f} [{hlo / H4:.3f}, {hhi / H4:.3f}] | {tacc:.3f} | {tb:.0f} [{tlo:.0f}, {thi:.0f}] | "
                  f"{tb / H:.3f} [{tlo / H:.3f}, {thi / H:.3f}] | {tb / r['S_bits']:.2g} |")
    compact(out)
    print("| run | N | state | S/H | analyzed questions / tokens | knowledge about those people (bits, trained phrasing) | "
          "fresh bits/token, counted values, best estimator: FP8 / NVFP4 / bf16 | fresh bits per learned bit: "
          "FP8 / NVFP4 / bf16 | FP8 fresh bits above the format-only control, per learned bit | "
          "FP8 fresh bits per bit of S | gate_proj output delta: mean relative norm, mean r99 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    g = lambda v: "inf" if not v else f"{v:.3g}"
    # sub10kbit learns the answer format and the value prior (loss falls) but recalls no fact: the control, per
    # token, at the same N or else the nearest (marked *). It moves every layer, so states confined to a few layers
    # (whose lower layers don't move at all) get no control.
    control = {(line["model"], line["n"]): line["fresh"]["fresh_per_token"]["fp8t"]["best_counted"] for line in out
               if line["group"] == "sub10kbit" and line["fresh"]}
    partial = {"tlora3kB", "tlora10kB", "tlora100kB"}
    for line in out:
        s = line["fresh"]
        if not s:
            continue
        people = s["questions"] // len(facts.ATTR_NAMES)
        learned = line["trainq_bits"][0] * people / line["n"]
        pt, tot = s["fresh_per_token"], s["totals"]
        per = {f: (tot[f]["best_counted"] / learned if learned > 0 else None) for f in tot}
        per_s = tot["fp8t"]["best_counted"] / line["S_bits"]
        near = sorted((abs(math.log(n / line["n"])), n) for (m, n) in control if m == line["model"])
        excess, mark = None, ""
        if near and learned >= 1 and line["group"] not in partial:
            n_c = near[0][1]
            mark = "" if n_c == line["n"] else "*"
            excess = (pt["fp8t"]["best_counted"] - control[(line["model"], n_c)]) * s["tokens"] / learned
        gate = s["by_map"].get("gate_proj", {})
        line["fresh_summary"] = {"people": people, "learned_bits": learned, "per_token": pt, "per_learned_bit": per,
                                 "fp8_per_state_bit": per_s, "fp8_excess_over_control_per_learned_bit": excess,
                                 "gate_rel_dnorm": gate.get("rel_dnorm"), "gate_r99": gate.get("r99")}
        print(f"| {s['name']} | {s['n']} | {line['group']} | {line['S_over_H']:.3g} | {s['questions']} / {s['tokens']} | "
              f"{learned:.0f} | {pt['fp8t']['best_counted']:.0f} / {pt['nvfp4']['best_counted']:.0f} / "
              f"{pt['bf16']['best_counted']:.0f} | {g(per['fp8t'])} / {g(per['nvfp4'])} / {g(per['bf16'])} | "
              f"{'n/a' if excess is None else f'{excess:.3g}{mark}'} | {per_s:.3g} | "
              f"{gate.get('rel_dnorm', 0):.2f}, {round(gate.get('r99') or 0)} |")
    json.dump(out, open(os.path.join(dirs[0], "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
