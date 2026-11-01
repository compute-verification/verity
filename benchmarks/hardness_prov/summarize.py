"""Markdown tables from erasure.py's and planner.py's JSON: `python summarize.py erasure.json [planner.json ...]`.

Erasure: per start and pass setting, the planted matrices' cheapest route on each chip at each τ (mean over beacons),
the erasure horizon (the first τ whose route costs at least `--erased` on both chips), the codes still at their start
value, the coin spread in LSBs and the served validation loss. With `--split`, fc and out separately (the scale game
rescales fc only).

Planner: per target family, setting and L, the cheapest route after L steps from the unplanned start, the plan through
the mean, and the known-coin control, with the fraction of codes landing exactly on the target.
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def erasure(doc, erased, split, chips=("h200", "sm120")):
    groups = defaultdict(list)
    for r in doc["runs"]:
        groups[(r["start"], r["tag"])].append(r)
    taus = doc["taus"]
    print(f"### Erasure, d = {doc['d']}, {doc['layers']} layers ({doc['pre']})\n")
    head = " | ".join(f"τ {t}" for t in taus)
    print(f"| start | setting | {head} | horizon | s0 at τ max | spread at τ max (LSB) | val served τ 0 → max |")
    print("|---|---|" + "---:|" * len(taus) + "---:|---:|---:|---|")
    for (start, tag), rs in sorted(groups.items()):
        parts = [("all", None)] if not split else [("fc", 0), ("out", 1)]
        for pname, off in parts:
            cells, horizon = [], None
            for i, t in enumerate(taus):
                cs = {}
                for chip in chips:
                    vals = []
                    for r in rs:
                        c = r["records"][i]["cost"][chip]
                        vals.append(c["best"] if off is None else np.mean(c["per_matrix"][off::2]))
                    cs[chip] = float(np.mean(vals))
                cells.append("/".join(f"{cs[c]:.3f}" for c in chips))
                if horizon is None and all(cs[c] >= erased for c in chips):
                    horizon = t
            s0 = np.mean([r["records"][-1]["s0"] for r in rs])
            sp = doc.get("spread", {}).get(f"{start}/{tag}", {})
            spt = sp.get(str(taus[-1]))
            v0 = np.mean([r["records"][0]["val_served"] for r in rs])
            v1 = np.mean([r["records"][-1]["val_served"] for r in rs])
            name = start if off is None else f"{start} ({pname})"
            print(f"| {name} | {tag} | " + " | ".join(cells) + f" | {horizon if horizon is not None else '>' + str(taus[-1])}"
                  f" | {s0:.3f} | {spt if spt is None else f'{spt:.2f}'} | {v0:.3f} → {v1:.3f} |")
    print()


GATE = ("sigma_q", "entropy_bits", "frac_abs_le15", "frac_abs_gt31", "frac_abs_gt63", "tail31", "frac_zero", "p_odd")


def gate(doc):
    """The public code statistics a gate could check, at τ 0 (one row per start and quantizer) and at τ max (per
    setting), with the top-r share of the codes' spectral energy, mean over planted matrices and beacons."""
    taus = doc["taus"]
    rs0 = doc["runs"][0]["records"][0]
    ranks = sorted(rs0.get("energy_top", {}), key=int)
    print(f"### Code statistics, d = {doc['d']} ({doc['pre']})\n")
    print("| start | setting | τ | " + " | ".join(GATE) + " | " + " | ".join(f"energy top {r}" for r in ranks)
          + " | best route sm120 | val served |")
    print("|---|---|---:|" + "---:|" * (len(GATE) + len(ranks) + 2))
    groups = defaultdict(list)
    for r in doc["runs"]:
        groups[(r["start"], r["tag"])].append(r)
    seen = set()
    for (start, tag), rs in sorted(groups.items()):
        quant = rs[0]["cfg"]["quant"]
        for i in ([0] if (start, quant) not in seen else []) + [len(taus) - 1]:
            recs = [r["records"][i] for r in rs]
            st = [float(np.mean([x.get("stats", {}).get(k, np.nan) for x in recs])) for k in GATE]
            en = [float(np.mean([x["energy_top"][k] for x in recs])) for k in ranks]
            cost = float(np.mean([x["cost"]["sm120"]["best"] for x in recs]))
            val = float(np.mean([x["val_served"] for x in recs]))
            setting = quant if i == 0 else tag
            print(f"| {start} | {setting} | {taus[i]} | " + " | ".join(f"{v:.3f}" for v in st) + " | "
                  + " | ".join(f"{v:.3f}" for v in en) + f" | {cost:.3f} | {val:.3f} |")
        seen.add((start, quant))
    print()


def constrained(doc):
    if not doc.get("constrained"):
        return
    print(f"### Constrained starts, d = {doc['d']}\n")
    print("| start | quant | val | val served |")
    print("|---|---|---:|---:|")
    for r in doc["constrained"]:
        print(f"| {r['family']}@{r['steps']} | {r['quant']} | {r['val']:.4f} | {r['val_served']:.4f} |")
    print()


def planner(doc):
    print(f"### Planner, d = {doc['d']}, M = {doc['m']} own beacons, {doc['iters']} iterations ({doc['pre']})\n")
    print("| target | setting | L | unplanned h200/sm120 | planned h200/sm120 | known-coin h200/sm120 | "
          "adaptive h200/sm120 | exact: planned, adaptive | miss (LSB): planned, adaptive | plan shift (LSB) | "
          "val served: planned, adaptive |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(doc["runs"], key=lambda r: (r["family"], r["tag"], r["L"])):
        def m(k, f):
            return float(np.mean([e[f] for e in r[k]])) if k in r else float("nan")
        print(f"| {r['family']} | {r['tag']} | {r['L']} | {m('unplanned', 'h200'):.3f}/{m('unplanned', 'sm120'):.3f} | "
              f"{m('planned', 'h200'):.3f}/{m('planned', 'sm120'):.3f} | {m('known', 'h200'):.3f}/{m('known', 'sm120'):.3f} | "
              f"{m('steer', 'h200'):.3f}/{m('steer', 'sm120'):.3f} | {m('planned', 'exact'):.3f}, {m('steer', 'exact'):.3f} | "
              f"{m('planned', 'miss_rms'):.2f}, {m('steer', 'miss_rms'):.2f} | {r['w0_shift_rms']:.2f} | "
              f"{m('planned', 'val_served'):.3f}, {m('steer', 'val_served'):.3f} |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--erased", type=float, default=0.99)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--gate", action="store_true", help="also the code statistics (records with `stats`)")
    args = ap.parse_args()
    for f in args.files:
        doc = json.loads(open(f).read())
        if "spread" in doc:
            constrained(doc)
            erasure(doc, args.erased, args.split)
            if args.gate:
                gate(doc)
        else:
            planner(doc)


if __name__ == "__main__":
    main()
