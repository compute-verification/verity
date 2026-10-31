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


def planner(doc):
    print(f"### Planner, d = {doc['d']}, M = {doc['m']} own beacons, {doc['iters']} iterations ({doc['pre']})\n")
    print("| target | setting | L | unplanned h200/sm120 | planned h200/sm120 | known-coin h200/sm120 | planned exact | "
          "planned miss (LSB) | plan shift (LSB) | val served |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(doc["runs"], key=lambda r: (r["family"], r["tag"], r["L"])):
        def m(k, f):
            return float(np.mean([e[f] for e in r[k]]))
        print(f"| {r['family']} | {r['tag']} | {r['L']} | {m('unplanned', 'h200'):.3f}/{m('unplanned', 'sm120'):.3f} | "
              f"{m('planned', 'h200'):.3f}/{m('planned', 'sm120'):.3f} | {m('known', 'h200'):.3f}/{m('known', 'sm120'):.3f} | "
              f"{m('planned', 'exact'):.3f} | {m('planned', 'miss_rms'):.2f} | {r['w0_shift_rms']:.2f} | "
              f"{m('planned', 'val_served'):.3f} |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--erased", type=float, default=0.99)
    ap.add_argument("--split", action="store_true")
    args = ap.parse_args()
    for f in args.files:
        doc = json.loads(open(f).read())
        if "spread" in doc:
            erasure(doc, args.erased, args.split)
        else:
            planner(doc)


if __name__ == "__main__":
    main()
