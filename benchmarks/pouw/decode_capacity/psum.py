"""Summarize psched.sh runs (dcap pfill rows): per (B, L) and arm, decode slowdown, filler share, and certified ÷ capacity.

  python3 psum.py RUN_OUT_DIR [RUN_OUT_DIR ...] [--limit 1.05]

certified ÷ capacity at a point = 0.99 × (the step's MACs ÷ (capacity × the step's time with the filler) + the filler's
share of the GPU's peak), FILL's convention (note:20261016T0318Z-draft-dram-free-filler). "base" is FILL v1's partitioned
filler in the same invocation; "sched" adds the phase filler. A row's best is the arm with the largest certified ÷
capacity among those whose slowdown is ≤ --limit.
"""

import argparse
import json
import pathlib

CERT = 0.99


def rows(dirs):
    out = []
    for d in dirs:
        p = pathlib.Path(d) / "pfill.jsonl"
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_run"] = pathlib.Path(d).parent.name
                out.append(r)
    return out


def cov(r, arm):
    """certified ÷ capacity, slowdown and filler share of an arm (base | sched | none)."""
    a = r["arms"]
    ref = (a["full"]["ms"] + a["full2"]["ms"]) / 2
    if arm == "none":
        return CERT * r["step_share_full"], 1.0, 0.0
    if arm == "base":
        if "base" not in a:
            return None
        slow, share = r["base"]["slowdown"], r["base"]["fill_share"]
    else:
        slow, share = r["sched"]["slowdown"], r["sched"]["fill_share"]
    step = r["step_share_full"] / slow * (a["full"]["mhz"] / (a["sched"]["mhz"] or a["full"]["mhz"]))
    return CERT * (step + share), slow, share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--limit", type=float, default=1.05)
    args = ap.parse_args()
    rs = rows(args.dirs)
    print("| run | B | L | shape | split only | base: slow, share, cert/C | sched: slow, split share, phase share, "
          "cert/C | phase tiles a step (att ms a layer, tile ms) | receipts (bad) |")
    print("|---|---|---|---|---|---|---|---|---|")
    best = {}
    for r in sorted(rs, key=lambda r: (r["B"], r["L"], r["fill_sms"], r["shape"])):
        b = cov(r, "base")
        s = cov(r, "sched")
        ph = r["phase"]
        bs = f"{b[1]:.3f}, {b[2]:.3f}, {b[0]:.3f}" if b else "–"
        print(f"| {r['_run'][-4:]} | {r['B']} | {r['L']} | {r['shape']} | {r['split_only']:.3f} | {bs} | "
              f"{s[1]:.3f}, {r['sched']['split_fill_share']:.3f}, {r['sched']['phase_share']:.3f}, {s[0]:.3f} | "
              f"{ph['tiles_a_step']:.0f} ({ph['att_layer_ms']:.4f}, {ph['tile_ms']:.4f}) | "
              f"{ph['rec_checked'] + r['rec_checked']} ({ph['rec_bad'] + r['rec_bad']}) |")
        for arm, c in (("base", b), ("sched", s)):
            if c and c[1] <= args.limit:
                k = (r["B"], r["L"])
                name = f"{r['fill_sms']} SMs" if arm == "base" else r["shape"]
                if c[0] > best.get(k, {}).get(arm, (0,))[0]:
                    best.setdefault(k, {})[arm] = (c[0], c[1], name)
    print(f"\nbest arm ≤ {args.limit:.2f} per point: FILL v1 split vs scheduled")
    print("| B | L | FILL v1 best: cert/C, slow (split) | scheduled best: cert/C, slow (arm) | gain |")
    print("|---|---|---|---|---|")
    for (b, l), v in sorted(best.items()):
        bb, ss = v.get("base"), v.get("sched")
        g = f"{ss[0] - bb[0]:+.3f}" if bb and ss else "–"
        print(f"| {b} | {l} | {f'{bb[0]:.3f}, {bb[1]:.3f} ({bb[2]})' if bb else '–'} | "
              f"{f'{ss[0]:.3f}, {ss[1]:.3f} ({ss[2]})' if ss else '–'} | {g} |")


if __name__ == "__main__":
    main()
