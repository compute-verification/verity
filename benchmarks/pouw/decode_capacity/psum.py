"""Summarize psched.sh runs (dcap pfill rows): per (B, L) and arm, decode slowdown, filler share and coverage.

  python3 psum.py RUN_OUT_DIR [RUN_OUT_DIR ...] [--limit 1.05]

Coverage at a decode point is read as the filler scorer reads a decode entry (benchmarks/pouw/filler/scoring.py,
filler-scorer/v1): the step plus the non-GEMM kernels, max(1.09 ms, 4.684 µs × B), those stretched by the step's
slowdown; (0.225 × serving MACs + 0.99 × filler MACs) ÷ (the int8 peak at the serving-alone clock × that time), the filler
running through the whole stretched step at its measured rate. "base" is FILL v1's partitioned filler in the same
invocation; "sched" adds the phase filler. A point's best is the arm with the largest coverage among those whose slowdown
is ≤ --limit.
"""

import argparse
import json
import pathlib

CERT_SERVING, CERT_FILLER = 0.225, 0.99
SMS, PER_SM = 188, 1024
H, LAYERS, NQ, NKV, HD, INTER, VOCAB = 4096, 32, 32, 8, 128, 14336, 128256
LIN_TOK = LAYERS * (H * (NQ + 2 * NKV) * HD + H * H + H * 2 * INTER + INTER * H)
LM_TOK = H * VOCAB
ATT_TOK_L = LAYERS * NQ * HD * 2


def other_ms(b):
    return max(1.09, 38.37 / 8192 * b)


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
    """(coverage, slowdown, filler share of the GPU's peak) of an arm: none | base | sched."""
    a = r["arms"]
    t0 = (a["full"]["ms"] + a["full2"]["ms"]) / 2
    mhz = (a["full"]["mhz"] + a["full2"]["mhz"]) / 2
    if arm == "none":
        slow, share = 1.0, 0.0
    elif arm == "base":
        if "base" not in a:
            return None
        slow, share = r["base"]["slowdown"], r["base"]["fill_share"]
    else:
        slow, share = r["sched"]["slowdown"], r["sched"]["fill_share"]
    b, l = r["B"], r["L"]
    t1 = t0 * slow + other_ms(b) * slow
    cap = PER_SM * SMS * mhz * 1e6 * t1 * 1e-3
    n = b * (LIN_TOK + LM_TOK + ATT_TOK_L * l)
    f = share * PER_SM * SMS * mhz * 1e6 * t1 * 1e-3
    return (CERT_SERVING * n + CERT_FILLER * f) / cap, slow, share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--limit", type=float, default=1.05)
    args = ap.parse_args()
    rs = rows(args.dirs)
    print("| run | B | L | shape | split only | base: slow, share, cov | sched: slow, split share, phase share, cov | "
          "phase tiles a step (att ms a layer, tile ms) | receipts (bad) |")
    print("|---|---|---|---|---|---|---|---|---|")
    best = {}
    for r in sorted(rs, key=lambda r: (r["B"], r["L"], r["fill_sms"], r["shape"])):
        b, s, n = cov(r, "base"), cov(r, "sched"), cov(r, "none")
        ph = r["phase"]
        bs = f"{b[1]:.3f}, {b[2]:.3f}, {b[0]:.3f}" if b else "–"
        print(f"| {r['_run'][-4:]} | {r['B']} | {r['L']} | {r['shape']} | {r['split_only']:.3f} | {bs} | "
              f"{s[1]:.3f}, {r['sched']['split_fill_share']:.3f}, {r['sched']['phase_share']:.3f}, {s[0]:.3f} | "
              f"{ph['tiles_a_step']:.0f} ({ph['att_layer_ms']:.4f}, {ph['tile_ms']:.4f}) | "
              f"{ph['rec_checked'] + r['rec_checked']} ({ph['rec_bad'] + r['rec_bad']}) |")
        k = (r["B"], r["L"])
        best.setdefault(k, {}).setdefault("none", (n[0], 1.0, "–"))
        for arm, c in (("base", b), ("sched", s)):
            if c and c[1] <= args.limit:
                name = f"{r['fill_sms']} SMs" if arm == "base" else r["shape"]
                if c[0] > best[k].get(arm, (0,))[0]:
                    best[k][arm] = (c[0], c[1], name)
    print(f"\nbest arm ≤ {args.limit:.2f} per point (scorer's coverage): FILL v1's split vs scheduled")
    print("| B | L | no filler | FILL v1 best: cov, slow (split) | scheduled best: cov, slow (arm) | gain |")
    print("|---|---|---|---|---|---|")
    for (b, l), v in sorted(best.items()):
        bb, ss = v.get("base"), v.get("sched")
        g = f"{ss[0] - bb[0]:+.3f}" if bb and ss else "–"
        print(f"| {b} | {l} | {v['none'][0]:.3f} | {f'{bb[0]:.3f}, {bb[1]:.3f} ({bb[2]})' if bb else '–'} | "
              f"{f'{ss[0]:.3f}, {ss[1]:.3f} ({ss[2]})' if ss else '–'} | {g} |")


if __name__ == "__main__":
    main()
