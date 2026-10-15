#!/usr/bin/env python3
"""epsilon from pouw-core's covert_bound_priced at the one-lane-op price (gpu-model/pouw-core.lean, §7.2-7.3: pS32 on the
measured line wP1mS), at the measured eta of bench.jsonl's arms.

    epsNum = ks cap + ku issue + km l2 - (Wb Nmin + ku Rmin + ku chi cap),   eps = epsNum / (ks cap)   (Lean's Nat: >= 0)

First it reproduces Lean's `epsS32_values` and `epsP1mu_values` exactly (exit 1 if not). Then, at each arm's eta:

- Lean's accounting (§7.3): the honest point is the filler's measured point (s_h steps), the useful call taking
  su = eta_cap cap of its steps, and Rmin as Lean's (the window's CUDA work stays forced). Unreceipted, the useful steps
  leave the floor (`epsNum_drop`); receipted at L = 32 they stay in it, so eps is Lean's receipted row at any eta.
- Forced work = receipts only: the same window, but Rmin is what the honest point's receipts force (`sat`): the filler's
  alu phi 24 q a step, duo 12 (3 ops a word at L = 32), duo2 24 (6 ops), an unreceipted call 0.
- The window alone: the receipted call fills the window by itself at its measured eta (no filler beside it), its receipts
  its only forced work: s = eta_cap cap steps, N = s / 256 blocks, Rmin = r_u s Nmin / N.

Each shape's arms at their faster tile config (table.best). Usage: eps.py BENCH.jsonl [nu ...]   (nu as fractions of all issue; default 0 0.005 0.04 and W11's upper ends,
0.00065 at delta = 0.1%, 0.00268 at 0.5% and 0.00534 at 1%: note:20261015T0713Z-draft-filler-issue-slack)
"""

import json
import math
import sys

from table import best

CAP = 14618880                 # tensor steps (32 s8 MACs) a us: 32 a SM-clock x 188 SMs x 2,430 MHz
ALL_ISSUE = 233902080          # q (quarter lane-ops) a us: 4 x 128 x 188 x 2,430
ISSUE0, CHI = 245597184, 4     # wP1mS: the measured line's intercept (q) and slope (q a step)
KS, KU, KM, WB = 32, 8, 32, 8192
L2_S32 = 1477593               # rate-capped words: S32 bulk plus L2 red
S_H, U_H = 8771328, 210511872  # the measured filler's point: 0.6 cap steps, 24 q a step
RMIN_H = 210278400
F_REF, F_MAX = 2092.0, 2430.0  # eta_cap = eta x 2,092 / 2,430 (the eta study's)
MU = 0.0011                    # detection slack: Nmin = floor(N (1 - mu)); Lean's rows differ by <= 2 blocks (< 4e-5 in eps)


def epsnum(issue, l2, nmin, rmin):
    return max(0, KS * CAP + KU * issue + KM * l2 - (WB * nmin + KU * rmin + KU * CHI * CAP))


def h_of(nu):
    return math.ceil(nu * ALL_ISSUE - 1e-9)


def eps(nu, nmin, rmin, l2=L2_S32):
    return epsnum(ISSUE0 + h_of(nu), l2, nmin, rmin) / (KS * CAP)


def blocks(steps):
    n = max(0, steps) // 256
    return n, math.floor(n * (1 - MU))


def reproduce():
    want = {(0, 1477593): 49462048, (0, 1784590): 59285952, (1169511, 1477593): 58818136, (1169511, 1784590): 68642040,
            (9356084, 1477593): 124310720, (9356084, 1784590): 134134624}
    bad = [(k, v, epsnum(ISSUE0 + k[0], k[1], 34225, RMIN_H)) for k, v in want.items()
           if epsnum(ISSUE0 + k[0], k[1], 34225, RMIN_H) != v]
    rows = [(33782, [53091104, 62447192, 127939776]), (32652, [62348064, 71704152, 137196736]),
            (25384, [121887520, 131243608, 196736192]), (23372, [138369824, 147725912, 213218496]),
            (17527, [186252064, 195608152, 261100736]), (3286, [302914336, 312270424, 377763008])]
    for nmin, vals in rows:
        for h, v in zip([0, 1169511, 9356084], vals):
            if epsnum(ISSUE0 + h, L2_S32, nmin, RMIN_H) != v:
                bad.append(((h, nmin), v, epsnum(ISSUE0 + h, L2_S32, nmin, RMIN_H)))
    if [h_of(x) for x in (0, 0.005, 0.04)] != [0, 1169511, 9356084]:
        bad.append(("h", [0, 1169511, 9356084], [h_of(x) for x in (0, 0.005, 0.04)]))
    return bad


def window(eta, ru, keep_rmin):
    """The honest point at the filler's line point (s_h steps), su = eta_cap cap of them the useful call's. ru: the useful
    call's receipt work in q a step (None: unreceipted, its steps outside the floor). keep_rmin: Rmin as Lean's (the
    window's CUDA work stays forced, §7.3); else Rmin is the honest point's receipts, u = 24 (s_h - su) + ru su."""
    su = min(S_H, math.ceil(eta * F_REF / F_MAX * CAP))
    n, nmin = blocks(S_H if ru is not None else S_H - su)
    if keep_rmin:
        return lambda nu: eps(nu, nmin, RMIN_H)
    u = 24 * (S_H - su) + (ru or 0) * su
    return lambda nu: eps(nu, nmin, u * nmin // max(n, 1))


def alone(eta, ru):
    s = math.ceil(eta * F_REF / F_MAX * CAP)
    n, nmin = blocks(s)
    return lambda nu: eps(nu, nmin, ru * s * nmin // max(n, 1))


def served_eta(rows, arm, m):
    """MAC-weighted eta over the four linears of one m (one layer's q/k/v, o, gate/up, down)."""
    sel = [r for r in rows if r["m"] == m and not r["m"] == r["n"] == r["k"]]
    macs = sum(r["m"] * r["n"] * r["k"] for r in sel)
    return macs / sum(r["m"] * r["n"] * r["k"] / r["eta"][arm] for r in sel)


def main():
    bad = reproduce()
    print(f"Lean reproduction: {'exact (epsS32_values, epsP1mu_values, h)' if not bad else bad}")
    if bad:
        return 1
    rows = [{"name": r["label"], "m": r["m"], "n": r["n"], "k": r["k"],
             "eta": {a: v["eta"] for a, v in r["arms"].items() if v["eta"] > 0}}
            for r in best([json.loads(x) for x in open(sys.argv[1]) if x.strip()])]
    nus = [float(x) for x in sys.argv[2:]] or [0, 0.005, 0.04, 0.00065, 0.00268, 0.00534]
    pts = [(r["name"], r["eta"]) for r in rows]
    for m in sorted({r["m"] for r in rows if r["m"] != r["n"] or r["n"] != r["k"]}, reverse=True):
        pts.append((f"served m={m} (MAC-weighted)", {a: served_eta(rows, a, m) for a in rows[0]["eta"]}))
    hdr = " | ".join(f"ν={x:g}" for x in nus)
    arms = [("b", None, "(b) unreceipted"), ("c", 12, "(c) duo /32"), ("c2", 24, "duo2 /32")]
    for keep, title in [(True, "Lean's accounting (§7.3): the useful steps inside the filler's honest point, Rmin as Lean's"),
                        (False, "Forced work = receipts only: the same window, Rmin the honest point's receipts "
                                "(filler 24 q a step, duo 12, duo2 24, unreceipted 0)")]:
        print(f"\n{title}. ε by ν:\n")
        print("| point | " + " | ".join(f"η {a}" for a, _, _ in arms) + " | " +
              " | ".join(f"{t}: {hdr}" for _, _, t in arms) + " |")
        print("|" + "---|" * (1 + len(arms) + len(arms) * len(nus)))
        for name, e in pts:
            cells = [f"{e[a]:.3f}" for a, _, _ in arms]
            for a, ru, _ in arms:
                f = window(e[a], ru, keep)
                cells += [f"{f(x):.4f}" for x in nus]
            print(f"| {name} | " + " | ".join(cells) + " |")
    print("\nThe receipted call alone in the window (no filler; its receipts its only forced work). ε by ν:\n")
    print(f"| point | (c) duo /32: {hdr} | duo2 /32: {hdr} |")
    print("|" + "---|" * (1 + 2 * len(nus)))
    for name, e in pts:
        print(f"| {name} | " + " | ".join(f"{alone(e['c'], 12)(x):.3f}" for x in nus) + " | " +
              " | ".join(f"{alone(e['c2'], 24)(x):.3f}" for x in nus) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
