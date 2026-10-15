"""The receipt φ cost tables, rendered as Markdown from one `phi.sh` run's output directory.

    python3 tables.py OUT_DIR [--merkle 0.4] [--n 16384,65536] > tables.md

Units. An SM issues 128 lane-operations a clock (4 SMSPs of 32 lanes), so a word that takes h lane-slots costs h/128
SM-clocks; IMAD and LOP3 run at 64 lanes a clock, 2 slots each. Pure: h = 128 / (words per SM-clock), and µs per 10^9
words is the whole GPU at the run's clock. Fused: a stride of L = 32 C k-steps is 16 C SM-clocks of one warp's FP8 or
INT8 MMAs (1024 MACs per SM-clock) per 512 accumulator words, so a loop that runs at rel of the bare loop's MMA rate
pays h_eff = (1/rel − 1)·L/8 slots of tensor time per word: the CUDA cost read through the tensor pipe.

Strides. With independent pipes (the Notion page's model, "Three lemmas toward ε < 1% PoUW", Receipt lemma),
saturation is h·N = W_G·t_h with t_h = (1 + 7L/n)·n³/(b³·W_T), that is L(1 + 7L/n) = (b³W_T/W_G)·h: 128 h on B200 and
8 h on sm_120 (W_T/W_G = 1024 MACs over 128 slots, b = 16), with decode overhead 7L/n. Measured, the CUDA cost is
tensor time at every stride, so the total is T(L) = 8 h/L + 7L/n, least at L* = √(8 h n/7) with T* = 2√(56 h/n).
h adds the page's Merkle term (0.4 slots a word, B200's, not measured here) to φ's.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

MODES = ("fp8", "fp8pp", "s8", "s8pp")
CS = (1, 2, 4, 8, 16, 32)
STEADY = (4, 8, 16, 32)
LOOP = 3  # UIADD3, UISETP, BRA
NOT_PHI = {"QMMA", "IMMA", "HMMA", "NOP", "UIADD3", "UISETP", "BRA"}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rate(r: dict) -> float:
    return next(v for k, v in r.items() if k.endswith("_per_sm_clk"))


def ncu(path: Path) -> dict[str, float]:
    out = {}
    if not path.exists():
        return out
    for row in csv.reader(path.read_text().splitlines()):
        if len(row) == 15 and row[12] != "Metric Name":
            try:
                out[row[12]] = float(row[14].replace(",", ""))
            except ValueError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--merkle", type=float, default=0.4)
    ap.add_argument("--n", default="16384,65536")
    args = ap.parse_args()
    o, ns = args.out, [int(x) for x in args.n.split(",")]
    pure, probe, fused = jsonl(o / "pure.jsonl"), jsonl(o / "probe.jsonl"), jsonl(o / "fused.jsonl")
    mix = json.loads((o / "sass-mix.json").read_text())
    kinds = [r["kind"] for r in pure]
    controls = list(dict.fromkeys(r["kind"] for r in fused if r["kind"] not in kinds and r["kind"] != "none"))
    sms, mhz = pure[0]["sms"], statistics.median(r["mhz"] for r in pure)

    body = {(m["mode"], m["index"], m.get("C"), m.get("acc")): m["loop_body"] for m in mix}
    print(f"Run `{o}`: commit {(o / 'commit.txt').read_text().strip()[:9]}, {sms} SMs at {mhz:.0f} MHz (median).\n")

    print("| kind | SASS/word (pure) | SASS/word (fused) | words/SM/clk | slots/word | µs per 10^9 words |")
    print("|---|---|---|---|---|---|")
    hp, ops = {}, {}
    for i, r in enumerate(pure):
        b, f = body[("pure", i, None, None)], body.get(("fused", i, 1, 2))
        per_fused = sum(v for op, v in f["base"].items() if op not in NOT_PHI) / 16 if f else float("nan")
        w = rate(r)
        hp[r["kind"]], ops[r["kind"]] = 128 / w, per_fused
        print(f"| {r['kind']} | {(b['total'] - LOOP) / 32:.2f} | {per_fused:.2f} | {w:.1f} | {128 / w:.2f} | "
              f"{1e9 / (w * sms * r['mhz'] * 1e6) * 1e6:.1f} |")

    print("\n| probe | steps/SM/clk | slots/step |")
    print("|---|---|---|")
    for r in probe:
        print(f"| {r['kind']} | {rate(r):.1f} | {128 / rate(r):.2f} |")

    base = {(r["mode"], r["C"]): rate(r) for r in fused if r["kind"] == "none"}
    rel = {(r["mode"], r["kind"], r["C"]): rate(r) / base[(r["mode"], r["C"])] for r in fused}

    def heff(mode: str, kind: str, c: int) -> float:
        return (1 / rel[(mode, kind, c)] - 1) * 32 * c / 8

    steady = {(m, k): statistics.median(heff(m, k, c) for c in STEADY) for m in MODES for k in kinds + controls}
    for mode in ("fp8", "s8"):
        print(f"\n{mode}: MMA rate over the bare loop's at L = 32 C, and h_eff (median over C = 4..32)\n")
        print("| kind | " + " | ".join(f"L={32 * c}" for c in CS) + " | h_eff |")
        print("|---|" + "---|" * (len(CS) + 1))
        for k in kinds + controls:
            cells = " | ".join(f"{rel[(mode, k, c)]:.3f}" for c in CS)
            print(f"| {k} | {cells} | {steady[(mode, k)]:.2f} |")

    print("\nh_eff by accumulator mode (median over C = 4..32)\n")
    print("| kind | " + " | ".join(MODES) + " |")
    print("|---|" + "---|" * len(MODES))
    for k in kinds + controls:
        print(f"| {k} | " + " | ".join(f"{steady[(m, k)]:.2f}" for m in MODES) + " |")

    print("\nncu at L = 32 (s8): pipe activity, % of peak sustained\n")
    cols = [("tensor", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"),
            ("alu", "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active"),
            ("fma", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"),
            ("issue", "smsp__issue_active.avg.pct_of_peak_sustained_active")]
    print("| kind | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for row, k in enumerate(["none"] + kinds + controls):
        m = ncu(o / f"ncu-fused-2-{row}.csv")
        if m:
            print(f"| {k} | " + " | ".join(f"{m.get(key, float('nan')):.1f}" for _, key in cols) + " |")

    def page_l(coef: float, h: float, n: int) -> float:
        return (-1 + math.sqrt(1 + 28 * coef * h / n)) / (14 / n)

    print(f"\nStrides and overhead, each h plus {args.merkle} for Merkle. The page's rule on B200 with h = SASS ops a "
          "word (its h_φ): L(1 + 7L/n) = 128 h, decode 7L/n. The same rule on sm_120 with h = pure slots: c = 8. "
          "Measured on sm_120 with h = s8 h_eff: L* = √(8hn/7), T* = 8h/L* + 7L*/n.\n")
    head = []
    for n in ns:
        head += [f"page B200: L, 7L/n (n={n})", f"rule sm_120: L, 7L/n (n={n})", f"measured: L*, T* (n={n})"]
    print("| kind | ops | pure slots | s8 h_eff | " + " | ".join(head) + " |")
    print("|---|---|---|---|" + "---|" * len(head))
    for k in kinds:
        hb, hr = ops[k] + args.merkle, hp[k] + args.merkle
        hm = max(steady[("s8", k)], 0.0) + args.merkle
        cells = []
        for n in ns:
            lb, ls = page_l(128, hb, n), page_l(8, hr, n)
            lstar = math.sqrt(8 * hm * n / 7)
            cells += [f"{lb:.0f}, {700 * lb / n:.1f}%", f"{ls:.0f}, {700 * ls / n:.2f}%",
                      f"{lstar:.0f}, {100 * (8 * hm / lstar + 7 * lstar / n):.1f}%"]
        print(f"| {k} | {ops[k]:.2f} | {hp[k]:.2f} | {steady[('s8', k)]:.2f} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
