"""The receipt φ cost tables, rendered as Markdown from one `phi.sh` run's output directory.

    python3 tables.py OUT_DIR [--merkle 0.4] [--n 16384,65536] [--mode s8] > tables.md

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

Each mode's rel is over its own bare loop: the interleaved modes' (fp8il, s8il) MMAs chain through a LOP3 each, and the
bare rates line shows what that costs against peak (8 MMA lanes a clock is 1024 MACs). The staggered modes (s8st,
s8ppst) start each warp of an SMSP a share of a stride late. `fused-occ2.jsonl`, when
present, is the same sweep with every kernel pinned to 2 blocks per SM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

MODES = ("fp8", "fp8pp", "s8", "s8pp", "fp8il", "s8il", "s8st", "s8ppst")
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
    ap.add_argument("--mode", default="s8", help="the fused mode whose h_eff the measured strides use")
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

    def sweep(rows: list[dict]) -> tuple[dict, dict, dict]:
        base = {(r["mode"], r["C"]): rate(r) for r in rows if r["kind"] == "none"}
        rel = {(r["mode"], r["kind"], r["C"]): rate(r) / base[(r["mode"], r["C"])] for r in rows}
        modes = [m for m in MODES if (m, 1) in base]
        steady = {(m, k): statistics.median((1 / rel[(m, k, c)] - 1) * 32 * c / 8 for c in STEADY)
                  for m in modes for k in kinds + controls}
        return base, rel, steady

    occ2 = o / "fused-occ2.jsonl"
    sweeps = {"": sweep(fused)} | ({" (2 blocks per SM)": sweep(jsonl(occ2))} if occ2.exists() else {})
    for tag, (base, rel, steady) in sweeps.items():
        modes = sorted({m for m, _ in base}, key=MODES.index)
        print(f"\nBare loop MMA lanes per SM-clock{tag}: "
              + ", ".join(f"{m} {statistics.median(base[(m, c)] for c in CS):.3f}" for m in modes))
        for mode in [m for m in ("fp8", "s8", "s8il", "s8ppst") if m in modes]:
            print(f"\n{mode}{tag}: MMA rate over the bare loop's at L = 32 C, and h_eff (median over C = 4..32)\n")
            print("| kind | " + " | ".join(f"L={32 * c}" for c in CS) + " | h_eff |")
            print("|---|" + "---|" * (len(CS) + 1))
            for k in kinds + controls:
                cells = " | ".join(f"{rel[(mode, k, c)]:.3f}" for c in CS)
                print(f"| {k} | {cells} | {steady[(mode, k)]:.2f} |")
        print(f"\nh_eff by accumulator mode{tag} (median over C = 4..32)\n")
        print("| kind | " + " | ".join(modes) + " |")
        print("|---|" + "---|" * len(modes))
        for k in kinds + controls:
            print(f"| {k} | " + " | ".join(f"{steady[(m, k)]:.2f}" for m in modes) + " |")
    steady = sweeps[""][2]

    cols = [("tensor", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"),
            ("alu", "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active"),
            ("fma", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"),
            ("issue", "smsp__issue_active.avg.pct_of_peak_sustained_active")]
    for title, stem in [("L = 32 (s8)", "ncu-fused-2"), ("L = 32 (s8il)", "ncu-fused-5"),
                        ("L = 256 (s8pp, 2 blocks per SM)", "ncu-fused-c8-3"),
                        ("L = 256 (s8il, 2 blocks per SM)", "ncu-fused-c8-5"),
                        ("L = 256 (s8ppst, 2 blocks per SM)", "ncu-fused-c8-7")]:
        rows = [(k, ncu(o / f"{stem}-{row}.csv")) for row, k in enumerate(["none"] + kinds + controls)]
        if not any(m for _, m in rows):
            continue
        print(f"\nncu at {title}: pipe activity, % of peak sustained\n")
        print("| kind | " + " | ".join(c for c, _ in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for k, m in rows:
            if m:
                print(f"| {k} | " + " | ".join(f"{m.get(key, float('nan')):.1f}" for _, key in cols) + " |")

    def page_l(coef: float, h: float, n: int) -> float:
        return (-1 + math.sqrt(1 + 28 * coef * h / n)) / (14 / n)

    print(f"\nStrides and overhead, each h plus {args.merkle} for Merkle. The page's rule on B200 with h = SASS ops a "
          "word (its h_φ): L(1 + 7L/n) = 128 h, decode 7L/n. The same rule on sm_120 with h = pure slots: c = 8. "
          f"Measured on sm_120 with h = {args.mode} h_eff: L* = √(8hn/7), T* = 8h/L* + 7L*/n.\n")
    head = []
    for n in ns:
        head += [f"page B200: L, 7L/n (n={n})", f"rule sm_120: L, 7L/n (n={n})", f"measured: L*, T* (n={n})"]
    print(f"| kind | ops | pure slots | {args.mode} h_eff | " + " | ".join(head) + " |")
    print("|---|---|---|---|" + "---|" * len(head))
    for k in kinds:
        hb, hr = ops[k] + args.merkle, hp[k] + args.merkle
        hm = max(steady[(args.mode, k)], 0.0) + args.merkle
        cells = []
        for n in ns:
            lb, ls = page_l(128, hb, n), page_l(8, hr, n)
            lstar = math.sqrt(8 * hm * n / 7)
            cells += [f"{lb:.0f}, {700 * lb / n:.1f}%", f"{ls:.0f}, {700 * ls / n:.2f}%",
                      f"{lstar:.0f}, {100 * (8 * hm / lstar + 7 * lstar / n):.1f}%"]
        print(f"| {k} | {ops[k]:.2f} | {hp[k]:.2f} | {steady[(args.mode, k)]:.2f} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
