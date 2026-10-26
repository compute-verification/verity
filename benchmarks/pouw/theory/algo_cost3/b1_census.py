"""A34: emulated b1's cost on the RTX PRO 6000 from the measured sm_120 census, in INT8 product-times a bit MAC.

A bit MAC b·c is one bit of AND(B word, C word); a dot product popcounts the ANDed words. A strategy takes g ANDed
words, reduces them by a carry-save tree of full adders (two LOP3s each: the XOR3 sum and the majority carry) to r
weighted words, popcounts those (POPC, on the XU), and adds the counts into accumulators, either with IADD3 on the ALU
(two counts an op, one accumulator a weight) or with IMAD on its own pipe (one count an op, the weight folded in).
`hs16` is Harley–Seal: 16 words a block into carried twos/fours/eights, so one POPC a block.

`dp4a2` runs on the IMAD pipe instead: two bits a byte, a = a0 + 128 a1 against b = b1 + 128 b0 (u8), so one IDP.4A.U8
adds a0b1 + 128 (a0b0 + a1b1) + 16384 a1b0 over four bytes, 8 bit MACs. Fifteen of them chained from zero keep the
middle field exact (at most 120, and the low one at most 60), and a LOP3 mask and half an IADD3 add it to the total.

Rates are warp instructions a SM-clock from `machine_model/data/rows.csv` (rows A, XU, I) and one instruction a
SMSP-clock of issue (DS_issue). The tensor row M (every MMA holds the pipe 16 SMSP-cycles) gives m16n8k32 IMMA's
16·8·32·4/16 = 1024 INT8 MACs a SM-clock. A warp instruction on 32 lanes of one word each covers 1024 bit MACs, so a
strategy's per-word counts over each pipe's rate are product-times directly. The LP mixes strategies to balance the
pipes, first the carry-save trees alone, then with dp4a2. It ignores operand delivery and the register file
(`b1alu_check.py --census` charges those the way D5 does).

    python3 b1_census.py [ROWS_CSV]
"""
import csv
import re
import sys
from pathlib import Path

from scipy.optimize import linprog

ROWS = Path(__file__).parents[2] / "machine_model/data/rows.csv"
MMA_MACS = 16 * 8 * 32 * 4 / 16

STRATEGIES = {"plain": (1, 0, 1), "fa3": (3, 1, 2), "fa7": (7, 4, 3), "fa15": (15, 11, 4), "hs16": (16, 15, 1)}
DP4A_CHAIN, DP4A_BITS = 15, 8


def rate(rows, name):
    return float(re.search(r"([0-9.]+) /SM/clk", rows[name]).group(1))


def per_word(g, fas, pops, acc):
    alu = g + 2 * fas + (pops / 2 if acc == "iadd3" else 0)
    imad = pops if acc == "imad" else 0
    return {"A": alu / g, "XU": pops / g, "I": imad / g, "issue": (alu + pops + imad) / g}


def dp4a2():
    """Per 1024 bit MACs, like `per_word`: 32 / 8 IDP.4A warp instructions and their chains' masks and adds."""
    n = 32 / DP4A_BITS
    alu = n * 1.5 / DP4A_CHAIN
    return {"A": alu, "XU": 0.0, "I": n, "issue": n + alu}


def cheapest(cols, cap):
    n = len(cols)
    res = linprog([0.0] * n + [1.0],
                  A_ub=[[t[p] for _, t in cols] + [-1.0] for p in cap], b_ub=[0.0] * len(cap),
                  A_eq=[[1.0] * n + [0.0]], b_eq=[1.0], bounds=[(0, None)] * (n + 1))
    assert res.status == 0, res.message
    return res.x[-1], {cols[i][0]: round(float(x), 3) for i, x in enumerate(res.x[:n]) if x > 1e-6}


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROWS
    rows = {r["row"]: r["description"] for r in csv.DictReader(src.open())}
    cap = {"A": rate(rows, "A"), "XU": rate(rows, "XU"), "I": rate(rows, "I"), "issue": 4.0}
    print(f"sm_120 census: ALU {cap['A']}, XU {cap['XU']}, IMAD {cap['I']}, issue {cap['issue']} warp instructions "
          f"a SM-clock; INT8 MMA {MMA_MACS:.0f} MACs a SM-clock")
    cols = []
    words = [(f"{name}/{acc}", per_word(g, fas, pops, acc))
             for name, (g, fas, pops) in STRATEGIES.items() for acc in ("iadd3", "imad")]
    for name, w in words + [("dp4a2", dp4a2())]:
        t = {p: w[p] / cap[p] * 1024 / MMA_MACS for p in cap}
        cols.append((name, t))
        print(f"  {name:11}: " + "  ".join(f"{p} {v:.3f}" for p, v in t.items())
              + f"  -> {max(t.values()):.3f} product-times a bit MAC")
    for label, cs in (("carry-save trees", cols[:-1]), ("with dp4a2", cols)):
        T, mix = cheapest(cs, cap)
        print(f"cheapest mix, {label}: {T:.3f} product-times a bit MAC, {mix}")


if __name__ == "__main__":
    main()
