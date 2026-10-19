"""Read sm90_clock's `lirate` (the fold with the lane alone, each checkpoint placement) beside its SASS counts.

For each `lirate<K,W>` kernel: the measured SMSP-cycles a unit a warp-slot (the median block, best rep), the loop's issue
count a unit (every instruction an issue slot), and its ALU-pipe count a unit at LOP3's half rate (2 cycles a warp
instruction for LOP3, IADD3, ISETP, SEL, LEA, SHF and PRMT). A loop bound by issue or by the ALU pipe runs at the larger
of the two, so `measured / max(issue, alu)` near 1 says each instruction costs what the sheet charges it and nothing
else; a REDG with a hidden cost (the L2's atomic unit) shows as CK_RED8 above it. The variants' differences are the
chip's price of moving the checkpoint's lane XOR to memory: CK_RED2 against CK_SEG is RT2-HOPPER-1's (134, 134) route
against LI-CP-W's (136, 136).

Run: `python lirate_analyze.py OUT` (OUT holds lirate.jsonl, and sass.json from the build, or pass `--sass FILE`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NAMES = ["ck_none", "ck_seg", "ck_red", "ck_red2", "ck_red8"]
ALU = ("LOP3", "IADD3", "ISETP", "SEL", "LEA", "SHF", "PRMT")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--sass", type=Path)
    args = ap.parse_args()
    rows = []
    for line in (args.out / "lirate.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    sass = json.loads((args.sass or args.out / "sass.json").read_text())["kernels"]
    best: dict[tuple[str, int], float] = {}
    for r in rows:
        if r.get("probe") == "lirate" and r["cuda"] == "no error":
            k = (r["name"], r["warps"])
            best[k] = min(best.get(k, float("inf")), r["cyc_unit_slot_med"])
    for (name, w), cyc in sorted(best.items(), key=lambda x: (x[0][1], NAMES.index(x[0][0]))):
        s = sass[f"lirate<{NAMES.index(name)},{w}>"]
        issue = s["loop"] / 4
        alu = 2 * sum(n for op, n in s["loop_base"].items() if op in ALU) / 4
        bound = max(issue, alu)
        print(f"{name:8s} W={w:2d} ({w // 4} a SMSP): {cyc:8.3f} SMSP-cycles a unit a warp-slot; SASS a unit: issue "
              f"{issue:6.2f}, ALU {alu:6.2f}; measured / max {cyc / bound:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
