"""Read sm90_clock's `lirate` (the fold with the lane alone, each checkpoint placement) beside its SASS counts.

For each `lirate<K,W>` kernel: the measured SMSP-cycles a unit a warp-slot (the median block, best rep), the loop's issue
count a unit (every instruction an issue slot), and its ALU-pipe count a unit at LOP3's half rate (2 cycles a warp
instruction for LOP3, IADD3, ISETP, SEL, LEA, SHF and PRMT). A loop bound by issue or by the ALU pipe runs at the larger
of the two, so `measured / max(issue, alu)` near 1 says each instruction costs what the sheet charges it and nothing
else; a REDG with a hidden cost (the L2's atomic unit) shows as CK_RED8 above it. The variants' differences are the
chip's price of moving the checkpoint's lane XOR to memory: CK_RED2 against CK_SEG is the red route's (134, 134)
against LI-CP-W's (136, 136), CK_SEG4 against CK_SEG is LI-CP-W's checkpoints four to a STG.128 (issue 135.25,
ALU 136), and CK_SEG4U and CK_SEG4M would move one LOP3 a checkpoint to two REDUX and a ULOP3 (136.25, 134) in all units
and in one of four (135.5, 135.5), but ptxas emits that XOR as two moves and a LOP3, and the uniform count printed for
each kernel says which form ran. Each REDG kernel's atomic lane-ops a SM-cycle (128 x REDG a unit over its cycles) is the
rate the memory side sustained. With `lirmix.jsonl`, each mix of NR in 8 units checkpointed by CK_RED2 against NR = 0
in the same loop: the cheater's saving within the atomic unit's spare capacity.

Run: `python lirate_analyze.py OUT` (OUT holds lirate.jsonl, and sass.json from the build, or pass `--sass FILE`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NAMES = ["ck_none", "ck_seg", "ck_red", "ck_red2", "ck_red8", "ck_seg4", "ck_seg4u", "ck_seg4m"]
ALU = ("LOP3", "IADD3", "ISETP", "SEL", "LEA", "SHF", "PRMT")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--sass", type=Path)
    args = ap.parse_args()
    rows = []
    lirate = args.out / "lirate.jsonl"
    for line in lirate.read_text().splitlines() if lirate.exists() else []:
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
        uni = sum(n for op, n in s["loop_base"].items() if op.startswith("U")) / 4
        bound = max(issue, alu)
        red = {"ck_red": 1, "ck_red2": 2, "ck_red8": 8}.get(name, 0)
        atom = f"; {128 * red / cyc:.3f} atomic lane-ops a SM-cycle" if red else ""
        print(f"{name:8s} W={w:2d} ({w // 4} a SMSP): {cyc:8.3f} SMSP-cycles a unit a warp-slot; SASS a unit: issue "
              f"{issue:6.2f}, ALU {alu:6.2f}, uniform {uni:5.2f}; measured / max {cyc / bound:.4f}{atom}")
    mix = args.out / "lirmix.jsonl"
    if mix.exists():
        runs: dict[int, list[float]] = {}
        for line in mix.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("probe") == "lirate_mix" and r["cuda"] == "no error":
                runs.setdefault(r["red2_of_8"], []).append(r["cyc_unit_slot_med"])
        base = min(runs.get(0, [float("nan")]))
        for nr, cs in sorted(runs.items()):
            cyc = min(cs)
            s = sass.get(f"lirate_mix<{nr},16>", {})
            issue = s.get("loop", 0) / 8
            print(f"mix {nr}/8 CK_RED2, W=16: {cyc:8.3f} SMSP-cycles a unit (reps {min(cs):.3f}-{max(cs):.3f}); SASS issue "
                  f"{issue:6.2f} a unit; against 0/8: {cyc - base:+.3f} a unit ({(cyc - base) / base:+.4%}); "
                  f"{256 * nr / 8 / cyc:.3f} atomic lane-ops a SM-cycle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
