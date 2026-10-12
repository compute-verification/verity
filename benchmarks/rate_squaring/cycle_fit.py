"""Fit a warp-squaring's cycles on a sub-partition to its loop body's SASS classes, over every ILP 1 form of the runs.

cycles ≈ w·W + m·M + a·A, with W the IMAD.WIDE family (the products), M the other IMADs (the fold's IMAD and IMAD.HI,
IMAD.SHL) and A every other instruction but the loop's control. Fitted twice: every coefficient free, and w held at
the pipe probe's 4.02. The coefficients are what an instruction costs in these mixes, where the pipes overlap, not
its pipe's occupancy (the free fit's w is under the probe's 4.02). Each run directory is a fetched `sq.sh` run (`out/sass-mix.json`, `out/time.jsonl`); a later
directory's form replaces an earlier one's. Then it prices FOLD 5 with its MOV and SEL removed, the per-column overhead
the column form's 713-cycle ceiling leaves out.

    python3 cycle_fit.py ~/.research/runs/r20261012-175857-517b ... ~/.research/runs/r20261012-185914-35f2
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from gamma import LANES_A_CLOCK, MHZ, rate_of_cycles, s_of

KERNEL = re.compile(r"_ZN2sq(?:5chain|10chain_held)ILi(\d+)ELi1E")
CONTROL = {"BRA", "BSSY", "BSYNC", "UIADD3", "UISETP", "ISETP", "EXIT", "NOP"}
PROBE_W = 4.02


def classes(full: dict[str, int]) -> tuple[int, int, int]:
    w = sum(n for op, n in full.items() if op.startswith("IMAD.WIDE"))
    m = sum(n for op, n in full.items() if op.startswith("IMAD") and not op.startswith("IMAD.WIDE"))
    a = sum(n for op, n in full.items() if not op.startswith("IMAD") and op.split(".")[0] not in CONTROL)
    return w, m, a


def forms(runs: list[Path]) -> dict[int, tuple[tuple[int, int, int], float, dict[str, int]]]:
    out = {}
    for run in runs:
        mix = json.loads((run / "out/sass-mix.json").read_text())
        times = [json.loads(line) for line in (run / "out/time.jsonl").read_text().splitlines() if line]
        for name, entry in mix.items():
            m = KERNEL.match(name)
            if not m or "loop_body" not in entry:
                continue
            fold = int(m.group(1))
            rows = [t for t in times if t["fold"] == fold and t["ilp"] == 1]
            if not rows:
                continue
            rate = max(rows, key=lambda t: t["waves"])["sq_per_s_median"]
            full = entry["loop_body"]["full"]
            out[fold] = (classes(full), LANES_A_CLOCK * MHZ * 1e6 / rate, full)
    return out


def main(argv: list[str]) -> None:
    data = forms([Path(a) for a in argv])
    folds = sorted(data)
    x = np.array([data[f][0] for f in folds], float)
    y = np.array([data[f][1] for f in folds])
    free, *_ = np.linalg.lstsq(x, y, rcond=None)
    held, *_ = np.linalg.lstsq(x[:, 1:], y - PROBE_W * x[:, 0], rcond=None)
    held = np.r_[PROBE_W, held]
    print(f"free fit: w {free[0]:.2f}  m {free[1]:.2f}  a {free[2]:.2f};  w held at {PROBE_W}: m {held[1]:.2f}  "
          f"a {held[2]:.2f}  (cycles a warp-squaring a sub-partition)")
    for f, (w, m, a), cyc in zip(folds, x.astype(int), y):
        p, q = x[folds.index(f)] @ free, x[folds.index(f)] @ held
        print(f"  FOLD {f:2}: W {w:3} M {m:3} A {a:3}  measured {cyc:5.0f}  free {p:5.0f} ({100 * (p - cyc) / cyc:+5.1f}%)"
              f"  held {q:5.0f} ({100 * (q - cyc) / cyc:+5.1f}%)")
    if 5 in data:
        (w, m, a), cyc, full = data[5]
        cut = full.get("MOV", 0) + full.get("SEL", 0)
        for name, co in (("free", free), ("held", held)):
            c = cyc - co[2] * cut
            print(f"FOLD 5 less its {cut} MOV and SEL ({name} fit): {c:.0f} cycles, "
                  f"{rate_of_cycles(c):.3e}/s, s {s_of(rate_of_cycles(c)):.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
