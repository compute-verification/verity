#!/usr/bin/env python3
"""Referee check of FOLD's fractional round (note:20261017T0320Z-draft-eps1-fold §9): adjacent committed words absorbed
once and twice, (c0, c1) = (1, 2) and (2, 1), against the shapes §9 cites, (1, 1) (g3-chains' run of two) and (2, 2)
(§8's even depth).

`chain_kernel.c` meets in the middle over every pair of words at width 6, so each count is exact: the whole all-state
kernel of the two-word chain, split into per-word elements (each word's own R-fold kernel) and cross-word elements (runs
of two). §9 says the fractional round is covered by §8; that holds at a key if the mixed shapes have no cross-word
element that (2, 2) doesn't also have.

  uv run python benchmarks/pouw/theory/chain_kernel_check.py [--quick] [--verdict PATH]
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from exact_kit.checks import Checks

HERE = Path(__file__).resolve().parent
N, R1, R2 = 6, 2, 4
W0, W1 = (0x1E, 0x37, 0x39, 0x2B), (0x05, 0x2A, 0x11, 0x3C)


def run(keys, tmp: Path):
    exe = tmp / "chain_kernel"
    subprocess.run(["gcc", "-O2", "-fopenmp", "-o", str(exe), str(HERE / "chain_kernel.c")], check=True)
    out = subprocess.run([str(exe), str(N), str(R1), str(R2), *(str(x) for x in W0 + W1)],
                         input="".join(f"{a} {b}\n" for a, b in keys), capture_output=True, text=True, check=True).stdout
    res, ex = {}, {}
    for line in out.splitlines():
        p = line.split()
        if p[0] == "S":
            res[(int(p[1]), int(p[2]), int(p[3]), int(p[4]))] = (int(p[5]), int(p[6]))
        elif p[0] == "N":
            ex.setdefault((int(p[1]), int(p[2]), int(p[3]), int(p[4])), []).append(tuple(int(x) for x in p[5:13]))
    return res, ex


def main() -> int:
    checks = Checks()
    rng = np.random.default_rng(20261017)
    m = (1 << N) - 1
    t = 1 << (N - 2)
    near1 = sorted({(s * t + e) & m for s in range(4) for e in (1, -1)})
    count = 8 if "--quick" in checks.args else 40
    keys = [(int(rng.integers(0, 1 << N)) | 1, int(rng.integers(0, 1 << N)) | 1) for _ in range(count)]
    keys += [(a, int(rng.integers(0, 1 << N)) | 1) for a in near1[:4]] + [(int(rng.integers(0, 1 << N)) | 1, a)
                                                                         for a in near1[:4]]
    with tempfile.TemporaryDirectory() as tmp:
        res, ex = run(keys, Path(tmp))
    mixed_only, cross = [], {s: 0 for s in ((1, 1), (1, 2), (2, 1), (2, 2))}
    for a0, a1 in keys:
        a0, a1 = a0 & m, a1 & m
        for s in cross:
            cross[s] += res[(a0, a1) + s][1] > 0
        if (res[(a0, a1, 1, 2)][1] or res[(a0, a1, 2, 1)][1]) and not res[(a0, a1, 2, 2)][1]:
            mixed_only.append((a0, a1, res[(a0, a1, 1, 2)], res[(a0, a1, 2, 1)], ex.get((a0, a1, 1, 2), [])[:2]))
    checks.check(f"n{N} ({len(keys)} keys): no key where the fractional round's mixed chain (1,2) or (2,1) has a "
                 "cross-word kernel element that (2,2) doesn't", not mixed_only,
                 f"keys with cross-word elements by shape {cross}; mixed-only keys {mixed_only[:4]}")
    checks.check(f"n{N}: the plain run of two (1,1) has cross-word elements at some key (the P/X/N group spans words), "
                 "so the count sees runs of two", cross[(1, 1)] > 0, f"{cross[(1, 1)]} keys")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
