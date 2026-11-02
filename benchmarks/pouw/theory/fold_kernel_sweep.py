#!/usr/bin/env python3
"""A wider sample for the corrected depth-R criterion: K_R = K_1 (the P/X/N group) at random key_ok_r11 keys and random words.

`fold_kernel_check.py` enumerates the whole all-state kernel of the same-word R-fold at width 8 on 56 structured keys, of
which 19 pass key_ok_r11. This runs the same enumerator (`fold_kernel.c`) on N random key_ok_r11 keys, each with its own
random word, R = 1..5, and checks that depth adds no kernel element and that every element keeps (G₃). Each key takes
about 11 s on four cores, so the batch progress goes to stderr.

  uv run python benchmarks/pouw/theory/fold_kernel_sweep.py [--keys N] [--verdict PATH]
"""
from __future__ import annotations

import collections
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fold_kernel_check as fk
from exact_kit.checks import Checks

N, R1, R2, RMAX, BATCH = 8, 3, 5, 5, 8


def main() -> int:
    checks = Checks()
    args = checks.args
    count = int(args[args.index("--keys") + 1]) if "--keys" in args else 64
    rng = np.random.default_rng(20261017 + count)
    odd = lambda: int(rng.integers(0, 1 << N)) | 1
    keys = []
    while len(keys) < count:
        a0, a1 = odd(), odd()
        if fk.key_ok_r11(a0, a1, N) and (a0, a1) not in keys:
            keys.append((a0, a1))
    sizes, extra, g3_bad = collections.Counter(), [], []
    with tempfile.TemporaryDirectory() as tmp:
        exe = fk.build(Path(tmp))
        for s in range(0, count, BATCH):
            word = tuple(int(x) for x in rng.integers(0, 1 << N, 4))
            res = fk.enumerate_kernels(exe, N, R1, R2, RMAX, word, keys[s:s + BATCH])
            for key, kr in res.items():
                sizes[len(kr[1]) + 1] += 1
                extra += [(key, r, len(kr[r] - kr[1])) for r in range(2, RMAX + 1) if kr[r] != kr[1]]
                g3_bad += [(key, w2) for r in range(1, RMAX + 1) for w2 in kr[r] if not fk.g3_ok(word, w2, N)]
            print(f"{min(s + BATCH, count)}/{count} keys: |K_1| {dict(sizes)}, K_R != K_1 at {len(extra)}, "
                  f"(G3) violations {len(g3_bad)}", file=sys.stderr, flush=True)
    checks.check(f"n{N}, {count} random key_ok_r11 keys, a random word every {BATCH}: K_1 is the 16-element P/X/N group",
                 set(sizes) == {16}, f"|K_1| histogram {dict(sizes)}")
    checks.check(f"n{N}, {count} random key_ok_r11 keys: K_R = K_1 for R = 2..{RMAX} (depth adds no kernel element)",
                 not extra, f"(key, R, extra) {extra[:6]}")
    checks.check(f"n{N}, {count} random key_ok_r11 keys: every kernel element keeps (G₃)", not g3_bad, str(g3_bad[:4]))
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
