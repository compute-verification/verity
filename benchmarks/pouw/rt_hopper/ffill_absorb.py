"""How many of ffill's FFMA steps an adversary's words can make absorbed (the chain unchanged, so the step
can be skipped with the receipt still right), with ffill_ref's exact FMA.

A step is absorbed when f·s is under half an ulp of x, so |x| must reach about 2^24·f·s. f = BIAS + w read as binary32
doubles for every 2^23 that w rises, so absorption needs the words of one chain to span about 2^26.7. At the design's
alphabets (s8 activations, |weight| ≤ 63) a k32 prefix moves at most 32·128·63 a segment and |w| ≤ K·8064. The
patterns drive the three filler words of a thread (REGS) to those extremes in every segment, every lane alike, which
is what skipping a warp instruction needs.

    python ffill_absorb.py
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dc"))
import ffill_ref as F  # noqa: E402

MAXP = 128 * 63
D = 32 * MAXP
PATTERNS = {
    "a up, b and c down": lambda j, S: ((j + 1) * D, -(j + 1) * D, -(j + 1) * D),
    "a up, b down, c up then down": lambda j, S: ((j + 1) * D, -(j + 1) * D, (min(j, S - 1 - j) + 1) * D),
    "all up": lambda j, S: ((j + 1) * D,) * 3,
}


def f_of(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", F.operand(w)))[0]


def absorbed(K: int, pattern, seeds: int = 5) -> tuple[int, int]:
    """(absorbed, total) chain steps over `seeds` tiles' coins."""
    hit = total = 0
    for sd in range(seeds):
        s, t, x, y = F.coins(sd, sd ^ 0x55, sd)
        n = 0
        for j in range(K // 32):
            for w in pattern(j, K // 32):
                nx, ny = F.step(x, y, w, s, t, n)
                hit += (nx == x) + (ny == y)
                total += 2
                x, y, n = nx, ny, n + 1
    return hit, total


def main() -> int:
    for K in (8192, 14336, 28672):
        wmax = K * MAXP
        print(f"K = {K}: |w| ≤ {wmax} (2^{math.log2(wmax):.1f}), f spans 2^{math.log2(f_of(wmax) / f_of(-wmax)):.1f}")
        for name, p in PATTERNS.items():
            h, tot = absorbed(K, p)
            print(f"  {name:<30} absorbed {h} of {tot} chain steps ({h / tot:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
