"""Whether the pinned sm_120 tensor-core models accumulate `exact_grid.py`'s integer grids exactly: one dot product of
depth k chained through the model's steps from a zero accumulator, against the exact integer sum.

    e4m3   sm120.mma.m16n8k32.e4m3 (BLACKWELL_SM120_E4M3_M16N8K32: one group of 32 products and the accumulator, a 26-bit
           truncating adder): operands E4M3 bytes of integers
    nvf4   sm120.mma.m16n8k64.e2m1.nvf4 (BLACKWELL_SM120_NVF4) with every UE4M3 block scale 1.0 (0x38): operands E2M1
           codes of {1, 2, 3, 4, 6}

All operands are positive, so the sum and every partial sum grow to k E[a] E[b]: the worst case for a bound on the
partial sums.  Each case states whether its integer sum is at most 2^24 (exactness predicted) and whether the model's
word equals it.

    python exact_check.py [--json OUT]
"""
from __future__ import annotations

import json
import random
import struct
import sys

import torch

from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E4M3_M16N8K32 as E4M3, BLACKWELL_SM120_NVF4 as NVF4

from exact_grid import e4m3_ints

E2M1_CODE = {1: 2, 2: 4, 3: 5, 4: 6, 6: 7}
UNIT = 0x38


def f32(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def e4m3_byte(v: int) -> int:
    return int(torch.tensor([float(v)]).to(torch.float8_e4m3fn).view(torch.uint8).item())


def chain(model, a: list[int], b: list[int], words, step) -> float:
    acc = 0
    for s in range(0, len(a), step):
        acc = words(acc, a[s:s + step], b[s:s + step])
    return f32(acc)


def case(name: str, fmt: str, k: int, values: list[int], rng: random.Random, fixed: int | None = None) -> dict:
    pick = (lambda: fixed) if fixed is not None else (lambda: rng.choice(values))
    a, b = [pick() for _ in range(k)], [pick() for _ in range(k)]
    exact = sum(x * y for x, y in zip(a, b))
    if fmt == "e4m3":
        enc = {v: e4m3_byte(v) for v in set(a) | set(b)}
        got = chain(E4M3, [enc[v] for v in a], [enc[v] for v in b], E4M3.step, 32)
    else:
        def words(acc, xa, xb):
            return NVF4.step_scaled(acc, xa, xb, [UNIT] * 4, [UNIT] * 4)
        got = chain(NVF4, [E2M1_CODE[v] for v in a], [E2M1_CODE[v] for v in b], words, 64)
    return {"case": name, "format": fmt, "k": k, "max": max(values) if fixed is None else fixed, "sum": exact,
            "within_2^24": exact <= 2 ** 24, "model": got, "equal": got == exact}


def main(argv: list[str]) -> int:
    rng = random.Random(0)
    cases = [
        case("e4m3 |x|<=16, k=65536, all 16 (k M^2 = 2^24)", "e4m3", 65536, [16], rng, 16),
        case("e4m3 |x|<=16, k=65536, random 1..16", "e4m3", 65536, list(range(1, 17)), rng),
        case("e4m3 M(4096)=64, k=4096, all 64 (= 2^24)", "e4m3", 4096, [64], rng, 64),
        case("e4m3 M(4096)=64, k=4096, random E4M3 integers <= 64", "e4m3", 4096, e4m3_ints(64)[1:], rng),
        case("e4m3 M(14336)=32, k=14336, random E4M3 integers <= 32", "e4m3", 14336, e4m3_ints(32)[1:], rng),
        case("e4m3 to 448, k=4096, random E4M3 integers <= 448 (k M^2 > 2^24)", "e4m3", 4096, e4m3_ints(448)[1:], rng),
        case("e4m3 {13, 15}, k=262144 (sum > 2^24)", "e4m3", 262144, [13, 15], rng),
        case("nvf4 unit scales, k=465984, all 6 (k 36 <= 2^24)", "nvf4", 465984, [6], rng, 6),
        case("nvf4 unit scales, k=465984, random {1,2,3,4,6}", "nvf4", 465984, [1, 2, 3, 4, 6], rng),
        case("nvf4 unit scales, k=14336, random {1,2,3,4,6}", "nvf4", 14336, [1, 2, 3, 4, 6], rng),
        case("nvf4 unit scales, k=2097152, {3, 6} (sum > 2^24)", "nvf4", 2097152, [3, 6], rng),
    ]
    for c in cases:
        print(json.dumps(c), flush=True)
    if "--json" in argv:
        with open(argv[argv.index("--json") + 1], "w") as f:
            json.dump(cases, f, indent=1)
    return 0 if all(c["equal"] for c in cases if c["within_2^24"]) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
