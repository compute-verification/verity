"""`circuit.h1zy`'s dequant against the device library's: `int8_exact.dequantize`'s expression,
(z.float() * s_a[:, None] * s_w).to(bfloat16), run by torch on the CPU, is the circuit's bit for bit but for the NaN encoding."""
from __future__ import annotations

import random
import struct

import numpy as np
import pytest
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import h1zy as Z

M32 = (1 << 32) - 1


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


INF, NAN, MAX, TINY = f32(float("inf")), 0x7FC00000, 0x7F7FFFFF, 0x00000001
#: (z, s_a, s_w) at the corners: zeros of each sign, the s32 extremes, z past 2^24 (int to f32 rounds), subnormal and huge
#: scales (an underflow to a subnormal or zero, an overflow to inf), inf and NaN scales, and served-like 1/127 scales
EDGES = [(0, f32(0.5), f32(0.25)), (0, f32(-1.0), f32(0.5)), (0, f32(-1.0), f32(-1.0)), (1, 0, f32(1.0)), (1, 0x80000000, f32(1.0)),
         ((1 << 31) - 1, f32(1.0), f32(1.0)), (-(1 << 31), f32(1.0), f32(1.0)), ((1 << 24) + 1, f32(1.0), f32(1.0)),
         ((1 << 24) + 3, f32(1.0), f32(3.0)), (12345, TINY, f32(1.0)), (3, TINY, f32(0.5)), (-7, f32(1e-20), f32(1e-20)),
         ((1 << 30), MAX, f32(2.0)), (-5, MAX, MAX), (2, INF, f32(1.0)), (0, INF, f32(1.0)), (9, NAN, f32(1.0)),
         (9, f32(1.0), NAN), (-1, f32(1 / 127), f32(1 / 127)), (1000, f32(3.1 / 127), f32(0.07 / 127)),
         (0x7F7F, f32(1.0), f32(1.0)), (0x8080, f32(1.0), f32(1.0)), (0x18181, f32(1.0), f32(1.0))]


def _rand(rng: random.Random) -> tuple[int, int, int]:
    """A served-like cell: an s32 dot of a K = 4096 tile and per-row scales of a layer's absmax range."""
    z = rng.randrange(-(1 << 26), 1 << 26) if rng.random() < 0.8 else rng.randrange(-(1 << 31), 1 << 31)
    return z & M32, f32(rng.uniform(1e-4, 0.1) / 127), f32(rng.uniform(1e-3, 0.5) / 127)


def _nan(y: int) -> bool:
    return (y & 0x7F80) == 0x7F80 and bool(y & 0x7F)


def test_the_dequant_is_torchs_scaled_mm_epilogue_but_for_the_nan_encoding():
    """`int8_exact.dequantize`'s expression, (z.float() * s_a[:, None] * s_w).to(bfloat16), on the CPU: equal but where the
    result is NaN, which the GPU's cvt.rn encodes 0x7FFF and torch's CPU cast 0x7FC0 (scalar) or 0xFFFF (vectorized)."""
    torch = pytest.importorskip("torch")
    rng = random.Random(34)
    cases = EDGES + [_rand(rng) for _ in range(2000)]
    z = torch.from_numpy(np.array([c[0] & M32 for c in cases], np.uint32).view(np.int32))
    sa = torch.from_numpy(np.array([c[1] for c in cases], np.uint32).view(np.float32))
    sw = torch.from_numpy(np.array([c[2] for c in cases], np.uint32).view(np.float32))
    y = (z.float() * sa * sw).to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16).tolist()
    ours = [Z.dequant(*c) for c in cases]
    assert [o for o, t in zip(ours, y) if not _nan(t)] == [t for t in y if not _nan(t)]
    assert [o for o, t in zip(ours, y) if _nan(t)] == [0x7FFF] * sum(map(_nan, y)) and any(map(_nan, y))
