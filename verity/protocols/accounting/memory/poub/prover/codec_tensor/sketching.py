"""The sketch arithmetic sketch-sha512's drivers share (`sketch_receipt`): a sketch vector's length, the exact integer
sketch S · e_v on the device and on the host, and the prmt that expands a sign selector into an MMA register.

~~~text
step     512 stored bytes of a task, m = ⌈SB / 512⌉ of them, the last zero-padded
vector   n = 32 m T bytes, T tasks' steps: a 4 KiB block at SB = 8192 and T = 8 (wt_set_pass's n)
sketch   σ_v = S · e_v over the integers for k × n signs S (bit 1 is −1, 0 is +1): |σ_v[c]| ≤ 128 n
~~~
"""
from __future__ import annotations

import numpy as np

STEP = 512


def steps(sb: int) -> int:
    return -(-sb // STEP)


def vector_bytes(sb: int, t: int) -> int:
    return 32 * steps(sb) * t


def sketch_torch(torch, vs, bits: np.ndarray, rows: int = 1 << 16):
    """S · e_v in float32 with TF32 off: exact, since every partial sum is an integer of magnitude at most 128 n < 2^24."""
    if 128 * bits.shape[1] >= 1 << 24:
        raise ValueError("float32 is exact only below n = 2^17")
    s = torch.from_numpy(1.0 - 2.0 * bits.astype(np.float32)).to(vs.device).t().contiguous()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = torch.empty(vs.shape[0], bits.shape[0], dtype=torch.int32, device=vs.device)
        for a in range(0, vs.shape[0], rows):
            out[a:a + rows] = (vs[a:a + rows].float() @ s).round().to(torch.int32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return out


def sketch(vs: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """S · e_v for every vector, exactly, through float64 BLAS (every partial sum an integer below 2^53)."""
    s = (1.0 - 2.0 * bits.astype(np.float64)).T
    out = np.empty((vs.shape[0], bits.shape[0]), np.int64)
    for a in range(0, vs.shape[0], 1 << 14):
        out[a:a + (1 << 14)] = np.rint(vs[a:a + (1 << 14)].astype(np.float64) @ s).astype(np.int64)
    return out


PRMT = ((0xFF01FF01, 0xFF01FF01), (0xFFFF0101, 0xFFFF0101), (0x01010101, 0xFFFFFFFF))


def prmt(lo: int, hi: int, sel: int) -> int:
    """PTX prmt.b32's default mode, on selector nibbles below 8."""
    src = lo | hi << 32
    return sum(((src >> 8 * ((sel >> 4 * j) & 7)) & 0xFF) << 8 * j for j in range(4))
