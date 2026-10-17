"""The DRAM-free filler's tile (dcap.cu's `dfill_k`) in Python: the receipt a tile of NC chains writes after S chained
int8 MMAs, from the round's coins (c0, c1) and the tile's id. Written from the matrices, as dcap.cu's `dfill_ref` is;
`dcap dgate` holds the GPU to `dfill_ref`, and tests/test_dfill_ref.py holds this file to receipts `dfill_ref` printed."""

from __future__ import annotations

import numpy as np

M64 = (1 << 64) - 1


def smix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def dseed(c0: int, c1: int, tile: int) -> int:
    return smix(smix(c0 ^ smix(tile)) ^ c1)


def dword(seed: int, tag: int, i: int, j: int) -> int:
    """Word (i, j) of generated matrix `tag` (1 + b: B_b, i = column; 3: C; 16 + c: chain c's A_0, i = row)."""
    return smix(seed ^ (tag << 48) ^ (i << 24) ^ j) & 0xFFFFFFFF


def drec(seed: int, c: int, r: int, w: int, word: int) -> int:
    return smix(seed ^ ((c * 16 + r) << 40) ^ (w << 32) ^ word)


def s8_rows(seed: int, tag: int, rows: int) -> np.ndarray:
    """rows x 32 s8, element 4j + q of row i in byte q of dword(seed, tag, i, j)."""
    words = np.array([[dword(seed, tag, i, j) for j in range(8)] for i in range(rows)], dtype="<u4")
    return words.view(np.int8).astype(np.int64)


def step(a: np.ndarray, bt: np.ndarray, c: np.ndarray) -> np.ndarray:
    """D = A B + C, then A' = D's bytes under the fixed map (the accumulator registers reread as the next A fragment)."""
    d = (c + a @ bt.T).astype("<i4")
    b = lambda x: np.ascontiguousarray(x).view(np.int8)
    top = np.concatenate([b(d[:8, 0::2]), b(d[8:, 1::2])], axis=1)
    bot = np.concatenate([b(d[8:, 0::2]), b(d[:8, 1::2])], axis=1)
    return np.concatenate([top, bot]).astype(np.int64)


def receipt(c0: int, c1: int, tile: int, nc: int, steps: int) -> int:
    seed = dseed(c0, c1, tile)
    bt = [s8_rows(seed, 1 + b, 8) for b in range(2)]
    c = np.array([[(dword(seed, 3, r, n) & 0xFFFFFF) - 0x800000 for n in range(8)] for r in range(16)], dtype=np.int64)
    h = 0
    for ch in range(nc):
        a = s8_rows(seed, 16 + ch, 16)
        for s in range(steps):
            a = step(a, bt[s & 1], c)
        words = a.astype(np.int8).view("<u4")
        for r in range(16):
            for w in range(8):
                h ^= drec(seed, ch, r, w, int(words[r, w]))
    return h
