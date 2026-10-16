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


# Map 2 (dcap.cu's MAP 2): a 128-bit tile seed (lo, hi) that every word and receipt term reads, and C in [-2^15, 2^15).
def dseed_x(c0: int, c1: int, tile: int, map_: int) -> tuple[int, int]:
    hi = smix(smix(c1 ^ smix(tile ^ 0xD1B54A32D192ED03)) ^ c0) if map_ == 2 else 0
    return dseed(c0, c1, tile), hi


def dword_x(seed: tuple[int, int], map_: int, tag: int, i: int, j: int) -> int:
    lo, hi = seed
    if map_ == 2:
        return smix(hi ^ smix(lo ^ (tag << 48) ^ (i << 24) ^ j)) & 0xFFFFFFFF
    return dword(lo, tag, i, j)


def dcval_x(seed: tuple[int, int], map_: int, r: int, n: int) -> int:
    if map_ == 2:
        return (dword_x(seed, map_, 3, r, n) & 0xFFFF) - 0x8000
    return (dword_x(seed, map_, 3, r, n) & 0xFFFFFF) - 0x800000


def drec_x(seed: tuple[int, int], map_: int, c: int, r: int, w: int, word: int) -> int:
    lo, hi = seed
    h = drec(lo, c, r, w, word)
    return smix(hi ^ h) if map_ == 2 else h


def s8_rows(seed: tuple[int, int], tag: int, rows: int, map_: int = 0) -> np.ndarray:
    """rows x 32 s8, element 4j + q of row i in byte q of word (i, j) of generated matrix `tag`."""
    words = np.array([[dword_x(seed, map_, tag, i, j) for j in range(8)] for i in range(rows)], dtype="<u4")
    return words.view(np.int8).astype(np.int64)


def step(a: np.ndarray, bt: np.ndarray, c: np.ndarray, map_: int = 0) -> np.ndarray:
    """D = A B + C, then A' = D's bytes under the fixed map (the accumulator registers reread as the next A fragment).
    Map 0 is FILL v1's (a' = d0, d2, d3, d1); maps 1 and 2 read the accumulator quad in register order (a' = d0, d1, d2, d3)."""
    d = (c + a @ bt.T).astype("<i4")
    b = lambda x: np.ascontiguousarray(x).view(np.int8)
    if map_:
        top = np.concatenate([b(d[:8, 0::2]), b(d[8:, 0::2])], axis=1)
        bot = np.concatenate([b(d[:8, 1::2]), b(d[8:, 1::2])], axis=1)
    else:
        top = np.concatenate([b(d[:8, 0::2]), b(d[8:, 1::2])], axis=1)
        bot = np.concatenate([b(d[8:, 0::2]), b(d[:8, 1::2])], axis=1)
    return np.concatenate([top, bot]).astype(np.int64)


def receipt(c0: int, c1: int, tile: int, nc: int, steps: int, map_: int = 0) -> int:
    seed = dseed_x(c0, c1, tile, map_)
    bt = [s8_rows(seed, 1 + b, 8, map_) for b in range(2)]
    c = np.array([[dcval_x(seed, map_, r, n) for n in range(8)] for r in range(16)], dtype=np.int64)
    h = 0
    for ch in range(nc):
        a = s8_rows(seed, 16 + ch, 16, map_)
        for s in range(steps):
            a = step(a, bt[s & 1], c, map_)
        words = a.astype(np.int8).view("<u4")
        for r in range(16):
            for w in range(8):
                h ^= drec_x(seed, map_, ch, r, w, int(words[r, w]))
    return h
