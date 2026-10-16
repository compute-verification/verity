"""pfill's tile (pfill.cuh's `dfill_k<NC>`, dcap.cu's filler unless built with -DDFILL_V1) in Python: the receipt a tile writes, from
the round's coins (c0, c1) and the tile's id, for each tile kind: the s8 tile on ±1 (K = 32), the e2m1 tile on ±1 (K = 64,
-DPFILL_E2M1) and the s8 tile on {0,1} bits (K = 32, -DPFILL_BITS, Theorem K's setting). Written from the matrices with
hashlib's SHA-512, as pfill.cuh's host `dfill_ref` is with its own; `dcap dgate` holds the GPU to that `dfill_ref`, and
tests/test_pfill_ref.py holds this file to that reference built with g++ (pfill_host.cpp)."""

from __future__ import annotations

import hashlib
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

_name = "dfill_ref"
if _name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_name, Path(__file__).with_name("dfill_ref.py"))
    sys.modules[_name] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules[_name])
smix, dseed = sys.modules[_name].smix, sys.modules[_name].dseed

W, TN = 8, 1024
DOM = 0x5046494C4C2D7631  # "PFILL-v1": the s8 tile
DOM4 = 0x5046494C4C2D6634  # "PFILL-f4": the e2m1 tile
DOM01 = 0x5046494C4C2D6231  # "PFILL-b1": the s8 tile on {0,1} bits
CTR_B, CTR_KEY = 1 << 56, 2 << 56
M32 = 0xFFFFFFFF
RAM, DUO, XOR = 4, 2, 1
S8, E2M1, BITS = 0, 1, 2  # pfill.cuh's PF_S8, PF_E2M1, PF_BITS
KINDS = (S8, E2M1, BITS)
DOMS = {S8: DOM, E2M1: DOM4, BITS: DOM01}


def block(c0: int, c1: int, tile: int, ctr: int, dom: int = DOM) -> np.ndarray:
    """A block's 16 row words: SHA-512 of the 40-byte (c0, c1, tile, dom, ctr), bytes 4r..4r+3 little-endian, so bit k
    of row r is entry 32 r + k's bit (entry value 1 - 2 bit on ±1, the bit itself on BITS)."""
    return np.frombuffer(hashlib.sha512(struct.pack(">5Q", c0, c1, tile, dom, ctr)).digest(), "<u4").astype(np.uint32)


def rows(c0: int, c1: int, tile: int, base: int, i: int, kind: int) -> np.ndarray:
    """Fragment or B block i's 16 rows over k < K: one block (s8 and bits, ctr base + i) or two (e2m1: ctr base + 2 i + h
    for k in 32 h .. 32 h + 31, in the high word for h = 1)."""
    if kind != E2M1:
        return block(c0, c1, tile, base + i, DOMS[kind]).astype(np.uint64)
    lo, hi = (block(c0, c1, tile, base + 2 * i + h, DOM4).astype(np.uint64) for h in (0, 1))
    return lo | (hi << np.uint64(32))


def key_ok(a: int) -> bool:
    m, r = a & 0xFFFFFF, a & 31
    return bool(a & 1) and 256 < m < (1 << 24) - 256 and r not in (1, 31)


def keys(rows: np.ndarray) -> tuple[int, int, int, int]:
    """(a0, a1, k0, k1): the first two row words that pass key_ok, then rows 14 and 15."""
    ok = [int(x) for x in rows if key_ok(int(x))]
    a = ok + [0x9E3779B9, 0x7F4A7C15][len(ok[:2]):]
    return a[0], a[1], int(rows[14]), int(rows[15])


def rotl(x: np.ndarray, r: int) -> np.ndarray:
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    x = x.astype(np.uint64)
    return sum(((x >> np.uint64(i)) & np.uint64(1)) for i in range(64)).astype(np.int64)


def words(a: np.ndarray, b: np.ndarray, kind: int) -> np.ndarray:
    """The committed words of A row bits a and B column bits b: s32 32 - 2 popcount(a ^ b) (s8), the bits of the f32
    64 - 2 popcount(a ^ b) (e2m1), or s32 popcount(a & b) (bits)."""
    if kind == BITS:
        return popcount(a & b).astype(np.uint32)
    if kind == E2M1:
        return (64 - 2 * popcount(a ^ b)).astype(np.float32).view(np.uint32)
    return ((32 - 2 * popcount(a ^ b)) & M32).astype(np.uint32)


def receipt(c0: int, c1: int, tile: int, nc: int, steps: int, kind: int = S8, w: int = W) -> int:
    """The tile's receipt: warp v's fragments v S .. v S + S - 1 (16 rows each) against all 128 w columns (w warps: the
    pump's tile is 8, dfill_p's PF_PW), fragment f's outputs absorbed by chain f % 4 of each lane in order of f, then of
    the B fragment j."""
    tn = 128 * w
    bb = np.concatenate([rows(c0, c1, tile, CTR_B, b, kind) for b in range(tn // 16)])  # bit k of bb[n]: entry (k, n)
    kb = block(c0, c1, tile, CTR_KEY, DOMS[kind])
    a0, a1, k0, k1 = (np.uint32(v) for v in keys(kb))
    lane = np.arange(32)
    g, t = lane >> 2, lane & 3
    pos = (np.arange(w)[:, None, None] * 32 + lane[None, :, None]) * 4 + np.arange(4)[None, None, :]
    y = pos.astype(np.uint32)
    z = (pos.astype(np.uint64) * 0x9E3779B9 & M32).astype(np.uint32)
    ar = np.stack([np.concatenate([rows(c0, c1, tile, 0, v * steps + f, kind) for f in range(steps)]
                                  or [np.zeros(0, np.uint64)]) for v in range(w)])  # ar[v, 16 f + i]: row i of fragment f
    jj = np.arange(tn // 8)
    b0 = bb[8 * jj[:, None] + 2 * t[None, :]]  # [j, lane]
    b1 = bb[8 * jj[:, None] + 2 * t[None, :] + 1]
    for fi in range((steps + 3) // 4):
        f = 4 * fi + np.arange(4)
        live = f < steps
        fc = np.minimum(f, max(steps - 1, 0))
        r0 = ar[:, 16 * fc[None, :] + g[:, None]]  # [w, lane, c]
        r1 = ar[:, 16 * fc[None, :] + g[:, None] + 8]
        d0, d1, d2, d3 = (words(r[None], b[:, None, :, None], kind) for r in (r0, r1) for b in (b0, b1))  # [j, w, lane, c]
        for j in range(tn // 8):
            ny, nz = y.copy(), z.copy()
            if nc == RAM:
                ny = (rotl(ny, 7) + d0[j]) * a0 + d1[j]
                ny = (rotl(ny, 19) + d2[j]) * a1 + d3[j]
            elif nc == DUO:
                for p, k, a, r in ((d0[j], k0, a0, 5), (d1[j], k1, a1, 11), (d2[j], k0, a0, 17), (d3[j], k1, a1, 29)):
                    u = ny ^ p ^ k
                    ny = u * a + nz
                    nz = rotl(u, r) + ny
            else:
                ny = ny ^ d0[j] ^ d1[j] ^ d2[j] ^ d3[j]
            y = np.where(live[None, None, :], ny, y)
            z = np.where(live[None, None, :], nz, z)
    key = dseed(c0, c1, tile)
    h = 0
    v = (z.astype(np.uint64) << np.uint64(32)) | y if nc == DUO else y.astype(np.uint64)
    for p, x in zip(pos.ravel().tolist(), v.ravel().tolist()):
        h ^= smix(smix(key ^ p) ^ x)
    return h
