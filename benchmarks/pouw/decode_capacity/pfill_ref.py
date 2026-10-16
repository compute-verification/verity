"""pfill's tile (pfill.cuh's `dfill_k<NC>`, dcap.cu's filler unless built with -DDFILL_V1) in Python: the receipt a tile writes, from
the round's coins (c0, c1) and the tile's id, for the s8 tile (K = 32), the e2m1 tile (K = 64, -DPFILL_E2M1) and the bU
tile (K = 32, A on {0, 1} and B on {±1, ..., ±64}, -DPFILL_BU). Written from the matrices with hashlib's SHA-512, as
pfill.cuh's host `dfill_ref` is with its own; `dcap dgate` holds the GPU to that `dfill_ref`, and tests/test_pfill_ref.py
holds this file to that reference built with g++ (pfill_host.cpp)."""

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
DOM_BU = 0x5046494C4C2D6255  # "PFILL-bU": the bU tile
CTR_B, CTR_KEY = 1 << 56, 2 << 56
M32 = 0xFFFFFFFF
RAM, DUO, XOR = 4, 2, 1
S8, E2M1, BU = 0, 1, 3  # the tile kinds, as pfill.cuh numbers them (a bool `e2m1` reads as S8 or E2M1)


def dom_of(kind: int) -> int:
    return {S8: DOM, E2M1: DOM4, BU: DOM_BU}[int(kind)]


def digest(c0: int, c1: int, tile: int, ctr: int, dom: int = DOM) -> bytes:
    return hashlib.sha512(struct.pack(">5Q", c0, c1, tile, dom, ctr)).digest()


def block(c0: int, c1: int, tile: int, ctr: int, dom: int = DOM) -> np.ndarray:
    """A block's 16 row words: SHA-512 of the 40-byte (c0, c1, tile, dom, ctr), bytes 4r..4r+3 little-endian, so bit k
    of row r is entry 32 r + k's bit (entry value 1 - 2 bit; bU's A: the bit)."""
    return np.frombuffer(digest(c0, c1, tile, ctr, dom), "<u4").astype(np.uint32)


def rows(c0: int, c1: int, tile: int, base: int, i: int, kind: int) -> np.ndarray:
    """Fragment or B block i's 16 rows over k < K: one block (s8 and bU's A, ctr base + i) or two (e2m1: ctr base + 2 i + h
    for k in 32 h .. 32 h + 31, in the high word for h = 1)."""
    if int(kind) != E2M1:
        return block(c0, c1, tile, base + i, dom_of(kind)).astype(np.uint64)
    lo, hi = (block(c0, c1, tile, base + 2 * i + h, DOM4).astype(np.uint64) for h in (0, 1))
    return lo | (hi << np.uint64(32))


def u_of(t: np.ndarray) -> np.ndarray:
    """bU's B entry from its digest byte t: (t & 0x7F) - 64 + ((t >> 6) & 1), uniform on {±1, ..., ±64}."""
    t = np.asarray(t, np.int64)
    return (t & 0x7F) - 64 + ((t >> 6) & 1)


def bu_columns(c0: int, c1: int, tile: int) -> np.ndarray:
    """bU's B, 32 x TN: block b (ctr 2^56 + b) holds columns 2b and 2b + 1, entry (k, 2b + i) from digest byte 32 i + k."""
    t = np.stack([np.frombuffer(digest(c0, c1, tile, CTR_B + b, DOM_BU), np.uint8).reshape(2, 32)
                  for b in range(TN // 2)]).reshape(TN, 32)
    return u_of(t).T


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


def words(x: np.ndarray, e2m1: bool) -> np.ndarray:
    """The committed words of a ^ b: s32 32 - 2 popcount (s8), or the bits of the f32 64 - 2 popcount (e2m1)."""
    if e2m1:
        return (64 - 2 * popcount(x)).astype(np.float32).view(np.uint32)
    return ((32 - 2 * popcount(x)) & M32).astype(np.uint32)


def bu_products(ar: np.ndarray, u: np.ndarray) -> np.ndarray:
    """bU's committed words, every warp's rows against every column: C = A u over k < 32 with A[w, r, k] = bit k of ar[w,
    r], as s32 bits ([W, 16 S, TN] uint32; |C| <= 2048)."""
    a = ((ar[..., None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).astype(np.int64)
    return ((a @ u) & M32).astype(np.uint32)


def receipt(c0: int, c1: int, tile: int, nc: int, steps: int, kind: int = S8) -> int:
    """The tile's receipt: warp w's fragments w S .. w S + S - 1 (16 rows each) against all TN columns, fragment f's
    outputs absorbed by chain f % 4 of each lane in order of f, then of the B fragment j."""
    kind = int(kind)
    e2m1 = kind == E2M1
    kb = block(c0, c1, tile, CTR_KEY, dom_of(kind))
    a0, a1, k0, k1 = (np.uint32(v) for v in keys(kb))
    lane = np.arange(32)
    g, t = lane >> 2, lane & 3
    pos = (np.arange(W)[:, None, None] * 32 + lane[None, :, None]) * 4 + np.arange(4)[None, None, :]
    y = pos.astype(np.uint32)
    z = (pos.astype(np.uint64) * 0x9E3779B9 & M32).astype(np.uint32)
    ar = np.stack([np.concatenate([rows(c0, c1, tile, 0, w * steps + f, kind) for f in range(steps)]
                                  or [np.zeros(0, np.uint64)]) for w in range(W)])  # ar[w, 16 f + i]: row i of fragment f
    jj = np.arange(TN // 8)
    n0 = 8 * jj[:, None] + 2 * t[None, :]  # [j, lane]: the lane's columns 8 j + 2 t and 8 j + 2 t + 1
    if kind == BU:
        cw = bu_products(ar, bu_columns(c0, c1, tile))
        ww = np.arange(W)[None, :, None, None]
    else:
        bb = np.concatenate([rows(c0, c1, tile, CTR_B, b, kind) for b in range(TN // 16)])  # bit k of bb[n]: entry (k, n)
        b0, b1 = bb[n0], bb[n0 + 1]
    for fi in range((steps + 3) // 4):
        f = 4 * fi + np.arange(4)
        live = f < steps
        fc = np.minimum(f, max(steps - 1, 0))
        i0 = 16 * fc[None, :] + g[:, None]  # [lane, c]: the lane's rows 16 f + g and 16 f + g + 8
        if kind == BU:
            d0, d1, d2, d3 = (cw[ww, (i0 + h)[None, None], (n0 + n)[:, None, :, None]] for h in (0, 8) for n in (0, 1))
        else:
            r0, r1 = ar[:, i0], ar[:, i0 + 8]  # [w, lane, c]
            d0, d1, d2, d3 = (words(r[None] ^ b[:, None, :, None], e2m1) for r in (r0, r1) for b in (b0, b1))  # [j, w, lane, c]
        for j in range(TN // 8):
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
