"""The pinned H1 tile (`pouw.tile`, `PouwH1Tile_v1{GemmAccumulatorS8_v2{K=64}, R=4, C=8}`) on many tiles at once:
`h1tile.reference`'s (z, receipt) for T tiles, word for word, its noise hashed with hashlib and everything else in numpy
over the tile axis.

At this tile the unit is one lane (G = Q = 1: rows and columns in slot order), two segments of 32 (KS = 2) and two digits
(m = 2), so the receipt folds four sets, T_0, T_1, O' and D_1, and is 8 words. Its noise is seed_j's blocks 0..3 (the rows'
dither bits, then each row's 32 digit coins) and 4..11 (the columns' coin bits, then U's 32 bytes), and its fold keys are
σ_R's block 0: 15 SHA-512 calls a tile, 17 compressions (`h1tile.unit_compressions(64, (4, 8))`)."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from verity.core.protocols.pouw.circuit import h1tile as Ht, seeds as SD, segment as SG

R, C, K = 4, 8, 64
LAY = Ht.Layout(R, C, K)
if (LAY.ks, LAY.m, LAY.lanes, LAY.rb, LAY.cb) != (2, 2, 1, 1, 1):
    raise ImportError(f"h1batch is written for the pinned tile's layout, not {LAY}")
RECEIPT_WORDS = 2 * LAY.sets
_Q = np.arange(32)
_ROW = np.array([SG.row_slot(q) for q in range(32)])
_COL = np.array([SG.col_slot(q) for q in range(32)])
_FX1 = np.array([SG.fx1_key(q) for q in range(32)], np.uint32)
_KK, _KA = np.array(SG.KK, np.uint32), np.array(SG.KA, np.uint32)
_BITS = np.arange(32, dtype=np.uint32)


def _halves(digests: np.ndarray) -> np.ndarray:
    """(..., 64) bytes -> (..., 16) big-endian u32 halves."""
    return digests.reshape(*digests.shape[:-1], 16, 4).astype(np.uint32) @ np.array([1 << 24, 1 << 16, 1 << 8, 1], np.uint32)


def _bits(halves: np.ndarray) -> np.ndarray:
    """(..., 16) halves -> (..., 512) bits, each half least significant bit first (`seeds.leaves(d, True)`)."""
    return ((halves[..., :, None] >> _BITS) & 1).reshape(*halves.shape[:-1], 512).astype(np.uint8)


def noise(salt: bytes, positions: Sequence[int], roots_a: np.ndarray, roots_b: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each tile's noise and fold keys (`h1tile.noise`, `fold_keys`): dither bits (T, 4, 64), digit coins (T, 4, 32), coin
    bits (T, 8, 64), U's int8 encodings (T, 8, 32) and the lane's 16 keys (T, 16), for tiles at ``positions`` whose
    operands commit to ``roots_a[t]`` and ``roots_b[t]`` (64 bytes each) under ``salt``."""
    n = len(positions)
    blocks = np.empty((n, 13, 64), np.uint8)
    nt = SD.NOISE_TAG
    for t, pos in enumerate(positions):
        ra, rb = bytes(roots_a[t]), bytes(roots_b[t])
        sj = SD.seed(salt, int(pos), ra, rb)
        for b in range(12):
            blocks[t, b] = np.frombuffer(hashlib.sha512(nt + sj + b.to_bytes(8, "big")).digest(), np.uint8)
        blocks[t, 12] = np.frombuffer(hashlib.sha512(nt + SD.receipt_seed(salt, int(pos), ra, rb) + bytes(8)).digest(),
                                      np.uint8)
    h = _halves(blocks)
    bits = _bits(h[:, :12])
    rows, cols = bits[:, :R], bits[:, R:]
    u = cols[:, :, K:K + 256].reshape(n, C, 32, 8).astype(np.uint32) @ (1 << np.arange(8, dtype=np.uint32))
    u129 = ((u % 129).astype(np.int64) - 64) & 0xFF
    return rows[:, :, :K], rows[:, :, K:K + 32], cols[:, :, :K], u129.astype(np.uint8), h[:, 12]


def _sx(v: np.ndarray) -> np.ndarray:
    return v.astype(np.int8).astype(np.int64)


def _absorb(y: np.ndarray, z: np.ndarray, p: np.ndarray) -> None:
    """fx1's absorb (`segment._fx1_absorb`) of a set p (T, 32) into the lane's chains y, z (T, 8), in place."""
    p = p * _FX1
    for t in range(8):
        h = 4 * (t % 2)
        for e in range(4):
            u = y[:, h + e] ^ p[:, 4 * t + e] ^ _KK[t]
            y[:, h + e] = u * _KA[t] + z[:, h + e]
            r = int(SG.ROT[t])
            z[:, h + e] = ((u << np.uint32(r)) | (u >> np.uint32(32 - r))) + y[:, h + e]


def tiles(a: np.ndarray, b: np.ndarray, salt: bytes, positions: Sequence[int], roots_a: np.ndarray, roots_b: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray]:
    """`h1tile.reference` of T tiles: ``a`` (T, 4, 64) and ``b`` (T, 8, 64) int8 codes (any integer dtype, read as their
    low byte), tile t at ``positions[t]`` with roots ``roots_a[t]``, ``roots_b[t]`` -> z (T, 32) u32 (row-major cells,
    the exact A Bᵀ) and the receipt (T, 8) u32."""
    a8, b8 = (np.asarray(x).astype(np.int64) & 0xFF for x in (a, b))
    n = len(positions)
    if a8.shape != (n, R, K) or b8.shape != (n, C, K):
        raise ValueError(f"{n} tiles take a (T, {R}, {K}) and b (T, {C}, {K}), not {a8.shape} and {b8.shape}")
    with np.errstate(over="ignore"):
        dither, coins, cbits, u, keys = noise(salt, positions, np.asarray(roots_a), np.asarray(roots_b))
        x = (a8 + dither) & 0xFF
        wp = (b8 + cbits + np.tile(u.astype(np.int64), 2)) & 0xFF
        sx, sw = _sx(x), _sx(wp)
        sets = []
        acc = np.zeros((n, 32), np.int64)
        for s in range(LAY.ks):
            seg = slice(32 * s, 32 * s + 32)
            acc = acc + np.einsum("tqk,tqk->tq", sx[:, _ROW, seg], sw[:, _COL, seg])
            sets.append(acc & 0xFFFFFFFF)
        y = sx[:, :, :32] + sx[:, :, 32:]
        q = y >> 7
        zd = np.where(q == 127, 0, coins & 1)
        a0 = (y & 127) | (zd << 7)
        a1 = (q + zd) & 0xFF
        su, sneg = _sx(u), _sx(-u.astype(np.int64) & 0xFF)
        op = sets[-1] + np.einsum("tqk,tqk->tq", sneg[:, _COL], _sx(a0)[:, _ROW])
        d1 = np.einsum("tqk,tqk->tq", su[:, _COL], _sx(a1)[:, _ROW])
        sets += [op & 0xFFFFFFFF, d1 & 0xFFFFFFFF]
        st_y, st_z = keys[:, :8].copy(), keys[:, 8:].copy()
        receipt = np.empty((n, RECEIPT_WORDS), np.uint32)
        for i, p in enumerate(sets):
            _absorb(st_y, st_z, p.astype(np.uint32))
            receipt[:, 2 * i] = np.bitwise_xor.reduce(st_y, axis=1)
            receipt[:, 2 * i + 1] = np.bitwise_xor.reduce(st_z, axis=1)
    z = (np.einsum("trk,tck->trc", _sx(a8), _sx(b8)).reshape(n, R * C) & 0xFFFFFFFF).astype(np.uint32)
    return z, receipt
