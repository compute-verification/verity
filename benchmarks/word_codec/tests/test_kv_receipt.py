"""The receipts inside attention (`kv_receipt.py`) against a lane-by-lane emulation of `kv.cu`'s `rc_sketch`: each
lane's B registers expanded from the packed sign table by `__byte_perm` (XORed with R for the bound form), and
`mma.m16n8k32.s8` from its A, B and D fragments."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import kv_receipt as R  # noqa: E402

SEED = hashlib.sha256(b"verity/word-codec/kv-receipt/test").digest()
RT, MDW, LAG, ZW = 2, 8, 2, 1024
CONSTS = {1: ((0xFF01FF01, 0xFF01FF01), (0xFFFF0101, 0xFFFF0101), (0x01010101, 0xFFFFFFFF)),
          2: ((0xFF00FF00, 0xFF00FF00), (0xFFFF0000, 0xFFFF0000), (0x00000000, 0xFFFFFFFF))}


def byte_perm(lo: int, hi: int, sel: int) -> int:
    src = (hi << 32 | lo).to_bytes(8, "little")
    return int.from_bytes(bytes(src[(sel >> 4 * b) & 7] for b in range(4)), "little")


def emulate(lanes: np.ndarray, table: np.ndarray, rc: int, z: np.ndarray | None) -> np.ndarray:
    """(blocks, MDW, 16, 16): each warp's acc[2][4] after each block, as kv.cu stores it."""
    tiles = lanes.shape[0]
    out = np.zeros((tiles // RT, MDW, 16, 16), np.int64)
    for i in range(MDW):
        acc = np.zeros((2, 16, 8), np.int64)
        for tile in range(tiles):
            x = lanes[tile, i].view("<u4")  # 32 lanes × 16 registers
            zr = z[tile - LAG] if z is not None and tile >= LAG else None
            for s in range(4):
                a = np.zeros((16, 32), np.int64)
                bm = np.zeros((2, 32, 8), np.int64)
                for lane in range(32):
                    g, t4 = lane // 4, lane % 4
                    regs = [int(v) for v in x[lane, 4 * s: 4 * s + 4]]
                    for q, (row, col) in enumerate(((g, 4 * t4), (g + 8, 4 * t4), (g, 16 + 4 * t4),
                                                    (g + 8, 16 + 4 * t4))):
                        a[row, col: col + 4] = np.frombuffer(regs[q].to_bytes(4, "little"), np.int8)
                    w = [int(v) for v in table[tile % RT, lane]]
                    sel = [w[0], w[0] >> 16, w[1], w[1] >> 16, w[2], w[2] >> 16]
                    for q in range(4):
                        r = 4 * s + q
                        lo, hi = CONSTS[rc][r % 3]
                        b = byte_perm(lo, hi, sel[2 * (r // 6) + (r % 6) // 3])
                        if zr is not None:
                            b ^= int(zr[(512 * i + 128 * s + 4 * lane + q) % ZW])
                        blk, h = q // 2, q % 2
                        bm[blk, 16 * h + 4 * t4: 16 * h + 4 * t4 + 4, g] = np.frombuffer(b.to_bytes(4, "little"),
                                                                                       np.int8)
                for blk in range(2):
                    acc[blk] += a @ bm[blk]
            if tile % RT == RT - 1:
                out[tile // RT, i] = np.concatenate([acc[0], acc[1]], axis=1)
                acc[:] = 0
    return out


@pytest.mark.parametrize("rc", [1, 2])
def test_sketch_matches_lane_emulation(rc):
    rng = np.random.default_rng(rc)
    tiles = 2 * RT
    lanes = rng.integers(0, 256, (tiles, MDW, 32, 64), dtype=np.uint8)
    bits = R.sign_bits(SEED, rc, 128 * RT)
    z = rng.integers(0, 1 << 32, (tiles, ZW), dtype=np.uint64).astype(np.uint32) if rc == 2 else None
    want = emulate(lanes, R.pack(bits, RT), rc, z)
    assert np.array_equal(R.sketch(lanes, bits, RT, z, LAG, ZW), want)


def test_unbound_is_signs_times_vectors():
    rng = np.random.default_rng(7)
    lanes = rng.integers(0, 256, (RT, MDW, 32, 64), dtype=np.uint8)
    bits = R.sign_bits(SEED, 1, 128 * RT)
    v = R.vectors(lanes, RT).astype(np.int64)
    s = 1 - 2 * bits.astype(np.int64)
    assert np.array_equal(R.sketch(lanes, bits, RT), v @ s.T)
    assert np.array_equal(np.sort(R.vectors(lanes, RT).view(np.uint8), axis=None), np.sort(lanes, axis=None))


def test_receipt_is_the_tags_construction():
    sk = np.arange(16, dtype=np.int64) - 8
    from verity.core.protocols.bandwidth import tags
    want = tags.Blake3Tag(name=R.NAMES[2]).keyed(SEED)(5, sk.astype("<i4").tobytes())
    assert R.receipt(SEED, 2, 5, sk) == want
    assert R.key_words(SEED, 2).tobytes() == tags.Blake3Tag(name=R.NAMES[2]).key(SEED)
