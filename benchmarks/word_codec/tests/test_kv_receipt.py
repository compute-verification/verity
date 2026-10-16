"""The receipts inside attention (`kv_receipt.py`) against a lane-by-lane emulation of `kv.cu`'s receipt forms: the
MMA warps fill each tile's B slot from `kv_receipt.rows` (XORed with the z RLAG tiles back for the bound form), and each
decoder lane loads its B registers from the slot and runs `mma.m16n8k32.s8` on its A, B and D fragments, one step for
each 16 bytes of its word (4 at p512, 8 at p1024; the bound form only at 64-byte words)."""

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
RT, LAG, ZW = 2, 2, 1024
MDW = {64: 8, 128: 4}  # decoder warps a CTA: 32 tokens a tile, a token's head share 512 bytes
CASES = [(1, 64), (2, 64), (1, 128)]


def slot(rows: np.ndarray, rc: int, tile: int, z: np.ndarray | None) -> np.ndarray:
    """Tile `tile`'s B slot, ZW words, as the MMA warps fill it."""
    b = np.tile(rows[tile % RT], ZW // rows.shape[1])
    if rc == 2 and tile >= LAG:
        b = b ^ z[tile - LAG]
    return b


def emulate(lanes: np.ndarray, rows: np.ndarray, rc: int, z: np.ndarray | None) -> np.ndarray:
    """(blocks, MDW, 16, 16): each warp's acc[2][4] after each block, as kv.cu stores it."""
    tiles, mdw, _, wb = lanes.shape
    out = np.zeros((tiles // RT, mdw, 16, 16), np.int64)
    for i in range(mdw):
        acc = np.zeros((2, 16, 8), np.int64)
        for tile in range(tiles):
            x = lanes[tile, i].view("<u4")  # 32 lanes × wb / 4 registers
            bs = slot(rows, rc, tile, z)
            base = 512 * i % ZW if rc == 2 else 0
            for s in range(wb // 16):
                a = np.zeros((16, 32), np.int64)
                bm = np.zeros((2, 32, 8), np.int64)
                for lane in range(32):
                    g, t4 = lane // 4, lane % 4
                    regs = [int(v) for v in x[lane, 4 * s: 4 * s + 4]]
                    for q, (row, col) in enumerate(((g, 4 * t4), (g + 8, 4 * t4), (g, 16 + 4 * t4),
                                                    (g + 8, 16 + 4 * t4))):
                        a[row, col: col + 4] = np.frombuffer(regs[q].to_bytes(4, "little"), np.int8)
                    for q in range(4):
                        b = int(bs[base + 128 * s + 4 * lane + q])
                        blk, h = q // 2, q % 2
                        bm[blk, 16 * h + 4 * t4: 16 * h + 4 * t4 + 4, g] = np.frombuffer(b.to_bytes(4, "little"),
                                                                                       np.int8)
                for blk in range(2):
                    acc[blk] += a @ bm[blk]
            if tile % RT == RT - 1:
                out[tile // RT, i] = np.concatenate([acc[0], acc[1]], axis=1)
                acc[:] = 0
    return out


@pytest.mark.parametrize("rc,wb", CASES)
def test_sketch_matches_lane_emulation(rc, wb):
    rng = np.random.default_rng(rc + wb)
    tiles = 2 * RT
    lanes = rng.integers(0, 256, (tiles, MDW[wb], 32, wb), dtype=np.uint8)
    bits = R.sign_bits(SEED, rc, 2 * wb * RT)
    z = rng.integers(0, 1 << 32, (tiles, ZW), dtype=np.uint64).astype(np.uint32) if rc == 2 else None
    want = emulate(lanes, R.rows(bits, RT, rc), rc, z)
    assert np.array_equal(R.sketch(lanes, bits, RT, z, LAG, ZW), want)


@pytest.mark.parametrize("rc,wb", CASES)
def test_rows_are_the_sign_bytes(rc, wb):
    steps = wb // 16
    bits = R.sign_bits(SEED, rc, 2 * wb * RT)
    by = R.rows(bits, RT, rc).astype("<u4").view(np.uint8).reshape(RT, steps, 32, 4, 4)  # τ, s, l, q, b
    for tau, s, lane, q, b in [(0, 0, 0, 0, 0), (1, steps - 1, 31, 3, 3), (0, 2, 13, 1, 2), (1, 1, 6, 2, 1)]:
        g, tq, blk, h = lane // 4, lane % 4, q // 2, q % 2
        neg = bits[8 * blk + g, 32 * steps * tau + 32 * s + 16 * h + 4 * tq + b]
        assert by[tau, s, lane, q, b] == (0xFF if neg else (0x01 if rc == 1 else 0x00))


def test_bound_takes_64_byte_words():
    with pytest.raises(ValueError):
        R.rows(R.sign_bits(SEED, 2, 256 * RT), RT, 2)


@pytest.mark.parametrize("wb", [64, 128])
def test_unbound_is_signs_times_vectors(wb):
    rng = np.random.default_rng(7)
    lanes = rng.integers(0, 256, (RT, MDW[wb], 32, wb), dtype=np.uint8)
    bits = R.sign_bits(SEED, 1, 2 * wb * RT)
    v = R.vectors(lanes, RT).astype(np.int64)
    s = 1 - 2 * bits.astype(np.int64)
    assert np.array_equal(R.sketch(lanes, bits, RT), v @ s.T)
    assert np.array_equal(np.sort(R.vectors(lanes, RT).view(np.uint8), axis=None), np.sort(lanes, axis=None))


def test_sha_receipt_is_seed_first_in_one_block():
    p = R.sha_prefix(SEED)
    assert len(p) == 128 and p.startswith(R.SHA_DOMAIN + b"b" + (32).to_bytes(8, "big") + SEED)
    sk = np.arange(128, dtype=np.int64).reshape(8, 16) - 64
    msg = p + (9).to_bytes(8, "big") + sk.astype("<i4").tobytes()
    assert len(msg) == 648 and R.sha_receipt(SEED, 9, sk) == hashlib.sha512(msg).digest()


def test_receipt_is_the_tags_construction():
    sk = np.arange(16, dtype=np.int64) - 8
    from verity.core.protocols.bandwidth import tags
    want = tags.Blake3Tag(name=R.NAMES[2]).keyed(SEED)(5, sk.astype("<i4").tobytes())
    assert R.receipt(SEED, 2, 5, sk) == want
    assert R.key_words(SEED, 2).tobytes() == tags.Blake3Tag(name=R.NAMES[2]).key(SEED)
