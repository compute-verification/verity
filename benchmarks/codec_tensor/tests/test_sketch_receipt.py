"""`sketch_receipt`'s blocks of a weight GEMM's store: a block is the GEMM's sketch vector, so the sketch warps' MMA
arithmetic over the stored bytes (`sketching.fragment_sketch`, under sketch-sha512's signs) is σ of each block, and a
launch's receipts from those sums are the driver-shaped `SketchReceipter`'s of its blocks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sketch_receipt as SR  # noqa: E402
import sketching as SK  # noqa: E402

SEED = SR.rb.seed(bytes(range(32)), 3, bytes(64))
SB, T = 8192, 8


def test_gemm_blocks_are_the_sketch_warps_vectors():
    nbs, nchunks, groups = 3, 1, 2
    store = np.random.default_rng(4).integers(0, 256, nbs * nchunks * SB, dtype=np.uint8)
    blocks = SR.gemm_blocks(store, nbs, nchunks, SB, groups, T)
    assert blocks.shape == (groups * nchunks * SK.tiles_per_cta(nbs, groups, T) * 16, SR.BLOCK)
    bits = np.unpackbits(SR.sign_bits(SEED), axis=1, bitorder="little")
    mma = SK.fragment_sketch(store, nbs, nchunks, SB, groups, T, SK.sign_fragments(bits, SK.steps(SB), T), 16)
    signs = SR.signs(SEED)
    assert np.array_equal(SR.sketches(signs, blocks), mma)
    j0 = 11
    receipter = SR.SketchReceipter(SEED)
    want = [receipter(j0 + v, blocks[v].tobytes(), b"\0") for v in range(len(blocks))]
    assert SR.receipts_of(SEED, mma, j0) == want


def test_reader_sketch64_fragments_are_the_64_row_sketch():
    """reader.cu's sketch64 arithmetic: each lane's A registers its 16 bytes of blocks g and g + 8 at 64 p + 16 t, its
    B registers one IMAD of a `pack64` word, put together across the warp as mma.m16n8k32's A (16 × 32) and B (32 × 8)
    at each k-step, sum to σ at 64 rows, the key binding k_s."""
    import reader as RD

    blocks = np.random.default_rng(64).integers(0, 256, (16, SR.BLOCK), dtype=np.uint8)
    blocks[0], blocks[1] = 0x80, 0x7F
    words = RD.pack64(SR.sign_bits(SEED, RD.ROWS64))
    e = blocks.view(np.int8).astype(np.int64)

    def reg(word: int, r: int) -> np.ndarray:
        x = (0x01010101 + 0xFE * ((word >> r) & 0x01010101)) & 0xFFFFFFFF
        return np.frombuffer(x.to_bytes(4, "little"), np.int8).astype(np.int64)

    d = np.zeros((8, 16, 8), np.int64)
    for p in range(64):
        for s in range(2):
            a, b = np.zeros((16, 32), np.int64), np.zeros((8, 32, 8), np.int64)
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                lo, hi = (e[r, 64 * p + 16 * t + 8 * s:][:8] for r in (g, g + 8))
                a[g, 4 * t:4 * t + 4], a[g + 8, 4 * t:4 * t + 4] = lo[:4], hi[:4]
                a[g, 16 + 4 * t:20 + 4 * t], a[g + 8, 16 + 4 * t:20 + 4 * t] = lo[4:], hi[4:]
                for nt in range(8):
                    word, r = int(words[p, lane, nt >> 1]), 4 * (nt & 1) + 2 * s
                    b[nt, 4 * t:4 * t + 4, g], b[nt, 16 + 4 * t:20 + 4 * t, g] = reg(word, r), reg(word, r + 1)
            d += a @ b
    sigma = d.transpose(1, 0, 2).reshape(16, RD.ROWS64)
    assert np.array_equal(sigma, SR.sketches(SR.signs(SEED, RD.ROWS64), blocks))
    assert not np.array_equal(SR.signs(SEED, RD.ROWS64)[:SR.ROWS], SR.signs(SEED))


def test_gemm_blocks_refuse_other_widths():
    store = np.zeros(2 * SB, dtype=np.uint8)
    with pytest.raises(ValueError, match="4096-byte block"):
        SR.gemm_blocks(store, 2, 1, SB, 1, 2)
