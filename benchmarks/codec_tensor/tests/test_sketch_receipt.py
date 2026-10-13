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


def test_gemm_blocks_refuse_other_widths():
    store = np.zeros(2 * SB, dtype=np.uint8)
    with pytest.raises(ValueError, match="4096-byte block"):
        SR.gemm_blocks(store, 2, 1, SB, 1, 2)
