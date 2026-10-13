"""Red team ra-remote-redteam-impl against the device side of the remote bandwidth audit: the fused weight GEMM's
receipt blocks (`sketch_receipt.gemm_blocks`, wt.cu's sketch warps) and attention's KV receipts (`kv_receipt`), each
against what the driver (`remote_bandwidth.receipts`, `audit.Session`) credits and opens.  Each test asserts the
property the driver's ledger needs and is marked xfail with its finding id while the device breaks it (the report is
note:memory-accounting/ra-remote-redteam-impl)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "word_codec"))

import kv_receipt as KR  # noqa: E402
import sketch_receipt as SR  # noqa: E402
import sketching as SK  # noqa: E402

SEED = SR.rb.seed(bytes(range(32)), 3, bytes(64))
SB, T = 8192, 8          # p1024's (and p512's) stored bytes a task, and the sketch's tasks a tile
SMS = 188                # an RTX PRO 6000's SMs: launch_gemm's default groups is SMS · per_sm // chunks
WORD = 128               # a p1024 codec word


def vectors_of_store_bytes(nbs: int, nchunks: int, groups: int) -> np.ndarray:
    """For each byte of each GEMM block (`gemm_blocks`), the store byte it holds, or −1 where it is padding."""
    idx = np.arange(nbs * nchunks * SB, dtype=np.int64) + 1
    out = np.zeros((SK.tiles_per_cta(nbs, groups, T) * groups * nchunks * 16, SR.BLOCK), np.int64)
    for b in range(4):
        out += SR.gemm_blocks(((idx >> (8 * b)) & 255).astype(np.uint8), nbs, nchunks, SB, groups, T).astype(np.int64) << (8 * b)
    return out - 1


@pytest.mark.xfail(strict=True, reason="RB-D1: a launch's GEMM blocks outnumber the stored bytes' blocks: zero padding "
                                       "where a CTA's last tile runs past its tasks is credited 4,096 bytes a block")
@pytest.mark.parametrize("n, kd, groups", [
    (4096, 4096, SMS // 8),          # 8B o_proj at launch_gemm's default groups, one CTA an SM: 1.44×
    (6144, 4096, SMS // 8),          # 8B qkv at the default: 1.44×
    (4096, 4096, 4096 // 16),        # groups = nbs, the most launch_gemm takes: 8×
])
def test_rbd1_a_launch_credits_no_more_blocks_than_it_stores(n, kd, groups) -> None:
    """`step.py` makes the pass's driver plan `Plan.whole(nvec + nkv)`, nvec = `wt_sketch_vectors(N, Kd, groups, T)`:
    every launch vector is a store block, credited 4,096 bytes by `ledger`.  groups is the launch's (`launch_gemm`, any
    value up to nbs), and a CTA's last tile of T tasks is zero-filled past its tasks, so the vectors outnumber the
    stored bytes / 4,096 by up to T."""
    nbs, nchunks = n // 16, kd // 512
    nvec = groups * nchunks * SK.tiles_per_cta(nbs, groups, T) * 16
    stored = nbs * nchunks * SB
    assert nvec * SR.BLOCK <= stored, f"{nvec} blocks credited for {stored // SR.BLOCK} stored: {nvec * SR.BLOCK / stored:.3f}×"


@pytest.mark.xfail(strict=True, reason="RB-D1: a GEMM block can be all padding, and its receipt is then computable "
                                       "with no read at all")
def test_rbd1_every_gemm_block_holds_stored_bytes() -> None:
    """17 row blocks over 2 groups: CTA 1 has 8 tasks, so its second tile is all zeros, and its 16 vectors' sketches
    are 0 whatever the store holds.  Their receipts are SHA512(prefix(seed) ‖ j ‖ 0^64), known from the seed alone."""
    nbs, nchunks, groups = 17, 1, 2
    rng = np.random.default_rng(1)
    stores = [rng.integers(0, 256, nbs * nchunks * SB, dtype=np.uint8) for _ in range(2)]
    signs = SR.signs(SEED)
    ys = [SR.receipts_of(SEED, SR.sketches(signs, SR.gemm_blocks(s, nbs, nchunks, SB, groups, T)), 0) for s in stores]
    free = [v for v in range(len(ys[0])) if ys[0][v] == ys[1][v] == SR.receipt_of(hashlib.sha512(SR.prefix(SEED)), v,
                                                                                   bytes(64))]
    assert len(free) == 16
    assert not free, f"blocks {free} have receipts that read no stored byte"


@pytest.mark.xfail(strict=True, reason="RB-D1: a GEMM block is not 32 whole codec words, so the driver's opening "
                                       "check (decode the block's words, match the payload leaf) can't open it")
def test_rbd1_each_codec_word_lies_in_one_gemm_block() -> None:
    """The driver opens block j by decoding its 32 words of 128 bytes at indices 32j + w (`codec.decode`).  A GEMM block
    is row r of a tile's A fragments, 32 scattered bytes of each 512-byte step: every 128-byte stored word lands in 4
    blocks, so no block is a run of whole words, and an honest fused receipt can't pass `Session.verify`'s `codec` and
    `payload-path` checks."""
    m = vectors_of_store_bytes(17, 1, 2)
    owners: dict[int, set[int]] = {}
    for v, row in enumerate(m):
        for x in row[row >= 0]:
            owners.setdefault(int(x) // WORD, set()).add(v)
    spread = {len(s) for s in owners.values()}
    assert spread == {4}
    assert spread == {1}


@pytest.mark.xfail(strict=True, reason="RB-D1: block j's bytes are the launch's cut, which neither the registration nor "
                                       "the plan binds")
def test_rbd1_a_block_is_a_function_of_the_store_and_its_index() -> None:
    """The cut follows `groups`, "which the prover states with the window": the same store's block 0 holds other bytes
    under another launch, so the verifier maps j to stored bytes on the prover's word."""
    a, b = vectors_of_store_bytes(32, 1, 1), vectors_of_store_bytes(32, 1, 2)
    assert np.array_equal(a[:16], b[:16])


@pytest.mark.xfail(strict=True, reason="RB-D2: attention's KV receipts are a third kind: other signs, another hash "
                                       "domain, and 4,096 bytes from 16 tokens' head shares under a half-block index")
def test_rbd2_a_kv_receipt_is_a_driver_receipt_of_one_registered_block() -> None:
    """The step commits its KV receipts in the pass's one tree after the weights' (`kv_rc_sha8_at`).  A half-block's
    receipt is SHA512(Z(kv-receipt domain ‖ frame(σ)) ‖ u64be half ‖ 8 sketches) under ro-sketch-blake3's signs over
    n = 32·S·RT bytes; its 8 vectors are half the bytes of 16 tokens.  The driver's block is one token-layer and its
    receipt `SketchReceipter`'s, so `Session.verify` rejects every honest KV opening."""
    rt, mdw, wb = 2, 4, 128                                          # p1024: 32 tokens a tile, 8 a decoder warp
    tiles = rt
    tile, warp, lane = np.meshgrid(np.arange(tiles), np.arange(mdw), np.arange(32), indexing="ij")
    token = (tile * 32 + 8 * warp + lane // 4)[..., None].repeat(wb, axis=3)
    tok = np.zeros(KR.vectors(np.zeros((tiles, mdw, 32, wb), np.uint8), rt).shape, np.int64)
    for b in range(2):
        tok += KR.vectors(((token >> (8 * b)) & 255).astype(np.uint8), rt).view(np.uint8).astype(np.int64) << (8 * b)
    half = tok[0, 0, :8]                                             # warp 0's first half-block: 8 vectors
    assert half.size == SR.BLOCK and len(np.unique(half)) == 16

    lanes = np.random.default_rng(2).integers(0, 256, (tiles, mdw, 32, wb), dtype=np.uint8)
    n = 32 * (wb // 16) * rt
    sk = KR.sketch(lanes, KR.sign_bits(SEED, 1, n), rt)[0, 0, :8]
    kv = KR.sha_receipt(SEED, 0, sk)
    block = KR.vectors(lanes, rt)[0, 0, :8].tobytes()
    assert kv == SR.SketchReceipter(SEED)(0, block, b"\0")
