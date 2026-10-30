"""The red team's attacks on the device side of the remote bandwidth audit: the fused weight GEMM's
receipt blocks (`sketch_receipt.gemm_vectors`, wt.cu's sketch warps) and attention's KV receipts (`kv_vectors`,
`kv_fragment_sketch`: kv.cu's form 13), each against what the driver (`remote_bandwidth.receipts`, `audit.Session`)
credits and opens.  Each test asserts the property the driver's ledger needs and was marked xfail with its finding id
while the device broke it; RB-D1's pass since
store-order/v0, RB-D2's since form 13."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import sketch_receipt as SR  # noqa: E402

SEED = SR.rb.seed(bytes(range(32)), 3, bytes(64))
SB = 8192                # p1024's (and p512's) stored bytes a task
SMS = 188                # an RTX PRO 6000's SMs: launch_gemm's default groups is SMS · per_sm // chunks
WORD = 128               # a p1024 codec word


def vectors_of_store_bytes(nbs: int, nchunks: int, groups: int) -> np.ndarray:
    """For each byte of each GEMM block (the launch's receipts, `gemm_vectors`), the store byte it holds, or −1 where it
    is padding."""
    return SR.gemm_vectors(nbs, nchunks, SB, groups)


@pytest.mark.parametrize("n, kd, groups", [
    (4096, 4096, SMS // 8),          # 8B o_proj at launch_gemm's default groups, one CTA an SM (was 1.44×)
    (6144, 4096, SMS // 8),          # 8B qkv at the default (was 1.44×)
    (4096, 4096, 4096 // 16),        # groups = nbs, the most launch_gemm takes (was 8×)
])
def test_rbd1_a_launch_credits_no_more_blocks_than_it_stores(n, kd, groups) -> None:
    """`step.py` makes the pass's driver plan `Plan.whole(nvec + nkv)`, nvec = `wt_sketch_vectors(N, Kd, groups, T)`
    (`gemm_receipts`' count, which bench.py's gate holds the build to): every launch vector is a store block, credited
    4,096 bytes by `ledger`.  groups is the launch's (`launch_gemm`, any value up to nbs)."""
    nbs, nchunks = n // 16, kd // 512
    nvec = len(SR.gemm_receipts(nbs, nchunks, SB, groups))
    stored = nbs * nchunks * SB
    assert nvec * SR.BLOCK <= stored, f"{nvec} blocks credited for {stored // SR.BLOCK} stored: {nvec * SR.BLOCK / stored:.3f}×"


def test_rbd1_every_gemm_block_holds_stored_bytes() -> None:
    """17 row blocks over 2 groups: CTA 1 has 8 tasks, so its second tile was all zeros, and its 16 vectors' sketches
    0 whatever the store held.  Their receipts were SHA512(prefix(seed) ‖ j ‖ 0^64), known from the seed alone."""
    nbs, nchunks, groups = 17, 1, 2
    rng = np.random.default_rng(1)
    stores = [rng.integers(0, 256, nbs * nchunks * SB, dtype=np.uint8) for _ in range(2)]
    m = vectors_of_store_bytes(nbs, nchunks, groups)
    ys = [SR.receipts_of(SEED, SR.sketches(SEED, np.where(m >= 0, s[np.maximum(m, 0)], 0).astype(np.uint8)), 0)
          for s in stores]
    receipter = SR.rb.SketchReceipter(SEED)
    free = [v for v in range(len(ys[0])) if ys[0][v] == ys[1][v] == receipter.receipt(v, bytes(64))]
    assert not free, f"blocks {free} have receipts that read no stored byte"


def test_rbd1_each_codec_word_lies_in_one_gemm_block() -> None:
    """The driver opens block j by decoding its 32 words of 128 bytes at indices 32j + w (`codec.decode`).  A GEMM block
    was row r of a tile's A fragments, 32 scattered bytes of each 512-byte step: every 128-byte stored word landed in 4
    blocks, so no block was a run of whole words, and an honest fused receipt couldn't pass `Session.verify`'s `codec`
    and `payload-path` checks."""
    m = vectors_of_store_bytes(17, 1, 2)
    owners: dict[int, set[int]] = {}
    for v, row in enumerate(m):
        for x in row[row >= 0]:
            owners.setdefault(int(x) // WORD, set()).add(v)
    spread = {len(s) for s in owners.values()}
    assert spread == {1}


def test_rbd1_a_block_is_a_function_of_the_store_and_its_index() -> None:
    """The cut followed `groups`, "which the prover states with the window": the same store's block 0 held other bytes
    under another launch, so the verifier mapped j to stored bytes on the prover's word."""
    a, b = vectors_of_store_bytes(32, 1, 1), vectors_of_store_bytes(32, 1, 2)
    assert np.array_equal(a[:16], b[:16])
    assert np.array_equal(a, b) and np.array_equal(a, vectors_of_store_bytes(32, 1, 32))


def test_rbd2_a_kv_receipt_is_a_driver_receipt_of_one_registered_block() -> None:
    """The step commits its KV receipts in the pass's one tree after the weights' (kv_attn form 13, `kv_rc_receipts`).
    Token-layer u·layers + l of the KV region is the store's block nvec + u·layers + l, 32 whole words of 128 bytes, and
    has one receipt at that index: σ is the KV heads' shares of form 13's MMA over the words its decoder lanes read
    (`kv_fragment_sketch`) under the pass's signs, and the receipt is `SketchReceipter`'s, so `Session.verify` takes an
    honest KV opening.  It was a third kind: other signs, another hash domain, and 4,096 bytes from 16 tokens' head
    shares under a half-block index."""
    slots, layers, nvec = 5, 3, 4
    blocks = nvec + slots * layers
    store = np.random.default_rng(2).integers(0, 256, blocks * SR.BLOCK, dtype=np.uint8)
    m = SR.kv_vectors(slots, layers, nvec * SR.BLOCK)
    assert SR.rb.plan_of(m, blocks) == SR.rb.Plan.of(range(nvec, blocks), blocks)
    sk = SR.kv_fragment_sketch(store[m], SR.kv_sign_frags(SR.sign_bits(SEED)))
    receipter = SR.rb.SketchReceipter(SEED)
    assert SR.receipts_of(SEED, sk, nvec) == [receipter(j, store[j * SR.BLOCK:(j + 1) * SR.BLOCK].tobytes(), b"\0")
                                              for j in range(nvec, blocks)]
