"""`sketch_receipt`'s blocks of a weight GEMM's store: a block is 4,096 stored bytes in store order (store-order/v0),
so the sketch warps' MMA arithmetic over the stored bytes (`fragment_sketch`, receipt_warp's lanes under sketch-sha512's
signs) is σ of each block, the launch's map is the plan the driver credits, and its receipts are the driver's
`SketchReceipter`'s of its blocks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sketch_receipt as SR  # noqa: E402

SEED = SR.rb.seed(bytes(range(32)), 3, bytes(64))
SB = 8192


@pytest.mark.parametrize("nbs, nchunks, groups", [(3, 1, 2), (5, 2, 1), (9, 1, 9), (17, 1, 2)])
def test_the_sketch_warps_sum_each_store_block(nbs, nchunks, groups):
    """Groups of SK_G = 4 tasks, a CTA's last group short (3 of 2 groups, 17 of 2), one task a CTA (9 of 9) and two
    chunks: each receipt is one store block's σ, every block once."""
    store = np.random.default_rng(nbs + groups).integers(0, 256, nbs * nchunks * SB, dtype=np.uint8)
    store[:2 * SR.BLOCK:2] = 0x80
    blocks = SR.gemm_blocks(store, SB)
    assert blocks.shape == (2 * nbs * nchunks, SR.BLOCK)
    mma = SR.fragment_sketch(store, nbs, nchunks, SB, groups, SR.sign_frags(SR.sign_bits(SEED)))
    assert np.array_equal(SR.sketches(SEED, blocks), mma)
    j0 = 11
    receipter = SR.rb.SketchReceipter(SEED)
    assert SR.receipts_of(SEED, mma, j0) == [receipter(j0 + v, blocks[v].tobytes()) for v in range(len(blocks))]


@pytest.mark.parametrize("nbs, nchunks, groups", [(3, 1, 2), (5, 2, 1), (9, 1, 9)])
def test_the_64_row_sketch_warps_sum_each_store_block(nbs, nchunks, groups):
    """At 64 rows receipt_warp's A is one AND of a bit plane (`sign_planes`) a register, 2^ρ times row ρ's bits, and
    a ones MMA's T recovers σ = T − 2 (Σ bit e): each receipt is the store block's σ at 64 rows, at the extremes
    (blocks of −128 and of 127, where |acc| reaches 2^26) too, and its receipts are the driver's at rows = 64."""
    store = np.random.default_rng(64 + nbs + groups).integers(0, 256, nbs * nchunks * SB, dtype=np.uint8)
    store[:SR.BLOCK], store[SR.BLOCK:2 * SR.BLOCK] = 0x80, 0x7F
    blocks = SR.gemm_blocks(store, SB)
    mma = SR.fragment_sketch64(store, nbs, nchunks, SB, groups, SR.sign_planes(SR.sign_bits(SEED, SR.ROWS64)))
    assert np.array_equal(SR.sketches(SEED, blocks, SR.ROWS64), mma)
    receipter = SR.rb.receipter(SR.rb.SKETCH_SHA512, SEED, 64)
    assert SR.receipts_of(SEED, mma, 5) == [receipter(5 + v, blocks[v].tobytes()) for v in range(len(blocks))]
    assert SR.receipts_of(SEED, mma[:, :SR.ROWS], 5)[0] != SR.rb.SketchReceipter(SEED)(5, blocks[0].tobytes())


def test_the_64_row_planes():
    """sign_words takes 64 rows as it takes 16; sign_planes' bit 8 b + ρ of lane (g, t)'s word 2 s + hh at p is row
    16 (ρ // 2) + g + 8 (ρ % 2)'s sign at 64 p + 16 t + 8 s + 4 hh + b; sign_frags refuses 64 rows."""
    bits = SR.sign_bits(SEED, SR.ROWS64)
    assert SR.sign_words(bits).tobytes() == bits.tobytes() and SR.sign_words(bits).size == SR.PLANE_WORDS
    pl = SR.sign_planes(bits).reshape(SR.BLOCK // 64, SR.LANES, 4)
    signs = SR.signs(SEED, SR.ROWS64)
    for p, lane, w, b, rho in ((0, 0, 0, 0, 0), (63, 31, 3, 3, 7), (17, 9, 2, 1, 4), (40, 22, 1, 2, 5)):
        g, t = divmod(lane, 4)
        bit = int(pl[p, lane, w]) >> (8 * b + rho) & 1
        assert 1 - 2 * bit == signs[16 * (rho >> 1) + g + 8 * (rho & 1), 64 * p + 16 * t + 8 * (w >> 1) + 4 * (w & 1) + b]
    with pytest.raises(ValueError, match="16 × 512"):
        SR.sign_frags(bits)
    with pytest.raises(ValueError, match="64 × 512"):
        SR.sign_planes(SR.sign_bits(SEED))


@pytest.mark.parametrize("groups", [1, 2, 5])
def test_a_launch_receipts_the_registered_blocks(groups):
    nbs, nchunks = 5, 2
    blocks = nbs * nchunks * SB // SR.BLOCK
    assert SR.rb.plan_of(SR.gemm_vectors(nbs, nchunks, SB, groups), blocks) == SR.rb.Plan.whole(blocks)
    assert sorted(SR.gemm_receipts(nbs, nchunks, SB, groups)) == list(range(blocks))


def test_sign_layouts():
    """sign_words is S's bits as bytes; sign_frags' selectors expand through prmt to each position's sign."""
    import sketching as SK

    bits = SR.sign_bits(SEED)
    assert SR.sign_words(bits).size == SR.SIGN_WORDS and SR.sign_words(bits).tobytes() == bits.tobytes()
    fr = SR.sign_frags(bits).reshape(SR.BLOCK // 128, SR.LANES, 4)
    assert fr.size == SR.FRAG_WORDS
    signs = SR.signs(SEED)
    for pp, lane, x, big_r in ((0, 0, 0, 0), (31, 31, 1, 7), (7, 13, 1, 3), (20, 6, 0, 5)):
        g, t = divmod(lane, 4)
        s, r = divmod(big_r, 4)
        sel = int(fr[pp, lane, 2 * x + big_r // 6]) >> (16 * ((big_r % 6) // 3))
        reg = SK.prmt(*SK.PRMT[big_r % 3], sel & 0xFFFF)
        pos = 64 * (2 * pp + x) + 16 * t + 8 * s + 4 * (r >> 1)
        got = [1 if (reg >> 8 * q) & 0xFF == 1 else -1 for q in range(4)]
        assert got == signs[g + 8 * (r & 1), pos:pos + 4].tolist()


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
    assert np.array_equal(sigma, SR.sketches(SEED, blocks, RD.ROWS64))
    assert not np.array_equal(SR.signs(SEED, RD.ROWS64)[:SR.ROWS], SR.signs(SEED))


def test_attention_sketches_each_token_layer():
    """kv_attn form 13: the KV heads' MMA shares over their words (`kv_fragment_sketch` from the A table
    `kv_sign_frags`) sum to σ of each token-layer, the heads' words partition the block, each head's share is S at its
    positions, and the receipts' map is the KV region's blocks in store order."""
    blocks = np.random.default_rng(13).integers(0, 256, (6, SR.BLOCK), dtype=np.uint8)
    blocks[0], blocks[1] = 0x80, 0x7F
    s, e = SR.signs(SEED), blocks.view(np.int8).astype(np.int64)
    sigma = SR.sketches(SEED, blocks)
    assert np.array_equal(SR.kv_fragment_sketch(blocks, SR.kv_sign_frags(SR.sign_bits(SEED))), sigma)
    pos = [SR.kv_positions(g) for g in range(SR.KV_HEADS)]
    assert np.array_equal(np.sort(np.concatenate(pos)), np.arange(SR.BLOCK))
    assert np.array_equal(sum(e[:, p] @ s[:, p].T.astype(np.int64) for p in pos), sigma)
    slots, layers, nvec = 3, 2, 7
    assert SR.rb.plan_of(SR.kv_vectors(slots, layers, nvec * SR.BLOCK), nvec + slots * layers) == SR.rb.Plan.of(
        range(nvec, nvec + slots * layers), nvec + slots * layers)


def test_the_chained_pass_completes_the_driver_s_messages():
    """reader.cu's chain_kernel completes `chain_messages`' templates from a pass's seed: the receipt prefix, the salts'
    input, the tree's binding, and with that the domain message whose SHA-512 is `rb.domain`'s id, at each plan."""
    import hashlib

    import reader as RD

    rb = SR.rb
    key = bytes(range(32))
    for plan in (rb.Plan(((901, 37), (5, 300), (2000, 18)), 4096), rb.Plan.whole(4096), rb.Plan.whole(1)):
        m = RD.chain_messages(plan, key)
        for seed in (SEED, rb.seed(b"\x01" * 32, 7, b"\x02" * 64)):
            def put(name, v):
                t, a = m[name]
                return t[:a] + v + t[a + 64:]
            assert put("receipt", seed) == rb.prefix(seed)
            assert put("salt", seed) == RD.store.SALT_DOMAIN + key + seed
            binding = hashlib.sha512(put("binding", seed)).digest()
            assert binding == rb.binding(seed, plan)
            assert hashlib.sha512(put("domain", binding)).digest() == rb.domain(seed, plan).domain_id
            assert len(m["binding"][0]) <= 239 and len(m["domain"][0]) <= 239 and len(m["receipt"][0]) == 128


def test_gemm_blocks_refuse_other_widths():
    with pytest.raises(ValueError, match="4096-byte blocks"):
        SR.gemm_blocks(np.zeros(6144, dtype=np.uint8), 6144)
    with pytest.raises(ValueError, match="two 4096-byte blocks"):
        SR.gemm_receipts(2, 1, 4096, 1)
