from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fill_ref as F  # noqa: E402

SEED = hashlib.sha256(b"test/fill").digest()


def test_sketch_is_the_signed_sum_of_each_block_read_as_int8():
    rng = np.random.default_rng(1)
    blocks = rng.integers(0, 256, size=(3, F.BLK), dtype=np.uint8)
    sb = F.signs(SEED)
    sigma = F.sign_matrix(sb)
    s = F.sketches(blocks, sigma)
    for b in range(3):
        for r in (0, 7, 15):
            want = 0
            for i in range(0, F.BLK, 97):
                bit = (sb[(F.BLK * r + i) // 8] >> ((F.BLK * r + i) % 8)) & 1
                want += (1 - 2 * bit) * int(np.int8(blocks[b, i]))
            sub = sum(int(sigma[r, i]) * int(np.int8(blocks[b, i])) for i in range(0, F.BLK, 97))
            assert sub == want
        assert s[b].dtype == np.int64 and abs(s[b]).max() <= 128 * F.BLK


def test_unit_root_is_the_tree_over_its_leaves_with_level_and_first_leaf_counters():
    rng = np.random.default_rng(2)
    unit = rng.integers(0, 256, size=F.UNIT_BYTES, dtype=np.uint8).tobytes()
    K, sigma = F.key_words(SEED), F.sign_matrix(F.signs(SEED))
    s = F.sketches(np.frombuffer(unit, dtype=np.uint8).reshape(F.UNIT_BLOCKS, F.BLK), sigma)
    u = 5

    def node(level: int, first: int) -> list[int]:
        if level == 0:
            return F.leaf(K, s[first - 32 * u], first)
        half = 1 << (level - 1)
        return F.parent(K, node(level - 1, first), node(level - 1, first + half), level, first)

    want = b"".join(w.to_bytes(4, "little") for w in node(5, 32 * u))
    assert F.unit_root(unit, u, K, sigma) == want


def test_a_chunk_spans_regions_by_whole_units_and_its_check_word_xors_the_roots():
    rng = np.random.default_rng(3)
    regions = [rng.integers(0, 256, size=n, dtype=np.uint8).tobytes() for n in (F.UNIT_BYTES + 100, 2 * F.UNIT_BYTES)]
    assert F.plan([len(r) for r in regions]) == [(0, 0), (1, 0), (1, F.UNIT_BYTES)]
    roots, x = F.chunk(regions, 0, 3, SEED, SEED)
    K, sigma = F.key_words(SEED), F.sign_matrix(F.signs(SEED))
    assert roots[1] == F.unit_root(regions[1][:F.UNIT_BYTES], 1, K, sigma)
    want = 0
    for r in roots:
        want ^= int.from_bytes(r[:4], "little")
    assert x == want


def test_the_key_and_the_signs_change_the_root():
    unit = bytes(range(256)) * (F.UNIT_BYTES // 256)
    other = hashlib.sha256(b"other").digest()
    base = F.chunk([unit], 0, 1, SEED, SEED)[0][0]
    assert F.chunk([unit], 0, 1, other, SEED)[0][0] != base
    assert F.chunk([unit], 0, 1, SEED, other)[0][0] != base
    flipped = bytearray(unit)
    flipped[F.UNIT_BYTES - 1] ^= 1
    assert F.chunk([bytes(flipped)], 0, 1, SEED, SEED)[0][0] != base


def test_a_v2_block_word_is_one_sha512_compression_of_the_tag_the_sketch_the_block_and_the_pass():
    s = np.arange(-8, 8, dtype=np.int64) * 1000
    msg = F.TAG_BLK + b"".join(int(v).to_bytes(4, "big", signed=True) for v in s) + (77).to_bytes(8, "big") + (3).to_bytes(8, "big")
    assert len(msg) == 96 and len(msg) <= 128 - 17  # one compression with its padding
    assert F.block_word(s, 77, 3) == int.from_bytes(hashlib.sha512(msg).digest()[:4], "big")
    assert F.block_word(s, 77, 4) != F.block_word(s, 77, 3) != F.block_word(s, 78, 3)


def test_v2_signs_are_fresh_each_pass_and_one_compression_each():
    a, b = F.signs_v2(SEED, 0), F.signs_v2(SEED, 1)
    assert len(a) == F.ROWS * F.BLK // 8 and a != b
    assert a[64:128] == hashlib.sha512(F.TAG_SGN + SEED + (0).to_bytes(8, "big") + (1).to_bytes(8, "big")).digest()
    assert len(F.TAG_SGN + SEED) + 16 <= 128 - 17


def test_a_v2_epoch_across_passes_xors_each_unit_word_under_its_pass_signs():
    rng = np.random.default_rng(4)
    regions = [rng.integers(0, 256, size=n, dtype=np.uint8).tobytes() for n in (2 * F.UNIT_BYTES, F.UNIT_BYTES + 9)]
    units = [regions[0][:F.UNIT_BYTES], regions[0][F.UNIT_BYTES:], regions[1][:F.UNIT_BYTES]]
    # the plan is 3 units: s = 2 .. 6 is pass 0's unit 2, pass 1's units 0 .. 2 and pass 2's unit 0
    want = 0
    for s in range(2, 7):
        p, u = divmod(s, 3)
        want ^= F.unit_word(units[u], u, p, F.sign_matrix(F.signs_v2(SEED, p)))
    assert F.epoch(regions, 2, 5, SEED) == want
    assert F.epoch(regions, 3, 1, SEED) != F.epoch(regions, 6, 1, SEED)  # unit 0 in passes 1 and 2


def test_pinned_v2_epoch():
    unit = bytes(range(256)) * (F.UNIT_BYTES // 256)
    assert F.epoch([unit, unit], 0, 3, SEED) == PINNED_V2


def test_pinned_root():
    unit = bytes(range(256)) * (F.UNIT_BYTES // 256)
    assert F.chunk([unit], 0, 1, SEED, SEED)[0][0].hex() == PINNED


PINNED = "00016451699de1ac6c130afeb67e713b66e93ed1d442c4be95821de368186235"
PINNED_V2 = 0xB1B8ED4C
