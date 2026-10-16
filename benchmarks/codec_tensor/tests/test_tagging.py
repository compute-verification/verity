"""`tagging`'s ro-blake3/v1 over the store's 4,088-byte blocks, against `verity/core/protocols/bandwidth/tags.py`, and a
Python transcript of `wt.cu`'s `tag_quad` (a lane a chunk, two parent steps over shuffles) against the same."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tagging  # noqa: E402

SEED = bytes(range(32))
J0 = (1 << 32) + 3


def _store(nbytes: int, seed: int = 1) -> bytes:
    return random.Random(seed).randbytes(nbytes)


def quad_tag(seed: bytes, j: int, blk: bytes) -> bytes:
    """`tag_quad` step by step: lane c's chunk c of u64be j ‖ blk, then lanes 0 and 2's parents, then lane 0's."""
    from verity.ml.crypto import blake3 as B

    kw = tuple(int.from_bytes(tagging.key(seed)[4 * i:4 * i + 4], "little") for i in range(8))
    msg = j.to_bytes(8, "big") + blk
    nch = -(-len(msg) // 1024)
    cvs = []
    for c in range(4):
        clen = min(1024, len(msg) - 1024 * c)
        nmb = -(-clen // 64) if clen > 0 else 0
        cv = kw
        for k in range(nmb):
            fl = B.KEYED_HASH | (B.CHUNK_START if k == 0 else 0)
            if k == nmb - 1:
                fl |= B.CHUNK_END | (B.ROOT if nch == 1 else 0)
            part = msg[1024 * c + 64 * k:1024 * c + 64 * k + min(64, clen - 64 * k)]
            cv = tuple(B.compress(cv, part.ljust(64, b"\0"), c, len(part), fl)[:8])
        cvs.append(cv)

    def words(*vs):
        return b"".join(w.to_bytes(4, "little") for v in vs for w in v)

    p = [tuple(B.compress(kw, words(cvs[c], cvs[c + 1]), 0, 64,
                          B.KEYED_HASH | B.PARENT | (B.ROOT if nch == 2 else 0))[:8]) for c in (0, 2)]
    r = tuple(B.compress(kw, words(p[0], cvs[2] if nch == 3 else p[1]), 0, 64, B.KEYED_HASH | B.PARENT | B.ROOT)[:8])
    return words(cvs[0] if nch == 1 else p[0] if nch == 2 else r)


@pytest.mark.parametrize("last", [8, 1008, 1016, 1024, 2040, 2048, 3064, 3072, TAG_BLOCK := tagging.TAG_BLOCK])
def test_quad_transcript_is_the_tag(last):
    blk = _store(last, last)
    j = J0 + last
    assert quad_tag(SEED, j, blk) == tagging.tag(SEED, j, blk)


def test_full_block_is_four_chunks():
    assert (8 + tagging.TAG_BLOCK) == 4 * 1024


def test_fast_tags_are_tags_py():
    pytest.importorskip("blake3")
    store = _store(3 * tagging.TAG_BLOCK + 2048)
    ys = tagging.tags_fast(store, J0, SEED)
    assert len(ys) == tagging.block_count(len(store)) == 4
    assert ys == [tagging.tag(SEED, J0 + jl, tagging.block(store, jl)) for jl in range(4)]


def test_root_is_level_wise_with_the_odd_node_promoted():
    for n in range(1, 40):
        level = [bytes([i]) * 64 for i in range(n)]
        want = tagging.root(level)
        while len(level) > 1:
            level = [tagging.node(level[i], level[i + 1]) if i + 1 < len(level) else level[i]
                     for i in range(0, len(level), 2)]
        assert level[0] == want


def test_root_small_cases():
    a, b, c = (bytes([i]) * 64 for i in range(3))
    assert tagging.root([a]) == a
    assert tagging.root([a, b, c]) == tagging.node(tagging.node(a, b), c)


def test_leaves_short_last():
    ys = [bytes([i]) * 32 for i in range(17)]
    lv = tagging.leaves(SEED, ys)
    assert len(lv) == 2 and all(len(x) == 64 for x in lv)
