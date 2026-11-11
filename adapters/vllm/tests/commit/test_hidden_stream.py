"""Binding tests for the hidden-stream commitment: openings against the prechallenge root, negatives.

CPU only: the streams are synthetic (random words in the M1 layout) -- the *acquisition* correctness (bit-identity,
oracle x-check, in-kernel SHA-256 == hashlib) is tested on the GPU by `out/gen/r9/cmt-hidden/src/hidden_tap.py xcheck`.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from verity_vllm.commit import hidden_stream as hs


def _layout() -> hs.StreamLayout:
    return hs.StreamLayout(HB=2, M=64, NB=2)


def _stream(layout: hs.StreamLayout, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**32, size=layout.total_words, dtype=np.uint64).astype(np.uint32)


def test_layout_word_index_roundtrip():
    L = _layout()
    for cls, kw in [("S", dict(col=200)), ("P", dict(col=5)), ("RP", dict(col=130)), ("ROW", dict(block=1, lane=3)),
                    ("ROW", dict(block=0, field=1)), ("O", dict(block=1, d=63)), ("FIN", dict(field=3))]:
        w = L.word_index(cls, 1, 17, **kw)
        u = L.unrank(w)
        assert u["cls"] == cls and u["slab"] == 1 and u["row"] == 17
    assert L.total_words % 64 == 0
    assert L.unrank(L.total_words - 1)["cls"] == "FIN"


def test_chunk_openings_verify_and_negatives():
    L = _layout()
    s = _stream(L)
    c = hs.commit_stream(s, L, launch_tag=11, src_mask=63, chunk_words=64)
    assert c.leaf_count == L.total_words // 64
    positions = [L.word_index("S", 0, 3, col=7), L.word_index("P", 1, 63, col=255), L.word_index("RP", 1, 9, col=101),
                 L.word_index("ROW", 0, 0, block=1, lane=2), L.word_index("O", 1, 40, block=0, d=17), L.word_index("FIN", 1, 63, field=1)]
    for w in positions:
        op = c.open(s, w)
        assert op.word == int(s[w])
        assert hs.verify_chunk_opening(c.root, L, 11, 63, 64, c.leaf_count, w, op)
        # (a) altered word inside the chunk
        ch = bytearray(op.chunk)
        ch[4 * op.offset] ^= 1
        bad = hs.ChunkOpening(op.index, op.offset, bytes(ch), op.path)
        assert not hs.verify_chunk_opening(c.root, L, 11, 63, 64, c.leaf_count, w, bad)
        # (b) two words of the chunk swapped (same multiset, different positions)
        ch = bytearray(op.chunk)
        i, j = 4 * op.offset, 4 * ((op.offset + 1) % 64)
        ch[i:i + 4], ch[j:j + 4] = op.chunk[j:j + 4], op.chunk[i:i + 4]
        assert not hs.verify_chunk_opening(c.root, L, 11, 63, 64, c.leaf_count, w, hs.ChunkOpening(op.index, op.offset, bytes(ch), op.path))
        # (c) substituted: the chunk (and its valid path) of another position presented for this one
        other = (w + 64 * 5) % L.total_words
        oo = c.open(s, other)
        assert not hs.verify_chunk_opening(c.root, L, 11, 63, 64, c.leaf_count, w, oo)
        assert not hs.verify_chunk_opening(c.root, L, 11, 63, 64, c.leaf_count, w, hs.ChunkOpening(op.index, op.offset, oo.chunk, oo.path))
        # (d) wrong launch tag / wrong geometry / wrong root
        assert not hs.verify_chunk_opening(c.root, L, 12, 63, 64, c.leaf_count, w, op)
        assert not hs.verify_chunk_opening(c.root, hs.StreamLayout(2, 64, 3), 11, 63, 64, c.leaf_count, w, op)
        c2 = hs.commit_stream(_stream(L, seed=2), L, launch_tag=11, src_mask=63)
        assert not hs.verify_chunk_opening(c2.root, L, 11, 63, 64, c.leaf_count, w, op)
        # (e) truncated / extended path
        assert not hs.verify_chunk_opening(c.root, L, 11, 63, 64, c.leaf_count, w, hs.ChunkOpening(op.index, op.offset, op.chunk, op.path[:-1]))


def test_chunk_commitment_is_deterministic_and_order_sensitive():
    L = _layout()
    s = _stream(L)
    a = hs.commit_stream(s, L, 3, 63)
    b = hs.commit_stream(s.copy(), L, 3, 63)
    assert a.root == b.root
    # reordering two chunks changes the root (position is bound), even though the multiset of chunks is the same
    s2 = s.copy()
    s2[0:64], s2[64:128] = s[64:128].copy(), s[0:64].copy()
    assert hs.commit_stream(s2, L, 3, 63).root != a.root
    assert hs.commit_stream(s, L, 4, 63).root != a.root
    assert hs.commit_stream(s, L, 3, 31).root != a.root


def _geo(L: hs.StreamLayout, seqlens_q=(50, 64), seqlens_k=(150, 256), causal=True) -> hs.ThreadGeometry:
    return hs.ThreadGeometry(HB=L.HB, b=L.HB, h=1, M=L.M, NB=L.NB, num_m_block=L.M // 64, seqlens_q=tuple(seqlens_q), seqlens_k=tuple(seqlens_k), causal=causal)


def _kernel_digests(s: np.ndarray, L: hs.StreamLayout, g: hs.ThreadGeometry, src: int, tag: int) -> np.ndarray:
    """What the H1 kernel would write: SHA-256 of every visited thread leaf, zero for unvisited slots."""
    d = np.zeros((g.leaf_count, 8), dtype=np.uint32)
    for bidb in range(g.b):
        for bidh in range(g.h):
            slab = bidb * g.h + bidh
            for mb in range(g.num_m_block):
                if mb * 64 >= g.seqlens_q[bidb]:
                    continue
                for nb in range(g.n_block_max(bidb, mb)):
                    for tidx in range(128):
                        msg = hs.thread_message(s, L, g, src, tag, bidb, slab, mb, nb, tidx)
                        d[g.leaf_index(slab, mb, nb, tidx)] = np.frombuffer(hashlib.sha256(msg).digest(), dtype=np.uint32)
    return d


@pytest.mark.parametrize("src", [31, 2, 4])
def test_thread_leaf_openings_and_negatives(src):
    L = _layout()
    s = _stream(L, seed=5)
    g = _geo(L)
    tag = 9
    d = _kernel_digests(s, L, g, src, tag)
    c = hs.ThreadLeafCommitment.from_digests(d, g, tag, src)
    # open a P value at canonical position (slab 1, row 40, col 200): owning thread, message regenerated from the stream
    slab, row, col = 1, 40, 200
    tidx, mi, t = hs.thread_of(row, col)
    mb, nb = row // 64, col // 128
    msg = hs.thread_message(s, L, g, src, tag, 1, slab, mb, nb, tidx)
    op = c.open(msg, g.leaf_index(slab, mb, nb, tidx))
    assert hs.verify_thread_opening(c.root, g, tag, src, 1, slab, mb, nb, tidx, op)
    if src & 2:
        # the word sits at a fixed offset of the message: ni index of col within the thread's 32 columns
        ni = [n for n in range(32) if nb * 128 + 8 * (n >> 1) + 2 * t + (n & 1) == col][0]
        off = hs.thread_word_offset(src, "P", mi, ni)
        assert struct.unpack_from("<I", msg, 4 * off)[0] == int(s[L.word_index("P", slab, row, col=col)])
    # negatives: altered word, swapped words, wrong position (header mismatch), wrong tag, substituted leaf, foreign path
    m2 = bytearray(msg); m2[64 + 3] ^= 0x80
    assert not hs.verify_thread_opening(c.root, g, tag, src, 1, slab, mb, nb, tidx, hs.ThreadOpening(op.index, bytes(m2), op.path))
    m3 = bytearray(msg); m3[64:68], m3[68:72] = msg[68:72], msg[64:68]
    if msg[64:68] != msg[68:72]:
        assert not hs.verify_thread_opening(c.root, g, tag, src, 1, slab, mb, nb, tidx, hs.ThreadOpening(op.index, bytes(m3), op.path))
    assert not hs.verify_thread_opening(c.root, g, tag, src, 1, slab, mb, nb, (tidx + 1) % 128, op)
    assert not hs.verify_thread_opening(c.root, g, tag + 1, src, 1, slab, mb, nb, tidx, op)
    other = hs.thread_message(s, L, g, src, tag, 0, 0, 0, 0, tidx)
    oo = c.open(other, g.leaf_index(0, 0, 0, tidx))
    assert hs.verify_thread_opening(c.root, g, tag, src, 0, 0, 0, 0, tidx, oo)
    assert not hs.verify_thread_opening(c.root, g, tag, src, 1, slab, mb, nb, tidx, oo)
    assert not hs.verify_thread_opening(c.root, g, tag, src, 1, slab, mb, nb, tidx, hs.ThreadOpening(op.index, msg, oo.path))
    # a stream altered in one bound word changes the digests -> different root
    s2 = s.copy(); s2[L.word_index("P" if src & 2 else "RP", slab, row, col=col)] ^= 1
    c2 = hs.ThreadLeafCommitment.from_digests(_kernel_digests(s2, L, g, src, tag), g, tag, src)
    assert c2.root != c.root


def test_thread_message_header_matches_kernel_convention():
    L = _layout(); s = _stream(L); g = _geo(L)
    msg = hs.thread_message(s, L, g, 31, 5, 0, 0, 0, 1, 37)
    hdr = struct.unpack("<16I", msg[:64])
    assert hdr[0] == hs.TAG and hdr[1] == (7 << 16) | 31 and hdr[3:7] == (0, 0, 1, 37) and hdr[7:10] == (50, 150, 5)
    assert hdr[10] == g.n_block_max(0, 0)
    # thread 37: warp 1, lane 5 -> rows 16+1=17 and 25, both < 50
    assert hdr[11] == 3
    assert len(msg) == 64 + 4 * (64 + 16 + 64 + 32 + 32)
