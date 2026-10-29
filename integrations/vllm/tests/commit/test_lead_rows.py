"""``verity_vllm.commit.lead_rows`` commits what the lead's kernel exports: its committed stream equals a twin built word by word
from the export's layout (``ssn.cu``'s ``word_at``) and ``layout.h``'s padding, on random words and ragged calls; each stream
tile is Lean's ``tileOf``; the chunks and segments cut where the layout says, the last short; the leaves are core's inline
frame-v3-sha512 leaves, distinct at every position for equal values; and the hashing meets ``CommitL1w``'s credited blocks."""
from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity_vllm.commit import lead_rows as L

SALT = hashlib.sha512(b"lead_rows/salt").digest()


def _word_at(i: int, j: int, t: int, S: int, n: int, bn: int) -> int:
    r, c = i & 63, j % bn
    w, g, eh, ni, q, el = r >> 4, r & 7, (r >> 3) & 1, c >> 3, (c & 7) >> 1, c & 1
    blk = ((i >> 6) * (n // bn) + j // bn) * S + t
    return blk * (64 * bn) + (ni * 128 + 32 * w + 4 * g + q) * 4 + 2 * eh + el


def _computed(m: int, n: int, bn: int) -> tuple[int, int]:
    return 64 * -(-m // 64), bn * -(-n // bn)


def _export(W: np.ndarray, S: int, bn: int) -> np.ndarray:
    """The kernel's export of the computed words W[i, j, t]."""
    M, N, _ = W.shape
    out = np.zeros(M * N * S, dtype="<i4")
    for i in range(M):
        for j in range(N):
            for t in range(S):
                out[_word_at(i, j, t, S, N, bn)] = W[i, j, t]
    return out


def _twin(W: np.ndarray, m: int, n: int, S: int, bn: int) -> bytes:
    """The committed stream word by word: the tiles in (rt, ct, t, ni, w) order inside pad_rows(m) x pad_cols(n), each in
    fragment order."""
    R, C = 16 * ((m + 15) // 16), 8 * ((n + 7) // 8)
    M, N, _ = W.shape
    out = []
    for rt in range(M // 64):
        for ct in range(N // bn):
            for t in range(S):
                for ni in range(bn // 8):
                    for w in range(4):
                        if 64 * rt + 16 * w >= R or bn * ct + 8 * ni >= C:
                            continue
                        for p in range(128):
                            g, q, eh, el = p >> 4, (p >> 2) & 3, (p >> 1) & 1, p & 1
                            out.append(W[64 * rt + 16 * w + 8 * eh + g, bn * ct + 8 * ni + 2 * q + el, t])
    return np.array(out, dtype="<i4").tobytes()


def _random_words(rng: np.random.Generator, m: int, n: int, S: int, bn: int) -> np.ndarray:
    M, N = _computed(m, n, bn)
    return rng.integers(-(2**31), 2**31, size=(M, N, S), dtype=np.int64).astype(np.int32)


SHAPES = [(1, 128, 3, 128), (16, 128, 2, 128), (17, 136, 3, 128), (64, 256, 2, 128), (65, 72, 2, 128), (100, 300, 2, 256),
          (1, 256, 1, 256), (130, 128, 1, 128)]


@pytest.mark.parametrize("m,n,S,bn", SHAPES)
def test_stream_is_the_twin_on_random_words(m, n, S, bn):
    W = _random_words(np.random.default_rng(m * 1000 + n + S), m, n, S, bn)
    got = L.stream(_export(W, S, bn), m, n, S, bn)
    assert got == _twin(W, m, n, S, bn)
    assert len(got) == 4 * L.pad_rows(m) * L.pad_cols(n) * S


@pytest.mark.parametrize("m,n,S,bn", SHAPES)
def test_each_stream_tile_is_leans_tile(m, n, S, bn):
    """Tile k of the stream holds Lean's tileOf (row block rb, column block cb, step t): row 16 rb + r and column 8 cb + c at
    fragment position 16 g + 4 q + 2 eh + el, r = 8 eh + g and c = 2 q + el."""
    W = _random_words(np.random.default_rng(7 + m + n), m, n, S, bn)
    words = np.frombuffer(L.stream(_export(W, S, bn), m, n, S, bn), dtype="<i4").reshape(-1, 128)
    M, N = _computed(m, n, bn)
    k = 0
    for rt in range(M // 64):
        for ct in range(N // bn):
            for t in range(S):
                for ni in range(bn // 8):
                    for w in range(4):
                        rb, cb = 4 * rt + w, ct * bn // 8 + ni
                        if 16 * rb >= L.pad_rows(m) or 8 * cb >= L.pad_cols(n):
                            continue
                        tile = W[16 * rb:16 * rb + 16, 8 * cb:8 * cb + 8, t]
                        frag = [tile[8 * ((p >> 1) & 1) + (p >> 4), 2 * ((p >> 2) & 3) + (p & 1)] for p in range(128)]
                        assert list(words[k]) == frag
                        k += 1
    assert k == len(words) == L.pad_rows(m) * L.pad_cols(n) * S // 128


def test_padding_rows_are_committed_and_the_rest_are_not():
    """Decode's one row: rows 1..15 are padding rows, committed with whatever the kernel computed there; rows 16..63 and the
    columns past pad_cols are computed and not committed."""
    m, n, S, bn = 1, 132, 2, 128
    M, N = _computed(m, n, bn)
    W = np.full((M, N, S), 3, dtype=np.int32)
    W[1:16] = 5
    W[16:] = 7
    W[:, 136:] = 7
    words = np.frombuffer(L.stream(_export(W, S, bn), m, n, S, bn), dtype="<i4")
    assert (words == 7).sum() == 0
    assert (words == 5).sum() == 15 * 136 * S
    assert (words == 3).sum() == 136 * S


def test_committed_tiles_are_the_streams_offsets():
    m, n, S, bn = 65, 200, 3, 128
    W = _random_words(np.random.default_rng(3), m, n, S, bn)
    ex = _export(W, S, bn)
    offs = L.committed_tiles(m, n, S, bn)
    assert b"".join(ex[o:o + 128].tobytes() for o in offs) == L.stream(ex, m, n, S, bn)
    assert len(offs) == L.pad_rows(m) * L.pad_cols(n) * S // 128


def test_stream_refuses_an_export_of_another_shape():
    with pytest.raises(ValueError):
        L.stream(np.zeros(64 * 128 * 2 - 1, dtype=np.int32), 1, 128, 2, 128)


def test_padding_matches_layout_h():
    for m in range(1, 300):
        assert L.pad_rows(m) == 16 * ((m + 15) // 16) and L.pad_cols(m) == 8 * ((m + 7) // 8)


def test_chunks_cut_at_256_KiB_and_the_last_is_short():
    m, n, S, bn = 17, 256, 9, 128
    words = L.stream(_export(_random_words(np.random.default_rng(9), m, n, S, bn), S, bn), m, n, S, bn)
    cs = L.chunks(words)
    total = L.pad_rows(m) * L.pad_cols(n) * S
    assert total == 73728 and len(cs) == -(-total // L.CHUNK_WORDS) == 2
    assert [len(c) for c in cs] == [4 * 65536, 4 * 8192] and b"".join(cs) == words


def test_chunks_never_span_calls():
    a, b = bytes(range(256)) * 1100, bytes(range(255, -1, -1)) * 3
    assert L.words_values([a, b]) == L.chunks(a) + L.chunks(b)
    assert [len(v) for v in L.words_values([a, b])] == [4 * 65536, 281600 - 4 * 65536, 768]


@pytest.mark.parametrize("rows,K", [(1, 2048), (3, 2048), (1, 8192), (1, 1056), (5, 96)])
def test_a_segments(rows, K):
    a = random.Random(rows * K).randbytes(rows * K)
    segs = L.a_segments(a)
    assert b"".join(segs) == a and len(segs) == -(-rows * K // L.A_SEGMENT)
    assert all(len(s) == L.A_SEGMENT for s in segs[:-1]) and 0 < len(segs[-1]) <= L.A_SEGMENT


def test_leaves_are_inline_frame_v3_leaves_at_their_rank():
    dom = L.instance_domain(SALT, L.a_position(2, 64, 5), 3)
    values = [b"\x01" * 1024, b"\x02" * 1024, b"\x03" * 32]
    sub = L.subtree(dom, L.A_SCHEMA, values)
    for r, v in enumerate(values):
        prefix = dom.leaf_prefix(r, r, L.A_SCHEMA, len(v))
        assert sub.levels[0][r] == hashlib.sha512(prefix + v).digest() == dom.leaf(r, r, L.A_SCHEMA, v)
    assert sub.levels[0][3] == dom.pad(3)
    assert sub.root == dom.node(1, 0, dom.node(0, 0, *sub.levels[0][:2]), dom.node(0, 1, *sub.levels[0][2:]))
    assert sub.depth == 2


def test_one_leaf_instance_is_its_leaf():
    dom = L.instance_domain(SALT, L.words_position(0), 1)
    sub = L.subtree(dom, L.WORDS_SCHEMA, [b"\x00" * 512])
    assert sub.levels == ((dom.leaf(0, 0, L.WORDS_SCHEMA, b"\x00" * 512),),) and sub.depth == 0


def test_equal_values_share_no_compression():
    """Zero data (decode's padding, a repeated prompt): equal chunks at different positions differ from the first block on."""
    dom = L.instance_domain(SALT, L.words_position(1), 300)
    firsts = {dom.leaf_prefix(r, r, L.WORDS_SCHEMA, 4 * L.CHUNK_WORDS)[:128] for r in range(300)}
    assert len(firsts) == 300


def test_hashing_meets_commit_l1w_credit():
    """CommitL1w credits the words 4 m n S / 128 + chunks blocks and A rows K / 128, over the distinct messages."""
    m, n, S, bn, K = 17, 256, 9, 128, 2048
    words = L.stream(_export(_random_words(np.random.default_rng(11), m, n, S, bn), S, bn), m, n, S, bn)
    a = random.Random(1).randbytes(m * K)
    for values, schema, floor in ((L.chunks(words), L.WORDS_SCHEMA, 4 * L.pad_rows(m) * L.pad_cols(n) * S / 128 + 2),
                                  (L.a_segments(a), L.A_SCHEMA, m * K / 128)):
        dom = L.instance_domain(SALT, 0, len(values))
        msgs = {dom.leaf_prefix(r, r, schema, len(v)) + v for r, v in enumerate(values)}
        assert len(msgs) == len(values)
        assert sum((len(x) + 17 + 127) // 128 for x in msgs) >= floor


def test_instance_domains():
    dom = L.instance_domain(SALT, L.a_position(3, 64, 63), 5)
    assert dom == CommitmentDomain(hashlib.sha512(L.BINDING_TAG + SALT + (3 * 64 + 63).to_bytes(8, "big")).digest(), -1,
                                   RangeIndexedDomain(0, 5), hash="sha512")
    ids = {L.instance_domain(s, p, 5).domain_id for s in (SALT, bytes(64)) for p in (0, 1, L.words_position(0))}
    assert len(ids) == 6
    assert L.a_position(10, 64, 63) < L.WORDS_POSITION <= L.words_position(0)
    for bad in ((0, 64, 64), (-1, 64, 0), (1 << 57, 64, 0)):
        with pytest.raises(ValueError):
            L.a_position(*bad)
