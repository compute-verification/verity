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


def test_a_leafs_message_holds_its_rank_and_position():
    """Each leaf of ``subtree`` is SHA-512 of its frame-v3 message, built here byte by byte: the frame, the tag, the domain id
    and then u64be-length-framed uint(rank), uint(position), the schema and the value, at ranks of one and two bytes."""
    n = 300
    dom = L.instance_domain(SALT, L.a_position(1, 64, 2), n)
    values = [r.to_bytes(2, "big") * 4 for r in range(n)]
    sub = L.subtree(dom, L.A_SCHEMA, values)
    frame = b"veritor/protocol/merkle/frame/v3\0" + (4).to_bytes(4, "big") + b"leaf"
    for r in (0, 1, 7, 255, 256, 299):
        u = r.to_bytes(max(1, (r.bit_length() + 7) // 8), "big")
        parts = (dom.domain_id, u, u, L.A_SCHEMA.encode(), values[r])
        message = frame + b"".join(len(p).to_bytes(8, "big") + p for p in parts)
        assert sub.levels[0][r] == hashlib.sha512(message).digest()
        assert message[121:121 + len(u)] == u == message[129 + len(u):129 + 2 * len(u)]


def _label(name: str, n: int) -> bytes:
    label = f"lead/keyed/vector/{name}".encode()
    return b"".join(hashlib.sha512(label + i.to_bytes(8, "big")).digest() for i in range(-(-n // 64)))[:n]


KS_W, SALT_W = hashlib.sha512(b"lead/keyed/vector/ks_w").digest(), hashlib.sha512(b"lead/keyed/vector/salt_w").digest()
WEIGHTS = hashlib.sha512(b"lead/keyed/vector/weights").digest()


def _window(calls: int = 64) -> L.Window:
    return L.Window(calls, ks_w=KS_W, y0=_label("y0", L.KEY_SALT_BYTES))


def test_window_reproduces_the_keyed_vectors():
    """``verity.ml.commitments.keyed``'s vectors through the committer: keyLeaf, d(A_j) at f 3, calls 64, j 5 over 2 rows of K
    2048 (4 segments), the forward's words over two calls (chunks of 65,536, 65,536, 32,768 and 1,024 words), and a one-leaf
    A at j 7."""
    w = _window()
    reg = w.register(WEIGHTS)
    assert reg == (WEIGHTS, w.key_leaf, 64)
    assert w.key_leaf.hex() == ("e121d1ae87e74b21ddb3585cff79d9c21aac50281d5e3a77fa18453007ae3f8d1315a9b3c0dda315ed6e199f57f62a9dee19a9fa4fcd25ac"
                                "67be7c612672c095")
    w.open(SALT_W)
    a = _label("a", 4096)
    assert w.a_root(3, 5, a).hex() == ("3bf65c529446a801fb137bc9eae0bae2c005c7d79fc48cd7b1a33b48e4dbf64a059f7100b300cb6f52d1d8ad39278e800c9"
                                       "4f2d0c5b010af31d49817ec11f4c1")
    words = [_label("words/0", 655360), _label("words/1", 4096)]
    assert w.words_root(3, words).hex() == ("d6776c42b1346f3ace17242764e08fcb23621d8adc544d190e197266a069f51d933b859b783b990cba2a1d3ee4226a"
                                            "92f8298acb94ec3caa003e4dbaae93b9d3")
    assert w.a_root(3, 7, a[:1024]).hex() == ("21ea90132419867d07995334c4021454dfcfe7ca594854e590049bca06089ed7618b6b94bd53c5c0cabdf425308480"
                                              "f464f1a22b8919e6e6a062514c74ecc12e")


def test_window_refuses_a_second_digest_at_a_position():
    w = _window()
    w.register(WEIGHTS)
    w.open(SALT_W)
    first = w.a_root(0, 1, b"\x01" * 2048)
    for again in (lambda: w.a_root(0, 1, b"\x01" * 2048), lambda: w.a_root(0, 1, b"\x02" * 64)):
        with pytest.raises(L.Refused):
            again()
    w.words_root(0, [b"\x03" * 512])
    with pytest.raises(L.Refused):
        w.words_root(0, [b"\x04" * 512])
    assert w.a_root(0, 2, b"\x01" * 2048) != first


def test_window_refuses_positions_outside_its_injective_map():
    """Positions f calls + j with j < calls and f calls + j < 2^63 are distinct and below every words position 2^63 + f; the
    window refuses a call count, a call or a forward outside them."""
    for calls in (0, -1, 1 << 63, 2.0):
        with pytest.raises(L.Refused):
            L.Window(calls)
    calls = 5
    positions = {L.a_position(f, calls, j) for f in range(7) for j in range(calls)}
    assert positions == set(range(7 * calls)) and max(positions) < L.words_position(0)
    w = _window(calls)
    w.register(WEIGHTS)
    w.open(SALT_W)
    for f, j in ((0, calls), (0, -1), (-1, 0), (((1 << 63) - 1) // calls, calls - 1), ((1 << 63) // calls + 1, 0)):
        with pytest.raises(L.Refused):
            w.a_root(f, j, b"\x05" * 64)
    for f in (-1, 1 << 63):
        with pytest.raises(L.Refused):
            w.words_root(f, [b"\x06" * 64])
    with pytest.raises(L.Refused):
        w.words_root(0, [b"\x06" * 64] * (calls + 1))
    assert w.a_root(((1 << 63) - 1) // calls - 1, calls - 1, b"\x05" * 64)
    assert w.words_root(0, [b"\x06" * 64] * calls)


def test_window_refuses_salt_w_before_its_registration():
    w = _window()
    with pytest.raises(L.Refused):
        w.open(SALT_W)
    with pytest.raises(L.Refused):
        w.a_root(0, 0, b"\x07" * 64)
    w.register(WEIGHTS)
    with pytest.raises(L.Refused):
        w.a_root(0, 0, b"\x07" * 64)
    with pytest.raises(L.Refused):
        w.open(SALT_W[:32])
    w.open(SALT_W)
    for late in (lambda: w.open(SALT_W), lambda: w.register(WEIGHTS)):
        with pytest.raises(L.Refused):
            late()
    with pytest.raises(AttributeError):
        w.calls = 65
    assert w.a_root(0, 0, b"\x07" * 64)


def test_windows_with_fresh_keys_differ():
    a, b = L.Window(4), L.Window(4)
    assert a.key_leaf != b.key_leaf
    for w in (a, b):
        w.register(WEIGHTS)
        w.open(SALT_W)
    assert a.a_root(0, 0, b"\x08" * 1024) != b.a_root(0, 0, b"\x08" * 1024)


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
