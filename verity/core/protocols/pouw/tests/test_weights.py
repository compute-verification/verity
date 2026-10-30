"""`weights.weights_roots`, the streaming b/<K> commitment the auditor registers a model with, against
`registered.Registered.commit` over the same rows and salts, and against `scalar_roots`, the per-row reference it replaced
(`hm96_salt`, `commit_string`, `registered.leaf`, `tree_levels`), for any chunking and any `workers`."""
from __future__ import annotations

import random
import struct

import numpy as np
import pytest

from verity.core.protocols.pouw import weights as Wt
from verity.core.protocols.pouw.audit import chacha20_block, hm96_salt
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.rowleaf import sha512_row_v2_digest
from verity.ml.crypto import chacha20

CIRCUIT = "ab" * 64
PREFIX = "h1/"
KEY = bytes(range(32))


def scalar_roots(circuit, prefix, linears, rows, key):
    """Each K's root one row at a time: every leaf from core's per-row path, kept, then `tree_levels`."""
    out = {}
    for i, k in enumerate(Wt.ks(linears)):
        mine = [w for w, (_n, kw) in enumerate(linears) if kw == k]
        width = 2 * Wt.b_words(k)
        dom = G.domain(circuit, Wt.value(prefix, k), Wt.b_words(k), sum(linears[w][0] for w in mine))
        leaves = []
        for w in mine:
            for row in rows(w):
                j, x = len(leaves), sha512_row_v2_digest(bytes(row), 8 * width)
                salt = hm96_salt(key, Wt.B_NONCE + i, j)
                leaves.append(G.leaf(dom, j, hm96.SHA512.commit_string(hm96.SHA512.default_key, x, salt), Wt.SCHEMA))
        out[k] = tree_levels(dom, leaves)[-1][0]
    return out


def linear(rng: random.Random, n: int, k: int) -> list[bytes]:
    return [Wt.row_bytes(bytes(rng.randrange(256) for _ in range(k)), struct.pack("<f", rng.uniform(1e-4, 1.0)))
            for _ in range(n)]


def model(seed: int = 0) -> tuple[list[tuple[int, int]], list[list[bytes]]]:
    rng = random.Random(seed)
    shapes = [(6, 8), (3, 8), (5, 12), (4, 8)]
    return shapes, [linear(rng, n, k) for n, k in shapes]


def committed(shapes, rows, key=KEY):
    """Each K's root as `Registered.commit` takes the rows, with `B_NONCE`'s salts."""
    out = {}
    for i, k in enumerate(Wt.ks(shapes)):
        words = [Wt.row_words(r) for w, (_n, kw) in enumerate(shapes) if kw == k for r in rows[w]]
        out[k] = G.Registered.commit(CIRCUIT, Wt.value(PREFIX, k), words,
                                     [hm96_salt(key, Wt.B_NONCE + i, j) for j in range(len(words))], schema=G.SCHEMA_V2)
    return out


@pytest.mark.parametrize("workers", [1, 3])
def test_roots_and_entries_are_registered_commits(workers):
    shapes, rows = model()
    roots = Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: iter(rows[w]), KEY, workers=workers)
    ref = committed(shapes, rows)
    assert sorted(roots) == [8, 12]
    for k, r in ref.items():
        assert roots[k] == r.root
        assert Wt.entry(CIRCUIT, PREFIX, k, r.leaves, roots[k]) == r.entry()
        assert r.words == Wt.b_words(k)


def test_weights_root_folds_by_ascending_k():
    shapes, rows = model()
    roots = Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w], KEY)
    assert Wt.weights_root(roots) == Wt.weights_root(dict(reversed(list(roots.items()))))
    assert Wt.weights_root(roots) != Wt.weights_root({**roots, 8: roots[12]})


def test_a_changed_code_scale_or_key_moves_the_root():
    shapes, rows = model()
    base = Wt.weights_root(Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w], KEY))
    code = [list(r) for r in rows]
    code[1][2] = bytes([code[1][2][0] ^ 1]) + code[1][2][1:]
    scale = [list(r) for r in rows]
    scale[3][0] = scale[3][0][:-1] + bytes([scale[3][0][-1] ^ 0x80])
    for other, key in ((code, KEY), (scale, KEY), (rows, bytes(32))):
        assert Wt.weights_root(Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: other[w], key)) != base


def test_refuses_rows_of_another_width_or_count():
    shapes, rows = model()
    with pytest.raises(ValueError, match="row 0 is"):
        Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: [r + b"\0\0" for r in rows[w]], KEY)
    with pytest.raises(ValueError, match="gave 5 rows"):
        Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w][:-1], KEY)
    with pytest.raises(ValueError, match="32 bytes"):
        Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w], KEY[:16])
    with pytest.raises(ValueError, match="positive even"):
        Wt.ks([(3, 7)])
    with pytest.raises(ValueError, match="whole u16"):
        Wt.row_bytes(b"\1\2\3", b"\0" * 4)


def random_model(seed: int) -> tuple[list[tuple[int, int]], list[list[bytes]]]:
    rng = random.Random(seed)
    shapes = [(rng.randint(1, 40), rng.choice([2, 4, 6, 8, 12, 16, 62, 256])) for _ in range(rng.randint(1, 6))]
    return shapes, [linear(rng, n, k) for n, k in shapes]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("chunk_bytes", [1, Wt.CHUNK_BYTES])
@pytest.mark.parametrize("workers", [1, 3])
def test_fast_roots_are_the_scalar_references(seed, chunk_bytes, workers, monkeypatch):
    """Random shapes, chunks of 16 rows (`CHUNK_BYTES` 1) or one chunk per tree, in-process or on three processes."""
    monkeypatch.setattr(Wt, "CHUNK_BYTES", chunk_bytes)
    shapes, rows = random_model(seed)
    want = scalar_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w], KEY)
    assert Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w], KEY, workers=workers) == want


@pytest.mark.parametrize("total", [1, 2, 3, 15, 16, 17, 31, 32, 33, 48, 64, 65])
@pytest.mark.parametrize("workers", [1, 2])
def test_edge_row_counts_around_chunks(total, workers, monkeypatch):
    """Σn about a 16-row chunk and a power of two, split over two linears of each K where it can be: partial last chunks,
    all-pad chunks, and a tree of one leaf."""
    monkeypatch.setattr(Wt, "CHUNK_BYTES", 1)
    rng = random.Random(total)
    shapes = [(n, k) for k in (2, 30) for n in ((total,) if total == 1 else (total // 2, total - total // 2))]
    rows = [linear(rng, n, k) for n, k in shapes]
    want = scalar_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w], KEY)
    assert Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w], KEY, workers=workers) == want


@pytest.mark.parametrize("workers", [1, 2])
def test_bytes_like_rows_and_slots_released(workers, monkeypatch):
    """Rows as bytearrays or memoryviews give the bytes' roots; a call's shared slots are gone after it ends, refused
    midway included."""
    monkeypatch.setattr(Wt, "CHUNK_BYTES", 1)
    shapes, rows = random_model(7)
    want = Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w], KEY, workers=1)
    for kind in (bytearray, memoryview):
        assert Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: map(kind, rows[w]), KEY, workers=workers) == want
    assert Wt._SLOTS == {}
    with pytest.raises(ValueError, match="gave"):
        Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w][:-1], KEY, workers=workers)
    assert Wt._SLOTS == {}


def test_chunk_rows_are_a_bounded_power_of_two():
    assert Wt.chunk_bits(2 * Wt.b_words(6144), 30) == 10
    assert Wt.chunk_bits(2 * Wt.b_words(256), 30) == 14
    assert Wt.chunk_bits(2 * Wt.b_words(256), 3) == 3
    assert Wt.chunk_bits(1 << 30, 30) == 4


@pytest.mark.parametrize("nonce", [0, Wt.B_NONCE, (1 << 64) - 1])
@pytest.mark.parametrize("first", [0, (1 << 32) - 2, 1 << 63])
def test_keystream_is_chacha20_block_and_hm96_salt(nonce, first):
    """numpy's keystream is the reference block function, across the counter's 32-bit carry, and leaf j's salt is blocks
    3j .. 3j + 2 of it."""
    key = bytes(range(7, 39))
    assert chacha20.keystream(key, nonce, first, 4) == b"".join(chacha20_block(key, first + i, nonce) for i in range(4))
    j = first // 3
    assert chacha20.keystream(key, nonce, 3 * j, 3) == hm96_salt(key, nonce, j)


@pytest.mark.parametrize("nonce", [0, Wt.B_NONCE, (1 << 64) - 1])
def test_hm96_salts_are_hm96_salt_at_each_row(nonce):
    """numpy's salts at any rows, in their order and repeated, are the reference's row by row, the counter's 32-bit carry
    and its last whole salt included; a row past it is refused."""
    key, last = bytes(range(3, 35)), ((1 << 64) - 3) // 3
    rows = [5, 0, (1 << 32) // 3, 1 << 40, 5, last, 2]
    assert chacha20.hm96_salts(key, nonce, rows) == b"".join(hm96_salt(key, nonce, j) for j in rows)
    assert chacha20.hm96_salts(key, nonce, np.array(rows[:3], dtype=np.int64)) == b"".join(hm96_salt(key, nonce, j) for j in rows[:3])
    assert chacha20.hm96_salts(key, nonce, []) == b""
    for bad in ([last + 1], [-1], [0.5]):
        with pytest.raises(ValueError):
            chacha20.hm96_salts(key, nonce, bad)


def test_progress_and_workers():
    shapes, rows = model()
    seen = []
    Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w], KEY, progress=lambda k, j: seen.append((k, j)), workers=1)
    assert seen == [(8, 0), (12, 0)]
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w], KEY, workers=bad)
    assert 1 <= Wt.DEFAULT_WORKERS <= 4


def test_a_rows_chunks_are_the_least_n_past_4096_whose_chunk_is_whole_512_code_blocks():
    """`chunks` at the served Ks (Llama's 4096 and 14336, GLM's 6144, 12288 and 16384) and every multiple of 32 below
    2^17 against its definition, and `leaf_words` a chunk's codes and the row's scale."""
    assert [Wt.chunks(k) for k in (1, 32, 448, 4096, 6144, 8192, 12288, 14336, 16384)] == [1, 1, 1, 1, 2, 2, 3, 4, 4]
    assert [Wt.leaf_words(k) for k in (64, 4096, 6144, 14336)] == [34, 2050, 1538, 1794]
    for k in range(32, 1 << 17, 32):
        try:
            n = Wt.chunks(k)
        except ValueError:
            assert k > Wt.CHUNK_MAX and all(k % (n * Wt.CHUNK_ALIGN) for n in range(1, k // Wt.CHUNK_ALIGN + 1)), k
            continue
        assert (n == 1) == (k <= Wt.CHUNK_MAX), k
        if n > 1:
            assert k // n <= Wt.CHUNK_MAX and k % (n * Wt.CHUNK_ALIGN) == 0, k
            assert not any(k % (x * Wt.CHUNK_ALIGN) == 0 for x in range(-(-k // Wt.CHUNK_MAX), n)), k
    for bad in (0, -32, 4096 + 32, 5000):
        with pytest.raises(ValueError):
            Wt.chunks(bad)
