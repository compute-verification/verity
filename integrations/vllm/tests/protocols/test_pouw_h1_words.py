"""`words.chunk_words` and `row_words` write each row's bytes once into the table: the same words, dtype and shape as the
rows split, each scale repeated, concatenated and cast, and as `Served.a_words`' rows padded to the Program's M first."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from verity.core.protocols.pouw import weights as Wt

from verity_vllm.protocols.pouw.h1 import served as SV
from verity_vllm.protocols.pouw.h1 import words as W


def _row_words(codes, scales):
    b = np.ascontiguousarray(np.asarray(codes, dtype=np.int8).view(np.uint8))
    b = b.reshape(len(b), -1)
    s = np.ascontiguousarray(np.asarray(scales, "<f4").reshape(len(b))).view(np.uint8).reshape(len(b), 4)
    return np.ascontiguousarray(np.concatenate([b, s], axis=1)).view("<u2").astype(np.uint16)


def _chunk_words(codes, scales, n):
    c = np.ascontiguousarray(np.asarray(codes, dtype=np.int8))
    r, k = c.shape
    return _row_words(c.reshape(n * r, k // n), np.repeat(np.asarray(scales, "<f4").reshape(r), n))


def _padded(codes, scales, n, rows):
    full, sc = np.zeros((rows, codes.shape[1]), np.int8), np.zeros(rows, "<f4")
    full[:len(codes)], sc[:len(codes)] = codes, scales
    return _chunk_words(full, sc, n)


def _same(got, want):
    assert got.dtype == want.dtype == np.uint16 and got.shape == want.shape and np.array_equal(got, want)


@pytest.mark.parametrize("r,k,n", [(1, 2, 1), (5, 16, 1), (5, 16, 2), (7, 64, 4), (33, 4096, 2), (3, 14336, 7)])
def test_the_words_are_the_split_repeated_and_concatenated_rows(r, k, n):
    rng = np.random.default_rng(r * k + n)
    codes = rng.integers(-128, 128, (r, k), dtype=np.int8)
    scales = (rng.standard_normal(r) * 3).astype(np.float64)
    _same(W.chunk_words(codes, scales, n), _chunk_words(codes, scales, n))
    _same(W.chunk_words(codes.view(np.uint8), scales.reshape(r, 1), n), _chunk_words(codes.view(np.uint8), scales, n))
    if n == 1:
        _same(W.row_words(codes, scales), _row_words(codes, scales))
    wide = np.zeros((r, 2 * k), np.int8)
    wide[:, ::2] = codes
    _same(W.chunk_words(wide[:, ::2], scales, n), _chunk_words(codes, scales, n))
    for rows in (r, r + 1, 3 * r + 5):
        _same(W.chunk_words(codes, scales, n, rows=rows), _padded(codes, scales, n, rows))
    _same(W.chunk_words(codes[:0], scales[:0], n, rows=4), _padded(codes[:0], scales[:0], n, 4))


def test_a_words_pads_to_the_programs_m():
    rng = np.random.default_rng(7)
    codes, scales = rng.integers(-128, 128, (5, 4096), dtype=np.int8), rng.random(5).astype("<f4")
    served = SimpleNamespace(m=12)
    call = SimpleNamespace(k=4096, index=3)
    _same(SV.Served.a_words(served, call, codes, scales), _padded(codes, scales, Wt.chunks(4096), 12))
    with pytest.raises(ValueError, match="K = 2048, not 4096"):
        SV.Served.a_words(served, call, codes[:, :2048], scales)


def test_refusals():
    codes, scales = np.zeros((4, 6), np.int8), np.ones(4, "<f4")
    with pytest.raises(ValueError, match="K = 6 is not 2 chunks"):
        W.chunk_words(codes, scales, 2)
    with pytest.raises(ValueError, match="4 rows past the table's 3"):
        W.chunk_words(codes, scales, 1, rows=3)
    with pytest.raises(ValueError, match="K = 5 is not 1 chunks"):
        W.row_words(np.zeros((2, 5), np.int8), np.ones(2))
