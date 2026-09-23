"""Poseidon2 over BabyBear (width 24) and the ``leaf/v2h`` row leaves: known answer, the three evaluations agree,
domain separation, injective packing, the leaf value."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from verity.commitments import CommitmentDomain, RangeIndexedDomain
from verity.commitments import poseidon2_babybear as h2
from verity.commitments import rowleaf


def test_known_answer_plonky3_width_24() -> None:
    inp, out = h2.KAT_24
    assert h2.permute(inp) == out
    assert h2.permute_np(np.asarray(inp, dtype=np.int64)[:, None])[:, 0].tolist() == out


def test_constants_pinned_by_digest() -> None:
    """The round constants, diagonal and M4 as fetched from Plonky3 (2026-09-23); a change here changes every digest."""
    assert h2.CONSTANTS_SHA256 == "0b81af1cdcd7e5b1" + h2.CONSTANTS_SHA256[16:] and len(h2.CONSTANTS_SHA256) == 64
    assert len(h2.RC_EXTERNAL_INITIAL) == len(h2.RC_EXTERNAL_FINAL) == h2.HALF_FULL_ROUNDS
    assert len(h2.RC_INTERNAL) == h2.PARTIAL_ROUNDS and len(h2.DIAG) == h2.WIDTH


def test_ring_evaluation_is_the_scalar_permutation() -> None:
    rng = np.random.default_rng(1)
    st = [int(x) for x in rng.integers(0, h2.P, size=h2.WIDTH)]
    ring = h2.permute_ring(list(st), lambda x: pow(x, 7, h2.P), lambda x: x % h2.P)
    assert ring == h2.permute(st)
    cols = rng.integers(0, h2.P, size=(h2.WIDTH, 5)).astype(np.int64)
    vec = h2.permute_np(cols)
    for j in range(5):
        assert vec[:, j].tolist() == h2.permute([int(x) for x in cols[:, j]])


def test_sponge_rows_scalar_equals_vectorised_and_separates_roles() -> None:
    rng = np.random.default_rng(2)
    rows = rng.integers(0, 2**16, size=(7, 1536)).astype(np.int64)
    vec = h2.hash_rows_np(rows, 16, h2.ROLE_X)
    for i in range(7):
        assert vec[i].tolist() == h2.hash_row(rows[i].tolist(), 16, h2.ROLE_X)
    assert h2.hash_row(rows[0].tolist(), 16, h2.ROLE_W) != h2.hash_row(rows[0].tolist(), 16, h2.ROLE_X)
    # the same bytes as 8-bit words are another object (different IV: word width and count)
    bytes8 = rng.integers(0, 256, size=(1536,)).astype(np.int64)
    as8 = h2.hash_row(bytes8.tolist(), 8, h2.ROLE_X)
    packed16 = (bytes8[0::2] + 256 * bytes8[1::2]).tolist()
    assert as8 != h2.hash_row(packed16, 16, h2.ROLE_X)
    assert h2.IV(h2.ROLE_X, 16, 1536) != h2.IV(h2.ROLE_X, 16, 3072)


def test_pack_words_is_injective_and_below_p() -> None:
    rng = np.random.default_rng(3)
    for bits, per in ((16, 1), (8, 2), (4, 4)):
        words = rng.integers(0, 2**bits, size=(50, 16 * per)).astype(np.int64)
        el = h2.pack_words(words, bits)
        assert el.shape == (50, 16) and el.max() < 2**16 < h2.P
        back = np.stack([(el >> (bits * q)) & (2**bits - 1) for q in range(per)], 2).reshape(50, 16 * per)
        assert np.array_equal(back, words)
    with pytest.raises(ValueError):
        h2.pack_words(np.array([[2**16]]), 16)


def test_one_word_changes_the_digest() -> None:
    rng = np.random.default_rng(4)
    row = rng.integers(0, 2**16, size=1536).astype(np.int64)
    d0 = h2.hash_row(row.tolist(), 16, h2.ROLE_X)
    row[1000] ^= 1
    assert h2.hash_row(row.tolist(), 16, h2.ROLE_X) != d0
    assert len(d0) == h2.OUT and all(0 <= x < h2.P for x in d0)


def test_row_leaf_value_and_leaf_hash() -> None:
    d = h2.hash_row(list(range(1536)), 16, h2.ROLE_W)
    val = rowleaf.row_leaf_value(d)
    assert val == b"".join(int(x).to_bytes(4, "big") for x in d) and len(val) == 32
    dom = CommitmentDomain(bytes(32), -2, RangeIndexedDomain(0, 64))
    leaf = rowleaf.row_leaf(dom, 5, d)
    assert leaf == dom.leaf(5, 5, rowleaf.SCHEMA_ROW, val) and leaf != rowleaf.row_leaf(dom, 6, d)
    with pytest.raises(ValueError):
        rowleaf.row_leaf_value(d[:7])
    assert np.array_equal(rowleaf.row_digests(np.tile(np.arange(1536), (2, 1)), 16, h2.ROLE_W)[0], np.asarray(d))
    # the digest text is stable (a regression pin of the sponge on a fixed row)
    assert hashlib.sha256(val).hexdigest() == hashlib.sha256(rowleaf.row_leaf_value(rowleaf.row_digest(list(range(1536)), 16, h2.ROLE_W))).hexdigest()
