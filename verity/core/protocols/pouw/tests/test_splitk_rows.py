"""An H1 table's leaves past K = 4096 (`h1zy.chunk_leaves`, `h1zy.chunk_words`): row r's chunk c is leaf n r + c, its
chunk's codes and then the row's scale again, and the pinned vectors (``splitk_rows_vectors``) that the GPU trees of
``a/<K>@<s>`` and ``b/<K>`` are gated against."""
from __future__ import annotations

import json

import pytest

import splitk_rows_vectors as V
from verity.core.protocols.pouw import weights as Wt
from verity.core.protocols.pouw.circuit import h1zy as Z


def test_the_pinned_vectors_are_the_chunk_rows():
    """``vectors/splitk_rows.json`` is `splitk_rows_vectors.vectors()`: ``a/14336@0`` over 64 rows (n = 4, 256 leaves of
    1794 words, binding the window) and ``b/6144`` over 64 rows (n = 2, 128 leaves of 1538 words, binding none), every
    leaf's words' SHA-512, row digest and frame leaf, and each value's domain and root."""
    got = V.vectors()
    assert json.loads(V.OUT.read_text()) == got
    shapes = [(c["value"], c["n"], c["leaves"], c["words"], c["window"]) for c in got["cases"]]
    assert shapes == [("a/14336@0", 4, 256, 1794, True), ("b/6144", 2, 128, 1538, False)]
    assert all(c["words"] == Wt.b_words(c["k_c"]) and len(c["leaf"]) == c["leaves"] for c in got["cases"])


@pytest.mark.parametrize("k", [64, 4096, 5120, 6144, 14336])
def test_a_rows_chunks_are_its_codes_in_turn_each_with_its_scale(k):
    row = V.rows(f"layout-{k}", k, 1)[0]
    n, leaves = Z.chunks(k), Z.chunk_leaves(row, k)
    kc = k // n
    assert len(leaves) == n and all(len(leaf) == kc + 4 and leaf[kc:] == row[k:] for leaf in leaves)
    assert b"".join(leaf[:kc] for leaf in leaves) == row[:k]
    assert (leaves == [row]) == (k <= Z.CHUNK_MAX)


def test_chunk_words_put_row_r_chunk_c_at_leaf_n_r_plus_c():
    k, rows = 14336, V.rows("order", 14336, 3)
    words = Z.chunk_words(rows, k)
    assert len(words) == 12 and all(len(w) == Wt.b_words(k // 4) for w in words)
    for r, row in enumerate(rows):
        for c, leaf in enumerate(Z.chunk_leaves(row, k)):
            assert words[4 * r + c] == Wt.row_words(leaf)


@pytest.mark.parametrize("k, size", [(6144, 6147), (6144, 6149), (6143, 6147)])
def test_a_row_of_another_width_or_an_odd_k_is_refused(k, size):
    with pytest.raises(ValueError, match="its K codes and its scale's 4 bytes, whole u16 words"):
        Z.chunk_leaves(bytes(size), k)
