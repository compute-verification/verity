"""`rows.registered` against `registered_value.Registered.commit`: over rows of 16-bit words, one row to a few dozen, one
word to a z/y record's 3072, both bit-row schemas and each domain binding, under the same salts: the entry, levels,
``b || c`` strings, salts, rows, paths and statement are the reference's."""
from __future__ import annotations

import os

import numpy as np
import pytest
from verity.commitments import registered_value as RV
from verity.commitments.periods import rows as CS
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2

PROGRAM, WINDOW = "ab" * 64, "cd" * 64


def _ref(words: np.ndarray, ys: bytes, schema: str, **binding) -> RV.Registered:
    n = len(words)
    return RV.Registered.commit(PROGRAM, "v", words.tolist(), [ys[192 * j:192 * (j + 1)] for j in range(n)], schema=schema,
                                **binding)


@pytest.mark.parametrize("n,w,schema,binding", [
    (1, 1, V2, {}), (2, 2, V2, {"window": WINDOW}), (3, 66, V2, {}), (5, 1026, V2, {"window": WINDOW}),
    (8, 3, V2, {"produced": "ef" * 64}), (4, 5, V2, {"session": "s-1"}), (33, 3072, V2, {"window": WINDOW}),
    (1, 1, SEG, {}), (3, 511, SEG, {}), (2, 512, SEG, {"window": WINDOW}), (5, 513, SEG, {}), (3, 2100, SEG, {})])
def test_a_value_is_what_the_reference_commits(n, w, schema, binding):
    rng = np.random.default_rng(n * 1000 + w)
    words = rng.integers(0, 1 << 16, (n, w), dtype=np.uint16)
    ys = os.urandom(192 * n)
    ref, got = _ref(words, ys, schema, **binding), CS.registered(PROGRAM, "v", words, ys, keep_rows=True, schema=schema,
                                                                   **binding)
    assert got.entry() == ref.entry() and got.root == ref.root and got.domain == ref.domain
    assert got.levels == ref.levels and got.bcs == ref.bcs and got.schema == schema
    assert list(got.salts) == list(ref.salts) and list(got.rows) == list(ref.rows)
    assert [got.path(p) for p in range(n)] == [ref.path(p) for p in range(n)]
    ps = [n - 1, 0, n // 2]
    assert got.statement(ps) == ref.statement(ps)
    assert got.salts.at(ps) == b"".join(ref.statement(ps)[1])


def test_rows_not_kept_are_refused_and_a_reused_salt_is_refused():
    words = np.arange(12, dtype=np.uint16).reshape(3, 4)
    ys = os.urandom(3 * 192)
    got = CS.registered(PROGRAM, "v", words, ys, window=WINDOW)
    assert got.levels == _ref(words, ys, V2, window=WINDOW).levels and len(got.rows) == 3
    with pytest.raises(LookupError, match="rebuilds"):
        got.rows[0]
    for bad in (ys[:-1], ys[:192] * 3):
        with pytest.raises(ValueError, match="never reused"):
            CS.registered(PROGRAM, "v", words, bad)


@pytest.mark.parametrize("words", [np.zeros((0, 4), np.uint16), np.zeros((2, 0), np.uint16), np.zeros((2, 3), np.int32),
                                   np.zeros(4, np.uint16)])
def test_a_value_of_no_rows_or_not_u16_words_is_refused(words):
    with pytest.raises(ValueError, match="16-bit words"):
        CS.registered(PROGRAM, "v", words, os.urandom(192 * max(1, words.shape[0])))
