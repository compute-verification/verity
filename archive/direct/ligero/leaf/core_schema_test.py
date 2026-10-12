"""B-Ligero's ``leaf = blake3`` is the core ``blake3-keyed/row/v2`` schema byte for byte: its native chunk-CV fold lands on
every full-row digest of ``verity.ml.commitments.frame_v3``'s vectors, with the same keys and schema string."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.commitments.frame_v3 import vectors as V
from verity.ml.commitments.rowleaf import SCHEMA_BLAKE3_ROW, blake3_row_key

from . import blake3 as L

ROWS = [r for r in V.load()["row_leaves"] if r["hash"] == "blake3" and r["n_words"] == L.K_VU]


def test_schema_and_keys_are_the_core_ones() -> None:
    assert L.SCHEMA == SCHEMA_BLAKE3_ROW
    assert all(L.key_for(role) == blake3_row_key(role) for role in (1, 2))


@pytest.mark.parametrize("case", ROWS, ids=lambda c: f"role{c['role']}-{c['word_bits']}b")
def test_native_fold_reproduces_the_core_vectors(case: dict) -> None:
    raw = bytes.fromhex(case["row_bytes"])
    words = np.frombuffer(raw, dtype="<u1" if case["word_bits"] == 8 else "<u2").astype(np.int64)[None, :]
    leaf = L.Blake3Leaf()
    assert leaf.leaf_bytes(leaf.native(words, word_bits=case["word_bits"], role=case["role"])[0]).hex() == case["digest"]
