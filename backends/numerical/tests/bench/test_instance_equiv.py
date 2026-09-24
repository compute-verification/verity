"""``instance_equiv``'s decoding and document (torch-free; the runner's loaders run on a pod)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from verity_numerical.bench import instance_equiv as ie
from verity_numerical.bench.tables import FROZEN_INSTANCES


def _vus(n: int, k: int, seed: int, fold: int = 1) -> list[tuple]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a, b = rng.integers(0, 256, k).tolist(), rng.integers(0, 256, k).tolist()
        accs = rng.integers(0, 1 << 32, 8, dtype=np.int64).tolist()[fold - 1::fold]
        out.append((a, b, accs, accs[-1]))
    return out


def _doc(fv, cv, dtype="<u1"):
    target = "fp8-ada-mma-draft/2026-09-22"
    frozen = FROZEN_INSTANCES[target]
    cand = dict(frozen, manifest_sha256="0" * 64, seed=20260922, recipe="r")
    return ie.document(target, frozen, cand, ie.arrays(fv, dtype), ie.arrays(cv, dtype), "c", "t")


def test_same_numbers_other_layout_are_equal():
    """A fold keeps every 4th accumulator public: same x, W and chain-end word, so equal."""
    doc = _doc(_vus(6, 32, 1), _vus(6, 32, 1, fold=4))
    assert doc["equal"] is True
    assert set(doc) == {"schema", "target", "frozen", "candidate", "canonical", "arrays", "equal", "tool"}
    assert doc["schema"] == "instance-equiv/v1"
    assert doc["arrays"]["x"]["shape"] == [6, 32] and doc["arrays"]["x"]["dtype"] == "|u1"
    assert doc["arrays"]["y"]["dtype"] == "<u4" and doc["arrays"]["y"]["shape"] == [6]
    assert all(v["frozen_sha256"] == v["candidate_sha256"] for v in doc["arrays"].values())
    json.dumps(doc)


def test_one_word_differs():
    fv, cv = _vus(6, 32, 1), _vus(6, 32, 1)
    cv[3][1][17] ^= 1
    doc = _doc(fv, cv)
    assert doc["equal"] is False
    assert doc["arrays"]["W"]["frozen_sha256"] != doc["arrays"]["W"]["candidate_sha256"]
    assert doc["arrays"]["x"]["frozen_sha256"] == doc["arrays"]["x"]["candidate_sha256"]


def test_bf16_words_little_endian():
    a = ie.arrays([([0x3F80, 0x0001], [0xFF7F, 0], [7], 7)], "<u2")
    assert a["x"].tobytes() == bytes([0x80, 0x3F, 0x01, 0x00])
    with pytest.raises(ValueError):
        ie.arrays([([1 << 16], [0], [0], 0)], "<u2")


def test_same_ref_compares_the_tables_keys():
    f = FROZEN_INSTANCES["fp8-ada-mma-draft/2026-09-22"]
    assert ie.same_ref(dict(f, seed=1, recipe="x"), f)
    assert not ie.same_ref(dict(f, range=[0, 4095]), f)
