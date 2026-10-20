"""RecCap (`verity_flock.rec_cap`), V*'s hold of the table's level-0 cap: on a held row (64 words) it outputs the words past
the node's 32, then zeros, so its outputs, which V* registers as zero rows, are all zero iff the row is a node's (`rec_live.
top_rows`: the node's 32 words, then 32 zero words). The unit through its M0 layout, and the Definition evaluated from its
descriptor, are that reference; the unit has no AND; and staging refuses tops that are not level 0's one cap. The Lean
verifier's side is `Flock.Test.RecCap` and `test_lean_rec_coins`."""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
from verity_flock import rec_cap as RC


def _rows(rng: random.Random) -> list[list[int]]:
    """A node's row (32 words, then zeros), a row one bit off past its node, and a row of random words."""
    node = [rng.randrange(1 << 16) for _ in range(RC.NODE_WORDS)] + [0] * (RC.TOP_WORDS - RC.NODE_WORDS)
    off = list(node)
    off[RC.NODE_WORDS + 7] ^= 1 << 3
    return [node, off, [rng.randrange(1 << 16) for _ in range(RC.TOP_WORDS)]]


def test_the_reference_is_zero_exactly_on_a_nodes_row():
    node, off, rand = _rows(random.Random(1))
    assert RC.ref(node) == [0] * RC.OUT_WORDS
    assert RC.ref(off) == [0] * 7 + [1 << 3] + [0] * (RC.OUT_WORDS - 8)
    assert RC.ref(rand)[:RC.TOP_WORDS - RC.NODE_WORDS] == rand[RC.NODE_WORDS:]
    with pytest.raises(ValueError, match="63 words"):
        RC.ref(node[:-1])


def test_the_unit_is_the_reference_with_no_and():
    low = RC.lowering(128)
    assert low.name == RC.unit_name(128) == "rec-cap/n128"
    assert not any(k == "and" for k in low.circuit.kind)
    assert low.layout.useful == 2049
    rows = _rows(random.Random(2))
    got, _cut, ok = low.evaluate(np.array(rows, np.uint64))
    assert int(ok) == 0b111
    assert [got[i].astype(np.uint16).tolist() for i in range(3)] == [RC.ref(r) for r in rows]


def test_the_definition_is_the_reference():
    from verity.ml.circuits.evaluation import evaluate_batch

    rows = _rows(random.Random(3))
    bits = np.array([[[(w >> j) & 1 for j in range(16)] for w in r] for r in rows], dtype=np.uint8)
    res = np.asarray(evaluate_batch(RC.definition(), bits)).reshape(3, RC.OUT_WORDS, 16)
    word = lambda w: int(sum(int(b) << j for j, b in enumerate(w)))  # noqa: E731
    assert RC.definition().id == "RecCap_v1"
    assert [[word(w) for w in res[t]] for t in range(3)] == [RC.ref(r) for r in rows]


def test_staging_refuses_tops_that_are_not_level_0s_one_cap(tmp_path):
    for own, first in (({"value": "rec-top-L1", "leaves": 128}, {0: 0, 1: 0}), ({"value": RC.TOP_VALUE, "leaves": 128},
                                                                               {0: 0, 1: 128})):
        with pytest.raises(ValueError, match="RecCap holds level 0's one cap rec-top-L0"):
            RC.stage(SimpleNamespace(own=own, first=first, reg=None), tmp_path / "cap")
