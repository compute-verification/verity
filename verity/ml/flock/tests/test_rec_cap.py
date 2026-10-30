"""RecCap (`verity_flock.rec_cap`), V*'s hold of each level's committed tops: on a held row (64 words) it outputs the words
past the node's 32, then zeros, so its outputs, which V* registers as zero rows, are all zero iff the row is a node's
(`rec_live.top_rows`: the node's 32 words, then 32 zero words). The unit through its M0 layout, and the Definition evaluated
from its descriptor, are that reference; the unit has no AND; its name and directory say its level; and staging refuses
tops that are not a level's (level 0's one cap, or a later level's rep 0 then rep 1). The Lean verifier's side is
`Flock.Test.RecCapSmall` (one of Flock's audit roots), `Flock.Test.RecCap` (built after landing by
`test_lean_rec_cap_guards`) and `test_lean_rec_coins`."""
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


def test_a_later_levels_cap_is_named_and_placed_by_its_level():
    assert RC.unit_name(128, 0) == "rec-cap/n128" and RC.unit_name(64, 3) == "rec-cap/n64-l3"
    assert RC.lowering(64, 3).name == RC.subcircuit(64, 3).id == "rec-cap/n64-l3"
    assert (RC.dir_name(0), RC.dir_name(3)) == ("cap", "cap/L3")
    assert (RC.level_of("rec-top-L0"), RC.level_of("rec-top-L5")) == (0, 5)
    for value in ("rec-top-stacked-L1", "rec-top-L", "rec-cap-zero"):
        with pytest.raises(ValueError, match="RecCap holds a level's tops rec-top-L<l>"):
            RC.level_of(value)


def test_staging_refuses_tops_that_are_not_a_levels(tmp_path):
    with pytest.raises(ValueError, match="RecCap at level 0 holds the reps' one cap rec-top-L0"):
        RC.stage(SimpleNamespace(own={"value": RC.TOP_VALUE, "leaves": 128}, first={0: 0, 1: 128}, reg=None), tmp_path / "cap")
    for first in ({0: 0, 1: 0}, {0: 0, 1: 128}, {0: 0}):
        with pytest.raises(ValueError, match="RecCap at level 1 holds rep 0's top then rep 1's"):
            RC.stage(SimpleNamespace(own={"value": "rec-top-L1", "leaves": 128}, first=first, reg=None), tmp_path / "cap")
    with pytest.raises(ValueError, match="RecCap holds a level's tops"):
        RC.stage(SimpleNamespace(own={"value": "rec-top-stacked-L1", "leaves": 128}, first={0: 0}, reg=None), tmp_path / "c")
