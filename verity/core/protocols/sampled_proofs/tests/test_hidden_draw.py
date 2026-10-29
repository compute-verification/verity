"""The sample over hidden positions (`one_stage.hidden_draw`): the honest table's positions are its live (call, row, tile)
triples one to one, and on any table, overlapping, out of order or with gaps, no two positions share a witness and each
witness is in range."""

import random

import pytest

from verity.core.protocols.sampled_proofs.one_stage import hidden_draw as HD


@pytest.mark.parametrize("width, rows", [(1, [3, 0, 2]), (4, [2, 0, 1, 5]), (24, [0, 0]), (3, [])])
def test_the_honest_table_is_its_triples_one_to_one(width, rows):
    calls = HD.table(width, rows)
    m = HD.total(width, rows)
    located = [HD.locate(width, calls, r) for r in range(m)]
    triples = [(c, j, t) for c, n in enumerate(rows) for j in range(n) for t in range(width)]
    assert located == triples
    assert HD.locate(width, calls, m) is None
    assert all(HD.witness(width, calls, w[0], r) == w for r, w in enumerate(located))


def _table(rng: random.Random) -> tuple[int, list[HD.Call]]:
    width = rng.randint(1, 4)
    return width, [HD.Call(rng.randint(0, 12), rng.randint(0, 3)) for _ in range(rng.randint(0, 5))]


@pytest.mark.parametrize("seed", range(40))
def test_no_two_positions_share_a_witness_on_any_table(seed):
    rng = random.Random(seed)
    width, calls = _table(rng)
    seen: dict[HD.Witness, int] = {}
    for c in range(-1, len(calls) + 1):
        for r in range(30):
            w = HD.witness(width, calls, c, r)
            if w is None:
                continue
            assert seen.setdefault(w, r) == r, (calls, w)
            assert w[0] == c and 0 <= w[1] < calls[c].rows and 0 <= w[2] < width
            assert calls[c].offset + w[1] * width + w[2] == r
        assert all(HD.locate(width, calls, r) in (None, *[HD.witness(width, calls, k, r) for k in range(len(calls))])
                   for r in range(30))


def test_an_overlapping_table_proves_fewer_positions():
    """Two calls over the same positions: each position has two witnesses, and the positions proved are at most the
    honest table's."""
    width, calls = 2, [HD.Call(0, 2), HD.Call(0, 2), HD.Call(9, 1)]
    proved = {r for r in range(20) for c in range(3) if HD.witness(width, calls, c, r) is not None}
    assert proved == {0, 1, 2, 3, 9, 10}
    assert len(proved) <= HD.total(width, [c.rows for c in calls])


@pytest.mark.parametrize("width", [0, -1, True, 1.0])
def test_a_width_not_positive_is_refused(width):
    with pytest.raises(ValueError):
        HD.offsets(width, [1])
