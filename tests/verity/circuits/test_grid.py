from __future__ import annotations

import pytest

from verity.circuits import codec
from verity.circuits.refs import Affine, Explicit, Grid, Strided, grid_ref, lane_bounds, refs_from_json, spell

KEYS = {"grid": [64, [["n", 2091, 0, 1, [[4, 0], [32, 64]]]]]}


def test_the_attention_keys_argument_is_four_members_a_rope_row_then_the_next():
    g = refs_from_json(KEYS)
    assert isinstance(g, Grid) and (g.run, g.periods, g.count) == (64, 128, 8192) and g.to_json() == KEYS
    assert list(g) == [("n", 2091, k // 256 * 64 + k % 64) for k in range(8192)]
    assert lane_bounds(g.run, g.lanes[0]) == (0, 2047)
    assert codec._checked_refs(g, lambda s, i: 2048) is g
    with pytest.raises(ValueError, match="grid-overrun"):
        codec._checked_refs(g, lambda s, i: 2047)


def test_reference_k_is_its_lane_its_periods_digits_and_its_offset():
    lanes = (("n", 3, 10, 1, ((2, 5), (3, 100))), ("p", 0, 7, -1, ((6, 2),)))
    want = []
    for t in range(6):
        for j, (s, i, b, e, lv) in enumerate(lanes):
            digits = [t % 2, t // 2] if j == 0 else [t]
            want += [(s, i, b + sum(d * st for d, (_r, st) in zip(digits, lv)) + o * e) for o in range(3)]
    assert [grid_ref(3, lanes, k)[1] for k in range(36)] == want == list(Grid(3, lanes))
    assert lane_bounds(3, lanes[0]) == (10, 10 + 5 + 200 + 2) and lane_bounds(3, lanes[1]) == (5, 17)


def test_a_grid_an_older_form_spells_is_spelled_by_it():
    assert spell(4, [("n", 1, 8, 1, ())]).to_json() == {"affine": ["n", 1, 8, 4]}
    assert spell(4, [("n", 1, 8, 2, ())]).to_json() == {"strided": ["n", 1, 8, 4, 0, 2, 4]}
    assert spell(4, [("n", 1, 8, 1, ((3, 10),))]).to_json() == {"strided": ["n", 1, 8, 4, 10, 1, 12]}
    assert spell(4, [("n", 1, 8, 1, ((3, 4),))]).to_json() == {"affine": ["n", 1, 8, 12]}
    assert spell(1, [("n", 1, 8, 1, ((2, 0), (3, 64)))]).to_json() == {"strided": ["n", 1, 8, 2, 64, 0, 6]}
    assert spell(2, [("n", 1, 0, 1, ((3, 2), (2, 6)))]).to_json() == {"affine": ["n", 1, 0, 12]}
    assert spell(1, [("n", 1, 5, 1, ()), ("p", 0, 2, 1, ())]).to_json() == {"explicit": [["n", 1, 5], ["p", 0, 2]]}
    assert spell(2, [("n", 1, 0, 1, ((3, 10),)), ("n", 1, 2, 1, ((3, 10),))]).to_json() == {"strided": ["n", 1, 0, 4, 10, 1, 12]}
    two = [("n", 1, 0, 1, ((3, 10),)), ("n", 2, 0, 1, ((3, 10),))]
    assert isinstance(spell(2, two), Grid) and list(spell(2, two)) == list(Grid(2, tuple(two)))
    for g in (spell(4, [("n", 1, 8, 1, ((3, 10),))]), spell(1, [("n", 1, 8, 1, ((2, 0), (3, 64)))])):
        assert isinstance(g, (Affine, Strided, Explicit))


@pytest.mark.parametrize("j,name", [
    ([0, [["n", 1, 0, 1, []]]], "grid-shape"),
    ([2, []], "grid-shape"),
    ([2, [["n", 1, 0, 1, [[1, 4]]]]], "grid-shape"),
    ([2, [["n", 1, 0, 1, [[2, 4]]], ["n", 2, 0, 1, [[3, 4]]]]], "grid-shape"),
    ([2, [["q", 1, 0, 1, []], ["n", 2, 0, 1, []]]], "grid-shape"),
    ([2, [["n", 1, 0, 1, [[3, 10]]]]], "grid-spelled"),
    ([1, [["n", 1, 0, 1, []], ["n", 2, 0, 1, []]]], "grid-spelled"),
    ([2, [["n", 1, 0, 1, [[3, 10]]], ["n", 1, 2, 1, [[3, 10]]]]], "grid-spelled"),
])
def test_the_decoder_refuses_a_malformed_grid_and_one_an_older_form_spells_by_name(j, name):
    with pytest.raises(ValueError, match=name):
        refs_from_json({"grid": j})
