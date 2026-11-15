"""A sample's place in its window's session (`window_place`): the place, the family and the run of a record's
window, held to a table of windows (`PLACES`), refusals among them."""
from __future__ import annotations

import pytest

from verity.protocols.proofs.spot.verifier import window_place as WP

#: `window_place.of`'s rows, a record's window and its place, None for a window that has none
PLACES = [
    ({"prompt": 4, "outputs": 4}, 68719476752),
    ({"prompt": 4, "outputs": 4, "stopped": False}, 68719476752),
    ({"prompt": 4, "outputs": 4, "stopped": True}, 68719476754),
    ({"prompt": 4, "outputs": 4, "stopped": True, "population": "stop"}, 68719476755),
    ({"prompt": 4, "outputs": 2}, 68719476744),
    ({"prompt": 2, "outputs": 8}, 34359738400),
    ({"prompt": 8, "outputs": 2}, 137438953480),
    ({"prompt": 0, "outputs": 0}, 0),
    ({"prompt": 0, "outputs": 1}, 4),
    ({"prompt": 1, "outputs": 0}, 17179869184),
    ({"prompt": 2**32 - 1, "outputs": 2**32 - 1, "stopped": True, "population": "stop"}, 2**66 - 1),
    ({"prompt": 2**32, "outputs": 4}, None),
    ({"prompt": 4, "outputs": 2**32}, None),
    ({"prompt": 4, "outputs": -1}, None),
    ({"prompt": "4", "outputs": 4}, None),
    ({"prompt": 4.0, "outputs": 4}, None),
    ({"prompt": True, "outputs": 4}, None),
    ({"prompt": None, "outputs": 4}, None),
    ({"outputs": 4}, None),
    ({"prompt": 4}, None),
    ({"prompt": 4, "outputs": 4, "stopped": 1}, None),
    ({"prompt": 4, "outputs": 4, "stopped": "true"}, None),
    ({"prompt": 4, "outputs": 4, "stopped": None}, None),
    ({"prompt": 4, "outputs": 4, "population": "stop"}, None),
    ({"prompt": 4, "outputs": 4, "stopped": False, "population": "stop"}, None),
    ({"prompt": 4, "outputs": 4, "stopped": True, "population": "spans"}, None),
    ({"prompt": 4, "outputs": 4, "stopped": True, "population": None}, None),
    ({"prompt": 4, "outputs": 4, "stopped": True, "population": 1}, None),
    ({"prompt": 4, "outputs": 4, "stopped": True, "population": ["stop"]}, None),
    ({}, None),
    (None, None),
    ([4, 4], None),
    ("window", None),
]

#: items a window holds beside the four a place reads, which move no place
MORE = {"kind": "served", "run": "ab" * 64, "row": 3, "check": "c", "stop": {"ids": [7], "outputs": 4}, "verdicts": True}


def test_a_windows_place_is_its_four_items_alone():
    """`of` is ((prompt · 2^32 + outputs) · 2 + stopped) · 2 + blocks over four items of a record's window: its
    ``prompt``, its ``outputs``, ``stopped`` true and, on a stop record, ``population`` "stop"; no other item moves
    it. The family is `FAMILY`, a slash and the place in decimal. Refused, by `of` and `family` alike: a window that is
    no object, a size that is no natural number below 2^32, a ``stopped`` that is neither true nor false, a
    ``population`` other than "stop", and the stop blocks of a class that is no stop class."""
    assert WP.SIZES == 2**32 and WP.FAMILY == "draw"
    for window, at in PLACES:
        both = (window, {**window, **MORE}) if isinstance(window, dict) else (window,)
        for w in both:
            if at is None:
                for f in (WP.of, WP.family):
                    with pytest.raises(ValueError, match="no place in a window's session|is no stop class"):
                        f(w)
            else:
                assert WP.of(w) == at and WP.family(w) == f"draw/{at}", window
    placed = [at for _, at in PLACES if at is not None]
    assert len(placed) == 11 and len(set(placed)) == 10 and len(PLACES) - len(placed) == 22


def test_a_windows_place_keeps_classes_and_populations_apart():
    """Two windows that differ in a size, in being a stop class's or in being a stop record's have two places, and a
    stop class's stop blocks have the place after its spans'."""
    windows = [{"prompt": p, "outputs": o, **more} for p in (0, 1, 4, 2**32 - 1) for o in (0, 1, 4, 2**32 - 1)
               for more in ({}, {"stopped": True}, {"stopped": True, "population": "stop"})]
    places = [WP.of(w) for w in windows]
    assert len(set(places)) == len(places) == 48
    for w in windows:
        if w.get("stopped") and "population" not in w:
            assert WP.of({**w, "population": "stop"}) == WP.of(w) + 1 == WP.of({**w, "stopped": False}) + 3


def test_a_windows_run_is_its_run_where_that_is_a_string():
    assert WP.run({"run": "ab", "prompt": 4}) == "ab" and WP.run({"run": ""}) == ""
    for window in ({"run": 7}, {"run": None}, {"run": ["ab"]}, {}, None, ["run"], "run"):
        assert WP.run(window) is None
