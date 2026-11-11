"""`engine.makers`: the parts plugins add to one seam, made once per object in the seam's declared order."""
from __future__ import annotations

import pytest

from verity_vllm.engine.makers import Makers


class Obj:
    pass


def test_each_object_is_made_its_parts_once_in_the_declared_order_without_the_nones():
    made: list = []
    m: Makers[str] = Makers(("first", "second", "third"), "a seam's NAMES")
    m.add("third", lambda obj: made.append("third") or "c")
    m.add("first", lambda obj: made.append("first") or "a")
    m.add("second", lambda obj: made.append("second"))
    a, b = Obj(), Obj()
    assert m.added == ("first", "second", "third")
    assert m.made(a) == ("a", "c") and m.made(a) is m.made(a)
    assert m.made(b) == ("a", "c")
    assert made == ["first", "second", "third"] * 2


def test_a_part_is_refused_without_its_place_twice_or_once_an_object_made_its_parts():
    m: Makers[str] = Makers(("first",), "a seam's NAMES")
    with pytest.raises(ValueError, match="give it its place in a seam's NAMES"):
        m.add("other", lambda obj: None)
    m.add("first", lambda obj: None)
    with pytest.raises(ValueError, match="already added to a seam's NAMES"):
        m.add("first", lambda obj: None)
    m = Makers(("first", "second"), "a seam's NAMES")
    m.add("first", lambda obj: "a")
    obj = Obj()
    m.made(obj)
    with pytest.raises(RuntimeError, match="before vLLM builds the engine"):
        m.add("second", lambda obj: None)
    m.clear()
    m.add("second", lambda obj: "b")
    assert m.made(obj) == ("b",)


def test_a_maker_that_raises_makes_nothing_and_is_asked_again():
    calls: list = []

    def make(obj: Obj) -> str:
        calls.append(obj)
        if len(calls) == 1:
            raise RuntimeError("not yet")
        return "a"

    m: Makers[str] = Makers(("first",), "a seam's NAMES")
    m.add("first", make)
    obj = Obj()
    with pytest.raises(RuntimeError, match="not yet"):
        m.made(obj)
    assert m.made(obj) == ("a",) and len(calls) == 2
