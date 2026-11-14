"""`ir_lower._boolean_build` picks a word primitive's builder independently of the order its Boolean Definitions were
registered in: two different builders for one word are refused by name in either order (`AmbiguousWord`, raised by
``key in PIECES`` too), one builder registered under two ids is that builder, and an ``opt_in`` candidate is never picked."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from verity_flock import ir_lower as IL

from verity.circuits.defs import REGISTRY

WORD = SimpleNamespace(id="BooleanBuildTestWord_v1")
A, B = "BooleanBuildTestA_v1", "BooleanBuildTestB_v1"


def one(C, a):
    return a


def two(C, a):
    return [a[0].inv().inv()]


@pytest.fixture
def register(monkeypatch):
    """Register ``(id, build, opt_in)`` candidates for `WORD` in the order given, with nothing `PIECES` cached for it."""
    def reg(*cands):
        for i, build, opt_in in cands:
            monkeypatch.setitem(REGISTRY.defs, i, SimpleNamespace(id=i, word=WORD, build=build, opt_in=opt_in))

    dict.pop(IL.PIECES, WORD.id, None)
    yield reg
    dict.pop(IL.PIECES, WORD.id, None)


@pytest.mark.parametrize("order", [1, -1], ids=["a-first", "b-first"])
def test_two_builders_for_one_word_are_refused_in_either_order(register, order):
    register(*[(A, one, False), (B, two, False)][::order])
    for ask in (lambda: IL._boolean_build(WORD.id), lambda: WORD.id in IL.PIECES, lambda: IL.PIECES.get(WORD.id)):
        with pytest.raises(IL.AmbiguousWord, match=f"{WORD.id} is built by more than one builder, of {A}, {B}: "):
            ask()
    assert not dict.__contains__(IL.PIECES, WORD.id)


@pytest.mark.parametrize("order", [1, -1], ids=["a-first", "b-first"])
def test_one_builder_under_two_ids_is_that_builder(register, order):
    register(*[(A, one, False), (B, one, False)][::order])
    assert IL._boolean_build(WORD.id) is one
    assert inspect.unwrap(IL.PIECES[WORD.id]) is one


@pytest.mark.parametrize("order", [1, -1], ids=["a-first", "b-first"])
def test_an_opt_in_candidate_is_never_picked(register, order):
    register(*[(A, one, False), (B, two, True)][::order])
    assert IL._boolean_build(WORD.id) is one
    assert inspect.unwrap(IL.PIECES[WORD.id]) is one


def test_a_word_no_boolean_definition_builds_has_no_builder():
    assert IL._boolean_build("BooleanBuildTestNoWord_v1") is None
