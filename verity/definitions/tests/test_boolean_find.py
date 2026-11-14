"""`verity.definitions.boolean`'s choice of a word's Boolean version does not follow the order its candidates were registered
in (`_load` gathers them in that order, `_find` chooses): two versions of one name give the newest in either order, and two
names for one word are refused as ambiguous in either order."""
from __future__ import annotations

import pytest

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT
from verity.circuits.boolean.trace import definition
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import REGISTRY, PrimitiveDefinition
from verity.circuits.types import Value
from verity.definitions import boolean as B

WORD = PrimitiveDefinition("BooleanFindTestWord", 1, (("x", Value(4)),), Value(4), lambda x: x ^ 15, register=False)
U4 = BL.bit_type(Value(4))


def _not(name: str, version: int):
    return definition(name, version, lambda C, a: [NOT(b) for b in a], (("x", U4),), U4, word=WORD, register=False)


@pytest.fixture
def fresh(monkeypatch):
    """`boolean`'s process caches (`_TABLE`, `_VERSIONS`) empty for the test and restored after it; ``reg`` registers
    Definitions in the order given."""
    monkeypatch.setattr(B, "_TABLE", [])
    monkeypatch.setattr(B, "_VERSIONS", {})

    def reg(*defs):
        for d in defs:
            monkeypatch.setitem(REGISTRY.defs, d.id, d)
    return reg


@pytest.mark.parametrize("order", [1, -1], ids=["older-first", "newer-first"])
def test_two_versions_of_one_name_give_the_newest_in_either_order(fresh, order):
    fresh(*[_not("BooleanFindTestNot", 2), _not("BooleanFindTestNot", 3)][::order])
    assert descriptor_id(B.boolean_version(WORD)) == "BooleanFindTestNot_v3"


@pytest.mark.parametrize("order", [1, -1], ids=["a-first", "b-first"])
def test_two_names_for_one_word_are_refused_as_ambiguous_in_either_order(fresh, order):
    fresh(*[_not("BooleanFindTestNotA", 2), _not("BooleanFindTestNotB", 2)][::order])
    with pytest.raises(B.NoBooleanVersion) as e:
        B.boolean_version(WORD)
    assert e.value.missing == {WORD.id: "ambiguous: ['BooleanFindTestNotA_v2', 'BooleanFindTestNotB_v2'] all name it as "
                                        "their word"}
