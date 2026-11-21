"""`verity.circuits.ops.boolean.versions`'s choice of a word's Boolean version does not follow the order its candidates were registered
in (`_load` gathers them in that order, `_find` chooses): two versions of one name, or two names, for one word are refused as
ambiguous in either order.  An ``opt_in`` version is weighed only with ``opt_in=True`` or when its family is named."""
from __future__ import annotations

import pytest
from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT
from verity.circuits.boolean.trace import definition
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import REGISTRY, PrimitiveDefinition
from verity.circuits.ops.boolean import versions as B
from verity.circuits.types import Value

import verity_vllm.definitions  # noqa: F401  (registers the vLLM registry with the Boolean-versions lookup)

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
def test_two_versions_of_one_name_are_refused_as_ambiguous_in_either_order(fresh, order):
    fresh(*[_not("BooleanFindTestNot", 2), _not("BooleanFindTestNot", 3)][::order])
    with pytest.raises(B.NoBooleanVersion) as e:
        B.boolean_version(WORD)
    assert e.value.missing == {WORD.id: "ambiguous: ['BooleanFindTestNot_v2', 'BooleanFindTestNot_v3'] all name it as "
                                        "their word"}


@pytest.mark.parametrize("order", [1, -1], ids=["a-first", "b-first"])
def test_two_names_for_one_word_are_refused_as_ambiguous_in_either_order(fresh, order):
    fresh(*[_not("BooleanFindTestNotA", 2), _not("BooleanFindTestNotB", 2)][::order])
    with pytest.raises(B.NoBooleanVersion) as e:
        B.boolean_version(WORD)
    assert e.value.missing == {WORD.id: "ambiguous: ['BooleanFindTestNotA_v2', 'BooleanFindTestNotB_v2'] all name it as "
                                        "their word"}


def test_an_opt_in_version_is_weighed_only_when_asked_for(fresh):
    newer = _not("BooleanFindTestNot", 3)
    newer.opt_in = True
    fresh(_not("BooleanFindTestNot", 2), newer)
    assert descriptor_id(B.boolean_version(WORD)) == "BooleanFindTestNot_v2"
    with pytest.raises(B.NoBooleanVersion, match=r"ambiguous: \['BooleanFindTestNot_v2', 'BooleanFindTestNot_v3'\]"):
        B.boolean_version(WORD, opt_in=True)
    assert descriptor_id(B.boolean_version(WORD, family="BooleanFindTestNot_v3")) == "BooleanFindTestNot_v3"
    assert descriptor_id(B.boolean_version(WORD)) == "BooleanFindTestNot_v2"
