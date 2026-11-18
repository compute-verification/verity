"""`DefinitionTable.add_family` and `resolve`: a registered id wins unless a family matches it, a family matching a registered
id that isn't its own is refused, two families matching one id are refused, a family's Definition has the id asked for, and a
width is made once."""
from __future__ import annotations

import re

import pytest

from verity.circuits.defs import DefinitionTable, PrimitiveDefinition
from verity.circuits.types import Value


def _prim(table: DefinitionTable, name: str, width: int) -> PrimitiveDefinition:
    p = PrimitiveDefinition(name, 1, (("a", Value(width)),), Value(1), lambda a: a & 1, conformance="exact-model-tested",
                            register=False)
    return table.add(p)


def _family(table: DefinitionTable, name: str = "Wx"):
    """A caching factory of ``{name}{w}_v1``, which registers what it makes, as the catalog's factories do."""
    made: dict[int, PrimitiveDefinition] = {}

    def make(w: int) -> PrimitiveDefinition:
        if w not in made:
            made[w] = _prim(table, f"{name}{w}", w)
        return made[w]

    return make


def test_a_registered_id_no_family_matches_wins():
    T = DefinitionTable()
    p = _prim(T, "Plain", 8)
    T.add_family(r"Wx(\d+)_v1", _family(T))
    assert T.resolve("Plain_v1") is p
    assert T.resolve("Absent_v1") is None


def test_a_family_matching_a_registered_id_not_its_own_is_refused_by_name():
    T = DefinitionTable()
    _prim(T, "Wx8", 8)
    with pytest.raises(ValueError, match=re.escape("Wx8_v1 is registered, and family")):
        T.add_family(r"Wx(\d+)_v1", lambda w: PrimitiveDefinition(f"Wx{w}", 1, (("a", Value(w)),), Value(1), lambda a: 0,
                                                                    conformance="exact-model-tested", register=False))
    assert not T.families
    U = DefinitionTable()
    U.add_family(r"Wx(\d+)_v1", lambda w: PrimitiveDefinition(f"Wx{w}", 1, (("a", Value(w)),), Value(1), lambda a: 0,
                                                                conformance="exact-model-tested", register=False))
    _prim(U, "Wx16", 16)
    with pytest.raises(ValueError, match=re.escape("Wx16_v1 is registered, and family")):
        U.resolve("Wx16_v1")


def test_two_families_matching_one_id_are_refused_by_name():
    T = DefinitionTable()
    T.add_family(r"Wx(\d+)_v1", _family(T))
    T.add_family(r"W\w(\d+)_v1", _family(T))
    with pytest.raises(ValueError, match=r"Wx8_v1: families 'Wx.*', 'W.*' all match it"):
        T.resolve("Wx8_v1")


def test_a_family_making_another_id_is_refused():
    T = DefinitionTable()
    T.add_family(r"Wx(\d+)_v1", lambda w: _prim(T, f"Other{w}", w))
    with pytest.raises(ValueError, match=re.escape("made Other8_v1 when asked for Wx8_v1")):
        T.resolve("Wx8_v1")


def test_a_width_is_made_once_and_its_own_registration_is_accepted():
    T = DefinitionTable()
    make = _family(T)
    early = make(4)
    T.add_family(r"Wx(\d+)_v1", make)
    assert T.resolve("Wx4_v1") is early
    first = T.resolve("Wx8_v1")
    assert first.id == "Wx8_v1" and T.resolve("Wx8_v1") is first and T.defs["Wx8_v1"] is first
