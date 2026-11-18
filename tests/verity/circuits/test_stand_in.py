from __future__ import annotations

import pytest

import stand_in as S
from verity.circuits.defs import PrimitiveDefinition
from verity.circuits.ops.boolean import versions as V
from verity.circuits.types import Value


def _xor8(name: str, module: str) -> PrimitiveDefinition:
    def f(a, b):
        return a ^ b
    f.__module__ = module
    return PrimitiveDefinition(name, 1, (("a", Value(8)), ("b", Value(8))), Value(8), f, register=False)


def test_a_stand_in_is_given_only_to_a_primitive_a_test_defines(monkeypatch):
    """A test's own primitive gets its stand-in; the catalog's without a body is refused, and so is a test's primitive under
    a catalog primitive's id."""
    catalog = _xor8("StandInCatalogXor8", "verity.circuits.ops.elsewhere")
    assert S._lookup(_xor8("StandInOwnXor8", __name__)).id == "StandIn[StandInOwnXor8]_v1"
    with pytest.raises(V.NoBooleanVersion):
        S._lookup(catalog)
    monkeypatch.setattr(S, "REGISTRY", {catalog.id: catalog})
    with pytest.raises(V.NoBooleanVersion):
        S._lookup(_xor8("StandInCatalogXor8", __name__))
