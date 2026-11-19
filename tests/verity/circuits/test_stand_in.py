from __future__ import annotations

import numpy as np
import pytest

import stand_in as S
from verity.circuits.defs import PrimitiveDefinition
from verity.circuits.evaluation.bits import evaluate_words
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


def _add(name: str, w: int) -> PrimitiveDefinition:
    return PrimitiveDefinition(name, 1, (("a", Value(w)), ("b", Value(w))), Value(w), lambda a, b: (a + b) % (1 << w),
                               register=False)


def test_a_stand_in_that_is_not_its_primitives_function_is_refused_by_name():
    with pytest.raises(S.StandInMismatch, match="StandInAdd8_v1"):
        S.stand_in_body(_add("StandInAdd8", 8))


def test_a_narrow_primitive_that_is_no_xor_gets_its_truth_table():
    p = _add("StandInAdd4", 4)
    a, b = np.meshgrid(np.arange(16, dtype=np.uint8), np.arange(16, dtype=np.uint8))
    assert np.array_equal(evaluate_words(S.stand_in_body(p), a.ravel(), b.ravel()), (a.ravel() + b.ravel()) % 16)
