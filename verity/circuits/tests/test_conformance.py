"""A Definition's conformance text starts with a label of `verity.circuits.defs.CONFORMANCE`, a closed set: any other text
is refused where the Definition is made, with the fix in the message."""
from __future__ import annotations

import pytest

from verity.circuits.defs import CONFORMANCE, CompositeDefinition, PrimitiveDefinition, conformance_label
from verity.circuits.types import Value

V8 = Value(8)


def _prim(conformance: str) -> PrimitiveDefinition:
    return PrimitiveDefinition("ConfProbe", 1, (("a", V8),), V8, lambda a: a, conformance=conformance, register=False)


def _comp(conformance: str) -> CompositeDefinition:
    return CompositeDefinition("ConfProbe", 1, (), lambda S: ((("a", V8),), V8), lambda B, S, a: a, conformance=conformance,
                               register=False)


@pytest.mark.parametrize("text", ["exact-model-tested", "exact-model-tested; the capture", "model-untested: read from the kernel",
                                  "superseded by ConfProbe_v2: why", "exact-model-tested at cc 12.0's DOT"])
def test_a_text_that_starts_with_a_label_is_that_label(text):
    for d in (_prim(text), _comp(text)):
        assert conformance_label(d.conformance) in CONFORMANCE and text.startswith(conformance_label(d.conformance))


@pytest.mark.parametrize("text", ["model-tested on synthetic rows", "adversarial: not a semantics", "predicted: its body's",
                                  "exact-model-testedness", "Exact-model-tested", "", "exact by construction"])
def test_any_other_text_is_refused_with_the_fix(text):
    for make in (_prim, _comp):
        with pytest.raises(ValueError, match=r"ConfProbe_v1: conformance .* doesn't start with a label of "
                                             r"verity\.ml\.circuits\.defs\.CONFORMANCE .*add a new label to CONFORMANCE"):
            make(text)
