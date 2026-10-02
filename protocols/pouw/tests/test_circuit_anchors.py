"""The anchored inputs' anchors (`verity_pouw.circuit.anchors`): the verifier builds them, refuses a prover's that differ,
and refuses a repeated call index in a window (the red team's X-SPC-72)."""
from __future__ import annotations

import pytest

import circuit_fixtures as F
from verity_pouw.circuit import ncp2, traced_as
from verity_pouw.circuit import reference as R
from verity_pouw.circuit.anchors import Anchors, Ledger
from verity_pouw.protocol import DomainError

SHAPE = ncp2.Shape(32, 64, 32)


def _anchors(call: R.Call) -> Anchors:
    return Anchors(call.salt, call.forward, call.layer, call.site, call.weight, tuple(call.rowids), tuple(call.colids))


def test_the_verifier_builds_the_anchored_inputs():
    """From its own anchors, the verifier's inputs are exactly the reference call's."""
    call = F.call(32, 64, 32)
    t = traced_as("ncp-v2-shift24", 32, 64, 32)
    args = call.args()
    assert t.call_args(_anchors(call), *args[:6]) == args
    assert t.weight_args(_anchors(call), call.weight_args()[0]) == call.weight_args()


def test_a_prover_s_anchored_inputs_must_be_the_verifier_s():
    call = F.call(32, 64, 32)
    an, fn = _anchors(call), ncp2.linear(SHAPE)
    an.check_call(fn, call.args())
    for field, value in (("rowids", list(range(32))), ("forward", [4, 0]), ("weight", [12, 0])):
        other = list(call.args())
        other[[p for p, _ in fn.signature()[0]].index(field)] = value
        with pytest.raises(DomainError):
            an.check_call(fn, other)
    an.check_weight(ncp2.weight(SHAPE), call.weight_args())


def test_anchors_that_cannot_be_a_call_s_are_refused():
    call = F.call(32, 64, 32)
    with pytest.raises(DomainError):
        Anchors(call.salt, 3, 5, 2, 11, (7,) * 32, tuple(call.colids)).check(SHAPE)       # repeated row ids
    with pytest.raises(DomainError):
        Anchors(call.salt, 3, 5, 2, 11, tuple(call.rowids), tuple(call.colids)[:16]).check(SHAPE)
    with pytest.raises(DomainError):
        Anchors(bytes(31), 3, 5, 2, 11, tuple(call.rowids), tuple(call.colids)).check(SHAPE)


def test_a_window_refuses_a_repeated_call_index():
    """(forward, position) names one call (A11, `UniqueCallIndices`): a second call with it is refused."""
    call = F.call(32, 64, 32)
    ledger = Ledger()
    ledger.admit(_anchors(call), SHAPE)
    ledger.admit(Anchors(call.salt, call.forward + 1, call.layer, call.site, call.weight, tuple(call.rowids), tuple(call.colids)), SHAPE)
    with pytest.raises(DomainError):
        ledger.admit(_anchors(call), SHAPE)
