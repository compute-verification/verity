"""A window's operand rows (`pouw.operands`): committed per window, held by the verifier only under the window's own
context, and their root X moving with every value's root."""
from __future__ import annotations

import hashlib

import pytest

from verity.core.protocols.pouw import operands as OP

CONTEXT, OTHER = hashlib.sha512(b"run").hexdigest(), hashlib.sha512(b"other run").hexdigest()
ROWS = {"a": [[1, 2, 3, 4], [0, 0, 0, 0]], "b": [[5, 6, 7, 8, 9, 10, 11, 12]]}


def block(context: str = CONTEXT, rows=ROWS) -> dict:
    return OP.block("p", OP.commit("p", context, rows))


def test_the_verifier_holds_the_operands_under_the_windows_context():
    b = block()
    assert OP.held({"context": CONTEXT, OP.KEY: b}) == b["values"]
    with pytest.raises(ValueError, match="domain other than"):
        OP.held({"context": OTHER, OP.KEY: b})


@pytest.mark.parametrize("bad", [
    lambda b: {**b, "values": {}},
    lambda b: {**b, "extra": 1},
    lambda b: {**b, "values": {**b["values"], "a": {**b["values"]["a"], "value": "b"}}},
    lambda b: {**b, "values": {**b["values"], "a": {**b["values"]["a"], "leaves": 3}}},
    lambda b: {**b, "values": {**b["values"], "a": {k: v for k, v in b["values"]["a"].items() if k != "root"}}},
    lambda b: {"program": "p"},
    lambda b: [1],
])
def test_a_malformed_block_is_refused(bad):
    with pytest.raises(ValueError):
        OP.held({"context": CONTEXT, OP.KEY: bad(block())})


def test_x_moves_with_every_root_and_the_program():
    b = block()
    assert OP.root(b) == OP.root(block(rows=ROWS) | {"values": b["values"]})
    assert OP.root(b) != OP.root(block()), "fresh salts make fresh roots"
    swapped = {"program": "p", "values": {**b["values"], "b": {**b["values"]["b"], "root": b["values"]["a"]["root"]}}}
    assert len({OP.root(b), OP.root(swapped), OP.root({**b, "program": "q"})}) == 3
