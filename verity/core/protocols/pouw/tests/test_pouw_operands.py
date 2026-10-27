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


def test_a_record_states_the_operands_with_their_gaps_or_why_they_are_refused():
    b = block()
    got = OP.stated({"context": CONTEXT, OP.KEY: b})
    assert got == {"held": True, "X": OP.root(b).hex(), "roots": {v: e["root"] for v, e in b["values"].items()},
                   "gaps": list(OP.GAPS)} and any("CIR-100" in g for g in got["gaps"])
    assert OP.stated({"context": CONTEXT}) is None
    bad = OP.stated({"context": OTHER, OP.KEY: b})
    assert bad["held"] is False and "domain other than" in bad["why"]


def test_x_is_the_root_of_the_windows_values_its_salt_epoch_binds():
    """X is SHA-512(tag ‖ u64be(len program) ‖ program ‖ per value in name order: u64be(len name) ‖ name ‖ domain ‖ root),
    the bytes ALLP-SALT's salt epoch root reads as the window's activations' root (``core_check.values_root``)."""
    b = block()
    want = b"verity/pouw-window-values/v1" + len(b"p").to_bytes(8, "big") + b"p"
    for v in ("a", "b"):
        e = b["values"][v]
        want += (1).to_bytes(8, "big") + v.encode() + bytes.fromhex(e["domain"]) + bytes.fromhex(e["root"])
    assert OP.root(b) == hashlib.sha512(want).digest()
