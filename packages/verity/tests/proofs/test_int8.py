"""``veritor.int8-q7``: exact int8 products, int32 sums and Q7 requantization, as words."""

from __future__ import annotations

import json

import pytest

from verity.errors import InvalidArtifact
from verity.proofs.gateset import GateSet, make_isa_gate_set
from verity.proofs.int8 import INT8_Q7_MAX_CONTRACTION, make_int8_q7_gate_set, requant_s8, requant_u7, signed, word


INT8_Q7_DIGEST = "88162784922551008cdff19a0a08a118d4bd6f02fbd65c71629c502a988b69c6"
"""Pinned: the Rust checker (`backends/sp1/common/src/gateset.rs`) resolves `veritor.int8-q7@1` to this digest."""


@pytest.fixture(scope="module")
def gates() -> GateSet:
    return make_int8_q7_gate_set()


def s8(value: int) -> int:
    return word(8, value)


def s32(value: int) -> int:
    return word(32, value)


def test_words_and_signed_readings_are_inverse() -> None:
    assert [signed(8, s8(v)) for v in (-128, -1, 0, 1, 127)] == [-128, -1, 0, 1, 127]
    assert s8(-1) == 0xFF and s8(-128) == 0x80 and s8(127) == 0x7F
    assert s32(-1) == 0xFFFFFFFF and signed(32, 0x80000000) == -(1 << 31)
    assert signed(32, 0x7FFFFFFF) == (1 << 31) - 1


def test_the_set_declares_two_widths_and_the_toy_costs(gates: GateSet) -> None:
    assert gates.id == "veritor.int8-q7@1"
    assert [gate.name for gate in gates] == [
        "add32",
        "in",
        "mul8",
        "requant_s8",
        "requant_u7",
        "weight",
    ]
    assert (gates.input_gates, gates.weight_gates) == (("in",), ("weight",))
    shapes = {gate.name: (gate.arity, gate.arg_widths, gate.width) for gate in gates}
    assert shapes == {
        "mul8": (2, (8, 8), 32),
        "add32": (2, (32, 32), 32),
        "requant_u7": (1, (32,), 8),
        "requant_s8": (1, (32,), 8),
        "in": (0, (), 8),
        "weight": (0, (), 8),
    }
    assert {gate.name: (gate.replay_cost, gate.proof_cost) for gate in gates} == {
        "mul8": (2, 2),
        "add32": (1, 1),
        "requant_u7": (1, 1),
        "requant_s8": (1, 1),
        "in": (0, 1),
        "weight": (0, 1),
    }
    assert INT8_Q7_MAX_CONTRACTION * (1 << 14) == 1 << 27  # the add32 domain fits int32


def test_the_manifest_pins_arg_widths_only_where_they_differ_and_the_digest(
    gates: GateSet,
) -> None:
    manifest = json.dumps(gates.manifest, sort_keys=True, separators=(",", ":"))
    assert (
        '{"arg_widths":[8,8],"arity":2,"name":"mul8","proof_cost":2,"replay_cost":2,'
        '"source":null,"width":32}' in manifest
    )
    assert (
        '{"arg_widths":[32],"arity":1,"name":"requant_u7","proof_cost":1,'
        '"replay_cost":1,"source":null,"width":8}' in manifest
    )
    assert (
        '{"arity":2,"name":"add32","proof_cost":1,"replay_cost":1,"source":null,'
        '"width":32}' in manifest
    )
    assert (
        '{"arity":0,"name":"in","proof_cost":1,"replay_cost":0,"source":"input","width":8}'
        in manifest
    )
    assert gates.digest == INT8_Q7_DIGEST == make_int8_q7_gate_set().digest
    assert gates.digest != make_isa_gate_set(8).digest


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (0, 0),
        (0, -128),
        (-128, -128),
        (127, -128),
        (-128, 127),
        (127, 127),
        (-1, 1),
        (-1, -1),
        (3, -7),
        (-100, 100),
    ],
)
def test_mul8_is_the_exact_signed_product_as_an_int32_word(
    gates: GateSet, a: int, b: int
) -> None:
    out = gates["mul8"].evaluate((s8(a), s8(b)))
    assert 0 <= out < 1 << 32
    assert signed(32, out) == a * b


def test_mul8_extremes(gates: GateSet) -> None:
    mul8 = gates["mul8"]
    assert mul8.evaluate((s8(-128), s8(-128))) == 16384
    assert signed(32, mul8.evaluate((s8(127), s8(-128)))) == -16256
    assert mul8.evaluate((s8(127), s8(127))) == 16129
    assert mul8.evaluate((0, s8(-1))) == 0
    with pytest.raises(InvalidArtifact, match="argument 0 is not a 8-bit value"):
        mul8.evaluate((256, 1))
    with pytest.raises(InvalidArtifact, match="expects 2 arguments"):
        mul8.evaluate((1,))


def test_add32_is_exact_on_the_domain_and_modular_at_the_int32_edge(
    gates: GateSet,
) -> None:
    add32 = gates["add32"]
    for a, b in ((0, 0), (16384, 16384), (-16256, 16256), (-1, 1), (-5, -7)):
        assert signed(32, add32.evaluate((s32(a), s32(b)))) == a + b
    # the largest partial sums of the declared domain, and their negatives, stay exact
    top = INT8_Q7_MAX_CONTRACTION * 16384
    assert signed(32, add32.evaluate((s32(top - 16384), s32(16384)))) == top
    assert signed(32, add32.evaluate((s32(-top + 16256), s32(-16256)))) == -top
    # outside it, the gate is what the word arithmetic says: modulo 2**32
    assert add32.evaluate((s32((1 << 31) - 1), 1)) == s32(-(1 << 31))
    assert add32.evaluate((s32(-(1 << 31)), s32(-1))) == s32((1 << 31) - 1)
    with pytest.raises(InvalidArtifact, match="argument 1 is not a 32-bit value"):
        add32.evaluate((0, 1 << 32))


REQUANT_CASES = [
    # (int32 input, requant_u7, requant_s8 as signed)
    (0, 0, 0),
    (1, 0, 0),
    (127, 0, 0),
    (128, 1, 1),
    (-1, 0, -1),  # floor(-1 / 128) = -1: u7 clips to 0, s8 keeps -1
    (-128, 0, -1),
    (-129, 0, -2),  # floor(-129 / 128) = -2
    (16256, 127, 127),  # 127 * 128 -> exactly 127
    (16383, 127, 127),
    (16384, 127, 127),  # -> 128, clips to 127
    (-16384, 0, -128),
    (-16385, 0, -128),  # -> -129, clips to -128 in s8, 0 in u7
    (-(1 << 31), 0, -128),
    ((1 << 31) - 1, 127, 127),
]


@pytest.mark.parametrize(("value", "u7", "signed_s8"), REQUANT_CASES)
def test_requant_floors_toward_minus_infinity_and_saturates(
    gates: GateSet, value: int, u7: int, signed_s8: int
) -> None:
    assert requant_u7(value) == u7 and requant_s8(value) == signed_s8
    assert gates["requant_u7"].evaluate((s32(value),)) == u7
    assert gates["requant_s8"].evaluate((s32(value),)) == s8(signed_s8)
    assert gates["requant_u7"].check((s32(value),), u7)
    assert gates["requant_s8"].check((s32(value),), s8(signed_s8))


def test_requant_outputs_are_8_bit_words_and_arity_one(gates: GateSet) -> None:
    for name in ("requant_u7", "requant_s8"):
        gate = gates[name]
        assert gate.arity == 1 and gate.width == 8 and gate.arg_widths == (32,)
        with pytest.raises(InvalidArtifact, match="expects 1 arguments"):
            gate.evaluate((0, 0))
        with pytest.raises(InvalidArtifact, match="argument 0 is not a 32-bit value"):
            gate.evaluate((1 << 32,))
        with pytest.raises(InvalidArtifact, match="output is not a 8-bit value"):
            gate.check((0,), 256)
    # u7 never produces the high bit; s8 does exactly for negatives
    assert all(gates["requant_u7"].evaluate((s32(t),)) < 128 for t in range(-300, 300))
    assert all(
        (gates["requant_s8"].evaluate((s32(t),)) >= 128) == (t < 0)
        for t in range(-300, 300)
    )
