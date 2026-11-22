"""The refusals of `verity.circuits.restrict`, one case each: the smallest input that trips that clause and no earlier one,
asserted by the clause's own text, so deleting the clause fails its case."""
from __future__ import annotations

import dataclasses
import json

import pytest

import stand_in  # noqa: F401  (the test primitives' Boolean bodies)
import test_codec_refusals as TC
import test_format_spec as spec
import test_restrict as TR
from verity.circuits import codec
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks
from verity.circuits.program import Program
from verity.circuits.refs import tuple_of
from verity.circuits.types import Tuple


def _bytes(program: Program) -> bytes:
    return b"".join(descriptor_chunks(program))


def _fanout() -> tuple[bytes, R.Bits]:
    _, c, bits = TR.toy("fanout")
    return c, bits


def _listing() -> list[str]:
    c, bits = _fanout()
    return R.restrict(c, TR.TOY_PARTITION, [(0, 1)], bits).listing().decode().split("\n")


def _unit(c: bytes, bits: R.Bits, units) -> None:
    R.restrict(c, TR.TOY_PARTITION, units, bits)


def _repeated_return() -> bytes:
    def body(B, S, x, k):
        u = B.call(TR.T, x, k)
        return tuple_of(u, u)
    return _bytes(TR._toy("RestrictToyTwice", (("x", TR.A4), ("k", TR.A4)), Tuple(TR.A4, TR.A4), body))


def _input_after_a_gate() -> bytes:
    program = Program(spec._comp("TLate", (), lambda S: ((("a", spec.V32), ("b", spec.V32), ("c", spec.V32)), spec.V32),
                                 lambda B, S, a, b, c: B.call(spec.TAdd, a, b)))
    d = json.loads(spec._canon(codec.encode_program(program)))
    nodes = d["definitions"][next(f for f in d["definitions"] if f.startswith("Root["))]["body"]
    nodes["nodes"][2], nodes["nodes"][3] = nodes["nodes"][3], nodes["nodes"][2]
    nodes["ret"] = {"affine": ["n", 2, 0, 1]}
    return json.dumps(d).encode()


def test_bytes_that_are_no_descriptor_are_refused():
    with pytest.raises(R.Refused) as e:
        R.program(b"not json")
    assert str(e.value).startswith("E_DESCRIPTOR: Expecting value")


def test_an_input_gate_has_no_lines_in_a_block():
    _, bits = _fanout()
    with pytest.raises(R.Refused) as e:
        bits.lines(0)
    assert str(e.value) == "E_INPUT_IN_BLOCK: gate 0 of C is an input gate inside the block"


def test_a_value_returned_twice_is_refused_as_two_port_bits_of_one_gate():
    with pytest.raises(R.Refused) as e:
        R.listing_of(_repeated_return())
    assert str(e.value) == "E_PORT_TWICE: two port bits are one gate of C on bits"


def test_an_input_node_after_a_computed_gate_is_refused_in_a_listing():
    with pytest.raises(R.Refused) as e:
        R.listing_of(_input_after_a_gate())
    assert str(e.value) == "E_FORMAT: an input node after a computed gate"


def test_a_query_other_than_a_template_query_is_refused_by_name():
    c, bits = _fanout()
    q = {"name": "Q_word", "version": 1, "params": {"X": 16}}
    with pytest.raises(R.Refused) as e:
        R.restrict(c, {"query": q}, [(0, 1)], bits)
    assert str(e.value) == f"E_QUERY: restrict covers the template queries, not {q!r}"


def test_a_template_query_naming_no_template_of_c_is_refused_by_its_own_reason():
    c, bits = _fanout()
    partition = {"query": {"name": "Q_template_instances", "version": 0, "params": {"templates": ["Nope_v1"]}}}
    with pytest.raises(R.Refused) as e:
        R.restrict(c, partition, [(0, 1)], bits)
    assert str(e.value).startswith("E_QUERY: ") and "Nope_v1" in str(e.value) and not str(e.value).startswith("E_QUERY: restrict covers")


@pytest.mark.parametrize("units", [[], [(1, 1)], [(7, 9)], [(1, 2), (0, 1)], [(0, 2), (1, 3)]])
def test_units_that_are_no_ascending_disjoint_ranges_of_the_population_are_refused_by_name(units):
    c, bits = _fanout()
    with pytest.raises(R.Refused) as e:
        _unit(c, bits, units)
    assert str(e.value) == f"E_UNITS: units {[tuple(u) for u in units]!r}: not ascending disjoint nonempty ranges within 0..3"


def test_a_listing_with_another_header_is_no_block_listing():
    lines = _listing()
    lines[0] = "verity/block-circuit/v0"
    with pytest.raises(R.Refused) as e:
        R.evaluate("\n".join(lines).encode(), [])
    assert str(e.value) == "E_FORMAT: not a block listing"


def test_a_listing_is_refused_inputs_of_another_count():
    lines = _listing()
    n = int(lines[1].split()[0])
    with pytest.raises(R.Refused) as e:
        R.evaluate("\n".join(lines).encode(), [0] * (n + 1))
    assert str(e.value) == f"E_FORMAT: {n + 1} input bits for {n}"


def test_a_listing_gate_of_no_known_kind_is_refused():
    lines = _listing()
    i = int(lines[1].split()[0]) + 2
    lines[i] = "Frob_v1 " + lines[i].split(" ", 1)[1]
    with pytest.raises(R.Refused) as e:
        R.evaluate("\n".join(lines).encode(), [0] * int(lines[1].split()[0]))
    assert str(e.value) == "E_KIND: Frob_v1"


def test_a_primitive_decoded_from_c_alone_is_not_evaluated():
    d, _ = TC.tiny()
    p = R.program(json.dumps(d).encode()).defs["TAdd32_v1"]
    with pytest.raises(ValueError) as e:
        p.evaluate(1, 2)
    assert str(e.value) == "decoded from C for its structure only"


def test_a_block_bit_outside_its_inputs_and_gates_names_no_bit_of_c():
    c, bits = _fanout()
    b = R.restrict(c, TR.TOY_PARTITION, [(0, 1)], bits)
    total = len(b.input_bits) + sum(hi - lo for lo, hi in b.gate_ranges)
    for i, left in ((-1, -1), (total, 0), (total + 5, 5)):
        with pytest.raises(IndexError) as e:
            b.to_c_bit(i)
        assert e.value.args == (left,)


def test_words_that_cut_a_member_are_refused_by_front_called_directly():
    _, bits = _fanout()
    body = bits.D.root.body
    k = next(k for k, n in enumerate(body.nodes) if n.form in ("call", "batch") and n.fn.gates >= 2)
    with pytest.raises(R.Refused) as e:
        R.front(bits.D, [(body.offsets[k] + 1, body.offsets[k + 1])])
    assert str(e.value) == f"E_UNITS: the units cut root node {k} inside a member"


def test_a_block_built_by_hand_without_an_input_a_gate_reads_names_the_bit():
    c, bits = _fanout()
    b = R.restrict(c, TR.TOY_PARTITION, [(1, 3)], bits)
    assert b.inputs
    with pytest.raises(R.Refused) as e:
        dataclasses.replace(b, inputs=b.inputs[:-1]).listing()
    assert str(e.value).startswith("E_UNREAD: a gate reads bit ") and str(e.value).endswith(" of C, which no port carries")
