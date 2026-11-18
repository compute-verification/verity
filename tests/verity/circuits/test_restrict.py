"""`verity.circuits.restrict` on the periods' chain: eight `Gemm_v2{K=16, N=16}` units at Ampere's dot, one wire between
consecutive units, blocks of two units and of one."""
from __future__ import annotations

import functools
import importlib.util
import json
import random
from pathlib import Path

import pytest

from verity.circuits import boolean as BL
from verity.circuits import restrict as R
from verity.circuits.codec import definition_digest, definition_entry, descriptor_chunks, descriptor_id
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation.reference import evaluate_region
from verity.circuits.program import Program, input_id
from verity.circuits.refs import array_of, tuple_of
from verity.circuits.types import Array, Tuple

_SPEC = importlib.util.spec_from_file_location("restrict_chain", Path(__file__).parent / "ops/test_boolean_block_circuit.py")
_BB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BB)
GID, _chain = _BB.GID, _BB._chain

PARTITION = {"query": {"name": "Q_template_instances", "version": 0, "params": {"templates": [GID]}}}
MEMBERS = [[(2 * i, 2 * i + 2)] for i in range(4)]
UNITS = [[(u, u + 1)] for u in range(8)]


@functools.cache
def chain():
    program = _chain()
    c = b"".join(descriptor_chunks(program))
    return program, c, R.Bits(R.program(c))


@functools.cache
def block(units: tuple) -> R.Block:
    _, c, bits = chain()
    return R.restrict(c, PARTITION, list(units), bits)


@functools.cache
def listing(units: tuple) -> bytes:
    return block(units).listing()


def trace(seed: int) -> dict[int, int]:
    program, _, _ = chain()
    C, rng, t = program.circuit, random.Random(seed), {}
    for g in C.gates():
        if (inp := input_id(g.prim.id)) is not None:
            t[g.index] = rng.getrandbits(inp[1])
    return evaluate_region(C, C[0:C.size], t)


def bit(t: dict[int, int], bits: R.Bits, x: int) -> int:
    g = bits.gate_of(x)
    return (t[g] >> bits.out[g].index(x)) & 1


@pytest.mark.parametrize("units", MEMBERS + UNITS[:2])
def test_a_blocks_circuit_computes_what_C_computes_there(units):
    program, _, bits = chain()
    assert program.circuit.size == bits.C.size
    b, t = block(tuple(units)), trace(sum(u for r in units for u in r))
    assert R.evaluate(listing(tuple(units)), [bit(t, bits, x) for x in b.input_bits]) == [bit(t, bits, x) for x in b.output_bits]


@pytest.mark.parametrize("units", MEMBERS + UNITS[:1] + UNITS[3:4])
def test_the_descriptor_laid_out_on_bits_is_the_listing_from_C(units):
    b = block(tuple(units))
    own, ref = R.listing_of(b.descriptor).split(b"\n"), listing(tuple(units)).split(b"\n")
    assert len(own) == len(ref)
    n_in = len(b.input_bits)
    kinds = [b.bits.kind[b.bits.gate_of(x)] for x in b.input_bits]
    assert [x for i, x in enumerate(ref) if i not in range(2, 2 + n_in)] == \
           [x for i, x in enumerate(own) if i not in range(2, 2 + n_in)]
    assert own[2:2 + n_in] == [(v if k == "input" else R.RUN).encode() for k, v in kinds]
    assert ref[2:2 + n_in] == [(v if k == "input" else R.WIRE).encode() for k, v in kinds]


def test_the_descriptor_is_a_program_of_Cs_definitions():
    _, c, bits = chain()
    b = block(tuple(MEMBERS[1]))
    assert json.loads(b.descriptor)["boolean"] == json.loads(c)["boolean"]
    own, cs = R.program(b.descriptor).defs, bits.D.defs
    shared = [k for k in own if k in cs]
    assert set(own) - set(shared) == {"Root[Block_v1]_v1"}
    assert all(definition_digest(definition_entry(own[k])) == definition_digest(definition_entry(cs[k])) for k in shared)
    assert b.descriptor == block(tuple(MEMBERS[2])).descriptor


def test_the_blocks_gates_are_Cs_computed_gates_each_once():
    _, _, bits = chain()
    computed = {x for g, (k, _) in enumerate(bits.kind) if k != "input" for x in range(bits.base[g], bits.base[g + 1])}
    for blocks in (MEMBERS, UNITS):
        got = [x for u in blocks for lo, hi in block(tuple(u)).gate_ranges for x in range(lo, hi)]
        assert len(got) == len(set(got)) and set(got) == computed


@pytest.mark.parametrize("units", MEMBERS + UNITS)
def test_the_map_is_injective_and_carries_the_ports_to_the_boundary(units):
    _, _, bits = chain()
    b = block(tuple(units))
    n_in, gates = len(b.input_bits), [x for lo, hi in b.gate_ranges for x in range(lo, hi)]
    image = [b.to_c(i) for i in range(n_in + len(gates))]
    assert len(set(image)) == len(image) and image == b.input_bits + gates
    assert list(b.inputs) == sorted(set(b.inputs)) and list(b.outputs) == sorted(set(b.outputs))
    (lo, hi), = b.words
    region = bits.C.region(lo, hi)
    assert list(b.inputs) == list(region.in_) and list(b.outputs) == list(region.out)
    assert b.input_bits == [x for g in b.inputs for x in bits.out[g]]
    lines = listing(tuple(units)).decode().splitlines()
    i, k, o = map(int, lines[1].split())
    assert (i, k, o) == (n_in, len(gates), len(b.output_bits))
    assert [image[int(s.split()[1])] for s in lines[2 + i + k:]] == b.output_bits


def test_a_wire_inside_a_block_is_no_port_unless_C_returns_it():
    _, _, bits = chain()
    root = set(bits.C.root_outputs())
    pair, first = block(tuple(MEMBERS[0])), block(tuple(UNITS[0]))
    inner = set(first.outputs) - set(pair.outputs)
    assert inner == set(first.outputs) - root
    assert not inner & set(pair.inputs)


def test_the_bytes_are_a_function_of_C_and_the_units():
    _, c, _ = chain()
    again = b"".join(descriptor_chunks(_chain()))
    b = R.restrict(again, PARTITION, MEMBERS[1])
    assert again == c and b.descriptor == block(tuple(MEMBERS[1])).descriptor and b.listing() == listing(tuple(MEMBERS[1]))


@pytest.mark.parametrize("units", [[], [(1, 1)], [(7, 9)], [(2, 4), (3, 5)], [(4, 6), (0, 2)]])
def test_units_not_ascending_disjoint_ranges_of_the_population_are_refused(units):
    _, c, bits = chain()
    with pytest.raises(R.Refused, match="E_UNITS"):
        R.restrict(c, PARTITION, units, bits)


def test_a_query_other_than_a_template_query_is_refused():
    _, c, bits = chain()
    with pytest.raises(R.Refused, match="E_QUERY"):
        R.restrict(c, {"query": {"name": "Q_word", "version": 1, "params": {"X": 16}}}, [(0, 1)], bits)
    with pytest.raises(R.Refused, match="E_QUERY"):
        R.restrict(c, {"query": {"name": "Q_template_instances", "version": 0, "params": {"templates": ["Nope_v1"]}}},
                   [(0, 1)], bits)


A4 = Array(4, BL.BIT)


def _t_body(B, S, a, k):
    return array_of([B.call(BL.XOR, BL.operands(a[i], B.call(BL.AND, BL.operands(k[i], a[(i + 1) % 4])))) for i in range(4)])


T = bind(CompositeDefinition("RestrictToyT", 1, (), lambda S: ((("a", A4), ("k", A4)), A4), _t_body, register=False))
TOY_PARTITION = {"query": {"name": "Q_template_instances", "version": 0, "params": {"templates": [descriptor_id(T)]}}}


def _toy(name: str, params, ret, body) -> Program:
    return Program(CompositeDefinition(name, 1, (), lambda S: (params, ret), body, register=False), registered=["k"])


def fanout() -> Program:
    """u0 = T(x, k), read by u1 and u2; k registered."""
    def body(B, S, x, k):
        u0 = B.call(T, x, k)
        return tuple_of(B.call(T, u0, k), B.call(T, u0, k))
    return _toy("RestrictToyFanout", (("x", A4), ("k", A4)), Tuple(A4, A4), body)


def pair() -> Program:
    """u1 reads u0, nothing else reads u0, and C returns u1 and u2."""
    def body(B, S, x, k):
        return tuple_of(B.call(T, B.call(T, x, k), k), B.call(T, x, k))
    return _toy("RestrictToyPair", (("x", A4), ("k", A4)), Tuple(A4, A4), body)


def batched() -> Program:
    """A batch of three units over x's rows, k shared, then a unit reading the batch's second."""
    def body(B, S, x, k):
        ys = B.batch(T, x, k, axes=(0, None))
        return tuple_of(ys, B.call(T, ys[1], k))
    return _toy("RestrictToyBatch", (("x", Array(3, A4)), ("k", A4)), Tuple(Array(3, A4), A4), body)


TOYS = {"fanout": (fanout, [[(0, 1)], [(1, 3)], [(0, 3)], [(2, 3)]]),
        "pair": (pair, [[(0, 2)], [(0, 1)], [(1, 3)], [(2, 3)]]),
        "batch": (batched, [[(0, 3)], [(1, 2)], [(0, 2)], [(2, 4)], [(3, 4)]])}


@functools.cache
def toy(name: str):
    program = TOYS[name][0]()
    c = b"".join(descriptor_chunks(program))
    return program, c, R.Bits(R.program(c))


@pytest.mark.parametrize("name,units", [(n, u) for n, (_f, us) in TOYS.items() for u in us])
def test_a_toys_block(name, units):
    program, c, bits = toy(name)
    b = R.restrict(c, TOY_PARTITION, units, bits)
    C, rng, t = program.circuit, random.Random(len(units[0])), {}
    for g in C.gates():
        if input_id(g.prim.id) is not None:
            t[g.index] = rng.getrandbits(1)
    evaluate_region(C, C[0:C.size], t)
    text = b.listing()
    assert R.evaluate(text, [bit(t, bits, x) for x in b.input_bits]) == [bit(t, bits, x) for x in b.output_bits]
    inside = {g for lo, hi in b.words for g in range(lo, hi)}
    reads = {o for g in C.gates() if g.index not in inside for o in g.operands}
    assert list(b.inputs) == sorted({o for g in inside for o in C.gate(g).operands} - inside)
    assert list(b.outputs) == sorted(inside & (reads | set(C.root_outputs())))
    own, ref = R.listing_of(b.descriptor).split(b"\n"), text.split(b"\n")
    n_in = len(b.input_bits)
    assert own[:2] == ref[:2] and own[2 + n_in:] == ref[2 + n_in:]
    assert own[2:2 + n_in] == [x.replace(b"wire", b"run") for x in ref[2:2 + n_in]]
    image = [b.to_c(i) for i in range(n_in + len(inside))]
    assert len(set(image)) == len(image)


def test_two_readers_of_one_writer_read_it_through_one_port_each_bit():
    _, c, bits = toy("fanout")
    readers, writer = R.restrict(c, TOY_PARTITION, [(1, 3)], bits), R.restrict(c, TOY_PARTITION, [(0, 1)], bits)
    u0 = set(writer.outputs)
    assert len(u0) == 4 and u0 <= set(readers.inputs) and len(readers.inputs) == len(set(readers.inputs))


def test_a_registered_value_is_a_registered_input_of_the_block():
    _, c, bits = toy("fanout")
    b = R.restrict(c, TOY_PARTITION, [(1, 3)], bits)
    d = json.loads(b.descriptor)
    assert any(k.startswith("RegisteredInput1_v1") for k in d["definitions"])
    assert [x for x in b.listing().split(b"\n")[2:2 + len(b.input_bits)]].count(b"registration") == 4


def test_a_wire_inside_a_two_unit_block_is_no_port():
    _, c, bits = toy("pair")
    b, u0 = R.restrict(c, TOY_PARTITION, [(0, 2)], bits), R.restrict(c, TOY_PARTITION, [(0, 1)], bits)
    assert set(u0.outputs) and not set(u0.outputs) & (set(b.outputs) | set(b.inputs))


def test_part_of_a_batch_is_one_call_a_member_and_all_of_it_the_batch():
    _, c, bits = toy("batch")
    forms = lambda units: [(n.form, n.n) for n in R.restrict(c, TOY_PARTITION, units, bits).program.root.body.nodes]
    assert forms([(0, 3)])[-1] == ("batch", 3)
    assert [f for f, _ in forms([(0, 2)])][-2:] == ["call", "call"] and forms([(1, 2)])[-1][0] == "call"


def test_the_bit_rule_is_one_function():
    assert R.split(16) == tuple(range(16))
    _, _, bits = chain()
    g = next(g for g, (k, _) in enumerate(bits.kind) if k == "input")
    assert bits.out[g] == tuple(bits.base[g] + j for j in R.split(16))
