"""`verity.circuits.restrict` on the periods' chain: eight `Gemm_v2{K=16, N=16}` units at Ampere's dot, one wire between
consecutive units, blocks of two units and of one."""
from __future__ import annotations

import functools
import importlib.util
import random
from pathlib import Path

import pytest

from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks
from verity.circuits.evaluation.reference import evaluate_region
from verity.circuits.program import input_id

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
    assert R.evaluate(b.text, [bit(t, bits, x) for x in b.inputs]) == [bit(t, bits, x) for x in b.outputs]


def test_the_blocks_gates_are_Cs_computed_gates_each_once():
    _, _, bits = chain()
    computed = {x for g, (k, _) in enumerate(bits.kind) if k != "input"
                for x in range(bits.base[g], bits.base[g + 1] if g + 1 < len(bits.base) else bits.size)}
    for blocks in (MEMBERS, UNITS):
        got = [x for u in blocks for x in block(tuple(u)).gate_indices()]
        assert len(got) == len(set(got)) and set(got) == computed


@pytest.mark.parametrize("units", MEMBERS + UNITS)
def test_the_ports_are_the_boundary_in_increasing_index_and_the_map_is_injective(units):
    _, _, bits = chain()
    b = block(tuple(units))
    gates = b.gate_indices()
    assert list(b.inputs) == sorted(set(b.inputs)) and gates == sorted(set(gates)) and not set(b.inputs) & set(gates)
    assert list(b.outputs) == sorted(set(b.outputs)) and set(b.outputs) <= set(gates)
    (lo, hi), = R.unit_gates(bits.D, PARTITION, units)
    region = bits.C.region(lo, hi)
    assert sorted({bits.gate_of(x) for x in b.inputs}) == list(region.in_)
    assert sorted({bits.gate_of(x) for x in b.outputs}) == list(region.out)
    lines = b.text.decode().splitlines()
    n_in, n_gates, n_out = map(int, lines[1].split())
    assert (n_in, n_gates, n_out) == (len(b.inputs), len(gates), len(b.outputs))
    outs = [int(s.split()[1]) for s in lines[2 + n_in + n_gates:]]
    assert [gates[k - n_in] for k in outs] == list(b.outputs)


def test_a_wire_inside_a_block_is_no_port_unless_C_returns_it():
    _, _, bits = chain()
    root = {x for g in bits.C.root_outputs() for x in bits.out[g]}
    pair, first = block(tuple(MEMBERS[0])), block(tuple(UNITS[0]))
    inner = set(first.outputs) - set(pair.outputs)
    assert inner == set(first.outputs) - root
    assert not inner & set(pair.inputs)


def test_the_text_is_a_function_of_C_and_the_units():
    program, c, _ = chain()
    again = b"".join(descriptor_chunks(_chain()))
    assert again == c and R.restrict(again, PARTITION, MEMBERS[1]).text == block(tuple(MEMBERS[1])).text


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


def test_the_bit_rule_is_one_function():
    assert R.split(16) == tuple(range(16))
    _, _, bits = chain()
    g = next(g for g, (k, _) in enumerate(bits.kind) if k == "input")
    assert bits.out[g] == tuple(bits.base[g] + j for j in R.split(16))
