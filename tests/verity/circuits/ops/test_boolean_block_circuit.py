"""`versions.boolean_table` and `versions.block_circuit` on the periods' chain: eight `Gemm_v2{K=16, N=16}` units at Ampere's
dot, one wire between them."""
from __future__ import annotations

from functools import partial

import pytest

from verity.circuits import boolean as BL
from verity.circuits import census as CE
from verity.circuits import count as K
from verity.circuits import program as PG
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.ops import gemm as G
from verity.circuits.ops.boolean import versions as V
from verity.circuits.program import NoBooleanBody, Program
from verity.circuits.types import Value
from verity.commitments.periods import interface as I

GEMM = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
GID = descriptor_id(GEMM)


def _chain() -> Program:
    n = 8
    wires = (I.Wire(GID, "x", GID, "o0", at=2, radix=(n - 2,), base=0, step=(1,)),)
    circuit = I.Circuit("chain", ((GEMM, n),), external=(f"{GID}/x",), wires=wires,
                        order=I.ordered({GID: n}, wires, {GID: range(0, n, 2)}))
    return circuit.program()[0]


def table_gates(fn, table: dict[str, int]) -> int:
    if isinstance(fn, PrimitiveDefinition):
        return table.get(descriptor_id(fn), 0)
    return sum((nd.n if nd.form in ("batch", "scan") else 1) * table_gates(nd.fn, table) for nd in fn.body.nodes)


def check_lifted(template, program: Program) -> object:
    """The template's block circuit from ``program``'s Boolean table, held to what its gate table (`count.table`) counts and to
    σ of every composite it reaches."""
    memo: dict = {}
    table = V.boolean_table(program)
    bits = V.block_circuit(template, table, memo)
    assert BL.is_boolean(bits)
    assert CE.boolean_gates([bits]) == [table_gates(template, K.table(program))]
    version = partial(V.block_circuit, table=table, memo=memo)
    for word in V._reachable([template]).values():
        if isinstance(word, PrimitiveDefinition) or BL.is_boolean(word):
            continue
        got = V.sigma_check(word, version(word), version)
        assert got["ok"], (descriptor_id(word), got["mismatches"])
    return bits


def test_every_primitive_of_the_chain_has_one_boolean_body_named_once():
    table = V.boolean_table(_chain())
    assert sorted(table) == ["AmpereBF16TcDot16_v2", "Const32[0x00000000]_v1", "F2fpBf16_v1"]
    assert {k: b.id for k, b in table.items()} == {"AmpereBF16TcDot16_v2": "AmpereBF16TcDot16_v3",
                                                  "Const32[0x00000000]_v1": "Const32[0x00000000]_v2",
                                                  "F2fpBf16_v1": "F2fpBf16_v2"}
    assert all(b.id == descriptor_id(b.fn) and len(b.digest) == 128 for b in table.values())
    with pytest.raises(TypeError):
        table["F2fpBf16_v1"] = table["F2fpBf16_v1"]  # type: ignore[index]


def test_the_chains_template_lifts_to_a_boolean_circuit_whose_gates_are_the_tables():
    bits = check_lifted(GEMM, _chain())
    assert CE.boolean_gates([bits]) == [400_256] and GEMM.gates == 48


def test_a_primitive_the_table_lacks_is_refused_by_name():
    table = dict(V.boolean_table(_chain()))
    del table["F2fpBf16_v1"]
    with pytest.raises(V.NoBooleanVersion, match="F2fpBf16_v1"):
        V.block_circuit(GEMM, table)


@pytest.fixture
def catalog_lookup():
    """`Program` building under the catalog's lookup (`versions.boolean_version`), as outside the suite, and the lookup the
    test found (the suite's stand-in, once `conftest.py` loads) restored after."""
    saved = list(PG._BODY_LOOKUP)
    PG.register_boolean_bodies(V.boolean_version)
    yield
    PG._BODY_LOOKUP[:] = saved


def test_a_program_with_a_word_primitive_with_no_boolean_body_is_refused_at_build_by_name(catalog_lookup):
    opaque = PrimitiveDefinition("BlockCircuitTestOpaque", 1, (("a", Value(8)),), Value(8), lambda a: a, register=False)
    wrap = bind(CompositeDefinition("BlockCircuitTestWrap", 1, (), lambda S: ((("a", Value(8)),), Value(8)),
                                    lambda B, S, a: B.call(opaque, a), register=False))
    with pytest.raises(NoBooleanBody, match="E_NO_BODY: .*BlockCircuitTestOpaque_v1"):
        Program(wrap)
