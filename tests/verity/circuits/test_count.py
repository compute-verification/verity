"""`count`: a circuit's Boolean gate count from its own Boolean bodies, held to the census of the circuit lifted to bits and to
the gate table, on the periods' chain and three small Definitions; and `count_vectors.jsonl`, the vectors Lean's count reads
(`uv run python tests/verity/circuits/test_count.py --write` rewrites them)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from verity.circuits import boolean as BL
from verity.circuits import census as CE
from verity.circuits import count as K
from verity.circuits.codec import canonical_json, descriptor_id, encode_program
from verity.circuits.defs import REGISTRY, CompositeDefinition, bind
from verity.circuits.ops import gemm as G
from verity.circuits.ops.boolean import versions as V
from verity.circuits.program import Program, is_input
from verity.circuits.refs import Coll, tuple_of
from verity.circuits.types import Array, Tuple, Value
from verity.commitments.periods import interface as I

PATH = Path(__file__).parent / "count_vectors.jsonl"
REWRITE = "the count's vectors differ: a change meant to move them rewrites them (module docstring)"
F32 = Value(32)
ADD = REGISTRY["F32Add_v2"]
GEMM = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)


def _chain() -> Program:
    gid, n = descriptor_id(GEMM), 8
    wires = (I.Wire(gid, "x", gid, "o0", at=2, radix=(n - 2,), base=0, step=(1,)),)
    circuit = I.Circuit("chain", ((GEMM, n),), external=(f"{gid}/x",), wires=wires,
                        order=I.ordered({gid: n}, wires, {gid: range(0, n, 2)}))
    return circuit.program()[0]


def _wrap(name: str, params, ret, body) -> Program:
    return Program(bind(CompositeDefinition(name, 1, (), lambda S: (params, ret), body, register=False)))


def _one_add() -> Program:
    return _wrap("CountOneAdd", (("a", F32), ("b", F32)), F32, lambda B, S, a, b: B.call(ADD, a, b))


def _batch_of_adds() -> Program:
    def body(B, S, a, b):
        return B.batch(ADD, a, b, axes=(0, None), n=5)
    return _wrap("CountBatchOfAdds", (("a", Array(5, F32)), ("b", F32)), Array(5, F32), body)


STEP = CompositeDefinition("CountRunningSum", 1, (), lambda S: ((("s", F32), ("x", F32)), Tuple(F32, F32)),
                           lambda B, S, s, x: (lambda t: tuple_of(t, t))(B.call(ADD, s, x)), register=False)


def _scan_of_adds() -> Program:
    def body(B, S, s, xs):
        out = B.scan(bind(STEP), s, xs=[xs], n=7)
        return Coll(Array(7, F32), out.refs.slice(1, 8))
    return _wrap("CountScanOfAdds", (("s", F32), ("xs", Array(7, F32))), Array(7, F32), body)


CASES = {"chain": _chain, "one_add": _one_add, "batch_of_adds": _batch_of_adds, "scan_of_adds": _scan_of_adds}


def _lifted(program: Program) -> int:
    """The census of the root's Calls lifted to bits by the circuit's own bodies (`versions.block_circuit`): an independent path."""
    table = V.boolean_table(program)
    memo: dict = {}
    return sum(CE.boolean_gates([V.block_circuit(nd.fn, table, memo)])[0] * (nd.n if nd.form in ("batch", "scan") else 1)
               for nd in program.root.body.nodes if not is_input(nd.fn))


def _vector(name: str) -> dict:
    program = CASES[name]()
    desc = encode_program(program)
    got = K.count(canonical_json(desc))
    root = descriptor_id(program.root)
    return {"name": name, "descriptor": canonical_json(desc).decode(), "root": root,
            "count": {k: [c.gates, c.ands] for k, c in sorted(got.items())}, "table": K.table(program)}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_count_from_the_descriptors_bytes_is_the_programs_and_the_lifted_circuits_census(name):
    program = CASES[name]()
    built, decoded = K.count(program), K.count(canonical_json(encode_program(program)))
    assert built == decoded
    assert built[descriptor_id(program.root)].gates == _lifted(program)


def test_the_chain_counts_each_template_once_and_its_eight_members_by_multiplying():
    got = K.count(_chain())
    assert got[descriptor_id(GEMM)] == K.Count(400_256, 133_216) and GEMM.gates == 48
    assert got[descriptor_id(_chain().root)] == got[descriptor_id(GEMM)] * 8


def test_a_batch_and_a_scan_multiply_their_callees_count():
    one = K.count(_one_add())[ADD.id]
    assert one == K.Count(2257, 781)
    assert K.count(_batch_of_adds())[descriptor_id(_batch_of_adds().root)] == one * 5
    assert K.count(_scan_of_adds())[descriptor_id(_scan_of_adds().root)] == one * 7


def test_the_table_is_the_gate_tables_on_each_word_primitive():
    for name in sorted(CASES):
        program = CASES[name]()
        table, unpriced = V.gate_table([program.root])
        assert not unpriced and K.table(program) == {k: v for k, v in table.items() if k in program.boolean}, name


def test_the_rule_is_the_censuss():
    assert K.gate(BL.xor_gate(3)) == K.Count(2, 0) and K.gate(BL.NOT) == K.Count(1, 0)
    assert K.gate(BL.and_gate(4)) == K.Count(3, 3) and K.gate(BL.constant(1)) == K.ZERO
    assert K.gate(ADD) is None
    assert CE.boolean_gates([BL.xor_gate(3), BL.NOT, BL.and_gate(4)]) == [2, 1, 3]


def test_a_word_primitive_without_a_body_is_refused_by_name():
    program = _one_add()
    program.boolean = {}
    with pytest.raises(K.NoBody, match="F32Add_v2"):
        K.count(program)


def test_the_vectors_are_the_counts():
    got = [json.loads(line) for line in PATH.read_text().splitlines()]
    assert [v["name"] for v in got] == sorted(CASES), REWRITE
    for v in got:
        assert v == _vector(v["name"]), REWRITE


if __name__ == "__main__" and sys.argv[1:] == ["--write"]:
    PATH.write_text("".join(json.dumps(_vector(n), sort_keys=True) + "\n" for n in sorted(CASES)))
