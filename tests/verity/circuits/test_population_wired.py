from __future__ import annotations

import pytest

from verity.circuits.codec import descriptor_id, program_digest
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.evaluation.reference import evaluate
from verity.circuits.population import mixed_population_program, wired_population_program
from verity.circuits.types import Array, Value

V16 = Value(16)
Add = PrimitiveDefinition("TWiredAdd16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: (a + b) & 0xFFFF, register=False)
Mul = PrimitiveDefinition("TWiredMul16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: (a * b + 1) & 0xFFFF, register=False)


def _pair(name, prim, ports):
    return bind(CompositeDefinition(name, 1, (), lambda S: (tuple((p, Array(2, V16)) for p in ports), Array(2, V16)),
                                    lambda B, S, a, b: B.batch(prim, a, b, axes=(0, 0)), register=False))


A, B = _pair("TWiredA", Add, ("x", "k")), _pair("TWiredB", Mul, ("y", "w"))
IA, IB = descriptor_id(A), descriptor_id(B)


def _record(na=2, nb=2, base=0, step=1, wires=True, reads=True, external=(f"{IA}/x", f"{IA}/k")):
    wire = {"reader": IB, "port": "y", "writer": IA, "out": "o0", "at": 0, "radix": [nb], "base": base, "step": [step]}
    read = {"reader": IB, "port": "w", "param": "w", "at": 0, "radix": [nb], "base": 0, "step": [2]}
    return {"templates": [[IA, na], [IB, nb]], "wires": [wire] if wires else [], "external": list(external),
            "registered": {"params": {"w": 2 * nb}, "tables": [], "reads": [read] if reads else [], "roots": [],
                           "openings": []}}


def test_the_wired_program_is_its_members_evaluated_in_the_records_order():
    prog = wired_population_program([(A, 2), (B, 2)], _record())[0]
    assert [n for n, _ in prog.fn.signature()[0]] == ["w", "0.x", "0.k"]
    w, x, k = [7, 9, 11, 13], [1, 2, 3, 4], [100, 200, 300, 400]
    a = [evaluate(A, x[2 * u:2 * u + 2], k[2 * u:2 * u + 2]) for u in range(2)]
    b = [evaluate(B, a[u], w[2 * u:2 * u + 2]) for u in range(2)]
    assert evaluate(prog.fn, w, x, k) == tuple(v for out in a + b for v in out)


def test_its_digest_moves_with_a_wire_and_with_a_count():
    digest = program_digest(wired_population_program([(A, 2), (B, 2)], _record())[0])
    assert program_digest(wired_population_program([(A, 2), (B, 2)], _record(base=1, step=-1))[0]) != digest
    assert program_digest(wired_population_program([(A, 3), (B, 2)], _record(na=3))[0]) != digest


def test_with_no_wires_and_no_reads_its_inputs_are_the_populations():
    rec = {**_record(wires=False, reads=False, external=[f"{t}/{p}" for t, ps in ((IA, "xk"), (IB, "yw")) for p in ps]),
           "registered": {"params": {"w": 4}, "tables": [], "reads": [], "roots": [], "openings": [], "values": ["w"]}}
    wired, mixed = wired_population_program([(A, 2), (B, 2)], rec)[0], mixed_population_program([(A, 2), (B, 2)],
                                                                                           values=(("w", 4, 16),))
    assert wired.fn.signature() == mixed.fn.signature()


def test_an_input_nothing_fills_is_refused_by_name():
    with pytest.raises(ValueError, match=r"TWiredA_v1 k unit 0 row 0: no wire"):
        wired_population_program([(A, 2), (B, 2)], _record(external=(f"{IA}/x",)))
    with pytest.raises(ValueError, match=r"TWiredB_v1 w unit 0 row 0: no wire"):
        wired_population_program([(A, 2), (B, 2)], _record(reads=False))
