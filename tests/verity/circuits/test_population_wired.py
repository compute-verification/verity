from __future__ import annotations

import pytest

from verity.circuits.codec import descriptor_id, program_digest
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.evaluation.reference import evaluate
from verity.circuits.population import BitsWord, WordBits, mixed_population_program, wired_population_program
from verity.circuits.program import is_input
from verity.circuits.refs import Concat, Strided
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


def test_a_row_nothing_fills_is_free():
    rec = _record()
    reg = rec["registered"]
    reg["params"].update(x=4, k=4)
    reg["reads"] = [{**reg["reads"][0], "radix": [1]}] + [
        {"reader": IA, "port": p, "param": p, "at": 0, "radix": [2], "base": 0, "step": [2]} for p in "xk"]
    prog = wired_population_program([(A, 2), (B, 2)], rec)[0]
    assert prog.annotations["free"] == [{"reader": IB, "port": "w", "at": 1, "units": 1}]
    assert [n for n, _ in prog.fn.signature()[0]] == ["k", "w", "x", "1.w"]
    assert wired_population_program([(A, 2), (B, 2)], _record())[0].annotations["free"] == [
        {"reader": IA, "port": "x", "at": 0, "units": 2}, {"reader": IA, "port": "k", "at": 0, "units": 2}]


def test_a_row_two_fill_is_refused():
    rec = _record()
    rec["registered"]["reads"].append({"reader": IB, "port": "y", "param": "w", "at": 1, "radix": [1], "base": 0, "step": [0]})
    with pytest.raises(ValueError, match=r"TWiredB_v1 y unit 1 row 0 is filled twice"):
        wired_population_program([(A, 2), (B, 2)], rec)


def test_a_program_with_no_width_changing_wire_keeps_its_digest():
    assert program_digest(wired_population_program([(A, 2), (B, 2)], _record())[0]) == (
        "a0fafd0c4293444de2f08bb1c7405f91fbf17a83d8d70ad3c66abc55ff235bda2c7c2c95f053f2c799bcf88a21a6c9dbe6f9a5bdf81d4f820980ab8002ac5a0f")
    assert wired_population_program([(A, 2), (B, 2)], _record())[1]["params"]["templates"] == [IA, IB]


def _parts(refs):
    return [q for p in refs.parts for q in _parts(p)] if isinstance(refs, Concat) else [refs]


def _root(prog):
    nodes = [n for n in prog.root.body.nodes if not is_input(n.fn)]
    return nodes, [len(_parts(a.refs)) for n in nodes for a in n.args]


def _every_other(nb, order=None):
    rec = _record(na=2 * nb, nb=nb, step=2)
    return {**rec, **({"order": order} if order else {})}


@pytest.mark.parametrize("nb", [2, 8, 64])
def test_a_regular_run_of_edges_is_one_strided_part_whatever_the_units(nb):
    prog = wired_population_program([(A, 2 * nb), (B, nb)], _every_other(nb))[0]
    nodes, parts = _root(prog)
    assert len(nodes) == 2 and parts == [1, 1, 1, 1]
    assert isinstance(nodes[1].args[0].refs, Strided)
    w, x, k = list(range(2 * nb)), list(range(1, 4 * nb + 1)), list(range(100, 100 + 4 * nb))
    a = [evaluate(A, x[2 * u:2 * u + 2], k[2 * u:2 * u + 2]) for u in range(2 * nb)]
    b = [evaluate(B, a[2 * u], w[2 * u:2 * u + 2]) for u in range(nb)]
    assert evaluate(prog.fn, w, x, k) == tuple(v for out in a + b for v in out)


@pytest.mark.parametrize("one_writer", [False, True])
def test_a_reader_range_is_one_batch_or_with_one_writer_a_batch_for_each_writer_batch_it_reads(one_writer):
    order = [[IA, 0, 4], [IA, 4, 4], [IB, 0, 4]]
    prog = wired_population_program([(A, 8), (B, 4)], _every_other(4, order), one_writer=one_writer)[0]
    nodes, parts = _root(prog)
    if one_writer:
        assert [(n.fn.id, n.n) for n in nodes] == [(A.id, 4), (A.id, 4), (B.id, 2), (B.id, 2)] and parts == [1] * 8
    else:
        assert [(n.fn.id, n.n) for n in nodes] == [(A.id, 4), (A.id, 4), (B.id, 4)] and parts == [1, 1, 1, 1, 2, 1]
    w, x, k = list(range(8)), list(range(1, 17)), list(range(100, 116))
    a = [evaluate(A, x[2 * u:2 * u + 2], k[2 * u:2 * u + 2]) for u in range(8)]
    b = [evaluate(B, a[2 * u], w[2 * u:2 * u + 2]) for u in range(4)]
    assert evaluate(prog.fn, w, x, k) == tuple(v for out in a + b for v in out)


BIT = Value(1)
Xor = PrimitiveDefinition("TWiredXor1", 1, (("a", BIT), ("b", BIT)), BIT, lambda a, b: a ^ b, register=False)
D = bind(CompositeDefinition("TWiredBits", 1, (), lambda S: ((("b", Array(32, BIT)), ("s", Array(32, BIT))), Array(32, BIT)),
                             lambda B_, S, b, s: B_.batch(Xor, b, s, axes=(0, 0)), register=False))
ID = descriptor_id(D)


def _bits(words):
    return [(v >> i) & 1 for v in words for i in range(16)]


def _words(bits):
    return [sum(b << i for i, b in enumerate(bits[h:h + 16])) for h in range(0, len(bits), 16)]


def _widths(**wire):
    return {"templates": [[IA, 2], [ID, 2], [IB, 2]],
            "wires": [{"reader": ID, "port": "b", "writer": IA, "out": "o0", "at": 0, "radix": [2], "base": 0, "step": [1]},
                      {"reader": IB, "port": "y", "writer": ID, "out": "o0", "at": 0, "radix": [2], "base": 0, "step": [1],
                       **wire}],
            "external": [f"{IA}/x", f"{IA}/k", f"{IB}/w"],
            "registered": {"params": {"s": 4}, "tables": [], "roots": [], "openings": [],
                           "reads": [{"reader": ID, "port": "s", "param": "s", "at": 0, "radix": [2], "base": 0, "step": [2]}]}}


def test_a_wire_or_read_between_words_and_bits_passes_through_their_wiring_least_significant_bit_first():
    prog, query = wired_population_program([(A, 2), (D, 2), (B, 2)], _widths())
    assert [n for n, _ in prog.fn.signature()[0]] == ["s", "0.x", "0.k", "2.w"]
    assert query["params"]["templates"] == [IA, ID, IB, descriptor_id(bind(BitsWord, W=16)), descriptor_id(bind(WordBits, W=16))]
    s, x, k, w = [0x8001, 0x00FF, 0x1234, 0xFFFF], [1, 2, 3, 4], [100, 200, 300, 400], [7, 9, 11, 13]
    a = [evaluate(A, x[2 * u:2 * u + 2], k[2 * u:2 * u + 2]) for u in range(2)]
    d = [tuple(p ^ q for p, q in zip(_bits(a[u]), _bits(s[2 * u:2 * u + 2]))) for u in range(2)]
    b = [evaluate(B, _words(d[u]), w[2 * u:2 * u + 2]) for u in range(2)]
    assert evaluate(prog.fn, s, x, k, w) == tuple(v for out in a + d + b for v in out)


def test_a_wire_between_two_word_widths_is_refused():
    Neg = PrimitiveDefinition("TWiredNeg32", 1, (("a", Value(32)),), Value(32), lambda a: -a & 0xFFFFFFFF, register=False)
    G = bind(CompositeDefinition("TWiredWide", 1, (), lambda S: ((("z", Array(1, Value(32))),), Array(1, Value(32))),
                                 lambda B_, S, z: B_.batch(Neg, z, axes=(0,)), register=False))
    rec = {"templates": [[IA, 1], [descriptor_id(G), 1]], "external": [f"{IA}/x", f"{IA}/k"],
           "wires": [{"reader": descriptor_id(G), "port": "z", "writer": IA, "out": "o0", "at": 0, "radix": [1], "base": 0,
                      "step": [0]}]}
    with pytest.raises(ValueError, match="only between W-bit words and their one-bit leaves"):
        wired_population_program([(A, 1), (G, 1)], rec)


@pytest.mark.parametrize("w", [16, 32])
def test_word_bits_and_bits_word_are_each_others_inverse_bit_i_leaf_i(w):
    x = 0x8001 | 0x5A << 4
    bits = evaluate(bind(WordBits, W=w), x)
    assert bits == tuple((x >> i) & 1 for i in range(w)) and evaluate(bind(BitsWord, W=w), *bits) == (x,)
