"""A Program holds no data (`verity.ml.circuits.constants`): what a program brings is a registered input, what a request brings a run
input, and a constant is an operation's own bits. `program_data` names every constant held anywhere else, by its Definition."""
from __future__ import annotations

import pytest

from verity.ml.circuits import codec
from verity.ml.circuits import constants as K
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.boolean import is_boolean
from verity.ml.circuits.evaluation import evaluate_region
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import REGISTERED_ID, Program, input_prim, is_input, is_registered
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.ops.scalar import I32Add

I32 = Value(32)
TEMP, TOP_P, SEVEN, POS, EPS = 0x3F4CCCCD, 0x3F733333, 7, 255, 0x3727C5AC


def _c(name, params, ret, body):
    return bind(CompositeDefinition(name, 1, (), lambda S: (params, ret), body, register=False))


#: (x + t) + 7: the 7 is the callee's own constant
Scale = _c("KScale", (("x", I32), ("t", I32)), I32, lambda B, S, x, t: B.call(I32Add, B.call(I32Add, x, t), B.call(const(32, SEVEN))))
#: (x + t) + eps, eps a parameter
ScaleBy = _c("KScaleBy", (("x", I32), ("t", I32), ("eps", I32)), I32,
             lambda B, S, x, t, eps: B.call(I32Add, B.call(I32Add, x, t), eps))


def _old(B, S, x):
    temp = B.call(const(32, TEMP))
    positions = B.batch(const(32, POS), axes=(), n=3)
    a = B.call(Scale, x, temp)
    b = B.call(Scale, a, positions[1])
    return tuple_of(b, B.call(I32Add, x, B.call(const(32, TOP_P))))


#: the program as it was: temperature, top-p and the Philox positions are root literals, and the 7 sits in KScale's body
Old = _c("KTop", (("x", I32),), Tuple(I32, I32), _old)


def _new(B, S, x, eps, temp, top_p, positions):
    a = B.call(ScaleBy, x, temp, eps)
    b = B.call(ScaleBy, a, positions[1], eps)
    return tuple_of(b, B.call(I32Add, x, top_p))


#: the same program holding no data: eps bound at registration; temperature, top-p and the positions per run
New = _c("KTopInputs", (("x", I32), ("eps", I32), ("temp", I32), ("top_p", I32), ("positions", Array(3, I32))),
         Tuple(I32, I32), _new)


@pytest.fixture(scope="module")
def new():
    return Program(New, registered=("eps",))


def test_a_program_holding_literals_holds_data():
    p = Program(Old)
    data = K.program_data(p.root)
    assert list(data) == [p.root.id, Scale.id]
    assert sorted(cid for _g, cid in data[p.root.id]) == sorted(
        [const(32, TEMP).id] + [const(32, POS).id] * 3 + [const(32, TOP_P).id])
    assert data[Scale.id] == [(1, const(32, SEVEN).id)]
    assert K.program_data(p.root, lambda fn: fn.id == Scale.id) == {p.root.id: data[p.root.id]}


#: an operation whose body calls KScale, and two programs: one reaching KScale only through it, one reaching it again outside
Op = _c("KOp", (("x", I32), ("t", I32)), I32, lambda B, S, x, t: B.call(Scale, x, t))
OnlyOp = _c("KOnlyOp", (("x", I32), ("t", I32)), I32, lambda B, S, x, t: B.call(Op, x, t))
Both = _c("KBoth", (("x", I32), ("t", I32)), Tuple(I32, I32), lambda B, S, x, t: tuple_of(B.call(Op, x, t), B.call(Scale, x, t)))


def test_a_definition_reached_first_under_an_operation_is_listed_on_a_later_path():
    """KScale is reached first inside the operation KOp, which the walk does not visit, and then from the root: that path
    lists its constant. Reached only through KOp, it holds none."""
    is_op = lambda fn: fn.id == Op.id  # noqa: E731
    assert K.program_data(Program(OnlyOp).root, is_op) == {}
    both = Program(Both)
    assert [n.fn.id for n in both.root.body.nodes if not is_input(n.fn)] == [Op.id, Scale.id]
    assert K.program_data(both.root, is_op) == {Scale.id: [(1, const(32, SEVEN).id)]}


#: a primitive that reads two tables by index, the library's and the program's
Lookup = PrimitiveDefinition("KLookup", 1, (("x", I32),), I32, lambda x: x, register=False,
                             tables=(("k_library", "library"), ("k_program", "registered")))
Reads = _c("KReads", (("x", I32),), I32, lambda B, S, x: B.call(Lookup, x))
ReadsUnderOp = _c("KReadsUnderOp", (("x", I32),), I32, lambda B, S, x: B.call(Reads, x))


def test_a_declared_table_is_data_unless_the_library_lists_it():
    root = Program(Reads).root
    assert K.program_data(root) == {Lookup.id: [(0, "k_library"), (0, "k_program")]}
    assert K.program_data(root, library=lambda n: n == "k_library") == {Lookup.id: [(0, "k_program")]}
    assert K.program_data(root, library=lambda n: True) == {Lookup.id: [(0, "k_program")]}
    assert K.program_data(Program(ReadsUnderOp).root, lambda fn: fn.id == Reads.id) == {}
    with pytest.raises(ValueError, match="table"):
        PrimitiveDefinition("KBadLookup", 1, (("x", I32),), I32, lambda x: x, register=False, tables=(("t", "public"),))


def test_a_strict_program_refuses_data(new):
    with pytest.raises(ValueError, match="holds data"):
        Program(Old, strict=K.program_data)
    assert Program(New, registered=("eps",), strict=K.program_data).bindings == new.bindings


def test_inputs_carry_the_values_and_the_program_holds_none(new):
    assert K.program_data(new.root) == {}
    assert new.bindings == (K.RUN, K.REGISTRATION, K.RUN, K.RUN, K.RUN)
    ins = K.inputs(new.circuit)
    assert ins == {K.REGISTRATION: [1], K.RUN: [0, 2, 3, 4, 5, 6]}
    for g in range(new.circuit.size):
        t = K.binding_time(new.circuit, g)
        assert (t is None) == (g not in ins[K.REGISTRATION] + ins[K.RUN])
    assert [n.fn.id for n in new.root.body.nodes if is_input(n.fn)] == [
        "Input32_v1", "RegisteredInput32_v1", "Input32_v1", "Input32_v1", "Input32_v1"]


@pytest.mark.parametrize("binding", K.BINDING_TIMES)
def test_an_input_node_appears_only_in_the_root(binding):
    """Red-team's L1 (`RgInner(x) = (x + In, x + In)`): an input node inside a Definition is a gate no parameter binds, so
    `inputs` would miss it while `binding_time` names it. The builder refuses it, directly under the root or deeper; the
    program function's own body is the root's, so there it is an input of the root."""
    inp = input_prim(32, binding)
    inner = _c(f"KInner_{binding}", (("x", I32),), Tuple(I32, I32),
               lambda B, S, x: tuple_of(B.call(I32Add, x, B.call(inp)), B.call(I32Add, x, B.call(inp))))
    outer = _c(f"KOuter_{binding}", (("x", I32),), Tuple(I32, I32), lambda B, S, x: B.call(inner, x))
    for fn in (outer, _c(f"KTop_{binding}", (("x", I32),), Tuple(I32, I32), lambda B, S, x: B.call(outer, x))):
        with pytest.raises(ValueError, match=rf"definition '{inner.id}' node 0: an input node \({inp.id}\) appears only in the root"):
            Program(fn)
    want = {K.REGISTRATION: [1, 3], K.RUN: [0]} if binding == K.REGISTRATION else {K.REGISTRATION: [], K.RUN: [0, 1, 3]}
    assert K.inputs(Program(inner).circuit) == want


@pytest.mark.parametrize("make", [lambda: Program(Old), lambda: Program(New), lambda: Program(New, registered=("eps",)),
                                  lambda: codec.decode_program(codec.encode_program(Program(New, registered=("eps",))), REGISTRY)])
def test_inputs_and_binding_time_agree(make):
    p = make()
    ins = K.inputs(p.circuit)
    assert {g: K.binding_time(p.circuit, g) for g in range(p.circuit.size) if K.binding_time(p.circuit, g)} == {
        g: when for when, gs in ins.items() for g in gs}


def test_registered_inputs_evaluate_and_cut_as_inputs(new):
    x = 5
    C = new.circuit
    t = evaluate_region(C, C[0:C.size], dict(enumerate([x, EPS, TEMP, TOP_P, POS, POS, POS])))
    got = [t[g] for g in C.root_outputs()]
    a = (x + TEMP + EPS) & 0xFFFFFFFF
    assert list(got) == [(a + POS + EPS) & 0xFFFFFFFF, (x + TOP_P) & 0xFFFFFFFF]
    assert [k for k, _fn in PO.calls(new)] == [5, 6, 7]
    run_only = Program(New)
    for q in (PO.query(), PO.call_query()):
        a, b = PO.evaluate(new, q), PO.evaluate(run_only, q)
        assert a.population == b.population and [a.locate(i) for i in range(a.population)] == [b.locate(i) for i in range(b.population)]
        assert PO.verify(PO.build(new, q), new).ok


def test_registered_inputs_round_trip_and_bind_the_digest(new):
    desc = codec.encode_program(new)
    back = codec.decode_program(desc, REGISTRY)
    assert [n.fn.id for n in back.root.body.nodes] == [n.fn.id for n in new.root.body.nodes]
    assert codec.encode_program(back) == desc
    assert codec.program_digest(desc) != codec.program_digest(codec.encode_program(Program(New)))


def test_registered_ids():
    for fid, ok in [("RegisteredInput32_v1", True), ("RegisteredInput1_v1", True), ("RegisteredInput032_v1", False),
                    ("RegisteredInput0_v1", False), ("RegisteredInput32_v2", False), ("Input32_v1", False),
                    ("RegisteredInput32_v1\n", False)]:
        assert bool(REGISTERED_ID.fullmatch(fid)) is ok, fid
    assert is_registered(input_prim(16, K.REGISTRATION)) and not is_registered(input_prim(16))
    assert is_input(input_prim(16, K.REGISTRATION)) and input_prim(16, K.REGISTRATION).binding == K.REGISTRATION
    assert is_boolean(input_prim(1, K.REGISTRATION)) and not is_boolean(input_prim(2, K.REGISTRATION))
    with pytest.raises(ValueError):
        Program(New, registered=("weights",))
    with pytest.raises(ValueError):
        input_prim(16, "build")
