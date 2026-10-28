"""Acceptance test: the circuit format's running example, exactly."""

from __future__ import annotations

import json
import random

import pytest

from verity.ml.circuits import (Array, FunctionCall, Program, Region, Tuple, Value, bind, composite, decode_program, descriptor_size,
                       encode_program, primitive, program_digest, query, GapError, QueryError, REGISTRY)
from verity.ml.circuits.defs import CompositeDefinition
from verity.ml.circuits.evaluation import evaluate_call, evaluate_region

V16 = Value(16)
MASK = 0xFFFF


@primitive("Mul16", 1, [("a", V16), ("b", V16)], V16, conformance="exact-model-tested")
def Mul16(a: int, b: int) -> int:
    return (a * b) & MASK


@primitive("Add16", 1, [("a", V16), ("b", V16)], V16, conformance="exact-model-tested")
def Add16(a: int, b: int) -> int:
    return (a + b) & MASK


@primitive("Relu16", 1, [("a", V16)], V16, conformance="exact-model-tested")
def Relu16(a: int) -> int:
    return 0 if a & 0x8000 else a


@primitive("Id16", 1, [("a", V16)], V16, conformance="exact-model-tested")
def Id16(a: int) -> int:
    return a


A2 = Array(2, V16)


@composite("Dot", 1, [], lambda S: ((("x", A2), ("w", A2)), V16))
def Dot(B, S, x, w):
    p0 = B.call(Mul16, x[0], w[0])
    p1 = B.call(Mul16, x[1], w[1])
    return B.call(Add16, p0, p1)


@composite("Row", 1, ["act"], lambda S: ((("x", A2), ("w", A2)), V16))
def Row(B, S, x, w):
    return B.call(S.act, B.call(Dot, x, w))


@composite("Linear", 1, ["M", "act"], lambda S: ((("x", A2), ("W", Array(S.M, A2))), Array(S.M, V16)))
def Linear(B, S, x, W):
    return B.batch(bind(Row, act=S.act), x, W, axes=(None, 0))


@composite("Inference", 1, ["B", "M", "act"],
           lambda S: ((("X", Array(S.B, A2)), ("W", Array(S.M, A2))), Array(S.B, Array(S.M, V16))))
def Inference(B, S, X, W):
    linear = bind(Linear, M=S.M, act=S.act)
    return B.batch(linear, X, W, axes=(0, None))


def make_program(act=Relu16, annotations=None) -> Program:
    return Program(bind(Inference, B=2, M=5, act=act), annotations=annotations)


def test_counts_and_layout():
    P = make_program()
    C = P.circuit
    assert C.size == 54
    assert P.input_gates == 14
    assert len(query(C, "//Input")) == 14
    assert len(query(C, "//Dot")) == 10
    assert len(query(C, "//Row")) == 10
    assert len(query(C, "//Linear")) == 2
    assert len(query(C, "//Mul16")) == 20
    # each call is one interval; Linear #0 = [14:34], Linear #1 = [34:54]
    assert [c.interval for c in query(C, "//Linear")] == [(14, 34), (34, 54)]
    rows = query(C, "//Row")
    assert [r.interval for r in rows] == [(14 + 4 * i, 18 + 4 * i) for i in range(10)]
    # gate i located by division: Row #3 = [26:30] -> Mul16, Mul16, Add16, Relu16
    assert [C[g].prim.id for g in range(26, 30)] == ["Mul16_v1", "Mul16_v1", "Add16_v1", "Relu16_v1"]
    # operand references are root-scoped gate identities: Input gates 0..3 are X, 4..13 are W
    g = C[14]  # first Mul16 of Linear #0, Row #0: x[0]*W[0][0]
    assert g.operands == (0, 4)
    assert C[15].operands == (1, 5)
    assert C[16].operands == (14, 15)
    assert C[17].operands == (16,)
    # Linear #1 shares W: its first Mul16 reads X[1][0] = Input 2 and W[0][0] = Input 4
    assert C[34].operands == (2, 4)


def test_groups_query_matches_spec():
    C = make_program().circuit
    gs = query(C, "//Linear ! groups(Row, 2)")
    assert [g.interval for g in gs] == [(14, 22), (22, 30), (30, 34), (34, 42), (42, 50), (50, 54)]
    assert all(isinstance(g, Region) for g in gs)
    # groups of 5: one per Linear; groups of 7 (nondivisible): still one block of 5 each
    assert [g.interval for g in query(C, "//Linear ! groups(Row, 5)")] == [(14, 34), (34, 54)]
    assert [g.interval for g in query(C, "//Linear ! groups(Row, 7)")] == [(14, 34), (34, 54)]
    assert [g.interval for g in query(C, "//Linear ! groups(Row, 3)")] == [(14, 26), (26, 34), (34, 46), (46, 54)]
    with pytest.raises(QueryError):
        query(C, "//Linear ! groups(Row, 0)")


def test_query_paths_and_scoping():
    C = make_program().circuit
    # /Linear : direct children of the root (batch members are direct children)
    assert [c.interval for c in query(C, "/Linear")] == [(14, 34), (34, 54)]
    assert query(C, "/Row") == []  # Rows are not direct children of the root
    assert [c.interval for c in query(C, "//Linear/Row")][:3] == [(14, 18), (18, 22), (22, 26)]
    assert len(query(C, "//Linear//Mul16")) == 20
    assert len(query(C, "//Linear/Mul16")) == 0
    assert len(query(C, "//Row/Dot")) == 10
    assert len(query(C, "//Row//Add16")) == 10
    assert query(C, "//Nothing") == []
    assert query(C, "//Nothing ! groups(Row, 2)") == []
    assert query(C, "//Linear ! groups(Nothing, 2)") == []  # empty selection -> no groups
    root = query(C, "/")
    assert len(root) == 1 and root[0].interval == (0, 54)
    with pytest.raises(QueryError):
        query(C, "")
    with pytest.raises(QueryError):
        query(C, "//Linear ! union(Row)")


def test_gap_rejected():
    # A body where two Row calls are separated by an unrelated gate: groups(Row, 2) must reject.
    @composite("Gappy", 1, [], lambda S: ((("x", A2), ("w", A2)), Array(2, V16)))
    def Gappy(B, S, x, w):
        r0 = B.call(bind(Row, act=Relu16), x, w)
        t = B.call(Id16, r0)
        r1 = B.call(bind(Row, act=Relu16), x, w)
        from verity.ml.circuits import array_of
        return array_of([t, r1])

    C = Program(bind(Gappy)).circuit
    with pytest.raises(GapError):
        query(C, "/ ! groups(Row, 2)")
    # k=1 never spans a gap
    assert [g.interval for g in query(C, "/ ! groups(Row, 1)")] == [(4, 8), (9, 13)]


def test_call_interface_vs_boundary():
    C = make_program().circuit
    lin0 = query(C, "//Linear")[0]
    assert isinstance(lin0, FunctionCall)
    # declared interface: x (2 leaves) then W (10 leaves), signature order
    assert lin0.args == tuple(range(0, 2)) + tuple(range(4, 14))
    assert lin0.returns == (17, 21, 25, 29, 33)
    assert lin0.in_ == tuple(range(0, 2)) + tuple(range(4, 14))
    assert lin0.out == (17, 21, 25, 29, 33)
    # Row #0 inside Linear #0: args = x[0], x[1], W[0][0], W[0][1]
    row0 = query(C, "//Row")[0]
    assert row0.args == (0, 1, 4, 5)
    assert row0.returns == (17,)
    # an explicit region's boundary is computed from wiring: [14:18] == Row #0
    reg = C[14:18]
    assert reg.in_ == (0, 1, 4, 5)
    assert reg.out == (17,)  # the Relu16 output is a root output
    # a non-contiguous subcircuit: the two Mul16 gates of Row #0
    S = C.gate_selection([15, 14])
    assert S.size == 2 and S[0].index == 14 and S[1].index == 15
    assert S.in_ == (0, 1, 4, 5)
    assert S.out == (14, 15)


def test_repeated_references_in_interface():
    # `@composite` registers globally: these names must not collide with the frontend examples' `F_v1`/`Top_v1`
    # (one id, one body — verity_vllm.program.frontend.check_identity)
    @composite("FRepeated", 1, [], lambda S: ((("a", V16), ("b", V16)), Tuple(V16, V16, V16)))
    def F(B, S, a, b):
        from verity.ml.circuits import tuple_of
        t = B.call(Add16, a, a)
        return tuple_of(t, a, t)

    @composite("TopRepeated", 1, [], lambda S: ((("v", V16),), V16))
    def Top(B, S, v):
        r = B.call(F, v, v)
        return B.call(Id16, r[0])

    C = Program(bind(Top)).circuit
    f = query(C, "//FRepeated")[0]
    assert f.args == (0, 0)  # [v, v]
    assert f.returns == (1, 0, 1)  # [t, v, t]
    assert f.in_ == (0,)  # distinct
    assert f.distinct_returns == (1,)
    assert f.out == (1,)  # the graph boundary: t is consumed by Id16


def test_evaluate_running_example():
    P = make_program()
    C = P.circuit
    rng = random.Random(1)
    X = [[rng.randrange(1 << 16) for _ in range(2)] for _ in range(2)]
    W = [[rng.randrange(1 << 16) for _ in range(2)] for _ in range(5)]
    t = {}
    for b in range(2):
        for k in range(2):
            t[b * 2 + k] = X[b][k]
    for m in range(5):
        for k in range(2):
            t[4 + m * 2 + k] = W[m][k]
    evaluate_region(C, C[0:54], t)
    outs = C.root_outputs()
    assert len(outs) == 10

    def ref(b, m):
        s = (X[b][0] * W[m][0] + X[b][1] * W[m][1]) & MASK
        return 0 if s & 0x8000 else s

    for b in range(2):
        for m in range(5):
            assert t[outs[b * 5 + m]] == ref(b, m)
    # evaluate a single Row call from bound arguments
    row = query(C, "//Row")[7]  # Linear #1, row 2
    got = evaluate_call(C, row, [X[1][0], X[1][1], W[2][0], W[2][1]])
    assert got == (ref(1, 2),)


def test_static_binding_change_keeps_identities_and_changes_digest():
    P1 = make_program(act=Relu16)
    P2 = make_program(act=Id16)
    assert P1.gates == P2.gates == 54
    q = "//Linear ! groups(Row, 2)"
    assert [g.interval for g in query(P1.circuit, q)] == [g.interval for g in query(P2.circuit, q)]
    assert [g.operands for g in P1.circuit[0:54]] == [g.operands for g in P2.circuit[0:54]]
    assert P1.circuit[17].prim.id == "Relu16_v1" and P2.circuit[17].prim.id == "Id16_v1"
    assert program_digest(P1) != program_digest(P2)


def test_annotations_do_not_change_digest_and_codec_round_trips():
    P = make_program()
    Pa = make_program(annotations={"note": "planner hint", "groups": [[14, 22]]})
    assert program_digest(P) == program_digest(Pa)
    desc = encode_program(P)
    s = json.dumps(desc)
    assert descriptor_size(P) > 0
    D = decode_program(json.loads(s), REGISTRY)
    assert D.gates == 54
    assert [g.interval for g in query(D.circuit, "//Linear ! groups(Row, 2)")] == \
        [(14, 22), (22, 30), (30, 34), (34, 42), (42, 50), (50, 54)]
    assert [g.operands for g in D.circuit[0:54]] == [g.operands for g in P.circuit[0:54]]
    assert [g.prim.id for g in D.circuit[0:54]] == [g.prim.id for g in P.circuit[0:54]]
    # re-encoding the decoded program yields the same digest
    assert program_digest(P) == program_digest(D)


def test_intrinsic_summary():
    lin = bind(Linear, M=5, act=Relu16)
    s = lin.intrinsic()
    assert s.gates == 20 and s.nodes == 1 and s.offsets == (0, 20)
    assert s.param_types == (A2, Array(5, A2))
