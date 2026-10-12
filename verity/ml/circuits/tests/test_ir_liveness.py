"""`verity.ml.circuits.liveness`: dead gates computed over the hierarchy against a flat scan of the lowered circuit, in both
senses (no consumer / no path to a protected output), and the per-parameter reads against the gate operands."""
from __future__ import annotations

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.liveness import DeadGateReport, Liveness, dead_gates
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

V16, V32 = Value(16), Value(32)
A2, A16 = Array(2, V16), Array(16, V16)


def prim(name: str, params, ret, fn) -> PrimitiveDefinition:
    return PrimitiveDefinition(name, 1, tuple(params), ret, fn, register=False)


def comp(name: str, params, ret, body) -> CompositeDefinition:
    return CompositeDefinition(name, 1, (), lambda S: (tuple(params), ret), body, register=False)


Mul16 = prim("TMul16", [("a", V16), ("b", V16)], V16, lambda a, b: (a * b) & 0xFFFF)
Add16 = prim("TAdd16", [("a", V16), ("b", V16)], V16, lambda a, b: (a + b) & 0xFFFF)
Relu16 = prim("TRelu16", [("a", V16)], V16, lambda a: 0 if a & 0x8000 else a)
Add32 = prim("TAdd32", [("a", V32), ("b", V32)], V32, lambda a, b: (a + b) & 0xFFFFFFFF)
Mul32 = prim("TMul32", [("a", V32), ("b", V32)], V32, lambda a, b: (a * b) & 0xFFFFFFFF)
Zero32 = prim("TZero32", [], V32, lambda: 0)
Dot16 = prim("TDot16", [("acc", V32), ("a", A16), ("b", A16)], V32, lambda acc, *ab: acc)
Cvt16 = prim("TCvt16", [("a", V32)], V16, lambda a: a >> 16)
Gather16 = prim("TGather16", [("i", V32), ("xs", Array(16, V16))], V16, lambda i, *xs: xs[i % 16])

# the IR running example: Inference{B, M, act} = batch of Linear{M} = batch of Row = act(Dot(x, w))
Dot2 = comp("TDot2", [("x", A2), ("w", A2)], V16,
            lambda Bd, S, x, w: Bd.call(Add16, Bd.call(Mul16, x[0], w[0]), Bd.call(Mul16, x[1], w[1])))
Row = CompositeDefinition("TRow", 1, ("act",), lambda S: ((("x", A2), ("w", A2)), V16),
                          lambda Bd, S, x, w: Bd.call(S.act, Bd.call(Dot2, x, w)), register=False)
Linear = CompositeDefinition("TLinear", 1, ("M", "act"), lambda S: ((("x", A2), ("W", Array(S.M, A2))), Array(S.M, V16)),
                             lambda Bd, S, x, W: Bd.batch(bind(Row, act=S.act), x, W, axes=(None, 0)), register=False)
Inference = CompositeDefinition("TInference", 1, ("B", "M", "act"),
                                lambda S: ((("X", Array(S.B, A2)), ("W", Array(S.M, A2))), Array(S.B, Array(S.M, V16))),
                                lambda Bd, S, X, W: Bd.batch(bind(Linear, M=S.M, act=S.act), X, W, axes=(0, None)),
                                register=False)


def _dot(Bd, S, x, w):
    acc = Bd.call(Zero32)
    for s in range(S.K // 16):
        acc = Bd.call(Dot16, acc, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
    return acc


Dot = CompositeDefinition("TDot", 1, ("K",), lambda S: ((("x", Array(S.K, V16)), ("w", Array(S.K, V16))), V32), _dot,
                          register=False)
Coord = CompositeDefinition("TGemmCoordinate", 1, ("K",), lambda S: ((("x", Array(S.K, V16)), ("wrow", Array(S.K, V16))), V16),
                            lambda Bd, S, x, wrow: Bd.call(Cvt16, Bd.call(bind(Dot, K=S.K), x, wrow)), register=False)
Gemm = CompositeDefinition("TGemm", 1, ("K", "N"),
                           lambda S: ((("x", Array(S.K, V16)), ("w", Array(S.N, Array(S.K, V16)))), Array(S.N, V16)),
                           lambda Bd, S, x, w: Bd.batch(bind(Coord, K=S.K), x, w, axes=(None, 0)), register=False)


def _butterfly(Bd, S, x):
    """An xor-butterfly all-reduce over L lanes of which only lane 0 of the last step is returned."""
    lanes = [x[m] for m in range(S.L)]
    h = 1
    while h < S.L:
        lanes = [Bd.call(Add32, lanes[m], lanes[m ^ h]) for m in range(S.L)]
        h *= 2
    return lanes[0]


Butterfly = CompositeDefinition("TButterfly", 1, ("L",), lambda S: ((("x", Array(S.L, V32)),), V32), _butterfly, register=False)
Embedding = CompositeDefinition(
    "TEmbedding", 1, (), lambda S: ((("tok", V32), ("table", Array(16, Array(8, V16)))), Array(8, V16)),
    lambda Bd, S, tok, table: Bd.batch(Gather16, tok, Coll(Array(8, Array(16, V16)), strided_view(table, 0, 16, 1, 8, 128)), axes=(None, 0)),
    register=False)
Inner = comp("TInner", [("x", V32)], Tuple(V32, V32), lambda Bd, S, x: tuple_of(Bd.call(Add32, x, x), Bd.call(Mul32, x, x)))
Middle = comp("TMiddle", [("x", V32)], Tuple(V32, V32), lambda Bd, S, x: (lambda r: tuple_of(r[1], r[0]))(Bd.call(Inner, x)))
StepPT = comp("TStepPT", [("carry", Tuple(V32, V32)), ("x", V32)], Tuple(Tuple(V32, V32), V32),
              lambda Bd, S, carry, x: (lambda s: tuple_of(tuple_of(carry[0], s), s))(Bd.call(Add32, carry[1], x)))
Row3 = comp("TRow3", [("x", Array(3, V32))], V32, lambda Bd, S, x: Bd.call(Add32, x[0], x[2]))
Col4 = comp("TCol4", [("x", Array(4, V32))], V32, lambda Bd, S, x: Bd.call(Add32, x[1], x[3]))


def _views(Bd, S, M):
    gap = Bd.batch(Row3, Coll(Array(4, Array(3, V32)), strided_view(M, 0, 3, 6, 2, 12)), axes=(0,))
    cols = Bd.batch(Col4, Coll(Array(3, Array(4, V32)), strided_view(M, 0, 4, 2, 6, 12)), axes=(0,))
    return tuple_of(gap, cols)


def _scan_pt(Bd, S, init, xs):
    res = Bd.scan(StepPT, init, xs=(xs,))
    return tuple_of(res[0][0], res[1][0])


def programs():
    yield "running example", Program(bind(Inference, B=2, M=5, act=Relu16))
    yield "gemm", Program(bind(Gemm, K=32, N=6))
    yield "butterfly", Program(bind(Butterfly, L=32))
    yield "embedding", Program(bind(Embedding))
    yield "dead return", Program(comp("TDeadRet", [("x", Array(1, V32))], V32,
                                      lambda Bd, S, x: (lambda r: Bd.call(Add32, r[0], r[0]))(Bd.call(Inner, x[0]))))
    yield "re-returned nested return", Program(comp("TReRet", [("x", Array(1, V32))], V32, lambda Bd, S, x: Bd.call(Middle, x[0])[0]))
    yield "dead inputs", Program(comp("TDeadIn", [("x", Array(4, V32))], V32, lambda Bd, S, x: Bd.call(Add32, x[0], x[1])))
    yield "strided views", Program(comp("TViews", [("M", Array(4, Array(6, V32)))], Tuple(Array(4, V32), Array(3, V32)), _views))
    yield "scan with pass-through carry", Program(comp("TScanPT", [("init", Tuple(V32, V32)), ("xs", Array(3, V32))],
                                                       Tuple(V32, V32), _scan_pt))


def brute(P: Program, transitive: bool) -> list[int]:
    C = P.circuit
    n = C.size
    if not transitive:
        consumed = bytearray(n)
        for g in range(n):
            for o in C.gate(g).operands:
                consumed[o] = 1
        for o in C.root_outputs():
            consumed[o] = 1
        return [g for g in range(n) if not consumed[g]]
    live = bytearray(n)
    for o in C.root_outputs():
        live[o] = 1
    for g in range(n - 1, -1, -1):
        if live[g]:
            for o in C.gate(g).operands:
                live[o] = 1
    return [g for g in range(n) if not live[g]]


def test_dead_gates_match_brute_force_in_both_senses():
    """Dead (no consumer, not protected) and unreachable (no path to a protected output), hierarchically computed,
    equal the flat scan of the lowered circuit, ids included."""
    for label, P in programs():
        for transitive in (False, True):
            r = dead_gates(P, limit=10 ** 9, transitive=transitive)
            expect = brute(P, transitive)
            assert r.total == len(expect), (label, transitive)
            assert r.ids == expect, (label, transitive)
            assert r.ids_complete and not r.approximate, (label, transitive)
            assert r.input == sum(1 for g in expect if g < P.input_gates), (label, transitive)
            assert r.on_s_out == 0  # a root output is protected, hence never dead


def test_transitive_dead_gates_are_a_superset():
    for label, P in programs():
        direct, trans = dead_gates(P, limit=10 ** 9), dead_gates(P, limit=10 ** 9, transitive=True)
        assert set(direct.ids) <= set(trans.ids), label


def test_dead_gate_ids_are_a_prefix_under_a_limit():
    P = Program(bind(Butterfly, L=32))
    full = dead_gates(P, limit=10 ** 9)
    part = dead_gates(P, limit=10)
    assert part.total == full.total == 31  # the lanes other than lane 0 at the last step
    assert part.ids == full.ids[:10] and not part.ids_complete
    assert dead_gates(P, limit=0).ids == [] and dead_gates(P, limit=0).total == 31


def test_report_json():
    r = dead_gates(Program(comp("TDeadIn2", [("x", Array(4, V32))], V32, lambda Bd, S, x: Bd.call(Add32, x[0], x[1]))))
    assert isinstance(r, DeadGateReport)
    assert r.to_json() == {"dead_computed_gates": 0, "dead_input_gates": 2, "dead_gates": 2, "ids": [2, 3],
                           "ids_complete": True, "on_S_out": 0, "approximate": False}


def test_parameter_reads_match_gate_operands():
    """`Liveness.analyse(spec, all returns live).reads` = per parameter, the leaves some gate of the lowered Program
    reads (every consumer counts, dead ones included)."""
    for spec in (bind(Inference, B=2, M=5, act=Relu16), bind(Gemm, K=32, N=6), bind(Butterfly, L=32), bind(Embedding)):
        P = Program(spec)
        C = P.circuit
        read = bytearray(C.size)
        for g in range(C.size):
            for o in C.gate(g).operands:
                read[o] = 1
        n = spec.body.ret_type.leaves
        reads = Liveness().analyse(spec, ((0, n),) if n else ()).reads
        assert len(reads) == len(spec.signature()[0])
        for pi, iv in enumerate(reads):
            gates = [ref[2] for ref in P.input_coll(pi).refs]
            expect = [leaf for leaf, g in enumerate(gates) if read[g]]
            assert [leaf for lo, hi in iv for leaf in range(lo, hi)] == expect, (spec.id, pi)
