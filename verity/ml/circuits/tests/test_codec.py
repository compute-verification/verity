"""Descriptor codec: nested-static dependency discovery (frontend rulings §8.3), identity conflicts (§8.1), and the
reference normalisation of `concat`.

Ported from veritor ``verity.ir/tests/test_codec.py`` (1-114, 131-224).  The tests over veritor's model registry (B1
attention padding, KV-cache aliasing, the pinned v0 digests) stayed behind; the one `b1.Gemm` reference became a local
composite."""

from __future__ import annotations

import json

import pytest

from verity.ml.circuits import Array, Program, Value, bind, composite, decode_program, encode_program, program_digest, REGISTRY
from verity.ml.circuits.codec import IdentityConflict
from verity.ml.circuits.defs import CompositeDefinition

from test_running_example import Add16, Dot, Id16, Mul16, Relu16

V16 = Value(16)
A2 = Array(2, V16)


# -- a function bundle: `ops` is a record-valued static holding function references ----------------------------

@composite("DotSwapped", 1, [], lambda S: ((("x", A2), ("w", A2)), V16))
def DotSwapped(B, S, x, w):
    """Same function as `Dot`, different body (products added in the other order)."""
    p0 = B.call(Mul16, x[0], w[0])
    p1 = B.call(Mul16, x[1], w[1])
    return B.call(Add16, p1, p0)


@composite("RowOps", 1, ["ops"], lambda S: ((("x", A2), ("w", A2)), V16))
def RowOps(B, S, x, w):
    return B.call(S.ops["act"], B.call(S.ops["dot"], x, w))


@composite("LinearOps", 1, ["M", "ops"], lambda S: ((("x", A2), ("W", Array(S.M, A2))), Array(S.M, V16)))
def LinearOps(B, S, x, W):
    return B.batch(bind(RowOps, ops=S.ops), x, W, axes=(None, 0))


@composite("TwoLinears", 1, ["M", "opsA", "opsB"],
           lambda S: ((("x", A2), ("W", Array(S.M, A2))), Array(2, Array(S.M, V16))))
def TwoLinears(B, S, x, W):
    """`bind(Row, ops=opsA)` and `bind(Row, ops=opsB)` coexist in one program (ruling §8.3)."""
    a = B.call(bind(LinearOps, M=S.M, ops=S.opsA), x, W)
    b = B.call(bind(LinearOps, M=S.M, ops=S.opsB), x, W)
    return B.call(bind(Stack2, M=S.M), a, b)


@composite("Stack2", 1, ["M"], lambda S: ((("a", Array(S.M, V16)), ("b", Array(S.M, V16))), Array(2, Array(S.M, V16))))
def Stack2(B, S, a, b):
    from verity.ml.circuits.refs import array_of
    return array_of([a, b])


OPS_A = {"dot": Dot, "act": Relu16}
OPS_B = {"dot": DotSwapped, "act": Id16}


def _same_circuit(P, D) -> None:
    n = P.gates
    assert D.gates == n
    assert [g.prim.id for g in D.circuit[0:n]] == [g.prim.id for g in P.circuit[0:n]]
    assert [tuple(g.operands) for g in D.circuit[0:n]] == [tuple(g.operands) for g in P.circuit[0:n]]
    assert D.circuit.root_outputs() == P.circuit.root_outputs()


def test_nested_static_functions_are_definitions_and_the_bundles_coexist():
    P = Program(bind(TwoLinears, M=3, opsA=OPS_A, opsB=OPS_B))
    desc = encode_program(P)
    ids = set(desc["definitions"])
    # one definition per function, referenced from both the bundle and the body nodes (sharing preserved)
    assert {"Dot_v1", "DotSwapped_v1", "Relu16_v1", "Id16_v1"} <= ids
    rows = sorted(i for i in ids if i.startswith("RowOps_v1{"))
    assert len(rows) == 2 and all("Dot" in r and ("Relu16" in r or "Id16" in r) for r in rows)
    lins = [i for i in ids if i.startswith("LinearOps_v1{")]
    assert len(lins) == 2
    for fid in rows + lins:
        assert desc["definitions"][fid]["bindings"]["ops"]["dict"]["dot"]["fn"] in ("Dot_v1", "DotSwapped_v1")
    # the bundle's functions are resolved into `definitions`, so decoding needs nothing outside the descriptor
    D = decode_program(json.loads(json.dumps(desc)), REGISTRY)
    _same_circuit(P, D)
    assert program_digest(D) == program_digest(P)
    # the decoded bundles are dicts of definitions again
    lin = D.defs[lins[0]]
    assert set(lin.bindings["ops"]) == {"dot", "act"}
    # the two Rows compute different functions: same gate count, different primitive at the activation
    prims = [g.prim.id for g in P.circuit[0:P.gates]]
    assert prims.count("Relu16_v1") == 3 and prims.count("Id16_v1") == 3


def test_function_in_a_bundle_that_no_node_calls_is_still_resolved():
    """Discovery follows the static value, not only the body nodes: an entry of the bundle that this body never
    calls (here `unused`) is a definition of the descriptor, and decoding resolves it."""

    @composite("RowOpsPartial", 1, ["ops"], lambda S: ((("x", A2), ("w", A2)), V16))
    def RowOpsPartial(B, S, x, w):
        return B.call(S.ops["act"], B.call(S.ops["dot"], x, w))

    ops = {"dot": Dot, "act": Relu16, "unused": (Id16, DotSwapped)}

    @composite("TopPartial", 1, ["ops"], lambda S: ((("x", A2), ("w", A2)), V16))
    def TopPartial(B, S, x, w):
        return B.call(bind(RowOpsPartial, ops=S.ops), x, w)

    P = Program(bind(TopPartial, ops=ops))
    desc = encode_program(P)
    assert {"Id16_v1", "DotSwapped_v1"} <= set(desc["definitions"])
    D = decode_program(json.loads(json.dumps(desc)), REGISTRY)
    _same_circuit(P, D)
    row = next(v for k, v in D.defs.items() if k.startswith("RowOpsPartial_v1{"))
    assert [f.id for f in row.bindings["ops"]["unused"]] == ["Id16_v1", "DotSwapped_v1"]



def _dot_like(body, name="DotConflict"):
    return CompositeDefinition(name, 1, (), lambda S: ((("x", A2), ("w", A2)), V16), body, register=False)


def _dot_body_a(B, S, x, w):
    return B.call(Add16, B.call(Mul16, x[0], w[0]), B.call(Mul16, x[1], w[1]))


def _dot_body_b(B, S, x, w):
    return B.call(Add16, B.call(Mul16, x[1], w[1]), B.call(Mul16, x[0], w[0]))


def test_two_bodies_for_one_fully_qualified_specialization_are_rejected():
    """Ruling §8.1: within one program a fully qualified specialization resolves to one body; a conflicting
    definition is an `IdentityConflict`, never first-wins."""
    da, db = _dot_like(_dot_body_a), _dot_like(_dot_body_b)
    top = CompositeDefinition("TopConflict", 1, (), lambda S: ((("x", A2), ("w", A2)), Array(2, V16)),
                              lambda B, S, x, w: __import__("verity.ml.circuits.refs", fromlist=["array_of"]).array_of(
                                  [B.call(da, x, w), B.call(db, x, w)]), register=False)
    P = Program(bind(top))
    with pytest.raises(IdentityConflict, match="DotConflict_v1"):
        encode_program(P)
    # ... whichever order the two are met in
    top2 = CompositeDefinition("TopConflict2", 1, (), lambda S: ((("x", A2), ("w", A2)), Array(2, V16)),
                               lambda B, S, x, w: __import__("verity.ml.circuits.refs", fromlist=["array_of"]).array_of(
                                   [B.call(db, x, w), B.call(da, x, w)]), register=False)
    with pytest.raises(IdentityConflict):
        encode_program(Program(bind(top2)))
    # the same body under two distinct definition objects is one definition, not a conflict
    da2 = _dot_like(_dot_body_a)
    top3 = CompositeDefinition("TopSame", 1, (), lambda S: ((("x", A2), ("w", A2)), Array(2, V16)),
                               lambda B, S, x, w: __import__("verity.ml.circuits.refs", fromlist=["array_of"]).array_of(
                                   [B.call(da, x, w), B.call(da2, x, w)]), register=False)
    desc = encode_program(Program(bind(top3)))
    assert desc["definitions"]["DotConflict_v1"]["form"] == "composite"


def test_conflict_through_a_static_bundle_is_rejected_by_the_binding_cache():
    """Function-valued statics are keyed by id, so `bind(Linear, ops={"dot": da})` and `bind(Linear, ops={"dot":
    db})` hit the same cache entry; with `da`/`db` different bodies under one id the second bind is a conflict
    (not a silent reuse of the first body — that would be first-wins one level below the codec)."""
    da, db = _dot_like(_dot_body_a), _dot_like(_dot_body_b)
    with pytest.raises(IdentityConflict, match="LinearOps_v1"):
        Program(bind(TwoLinears, M=2, opsA={"dot": da, "act": Relu16}, opsB={"dot": db, "act": Relu16}))
    # the same body under two distinct objects binds to the shared specialization (M=4: a fresh cache entry)
    da2 = _dot_like(_dot_body_a)
    P = Program(bind(TwoLinears, M=4, opsA={"dot": da, "act": Relu16}, opsB={"dot": da2, "act": Relu16}))
    desc = encode_program(P)
    assert sum(1 for k in desc["definitions"] if k.startswith("LinearOps_v1{")) == 1


def test_unbound_composite_as_a_static_value_is_not_encodable():
    @composite("NeedsK", 1, ["K"], lambda S: ((("x", Array(S.K, V16)),), V16))
    def NeedsK(B, S, x):
        return B.call(Id16, x[0])

    @composite("MidUnbound", 1, ["f"], lambda S: ((("x", A2),), V16))
    def MidUnbound(B, S, x):
        return B.call(bind(S.f, K=2), x)

    @composite("TopUnbound", 1, ["f"], lambda S: ((("x", A2),), V16))
    def TopUnbound(B, S, x):
        return B.call(bind(MidUnbound, f=S.f), x)

    # (the program function itself is inlined into the root; the unbound slot is in `MidUnbound`'s record)
    with pytest.raises(TypeError, match="unbound static parameters"):
        encode_program(Program(bind(TopUnbound, f=NeedsK)))


def test_a_self_compare_round_trips():
    """A batch whose two arguments are the same multi-part sequence (the top-p keep word's NaN test, `x == x`): its second argument
    is written out, not as an alias to its own node, which a reader cannot resolve (PROTOCOL.md §4.5)."""
    from verity.ml.circuits.refs import Affine, Coll, concat
    from verity.ml.circuits.ops.scalar import BIT, F32, F32Eq

    F2 = Array(2, F32)

    def body(B, S, a, b):
        x = Coll(F2, concat([Affine("p", 0, 0, 1), Affine("p", 1, 1, 1)]))
        return B.batch(F32Eq, x, x, axes=(0, 0))

    sig = lambda S: ((("a", F2), ("b", F2)), Array(2, BIT))  # noqa: E731
    fn = bind(CompositeDefinition("SelfCompare", 1, (), sig, body, register=False))
    P = Program(CompositeDefinition("CallsSelfCompare", 1, (), sig, lambda B, S, a, b: B.call(fn, a, b), register=False))
    desc = json.loads(json.dumps(encode_program(P)))
    (node,) = desc["definitions"][fn.id]["body"]["nodes"]
    assert node["args"][0]["refs"] == node["args"][1]["refs"] and "alias" not in json.dumps(node)
    _same_circuit(P, decode_program(desc, REGISTRY))


# -- zero-stride broadcasts (ruling §8.5): `[z, z, z]` compacts, `[z0, z1, z2]` does not -------------------------

def test_concat_normalises_repeated_references_to_one_value_into_a_zero_stride_strided():
    from verity.ml.circuits.refs import Affine, Concat, Strided, concat, is_broadcast
    z = Affine("n", 3, 0, 1)  # one produced value
    r = concat([z] * 64)
    assert isinstance(r, Strided) and (r.outer_stride, r.inner_stride, r.inner, r.count) == (0, 0, 1, 64)
    assert all(r[j] == ("n", 3, 0) for j in range(64)) and is_broadcast(r)
    # slicing keeps compactness; so does concatenating broadcasts of the same leaf (the FA2 pad rows)
    assert isinstance(r.slice(5, 40), Strided) and r.slice(5, 40).count == 35
    rows = concat([Affine("p", 1, 0, 128), r, r, r])
    assert isinstance(rows, Concat) and len(rows.parts) == 2 and rows.parts[1].count == 192
    assert [rows[j] for j in (0, 127, 128, 319)] == [("p", 1, 0), ("p", 1, 127), ("n", 3, 0), ("n", 3, 0)]
    # three *distinct* produced values are three references: never merged
    d = concat([Affine("n", 3, 0, 1), Affine("n", 4, 0, 1), Affine("n", 5, 0, 1)])
    assert isinstance(d, Concat) and len(d.parts) == 3
    # ... and a run of the same leaf followed by a different one keeps both
    m = concat([z, z, Affine("n", 4, 0, 1)])
    assert isinstance(m, Concat) and [type(p).__name__ for p in m.parts] == ["Strided", "Affine"]
    assert list(m) == [("n", 3, 0), ("n", 3, 0), ("n", 4, 0)]
    # contiguous references are still one affine run
    assert isinstance(concat([Affine("n", 3, 0, 1), Affine("n", 3, 1, 1)]), Affine)


