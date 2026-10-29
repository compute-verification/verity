"""Descriptor codec: nested-static dependency discovery (frontend rulings §8.3), identity conflicts (§8.1),
and digest stability of the v0 encoding for programs whose encoding did not change."""

from __future__ import annotations

import json

import pytest

from verity.ml.circuits import Array, Program, Value, bind, composite, decode_program, encode_program, program_digest, REGISTRY
from verity.ml.circuits.codec import IdentityConflict
from verity.ml.circuits.defs import CompositeDefinition
from verity.ml.definitions.registry import b1

from tests.program.test_running_example import Add16, Dot, Id16, Mul16, Relu16, make_program

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


def test_uncalled_role_of_the_program_function_itself_is_a_definition():
    """The program function is inlined into the root, so no node names it; its binding record (`program_fn`)
    is still the program's identity and every function it references is resolved — FE2's authored `Row{ops=
    {add, spare}}` where `spare` is bound but never called (test_frontend_rulings: "DC: nested-static discovery")."""
    @composite("RowUncalled", 1, ["ops"], lambda S: ((("x", A2), ("w", A2)), V16))
    def RowUncalled(B, S, x, w):
        return B.call(S.ops["dot"], x, w)

    P = Program(bind(RowUncalled, ops={"dot": Dot, "spare": bind(b1.Gemm, K=32, N=4)}))
    desc = encode_program(P)
    assert "Gemm_v1{K=32,N=4}" in desc["definitions"] and "Gemm_v1{K=32,N=4}" in desc["program_fn"]
    assert not any("Gemm" in _spec(n) for n in desc["definitions"][desc["root"]]["body"]["nodes"] for _spec in [lambda n: str(n["fn"])])
    D = decode_program(json.loads(json.dumps(desc)), REGISTRY)
    _same_circuit(P, D)
    assert program_digest(encode_program(D)) == program_digest(desc)


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


def test_duplicate_definition_objects_are_identity_checked_once_per_object(monkeypatch):
    """A body that calls the SAME re-decoded definition object from many nodes (a workload Program composed
    from separately decoded request components: every node of the second component refers to a duplicate object) is identity-
    checked once per distinct duplicate object, not once per calling node -- the OLMoE B2 GP-01 compose re-encoded 95 definitions
    146,523 times (> 45 min; 8 s deduplicated); the descriptor and digest are unchanged, and a conflicting duplicate is still refused."""
    from verity.ml.circuits import codec as C
    from verity.ml.circuits.refs import array_of
    N = 40
    def wrapped(dot):                                                          # a root that CALLS the dot (the program fn itself is inlined)
        return CompositeDefinition("WrapDot", 1, (), lambda S: ((("x", A2), ("w", A2)), V16), lambda B, S, x, w: B.call(dot, x, w), register=False)
    desc_a = encode_program(Program(bind(wrapped(_dot_like(_dot_body_a)))))

    def decoded_dot(desc):                                                      # the DotConflict_v1 definition object of one decode
        root = decode_program(desc, REGISTRY).root
        return next(n.fn for n in root.body.nodes if C._spec_id(n.fn) == "DotConflict_v1")
    r1, r2 = decoded_dot(desc_a), decoded_dot(desc_a)                            # two decodes: two objects, one id
    rid = "DotConflict_v1"
    assert r1 is not r2 and C._spec_id(r1) == C._spec_id(r2) == rid

    def body(B, S, x, w):
        return array_of([B.call(r1, x, w)] + [B.call(r2, x, w) for _ in range(N)])
    top = CompositeDefinition("TopManyDup", 1, (), lambda S: ((("x", A2), ("w", A2)), Array(N + 1, V16)), body, register=False)
    P = Program(bind(top))
    v0_checks: list[str] = []
    orig = C._encode_definition

    def counting(fn, *a, **k):
        if not a and not k and C._spec_id(fn) == rid:                        # the identity check's self-contained (v0) encodes
            v0_checks.append(C._spec_id(fn))
        return orig(fn, *a, **k)
    monkeypatch.setattr(C, "_encode_definition", counting)
    desc = encode_program(P)
    assert len(v0_checks) == 2, v0_checks                                    # one duplicate object -> one comparison (2 encodes), not N
    assert desc["definitions"][rid]["form"] == "composite"
    root_nodes = decode_program(desc, REGISTRY).root.body.nodes
    assert sum(1 for n in root_nodes if n.form == "call" and C._spec_id(n.fn) == rid) == N + 1      # every call survived the dedup
    monkeypatch.setattr(C, "_encode_definition", orig)
    assert program_digest(desc) == program_digest(encode_program(P))          # deterministic; the dedup does not enter the descriptor
    # the same shape with a CONFLICTING re-decoded duplicate (same id, other body) is still refused
    r3 = decoded_dot(encode_program(Program(bind(wrapped(_dot_like(_dot_body_b))))))

    def body_conflict(B, S, x, w):
        return array_of([B.call(r1, x, w)] + [B.call(r3, x, w) for _ in range(N)])
    topc = CompositeDefinition("TopManyDupConflict", 1, (), lambda S: ((("x", A2), ("w", A2)), Array(N + 1, V16)), body_conflict, register=False)
    assert r3 is not r1 and C._spec_id(r3) == rid
    with pytest.raises(IdentityConflict, match="DotConflict"):
        encode_program(Program(bind(topc)))


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


def test_attention_head_zero_padding_references_one_zero_gate():
    """The FA2 `Clear_OOB_MN` pad of `AttentionHead{T}` is D * (BN - nvis) references to the body's single
    `Const16[0]` gate (and l0/o0 to its single `Const32[0]`): one value each, broadcast, not n produced zeros."""
    from verity.ml.circuits.refs import Strided
    sp = bind(b1.AttentionHead, T=5, D=64)
    body = sp.body
    consts = [(k, n.fn.id) for k, n in enumerate(body.nodes) if getattr(n.fn, "family", None) == "Const"]
    assert consts == [(0, "Const16[0x0000]_v1"), (1, "Const32[0x00000000]_v1")]
    blk = body.nodes[2]
    kb, vb, l0, o0 = blk.args[1].refs, blk.args[2].refs, blk.args[4].refs, blk.args[5].refs
    for pad in (kb.parts[1], vb.parts[1]):
        assert isinstance(pad, Strided) and pad.count == 64 * (b1.BN - 5) and pad[0] == pad[pad.count - 1] == ("n", 0, 0)
    assert isinstance(l0, Strided) and l0.count == 4 and l0[3] == ("n", 1, 0)
    assert isinstance(o0, Strided) and o0.count == 64 and o0[63] == ("n", 1, 0)
    # the circuit is unchanged: the pad rows resolve to the one zero gate
    P = Program(sp)
    C = P.circuit
    call = next(c for c in C.root_calls() if c.name == "AttnBlock")
    args = call.args
    q = 64
    k_rows = args[q:q + b1.BN * 64]
    assert len(set(k_rows[5 * 64:])) == 1 and C.gate(k_rows[5 * 64]).prim.id == "Const16[0x0000]_v1"


# -- schema v1: flat concats and the reference-sequence alias (descriptor-codec.md §B.2) -----------------------------

def test_v1_alias_writes_a_repeated_reference_sequence_as_a_pass_through_to_the_earlier_argument():
    from types import SimpleNamespace
    from verity.ml.circuits.codec import SCHEMA_V0, SCHEMA_V1, _BodyEncoder, _Tables, _decode_refs
    from verity.ml.circuits.refs import Affine, Concat
    a0, a1, a2, a3 = (Affine("n", i, 0, 4) for i in range(4))
    A = Concat([a0, a1])
    Bn = Concat([A, a2, a3])  # nested, as the builder produces it for the growing KV cache
    enc = _BodyEncoder(SCHEMA_V1, _Tables())
    jA = enc.refs(A, (5, 0))
    assert jA == {"concat": [a0.to_json(), a1.to_json()]}
    jB = enc.refs(Bn, (6, 1))
    assert jB == {"concat": [{"alias": [5, 0]}, a2.to_json(), a3.to_json()]}  # flat: the nesting is not written
    assert enc.refs(Concat([a0, a1]), (7, 0)) == {"alias": [5, 0]}  # the whole sequence again: just the alias
    assert enc.refs(Concat([a0, a1, a2, a3]), None) == {"alias": [6, 1]}  # `ret` may alias; it is not a target
    assert enc.refs(Concat([a1, a2]), (8, 0)) == {"concat": [a1.to_json(), a2.to_json()]}  # not a prefix: verbatim
    assert _BodyEncoder(SCHEMA_V0).refs(Bn, (6, 1)) == Bn.to_json()  # v0 is untouched
    # decoding splices the aliased sequence back in, flat, with the same ordered references
    nodes = [SimpleNamespace(args=[SimpleNamespace(refs=None)])] * 5
    nodes.append(SimpleNamespace(args=[SimpleNamespace(refs=_decode_refs(jA, nodes, SCHEMA_V1))]))
    dB = _decode_refs(jB, nodes, SCHEMA_V1)
    assert isinstance(dB, Concat) and len(dB.parts) == 4 and list(dB) == list(Bn)
    with pytest.raises(ValueError):
        _decode_refs({"alias": [9, 0]}, nodes, SCHEMA_V1)  # aliases point backwards only


@pytest.mark.slow
def test_v1_kv_cache_arguments_alias_the_previous_position_and_denote_the_same_program():
    from verity_vllm.program import profile_descriptor as PD
    from verity.ml.circuits.codec import SCHEMA_V0, SCHEMA_V1, encode_program as enc
    from verity_vllm.program.descriptor_equivalence import canonical_runs
    LP, STEPS = 4, 2
    desc = PD.build("B0", LP, STEPS)
    assert desc["schema"] == SCHEMA_V1
    root = desc["definitions"][desc["root"]]["body"]
    fn_of = desc["callees"].__getitem__  # v1: callee ids are interned
    attn = [(k, n) for k, n in enumerate(root["nodes"]) if fn_of(n["fn"]).startswith("Attention_v1{")]
    for k, n in attn:
        T = int(fn_of(n["fn"]).split("T=")[1].split(",")[0])
        kc, vc = n["args"][1]["refs"], n["args"][2]["refs"]
        if T >= 3:  # position T's cache is position T-1's cache (an earlier Attention node's argument) plus one row
            assert kc["concat"][0]["alias"][1] == 1 and vc["concat"][0]["alias"][1] == 2
            j = kc["concat"][0]["alias"][0]
            assert j < k and fn_of(root["nodes"][j]["fn"]).startswith(f"Attention_v1{{T={T - 1},")
            assert len(kc["concat"]) == 2 and "affine" in kc["concat"][1]
        elif T == 2:
            assert all("affine" in p for p in kc["concat"]) and len(kc["concat"]) == 2
        else:
            assert "affine" in kc
    assert len(attn) == b1.B0_CONFIG["NL"] * (LP + STEPS)
    # the v0 (verbatim) and v1 (aliased) descriptors decode to the same reference sequences everywhere ...
    D1 = decode_program(json.loads(json.dumps(desc)), REGISTRY)
    d0 = enc(D1, schema=SCHEMA_V0)
    D0 = decode_program(json.loads(json.dumps(d0)), REGISTRY)
    assert d0["definitions"].keys() == desc["definitions"].keys()
    for fid in desc["definitions"]:
        f0, f1 = D0.defs[fid], D1.defs[fid]
        if not hasattr(f0, "body"):
            continue
        for n0, n1 in zip(f0.body.nodes, f1.body.nodes, strict=True):
            assert n0.fn.id == n1.fn.id and n0.out == n1.out
            for x0, x1 in zip(n0.args, n1.args, strict=True):
                assert x0.type == x1.type and canonical_runs(x0.refs) == canonical_runs(x1.refs)
        assert canonical_runs(f0.body.ret) == canonical_runs(f1.body.ret)
    # ... and the v1 digest is a function of the denoted program, not of the writer's nesting: the v0 descriptor
    # re-encoded as v1 has the v1 digest, and v1 round-trips byte-for-byte
    assert program_digest(enc(D0, schema=SCHEMA_V1)) == program_digest(desc) == program_digest(enc(D1, schema=SCHEMA_V1))
    assert program_digest(d0) != program_digest(desc) and len(json.dumps(d0)) > len(json.dumps(desc))


# -- digest stability: the v0 encoding of programs whose reference construction did not change ----------------------

V0_DIGESTS = {  # recorded with the codec at vllm-poc e47be70
    "running-example": "d4ab767ce4d2a53369a712b45ab1d534d338a559067eb7f2e737c8b11126ced6877d3402dd2c45c44f511d353d27169440152bcb85d5079c73e76626eb5b5d8d",
    "running-example-Id16": "7ab1b121aefdc5b77e44ea6eefaa06c71192b316e4cacd7bbe3cbe67ef604a25b0aead3ac174b6adf90e118877a06ae6d4674619b0ef0cac05fbded0045ce7ec",
    "Gemm{K=32,N=4}": "73b31a201e80a139da6a2c8ff9fdf8de66e1d4c79a5994be754574d3677d40c1d0838cb571b6bd43e8448e1c5b4bc8bb4cd259e5e1a3e7fcb9c3921645f2cc11",  # the re-key (AmpereBF16TcDot16_v2); 708fd264 = docs/data/frontend acceptance
}
# Programs with broadcasts or KV caches (B0/B1 `Serve`, `AttentionHead`) moved: descriptor-codec.md records old -> new.


def _v0_programs():
    return {
        "running-example": make_program(),
        "running-example-Id16": make_program(act=Id16),
        "Gemm{K=32,N=4}": Program(bind(b1.Gemm, K=32, N=4)),
    }


@pytest.mark.parametrize("label", sorted(V0_DIGESTS))
def test_v0_digests_of_programs_without_nested_statics_did_not_move(label):
    from verity.ml.circuits.codec import SCHEMA_V0
    P = _v0_programs()[label]
    assert program_digest(encode_program(P, schema=SCHEMA_V0)) == V0_DIGESTS[label]
