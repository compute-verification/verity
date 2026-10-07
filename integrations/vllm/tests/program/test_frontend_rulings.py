"""The §8 rulings in the torch frontend (frontend design §8/§9), one test each.
Needs torch (the pod); skips cleanly without it."""

from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")

from verity.ml.circuits import Program, bind, decode_program, encode_program, program_digest  # noqa: E402
from verity.ml.circuits.defs import REGISTRY  # noqa: E402
from verity_vllm.program.frontend import (Arr, DeadGateWarning, FrontendError, Ops, batch, check_identity, export,  # noqa: E402
                                export_report, function, ops, scan)
from verity_vllm.program.frontend import examples as ex  # noqa: E402
from verity_vllm.program.frontend.correspond import correspond  # noqa: E402
from verity.ml.circuits.refs import Affine, Concat, Explicit, Strided  # noqa: E402
from verity_vllm.program.registry import b1  # noqa: E402
from verity_vllm.program.registry.prims import BF16, F32, I32  # noqa: E402

meta = lambda *shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")  # noqa: E731


def _root_body(P: Program):
    return P.root.body


def _node_refs(P: Program, k: int, arg: int):
    return _root_body(P).nodes[k].args[arg].refs


# -- ruling 6: S.out is the graph boundary; dead gates are legal, reported, warned about, strict-able -----------


def test_r6_dead_gates_reported_warned_strict_and_suppressed():
    @function(name="WithDead")
    def WithDead(x: Arr[8, BF16], w: Arr[8, BF16]) -> Arr[8, BF16]:
        unused = ex.BiasAdd(x, w)  # noqa: F841  produced, never consumed
        return ex.BiasAdd(w, x)

    xs = (meta(8), meta(8))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = export_report(WithDead, xs)
    assert [w for w in caught if issubclass(w.category, DeadGateWarning)], "authoring mode warns"
    assert rep.dead.total == 8 and rep.dead.computed == 8 and rep.dead.ids_complete
    assert rep.dead.on_s_out == 0  # dead by existence is not boundary membership
    roots = set(rep.program.circuit.root_outputs())
    assert not (set(rep.dead.ids) & roots) and len(roots) == 8
    assert rep.unreachable.total == 8  # nothing else fed only the dead gates
    assert rep.to_json()["dead_gates"]["dead_gates"] == 8 and rep.to_json()["dead_gate_policy"] == "warn"
    # the descriptor still carries the dead gates: they are part of the described computation (§8.6)
    assert rep.gates == 16 + 16 and rep.input_gates == 16
    with pytest.raises(FrontendError) as ei:
        export(WithDead, xs, dead_gates="error")
    assert ei.value.code == "E_DEAD_GATES"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export(WithDead, xs, dead_gates="ignore")
    assert not [w for w in caught if issubclass(w.category, DeadGateWarning)]
    with pytest.raises(ValueError):
        export(WithDead, xs, dead_gates="loud")


def test_r6_declared_return_that_nothing_reads_is_not_on_s_out():
    """A composite returns (t, u); the caller reads only t. `u` is a declared return, not a boundary value."""
    @function(name="Pair6")
    def Pair(x: Arr[4, BF16]):
        return (ex.BiasAdd(x, x, name="t"), ex.BiasAdd(x, x, name="u"))

    @function(name="Top6")
    def Top(x: Arr[4, BF16]) -> Arr[4, BF16]:
        t, u = Pair(x)
        return t

    rep = export_report(Top, (meta(4),), dead_gates="ignore")
    call = next(c for c in rep.program.circuit.root_calls() if c.node.fn.base_id == "Pair6_v1")
    assert len(call.distinct_returns) == 8 and len(call.out) == 4  # 8 declared, 4 on the boundary
    assert rep.dead.total == 4


# -- ruling 3: ops bundles are record statics; bodies shared; bundles coexist; E_IDENTITY only on conflict -----


def _row_family():
    @function(name="Row3", static=("ops",))
    def Row(x: Arr[2, BF16], w: Arr[2, BF16], *, ops) -> BF16:
        return ops.add(ops.mul(x[0], w[0]), ops.mul(x[1], w[1]))

    @function(name="Two3", static=("opsA", "opsB"))
    def Two(x: Arr[2, BF16], W: Arr[3, 2, BF16], *, opsA, opsB) -> Arr[2, 3, BF16]:
        a = batch(Row.bind(ops=opsA), x, W, axes=(None, 0))
        b = batch(Row.bind(ops=opsB), x, W, axes=(None, 0))
        a2 = batch(Row.bind(ops=opsA), x, W, axes=(None, 0))  # second call site of the same specialisation
        return torch.stack([a, torch.stack([b[0], a2[1], b[2]])])

    return Row, Two


def test_r3_ops_bundles_are_identity_shared_and_coexist():
    Row, Two = _row_family()
    opsA = Ops(add=ops.Bf16Add, mul=ops.Bf16MulBf16)
    opsB = Ops(add=ops.Bf16AddF2fp, mul=ops.Bf16MulBf16)
    rep = export_report(Two, static={"opsA": opsA, "opsB": opsB}, dead_gates="ignore")
    defs = encode_program(rep.program)["definitions"]
    rows = sorted(k for k in defs if k.startswith("Row3_v1{"))
    assert len(rows) == 2, rows  # Row(opsA) and Row(opsB) coexist as two fully qualified specialisations
    assert all("Bf16Add" in k and "Bf16MulBf16" in k for k in rows)  # the selected functions are in the identity
    # bodies are emitted once and shared: 3 call sites, 2 bodies
    batch_nodes = [n for n in rep.program.fn.body.nodes if n.form == "batch"]
    assert len(batch_nodes) == 3 and len({n.fn.id for n in batch_nodes}) == 2
    # and the descriptor round-trips (every role is called in the body, so nothing depends on nested-static discovery)
    P2 = decode_program(encode_program(rep.program), REGISTRY)
    assert program_digest(P2) == rep.digest


def test_r3_e_identity_only_for_conflicting_bodies():
    @function(name="Rowish3")
    def Rowish(x: Arr[8, BF16], *, act) -> Arr[8, BF16]:
        return batch(act, x, x, axes=(0, 0))

    xs = (meta(8),)
    export(Rowish, xs, static={"act": ops.Bf16Add}, dead_gates="ignore")
    with pytest.raises(FrontendError) as ei:  # same specialisation `Rowish3_v1`, a different body
        export(Rowish, xs, static={"act": ops.Bf16AddF2fp}, dead_gates="ignore")
    assert ei.value.code == "E_IDENTITY"

    @function(name="Indifferent3")
    def Indifferent(x: Arr[8, BF16], *, flavour) -> Arr[8, BF16]:
        return batch(ops.Bf16Add, x, x, axes=(0, 0))  # the body does not depend on `flavour`

    d1 = program_digest(export(Indifferent, xs, static={"flavour": "a"}, dead_gates="ignore"))
    d2 = program_digest(export(Indifferent, xs, static={"flavour": "b"}, dead_gates="ignore"))  # same body: accepted
    assert d1 == d2


def test_r3_bundle_with_an_uncalled_role_round_trips():
    """A role bound in the bundle but not called by the body is part of the identity (`Row{ops={...}}`) and
    must be a definition of the descriptor even though no node references it; `encode_program.visit` only
    follows function-valued statics at the top level today (DC owns the codec fix)."""
    @function(name="RowU3", static=("ops",))
    def Row(x: Arr[2, BF16], w: Arr[2, BF16], *, ops) -> BF16:
        return ops.add(x[0], w[1])

    bundle = Ops(add=ops.Bf16Add, spare=ops.Gemm(K=32, N=4))
    rep = export_report(Row, static={"ops": bundle}, dead_gates="ignore")
    desc = encode_program(rep.program)
    assert any(k.startswith("Gemm_v1{") for k in desc["definitions"])
    decode_program(desc, REGISTRY)


# -- ruling 1: one body per fully qualified specialisation; registry version in the provenance -----------------


def test_r1_conflicting_definitions_are_rejected_not_first_wins():
    @function(name="Gemm", version=1, static=("K", "N"))  # claims TA1's registered identity with another body
    def NotGemm(x: Arr["K", BF16], w: Arr["N", "K", BF16], *, K, N) -> Arr["N", BF16]:
        return batch(ops.Bf16Add, w[:, 0], w[:, 1], axes=(0, 0))

    with pytest.raises(FrontendError) as ei:
        export(NotGemm, static={"K": 32, "N": 4}, dead_gates="ignore")
    assert ei.value.code == "E_IDENTITY" and "registered" in ei.value.message
    # an authored function with TA1's identity *and* TA1's body is accepted (the id denotes that body)
    P = export(ex.Gemm, static={"K": 32, "N": 4}, dead_gates="ignore")
    assert program_digest(P) == program_digest(Program(bind(b1.Gemm, K=32, N=4)))


def test_r1_two_bodies_in_one_program_are_rejected():
    @function(name="Leaf1", static=("K",))
    def LeafA(x: Arr["K", BF16], *, K) -> Arr["K", BF16]:
        return batch(ops.Bf16Add, x, x, axes=(0, 0))

    @function(name="Leaf1", static=("K",))
    def LeafB(x: Arr["K", BF16], *, K) -> Arr["K", BF16]:
        return batch(ops.Bf16AddF2fp, x, x, axes=(0, 0))

    @function(name="Both1")
    def Both(x: Arr[4, BF16]) -> Arr[4, BF16]:
        return LeafB(LeafA(x, K=4), K=4)

    with pytest.raises(FrontendError) as ei:
        export(Both, (meta(4),), dead_gates="ignore")
    assert ei.value.code == "E_IDENTITY" and "two different bodies claim Leaf1_v1{K=4}" in ei.value.message
    # and the program-level check catches the same conflict on an already-built program (no first-wins on decode paths)
    from verity.ml.circuits.defs import CompositeDefinition, Builder
    from verity.ml.circuits.refs import Coll
    from verity_vllm.program.registry.prims import Bf16Add, Bf16AddF2fp
    from verity.ml.circuits.types import Array

    def leaf(prim):
        return CompositeDefinition("Leaf1x", 1, ("K",), lambda S: ((("x", Array(S.K, BF16)),), Array(S.K, BF16)),
                                   lambda B, S, x: B.batch(prim, x, x, axes=(0, 0)), register=False)

    la, lb = leaf(Bf16Add), leaf(Bf16AddF2fp)
    both = CompositeDefinition("Both1x", 1, (), lambda S: ((("x", Array(4, BF16)),), Array(4, BF16)),
                               lambda B, S, x: B.call(bind(lb, K=4), B.call(bind(la, K=4), x)), register=False)
    with pytest.raises(FrontendError) as ei:
        check_identity(Program(bind(both)))
    assert ei.value.code == "E_IDENTITY" and "within one program" in ei.value.message


def test_r1_registry_version_in_provenance():
    rep = export_report(ex.Gemm, static={"K": 32, "N": 4}, dead_gates="ignore")
    reg = rep.provenance["registry"]
    assert len(reg["digest"]) == 64 and reg["primitives"] > 20 and "verity_vllm.program.registry.prims" in reg["sources"]
    assert rep.provenance["torch"] == torch.__version__ and rep.provenance["mode"] == "authoring"
    assert rep.provenance["rules"] == []


# -- ruling 2: no silent padding ------------------------------------------------------------------------------


def test_r2_no_silent_padding_of_scan_or_attention():
    SumStep = _sum_step()

    @function(name="Sum2", static=("N",))
    def Sum(xs: Arr["N", BF16], *, N) -> BF16:
        carry, ys = scan(SumStep, xs[0], xs=(xs[1:],))
        return carry

    for n in (3, 5, 9):
        P = export(Sum, static={"N": n}, dead_gates="ignore")
        scans = [nd for nd in P.fn.body.nodes if nd.form == "scan"]
        assert len(scans) == 1 and scans[0].n == n - 1, n  # exactly the authored length, never a padded maximum
    # attention: per-T specialisations with the authored block structure (T=5: one partial block, T=70: 6 + 64)
    from tests.program import b1_authored as A
    ids5 = set(encode_program(Program(A.AttentionHead.specialize(T=5, D=16))).get("definitions", {}))
    ids70 = set(encode_program(Program(A.AttentionHead.specialize(T=70, D=16))).get("definitions", {}))
    assert {k for k in ids5 if k.startswith("AttnBlock")} == {"AttnBlock_v1{D=16,NVIS=5,FIRST=true}"}
    assert {k for k in ids70 if k.startswith("AttnBlock")} == {"AttnBlock_v1{D=16,NVIS=6,FIRST=true}", "AttnBlock_v1{D=16,NVIS=64,FIRST=false}"}
    assert Program(A.AttentionHead.specialize(T=5, D=16)).gates != Program(A.AttentionHead.specialize(T=70, D=16)).gates


def _sum_step():
    @function(name="SumStep2")
    def SumStep(carry: BF16, x: BF16):
        s = ops.Bf16Add(carry, x)
        return (s, s)
    return SumStep


# -- ruling 4: derivation provenance --------------------------------------------------------------------------


def test_r4_derivation_provenance_record_for_the_module_route():
    class Lin(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(4, 32, dtype=torch.bfloat16))
            self.b = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))

        def forward(self, x):
            return ex.BiasAdd(ex.Gemm(x, self.w), self.b)

    rep = export_report(Lin(), (meta(32),), name="Lin4", dead_gates="ignore")
    prov = rep.provenance
    assert prov["mode"] == "derivation" and prov["torch"] == torch.__version__
    (rule,) = prov["rules"]
    assert rule["rule"] == "ModuleWeights_v1" and rule["kind"] == "hierarchy-around-unchanged-gates"
    assert rule["source"]["parameters"] == ["b", "w"] and rule["source"]["module"].endswith("Lin")
    assert rule["preconditions_checked"] and all(rule["preconditions_checked"].values())
    assert rule["result"]["digest"] == rep.digest and rule["result"]["root"] == rep.root
    assert set(rule["preconditions"]) == set(rule["preconditions_checked"])


# -- ruling 5: references -------------------------------------------------------------------------------------


def test_r5_broadcasts_are_zero_stride_strided_and_distinct_values_are_not_merged():
    @function(name="Bcast5")
    def Bcast(x: Arr[3, BF16]) -> Arr[3, BF16]:
        z = ops.const(16, 0)
        rep = z.expand(3)  # one gate broadcast
        st = torch.stack([z, z, z])  # the same reference three times
        three = torch.stack([ops.const(16, 0) for _ in range(3)])  # three produced values
        a = ex.BiasAdd(x, rep)
        b = ex.BiasAdd(a, st)
        return ex.BiasAdd(b, three)

    P = export(Bcast, (meta(3),), dead_gates="ignore")
    nodes = P.fn.body.nodes
    adds = [n for n in nodes if n.form == "call" and n.fn.base_id == "BiasAdd_v1"]
    r_rep, r_st, r_three = adds[0].args[1].refs, adds[1].args[1].refs, adds[2].args[1].refs
    assert isinstance(r_rep, Strided) and r_rep.outer_stride == 0 and r_rep.inner_stride == 0 and r_rep.count == 3
    assert r_st.to_json() == r_rep.to_json()  # stack of one repeated reference == the broadcast
    assert not isinstance(r_three, Strided) and len({r_three[j] for j in range(3)}) == 3  # never merged
    assert [r_rep[j] for j in range(3)] == [r_rep[0]] * 3


def test_r5_inner_axis_select_and_permute_are_strided_views():
    @function(name="Views5")
    def Views(kc: Arr[7, 3, 8, BF16], v: Arr[8, 8, BF16]) -> Arr[8, BF16]:
        head = kc.select(1, 1)  # [7][8]: one KV head out of [T][KVH][D]
        row = ex.BiasAdd(head[0], head[6])
        col = v.select(1, 0)  # [8]: column 0 of v
        return ex.BiasAdd(row, col)

    P = export(Views, (meta(7, 3, 8), meta(8, 8)), dead_gates="ignore")
    adds = [n for n in P.fn.body.nodes if n.form == "call"]
    h0, h6 = adds[0].args[0].refs, adds[0].args[1].refs
    assert not isinstance(h0, (Explicit, Concat)) and [h0[j][2] for j in range(8)] == [8 + j for j in range(8)]
    assert not isinstance(h6, (Explicit, Concat)) and [h6[j][2] for j in range(8)] == [6 * 24 + 8 + j for j in range(8)]
    col = adds[1].args[1].refs  # column 0 of v as a view of the parameter: one Strided
    assert isinstance(col, Strided) and [col[j][2] for j in range(8)] == [j * 8 for j in range(8)]


def test_r5_permute_fallback_is_explicit_under_a_limit():
    from verity_vllm.program.frontend import torch_frontend as tf

    @function(name="Perm5")
    def Perm(x: Arr[2, 3, 4, BF16]) -> Arr[4, 3, 2, BF16]:
        y = torch.cat([x[1:2], x[0:1]])  # a composite reference sequence: no single affine run underneath
        return y.permute(2, 1, 0)

    P = export(Perm, (meta(2, 3, 4),), dead_gates="ignore")
    ret = P.fn.body.ret
    assert isinstance(ret, Explicit) and ret.count == 24
    old = tf._EXPLICIT_LIMIT
    tf._EXPLICIT_LIMIT = 8
    try:
        @function(name="Perm5b")
        def Perm2(x: Arr[2, 3, 4, BF16]) -> Arr[4, 3, 2, BF16]:
            return torch.cat([x[1:2], x[0:1]]).permute(2, 1, 0)

        with pytest.raises(FrontendError) as ei:
            export(Perm2, (meta(2, 3, 4),), dead_gates="ignore")
        assert ei.value.code == "E_UNSUPPORTED_VIEW"
    finally:
        tf._EXPLICIT_LIMIT = old


def test_r5_three_level_view_is_a_concat_of_strided_pieces():
    @function(name="Perm5c")
    def Perm(x: Arr[2, 3, 4, BF16]) -> Arr[4, 3, 2, BF16]:
        return x.permute(2, 1, 0)

    P = export(Perm, (meta(2, 3, 4),), dead_gates="ignore")
    ret = P.fn.body.ret
    assert isinstance(ret, Concat) and len(ret.parts) == 4 and all(isinstance(p, Strided) for p in ret.parts)
    expect = [k * 1 * 0 + j * 4 + i * 12 for k in range(4) for j in range(3) for i in range(2)]
    assert [ret[n][2] for n in range(24)] == [k + 4 * j + 12 * i for k in range(4) for j in range(3) for i in range(2)]
    assert expect  # (documenting the index map: out[k][j][i] = x[i][j][k])


def test_r5_authored_attention_head_corresponds_to_ta1_with_smaller_encoding():
    from tests.program import b1_authored as A
    P_a = Program(A.AttentionHead.specialize(T=70, D=16))
    P_t = Program(bind(b1.AttentionHead, T=70, D=16))
    r = correspond(P_a, P_t)
    # Since DC's builder fix B.1 (TA1's concat() emits the same zero-stride Strided as the frontend's view builder) the
    # authored and TA1 descriptors are byte-identical -- the stronger fact (descriptor-codec.md §6); before it the
    # authored one was smaller and corresponded through one same-references run.
    assert r.corresponds and r.digest_equal and not r.mismatches
    from verity.ml.circuits.codec import descriptor_size
    assert descriptor_size(P_a) == descriptor_size(P_t)
