"""The binder's passes (`verity_vllm.query.manifest.bind`) on hand-written manifest rows: what each pass binds, and what it refuses to guess."""
from __future__ import annotations

import pytest

from verity_vllm.query.manifest.bind import Binder, Port, coverage_check, source_ports

LIN, ATT = "model.layers.0.mlp.down_proj", "model.layers.0.self_attn"
QUANT = {"x_q": Port(frozenset({"DeepGemmFp8Linear_v1"}), "0"), "x_s": Port(frozenset({"DeepGemmFp8Linear_v1"}), "1")}


def row(op: str, member: str, *, step: int = 0, inv: int = 0, spec: str = "Gemm_v1{K=4}", port: str | None = None, cid: int = 0, **kw) -> dict:
    return {"family": "instance_outputs", "request_id": "r0", "step": step, "invocation": inv, "rank": 0, "op_path": op,
            "output_member": member, "spec": spec, "port": port or member, "position": [cid, 0 if port in (None, "out") else int(port)], **kw}


def deepgemm(op: str = LIN, inv: int = 0, cid: int = 0) -> list[dict]:
    return [row(op, p, inv=inv, spec="DeepGemmFp8Linear_v1{K=4}", port=p, cid=cid) for p in "012"]


def test_slots_bind_the_members_left_in_program_order_or_none():
    norm = [row("model.norm", p, spec="RMSNormFused_v1{N=4}", port=p) for p in "01"]
    b = Binder(norm + [row(LIN, "out")])
    assert b.bind(0, ["model.norm/1", "model.norm/0"]) == [("model.norm", "1", 0, "slot"), ("model.norm", "0", 0, "slot")]
    assert b.bind(0, [LIN]) == [(LIN, "out", 0, "slot")], "a bare module is slot 0"
    assert b.bind(0, [f"{LIN}/0"]) == [(LIN, "out", 0, "slot")], "one member left: slot 0 binds it"
    assert b.bind(0, [f"{LIN}/1"]) == [(LIN, "", 0, "unbound")], "slot 1 alone is not slot 0"
    assert b.bind(0, ["model.norm/0"]) == [("model.norm", "", 0, "unbound")], "one slot for two members: neither"
    assert b.bind(0, [LIN, f"{LIN}/0"]) == [(LIN, "out", 0, "slot"), (LIN, "", 0, "unbound")], "slot 0 twice at one occurrence"


def test_a_declared_port_binds_its_definitions_port_and_slot_0_the_one_left():
    b = Binder(deepgemm(), ports=QUANT)
    assert b.bind(0, [f"{LIN}/x_q", f"{LIN}/x_s", f"{LIN}/0"]) == [(LIN, "0", 0, "port"), (LIN, "1", 0, "port"), (LIN, "2", 0, "slot")]
    assert b.bind(0, [f"{LIN}/0"]) == [(LIN, "", 0, "unbound")], "y alone never binds the quant branch's port 0"
    assert Binder(deepgemm()).bind(0, [f"{LIN}/x_q"]) == [(LIN, "x_q", 0, "unbound")], "undeclared, the label is its own extra"


def test_a_row_parallel_fp8_linear_binds_its_quant_branch_partial_and_reduced_output_at_either_kernel():
    """DeepSeek-V4-Pro's `wo_b` on TP rank 3 as its manifest of record spells it: at the prefill DeepGEMM's ports 0 / 1 and its y, the
    rank's partial (port 2), then the multimem all-reduce; at decode FlashInfer's y (the partial), then the MNNVL all-reduce."""
    wo_b = "model.layers.0.attn.wo_b"
    prefill = [row(wo_b, m, spec="DeepGemmFp8Linear_v1", port=p, cid=7560, rank=3) for m, p in (("0", "0"), ("1", "1"), ("part_rank3", "2"))]
    prefill.append(row(wo_b, "out", spec="AllReduceMultimem_v1", port="out", cid=7816, rank=3))
    decode = [row(wo_b, "part_rank3", step=1, spec="FlashInferFp8Linear_v1", port="out", cid=971374, rank=3),
              row(wo_b, "out", step=1, spec="AllReduceMnnvl_v1{WORLD=8,N=7168,SHOT=1}", port="out", cid=971375, rank=3)]
    b = Binder(prefill + decode, rank=3, ports=QUANT)
    events = [f"{wo_b}/x_q", f"{wo_b}/x_s", f"{wo_b}/part_rank3", wo_b]
    assert b.bind(0, events) == [(wo_b, "0", 0, "port"), (wo_b, "1", 0, "port"), (wo_b, "part_rank3", 0, "label"), (wo_b, "out", 0, "slot")]
    assert b.bind(1, events[2:]) == [(wo_b, "part_rank3", 0, "label"), (wo_b, "out", 0, "slot")]
    assert b.bind(1, events)[:2] == [(wo_b, "x_q", 0, "unbound"), (wo_b, "x_s", 0, "unbound")], "FlashInfer's GEMM has no DeepGEMM port"
    assert b.bind(0, events[2:]) == [(wo_b, "part_rank3", 0, "label"), (wo_b, "", 0, "unbound")], "no quant branch: the return binds nothing"


def test_a_declared_labels_occurrences_bind_its_rows_in_invocation_order_when_the_counts_agree():
    b = Binder(deepgemm(inv=0, cid=0) + deepgemm(inv=1, cid=5), ports=QUANT)
    names = [f"{LIN}/x_q", f"{LIN}/0", f"{LIN}/x_q", f"{LIN}/0"]
    assert b.bind(0, names) == [(LIN, "0", 0, "port"), (LIN, "", 0, "unbound"), (LIN, "0", 1, "port"), (LIN, "", 1, "unbound")]
    assert b.bind(0, [f"{LIN}/x_q"]) == [(LIN, "x_q", 0, "unbound")], "one occurrence for two rows: none"


def test_a_call_boundary_port_binds_the_one_row_of_its_module_body_its_definitions_produce():
    sparse = Port(frozenset({"SparseMla_v1", "SparseMlaSplit_v1"}), "out", call_boundary=True)
    rows = [row(f"{ATT}/dsa_indexer_attn_3", "out", spec="SparseMlaSplit_v1{T=1}", family="call_boundaries"),
            row(f"{ATT}/bmm_2", "out", spec="BatchedGemm_v1{B=8}", family="call_boundaries"), row(ATT, "out", spec="BatchedGemm_v1{B=8}")]
    b = Binder(rows, ports={"attn_mqa": sparse})
    assert b.bind(0, [f"{ATT}/attn_mqa", f"{ATT}/out"]) == [(f"{ATT}/dsa_indexer_attn_3", "out", 0, "port"), (ATT, "out", 0, "label")]
    two = Binder(rows + [row(f"{ATT}/dsa_indexer_attn_9", "out", spec="SparseMla_v1{T=1}", family="call_boundaries")], ports={"attn_mqa": sparse})
    assert two.bind(0, [f"{ATT}/attn_mqa"]) == [(ATT, "attn_mqa", 0, "unbound")], "two sparse MLA rows for one tensor: neither"


def test_a_call_boundary_each_request_spells_by_its_own_fx_node_binds_every_requests_row_under_its_own_spelling():
    """[GLM-5.3 B8 mixed arrivals] A request's Program numbers its fx nodes over its own steps: at engine step 9, r0 at its step 9
    spells the W_UK_T absorb `bmm_1404` and r1, r2 at their step 0 `bmm`.  The step's one tensor binds every request's row, `spellings`
    names each request's own, and requests with different counts bind none."""
    absorb = {"q_nope_absorb": Port(frozenset({"BatchedGemm_v1"}), "out", call_boundary=True)}

    def cb(fx: str, rid: str, cid: int) -> dict:
        n = 1 if rid == "r0" else 11
        return row(f"{ATT}/{fx}", "out", spec="BatchedGemm_v1{B=8}", family="call_boundaries", request_id=rid, cid=cid, engine_step=9,
                   element_range=[0, 4096 * n])

    rows = [cb("bmm_1404", "r0", 9), cb("bmm", "r1", 1), cb("bmm", "r2", 1)]
    b = Binder(rows, ports=absorb)
    r0, r1 = (f"{ATT}/bmm_1404", "out", 0), (f"{ATT}/bmm", "out", 0)
    assert b.bind(9, [f"{ATT}/q_nope_absorb"]) == [(*r0, "port")]
    assert b.spellings(9, r0) == b.spellings(9, r1) == {"r0": r0, "r1": r1, "r2": r1}
    assert b.spellings(9, (ATT, "out", 0)) == {}, "no call-boundary port declares it"
    cov = coverage_check({"manifest_digest": "d", "identities": rows}, {9: [{"name": f"{ATT}/q_nope_absorb", "numel": 4096 * 23}]}, ports=absorb)
    assert (cov["ok"], cov["missing_n"], cov["committed_unmatched_n"]) == (True, 0, 0)

    two = Binder(rows + [cb("bmm_1405", "r0", 10), cb("bmm_1", "r1", 2), cb("bmm_1", "r2", 2)], ports=absorb)
    assert two.bind(9, [f"{ATT}/q_nope_absorb"] * 2) == [(*r0, "port"), (f"{ATT}/bmm_1405", "out", 0, "port")], "occurrence k: each request's k-th"
    assert two.spellings(9, (f"{ATT}/bmm_1", "out", 0)) == {"r0": (f"{ATT}/bmm_1405", "out", 0), "r1": (f"{ATT}/bmm_1", "out", 0),
                                                            "r2": (f"{ATT}/bmm_1", "out", 0)}
    odd = Binder(rows + [cb("bmm_1405", "r0", 10)], ports=absorb)
    assert odd.bind(9, [f"{ATT}/q_nope_absorb"]) == [(ATT, "q_nope_absorb", 0, "unbound")], "r0 has two, r1 and r2 one: none"


def test_each_request_spells_a_call_boundary_in_its_own_program_order_past_a_key_both_requests_spell():
    """[GLM-5.3 B8 greedy Commit] At engine step 9 the layer's own output (`self_attn` `out`, a BatchedGemm_v1
    Call as the absorb is) is one key that r0 at its step 9 and r1 at its step 0 both spell, at a different place in each one's
    Program.  Ordered by one request's position for it, r0's absorb `bmm_1404` was paired with r1's `self_attn` output: r1's `bmm` was
    left unbound and its output bound twice, at 494 identities per rank."""
    absorb = {"q_nope_absorb": Port(frozenset({"BatchedGemm_v1"}), "out", call_boundary=True)}

    def at9(op: str, rid: str, cid: int, family: str = "call_boundaries") -> dict:
        return row(op, "out", spec="BatchedGemm_v1{B=8}", family=family, request_id=rid, cid=cid, engine_step=9, element_range=[0, 1])

    r1 = [at9(f"{ATT}/bmm", "r1", 100), at9(ATT, "r1", 265, "instance_outputs")]
    r0 = [at9(f"{ATT}/bmm_1404", "r0", 88206), at9(ATT, "r0", 88221, "instance_outputs")]
    absorb_r0, absorb_r1, out = (f"{ATT}/bmm_1404", "out", 0), (f"{ATT}/bmm", "out", 0), (ATT, "out", 0)
    for rows in (r1 + r0, r0 + r1):
        b = Binder(rows, ports=absorb)
        assert b.bind(9, [f"{ATT}/q_nope_absorb", f"{ATT}/out"]) == [(*absorb_r0, "port"), (*out, "label")]
        assert b.spellings(9, absorb_r0) == {"r0": absorb_r0, "r1": absorb_r1}, "each request's absorb, whichever request's rows come first"
        cov = coverage_check({"manifest_digest": "d", "identities": rows},
                             {9: [{"name": f"{ATT}/q_nope_absorb", "numel": 1}, {"name": f"{ATT}/out", "numel": 1}]}, ports=absorb)
        assert (cov["ok"], cov["missing_n"], cov["duplicate_names_n"]) == (True, 0, 0)


def test_labels_declaring_one_port_bind_its_rows_in_commit_order_and_a_lone_call_its_unqualified_module():
    """DeepSeek-V4-Pro's mHC as its manifest of record spells it: layer 1's attention and FFN post_pre are two fused post_pre Calls
    (`<layer>/<fx node>`, contested), layer 0's FFN post_pre the layer's one (`<layer>`); the taps commit residual (port 0) per side."""
    pp = frozenset({"MhcFusedPostPre_v1", "MhcPostPreTf32_v1"})
    residual = Port(pp, "0", call_boundary=True)
    l0, l1 = "model.layers.0", "model.layers.1"
    rows = [row(l0, "pre.0", spec="MhcPreBroadcast_v1", port="0", cid=1), row(l0, "post_pre.0", spec="MhcPostPreTf32_v1", port="0", cid=5),
            row(f"{l1}/mhc_fused_post_pre_tilelang_2", "post_pre.0", spec="MhcFusedPostPre_v1", port="0", cid=9),
            row(f"{l1}/mhc_fused_post_pre_tilelang_7", "post_pre.0", spec="MhcFusedPostPre_v1", port="0", cid=14)]
    b = Binder(rows, ports={"hc_attn_residual": residual, "hc_ffn_residual": residual})
    assert b.bind(0, [f"{l0}/pre.0", f"{l0}/hc_ffn_residual", f"{l1}/hc_attn_residual", f"{l1}/hc_ffn_residual"]) == [
        (l0, "pre.0", 0, "label"), (l0, "post_pre.0", 0, "port"),
        (f"{l1}/mhc_fused_post_pre_tilelang_2", "post_pre.0", 0, "port"), (f"{l1}/mhc_fused_post_pre_tilelang_7", "post_pre.0", 0, "port")]
    assert b.bind(0, [f"{l1}/hc_ffn_residual"]) == [(l1, "hc_ffn_residual", 0, "unbound")], "one side for two Calls: neither"


def test_a_label_binds_the_member_it_equals_at_its_occurrence():
    b = Binder([row("model.embed_tokens", "part_rank3"), row("model.embed_tokens", "in", family="input_token_ids"),
                row("runner.sampler", "sampled_token_ids", step=1), row("moe/L3", "topk_ids", family="moe_stream")])
    assert b.bind(0, ["model.embed_tokens/in", "model.embed_tokens/part_rank3", "model.embed_tokens/part_rank3", "moe/L3/topk_ids"]) == [
        ("model.embed_tokens", "in", 0, "label"), ("model.embed_tokens", "part_rank3", 0, "label"),
        ("model.embed_tokens", "part_rank3", 1, "unbound"), ("moe/L3", "topk_ids", 0, "label")]
    assert b.bind(1, ["runner.sampler/sampled_token_ids", "model.embed_tokens/in"]) == [
        ("runner.sampler", "sampled_token_ids", 0, "label"), ("model.embed_tokens", "in", 0, "unbound")], "rows bind at their own step"


def test_a_promoted_identity_binds_only_at_its_consumers_input():
    rows = [row("model", "out", spec="Bf16MulScalarTensor_v1{N=4}", promoted=True)]
    b = Binder(rows, rebind={"model.norm/in": ("model", "out")})
    assert b.bind(0, ["model.norm/in"]) == [("model", "out", 0, "consumer-input")]
    assert b.bind(0, ["model"]) == [("model", "", 0, "unbound")], "the producer's own output is not the promoted Value"
    assert b.bind(0, ["model/out"]) == [("model", "out", 0, "unbound")]


def test_a_shared_instance_binds_its_canonical_path_and_invocation():
    rope = [row("m.rope0", "query", inv=k, spec="RoPE_v1{D=4}", port="out", cid=k) for k in range(2)]
    b = Binder(rope, aliases={"m.rope1": ["m.rope0", 1]})
    assert b.bind(0, ["m.rope0/query", "m.rope1/query"]) == [("m.rope0", "query", 0, "label"), ("m.rope0", "query", 1, "label")]


def test_rows_are_placed_on_the_engine_step_axis_as_the_population_is():
    rows = [row(LIN, "out", step=1, request_id="r1", engine_step=4, rank=2),
            {**row("fa2.m1/L0", "g1x2x3", family="fa2_hidden_m1_stream", rank=2), "request_id": "*", "step": "*", "engine_step": 5},
            row(LIN, "out", step=[0, 1], request_id="w-7", rank="*")]
    b = Binder(rows, rank=2, arrivals={"r1": 2, "r0": 0}, declared={"w-7": "r0"})
    assert sorted(b.rows) == [0, 1, 3, 5], "request-relative steps + arrival; the batch row's absolute engine step; a step range"
    assert sorted(Binder(rows, rank=2).rows) == [0, 1, 4, 5], "without the schedule: the rows' own engine steps"
    assert not Binder([row(LIN, "out", rank=1)], rank=0).rows


def test_two_sources_declaring_one_label_differently_are_refused():
    class S:
        def __init__(self, p):
            self.p = p

        def ports(self):
            return self.p

    assert source_ports([S(QUANT), S({"x_q": QUANT["x_q"]}), object()]) == QUANT
    with pytest.raises(ValueError, match="x_q"):
        source_ports([S(QUANT), S({"x_q": QUANT["x_s"]})])
