"""The binder's passes (`verity_vllm.query.manifest.bind`) on hand-written manifest rows: what each pass binds, and what it refuses to guess."""
from __future__ import annotations

import pytest

from verity_vllm.query.manifest.bind import Binder, Port, coverage_check, source_ports

LIN, ATT = "model.layers.0.mlp.down_proj", "model.layers.0.self_attn"
QUANT = {"x_q": Port(frozenset({"DeepGemmFp8Linear_v1"}), "0"), "x_s": Port(frozenset({"DeepGemmFp8Linear_v1"}), "1")}


def row(op: str, member: str, *, step: int | list[int] = 0, inv: int = 0, spec: str = "Gemm_v1{K=4}", port: str | None = None, cid: int = 0, **kw) -> dict:
    return {"family": "instance_outputs", "request_id": "r0", "step": step, "invocation": inv, "rank": 0, "op_path": op,
            "output_member": member, "spec": spec, "port": port or member, "position": [cid, 0 if port is None or port == "out" else int(port)], **kw}


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


def test_a_row_holds_its_keys_fields_itself_or_under_identity():
    flat = row(LIN, "out")
    want = Binder([flat]).rows
    assert want and Binder([{"identity": flat}]).rows == want, "the fields under `identity` read as the row's own"
    empties: tuple[object, ...] = (None, {}, [], 0, "", False)
    for empty in empties:
        assert Binder([{**flat, "identity": empty}]).rows == want, "an empty `identity` adds nothing"
    with pytest.raises(TypeError, match="is not a mapping"):
        Binder([{**flat, "identity": ["x"]}])


@pytest.mark.parametrize("no_mapping", [["x"], [], "x", 1, None])
def test_a_row_that_is_no_mapping_is_refused_before_its_identity_is_read(no_mapping):
    with pytest.raises(TypeError, match="is not a mapping"):
        Binder([no_mapping])


def test_two_sources_declaring_one_label_differently_are_refused():
    class S:
        def __init__(self, p):
            self.p = p

        def ports(self):
            return self.p

    assert source_ports([S(QUANT), S({"x_q": QUANT["x_q"]}), object()]) == QUANT
    with pytest.raises(ValueError, match="x_q"):
        source_ports([S(QUANT), S({"x_q": QUANT["x_s"]})])


def test_where_a_row_and_its_identity_disagree_the_identity_is_read():
    flat, other = row(LIN, "out"), row(ATT, "0", step=1, inv=2)
    assert Binder([flat]).rows != Binder([other]).rows
    assert Binder([{**flat, "identity": other}]).rows == Binder([other]).rows, "every field under `identity` is read before the row's own"
    assert Binder([{**flat, "identity": {"op_path": ATT}}]).rows == Binder([{**flat, "op_path": ATT}]).rows, "and one field alone is"


def test_an_incomplete_manifest_covers_nothing_and_names_what_it_leaves_out():
    rows = [row(LIN, "out"), row(LIN, "out", step=1)]
    lay = {0: [{"name": LIN, "numel": 4}], 1: [{"name": LIN, "numel": 4}]}
    got = coverage_check({"complete": False, "unmodelled": {"Conv_v1": 2, "Abs_v1": 1}, "identities": rows, "manifest_digest": "d" * 64}, lay)
    assert not got["ok"] and got["error"] == "manifest incomplete: unmodelled Program families Abs_v1 x1; Conv_v1 x2"
    assert (got["checked"], got["missing_n"], got["missing_by_family"]) == (2, 2, {"(unmodelled)": 2})
    assert got["families_missing_entirely"] == ["Abs_v1", "Conv_v1"] and got["unmodelled"] == {"Conv_v1": 2, "Abs_v1": 1}
    assert got["manifest_digest"] == "d" * 64 and got["missing"] == [] and got["committed_extras_n"] == 0
    bare = coverage_check({"complete": False}, lay)             # one that names no family and holds no row is refused the same way
    assert not bare["ok"] and (bare["checked"], bare["unmodelled"], bare["families_missing_entirely"]) == (0, {}, [])
    assert bare["error"] == "manifest incomplete: unmodelled Program families "


MOE = "moe/L0"


def composed(rows: list[dict], *, tokens: int | None = None, **more) -> dict:
    """A global manifest of one engine step that batches r0 (3 token rows) and r1 (1); `tokens`: the rows the step states."""
    step = {"engine_step": 0, "members": [{"request_id": "r0", "rows": 3}, {"request_id": "r1", "rows": 1}]}
    return {"global": True, "engine_steps": [step if tokens is None else {**step, "rows": tokens}], "identities": rows, "manifest_digest": "d" * 64, **more}


def test_a_row_with_no_step_of_its_own_is_on_no_engine_step_unless_it_states_one():
    every = {**row(LIN, "out"), "step": "*"}
    none = {k: v for k, v in row(LIN, "out").items() if k != "step"}
    assert not Binder([every]).rows and not Binder([none]).rows
    assert not Binder([every, none], arrivals={"r0": 2}).rows, "an arrival moves a step: it gives a row none"
    assert sorted(Binder([{**every, "engine_step": 3}, {**none, "engine_step": 5}]).rows) == [3, 5]


def test_a_row_of_no_request_stays_on_its_own_engine_step_under_a_schedule():
    null = {**row(LIN, "out", step=1, engine_step=4), "request_id": None}
    absent = {k: v for k, v in row(ATT, "out", step=1, engine_step=6).items() if k != "request_id"}
    assert sorted(Binder([null, absent], arrivals={"r0": 2}).rows) == [4, 6], "no request, so no arrival moves its step"
    assert sorted(Binder([null, absent]).rows) == [4, 6]
    assert sorted(Binder([{**null, "request_id": "r0"}], arrivals={"r0": 2}).rows) == [3], "the same row of a request is moved"


def test_coverage_is_of_one_ranks_rows_and_of_one_requests_when_one_is_named():
    rows = [row(LIN, "out", element_range=[0, 4]), row(ATT, "out", request_id="r1", element_range=[0, 4]),
            row(ATT, "out", inv=1, rank=1, element_range=[0, 4]), row("model.norm", "out", request_id="*", element_range=[0, 2])]
    manifest = {"identities": rows, "manifest_digest": "d" * 64}
    lay = {0: [{"name": LIN, "numel": 4}, {"name": "model.norm", "numel": 2}]}
    every = coverage_check(manifest, lay)
    assert (every["ok"], every["checked"], every["requests_with_missing"]) == (False, 3, ["r1"]), "rank 0's three rows; r1's is not committed"
    r0 = coverage_check(manifest, lay, request_id="r0")
    assert (r0["ok"], r0["checked"], r0["missing_n"]) == (True, 2, 0), "r0's row and the row of every request"
    r1 = coverage_check(manifest, lay, request_id="r1")
    assert (r1["ok"], r1["checked"], [m["op_path"] for m in r1["missing"]]) == (False, 2, [ATT])
    rank1 = coverage_check(manifest, lay, rank=1)
    assert (rank1["checked"], [(m["op_path"], m["invocation"]) for m in rank1["missing"]], rank1["committed_extras_n"]) == (1, [(ATT, 1)], 2)


def test_a_requests_range_of_a_batched_moe_plane_must_fit_inside_its_own_token_rows():
    """In a global manifest a MoE block plane is one tensor a step over its requests: r0 has 3 of the step's 4 token rows."""
    def why(layout: dict, hi: int, *, request: str = "r0", tokens: int | None = None) -> str | None:
        plane = row(MOE, "up", family="moe_block_stream", request_id=request, element_range=[0, hi])
        got = coverage_check(composed([plane], tokens=tokens), {0: [{"name": f"{MOE}/up", **layout}]})
        assert got["ok"] == (not got["missing"]) and got["checked"] == 1
        return got["missing"][0]["why"] if got["missing"] else None

    assert why({"numel": 12}, 9) is None, "3 words a token row: r0's three rows hold 9"
    assert why({"numel": 12}, 10) == "the request-local range ends at 10; the request's 3 token rows of the plane hold 9 words"
    assert why({"numel": 13}, 1) == "13 words do not split into the step's 4 token rows"
    assert why({"numel": 12}, 1, request="r2") == "the request has no declared rows at the step"
    assert why({"numel": 12}, 1, tokens=0) == "the step declares no token rows"
    assert why({"numel": 12}, 6, tokens=6) is None, "the rows the step states, not its members' sum: 2 words a token row"
    assert why({"numel": 12}, 7, tokens=6) == "the request-local range ends at 7; the request's 3 token rows of the plane hold 6 words"
    assert why({"shape": [8, 2]}, 12) is None, "two rows a token: r0's share is 6 rows of 2 words"
    assert why({"shape": [8, 2]}, 13) == "the request-local range ends at 13; the request's 3 token rows of the plane hold 12 words"
    for rows in (6, 2, 0):
        assert why({"shape": [rows, 2]}, 1) == f"{rows} rows, not a multiple of the step's 4 token rows: the plane covers no request"


def test_a_global_manifests_token_ids_are_held_to_the_declared_composition_of_each_engine_step():
    """The packed ids member has the step's declared token rows and the sampled ids one slot a sequence; a member the manifest's
    `composition_reference` does not name is held to nothing, and one that is not committed is missing, not mismatched."""
    ids, sampled = ("model.embed", "in"), ("runner.sampler", "sampled_token_ids")
    rows = [row(*ids, family="input_token_ids", element_range=[0, 3]), row(*sampled, family="sampled_token_ids", element_range=[0, 1])]
    ref = {f: {"op_path": op, "output_member": member} for f, (op, member) in (("input_token_ids", ids), ("sampled_token_ids", sampled))}

    def check(n_ids: int = 4, n_sampled: int = 2, reference: dict | None = None, request_id: str | None = None, **step) -> dict:
        manifest = composed(rows, composition_reference=ref if reference is None else reference)
        manifest["engine_steps"][0].update(step)
        lay = [{"name": "/".join(m), "numel": n} for m, n in ((ids, n_ids), (sampled, n_sampled)) if n]
        return coverage_check(manifest, {0: lay}, request_id=request_id)

    def mismatch(member: tuple[str, str], declared: int, committed: int) -> list[dict]:
        return [{"engine_step": 0, "member": "/".join(member), "declared": declared, "committed": committed, "requests": ["r0", "r1"]}]

    whole = check()
    assert (whole["ok"], whole["composition_mismatch_n"], whole["composition_mismatch"]) == (True, 0, [])
    short = check(n_ids=3)
    assert (short["ok"], short["missing_n"], short["steps_with_missing"]) == (False, 0, [0]), "every identity is covered: the step's size is not"
    assert (short["composition_mismatch_n"], short["composition_mismatch"]) == (1, mismatch(ids, 4, 3))
    assert check(n_sampled=1)["composition_mismatch"] == mismatch(sampled, 2, 1) and check(n_sampled=3)["composition_mismatch"] == mismatch(sampled, 2, 3)
    assert check(n_ids=5, rows=5)["ok"] and check(rows=5)["composition_mismatch"] == mismatch(ids, 5, 4), "the rows the step states, not its members' sum"
    assert check(n_sampled=3, n_seq=3)["ok"] and check(n_seq=3)["composition_mismatch"] == mismatch(sampled, 3, 2)
    wide = {**ref, "input_token_ids": {**ref["input_token_ids"], "per_row": 2}}
    assert check(n_ids=8, reference=wide)["ok"] and check(reference=wide)["composition_mismatch"] == mismatch(ids, 4, 4)
    unnamed = check(n_ids=3, reference={"sampled_token_ids": ref["sampled_token_ids"]})
    assert (unnamed["ok"], unnamed["composition_mismatch"]) == (True, []), "no reference for the ids: their size is held to nothing"
    absent = check(n_sampled=0)
    assert (absent["ok"], [m["output_member"] for m in absent["missing"]], absent["composition_mismatch"]) == (False, ["sampled_token_ids"], [])
    assert check(n_ids=3, request_id="r0")["ok"], "one request's coverage does not see the step's size"
