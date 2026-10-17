"""The binder's passes (`verity_vllm.commit.bind`) on hand-written manifest rows: what each pass binds, and what it refuses to guess."""
from __future__ import annotations

import pytest

from verity_vllm.commit.bind import Binder, Port, source_ports

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
