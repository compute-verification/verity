"""`Q_module_body_v1` and the v1 bridge on a hand-built request Program (no registry, no artifact).

    ids --Gather[embed]--> h0 --Mul[model]--> h1 --Add[model]--> h2 --RmsNorm[layernorm]--> h3 --Gemm[qkv_proj]--> q
      --RoPE[rotary_emb]/query--> qr --Add[model]--> h4 --Gemm[lm_head, tied embed weight]--> logits --Argmax[root]--> token

`model`'s own body has three Calls (Mul, Add, Add); `h1` never leaves it (interior), `h2` and `h4` do (boundary -- the F-r19-int-20
class: an interior-looking Call whose output another module reads is required because the dataflow says so, not by a pass).
"""

from __future__ import annotations

import pytest
from verity.verification.query import ValueRef, boundary
from verity_vllm.query.correspondence import Correspondence, implementation_prefix, parse_call_name
from verity_vllm.query.module_body import Q_module_body_v1
from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
from verity_vllm.query.v1_bridge import NAMED_RESIDUALS, member_name, request_manifest, shared_instances, under

W = "model.model"                                  # the export wrapper holds the served model under `model`
EMBED_W, QKV_W = f"{W}.embed.weight", f"{W}.layers.0.self_attn.qkv_proj.weight"


def _port(name="out", n=4, width=16):
    return Port(name, 0, n, n, width)


def _call(cid, family, name, reads, ports=("out",), n=4):
    return CallView(cid, f"{family}{{N={n}}}", family, "call", name, tuple(_port(p, n) for p in ports), tuple(reads))


def program() -> ProgramView:
    weights = InputView(-1, "weights", "{...}", (Port(EMBED_W, 0, 8, 8, 16), Port(QKV_W, 8, 16, 8, 16)))
    ids = InputView(-2, "ids", "Value<32>", (Port("ids", 0, 1, 1, 32),))
    R = ValueRef
    calls = (
        _call(0, "Gather_v1", f"{W}.embed/embedding/out[0]", [R(-2, "ids"), R(-1, EMBED_W)]),
        _call(1, "Mul_v1", f"{W}/mul[0]", [R(0, "out")]),
        _call(2, "Add_v1", f"{W}/add[0]", [R(1, "out")]),
        _call(3, "RmsNorm_v1", f"{W}.layers.0.input_layernorm/rms_norm/out[0]", [R(2, "out")]),
        _call(4, "Gemm_v1", f"{W}.layers.0.self_attn.qkv_proj/linear[0]", [R(3, "out"), R(-1, QKV_W)]),
        _call(5, "RoPE_v1", f"{W}.layers.0.self_attn.rotary_emb/rotary_embedding/query[0]", [R(4, "out")]),
        _call(6, "Add_v1", f"{W}/add_1[0]", [R(2, "out"), R(5, "out")]),
        _call(7, "Gemm_v1", f"{W}.lm_head/linear[0]", [R(6, "out"), R(-1, EMBED_W)]),
        _call(8, "ArgmaxTokenSelect_v1", "argmax[0]", [R(7, "out")], n=1),
    )
    return ProgramView(calls, (weights, ids), (ValueRef(8, "out"),), "synthetic", "synthetic")


def test_parse_call_name():
    n = parse_call_name("model.a.b/node/query[7]")
    assert (n.module, n.node, n.hint, n.row, n.suffix) == ("model.a.b", "node", "query", "7", "")
    n = parse_call_name("argmax_2[0]")
    assert (n.module, n.node, n.hint, n.row) == ("", "argmax_2", "", "0")
    n = parse_call_name("model.a/all_reduce[3]/all_reduce")
    assert (n.module, n.node, n.row, n.suffix) == ("model.a", "all_reduce", "3", "all_reduce")
    n = parse_call_name("model.a/node[t=3]")
    assert n.row == "t=3"


def test_implementation_prefix_is_the_common_suffix_map():
    mods = ["model.model.layers.0.self_attn.attn", "model.model.layers.1.self_attn.attn", "model.model"]
    assert implementation_prefix(mods, ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]) == "model."
    assert implementation_prefix(mods, ["model.model.layers.0.self_attn.attn"]) == ""
    assert implementation_prefix(mods, []) == ""
    assert implementation_prefix(mods, ["not.in.the.program"]) == ""


def test_module_bodies_and_boundary():
    P = program()
    corr = Correspondence(P, implementation_paths=[f"model.layers.0.self_attn.qkv_proj"])
    assert corr.prefix == "model."
    assert corr.implementation_path(f"{W}.embed") == "model.embed"
    B = boundary(P, Q_module_body_v1(P, corr))
    assert {u.id for u in B.units} == {W, f"{W}.embed", f"{W}.layers.0.input_layernorm", f"{W}.layers.0.self_attn.qkv_proj",
                                        f"{W}.layers.0.self_attn.rotary_emb", f"{W}.lm_head", ""}
    assert B.owner[1] == B.owner[2] == B.owner[6] == W
    R = ValueRef
    assert R(1, "out") not in B.values, "h1 is interior to `model`'s own body"
    assert {R(2, "out"), R(6, "out")} <= B.values, "h2 / h4 leave `model`'s body: boundary by dataflow (the F-r19-int-20 class)"
    assert B.values == {R(-2, "ids"), R(-1, EMBED_W), R(-1, QKV_W), R(0, "out"), R(2, "out"), R(3, "out"), R(4, "out"), R(5, "out"),
                        R(6, "out"), R(7, "out"), R(8, "out")}


def test_query_rejects_foreign_correspondence():
    P, P2 = program(), program()
    with pytest.raises(ValueError):
        Q_module_body_v1(P, Correspondence(P2))


def test_member_name_structure_then_residual():
    P = program()
    corr = Correspondence(P)
    gemm, rope, norm = P.call(4), P.call(5), P.call(3)
    assert member_name(gemm, gemm.port("out"), corr.name(4), own_parameter=True) == ("0", "out", NAMED_RESIDUALS[0])
    assert member_name(gemm, gemm.port("out"), corr.name(4), own_parameter=False)[:2] == ("out", "out")
    assert member_name(rope, rope.port("out"), corr.name(5), own_parameter=False)[0] == "0"
    assert member_name(norm, norm.port("out"), corr.name(3), own_parameter=False) == ("out", "out", None)
    two = _call(9, "Split_v1", "m/split[0]", [], ports=("0", "1"))
    assert member_name(two, two.port("1"), parse_call_name(two.name), own_parameter=False) == ("1", "1", None)


def test_under():
    assert under("a.b.c.weight", "a.b.c") and under("a.b.c.weight", "a")
    assert not under("a.b.c.weight", "a.b.c.weight") and not under("a.b.c.weight", "a.b.x") and not under("a.b", "")


def test_shared_instances_by_shared_parameter_and_signature():
    R = ValueRef
    w = InputView(-1, "weights", "{...}", (Port("m.rope0.inv_freq", 0, 4, 4, 32),))
    calls = (
        _call(0, "RoPE_v1", "m.rope0/rotary_embedding/query[0]", [R(-1, "m.rope0.inv_freq")]),
        _call(1, "RoPE_v1", "m.rope1/rotary_embedding/query[0]", [R(-1, "m.rope0.inv_freq")]),
        _call(2, "RoPE_v1", "m.rope2/rotary_embedding/query[0]", [R(-1, "m.rope0.inv_freq")]),
        _call(3, "Gemm_v1", "m.head/linear[0]", [R(-1, "m.rope0.inv_freq")]),          # a different body reading the same parameter
    )
    P = ProgramView(calls, (w,), (R(3, "out"),), None, "synthetic")
    corr = Correspondence(P)
    aliases, faults = shared_instances(P, corr, {R(-1, "m.rope0.inv_freq"): "m.rope0.inv_freq"})
    assert faults == []
    assert aliases == {"m.rope0": ("m.rope0", 0), "m.rope1": ("m.rope0", 1), "m.rope2": ("m.rope0", 2)}, "the Gemm body has another signature: not aliased"


def test_request_manifest_rows():
    P = program()
    corr = Correspondence(P, implementation_paths=[f"model.layers.0.self_attn.qkv_proj"])
    res = {"program": {"digest": "d0", "function": "f"}, "attention_impls_observed": {}, "target": {"declared": {"compute_capability": [8, 9]}}}
    r = request_manifest(P, corr, res, request_id="r0")
    man = r.manifest
    rows = {(x["family"], x["op_path"], x["output_member"], tuple(x["element_range"])) for x in man["identities"]}
    assert rows == {
        ("weights", "model.embed.weight", "weight", (0, None)), ("weights", "model.layers.0.self_attn.qkv_proj.weight", "weight", (0, None)),
        ("instance_outputs", "model.embed", "out", (0, 4)),
        ("instance_outputs", "model", "out", (0, 8)),                                  # h2 + h4: two crossing rows of `model`'s own body
        ("instance_outputs", "model.layers.0.input_layernorm", "out", (0, 4)),
        ("instance_outputs", "model.layers.0.self_attn.qkv_proj", "0", (0, 4)),        # Gemm_v1:out->0 (own parameter)
        ("instance_outputs", "model.layers.0.self_attn.rotary_emb", "0", (0, 4)),      # RoPE_v1:query->0
        ("instance_outputs", "model.lm_head", "out", (0, 4)),                          # tied weight is not its own: structural `out`
        ("sampled_token_ids", "runner.sampler", "sampled_token_ids", (0, 1)),
        ("input_token_ids", "model.embed", "in", (0, 1)),
    }
    assert man["complete"] and man["unmodelled"] == {}
    assert man["query"]["residuals_applied"] == {"Gemm_v1:out->0": 1, "RoPE_v1:query->0": 1, "encoding:in": 1, "encoding:sampled_token_ids": 1}
    assert man["populations"]["units"] == 7 and man["populations"]["steps"] == 1
    assert man["wrapper_prefix"] == "model."
    assert all(x["step"] == 0 for x in man["identities"] if x["family"] != "weights")
    assert [x["output_member"] for x in man["identities"] if x["op_path"] == "model"] == ["out"]
    body = next(x for x in man["identities"] if x["op_path"] == "model")
    assert body["boundary"] == {"crossing_rows": 2, "returned_rows": 2, "structural_member": "out"}   # a crossing Value is a returned tensor too


def test_request_manifest_refuses_a_tp_rank_program():
    """A rank Program's population of record names its rank-local partials (TP-04); the v2 query does not, so it refuses by name."""
    P = program()
    res = {"program": {"digest": "d0", "function": "f"}, "target": {"declared": {"compute_capability": [8, 9]}}, "tp": {"world": 2, "rank": 1}}
    with pytest.raises(ValueError, match="TP rank Program .*required_manifest build-global"):
        request_manifest(P, Correspondence(P), res, request_id="r0", rank=1, ranks=2)
    with pytest.raises(ValueError, match="rank Program mismatch"):
        request_manifest(P, Correspondence(P), res, request_id="r0")


def test_return_slot_policy_is_switchable():
    """A recorded return slot the dataflow never reads is required only under the RETURN-SLOT policy."""
    P = program()
    R = ValueRef
    dangling = _call(8, "RmsNorm_v1", f"{W}.norm/rms_norm/out[0]", [R(6, "out")])         # returned by `norm`, read by nobody
    select = _call(9, "ArgmaxTokenSelect_v1", "argmax[0]", [R(7, "out")], n=1)
    P2 = ProgramView(P.calls[:8] + (dangling, select), P.params, (R(9, "out"),), None, "synthetic")
    res = {"program": {"digest": "d0", "function": "f"}, "target": {"declared": {"compute_capability": [8, 9]}}}
    with_slots = request_manifest(P2, Correspondence(P2), res, request_id="r0").manifest
    without = request_manifest(P2, Correspondence(P2), res, request_id="r0", return_slots=False).manifest
    paths = lambda m: {x["op_path"] for x in m["identities"] if x["family"] == "instance_outputs"}  # noqa: E731
    assert f"{W}.norm" in paths(with_slots) and f"{W}.norm" not in paths(without)
    assert with_slots["query"]["policy"]["protocol_required"][:2] == ["sampling-event outputs", "recorded return slots (RETURN-SLOT)"]
    assert without["query"]["policy"]["protocol_required"] == ["sampling-event outputs"]


def test_tensor_return_policy_requires_the_whole_node_output():
    """A fused norm over 3 rows returns (normed, residual); only the last row's `0` is read (by the logits).  The module returned the
    whole tensors: all 3 rows of `0` and of `1` are required (TENSOR-RETURN); off-policy only the crossing row is."""
    R = ValueRef
    w = InputView(-1, "weights", "{...}", (Port(f"{W}.lm_head.weight", 0, 4, 4, 16),))
    h = InputView(-2, "h", "Array<3,Value<16>>", (Port("h", 0, 12, 12, 16),))
    calls = tuple(_call(i, "FusedNorm_v1", f"{W}.norm/fused_add_rms_norm_9[{i}]", [R(-2, "h")], ports=("0", "1")) for i in range(3)) + (
        _call(3, "Gemm_v1", f"{W}.lm_head/linear[0]", [R(2, "0"), R(-1, f"{W}.lm_head.weight")]),
        _call(4, "ArgmaxTokenSelect_v1", "argmax[0]", [R(3, "out")], n=1))
    P = ProgramView(calls, (w, h), (R(4, "out"),), None, "synthetic")
    res = {"program": {"digest": "d0", "function": "f"}, "target": {"declared": {"compute_capability": [8, 9]}}}
    on = request_manifest(P, Correspondence(P), res, request_id="r0").manifest
    off = request_manifest(P, Correspondence(P), res, request_id="r0", return_slots=False).manifest
    norm = lambda m: {(x["output_member"], x["rows"]) for x in m["identities"] if x["op_path"] == f"{W}.norm" and x["family"] == "instance_outputs"}  # noqa: E731
    assert norm(on) == {("0", 3), ("1", 3)}
    assert norm(off) == {("0", 1)}
    row0 = next(x for x in on["identities"] if x["op_path"] == f"{W}.norm" and x["output_member"] == "0")
    assert row0["boundary"] == {"crossing_rows": 1, "returned_rows": 3, "structural_member": "0"}
