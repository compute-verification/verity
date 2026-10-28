"""`Q_module_body_v1` and the v1 bridge on a hand-built request Program (no registry, no artifact).

    ids --Gather[embed]--> h0 --Mul[model]--> h1 --Add[model]--> h2 --RmsNorm[layernorm]--> h3 --Gemm[qkv_proj]--> q
      --RoPE[rotary_emb]/query--> qr --Add[model]--> h4 --Gemm[lm_head, tied embed weight]--> logits --Argmax[root]--> token

`model`'s own body has three Calls (Mul, Add, Add); `h1` never leaves it (interior), `h2` and `h4` do (boundary: an interior-looking Call whose output another module reads is required because the dataflow says so, not by a pass).
"""

from __future__ import annotations

import pytest
from verity_vllm.query.units import ValueRef, boundary
from verity_vllm.correspondence.reader_for_query import Correspondence, implementation_prefix, parse_call_name
from verity_vllm.query.module_body import Q_module_body_v1
from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
from verity_vllm.query.manifest import format as fmt
from verity_vllm.query.compose import merge_ranks
from verity_vllm.query.required import (Q_MODULE_BODY_ID, Q_WORD_ID, member_name, partition_of, request_manifest, shared_instances,
                                        under)

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
    # GLM-5.3: MLA attention is a custom op inside `self_attn`, which is the Program's module; the observed `.attn` is its child
    glm = ["inner.model.model.layers.0.self_attn", "inner.model.model.layers.1.self_attn", "inner.model.model.embed_tokens"]
    assert implementation_prefix(glm, ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]) == "inner.model."


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
    assert {R(2, "out"), R(6, "out")} <= B.values, "h2 / h4 leave `model`'s body: boundary by dataflow"
    assert B.values == {R(-2, "ids"), R(-1, EMBED_W), R(-1, QKV_W), R(0, "out"), R(2, "out"), R(3, "out"), R(4, "out"), R(5, "out"),
                        R(6, "out"), R(7, "out"), R(8, "out")}


def test_query_rejects_foreign_correspondence():
    P, P2 = program(), program()
    with pytest.raises(ValueError):
        Q_module_body_v1(P, Correspondence(P2))


def test_member_name_is_the_programs_own():
    """The port of a multi-port Definition (hint-qualified when the Build recorded one), else the recorded hint, else `out`."""
    P = program()
    corr = Correspondence(P)
    gemm, rope, norm = P.call(4), P.call(5), P.call(3)
    assert member_name(gemm, gemm.port("out"), corr.name(4)) == "out"
    assert member_name(rope, rope.port("out"), corr.name(5)) == "query"
    assert member_name(norm, norm.port("out"), corr.name(3)) == "out"
    two = _call(9, "Split_v1", "m/split[0]", [], ports=("0", "1"))
    assert member_name(two, two.port("1"), parse_call_name(two.name)) == "1"
    hinted = _call(10, "Split_v1", "m/split/index_q[0]", [], ports=("0", "1"))
    assert member_name(hinted, hinted.port("1"), parse_call_name(hinted.name)) == "index_q.1"


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
    r = request_manifest(P, corr, res, request_id="r0", query=Q_MODULE_BODY_ID)
    man = r.manifest
    rows = {(x["family"], x["op_path"], x["output_member"], tuple(x["element_range"])) for x in man["identities"]}
    assert rows == MODULE_BODY_ROWS == {
        ("weights", "model.embed.weight", "weight", (0, None)), ("weights", "model.layers.0.self_attn.qkv_proj.weight", "weight", (0, None)),
        ("instance_outputs", "model.embed", "out", (0, 4)),
        ("instance_outputs", "model/add", "out", (0, 4)),                              # h2 and h4 cross `model`'s own body: two producers,
        ("instance_outputs", "model/add_1", "out", (0, 4)),                            # each spelt at its fx node
        ("instance_outputs", "model.layers.0.input_layernorm", "out", (0, 4)),
        ("instance_outputs", "model.layers.0.self_attn.qkv_proj", "out", (0, 4)),
        ("instance_outputs", "model.layers.0.self_attn.rotary_emb", "query", (0, 4)),  # the Build's hint names the Call
        ("instance_outputs", "model.lm_head", "out", (0, 4)),
        ("sampled_token_ids", "runner.sampler", "sampled_token_ids", (0, 1)),
        ("input_token_ids", "model.embed", "in", (0, 1)),
    }
    assert man["complete"] and man["unmodelled"] == {}
    assert man["populations"]["units"] == 7 and man["populations"]["steps"] == 1
    assert man["wrapper_prefix"] == "model."
    assert all(x["step"] == 0 for x in man["identities"] if x["family"] != "weights")
    assert not [x for x in man["identities"] if x["op_path"] == "model"]
    body = next(x for x in man["identities"] if x["op_path"] == "model/add")
    assert body["boundary"] == {"crossing_rows": 1, "returned_rows": 1}   # a crossing Value is a returned tensor too
    assert (body["spec"], body["port"], body["position"]) == ("Add_v1", "out", [2, 0]), "the first Call of the group, its port and index"
    assert man["query"]["query_id"] == Q_MODULE_BODY_ID and "partition" not in man["query"]


MODULE_BODY_ROWS = {
    ("weights", "model.embed.weight", "weight", (0, None)), ("weights", "model.layers.0.self_attn.qkv_proj.weight", "weight", (0, None)),
    ("instance_outputs", "model.embed", "out", (0, 4)),
    ("instance_outputs", "model/add", "out", (0, 4)), ("instance_outputs", "model/add_1", "out", (0, 4)),
    ("instance_outputs", "model.layers.0.input_layernorm", "out", (0, 4)), ("instance_outputs", "model.layers.0.self_attn.qkv_proj", "out", (0, 4)),
    ("instance_outputs", "model.layers.0.self_attn.rotary_emb", "query", (0, 4)), ("instance_outputs", "model.lm_head", "out", (0, 4)),
    ("sampled_token_ids", "runner.sampler", "sampled_token_ids", (0, 1)), ("input_token_ids", "model.embed", "in", (0, 1)),
}


def test_the_query_of_record_commits_every_call_boundary_and_names_the_partition():
    """Under Q_word, h1 (Mul -> Add inside `model`'s own body) crosses Calls: required, spelt under its fx node (`call_boundaries`); every
    module-body row stays as it was.  The header names the query and, given the Program's SHA-512, its verity/partition/v1 object."""
    P = program()
    corr = Correspondence(P, implementation_paths=[f"model.layers.0.self_attn.qkv_proj"])
    res = {"program": {"digest": "d0", "function": "f"}, "attention_impls_observed": {}, "target": {"declared": {"compute_capability": [8, 9]}}}
    sha = "ab" * 64
    man = request_manifest(P, corr, res, request_id="r0", program_digest=sha).manifest
    rows = {(x["family"], x["op_path"], x["output_member"], tuple(x["element_range"])) for x in man["identities"]}
    assert rows == MODULE_BODY_ROWS | {("call_boundaries", "model/mul", "out", (0, 4))}
    assert man["complete"] and man["unmodelled"] == {}
    q = man["query"]
    assert q["query_id"] == Q_WORD_ID and q["boundary"] == "call" and q["engine"] == "q-word"
    assert "call_boundaries" in q["required_families"]
    assert q["partition"] == partition_of(sha)
    assert q["partition"]["object"] == {"format": "verity/partition/v1", "program": sha, "query": {"name": "Q_word", "version": 1, "params": {"X": 16, "W": 32}}}
    assert man["populations"]["units"] == 9 and man["populations"]["boundary_values"] == 12
    assert "partition" not in request_manifest(P, corr, res, request_id="r0").manifest["query"], "no SHA-512, no partition object"


DOWN_W, LM_W = f"{W}.layers.0.mlp.down_proj.weight", f"{W}.lm_head.weight"
SITE = f"{W}.logits_processor.all_gather2_0"


def tp_program(*, two_operands: bool = False) -> ProgramView:
    """A world-2 rank Program at toy size: the embedding shard all-reduced in its own body, a row-parallel down_proj (Gemm + all-reduce in
    one body), and lm_head's vocab shard gathered at a collective SITE (a module body of the gather only) under logits_processor.  Every
    collective reads the rank's own partial and a `peers_<k>` input (the received peer partial) that only collectives read."""
    weights = InputView(-1, "weights", "{...}", (Port(EMBED_W, 0, 8, 8, 16), Port(DOWN_W, 8, 16, 8, 16), Port(LM_W, 24, 8, 8, 16)))
    ids = InputView(-2, "ids", "Value<32>", (Port("ids", 0, 1, 1, 32),))
    peers = tuple(InputView(-3 - k, f"peers_{k}", "Array<1,Array<4,Value<16>>>", (Port(f"peers_{k}", 0, 4, 4, 16),)) for k in range(3))
    R = ValueRef
    calls = (
        _call(0, "Gather_v1", f"{W}.embed/embedding/out[0]", [R(-2, "ids"), R(-1, EMBED_W)]),
        _call(1, "AllReduce2_v1", f"{W}.embed/all_reduce[0]/all_reduce", [R(0, "out"), R(-3, "peers_0")]),
        _call(2, "RmsNorm_v1", f"{W}.layers.0.input_layernorm/rms_norm/out[0]", [R(1, "out")]),
        _call(3, "Gemm_v1", f"{W}.layers.0.mlp.down_proj/linear[0]", [R(2, "out"), R(-1, DOWN_W)]),
        _call(4, "AllReduce2_v1", f"{W}.layers.0.mlp.down_proj/all_reduce[0]/all_reduce",
              [R(3, "out"), R(-4, "peers_1")] + ([R(2, "out")] if two_operands else [])),
        _call(5, "Add_v1", f"{W}/add[0]", [R(1, "out"), R(4, "out")]),
        _call(6, "Gemm_v1", f"{W}.lm_head/linear[0]", [R(5, "out"), R(-1, LM_W)], n=2),
        _call(7, "AllGather2_v1", f"{SITE}/all_gather[0]/all_gather", [R(6, "out"), R(-5, "peers_2")]),
        _call(8, "ArgmaxTokenSelect_v1", "argmax[0]", [R(7, "out")], n=1),
    )
    return ProgramView(calls, (weights, ids) + peers, (ValueRef(8, "out"),), "synthetic", "synthetic")


def _tp_manifest(rank: int, P: ProgramView | None = None) -> dict:
    P = P or tp_program()
    res = {"program": {"digest": f"d{rank}", "function": "f"}, "target": {"declared": {"compute_capability": [8, 9]}}, "tp": {"world": 2, "rank": rank}}
    return request_manifest(P, Correspondence(P), res, request_id="r0", rank=rank, ranks=2).manifest


def test_a_rank_program_requires_its_rank_local_partials_beside_every_collective_output():
    """A collective's output and its rank-local partial are required on every rank.  A partial produced in the collective's own body
    is its producer's Value (`part_rank<r>`); a collective site whose operand another body produced names its partial itself (`shard_rank<r>`,
    1/world of the gathered width) at the enclosing module, invocation 0.  The received `peers_<k>` inputs are neither token ids nor rows."""
    for rank in (0, 1):
        m = _tp_manifest(rank)
        assert m["complete"], m["unmodelled"]
        rows = {(x["family"], x["op_path"], x["output_member"]): x for x in m["identities"] if x["family"] != "weights"}
        partials = {(op, mem): x["element_range"][1] for (fam, op, mem), x in rows.items() if fam == "tp_rank_partials"}
        assert partials == {(f"{W}.embed", f"part_rank{rank}"): 4, (f"{W}.layers.0.mlp.down_proj", f"part_rank{rank}"): 4,
                            (f"{W}.logits_processor", f"shard_rank{rank}"): 2}
        assert rows[("tp_rank_partials", f"{W}.logits_processor", f"shard_rank{rank}")]["program_op_path"] == SITE
        assert rows[("instance_outputs", f"{W}.logits_processor", "out")]["invocation"] == 0
        assert rows[("instance_outputs", f"{W}.layers.0.mlp.down_proj", "out")]["spec"].startswith("AllReduce2_v1"), "the all-reduce's output"
        assert ("instance_outputs", f"{W}.embed", "out") in rows
        assert {x["op_path"] for x in m["identities"] if x["family"] == "input_token_ids"} == {f"{W}.embed"}
        assert m["op_path_aliases"][SITE] == [f"{W}.logits_processor", 0]
    res = {"program": {"digest": "d0", "function": "f"}, "target": {"declared": {"compute_capability": [8, 9]}}, "tp": {"world": 2, "rank": 1}}
    with pytest.raises(ValueError, match="rank Program mismatch"):
        request_manifest(tp_program(), Correspondence(tp_program()), res, request_id="r0")


def test_a_collective_with_two_rank_local_operands_is_named_and_the_rank_manifest_is_incomplete():
    m = _tp_manifest(0, tp_program(two_operands=True))
    assert not m["complete"] and any("2 rank-local operands" in k for k in m["unmodelled"])


def test_merge_ranks_binds_every_collective_output_to_the_peers_partial_and_keeps_each_ranks_digest():
    m0, m1 = _tp_manifest(0), _tp_manifest(1)
    g = merge_ranks([m1, m0])
    assert g["complete"] and g["tp_peer_binding"]["n_unbound"] == 0 and g["tp_peer_binding"]["bound"] == 6
    assert g["program_digest_kind"] == "tp-ranks-fold" and g["program_digest"] == fmt.tp_ranks_fold_digest([m0["program_digest"], m1["program_digest"]])
    assert g["program_digest_per_rank"] == {"0": m0["program_digest"], "1": m1["program_digest"]}
    assert {x["program_digest"] for x in g["identities"] if x["rank"] == 1} == {m1["program_digest"]}
    assert len(g["identities"]) == len(m0["identities"]) + len(m1["identities"])
    dead = dict(m1["identities"][-1], output_member="dead")
    listed = merge_ranks([m0, dict(m1, unreturned=[dead])])
    assert listed["unreturned"] == [dead] and listed["unreturned_digest"] == fmt.manifest_digest([dead]) != g["unreturned_digest"]
    lost = dict(m1, identities=[x for x in m1["identities"] if x["output_member"] != "shard_rank1"])
    bad = merge_ranks([m0, lost])
    assert not bad["complete"] and bad["tp_peer_binding"]["n_unbound"] == 1 and any(k[:16] == "tp peer binding:" for k in bad["unmodelled"])
    with pytest.raises(ValueError, match="needs every rank"):
        merge_ranks([m0, m0])
    with pytest.raises(ValueError, match="one Program digest"):
        merge_ranks([m0, dict(m1, program_digest=m0["program_digest"])])


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
    assert row0["boundary"] == {"crossing_rows": 1, "returned_rows": 3}
