"""The query reads the owning module off ONE reader -- the `runtime-correspondence/v1` record -- and falls back to the v1-annotation
names only when a Build carries none; the source is recorded, and on the same Program both give the same partition and manifest."""
from __future__ import annotations

import gzip
import json

import pytest
from verity.verification.query import ValueRef, boundary
from verity_vllm import runtime_correspondence as RC
from verity_vllm.query.correspondence import CallName, Correspondence, call_name_of, parse_call_name
from verity_vllm.query.module_body import Q_module_body_v1
from verity_vllm.query.program_view import InputView, Port, ProgramView
from verity_vllm.query.v1_bridge import merge_ranks, request_manifest

from .test_module_body import W, _call, program


def result_json() -> dict:
    return {"program": {"digest": "d0", "function": "f"}, "attention_impls_observed": {}, "target": {"declared": {"compute_capability": [8, 9]}}}


def _record(P) -> dict:
    """The record the frontend would have emitted for `program()`: one occurrence per (module, node), Call k -> its occurrence; the
    frontend's naming suffix kept as `member` / `row` where its parser reads it, RAW as `hint` where it does not (`.../<fx>/<member>[row]`)."""
    occs, calls, seen = [], [], {}
    for c in P.calls:
        n = parse_call_name(c.name)
        key = (n.module, n.node)
        if key not in seen:
            seen[key] = len(occs)
            occs.append({"node": n.node, "module": n.module, "module_class": None, "op": "aten.op", "rule": None, "outputs": [{"shape": [4], "dtype": "bf16"}],
                         "written": [], "inputs": [], "lifetime": {}, "collective": None})
        if n.hint:                                                              # `<module>/<fx>/<member>[row]`: the record keeps the raw name
            calls.append({"occ": seen[key], "member": None, "row": None, "hint": c.name, "returns": [{"port": 0, "output": 0, "leaves": [0, 4]}], "args": []})
        else:
            calls.append({"occ": seen[key], "member": None, "row": int(n.row), "hint": None, "returns": [{"port": 0, "output": 0, "leaves": [0, 4]}], "args": []})
    return {"schema": RC.SCHEMA, "body": "program_fn", "inputs": ["ids"], "weights": None, "rank": None, "occurrences": occs, "calls": calls}


def test_record_and_fallback_agree_call_for_call_and_the_source_is_recorded():
    P = program()
    table = RC.RuntimeCorrespondence.from_json(_record(P))
    canon = Correspondence(P, table=table)
    fallback = Correspondence(P)
    assert canon.source == "correspondence" and fallback.source == "v1_annotations"
    assert canon.digest == RC.correspondence_digest(table.doc) and fallback.digest is None
    for c in P.calls:
        assert canon.module_of(c.id) == fallback.module_of(c.id), c.name
        assert canon.name(c.id) == fallback.name(c.id), c.name
    Qa, Qb = Q_module_body_v1(P, canon), Q_module_body_v1(P, fallback)
    assert {u.id: u.calls for u in Qa(P)} == {u.id: u.calls for u in Qb(P)}
    assert boundary(P, Qa).values == boundary(P, Qb).values
    res = result_json()
    ma = request_manifest(P, canon, res, request_id="r0").manifest
    mb = request_manifest(P, fallback, res, request_id="r0").manifest
    assert ma["manifest_digest"] == mb["manifest_digest"] and ma["identities"] == mb["identities"]
    assert ma["correspondence"]["source"] == "correspondence" and ma["correspondence"]["digest"] == canon.digest and ma["correspondence"]["schema"] == RC.SCHEMA
    assert mb["correspondence"]["source"] == "v1_annotations" and "fallback" in mb["correspondence"] and mb["correspondence"]["digest"] is None
    pa = canon.provenance()
    assert pa["source"] == "correspondence" and "problem" not in pa


def test_call_name_of_the_record():
    rec = RC.CallCorrespondence(0, f"{W}.a", None, "linear", "aten.mm", None, "0", 7, None, (), (), {}, None, None)
    assert call_name_of(rec) == CallName(f"{W}.a", "linear", "0", "7", "")
    raw = RC.CallCorrespondence(1, f"{W}.b", None, "rope", "aten.op", None, None, None, f"{W}.b/rope/query[3]", (), (), {}, None, None)
    assert call_name_of(raw) == CallName(f"{W}.b", "rope", "query", "3", "")
    root = RC.CallCorrespondence(2, "", None, None, None, None, None, None, "argmax_2[0]", (), (), {}, None, None)
    assert call_name_of(root) == CallName("", "argmax_2", "", "0", "")
    coll = RC.CallCorrespondence(3, f"{W}.c", None, "all_reduce", "verity_tp.all_reduce", None, None, None, f"{W}.c/all_reduce[3]/all_reduce", (), (), {}, None, None)
    assert call_name_of(coll) == CallName(f"{W}.c", "all_reduce", "", "3", "all_reduce")
    parsed = RC.CallCorrespondence(4, f"{W}.c", None, "all_reduce2_7", "verity_tp.all_reduce", None, "all_reduce", 3, None, (), (), {}, None,
                                   RC.CollectiveOccurrence("all_reduce", 7, (0, 1), 0, 1, "peers_7"))
    assert call_name_of(parsed) == CallName(f"{W}.c", "all_reduce2_7", "", "3", "all_reduce"), "a collective's kind is its tag, not a return slot"


ATTN = f"{W}.layers.0.self_attn"
QKV_W, O_W = f"{ATTN}.qkv_proj.weight", f"{ATTN}.o_proj.weight"


def tp_attention_program() -> ProgramView:
    """A world-2 rank Program shaped like OLMoE attention: q and k all-gathered from `self_attn`'s own forward (each reading qkv_proj's
    output: two collective SITES), their norms, and a row-parallel o_proj (Gemm + all-reduce in one body).  The fallback names spell the
    sites as the v1 frontend does, `<self_attn>.<fx node>`."""
    R = ValueRef
    weights = InputView(-1, "weights", "{...}", (Port(QKV_W, 0, 8, 8, 16), Port(O_W, 8, 8, 8, 16)))
    ids = InputView(-2, "ids", "Value<32>", (Port("ids", 0, 1, 1, 32),))
    peers = tuple(InputView(-3 - k, f"peers_{k}", "Array<1,Array<4,Value<16>>>", (Port(f"peers_{k}", 0, 4, 4, 16),)) for k in range(3))
    calls = (
        _call(0, "Gemm_v1", f"{ATTN}.qkv_proj/linear[0]", [R(-2, "ids"), R(-1, QKV_W)]),
        _call(1, "AllGather2_v1", f"{ATTN}.all_gather2_5/all_gather2_5[0]/all_gather", [R(0, "out"), R(-3, "peers_0")], n=8),
        _call(2, "AllGather2_v1", f"{ATTN}.all_gather2_6/all_gather2_6[0]/all_gather", [R(0, "out"), R(-4, "peers_1")], n=8),
        _call(3, "RmsNorm_v1", f"{ATTN}.q_norm/rms_norm/out[0]", [R(1, "out")], n=8),
        _call(4, "RmsNorm_v1", f"{ATTN}.k_norm/rms_norm/out[0]", [R(2, "out")], n=8),
        _call(5, "Add_v1", f"{ATTN}.attn/attn[0]", [R(3, "out"), R(4, "out")]),
        _call(6, "Gemm_v1", f"{ATTN}.o_proj/linear[0]", [R(5, "out"), R(-1, O_W)]),
        _call(7, "AllReduce2_v1", f"{ATTN}.o_proj/all_reduce2_7[0]/all_reduce", [R(6, "out"), R(-5, "peers_2")]),
        _call(8, "ArgmaxTokenSelect_v1", "argmax[0]", [R(7, "out")], n=1),
    )
    return ProgramView(calls, (weights, ids) + peers, (R(8, "out"),), "synthetic", "synthetic")


def _tp_record(P: ProgramView, rank: int) -> dict:
    """The record the frontend emits for `tp_attention_program()`: a collective's occurrence carries `collective`, its Call the kind as
    `member`, and a site's owning module is the module whose forward called it (`self_attn`), not the fallback's site path."""
    occs, calls = [], []
    for c in P.calls:
        n = parse_call_name(c.name)
        kind = n.suffix or None
        module = ATTN if n.module.startswith(f"{ATTN}.all_gather2_") else n.module
        coll = None
        if kind:
            k = len([o for o in occs if o["collective"]])
            coll = {"kind": kind, "occurrence": k, "operand_ranks": [rank, 1 - rank], "own_operand": 0, "peer_operand": 1, "peer_input": f"peers_{k}"}
        occs.append({"node": n.node, "module": module, "module_class": None, "op": "aten.op", "rule": None, "outputs": [{"shape": [4], "dtype": "bf16"}],
                     "written": [], "inputs": [], "lifetime": {}, "collective": coll})
        ret = [{"port": 0, "output": 0, "leaves": [0, 4]}]
        if n.hint:
            calls.append({"occ": len(occs) - 1, "member": None, "row": None, "hint": c.name, "returns": ret, "args": []})
        else:
            calls.append({"occ": len(occs) - 1, "member": kind, "row": int(n.row), "hint": None, "returns": ret, "args": []})
    return {"schema": RC.SCHEMA, "body": "program_fn", "inputs": ["ids"], "weights": None, "rank": {"world": 2, "rank": rank},
            "occurrences": occs, "calls": calls}


def test_a_rank_programs_collective_sites_are_the_same_identities_under_the_record_and_the_fallback():
    """Row #70 (OLMoE TP2, a Build with the record): the record owns the q / k gathers by `self_attn` and spells a collective's kind as
    its member; the manifest must not see either -- both gathers at (self_attn, 0 / 1) with their shards, o_proj's all-reduce at slot 0,
    identity for identity the fallback's."""
    P = tp_attention_program()
    merged = []
    for rank in (0, 1):
        canon, fallback = Correspondence(P, table=RC.RuntimeCorrespondence.from_json(_tp_record(P, rank))), Correspondence(P)
        assert canon.module_of(1) == ATTN and fallback.module_of(1) == f"{ATTN}.all_gather2_5"
        assert canon.site_path(1) == fallback.site_path(1) == f"{ATTN}.all_gather2_5"
        assert canon.name(7) == fallback.name(7)
        res = dict(result_json(), program={"digest": f"d{rank}", "function": "f"}, tp={"world": 2, "rank": rank})
        ma = request_manifest(P, canon, res, request_id="r0", rank=rank, ranks=2).manifest
        mb = request_manifest(P, fallback, res, request_id="r0", rank=rank, ranks=2).manifest
        assert ma["complete"] and mb["complete"], (ma["unmodelled"], mb["unmodelled"])
        assert ma["identities"] == mb["identities"] and ma["manifest_digest"] == mb["manifest_digest"]
        assert ma["op_path_aliases"] == mb["op_path_aliases"] == {f"{ATTN}.all_gather2_5": [ATTN, 0], f"{ATTN}.all_gather2_6": [ATTN, 1]}
        rows = {(x["family"], x["op_path"], x["invocation"], x["output_member"], tuple(x["element_range"])) for x in ma["identities"]}
        assert {r[0:1] + r[2:] for r in rows if r[1] == ATTN} == {
            ("instance_outputs", 0, "out", (0, 8)), ("instance_outputs", 1, "out", (0, 8)),
            ("tp_rank_partials", 0, f"shard_rank{rank}", (0, 4)), ("tp_rank_partials", 1, f"shard_rank{rank}", (0, 4))}
        assert ("instance_outputs", f"{ATTN}.o_proj", 0, "0", (0, 4)) in rows
        merged.append(ma)
    g = merge_ranks(merged)
    assert g["complete"] and g["tp_peer_binding"]["n_unbound"] == 0 and g["tp_peer_binding"]["bound"] == 6


def test_a_record_of_another_program_is_refused():
    P = program()
    doc = _record(P)
    doc["calls"] = doc["calls"][:-1]
    with pytest.raises(ValueError, match="another Program"):
        Correspondence(P, table=RC.RuntimeCorrespondence.from_json(doc))


def test_of_reads_the_build_dir_and_picks(tmp_path):
    P = program()
    doc = _record(P)
    new = tmp_path / "build_request"
    new.mkdir()
    with gzip.open(new / "descriptor.json.gz", "wt") as f:
        json.dump({"annotations": {"correspondence": doc, "node_hints": {}}}, f)
    (new / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": "p" * 64, "correspondence_digest": RC.correspondence_digest(doc)}}))
    old = tmp_path / "build_request_v1"
    old.mkdir()
    (old / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": "p" * 64}}))
    with gzip.open(old / "descriptor.json.gz", "wt") as f:
        json.dump({"annotations": {"node_hints": {}}}, f)
    a, b, c = Correspondence.of(P, str(new)), Correspondence.of(P, str(old)), Correspondence.of(P, None)
    assert a.source == "correspondence" and a.digest == RC.correspondence_digest(doc) and a.problem is None
    assert b.source == "v1_annotations" and c.source == "v1_annotations"
    (new / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": "p" * 64, "correspondence_digest": "0" * 64}}))
    d = Correspondence.of(P, str(new))
    assert d.source == "correspondence" and d.problem and "!=" in d.problem and d.provenance()["problem"] == d.problem
