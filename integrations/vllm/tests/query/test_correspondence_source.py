"""The query reads the owning module off ONE reader -- the `runtime-correspondence/v1` record -- and falls back to the v1-annotation
names only when a Build carries none; the source is recorded, and on the same Program both give the same partition and manifest."""
from __future__ import annotations

import gzip
import json

import pytest
from verity.verification.query import boundary
from verity_vllm import runtime_correspondence as RC
from verity_vllm.query.correspondence import CallName, Correspondence, call_name_of, parse_call_name
from verity_vllm.query.module_body import Q_module_body_v1
from verity_vllm.query.v1_bridge import request_manifest

from .test_module_body import W, program


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
