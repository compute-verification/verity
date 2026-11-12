"""`read_descriptor_annotation`: the `annotations[namespace]` of a descriptor file read WITHOUT `json.load` of the
descriptor -- its `definitions` (the Program body) streamed through and dropped -- is exactly what `json.load` gives, at every chunk
size (values cut anywhere: inside a node, a string, a number), and `read_program_dir` is unchanged in what it returns."""
from __future__ import annotations

import gzip
import json

import pytest

from verity_vllm.correspondence import runtime as RC


def descriptor(n_nodes: int = 60) -> dict:
    nodes = [{"fn": f"Gemm_v1{{K={k},N=8}}", "form": "call", "out": "Array<8,Value<16>>", "name": f"m.l{k}/linear[{k}]",
              "args": [[["n", j, 0, 12345678 + j] for j in range(k)], [["p", 1, 0, 64]]], "flag": k % 2 == 0, "f": 1.5e-7 * k, "none": None}
             for k in range(n_nodes)]
    corr = {"schema": RC.SCHEMA, "body": "program_fn", "inputs": ["ids"], "weights": ["m.w"], "rank": None,
            "occurrences": [{"node": f"linear_{k}", "module": f"m.l{k}", "module_class": "Linear", "op": "aten.mm", "rule": None,
                             "outputs": [{"shape": [1, 8], "dtype": "bf16"}], "written": [], "inputs": [], "lifetime": {"in_place": False}, "collective": None}
                            for k in range(n_nodes)],
            "calls": [{"occ": k, "member": None, "row": k, "hint": None, "returns": [{"port": 0, "output": 0, "leaves": [0, 8]}], "args": [{"input": None}]}
                      for k in range(n_nodes)]}
    return {"schema": "verity-ir/program/v1", "root": "root_v1", "program_fn": "pf_v1{}",
            "definitions": {"pf_v1{}": {"kind": "composite", "params": [["ids", "Value<32>"], ["weights", "{m.w: Array<8,Array<8,Value<16>>>}"]],
                                        "body": {"nodes": nodes, "ret": [["n", n_nodes - 1, 0, 8]]}, "empty_list": [], "empty_obj": {}},
                            "Gemm_v1{K=1,N=8}": {"kind": "primitive", "statics": {"K": 1, "N": 8}, "text": "a \"quoted\" string, with ] and } inside"}},
            "annotations": {"node_hints": {str(k): f"m.l{k}/linear[{k}]" for k in range(n_nodes)}, RC.NAMESPACE: corr, "other": [1, 2.5, "x", None, True, {"a": [[]]}]},
            "tables": {"strings": ["s" * 40 for _ in range(30)]}}


@pytest.mark.parametrize("chunk", [7, 64, 1000, 1 << 22])
@pytest.mark.parametrize("gz", [False, True])
def test_streamed_annotation_equals_json_load(tmp_path, monkeypatch, chunk, gz):
    desc = descriptor()
    p = tmp_path / ("descriptor.json.gz" if gz else "descriptor.json")
    text = json.dumps(desc, indent=1 if chunk == 64 else None)
    if gz:
        with gzip.open(p, "wt") as f:
            f.write(text)
    else:
        p.write_text(text)
    monkeypatch.setattr(RC, "_CHUNK", chunk)
    got = RC.read_descriptor_annotation(str(p))
    assert got == desc["annotations"][RC.NAMESPACE]
    assert RC.correspondence_digest(got) == RC.correspondence_digest(desc)
    assert RC.read_descriptor_annotation(str(p), "node_hints") == desc["annotations"]["node_hints"]
    assert RC.read_descriptor_annotation(str(p), "other") == desc["annotations"]["other"]
    assert RC.read_descriptor_annotation(str(p), "absent") is None


def test_descriptor_without_annotations(tmp_path):
    p = tmp_path / "descriptor.json"
    p.write_text(json.dumps({"schema": "x", "definitions": {"a": {"body": {"nodes": [1, 2, [3, [4]]]}}}}))
    assert RC.read_descriptor_annotation(str(p)) is None
    p.write_text(json.dumps({"schema": "x", "annotations": None}))
    assert RC.read_descriptor_annotation(str(p)) is None


def test_read_program_dir_through_the_stream(tmp_path):
    desc = descriptor(12)
    with gzip.open(tmp_path / "descriptor.json.gz", "wt") as f:
        json.dump(desc, f)
    digest = RC.correspondence_digest(desc)
    (tmp_path / "result.json").write_text(json.dumps({"program": {"digest": "d", "correspondence_digest": digest}}))
    table, dg, problem = RC.read_program_dir(str(tmp_path))
    assert problem is None and dg == digest and len(table) == 12 and table.owning_module(3) == "m.l3"
    assert table.doc == desc["annotations"][RC.NAMESPACE]
    (tmp_path / "result.json").write_text(json.dumps({"program": {"digest": "d", "correspondence_digest": "0" * 64}}))
    table, dg, problem = RC.read_program_dir(str(tmp_path))
    assert table is not None and dg == digest and problem and "correspondence_digest" in problem
    (tmp_path / "result.json").write_text(json.dumps({"program": {"digest": "d"}}))
    assert RC.read_program_dir(str(tmp_path)) == (None, None, None)


def test_the_plain_descriptor_is_read_where_no_gzipped_one_is(tmp_path):
    desc = descriptor(3)
    doc, digest = desc["annotations"][RC.NAMESPACE], RC.correspondence_digest(desc)
    (tmp_path / "result.json").write_text(json.dumps({"program": {"digest": "d", "correspondence_digest": digest}}))
    (tmp_path / "descriptor.json").write_text(json.dumps(desc))
    assert RC.read_program_document(str(tmp_path)) == (doc, digest, None)
    table, dg, problem = RC.read_program_dir(str(tmp_path))
    assert problem is None and dg == digest and table is not None and table.doc == doc
    other = descriptor(5)                                                          # beside a gzipped descriptor the plain one is not read
    with gzip.open(tmp_path / "descriptor.json.gz", "wt") as f:
        json.dump(other, f)
    got, dg, problem = RC.read_program_document(str(tmp_path))
    assert got == other["annotations"][RC.NAMESPACE] and dg == RC.correspondence_digest(other) != digest
    assert problem and "correspondence_digest" in problem                          # the result records the plain one's digest


def test_a_program_dir_with_no_descriptor_or_no_record_in_it(tmp_path):
    assert RC.read_program_document(str(tmp_path)) == (None, None, None)           # nothing beside it: no record, and no problem
    assert RC.read_program_dir(str(tmp_path)) == (None, None, None)
    bare = tmp_path / "bare"                                                       # a descriptor that carries no record, and no result
    bare.mkdir()
    (bare / "descriptor.json").write_text(json.dumps({"schema": "x", "annotations": {"node_hints": {}}}))
    assert RC.read_program_document(str(bare)) == (None, None, None)
    # a result that records the record's digest is a problem where no descriptor is beside it, and where the descriptor carries no record
    recorded = json.dumps({"program": {"digest": "d", "correspondence_digest": "0" * 64}})
    (tmp_path / "result.json").write_text(recorded)
    problem = f"{tmp_path}: result.json records a correspondence_digest but no descriptor is beside it"
    assert RC.read_program_document(str(tmp_path)) == (None, None, problem)
    assert RC.read_program_dir(str(tmp_path)) == (None, None, problem)
    (bare / "result.json").write_text(recorded)
    doc, digest, problem = RC.read_program_document(str(bare))
    assert doc is None and digest is None
    assert problem == f"{bare}: result.json records a correspondence_digest but the descriptor carries no {RC.NAMESPACE!r} annotation"
