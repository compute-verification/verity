"""`query_artifact`: the four identities are stable and independent (change the query -> `query_id` changes,
`program_digest` does not; change the Program -> the reverse; `binding_id` changes with either), the query document
carries no use configuration, the use document has the schema G8 reads, and `program_digest` is the codec's own."""
from __future__ import annotations

import hashlib
import json

import pytest

from verity.ml.circuits import codec
from verity_vllm.query import query_artifact as QA
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import bind
from verity.ml.circuits.partition import PartitionVerdict, ReadinessReport, BlockVerdict, WidthVerdict
from verity.ml.circuits.program import Program
from verity.ml.definitions.registry import b1

Q1 = {"op": "override", "base": {"op": "default", "w": 32},
      "cases": [{"path": "//Gemm_v1{*}", "family": {"op": "members", "path": "//Gemm_v1{K=*,N=*}", "sel": "@batch:GemmCoordinate_v1", "k": 1}}]}
Q2 = {"op": "override", "base": {"op": "default", "w": 16}, "cases": Q1["cases"]}


@pytest.fixture(scope="module")
def gemm():
    return Program(bind(b1.Gemm, K=32, N=6))


@pytest.fixture(scope="module")
def gemm8():
    return Program(bind(b1.Gemm, K=32, N=8))


def verdict(status: str = "ok") -> BlockVerdict:
    return BlockVerdict(PartitionVerdict("exact", (), {"path": "tiling"}), WidthVerdict(32, 16, (), (), "ok"), 230,
                     ReadinessReport(32, True, (), 32, 3, 0), status)


def test_program_digest_is_the_codecs_untouched():
    assert QA.program_digest is codec.program_digest


def test_query_id_is_sha512_of_canonical_json_and_stable():
    qid = QA.query_id(Q1)
    assert qid == hashlib.sha512(canonical_json(Q1)).hexdigest()
    assert QA.query_id(json.loads(json.dumps(Q1))) == qid                         # key order / round trip irrelevant
    assert QA.query_id({"cases": Q1["cases"], "op": "override", "base": Q1["base"]}) == qid
    assert QA.query_id(Q2) != qid


def test_query_id_accepts_objects_with_to_json():
    class Q:
        def to_json(self):
            return Q1
    assert QA.query_id(Q()) == QA.query_id(Q1)
    with pytest.raises(TypeError):
        QA.query_id(object())


def test_identities_are_independent(gemm, gemm8):
    d1, d8 = QA.program_digest(gemm), QA.program_digest(gemm8)
    assert d1 != d8
    q1, q2 = QA.query_id(Q1), QA.query_id(Q2)
    # change the query: query_id and binding_id change, program_digest does not
    doc_a, doc_b = QA.query_document(Q1, gemm), QA.query_document(Q2, gemm)
    assert doc_a["query_id"] == q1 and doc_b["query_id"] == q2
    assert doc_a["program_digest"] == doc_b["program_digest"] == d1
    assert doc_a["binding_id"] != doc_b["binding_id"]
    # change the Program: program_digest and binding_id change, query_id does not
    doc_c = QA.query_document(Q1, gemm8)
    assert doc_c["query_id"] == q1 and doc_c["program_digest"] == d8 and doc_c["binding_id"] != doc_a["binding_id"]
    # the digest of the Program is untouched by producing artifacts
    assert QA.program_digest(gemm) == d1


def test_binding_id_construction_and_stability(gemm):
    q1, d1 = QA.query_id(Q1), QA.program_digest(gemm)
    expected = hashlib.sha512(("verity-ir/query/v1" + q1 + d1).encode()).hexdigest()
    assert QA.binding_id(q1, d1) == expected
    assert QA.query_document(Q1, gemm)["binding_id"] == expected
    assert QA.query_document(Q1, d1)["binding_id"] == expected                      # a digest string works too
    assert QA.binding_id(q1, d1, schema="verity-ir/query/v2") != expected
    with pytest.raises(ValueError):
        QA.binding_id("abc", d1)


def test_query_document_carries_no_use_configuration(gemm):
    doc = QA.query_document(Q1, gemm, provenance={"author": "test"})
    assert doc["schema"] == "verity-ir/query/v1"
    assert set(doc) == {"schema", "query_id", "program_digest", "binding_id", "query", "provenance"}
    assert "kind" not in doc and "w_max" not in doc and "purpose" not in doc
    assert doc["query"] == Q1
    assert QA.check_query_document(doc) == []
    bad = dict(doc, kind="vu-partition")
    assert any("use configuration" in p for p in QA.check_query_document(bad))
    tampered = dict(doc, query=Q2)
    assert any("query_id" in p for p in QA.check_query_document(tampered))
    rebound = dict(doc, program_digest="0" * 128)
    assert any("binding_id" in p for p in QA.check_query_document(rebound))


def test_use_document_schema(gemm):
    qdoc, udoc = QA.vu_use_document(Q1, gemm, verdict(), w_max=32, utc="2026-09-17T22:00:00Z")
    assert udoc["schema"] == "verity-ir/query-use/v1"
    assert set(udoc) == {"schema", "binding_id", "use", "verdict", "evaluator"}
    assert udoc["binding_id"] == qdoc["binding_id"]
    assert udoc["use"] == {"kind": "vu-partition", "w_max": 32}
    assert udoc["verdict"]["status"] == "ok" and udoc["verdict"]["count"] == 230
    assert udoc["verdict"]["partition"]["status"] == "exact" and udoc["verdict"]["width"]["w_max"] == 32
    assert udoc["verdict"]["readiness_32"]["ok"] is True
    assert udoc["evaluator"] == {"version": QA.EVALUATOR_VERSION, "utc": "2026-09-17T22:00:00Z"}
    json.dumps(udoc)
    assert QA.check_use_document(udoc, QA.program_digest(gemm), qdoc) == []
    assert QA.check_use_document(udoc, "0" * 128, qdoc) != []                         # bound to another Program
    assert QA.check_use_document(dict(udoc, binding_id="1" * 128), None, qdoc) != []
    assert QA.check_use_document({"schema": "verity-ir/query-use/v1"}) != []


def test_use_document_from_plain_dict_verdict_and_default_utc():
    udoc = QA.use_document("a" * 128, {"status": "fail"}, w_max=16)
    assert udoc["verdict"] == {"status": "fail"} and udoc["use"]["w_max"] == 16
    assert udoc["evaluator"]["utc"].endswith("Z") and len(udoc["evaluator"]["utc"]) == 20
    with pytest.raises(ValueError):
        QA.use_document("nope", verdict())


def test_same_query_over_two_programs_shares_query_id(gemm, gemm8):
    """A canonical VU query is reused across Programs of one profile: one query_id, many binding_ids."""
    a, b = QA.query_document(Q1, gemm), QA.query_document(Q1, gemm8)
    assert a["query_id"] == b["query_id"] and a["binding_id"] != b["binding_id"]


def test_query_document_round_trips_through_json(gemm):
    doc = QA.query_document(Q1, gemm)
    back = json.loads(json.dumps(doc))
    assert back == doc and QA.check_query_document(back) == []
