"""`query.manifest.verify.verify`: the record of a manifest of record, and what it says of one that is not, cause by cause."""
from __future__ import annotations

import pytest

from verity_vllm.query.manifest import verify as V
from verity_vllm.query.manifest.format import manifest_digest

PROGRAM = "ab" * 64
OTHER = "cd" * 64


def _row(invocation):
    return {"family": "aten", "program_digest": PROGRAM, "request_id": "r0", "step": 0, "invocation": invocation,
            "op_path": "model.layers.0.mlp", "output_member": "out", "element_range": [0, 8], "rank": 0}


ROWS = [_row(0), _row(1)]
DIGEST = manifest_digest(ROWS)
ONE_ROW = manifest_digest(ROWS[:1])


def _manifest(**changed):
    return {"identities": ROWS, "manifest_digest": DIGEST, "program_digest": PROGRAM, "complete": True,
            "populations": {"identities": len(ROWS)}, "query": {"engine": "v2-query"}, **changed}


def test_the_manifest_of_record_verifies_and_its_record_says_no_why():
    assert V.verify(_manifest(), _manifest(), manifest_path="commit/manifest.json") == {
        "schema": "verity-sweep/manifest-verify/v1", "manifest": "commit/manifest.json", "used_digest": DIGEST, "used_rows_digest": DIGEST,
        "rebuilt_digest": DIGEST, "used_program_digest": PROGRAM, "rebuilt_program_digest": PROGRAM, "used_identities": 2,
        "rebuilt_identities": 2, "rebuilt_complete": True, "used_query": "v2-query", "rebuilt_engine": "v2-query", "ok": True}


FULL = "pass the FULL digest (workload_program.json), not a prefix"
CAUSES = [
    ("rows taken out after the digest", {"identities": ROWS[:1]}, {},
     (f"manifest_digest {DIGEST[:16]} != digest of its rows {ONE_ROW[:16]} (edited after digesting); "
      f"rows digest {ONE_ROW[:16]} != the Program x workload's manifest {DIGEST[:16]} (1 vs 2 identities): not the manifest of record")),
    ("a digest that is honest, of fewer rows than the Program requires", {"identities": ROWS[:1], "manifest_digest": ONE_ROW}, {},
     f"rows digest {ONE_ROW[:16]} != the Program x workload's manifest {DIGEST[:16]} (1 vs 2 identities): not the manifest of record"),
    ("a digest of 32 hex digits", {"manifest_digest": "e" * 32}, {},
     f"manifest_digest {'e' * 16} != digest of its rows {DIGEST[:16]} (edited after digesting)"),
    ("a digest of 64 hex digits", {"manifest_digest": "e" * 64}, {},
     (f"manifest_digest {'e' * 16} != digest of its rows {DIGEST[:16]} "
      "(a 64-hex SHA-256 digest from before manifest digests became SHA-512: rebuild the manifest)")),
    ("a prefix of the Program's digest", {"program_digest": PROGRAM[:16]}, {},
     f"program_digest {PROGRAM[:16]} (16 hex) != the rebuilt {PROGRAM} (128 hex): {FULL}"),
    ("a digest that agrees in its first 16 digits", {"program_digest": PROGRAM[:16] + "0" * 112}, {},
     f"program_digest {PROGRAM[:16] + '0' * 112} (128 hex) != the rebuilt {PROGRAM} (128 hex): {FULL}"),
    ("another Program", {"program_digest": OTHER}, {}, f"program_digest {OTHER[:16]} != the Build's {PROGRAM[:16]}"),
    ("another Program's digest at half its width", {"program_digest": OTHER[:64]}, {},
     f"program_digest {OTHER[:64]} (64 hex) != the rebuilt {PROGRAM} (128 hex): {FULL}"),
    ("a rebuilt manifest that is not complete", {}, {"complete": False, "unmodelled": {"Softmax_v1 under model.layers.0": 2}},
     "the Program x workload manifest is incomplete: {'Softmax_v1 under model.layers.0': 2}"),
    ("a rebuilt manifest that does not say it is complete", {}, {"complete": None}, "the Program x workload manifest is incomplete: None"),
    ("a rebuilt manifest whose complete is 1 and not true", {}, {"complete": 1}, "the Program x workload manifest is incomplete: None"),
]


@pytest.mark.parametrize("used,rebuilt,why", [c[1:] for c in CAUSES], ids=[c[0] for c in CAUSES])
def test_a_manifest_that_is_not_the_manifest_of_record_fails_with_each_cause_named(used, rebuilt, why):
    rec = V.verify(_manifest(**used), _manifest(**rebuilt))
    assert rec["ok"] is False and rec["why"] == why and rec["manifest"] is None


def test_a_used_manifest_with_no_rows_and_no_query_digests_as_an_empty_one_and_fails():
    rec = V.verify({}, _manifest())
    assert rec["ok"] is False and rec["used_identities"] == 0 and rec["used_rows_digest"] == manifest_digest([])
    assert rec["used_digest"] is None and rec["used_program_digest"] is None and rec["used_query"] is None
    assert V.verify(_manifest(query="v2-query"), _manifest())["used_query"] is None
