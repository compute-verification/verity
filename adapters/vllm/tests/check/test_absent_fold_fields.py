"""What the fold's comparisons make of a field a document leaves out.

`compare_tokens` reads the accepted tokens of each request from the capture's tokens file, and `compare_structure` the
extra nodes from the matcher's report, which lists them only for a superset.  Each test leaves the field out (absent,
null, empty, or any other value that tests false: `left_out`) and states what the comparison then reports.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from tests.check.test_absent_commit_fields import put
from tests.check.test_absent_link_fields import left_out
from tests.observe.test_fold_synthetic import (
    BN,
    REQ,
    SAMPLED,
    C,
    _real_memory_available,
    install_stub_memory,
    toy_log,
    toy_profile,
)
from verity_vllm.check import fold_compare as cmp
from verity_vllm.observe.fold.fold import FoldResult, fold

TOKENS = [int(t) for t in SAMPLED]


@pytest.fixture(scope="module")
def folded() -> FoldResult:
    if not _real_memory_available():
        install_stub_memory()
    return fold(toy_log(), toy_profile(), strict=True)


def tokens_report(tmp_path, folded: FoldResult, requests: Any) -> dict[str, Any]:
    """`compare_tokens` over a tokens file that gives the sampled tokens and, under `requests`, what the test states."""
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps(put({"sampled": TOKENS, "requests": None}, ("requests",), requests)))
    return cmp.compare_tokens(folded, path, None)


def accepted_mismatches(report: dict[str, Any]) -> list[str]:
    return [m for m in report["mismatches"] if m.startswith("accepted tokens")]


@left_out([])
def test_a_tokens_file_that_names_no_request_records_no_accepted_tokens(tmp_path, folded, stated):
    report = tokens_report(tmp_path, folded, stated)
    assert report["fold"] == {REQ: TOKENS}
    assert report["accepted"] is None and accepted_mismatches(report) == []


@left_out([])
def test_a_request_that_states_no_accepted_token_records_none(tmp_path, folded, stated):
    request = put({"request_id": REQ, "accepted_record_token_ids": TOKENS}, ("accepted_record_token_ids",), stated)
    report = tokens_report(tmp_path, folded, [request])
    assert report["accepted"] is None and accepted_mismatches(report) == [] and report["status"] == "pass"


def test_the_accepted_tokens_of_a_request_are_held_against_the_folds(tmp_path, folded):
    same = tokens_report(tmp_path, folded, [{"request_id": REQ, "accepted_record_token_ids": TOKENS}, 5])
    assert same["accepted"] == {REQ: TOKENS} and same["mismatches"] == [] and same["status"] == "pass"
    other = tokens_report(tmp_path, folded, [{"request_id": REQ, "accepted_record_token_ids": [1, 2]}])
    assert other["accepted"] == {REQ: [1, 2]} and other["status"] == "fail"
    assert other["mismatches"] == [f"accepted tokens for {REQ!r} differ: [1, 2] vs fold {TOKENS}"]


def test_a_tokens_file_that_is_not_there_gives_no_accepted_tokens(tmp_path, folded):
    report = cmp.compare_tokens(folded, tmp_path / "tokens.json", None)
    assert report["observer"] is None and "accepted" not in report and report["status"] == "pass"


def structure_report(monkeypatch, folded: FoldResult, matcher) -> dict[str, Any]:
    monkeypatch.setattr(cmp, "match_program", matcher)
    report = cmp.compare_structure(folded, C=C, LP=2, STEPS=1, BN=BN)
    assert sorted(report) == ["classes", "extra_by_spec", "family", "histogram", "matched", "n_derived_nodes", "n_extra", "n_family_leaves",
                              "reason", "seconds", "status", "superset", "unclassified"]
    return report


@left_out([])
@pytest.mark.parametrize(("stated_by_the_matcher", "status"), [({"matched": True}, "pass"), ({"matched": False, "superset": True}, "pass"),
                                                             ({"matched": False, "reason": "node 3 differs"}, "fail")])
def test_a_report_that_lists_no_extra_node_counts_none(monkeypatch, folded, stated, stated_by_the_matcher, status):
    matched = put(dict(stated_by_the_matcher, extra_nodes=None), ("extra_nodes",), stated)
    report = structure_report(monkeypatch, folded, lambda *a, **kw: dict(matched))
    assert (report["n_extra"], report["extra_by_spec"], report["classes"], report["unclassified"]) == (0, {}, {}, {})
    assert report["status"] == status and report["reason"] == stated_by_the_matcher.get("reason")
    assert (report["matched"], report["superset"]) == (bool(stated_by_the_matcher.get("matched")), bool(stated_by_the_matcher.get("superset")))


def test_a_matcher_that_raises_fails_the_comparison_and_counts_no_extra_node(monkeypatch, folded):
    def raises(*a, **kw):
        raise ValueError("no such family")

    report = structure_report(monkeypatch, folded, raises)
    assert report["status"] == "fail" and report["reason"] == "ValueError: no such family"
    assert (report["matched"], report["superset"], report["n_extra"], report["extra_by_spec"], report["classes"]) == (False, False, 0, {}, {})


def test_the_extra_nodes_of_a_superset_are_counted_by_their_spec(folded):
    report = cmp.compare_structure(folded, C=C, LP=2, STEPS=1, BN=BN)
    assert (report["status"], report["matched"], report["superset"], report["n_extra"]) == ("pass", False, True, 1)
    assert list(report["extra_by_spec"].values()) == [1] and next(iter(report["extra_by_spec"])).startswith("RMSNormFusedCuda_v2{")
    assert len(report["classes"]) == 1 and report["unclassified"] == {}
