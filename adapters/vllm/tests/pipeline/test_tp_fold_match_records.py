"""`verity_vllm.pipeline.tp.fold_match` on documents that hold less than a finished stage's: a fold that left no summary or a
summary without its statistics, a compare or a GM-01 that wrote no document, a decomposition whose requests state no totals, a
rank with no observer log (so no compare and no GM-01), a cross-rank check that stopped before it listed what failed, and the
inputs that check was not given.  No stage's subprocess runs here: `_run` only returns a code, and `main` runs on stand-in stages."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.check.match import rank_match as RM
from verity_vllm.pipeline import match as PM
from verity_vllm.pipeline.tp import fold_match as FM

LINKS = {"schema": "cb-a/tp_links/v2", "world": 2, "rank": 1, "links": [{"k": 0, "module": "m", "peers": []}]}
SUMMARY: dict[str, Any] = {"errors": 0, "unresolved": 0, "n_instances": 9, "n_steps": 3, "error_examples": [{"e": 0}, {"e": 1}, {"e": 2}, {"e": 3}],
                           "fold": {"tp": {"rank": 1, "world": 2}, "collectives": [{"step": 0, "k": 0}, {"step": 0, "k": 1}], "collectives_by_kind": {"all_reduce": 2}}}


def _exits(monkeypatch, rc: int) -> None:
    monkeypatch.setattr(FM, "_run", lambda cmd, log, timeout=None: rc)


def _fold(tmp_path: Path, monkeypatch, summary: dict | None, rc: int = 0) -> dict:
    out = tmp_path / "rank1"
    out.mkdir()
    for name in ("program.json", "instances.jsonl"):
        (out / name).write_text("")
    if summary is not None:
        (out / "fold_summary.json").write_text(json.dumps(summary))
    _exits(monkeypatch, rc)
    return FM.fold_rank("py", tmp_path / "log.rank1.jsonl.gz", "profile", out, None)


# the fold

def test_a_fold_reports_its_summarys_rank_collectives_and_first_three_error_examples(tmp_path, monkeypatch):
    fold = _fold(tmp_path, monkeypatch, SUMMARY)
    assert fold["ok"] is True and fold["tp"] == {"rank": 1, "world": 2} and fold["collectives"] == 2
    assert fold["collectives_by_kind"] == {"all_reduce": 2} and fold["error_examples"] == [{"e": 0}, {"e": 1}, {"e": 2}]


@pytest.mark.parametrize("summary", [None, {}, {"fold": None}, {"fold": {}}, {"fold": {"collectives": None}, "error_examples": None}],
                         ids=["no summary", "an empty summary", "null statistics", "empty statistics", "a null list of collectives"])
def test_a_fold_whose_summary_holds_no_statistics_states_no_rank_and_no_collective(tmp_path, monkeypatch, summary):
    fold = _fold(tmp_path, monkeypatch, summary)
    assert (fold["tp"], fold["collectives"], fold["collectives_by_kind"], fold["error_examples"]) == (None, 0, None, [])
    assert fold["ok"] is True and fold["errors"] is None


def test_a_fold_that_failed_before_its_summary_is_not_ok(tmp_path, monkeypatch):
    fold = _fold(tmp_path, monkeypatch, None, rc=1)
    assert (fold["ok"], fold["rc"], fold["tp"], fold["collectives"], fold["error_examples"]) == (False, 1, None, 0, [])


# the compare

def _compare(tmp_path: Path, monkeypatch, rc: int, derived: list[str], **documents: dict) -> dict:
    for name, doc in documents.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(doc))
    _exits(monkeypatch, rc)
    return FM.compare_rank("py", tmp_path, tmp_path / "build", derived, "dag", None)


def test_a_decomposition_that_wrote_no_document_fails_by_its_code_and_counts_nothing(tmp_path, monkeypatch):
    got = _compare(tmp_path, monkeypatch, 3, ["d1"])
    assert got["verdict"] == "FAIL batch_decomp None rc=3: "
    assert (got["requests"], got["canonical_equal_all"], got["operands_different"], got["unattributed"], got["per_request"]) == (0, False, 0, None, [])


def test_a_decomposition_adds_the_differing_operands_of_the_requests_that_state_their_totals(tmp_path, monkeypatch):
    requests = [{"request_id": "q0", "derived": "d0", "canonical_equal": True, "totals": {"operands_different": 2}},
                {"request_id": "q1", "derived": "d1", "canonical_equal": True}, {"request_id": "q2", "canonical_equal": True, "totals": None}]
    got = _compare(tmp_path, monkeypatch, 3, ["d1"], match_decomp={"verdict": "FAIL", "reasons": ["a", "b"], "requests": requests, "attribution": {"unattributed": 4}})
    assert got["verdict"] == "FAIL batch_decomp FAIL rc=3: a; b"
    assert (got["requests"], got["canonical_equal_all"], got["operands_different"], got["unattributed"]) == (3, True, 2, 4)
    assert [r["derived"] for r in got["per_request"]] == ["d0", "d1", None]
    passed = _compare(tmp_path, monkeypatch, 0, ["d1"], match_decomp={"verdict": "PASS", "requests": requests[1:]})
    assert passed["verdict"] == "PASS" and passed["unattributed"] is None and passed["operands_different"] == 0


def test_a_compare_that_wrote_no_document_fails_with_nothing_compared(tmp_path, monkeypatch):
    got = _compare(tmp_path, monkeypatch, 1, [])
    assert got["verdict"] == "FAIL canonical_equal=None dag_multiset_equal=False steps=None/None operands_different=None rc=1"
    assert (got["steps"], got["totals"], got["histogram_diff_T_normalised"], got["first_diff"], got["dag_multiset"]) == ([None, None], {}, 0, None, None)


def test_a_compare_reads_the_step_comparison_under_its_documents_compare_block(tmp_path, monkeypatch):
    block = {"totals": {"operands_different": 0}, "a": {"steps": 3}, "b": {"steps": 3}, "canonical_sha256": {"equal": True}, "diffs": [{"at": 1}]}
    got = _compare(tmp_path, monkeypatch, 0, [], match_compare={"compare": block, "histogram_diff_T_normalised": {"Gemm": 1, "Add": 2}})
    assert got["verdict"] == "PASS" and got["criterion"] == "sequence" and got["steps"] == [3, 3] and got["canonical_equal"] is True
    assert got["histogram_diff_T_normalised"] == 2 and got["first_diff"] == {"at": 1} and got["totals"] == {"operands_different": 0}
    fewer = _compare(tmp_path, monkeypatch, 0, [], match_compare={"compare": dict(block, b={"steps": 2}, totals=None)})
    assert fewer["verdict"] == "FAIL canonical_equal=True dag_multiset_equal=False steps=3/2 operands_different=None rc=0"


# GM-01

@pytest.mark.parametrize("rc, verdict", [(2, "rc=2"), (124, "TIMEOUT")])
def test_a_global_match_that_wrote_no_document_is_named_by_its_code_and_lists_no_check(tmp_path, monkeypatch, rc, verdict):
    _exits(monkeypatch, rc)
    got = FM.global_match_rank("py", tmp_path, tmp_path / "gp.json", tmp_path / "wl.json", "dag", None)
    assert (got["verdict"], got["ok"], got["checks"], got["generated_lengths"], got["failed_checks"]) == (verdict, False, {}, {}, None)


@pytest.mark.parametrize("lists", [{}, {"checks": None, "requests": None}, {"checks": [], "requests": []}], ids=["no lists", "null lists", "empty lists"])
def test_a_global_match_document_gives_its_checks_by_id_and_its_lengths_by_request(tmp_path, monkeypatch, lists):
    _exits(monkeypatch, 0)
    (tmp_path / "global_match.json").write_text(json.dumps({"verdict": "PASS", **lists}))
    got = FM.global_match_rank("py", tmp_path, tmp_path / "gp.json", tmp_path / "wl.json", "dag", None)
    assert (got["verdict"], got["ok"], got["checks"], got["generated_lengths"]) == ("PASS", True, {}, {})
    doc = {"verdict": "FAIL", "checks": [{"id": "GM-01a", "status": "PASS"}, {"id": "GM-01b", "status": "FAIL"}], "requests": [{"request_id": "q0", "generated_length": 2}]}
    (tmp_path / "global_match.json").write_text(json.dumps(doc))
    got = FM.global_match_rank("py", tmp_path, tmp_path / "gp.json", tmp_path / "wl.json", "dag", None)
    assert (got["ok"], got["checks"], got["generated_lengths"]) == (False, {"GM-01a": "PASS", "GM-01b": "FAIL"}, {"q0": 2})


# a rank's links at B>1

def _program(tmp_path: Path, name: str, links: bool = True) -> str:
    (tmp_path / name).mkdir()
    if links:
        (tmp_path / name / "tp_links.json").write_text(json.dumps(LINKS))
    return str(tmp_path / name)


def test_a_rank_with_further_programs_has_each_requests_links_from_its_programs_dir(tmp_path):
    compare = {"per_request": [{"request_id": "q0", "derived": _program(tmp_path, "d0")}, {"request_id": "q1", "derived": _program(tmp_path, "d1")}]}
    got = FM.links_for_rank(1, tmp_path / "build", compare, ["d1"])
    assert isinstance(got, dict) and sorted(got) == ["q0", "q1"] and [ln["k"] for ln in got["q1"]] == [0]


def test_a_request_with_no_program_dir_or_no_links_file_leaves_the_rank_without_links(tmp_path):
    with_links, without = _program(tmp_path, "d0"), _program(tmp_path, "d1", links=False)
    for requests in ([], [{"request_id": "q0", "derived": with_links}, {"request_id": "q1", "derived": None}],
                     [{"request_id": "q0", "derived": with_links}, {"request_id": "q1"}], [{"request_id": "q0", "derived": with_links}, {"request_id": "q1", "derived": without}]):
        assert FM.links_for_rank(1, tmp_path / "build", {"per_request": requests}, ["d1"]) is None


@pytest.mark.parametrize("compare", [{}, {"per_request": None}, {"per_request": []}], ids=["no list", "a null list", "an empty list"])
def test_a_compare_record_that_lists_no_request_has_no_links_for_the_rank(tmp_path, compare):
    assert FM.links_for_rank(1, tmp_path / "build", compare, ["d1"]) is None


@pytest.mark.parametrize("text", ["{", "[1]", '{"world": Infinity, "links": []}', '{"rank": -Infinity, "links": []}',
                                  '{"links": [{"k": 0, "peer_rank": Infinity}]}', '{"links": [{"k": Infinity}]}'],
                         ids=["not JSON", "a list", "an infinite world", "an infinite rank", "an infinite peer rank", "an infinite collective"])
def test_a_links_file_that_is_not_one_leaves_the_rank_without_links(tmp_path, text):
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "tp_links.json").write_text(text)
    assert FM.links_for_rank(1, tmp_path / "build", {}, []) is None


# the cross-rank check's record

def test_an_input_the_cross_rank_check_was_not_given_fails_it_by_name_beside_what_it_failed_itself():
    whole: dict[str, Any] = {"verdict": "PASS", "checks": [], "failed": []}
    FM._fails_too(whole, "recorder_missing")
    assert whole == {"verdict": "FAIL", "checks": [], "failed": ["recorder_missing"]}
    failed: dict[str, Any] = {"verdict": "FAIL", "failed": ["count", "peer"]}
    FM._fails_too(failed, "links_missing")
    FM._fails_too(failed, "provenance_missing")
    assert failed == {"verdict": "FAIL", "failed": ["count", "peer", "links_missing", "provenance_missing"]}


def test_a_cross_rank_check_that_stopped_before_listing_what_failed_lists_the_missing_input_alone():
    stopped: dict[str, Any] = {"verdict": "FAIL", "checks": [{"name": "ranks", "pass": False}]}
    FM._fails_too(stopped, "recorder_missing")
    assert stopped["failed"] == ["recorder_missing"] and stopped["verdict"] == "FAIL"


@pytest.mark.parametrize("record", [{"fold": {"ok": False}}, {"fold": {"ok": True}, "compare": None, "global_match": {}}], ids=["no stage", "null and empty stages"])
def test_a_rank_that_did_not_run_a_stage_has_an_empty_record_of_it(record):
    assert FM._stage({"1": record}, 1, "compare") == {} and FM._stage({"1": record}, 1, "global_match") == {}
    assert FM._stage({"1": {"compare": {"verdict": "PASS"}}}, 1, "compare") == {"verdict": "PASS"}


# the whole command, on stand-in stages

def _by_stage(doc: dict) -> dict:
    """The verdict document's block of verdicts by stage: the one that holds `fold_per_rank`."""
    return next(block for block in doc.values() if isinstance(block, dict) and "fold_per_rank" in block)


FOLD = {"rc": 0, "seconds": 0.0, "ok": True, "kept": False, "errors": 0, "unresolved": 0, "unsupported": 0, "n_instances": 9, "n_steps": 3, "digest": "dd",
        "collectives": 2, "collectives_by_kind": {"all_reduce": 2}, "error_examples": []}
CHECKED = {"verdict": "PASS", "checks": [{"name": "count", "pass": True}], "failed": [], "n_collectives": {"0": 2, "1": 2}, "peer_binding": {"a": 1, "b": 2}}


def _main(tmp_path: Path, monkeypatch, *, logs=(0, 1), recorders=(0, 1), links=(0, 1), gprog: bool = True, provenance: bool = True,
          checked: dict | None = None, summary: dict | None = None, states: dict | None = None) -> tuple[int, dict, dict, list]:
    """`tp-fold-match` of a two-rank capture whose stages are stand-ins: (exit code, tp2_fold_match.json, tp2_rank_match.json or {},
    the collectives the cross-rank check was given and which of its other inputs it got)."""
    capture, out = tmp_path / "capture", tmp_path / "out"
    given: list = []
    capture.mkdir()
    for r in (0, 1):
        build = tmp_path / f"b{r}"
        build.mkdir()
        if r in logs:
            (capture / f"log.rank{r}.jsonl.gz").write_text("")
        if r in recorders:
            (capture / f"coll.rank{r}.jsonl").write_text(json.dumps({"rank": r}) + "\n")
        if r in links:
            (build / "tp_links.json").write_text(json.dumps(dict(LINKS, rank=r)))
    (tmp_path / "tp2_match.json").write_text(json.dumps({"provenance": {"recorder_sha256": {"0": "aa"}}}))

    def fold_rank(py, log, profile, rank_dir, timeout, reuse=False, construction=()):
        rank = int(rank_dir.name[4:])
        rank_dir.mkdir(parents=True, exist_ok=True)
        (rank_dir / "instances.jsonl").write_text("")
        (rank_dir / "fold_summary.json").write_text(json.dumps(SUMMARY if summary is None else summary))
        return dict(FOLD, tp={"rank": rank, "world": 2} if states is None else states)

    def cross_rank_check(colls, **inputs):
        given.append((colls, sorted(name for name in ("recorder", "links", "provenance", "recorder_sha256") if inputs[name] is not None)))
        return json.loads(json.dumps(CHECKED if checked is None else checked))

    monkeypatch.setattr(FM, "fold_rank", fold_rank)
    monkeypatch.setattr(FM, "compare_rank", lambda *a, **k: {"comparator": "cb_a.compare", "verdict": "PASS", "seconds": 0.0})
    monkeypatch.setattr(FM, "global_match_rank", lambda *a, **k: {"verdict": "PASS", "ok": True, "failed_checks": [], "achieved_concurrency": None, "seconds": 0.0})
    monkeypatch.setattr(PM, "fold_construction_args", lambda workload, dirs: [])
    monkeypatch.setattr(RM, "cross_rank_check", cross_rank_check)
    monkeypatch.setattr(RM, "own_producers_from_record", lambda path: {})
    rc = FM.main(FM.Options(capture=str(capture), profile="profile", out=str(out), build_rank=[str(tmp_path / "b0"), str(tmp_path / "b1")],
                            gprog_rank=["0=g0.json", "1=g1.json"] if gprog else [], workload="workload.json" if gprog else None,
                            tp2_match=str(tmp_path / "tp2_match.json") if provenance else None))
    rank_match = out / "tp2_rank_match.json"
    return rc, json.loads((out / "tp2_fold_match.json").read_text()), json.loads(rank_match.read_text()) if rank_match.exists() else {}, given


def test_a_fold_match_whose_every_stage_passes_passes_and_counts_the_peer_bindings(tmp_path, monkeypatch):
    rc, doc, rank_match, given = _main(tmp_path, monkeypatch)
    assert rc == 0 and doc["pass"] is True and doc["problems"] == []
    assert _by_stage(doc) == {"fold_per_rank": {"0": True, "1": True}, "compare_per_rank": {"0": "PASS", "1": "PASS"},
                              "global_match_per_rank": {"0": "PASS", "1": "PASS"}, "cross_rank": "PASS"}
    assert doc["cross_rank"]["peer_binding_n"] == 2 and "peer_binding" not in doc["cross_rank"] and rank_match["peer_binding"] == {"a": 1, "b": 2}
    assert given == [({0: SUMMARY["fold"]["collectives"], 1: SUMMARY["fold"]["collectives"]}, ["links", "provenance", "recorder", "recorder_sha256"])]


@pytest.mark.parametrize("summary", [{}, {"fold": None}, {"fold": {"collectives": None}}], ids=["an empty summary", "null statistics", "a null list"])
def test_a_rank_whose_fold_summary_lists_no_collective_gives_the_cross_rank_check_none(tmp_path, monkeypatch, summary):
    assert _main(tmp_path, monkeypatch, summary=summary)[3][0][0] == {0: [], 1: []}


def test_a_rank_with_no_observer_log_fails_the_match_and_ran_no_compare_and_no_global_match(tmp_path, monkeypatch):
    rc, doc, rank_match, given = _main(tmp_path, monkeypatch, logs=(0,))
    assert rc == 1 and doc["pass"] is False and doc["problems"] == ["rank 1: partner rank missing (no log.rank1.jsonl.gz)"]
    assert _by_stage(doc) == {"fold_per_rank": {"0": True, "1": False}, "compare_per_rank": {"0": "PASS", "1": None},
                              "global_match_per_rank": {"0": "PASS", "1": None}, "cross_rank": "FAIL"}
    assert doc["cross_rank"]["failed"] == ["partner_rank"] and doc["cross_rank"]["peer_binding_n"] == 0
    assert rank_match == {} and given == []


def test_a_rank_given_no_global_program_fails_the_match_with_no_global_match_record(tmp_path, monkeypatch):
    rc, doc, _, _ = _main(tmp_path, monkeypatch, gprog=False)
    assert rc == 1 and _by_stage(doc)["global_match_per_rank"] == {"0": None, "1": None} and _by_stage(doc)["cross_rank"] == "PASS"
    assert [p.split(":")[0] + ":" + p.split(":")[1].split("(")[0] for p in doc["problems"]] == ["rank 0: GM-01 stage not run ", "rank 1: GM-01 stage not run "]


@pytest.mark.parametrize("states", [None, {}], ids=["no header", "an empty header"])
def test_a_log_whose_header_states_no_rank_is_no_ranks_log(tmp_path, monkeypatch, states):
    rc, doc, _, _ = _main(tmp_path, monkeypatch, states={} if states is None else states, summary={"errors": 0} if states is None else None)
    assert rc == 1 and doc["problems"] == [f"rank {r}: the log's RunHeader.notes.tp2 states rank None / world None (swapped or mislabelled rank)" for r in (0, 1)]


@pytest.mark.parametrize("listed", [{}, {"checks": None}, {"checks": []}], ids=["no list", "a null list", "an empty list"])
def test_a_cross_rank_check_that_lists_no_check_is_logged_with_none(tmp_path, monkeypatch, caplog, listed):
    checked = {k: v for k, v in CHECKED.items() if k != "checks"} | listed
    with caplog.at_level("INFO", logger=FM.logger.name):
        rc, doc, _, _ = _main(tmp_path, monkeypatch, checked=checked)
    assert rc == 0 and doc["cross_rank"]["verdict"] == "PASS"
    assert [m for m in caplog.messages if "cross-rank check" in m] == ["[tp2foldmatch] cross-rank check: PASS failed=[] collectives={'0': 2, '1': 2} checks=[]"]


STOPPED = {"verdict": "FAIL", "checks": [{"name": "count", "pass": False, "fails": ["rank 1: duplicate (step, k) among its collectives"]}]}


@pytest.mark.parametrize("checked, own", [(None, []), (dict(CHECKED, verdict="FAIL", failed=["peer"]), ["peer"]), (STOPPED, [])],
                         ids=["a check that passed", "a check that failed", "a check that stopped early"])
def test_the_inputs_the_cross_rank_check_was_not_given_fail_it_by_name_after_its_own(tmp_path, monkeypatch, checked, own):
    rc, doc, rank_match, given = _main(tmp_path, monkeypatch, recorders=(0,), links=(1,), provenance=False, checked=checked)
    assert rc == 1 and given[0][1] == []
    assert doc["cross_rank"]["failed"] == own + ["recorder_missing", "links_missing", "provenance_missing"] == rank_match["failed"]
    assert doc["cross_rank"]["verdict"] == "FAIL" == rank_match["verdict"] and _by_stage(doc)["cross_rank"] == "FAIL"
    assert doc["cross_rank"]["peer_binding_n"] == (0 if checked is STOPPED else 2)
    assert [p for p in doc["problems"] if "cannot run" in p or "REQUIRED" in p] == [
        "rank 0: no tp_links.json for the rank Program(s); the link check cannot run",
        f"rank 1: no collective recorder rows ({tmp_path / 'capture' / 'coll.rank1.jsonl'}); the value check cannot run",
        "no --tp2-match: the stored-step provenance check (received == Def(part_rank0 .. part_rank{world-1})) is REQUIRED"]
