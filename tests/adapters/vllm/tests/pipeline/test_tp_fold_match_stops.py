"""`verity_vllm.pipeline.tp.fold_match` where a stage stops or a command line is refused: a stage's subprocess that outlives its
timeout, a rank whose fold is not ok, a GM-01 that fails, a GP-01 Program given without a workload, the shared X-09 (GM-01 before
the compare, and once), a rank's further request Programs in the spelling for any world, a provenance record that cannot be read
or holds none, a stage's document that parses and is not a mapping, and the command lines `main` refuses.  No stage's
subprocess runs here: `subprocess.run` is a stand-in where `_run` is the subject, and `main` runs on the stand-in stages of
`test_tp_fold_match_records`."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.pipeline import test_tp_fold_match_records as FR
from verity_vllm.check.match import rank_match as RM
from verity_vllm.config import UsageError
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import match as PM
from verity_vllm.pipeline.tp import fold_match as FM

PASSED = {"verdict": "PASS", "ok": True, "failed_checks": [], "achieved_concurrency": None, "seconds": 0.0}
NOT_RUN = "rank {r}: GM-01 stage not run (--gprog-rank{r} / --workload absent): the global Match over the rank's GP-01 workload Program is required"
NO_WORKLOAD = "--gprog given without --workload"


# a stage's subprocess

def test_a_stage_that_outlives_its_timeout_exits_124_and_its_log_says_after_how_long(tmp_path, monkeypatch):
    def run(cmd, **kw):
        kw["stdout"].write("half a line")
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(FM.subprocess, "run", run)
    log = tmp_path / "rank1" / "fold.log"
    assert FM._run(["py", "-m", "stage"], log, 7) == 124
    assert log.read_text() == "$ py -m stage\nhalf a line\nTIMEOUT after 7s\n"


def test_a_stage_that_ends_exits_with_its_own_code_and_its_log_names_no_timeout(tmp_path, monkeypatch):
    asked: dict[str, Any] = {}

    def run(cmd, **kw):
        asked.update(kw, cmd=cmd)
        return SimpleNamespace(returncode=3)

    monkeypatch.setattr(FM.subprocess, "run", run)
    log = tmp_path / "rank1" / "fold.log"
    assert FM._run(["py", "-m", "stage"], log, 7) == 3 and log.read_text() == "$ py -m stage\n"
    assert (asked["cmd"], asked["timeout"], asked["stderr"]) == (["py", "-m", "stage"], 7, subprocess.STDOUT)
    assert FM._run(["py"], log) == 3 and asked["timeout"] is None


# a stage's document that is not a mapping

#: JSON that parses to something no stage writes: each stage's document is a mapping
NOT_MAPPINGS = ["[1]", "[]", '"fold"', "7", "true"]


@pytest.mark.parametrize("text", NOT_MAPPINGS)
def test_a_stages_document_that_is_not_a_mapping_reads_as_none_written(tmp_path, text):
    (tmp_path / "document.json").write_text(text)
    assert FM._load(tmp_path / "document.json") is None
    (tmp_path / "document.json").write_text('{"verdict": "PASS"}')
    assert FM._load(tmp_path / "document.json") == {"verdict": "PASS"}


@pytest.mark.parametrize("text", NOT_MAPPINGS)
def test_a_fold_summary_that_is_not_a_mapping_states_no_rank_and_no_collective(tmp_path, monkeypatch, text):
    fold = FR._fold(tmp_path, monkeypatch, None, text=text)
    assert (fold["tp"], fold["collectives"], fold["collectives_by_kind"], fold["error_examples"], fold["errors"]) == (None, 0, None, [], None)


@pytest.mark.parametrize("text", NOT_MAPPINGS)
def test_a_compare_a_decomposition_or_a_global_match_that_is_not_a_mapping_fails_as_one_that_wrote_none(tmp_path, monkeypatch, text):
    for name in ("match_compare", "match_dag", "match_decomp", "global_match"):
        (tmp_path / f"{name}.json").write_text(text)
    compared = FR._compare(tmp_path, monkeypatch, 0, [])
    assert compared["verdict"] == "FAIL canonical_equal=None dag_multiset_equal=False steps=None/None operands_different=None rc=0"
    assert compared["dag_multiset"] is None           # the record holds no document, where it would hold the one that is not a mapping
    assert FR._compare(tmp_path, monkeypatch, 0, ["d1"])["verdict"] == "FAIL batch_decomp None rc=0: "
    got = FM.global_match_rank("py", tmp_path, tmp_path / "gp.json", tmp_path / "wl.json", "dag", None)
    assert (got["verdict"], got["ok"], got["checks"], got["generated_lengths"]) == ("rc=0", False, {}, {})


@pytest.mark.parametrize("text", NOT_MAPPINGS)
def test_a_kept_document_that_is_not_a_mapping_is_not_reused_and_its_stage_runs_again(tmp_path, monkeypatch, text):
    ran: list[list[str]] = []

    def run(cmd: list[str], log: Path, timeout: int | None = None) -> int:
        ran.append(cmd)
        return 5

    monkeypatch.setattr(FM, "_run", run)
    document = tmp_path / "match_compare.json"
    document.write_text(text)
    assert FM._run_unless(True, document, ["py", "-m", "stage"], tmp_path / "match_compare.log", None) == 5 and ran == [["py", "-m", "stage"]]
    document.write_text("{}")
    assert FM._run_unless(True, document, ["py", "-m", "stage"], tmp_path / "match_compare.log", None) == 0 and len(ran) == 1


# a rank's stages

def _stages(monkeypatch, *, global_match: dict[str, dict] | None = None) -> list[tuple]:
    """Stand-ins for the compare and GM-01 that note, in the list returned, (the rank's directory, the stage, what tells the
    shared X-09 from the other: the compare's `shared`, GM-01's `x09`).  `global_match` gives a rank's GM-01 record by its
    directory's name; any other rank's passes."""
    ran: list[tuple] = []

    def compare_rank(py, rank_dir, build, derived, within, timeout, reuse=False, shared=False):
        ran.append((rank_dir.name, "compare", shared))
        requests = [{"request_id": f"q{i}", "derived": d} for i, d in enumerate([str(build), *derived])]
        return {"comparator": "cb_a.compare", "verdict": "PASS", "seconds": 0.0, "per_request": requests}

    def global_match_rank(py, rank_dir, gprog, workload, within, timeout, reuse=False, x09=None):
        ran.append((rank_dir.name, "GM-01", x09))
        return dict((global_match or {}).get(rank_dir.name, PASSED))

    monkeypatch.setattr(FM, "compare_rank", compare_rank)
    monkeypatch.setattr(FM, "global_match_rank", global_match_rank)
    return ran


def _of(ran: list[tuple], rank: int) -> list[tuple]:
    """What rank `rank` ran, in its own order (the ranks run at once, so the order between two ranks is not a fact)."""
    return [stage[1:] for stage in ran if stage[0] == f"rank{rank}"]


def test_a_rank_whose_fold_is_not_ok_is_named_with_its_first_examples_and_runs_no_stage_after_it(tmp_path, monkeypatch):
    capture, out, given = FR._stand_ins(tmp_path, monkeypatch)
    ran, folds = _stages(monkeypatch), FM.fold_rank
    examples = [{"e": "x" * 200}, {"e": "y" * 200}]

    def fold_rank(py, log, profile, rank_dir, *rest, **kw):
        fold = folds(py, log, profile, rank_dir, *rest, **kw)
        # the rank its log's header states is another's: a fold that is not ok is named for the fold alone
        return fold if rank_dir.name == "rank0" else dict(fold, ok=False, rc=1, errors=2, unresolved=5, error_examples=examples, tp={"rank": 0, "world": 2})

    monkeypatch.setattr(FM, "fold_rank", fold_rank)
    rc, doc, rank_match, given = FR._ran(FM.main(FR._command(tmp_path, capture, out)), out, given)
    assert rc == 1 and doc["pass"] is False
    assert doc["problems"] == ["rank 1: fold not ok (rc 1, errors 2, unresolved 5): " + json.dumps(examples)[:300]]
    assert _of(ran, 1) == [] and _of(ran, 0) == [("compare", False), ("GM-01", None)]
    assert FR._by_stage(doc) == {"fold_per_rank": {"0": True, "1": False}, "compare_per_rank": {"0": "PASS", "1": None},
                                 "global_match_per_rank": {"0": "PASS", "1": None}, "cross_rank": "FAIL"}
    assert doc["cross_rank"]["failed"] == ["partner_rank"] and doc["cross_rank"]["why"] == "ranks folded [0] of [0, 1]: the cross-rank check needs every rank's fold"
    assert rank_match == {} and given == []


@pytest.mark.parametrize("verdict, failed, named", [("FAIL", ["GM-01b", "GM-01d"], "rank 1: GM-01 FAIL ['GM-01b', 'GM-01d']"), ("TIMEOUT", None, "rank 1: GM-01 TIMEOUT None")],
                         ids=["a document that fails two checks", "no document: it timed out"])
def test_a_rank_whose_global_match_does_not_pass_is_named_with_its_verdict_and_the_checks_it_failed(tmp_path, monkeypatch, verdict, failed, named):
    capture, out, given = FR._stand_ins(tmp_path, monkeypatch)
    ran = _stages(monkeypatch, global_match={"rank1": dict(PASSED, verdict=verdict, ok=False, failed_checks=failed)})
    rc, doc, _, given = FR._ran(FM.main(FR._command(tmp_path, capture, out)), out, given)
    assert rc == 1 and doc["pass"] is False and doc["problems"] == [named]
    assert FR._by_stage(doc)["global_match_per_rank"] == {"0": "PASS", "1": verdict} and FR._by_stage(doc)["cross_rank"] == "PASS"
    assert _of(ran, 1) == [("compare", False), ("GM-01", None)] and len(given) == 1


def test_a_global_program_given_without_a_workload_is_named_for_its_rank_and_no_global_match_runs(tmp_path, monkeypatch):
    capture, out, given = FR._stand_ins(tmp_path, monkeypatch)
    ran = _stages(monkeypatch)
    options = FM.Options(capture=str(capture), profile="profile", out=str(out), build_rank=[str(tmp_path / "b0"), str(tmp_path / "b1")],
                         gprog_rank=["1=g1.json"], workload=None, tp2_match=str(tmp_path / "tp2_match.json"))
    rc, doc, _, _ = FR._ran(FM.main(options), out, given)
    assert rc == 1 and doc["problems"] == [NOT_RUN.format(r=0), NOT_RUN.format(r=1), NO_WORKLOAD]
    assert sorted(ran) == [("rank0", "compare", False), ("rank1", "compare", False)]
    assert FR._by_stage(doc)["global_match_per_rank"] == {"0": None, "1": None}


# the shared X-09

def _shared(tmp_path: Path, monkeypatch, **options: Any) -> tuple[int, dict, list[tuple], list[str]]:
    """`tp-fold-match` of a capture whose rank 1 has two further request Programs and whose rank 0 has none: (exit code,
    tp2_fold_match.json, what each rank ran, rank 1's Program dirs)."""
    capture, out, given = FR._stand_ins(tmp_path, monkeypatch)
    ran = _stages(monkeypatch)
    further = [FR._program(tmp_path, "d1a"), FR._program(tmp_path, "d1b")]
    given_options: dict[str, Any] = {"gprog_rank": ["0=g0.json", "1=g1.json"], "workload": "workload.json"} | options
    rc = FM.main(FM.Options(capture=str(capture), profile="profile", out=str(out), build_rank=[str(tmp_path / "b0"), str(tmp_path / "b1")],
                            derived_rank=[f"1={d}" for d in further], tp2_match=str(tmp_path / "tp2_match.json"), **given_options))
    return rc, FR._ran(rc, out, given)[1], ran, [str(tmp_path / "b1"), *further]


def test_under_shared_x09_a_rank_with_further_programs_runs_its_global_match_first_and_once_over_all_of_them(tmp_path, monkeypatch):
    rc, doc, ran, programs = _shared(tmp_path, monkeypatch, shared_x09=True)
    assert rc == 0 and doc["pass"] is True and doc["problems"] == []
    assert _of(ran, 1) == [("GM-01", programs), ("compare", True)]
    assert FR._by_stage(doc)["global_match_per_rank"] == {"0": "PASS", "1": "PASS"}


def test_under_shared_x09_a_rank_with_one_program_compares_first_and_shares_nothing(tmp_path, monkeypatch):
    assert _of(_shared(tmp_path, monkeypatch, shared_x09=True)[2], 0) == [("compare", False), ("GM-01", None)]


def test_without_shared_x09_a_rank_with_further_programs_compares_first_in_a_process_of_its_own(tmp_path, monkeypatch):
    rc, doc, ran, _ = _shared(tmp_path, monkeypatch)
    assert rc == 0 and doc["pass"] is True and _of(ran, 1) == [("compare", False), ("GM-01", None)]


@pytest.mark.parametrize("lacked, problems", [
    ({"gprog_rank": ["0=g0.json"]}, [NOT_RUN.format(r=1)]),
    ({"workload": None}, [NOT_RUN.format(r=0), NO_WORKLOAD, NOT_RUN.format(r=1), NO_WORKLOAD]),
], ids=["no GP-01 Program for the rank", "no workload"])
def test_under_shared_x09_a_rank_with_no_global_match_to_run_compares_in_a_process_of_its_own(tmp_path, monkeypatch, lacked, problems):
    rc, doc, ran, _ = _shared(tmp_path, monkeypatch, shared_x09=True, **lacked)
    assert rc == 1 and _of(ran, 1) == [("compare", False)] and doc["problems"] == problems


def test_a_shared_x09_whose_global_match_fails_is_named_once(tmp_path, monkeypatch):
    capture, out, given = FR._stand_ins(tmp_path, monkeypatch)
    ran = _stages(monkeypatch, global_match={"rank1": dict(PASSED, verdict="FAIL", ok=False, failed_checks=["GM-01a"])})
    further = FR._program(tmp_path, "d1a")
    rc, doc, _, _ = FR._ran(FM.main(FR._command(tmp_path, capture, out, "--shared-x09", "--derived-rank1", further)), out, given)
    assert rc == 1 and doc["problems"] == ["rank 1: GM-01 FAIL ['GM-01a']"]
    assert _of(ran, 1) == [("GM-01", [str(tmp_path / "b1"), further]), ("compare", True)]


# the spelling for any world

def _any_world(tmp_path: Path, capture: Path, out: Path, *argv: str) -> FM.Options:
    """What `verity-vllm tp-fold-match` reads from a command line that names the rank Builds in rank order, and `argv` after it."""
    return cli.parse("tp-fold-match", ["--capture", str(capture), "--profile", "profile", "--out", str(out), "--build-rank", str(tmp_path / "b0"),
                                       "--build-rank", str(tmp_path / "b1"), "--gprog-rank", "0=g0.json", "--gprog-rank", "1=g1.json",
                                       "--workload", str(tmp_path / "workload.json"), "--tp2-match", str(tmp_path / "tp2_match.json"), *argv], {})


def test_derived_rank_names_a_ranks_further_request_programs_by_its_number_in_the_order_given(tmp_path, monkeypatch):
    capture, out, given = FR._stand_ins(tmp_path, monkeypatch)
    _stages(monkeypatch)
    further = {0: [FR._program(tmp_path, "d0a")], 1: [FR._program(tmp_path, "d1b"), FR._program(tmp_path, "d=1a")]}
    builds = {r: str(tmp_path / f"b{r}") for r in (0, 1)}
    folded: list = []
    linked: list = []

    def cross_rank_check(colls, **inputs):
        linked.append({rank: sorted(by_request) for rank, by_request in inputs["links"].items()})
        return json.loads(json.dumps(FR.CHECKED))

    monkeypatch.setattr(PM, "fold_construction_args", FR._noting(folded))
    monkeypatch.setattr(RM, "cross_rank_check", cross_rank_check)
    argv = ["--derived-rank", f"1={further[1][0]}", "--derived-rank", f"0={further[0][0]}", "--derived-rank", f"1={further[1][1]}", "--world", "3"]
    rc, doc, _, _ = FR._ran(FM.main(_any_world(tmp_path, capture, out, *argv)), out, given)
    assert rc == 0 and doc["pass"] is True and doc["problems"] == []
    assert folded == [[builds[0], *further[0]], [builds[1], *further[1]]]
    assert linked == [{0: ["q0", "q1"], 1: ["q0", "q1", "q2"]}]
    assert doc["world"] == 2, "the world is the number of rank Builds named, whatever --world says"


# the command lines `main` refuses

BOTH = "give either --build-rank DIR (repeated, rank order) or --build-rank0/--build-rank1, not both"
NEITHER = "--build-rank DIR (repeated, rank order) or --build-rank0 and --build-rank1 required"


@pytest.mark.parametrize("builds, world, why", [
    ({"build_rank": ["b0", "b1"], "build_rank0": "b0"}, 2, BOTH),
    ({"build_rank": ["b0", "b1"], "build_rank1": "b1"}, 2, BOTH),
    ({"build_rank": ["b0", "b1"], "build_rank0": "b0", "build_rank1": "b1"}, 2, BOTH),
    ({}, 2, NEITHER),
    ({"build_rank0": "b0"}, 2, NEITHER),
    ({"build_rank1": "b1"}, 2, NEITHER),
    ({"build_rank": ["b0"]}, 2, "world 1 vs 1 rank Builds"),
    ({"build_rank0": "b0", "build_rank1": "b1"}, 3, "world 3 vs 2 rank Builds"),
    ({"build_rank0": "b0", "build_rank1": "b1"}, 1, "world 1 vs 2 rank Builds"),
], ids=["both spellings, rank 0", "both spellings, rank 1", "both spellings, both ranks", "no rank Build", "rank 0's Build alone", "rank 1's Build alone",
        "one rank Build in rank order", "two rank Builds for a world of three", "two rank Builds for a world of one"])
def test_a_command_line_whose_rank_builds_are_not_one_worlds_is_refused_before_any_stage(tmp_path, monkeypatch, builds, world, why):
    capture, out, given = FR._stand_ins(tmp_path, monkeypatch)
    ran = _stages(monkeypatch)
    folded: list = []
    monkeypatch.setattr(PM, "fold_construction_args", FR._noting(folded))
    with pytest.raises(UsageError) as refused:
        FM.main(FM.Options(capture=str(capture), profile="profile", out=str(out), world=world, **builds))
    assert str(refused.value) == why
    assert (ran, folded, given) == ([], [], []) and not (out / "tp2_fold_match.json").exists()


# the provenance record

def _provenance(tmp_path: Path, monkeypatch, write: Any) -> tuple[int, dict, dict, list, Path]:
    """`tp-fold-match` whose `--tp2-match` names what `write` leaves at the path: (exit code, tp2_fold_match.json,
    tp2_rank_match.json, what the cross-rank check was given, the path)."""
    capture, out, given = FR._stand_ins(tmp_path, monkeypatch)
    path = tmp_path / "tp2_match.json"
    path.unlink()
    write(path)
    return *FR._ran(FM.main(FR._command(tmp_path, capture, out)), out, given), path


def test_the_provenance_record_read_is_named_by_its_path_its_digest_and_the_recorder_rows_it_binds(tmp_path, monkeypatch):
    text = json.dumps({"provenance": {"recorder_sha256": {"0": "aa", "1": "bb"}, "pass": True}})
    rc, doc, _, given, path = _provenance(tmp_path, monkeypatch, lambda p: p.write_text(text))
    assert rc == 0 and doc["problems"] == [] and given[0][1] == ["links", "provenance", "recorder", "recorder_sha256"]
    assert doc["tp2_match"] == {"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest(), "recorder_sha256": {"0": "aa", "1": "bb"}}


@pytest.mark.parametrize("text", ["{}", '{"provenance": null}', "{", FR.NESTED, *NOT_MAPPINGS,
                                  '{"provenance": [1]}', '{"provenance": []}', '{"provenance": "pass"}', '{"provenance": 0}', '{"provenance": true}'],
                         ids=["no provenance", "a null provenance", "not JSON", "nested past the parser's depth", *NOT_MAPPINGS,
                              "a list for it", "an empty list for it", "text for it", "a number for it", "true for it"])
def test_a_match_record_that_holds_no_provenance_is_named_by_its_digest_and_fails_the_cross_rank_check_by_name(tmp_path, monkeypatch, text):
    rc, doc, rank_match, given, path = _provenance(tmp_path, monkeypatch, lambda p: p.write_text(text))
    assert rc == 1 and doc["problems"] == [f"{path}: no provenance record"]
    assert doc["tp2_match"] == {"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest(), "recorder_sha256": None}
    assert given[0][1] == ["links", "recorder", "recorder_sha256"]
    assert doc["cross_rank"]["failed"] == ["provenance_missing"] == rank_match["failed"] and doc["cross_rank"]["verdict"] == "FAIL"


@pytest.mark.parametrize("write", [lambda p: None, Path.mkdir], ids=["no file", "a directory"])
def test_a_match_record_that_cannot_be_read_is_named_as_unreadable_and_as_holding_no_provenance(tmp_path, monkeypatch, write):
    rc, doc, rank_match, given, path = _provenance(tmp_path, monkeypatch, write)
    assert rc == 1 and len(doc["problems"]) == 2 and "tp2_match" not in doc
    assert doc["problems"][0].startswith(f"{path}: unreadable ([Errno ") and doc["problems"][0].endswith(f"'{path}')")
    assert doc["problems"][1] == f"{path}: no provenance record"
    assert given[0][1] == ["links", "recorder", "recorder_sha256"] and rank_match["failed"] == ["provenance_missing"]
