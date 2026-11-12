"""What the Match's commands say beside the documents they write goes to the package's logger, and nothing to stdout:
`verity_vllm.check.match.batch_decomp.main` and `verity_vllm.check.match.program_dag.main` log the result they wrote,
`verity_vllm.check.match.global_match.main` logs what is about to run before it runs, its brief after, and as an error the X-09
document it could not write, `verity_vllm.check.match.instance_sequence` names in a warning a registry module that failed to
import, and `verity_vllm.check.match.global_match_fast` names in a warning a forked worker's result file it could not remove.

`batch_decomp` and `program_dag` run on the toy of `test_batch_decomp_e2e.py`; `global_match.main` runs over a stand-in for its
checker, since the lines under test are `main`'s own.

`batch_decomp.main` and `global_match.main` tune the collector of the process that runs them. Under a test that process is pytest's,
so every test here runs with the setting that leaves the interpreter's thresholds alone, and is held to having left them."""
from __future__ import annotations

import collections
import gc
import json
import logging
import os
import tempfile

import pytest
from verity.definitions.registry import catalog

from tests.check import test_batch_decomp_e2e as e2e
from verity_vllm import gc_tuning
from verity_vllm.check.match import batch_decomp as bd
from verity_vllm.check.match import global_match as gm
from verity_vllm.check.match import global_match_fast as gmf
from verity_vllm.check.match import instance_sequence as seq
from verity_vllm.check.match import program_dag
from verity_vllm.check.match import program_loaders as loaders

memory = e2e.memory                          # the toy's fixtures, under their own names
derived_is_a_fold = e2e.derived_is_a_fold


@pytest.fixture(autouse=True)
def the_collector_is_left_as_it_was(monkeypatch):
    monkeypatch.setenv(gc_tuning.ENV, "default")
    before = gc.get_threshold()
    yield
    assert gc.get_threshold() == before, "a command tuned the collector of the test process"


def _logged(caplog, module) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == module.__name__]


# ------------------------------------------------------------------------------------------------------------------ batch-decomp


def test_batch_decomp_writes_its_document_and_logs_its_brief(memory, tmp_path, derived_is_a_fold, caplog, capsys):
    d, _ = e2e.fold_batched([e2e.R0, e2e.R1], tmp_path / "b2")
    derived = [e2e.fold_alone(e2e.R0, tmp_path / "LP3_T2"), e2e.fold_alone(e2e.R1, tmp_path / "LP2_T1")]
    out = tmp_path / "decomp.json"
    with caplog.at_level(logging.INFO, logger="verity_vllm"):
        rc = bd.main(bd.Options(oracle=d, derived=derived, out=str(out)))
    doc = json.loads(out.read_text())
    (rec,) = _logged(caplog, bd)
    brief = json.loads(rec.getMessage())
    assert rc == 0 and rec.levelno == logging.INFO and brief["verdict"] == doc["verdict"] == "PASS"
    assert (brief["within_step"], brief["seconds"], brief["xreq_total"]) == (doc["within_step_criterion"], doc["seconds"], 0)
    assert [(e["request_id"], e["LP"], e["T"], e["canonical_equal"]) for e in brief["requests"]] == [("r0", 3, 2, True), ("r1", 2, 1, True)]
    assert "unattributed_examples" not in brief["attribution"] and brief["attribution"]["unattributed"] == 0
    assert capsys.readouterr().out == ""


def test_batch_decomp_exits_4_on_a_shape_that_is_not_derived_and_3_on_a_failure(memory, tmp_path, derived_is_a_fold, caplog):
    d, _ = e2e.fold_batched([e2e.R0, e2e.R1], tmp_path / "b2")
    with caplog.at_level(logging.INFO, logger="verity_vllm"):
        assert bd.main(bd.Options(oracle=d, derived=[e2e.fold_alone(e2e.R0, tmp_path / "LP3_T2")], out=str(tmp_path / "missing.json"))) == 4
        # r1's Program replaced by r0's under r1's shape: its instances differ
        wrong = e2e.fold_alone(e2e.R0, tmp_path / "LP2_T1")
        (tmp_path / "LP2_T1" / "construction_manifest.json").write_text(json.dumps({"structural_inputs": {"LP": 2, "T": 1}}))
        assert bd.main(bd.Options(oracle=d, derived=[e2e.fold_alone(e2e.R0, tmp_path / "LP3_T2"), wrong], out=str(tmp_path / "fail.json"))) == 3
    missing, failed = (json.loads(r.getMessage()) for r in _logged(caplog, bd))
    assert (missing["verdict"], missing["missing_derived"]) == ("MISSING_DERIVED", [[2, 1]])
    assert failed["verdict"] == "FAIL" and failed["reasons"] == json.loads((tmp_path / "fail.json").read_text())["reasons"]


# ------------------------------------------------------------------------------------------------------------------ program-dag


def test_program_dag_writes_its_result_logs_it_and_exits_by_it(memory, tmp_path, monkeypatch, caplog, capsys):
    """The toy has no derived dir: a single request's fold stands in for it, as it does for `batch_decomp`."""
    monkeypatch.setattr(loaders, "load_derived", lambda d, label, compact: loaders.load_oracle(d, label, compact=compact))
    a, b = e2e.fold_alone(e2e.R0, tmp_path / "a"), e2e.fold_alone(e2e.R1, tmp_path / "b")
    same, other = tmp_path / "same.json", tmp_path / "other.json"
    with caplog.at_level(logging.INFO, logger="verity_vllm"):
        assert program_dag.main(program_dag.Options(a=a, b=a, out=str(same))) == 0
        assert program_dag.main(program_dag.Options(a=a, b=b, out=str(other))) == 1
    first, second = _logged(caplog, program_dag)
    assert json.loads(first.getMessage()) == json.loads(same.read_text()) and json.loads(same.read_text())["equal"] is True
    assert json.loads(second.getMessage()) == json.loads(other.read_text()) and json.loads(other.read_text())["equal"] is False
    assert first.levelno == second.levelno == logging.INFO and capsys.readouterr().out == ""


# ------------------------------------------------------------------------------------------------------------------ global-match


def _result(**fields) -> dict:
    """What `main` reads of a checker's result."""
    return {"verdict": "PASS", "comparator": gm.COMPARATOR, "rules_version": gm.RULES_VERSION, "within_step_criterion": "dag",
            "checks": [{"id": "G1", "status": "pass"}, {"id": "G3", "status": "pass"}], "failed_checks": [], "reasons": [],
            "requests": [{"request_id": "r0", "LP": 3, "n_instances": 7, "not_in_the_brief": 1}], "seconds": 0.5, **fields}


def _stand_in(monkeypatch, tmp_path, result, before=lambda: None):
    """`global-match` over the record implementation with its checker replaced: it returns `result`, after `before()`."""
    gpp = tmp_path / "gp.json"
    gpp.write_text(json.dumps({"schema": gm.GP_SCHEMA, "requests": []}))
    monkeypatch.setenv(gm.IMPL_ENV, "record")

    def check(gp, oracle, **kw):
        before()
        return dict(result)
    monkeypatch.setattr(gm, "check", check)
    return str(gpp)


def test_global_match_logs_what_is_about_to_run_before_it_runs_and_its_brief_after(tmp_path, monkeypatch, caplog, capsys):
    before: list[list[str]] = []
    gpp = _stand_in(monkeypatch, tmp_path, _result(), lambda: before.append([r.getMessage() for r in _logged(caplog, gm)]))
    out = tmp_path / "gm.json"
    with caplog.at_level(logging.INFO, logger="verity_vllm"):
        rc = gm.main(gm.Options(oracle="the-oracle", global_program=gpp, out=str(out)))
    banner, brief = _logged(caplog, gm)
    assert rc == 0 and before == [[banner.getMessage()]]                       # the banner, and only it, before the checker ran
    assert banner.getMessage() == gm.impl_banner(gm, {"schema": gm.GP_SCHEMA, "requests": []}) and banner.levelno == logging.INFO
    assert banner.getMessage().startswith("# global_match impl=record")
    said, doc = json.loads(brief.getMessage()), json.loads(out.read_text())
    assert brief.levelno == logging.INFO and (said["verdict"], said["within_step"], said["seconds"]) == ("PASS", "dag", 0.5)
    assert said["checks"] == [["G1", "pass"], ["G3", "pass"]] and said["impl"] == doc["impl"] and said["impl"]["name"] == "record"
    assert said["requests"][0]["request_id"] == "r0" and "not_in_the_brief" not in said["requests"][0]
    assert capsys.readouterr().out == ""


def test_global_match_says_as_an_error_that_x09_left_no_document_to_write(tmp_path, monkeypatch, caplog):
    x09 = {"verdict": "ERROR", "error": "the decomposition raised"}
    gpp = _stand_in(monkeypatch, tmp_path, _result(verdict="FAIL", x09=x09))
    dout = tmp_path / "decomp.json"
    with caplog.at_level(logging.INFO, logger="verity_vllm"):
        rc = gm.main(gm.Options(oracle="the-oracle", global_program=gpp, out=str(tmp_path / "gm.json"), decomp_out=str(dout)))
    banner, error = _logged(caplog, gm)                                        # no brief: the command stopped at the missing document
    assert rc == 3 and not dout.exists() and (banner.levelno, error.levelno) == (logging.INFO, logging.ERROR)
    assert json.loads(error.getMessage()) == {"decomp_out": str(dout), "error": "X-09 produced no document", "x09": x09}


# ------------------------------------------------------------------------------------------------------------------ the registry


def test_a_registry_module_that_failed_to_import_is_named_in_a_warning(monkeypatch, caplog):
    registry = object()
    monkeypatch.setattr(catalog, "load", lambda: (registry, ["hopper: needs a newer Python"]))
    with caplog.at_level(logging.WARNING, logger="verity_vllm"):
        assert seq._registry() is registry
    (rec,) = _logged(caplog, seq)
    assert rec.levelno == logging.WARNING
    assert rec.getMessage() == "[compare] registry module unavailable (hopper: needs a newer Python): Programs naming its Definitions will not decode"


# ------------------------------------------------------------------------------------------------------------------ the forked workers


def test_a_workers_result_file_left_behind_is_named_in_a_warning_and_its_results_still_return(tmp_path, monkeypatch, caplog):
    """Two items on two forked workers, whose result files the parent reads and then cannot remove."""
    unlink = os.unlink

    def refused(path, *args, **kwargs):
        if os.path.dirname(path) == str(tmp_path):
            raise PermissionError(13, "Permission denied", path)
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(gmf, "stats", collections.Counter())
    monkeypatch.setattr(os, "unlink", refused)
    with caplog.at_level(logging.WARNING, logger="verity_vllm"):
        assert gmf.request_legs([3, 4], lambda x: x * x, 2) == [9, 16]
    left = sorted(str(f) for f in tmp_path.glob("*.pkl"))
    records = _logged(caplog, gmf)
    assert len(left) == 2 and [r.levelno for r in records] == [logging.WARNING, logging.WARNING]
    assert sorted(r.getMessage().split(" result file ")[1].split(" is left behind: ")[0] for r in records) == left
    assert all(r.getMessage().startswith("worker ") and "PermissionError(13" in r.getMessage() for r in records)
