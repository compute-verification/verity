"""What the Match's commands say beside the documents they write goes to the package's logger, and nothing to stdout:
`verity_vllm.check.match.batch_decomp.main` and `verity_vllm.check.match.program_dag.main` log the result they wrote,
`verity_vllm.check.match.global_match.main` logs what is about to run before it runs, its brief after, and as an error the X-09
document it could not write, `verity_vllm.check.match.instance_sequence` names in a warning a registry module that failed to
import, and `verity_vllm.check.match.global_match_fast` names in a warning a forked worker's result file it could not remove.

`batch_decomp` and `program_dag` run on the toy of `test_batch_decomp_e2e.py`; `global_match.main` runs over a stand-in for its
checker, since the lines under test are `main`'s own.

The capture log is also an input the two commands take by path: `batch-decomp --log` and `global-match --log` name a capture
log kept outside `<oracle>/capture/`, and `global-match --tokens` the tokens file.  Those tests run each command from its
command line on a record laid out that way, `global-match` over its checker of record.

A request that fails on a forked worker is said by the parent, which raises: with each failed worker's error and where it was
raised, with the items and the wait status of a worker that left no result (it died, or could not write its error, which it
then says on stderr), and with an item that has no result counted as missing.  No partial list is returned, and no result file
is left behind."""
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
from tests.check import test_global_match as tgm
from verity_vllm import gc_tuning
from verity_vllm.check.match import batch_decomp as bd
from verity_vllm.check.match import global_match as gm
from verity_vllm.check.match import global_match_fast as gmf
from verity_vllm.check.match import instance_sequence as seq
from verity_vllm.check.match import program_dag
from verity_vllm.check.match import program_loaders as loaders
from verity_vllm.pipeline import cli

memory = e2e.memory                          # the toy's fixtures, under their own names
derived_is_a_fold = e2e.derived_is_a_fold


@pytest.fixture(autouse=True)
def _the_collector_as_pytest_has_it(monkeypatch):
    """`global-match` and `batch-decomp` tune the collector of the process their `main` runs in, which here is pytest's: `tune`
    is stood in for, and a command that reads it from anywhere else fails the test that ran it."""
    monkeypatch.setattr(gc_tuning, "tune", lambda: None)
    before = gc.get_threshold()
    yield
    after = gc.get_threshold()
    gc.set_threshold(*before)
    assert after == before, "a command's `main` tuned this process's collector: it does not read `verity_vllm.gc_tuning.tune`"


def _logged(caplog, module) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == module.__name__]


def _kept_elsewhere(oracle, tmp_path, name: str) -> str:
    """Moves `<oracle>/capture/<name>` out of the record's directory and returns where it is now."""
    kept = tmp_path / "kept-elsewhere"
    kept.mkdir(exist_ok=True)
    os.replace(os.path.join(oracle, "capture", name), kept / f"the-{name}")
    return str(kept / f"the-{name}")


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


def test_batch_decomp_reads_the_capture_log_that_its_log_option_names(memory, tmp_path, derived_is_a_fold):
    d, _ = e2e.fold_batched([e2e.R0, e2e.R1], tmp_path / "b2")
    args = ["batch-decomp", "--oracle", d, "--derived", e2e.fold_alone(e2e.R0, tmp_path / "LP3_T2"),
            "--derived", e2e.fold_alone(e2e.R1, tmp_path / "LP2_T1")]
    in_place, named = tmp_path / "in_place.json", tmp_path / "named.json"
    assert cli.main([*args, "--out", str(in_place)]) == 0
    log = _kept_elsewhere(d, tmp_path, "log.jsonl.gz")
    with pytest.raises(FileNotFoundError, match="capture/log.jsonl.gz"):        # the default path, which no longer has it
        cli.main([*args, "--out", str(tmp_path / "unread.json")])
    assert cli.main([*args, "--log", log, "--out", str(named)]) == 0
    a, b = json.loads(in_place.read_text()), json.loads(named.read_text())
    assert (a["log"], b["log"]) == (os.path.join(d, "capture", "log.jsonl.gz"), log) and a["verdict"] == b["verdict"] == "PASS"
    varies = ("log", "utc", "seconds")                                         # the path given, and when and how long each run was
    assert {k: v for k, v in b.items() if k not in varies} == {k: v for k, v in a.items() if k not in varies}
    assert b["engine_steps"] == 3 and [e["request_id"] for e in b["requests"]] == ["r0", "r1"]


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


def test_global_match_reads_the_capture_log_and_the_tokens_file_that_its_options_name(memory, tmp_path):
    """The checker of record on the toy's two requests, X-09 included: the document is the one the record gives with both files in
    place, and each file is missed by name of what it leaves undone when its option is not given."""
    gp, oracle = tgm.assemble([tgm.R0, tgm.R1], tmp_path / "record")
    gpp = tmp_path / "gp.json"
    gpp.write_text(json.dumps(gp))
    args = ["global-match", "--oracle", oracle, "--global-program", str(gpp), "--no-alternate"]

    def run(name: str, *options: str) -> tuple[int, dict]:
        out = tmp_path / f"{name}.json"
        rc = cli.main([*args, *options, "--out", str(out)])
        return rc, json.loads(out.read_text())
    rc, in_place = run("in_place")
    assert rc == 0 and in_place["verdict"] == "PASS" and tgm._check(in_place, "G6")["tokens_json"] is True, in_place["reasons"]
    tokens = _kept_elsewhere(oracle, tmp_path, "tokens.json")
    rc, no_tokens = run("no_tokens")                                           # the default path, which no longer has it
    assert rc == 3 and no_tokens["failed_checks"] == ["G6"] and tgm._check(no_tokens, "G6")["tokens_json"] is False
    log = _kept_elsewhere(oracle, tmp_path, "log.jsonl.gz")
    with pytest.raises(FileNotFoundError, match="capture/log.jsonl.gz"):
        run("no_log", "--tokens", tokens)
    rc, named = run("named", "--log", log, "--tokens", tokens)
    assert rc == 0 and named["log"] == log and in_place["log"] == os.path.join(oracle, "capture", "log.jsonl.gz")
    assert named["verdict"] == "PASS" and tgm._check(named, "G6")["tokens_json"] is True, named["reasons"]
    assert [(k["id"], k["status"]) for k in named["checks"]] == [(k["id"], k["status"]) for k in in_place["checks"]]
    assert named["x09"]["verdict"] == in_place["x09"]["verdict"] == "PASS"     # X-09 read the same log, by the same option
    assert [(e["request_id"], e["n_instances"]) for e in named["requests"]] == [(e["request_id"], e["n_instances"]) for e in in_place["requests"]]


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

on_workers = gmf.request_legs     # (items, what to run on each, the number of workers) -> the results in the items' order


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """The workers' result files go to the test's directory and the module's counters are the test's own; when the test is over
    the parent has removed every result file, a failed worker's too."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(gmf, "stats", collections.Counter())
    yield tmp_path
    assert list(tmp_path.glob("*.pkl")) == []


def _square_but_four(x: int) -> int:
    if x == 4:
        raise ValueError("no request 4")
    return x * x


def _refuse(x: int) -> int:
    raise KeyError(x)


def _die_at_four(x: int) -> int:
    if x == 4:
        os._exit(3)     # the worker is gone before it writes its result
    return x * x


def _fail_at_four_with_no_room_to_say_so(x: int) -> int:
    if x == 4:
        def no_room(*args, **kwargs):
            raise OSError(28, "No space left on device")
        pytest.MonkeyPatch().setattr(gmf.pickle, "dump", no_room)     # in the worker, which exits with it; the parent's is its own
        raise ValueError("no request 4")
    return x * x


def _the_worker_named(text: str) -> tuple[str, str]:
    """(the items in the worker's brackets, what is said of it), of a failure naming one worker that is not this process."""
    named, _, said = text.partition(") ")
    worker, _, items = named.partition(" (")
    assert worker.startswith("1 of 2 per-request workers failed: worker ") and int(worker.rsplit(" ", 1)[1]) != os.getpid()
    return items.rsplit(" ", 1)[1], said


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
        assert on_workers([3, 4], lambda x: x * x, 2) == [9, 16]
    left = sorted(str(f) for f in tmp_path.glob("*.pkl"))
    records = _logged(caplog, gmf)
    assert len(left) == 2 and [r.levelno for r in records] == [logging.WARNING, logging.WARNING]
    assert sorted(r.getMessage().split(" result file ")[1].split(" is left behind: ")[0] for r in records) == left
    assert all(r.getMessage().startswith("worker ") and "PermissionError(13" in r.getMessage() for r in records)


def test_a_request_that_raises_on_a_worker_fails_the_call_with_its_error_and_where_it_was_raised(results_dir):
    """Two items on two workers.  The first worker's result is read and not returned."""
    with pytest.raises(RuntimeError) as failed:
        on_workers([3, 4], _square_but_four, 2)
    text = str(failed.value)
    assert text.startswith("1 of 2 per-request workers failed: ValueError('no request 4')\nTraceback (most recent call last):\n")
    assert text.endswith('in _square_but_four\n    raise ValueError("no request 4")\nValueError: no request 4\n')
    assert list(gmf.stats.values()) == [1]     # the one result that a worker did hand back was counted


def test_every_failed_worker_is_named_in_the_workers_order(results_dir):
    """Three items on two workers, dealt in turn: the first worker has the first and third, and stops at the first."""
    with pytest.raises(RuntimeError) as failed:
        on_workers([3, 4, 5], _refuse, 2)
    text = str(failed.value)
    assert text.startswith("2 of 2 per-request workers failed: KeyError(3)\nTraceback (most recent call last):\n")
    assert "\nKeyError: 3\n | KeyError(4)\nTraceback (most recent call last):\n" in text and text.endswith("\nKeyError: 4\n")
    assert "KeyError(5)" not in text and not gmf.stats


def test_a_worker_that_dies_without_a_result_is_named_with_its_items_and_its_wait_status(results_dir):
    """The second worker exits with status 3 inside the request, so its result file is empty; 768 is that exit as `waitpid` gives it."""
    with pytest.raises(RuntimeError) as failed:
        on_workers([3, 4], _die_at_four, 2)
    assert _the_worker_named(str(failed.value)) == ("[1]", "left no result, wait status 768: EOFError('Ran out of input')")


def test_a_worker_that_cannot_write_its_error_says_so_on_stderr_and_is_named_by_the_parent(results_dir, capfd):
    """The request fails and the worker's write of that failure fails too: the worker says both on stderr and exits 1 (256 as
    `waitpid` gives it), and the parent finds the file the worker opened and left empty."""
    with pytest.raises(RuntimeError) as failed:
        on_workers([3, 4], _fail_at_four_with_no_room_to_say_so, 2)
    assert _the_worker_named(str(failed.value)) == ("[1]", "left no result, wait status 256: EOFError('Ran out of input')")
    out, err = capfd.readouterr()
    said, _, could_not = err.partition(" raised ValueError('no request 4') and could not write it to ")
    assert out == "" and said.startswith("global_match_fast: ") and said.rsplit(" ", 1)[1].isdigit()
    path, _, why = could_not.partition(": ")
    assert os.path.dirname(path) == str(results_dir) and why == "OSError(28, 'No space left on device')\n"


def test_an_item_whose_result_is_nothing_counts_as_missing(results_dir):
    """A worker's results are facts, one an item; nothing for an item is an incomplete set, not a list with a gap."""
    with pytest.raises(RuntimeError, match="^per-request workers returned an incomplete set of "):
        on_workers([3, 4], lambda x: None if x == 4 else x * x, 2)


def test_in_this_process_a_request_raises_its_own_error(results_dir):
    """With one worker asked for, or one item, nothing is forked and the request's error is the call's."""
    with pytest.raises(ValueError, match="^no request 4$"):
        on_workers([3, 4], _square_but_four, 1)
    with pytest.raises(KeyError):
        on_workers([4], _refuse, 2)
    assert on_workers([3, 5], _square_but_four, 1) == [9, 25] and not gmf.stats
