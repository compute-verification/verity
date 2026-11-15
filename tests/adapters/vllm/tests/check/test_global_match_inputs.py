"""verity_vllm.check.match.global_match: where its command takes the declaration from and what it refuses to start with, and what
the checker says of a criterion or a declaration it does not know, of the other criterion's failure, and of a component dir.

`main` runs over a stand-in for its checker where the lines under test are `main`'s own; the checker runs on the toy's two
requests of `test_global_match.py`, in the record implementation and in the fast one."""
from __future__ import annotations

import dataclasses
import gc
import json
from typing import Any

import pytest

from tests.check import test_batch_decomp_e2e as e2e
from tests.check import test_global_match as tgm
from verity_vllm import gc_tuning
from verity_vllm.check.match import batch_decomp as bd
from verity_vllm.check.match import global_match as gm
from verity_vllm.check.match import global_match_fast as gmf
from verity_vllm.config import UsageError

memory = e2e.memory                          # the toy's fixture, under its own name
IMPLS = pytest.mark.parametrize("impl", gm.IMPLS)
OTHER_SCHEMA = "cb-a/global-program/v0"


@pytest.fixture(autouse=True)
def _the_collector_as_pytest_has_it(monkeypatch):
    """`global-match` tunes the collector of the process its `main` runs in, which here is pytest's: `tune` is stood in for."""
    monkeypatch.setattr(gc_tuning, "tune", lambda: None)
    before = gc.get_threshold()
    yield
    assert gc.get_threshold() == before


@pytest.fixture
def stand_ins(monkeypatch):
    """The three things `main` hands the declaration through, each replaced by one that says it was called and with what: the
    Build artifact's reader, the mock's assembler and the checker."""
    calls: list[tuple] = []

    def from_gprog(artifact, workload=None):
        calls.append(("from_gprog", artifact, workload))
        return {"schema": gm.GP_SCHEMA, "requests": [], "read_from": "the Build's artifact"}

    def mock_global_program(workload, components):
        calls.append(("mock_global_program", workload, components))
        return {"schema": gm.GP_SCHEMA, "requests": [], "read_from": "the workload and the components"}

    def check(gp, oracle, **options):
        calls.append(("check", gp, oracle, options))
        return {"verdict": "PASS", "comparator": gm.COMPARATOR, "rules_version": gm.RULES_VERSION, "checks": [], "requests": [],
                "seconds": 0.0}
    monkeypatch.setattr(gm, "from_gprog", from_gprog)
    monkeypatch.setattr(gm, "mock_global_program", mock_global_program)
    monkeypatch.setattr(gm, "check", check)
    monkeypatch.setenv(gm.IMPL_ENV, "record")
    return calls


# ------------------------------------------------------------------------------------------------------------------ the command


def test_decomp_out_without_the_x09_run_is_a_usage_error_before_anything_is_read(stand_ins, tmp_path):
    gpp = tmp_path / "no-such-declaration.json"
    with pytest.raises(UsageError) as refused:
        gm.main(gm.Options(oracle=str(tmp_path / "no-record"), global_program=str(gpp), no_x09=True, decomp_out=str(tmp_path / "x09.json")))
    assert str(refused.value) == "--decomp-out needs the X-09 run (drop --no-x09)" and stand_ins == []


def test_with_no_declaration_file_and_no_workload_the_command_is_a_usage_error(stand_ins, tmp_path):
    with pytest.raises(UsageError) as refused:
        gm.main(gm.Options(oracle=str(tmp_path / "no-record"), component=["a-component"]))
    assert str(refused.value) == "--workload is required without --global-program" and stand_ins == []


SOURCES: dict[str, tuple[dict[str, Any], tuple | None, str]] = {
    "the declaration's own file, whatever else is named": (
        {"global_program": "FILE", "gprog": "the-artifact.json", "workload": "the-workload.json", "component": ["c0"]},
        None, "its own file"),
    "the Build's artifact, read with the workload named": (
        {"gprog": "the-artifact.json", "workload": "the-workload.json", "component": ["c0"]},
        ("from_gprog", "the-artifact.json", "the-workload.json"), "the Build's artifact"),
    "the Build's artifact alone": (
        {"gprog": "the-artifact.json"}, ("from_gprog", "the-artifact.json", None), "the Build's artifact"),
    "the mock, of the workload and the components in the order given": (
        {"workload": "the-workload.json", "component": ["c1", "c0"]},
        ("mock_global_program", "the-workload.json", ["c1", "c0"]), "the workload and the components"),
}


@pytest.mark.parametrize("source", list(SOURCES))
def test_the_declaration_is_its_own_file_else_the_builds_artifact_else_the_mock(source: str, stand_ins, tmp_path):
    """One reader is called, the first that an option names, and what it returns is what the checker is handed and what the
    command writes beside its document."""
    options, reader, read_from = SOURCES[source]
    if "global_program" in options:
        gpp = tmp_path / "declaration.json"
        gpp.write_text(json.dumps({"schema": gm.GP_SCHEMA, "requests": [], "read_from": "its own file"}))
        options = {**options, "global_program": str(gpp)}
    out = tmp_path / "gm.json"
    rc = gm.main(gm.Options(oracle="the-oracle", out=str(out), **options))
    *readers, (name, gp, oracle, passed) = stand_ins
    assert rc == 0 and readers == ([reader] if reader else []) and (name, oracle) == ("check", "the-oracle")
    assert gp == {"schema": gm.GP_SCHEMA, "requests": [], "read_from": read_from}
    assert json.loads((tmp_path / "gm_global_program.json").read_text()) == gp
    assert passed["run_x09"] is True and passed["record_alternate"] is True and passed["within_step"] == gm.DEFAULT_WITHIN_STEP


def test_the_checker_is_handed_what_the_commands_options_say(stand_ins, tmp_path):
    gpp = tmp_path / "declaration.json"
    gpp.write_text(json.dumps({"schema": gm.GP_SCHEMA, "requests": []}))
    rc = gm.main(gm.Options(oracle="the-oracle", global_program=str(gpp), log="a-log", tokens="some-tokens", tokens_optional=True,
                            no_x09=True, within_step="sequence", no_alternate=True, derived=["d1", "d0"]))
    ((name, _gp, oracle, passed),) = stand_ins
    assert rc == 0 and (name, oracle) == ("check", "the-oracle")
    assert passed == {"log_path": "a-log", "tokens_path": "some-tokens", "tokens_optional": True, "run_x09": False,
                      "within_step": "sequence", "record_alternate": False, "_loaded": {gm.X09_EXTRA_DERIVED: ["d1", "d0"]}}


# ------------------------------------------------------------------------------------------------------------------ the checker


@IMPLS
def test_a_criterion_the_checker_does_not_know_is_refused_before_the_record_is_read(impl: str, tmp_path, monkeypatch):
    monkeypatch.setenv(gm.IMPL_ENV, impl)
    with pytest.raises(ValueError) as refused:
        gm.check({"schema": gm.GP_SCHEMA, "requests": []}, str(tmp_path / "no-record"), within_step="order")
    assert str(refused.value) == "within_step must be one of ('sequence', 'dag'), got 'order'"


@IMPLS
def test_a_declaration_of_another_schema_fails_by_name_with_no_check_run(impl: str, memory, tmp_path, monkeypatch):
    """The checker reads the record, states what the declaration says of itself and fails on its schema, by name, with no check
    run and no request compared.

    Today's answer for the command, not what it should do: it writes that document and the declaration beside it, and then
    stops with a KeyError where it makes its brief, which reads the seconds a document that ends this early does not hold.  It
    should say its brief and exit 3, as it does for any other failed Match."""
    monkeypatch.setenv(gm.IMPL_ENV, impl)
    gp, oracle = tgm.assemble([tgm.R0, tgm.R1], tmp_path / "record")
    other = {**gp, "schema": OTHER_SCHEMA}
    res = gm.check(other, oracle)
    assert (res["verdict"], res["checks"], res["requests"]) == ("FAIL", [], [])
    assert res["reasons"] == [f"global program schema '{OTHER_SCHEMA}' != '{gm.GP_SCHEMA}'"]
    assert res["global_program"]["schema"] == OTHER_SCHEMA and res["global_program"]["n_requests"] == 2
    assert gm.check({"schema": gm.GP_SCHEMA[:-1]}, oracle)["reasons"] == [f"global program schema '{gm.GP_SCHEMA[:-1]}' != '{gm.GP_SCHEMA}'"]
    gpp, out = tmp_path / "other.json", tmp_path / "gm.json"
    gpp.write_text(json.dumps(other))
    with pytest.raises(KeyError, match="^'seconds'$"):
        gm.main(gm.Options(oracle=oracle, global_program=str(gpp), out=str(out)))
    written = json.loads(out.read_text())
    assert (written["verdict"], written["reasons"], written["impl"]["name"]) == ("FAIL", res["reasons"], impl)
    assert json.loads((tmp_path / "gm_global_program.json").read_text()) == other


@IMPLS
def test_the_other_criterions_failure_is_recorded_beside_the_verdict_and_does_not_change_it(impl: str, memory, tmp_path, monkeypatch):
    """The verdict of record is made before the other criterion runs.  An error of that run is kept as the alternate verdict
    ERROR with the error's first 300 characters, and the document is the one the criterion of record gives."""
    monkeypatch.setenv(gm.IMPL_ENV, impl)
    gp, oracle = tgm.assemble([tgm.R0, tgm.R1], tmp_path / "record")
    checker, asked = gm.check, []

    def the_other_criterion_raises(*args, **options):
        asked.append((options["within_step"], options["run_x09"], options["record_alternate"]))
        raise RuntimeError("the other criterion raised, " + "at length " * 40)
    alone = checker(gp, oracle, run_x09=False, record_alternate=False)
    monkeypatch.setattr(gm, "check", the_other_criterion_raises)
    res = checker(gp, oracle, run_x09=False)
    said = repr(RuntimeError("the other criterion raised, " + "at length " * 40))
    assert len(said) > 300 and res["alternate_verdict"] == {"criterion": "sequence", "verdict": "ERROR", "error": said[:300]}
    assert asked == [("sequence", False, False)] and res["within_step_criterion"] == "dag"
    assert (res["verdict"], res["failed_checks"], res["reasons"]) == ("PASS", [], []) == (alone["verdict"], alone["failed_checks"], alone["reasons"])
    assert [(k["id"], k["status"]) for k in res["checks"]] == [(k["id"], k["status"]) for k in alone["checks"]]
    assert [p["name"] for p in res["phases"]] == [*(p["name"] for p in alone["phases"]), "alternate_criterion"]
    res = checker(gp, oracle, run_x09=False, within_step="sequence")
    assert res["alternate_verdict"]["criterion"] == "dag" and asked[1:] == [("dag", False, False)]


@pytest.mark.parametrize("fast", [False, True], ids=gm.IMPLS)
def test_a_component_dir_loads_as_a_fold_only_when_it_holds_a_folds_program(fast: bool, tmp_path, monkeypatch):
    """The operations the checker runs load a component through its derived loader, but for a dir that holds `program.json`, a
    fold standing in for a component, which loads as the fold it is."""
    loaded: list[tuple] = []
    made = gmf.ops() if fast else bd.record_ops()
    stub = dataclasses.replace(made, load_oracle=lambda d, label: loaded.append(("as a fold", d, label)),
                               load_derived=lambda d, label: loaded.append(("as derived", d, label)))
    monkeypatch.setattr(gmf if fast else bd, "ops" if fast else "record_ops", lambda: stub)
    derived, fold = tmp_path / "derived", tmp_path / "fold"
    derived.mkdir()
    fold.mkdir()
    (fold / "program.json").write_text("{}")
    ops = gm.match_ops(fast)
    ops.load_derived(str(derived), "a component")
    ops.load_derived(str(fold), "a fold in its place")
    assert loaded == [("as derived", str(derived), "a component"), ("as a fold", str(fold), "a fold in its place")]
    assert ops.load_oracle is stub.load_oracle
