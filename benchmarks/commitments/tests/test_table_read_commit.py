"""table_read_commit keeps what it measured: each arm's result is on disk when the arm ends, whatever stops the run after."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from research import result as research_result

ROOT = Path(__file__).resolve().parents[3]
SMALL = ["--reads", "4", "--layers", "2", "--reps", "1"]


@pytest.fixture
def bench():
    spec = importlib.util.spec_from_file_location("table_read_commit", ROOT / "benchmarks" / "commitments" / "table_read_commit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _only_batch_1(bench, monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    plan = bench.host_plan

    def host_plan(batch: int, *args):
        if batch != 1:
            raise error
        return plan(batch, *args)

    monkeypatch.setattr(bench, "host_plan", host_plan)


def test_a_run_stopped_after_one_arm_leaves_that_arms_result_on_disk(bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _only_batch_1(bench, monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        bench.main([*SMALL, "--batch", "1,2,3", "--arms", "plan", "--out", str(tmp_path)])
    doc = json.loads((tmp_path / "table_read_commit.json").read_text())
    assert list(doc["arms"]) == ["plan.1"] and doc["pending"] == ["plan.2", "plan.3"]
    arm = doc["arms"]["plan.1"]
    assert arm["status"] == "done" and arm["result"]["tables"] == 2 * len(bench.ROWS) and arm["result"]["rows"] == 8 * len(bench.ROWS)
    assert arm["result"]["plan_ms"] > 0 and arm["start"].endswith("Z") and arm["end"].endswith("Z")
    assert not (tmp_path / "result.json").exists() and not list(tmp_path.glob("*.tmp"))


def test_an_arm_that_fails_is_recorded_and_the_arms_after_it_run(bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    _only_batch_1(bench, monkeypatch, ValueError("no layout"))
    assert bench.main([*SMALL, "--batch", "2,1", "--arms", "plan", "--out", str(tmp_path)]) == 1
    doc = json.loads((tmp_path / "table_read_commit.json").read_text())
    assert {n: a["status"] for n, a in doc["arms"].items()} == {"plan.2": "failed", "plan.1": "done"} and doc["pending"] == []
    assert doc["arms"]["plan.2"]["error"] == "ValueError: no layout"
    check = research_result.check(tmp_path, None)
    assert check.valid, check.errors
    assert check.validation_status == "failed" and json.loads(check.doc["validation"]["detail"])["failed"] == ["plan.2"]
    assert [m["name"] for m in check.doc["measurements"]] == ["plan.1.plan_ms", "workload_wall"]
    log = [ln.split(" table_read_commit: ")[1] for ln in capsys.readouterr().err.splitlines() if " table_read_commit: " in ln]
    assert log[0] == "arm plan.2 start" and log[1].startswith("arm plan.2 failed in ") and log[1].endswith(": ValueError: no layout")
    assert log[2] == "arm plan.1 start" and log[3].startswith("arm plan.1 done in ")


def test_a_whole_run_passes_and_arms_selects_in_the_arms_order(bench, tmp_path: Path) -> None:
    assert bench.main([*SMALL, "--batch", "1", "--arms", "plan", "--out", str(tmp_path)]) == 0
    check = research_result.check(tmp_path, None)
    assert check.valid, check.errors
    assert check.validation_status == "passed" and check.doc["validation"]["checker"] == "every arm ended done"
    assert bench.parse(["--arms", "cpu,tables.twin,plan,gate"]).arms == ["gate", "tables.twin", "plan", "cpu"]
    for bad in ("beside", "hash", "tables"):
        with pytest.raises(SystemExit):
            bench.parse(["--arms", bad])


def test_the_checker_names_only_the_checks_of_the_arms_done(bench) -> None:
    step = "a small step's device and CPU-twin roots equal row_tables.step_roots"
    strings = "is history.check's over its strings"
    assert bench.checker({"plan.1": {}, "cpu": {}}) == "every arm ended done"
    assert bench.checker({"gate": {"twin": True}}) == f"every arm ended done; {step.replace('device and ', '')}"
    assert bench.checker({"gate": {"twin": True, "device": False}, "tables.python": {}}) == \
        f"every arm ended done; {step}; each history table's root (python) {strings}"
    assert bench.checker({"tables.twin": {}}) == f"every arm ended done; each history table's root (twin) {strings}"
    assert bench.checker({"tables.device": {}, "tables.twin": {}}) == \
        f"every arm ended done; each history table's root (twin, device) {strings}; " \
        "the device table's root is its CPU twin's over its rows under its salts"


def test_the_tables_gate_holds_each_check_and_the_device_tables_twin(bench) -> None:
    assert bench.gates({"plan.1": {}}) == {}
    assert bench.gates({"tables.twin": {"check": True}, "tables.device": {"check": True, "twin": True}}) == {"tables": True}
    assert bench.gates({"tables.twin": {"check": True}, "tables.device": {"check": True, "twin": False}}) == {"tables": False}
    assert bench.gates({"tables.python": {"check": False}}) == {"tables": False}
