"""A trial run (`VERITY_BENCH_TRIAL`): its measurements go beside its result, which says trial and holds none, so none
reaches a metric's line; without the variable a result is written byte for byte as before."""

from __future__ import annotations

import hashlib
import json

import pytest
from kit_standins import RUN, _result
from kit_standins import no_gpu as no_gpu
from verity_bench import kit, overhead_node

#: the sha256 of the result `_write` wrote before trials existed, its measurement holding its metric's version
BEFORE = "0e72ba6f4628fcae17f495c6860e18569eac0bd6a6afb921b7ada62a0edb56e4"


def _write(out):
    w = kit.Writer(out, "r20261117-200000-0001", {"commit": "c0ffee", "card": "none"}, "bench-result/v1", ["bash", "job.sh"])
    w.fingerprint(script="job.sh")
    w.measurement("benchmarks/x", 1.5, "s", mark="measured", member="m-h200", extra={"n": 3})
    w.validation("passed", "job.sh", "held")


@pytest.fixture
def said(monkeypatch):
    """What the writer leaves to say at the process's exit."""
    registered: list = []
    monkeypatch.setattr(kit.atexit, "register", lambda f, *args: registered.append((f, args)))
    monkeypatch.setattr(kit, "_TRIALS_SAID", set())
    return registered


@pytest.mark.parametrize("value", [None, "", "0"])
def test_without_a_trial_a_result_is_written_byte_for_byte_as_before(tmp_path, monkeypatch, value):
    for name in ("RESEARCH_RUN_DIR", "RESEARCH_SOURCE_SHA", kit.TRIAL):
        monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv(kit.TRIAL, value)
    _write(tmp_path)
    assert hashlib.sha256((tmp_path / kit.RESULT).read_bytes()).hexdigest() == BEFORE
    assert [p.name for p in tmp_path.iterdir()] == [kit.RESULT]


def test_a_trial_is_the_variable_set_to_anything_but_0():
    assert [kit.is_trial(env) for env in ({}, {kit.TRIAL: ""}, {kit.TRIAL: "0"}, {kit.TRIAL: "1"}, {kit.TRIAL: "yes"})] == [
        False, False, False, True, True]
    assert kit.trial_attempt({"params": {"env": {kit.TRIAL: "1"}}})
    assert not kit.trial_attempt({"params": {"env": {"VERITY_BENCH_VENV": "/v"}}}) and not kit.trial_attempt({})


def test_a_trial_writes_its_measurements_beside_its_result_and_none_in_it(tmp_path, no_gpu, monkeypatch, capsys, said):
    from research import result as research_result

    run_dir = tmp_path / "run"
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.setenv(kit.TRIAL, "1")
    w = kit.writer(run_dir / "out", "bench-result/v1")
    w.fingerprint(script="job.sh")
    wall = w.measurement("wall", 1.5, "s", mark="measured", member="m-h200")
    gamma = kit.writer(tmp_path / "elsewhere").measurement("gamma", 0.5, "ratio", mark="measured")
    block = w.validation("passed", "job.sh", "held")

    result = _result(run_dir)
    assert result["measurements"] == [] and result["trial"] is True and result["kind"] == "bench-result/v1"
    assert result["workload_fingerprint"]["script"] == "job.sh" and result["validation"] == block
    check = research_result.check(run_dir, RUN)
    assert check.valid and check.validation_status == "passed"
    trial = run_dir / "out" / kit.TRIAL_FILE
    assert json.loads(trial.read_text()) == [wall, gamma]
    plain = kit.Writer(tmp_path / "plain", RUN, w.stamp, trial=False)
    assert plain.measurement("wall", 1.5, "s", mark="measured", member="m-h200") == wall
    with pytest.raises(ValueError, match="already holds a measurement named wall"):
        w.measurement("wall", 2.0, "s", mark="measured", member="m-h200")

    assert len(said) == 1
    f, args = said[0]
    f(*args)
    assert capsys.readouterr().err == (f"a trial run (VERITY_BENCH_TRIAL): its 2 measurements are in {trial}, and none "
                                       "reaches a metric's line\n")


def test_a_trials_whole_entries_go_beside_its_result_too(tmp_path, no_gpu, monkeypatch, said):
    run_dir, out = tmp_path / "r1", tmp_path / "r1" / "out"
    out.mkdir(parents=True)
    monkeypatch.setenv(kit.TRIAL, "1")
    whole = {"name": "cost", "value": None, "unit": "tokens/s", "run": "another", "rung": 3, "inputs": ["a load"]}
    overhead_node.write_result(run_dir, "r1", out, "record.json", True, "said", {"door": "cost"}, [whole],
                               measured=[{"name": "C", "value": 512, "unit": "m"}])
    result = _result(run_dir)
    assert result["measurements"] == [] and result["trial"] is True and result["validation"]["status"] == "passed"
    held, own = json.loads((run_dir / kit.TRIAL_FILE).read_text())
    assert held == whole and (own["name"], own["value"]) == ("C", 512)
