"""A number reaches the Metrics tab once, from the run that measured it: `kit.report(..., from_run=RUN)` of another run's
record prints and writes its document and writes no measurement, and each measurement a research run writes holds the
command the run was launched as."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kit_standins import no_gpu as no_gpu
from verity_bench import kit

THIS, OTHER = "r00000000-000000-aaaa", "r00000000-000000-bbbb"
#: a job.json's command as research writes it: its slot wrapper, then the run's own words, already redacted
COMMAND = ["env", "CHECK_SLOTS=/slots", "python3", "slot.py", "--", "nice", "-n", "19", "uv", "run", "python",
           "benchmarks/compute/run.py", "--from", OTHER]
MEASURED = [{"name": "compute/share", "value": 0.75, "unit": "ratio"}]


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch, no_gpu) -> Path:
    """A research run around the command: its directory, its id and its job.json."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "job.json").write_text(json.dumps({"run": {"id": THIS}, "execution": {"command": COMMAND}}))
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run))
    monkeypatch.setenv("RESEARCH_RUN_ID", THIS)
    monkeypatch.delenv("OUT", raising=False)
    return run


def test_a_reprint_of_another_runs_record_prints_and_writes_its_document_and_no_measurement(run_dir, capsys):
    assert kit.report("compute", "compute: 0.75", {"value": 0.75, "run": OTHER}, MEASURED, from_run=OTHER) == 0
    out, err = capsys.readouterr()
    assert out.splitlines() == ["compute: 0.75", json.dumps({"value": 0.75, "run": OTHER})]
    assert err.strip() == kit.kit_report.REPRINT.format(name="compute", run=OTHER)
    assert json.loads((run_dir / "out" / "compute.json").read_text()) == {"value": 0.75, "run": OTHER}
    assert not (run_dir / kit.RESULT).exists()


def test_a_command_reading_its_own_runs_record_writes_its_measurements_with_the_runs_command(run_dir, capsys):
    """compute's serve step reads the record of the run it is in, and so is no reprint."""
    assert kit.report("compute", "compute: 0.75", {"value": 0.75, "run": THIS}, MEASURED, from_run=THIS) == 0
    assert capsys.readouterr().err == ""
    (m,) = json.loads((run_dir / kit.RESULT).read_text())["measurements"]
    assert (m["name"], m["value"], m["run"], m["command"]) == ("compute/share", 0.75, THIS, COMMAND)


def test_from_run_is_a_run_id(run_dir):
    for bad in ("", 7):
        with pytest.raises(ValueError, match="from_run names the run whose record was read"):
            kit.report("compute", "compute: 0.75", {"value": 0.75}, MEASURED, from_run=bad)  # type: ignore[arg-type]


def test_a_measurement_holds_no_command_outside_a_research_run_or_where_job_json_gives_none(tmp_path, monkeypatch, no_gpu):
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    assert kit.run_command() is None
    assert "command" not in kit.writer(tmp_path / "a").measurement("x", 1, "s", mark="measured")
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run))
    for job in (None, {}, {"execution": {}}, {"execution": {"command": "uv run x"}}, {"execution": {"command": [1]}}):
        if job is not None:
            (run / "job.json").write_text(json.dumps(job))
        assert kit.run_command() is None


def test_an_extra_key_cant_stand_in_for_the_runs_command(tmp_path, monkeypatch, no_gpu):
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    with pytest.raises(ValueError, match="extra key 'command'"):
        kit.writer(tmp_path).measurement("x", 1, "s", mark="measured", extra={"command": ["uv", "run", "x"]})
