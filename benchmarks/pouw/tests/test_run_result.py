"""run_result.write: a benchmark's result.json under `research run`, and nothing outside one; and PoUW's two writers of a
run's result (it and `exact_kit.checks`) beside the harness kit's writer. The three write one result key for key, and a
measurement of the kit's holds more. They differ in what a run's second writer does: each of these two writes the result
whole, so the last one's stands, and the kit holds one validation a result and refuses the second."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest
from exact_kit.checks import Checks
from verity_bench import kit

POUW = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pouw_run_result", POUW / "run_result.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def _write(run_dir: Path, out: Path, ok: bool = True) -> dict:
    R.write(out, ok, "a checker", "a detail", [("items", 3, "count"), ("ratio", 1.5, "x")])
    return json.loads((run_dir / "result.json").read_text())


def test_a_run_gets_its_result_with_the_record_as_evidence(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    (run_dir / "out").mkdir(parents=True)
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", "abc123")
    doc = _write(run_dir, run_dir / "out" / "bench.json")
    assert (doc["schema"], doc["run_id"]) == ("research/result/v0.1", "r-test")
    assert doc["workload_fingerprint"] == {"source_sha": "abc123"}
    assert doc["validation"] == {"status": "passed", "checker": "a checker", "detail": "a detail",
                                 "evidence": ["out/bench.json"]}
    assert doc["measurements"] == [{"name": "items", "value": 3.0, "unit": "count"},
                                   {"name": "ratio", "value": 1.5, "unit": "x"}]
    assert doc["measurement_files"] == ["out/bench.json"] and doc["artifacts"] == []
    assert _write(run_dir, run_dir / "out" / "bench.json", ok=False)["validation"]["status"] == "failed"
    assert sorted(p.name for p in run_dir.iterdir()) == ["out", "result.json"]


def test_a_record_outside_the_run_directory_is_no_evidence(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    doc = _write(run_dir, tmp_path / "elsewhere" / "bench.json")
    assert doc["validation"]["evidence"] == [] and doc["measurement_files"] == []
    assert doc["validation"]["status"] == "passed" and len(doc["measurements"]) == 2


def test_outside_a_run_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    R.write(tmp_path / "bench.json", True, "a checker", "a detail", [])
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    R.write(tmp_path / "bench.json", True, "a checker", "a detail", [])
    assert list(tmp_path.iterdir()) == []


def _kits(run_dir: Path, ok: bool, checker: str, detail: str, record: str, measurements, **fingerprint) -> dict:
    """The same result through the kit's calls: what ran, the record as the measurement file and the evidence, the
    validation, and each measurement as the float the old writers write."""
    w = kit.writer(run_dir / "out")
    w.fingerprint(**fingerprint)
    w.measurement_file(record)
    w.validation("passed" if ok else "failed", checker, detail, [record])
    for name, value, unit in measurements:
        w.measurement(name, float(value), unit)
    return json.loads((run_dir / "result.json").read_text())


@pytest.fixture
def a_run(tmp_path, monkeypatch) -> Path:
    """A research run's directory, with its `out/`, on a host with no GPU."""
    run_dir = tmp_path / "run"
    (run_dir / "out").mkdir(parents=True)
    for name, value in (("RESEARCH_RUN_DIR", str(run_dir)), ("RESEARCH_RUN_ID", "r-test"), ("RESEARCH_SOURCE_SHA", "abc123"),
                        ("OUT", str(run_dir / "out"))):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("GPU_LEASE_UUID", raising=False)
    monkeypatch.setattr(kit.shutil, "which", lambda name: None)
    return run_dir


def _the_same_key_for_key(theirs: dict, ours: dict) -> None:
    """One result, key for key: the same keys, each the same value, and each measurement's name, value and unit. A
    measurement of the kit's holds more (its commit, run, interval, target, verdict, inputs and host stamp)."""
    assert ours.keys() == theirs.keys()
    assert {k: ours[k] for k in ours if k != "measurements"} == {k: theirs[k] for k in theirs if k != "measurements"}
    assert [{k: m[k] for k in ("name", "value", "unit")} for m in ours["measurements"]] == theirs["measurements"]
    assert all(isinstance(m["value"], float) for m in theirs["measurements"] + ours["measurements"])


def test_run_results_writer_and_the_kits_write_one_result_key_for_key(a_run):
    measurements = [("items", 3, "count"), ("ratio", 1.5, "x")]
    theirs = _write(a_run, a_run / "out" / "bench.json")
    (a_run / "result.json").unlink()
    ours = _kits(a_run, True, "a checker", "a detail", "out/bench.json", measurements)
    _the_same_key_for_key(theirs, ours)
    assert (theirs["workload_fingerprint"], theirs["measurement_files"], theirs["artifacts"]) == (
        {"source_sha": "abc123"}, ["out/bench.json"], [])


def test_exact_kits_writer_and_the_kits_write_one_result_key_for_key(a_run, tmp_path):
    script = tmp_path / "probe.py"
    script.write_text("# a script the verdict names\n")
    checks = Checks(argv=["--k", "4"], script=str(script), out=io.StringIO())
    checks.check("a", True)
    checks.check("b", False)
    assert checks.finish() == 1
    theirs = json.loads((a_run / "result.json").read_text())
    (a_run / "result.json").unlink()
    verdict = checks.verdict()
    ours = _kits(a_run, False, "exact_kit.checks", checks.summary(), "out/probe.verdict.json",
                 [(name, value, "count") for name, value in verdict["counts"].items()],
                 script=verdict["script"], args=verdict["args"], mutations_run=verdict["mutations_run"])
    _the_same_key_for_key(theirs, ours)
    assert theirs["workload_fingerprint"] == {"script": {"name": "probe.py", "sha256": checks.verdict()["script"]["sha256"]},
                                              "args": ["--k", "4"], "mutations_run": False, "source_sha": "abc123"}
    assert (theirs["measurement_files"], theirs["artifacts"]) == (["out/probe.verdict.json"], [])


def test_a_runs_second_writer_replaces_the_result_where_the_kit_refuses_a_second_validation(a_run):
    """`pearlc_k256_gate.sh`'s run is many processes that each write a whole result, and the run's is the last one's,
    the verdict's; `theory/run.py`'s is each script's and then its own. Through the kit the first one's would stand."""
    _write(a_run, a_run / "out" / "census.json")
    R.write(a_run / "out" / "verdict.json", False, "the verdict", "a part fails", [("parts", 2, "count")])
    last = json.loads((a_run / "result.json").read_text())
    assert (last["validation"]["checker"], last["validation"]["status"]) == ("the verdict", "failed")
    assert last["measurement_files"] == ["out/verdict.json"] and [m["name"] for m in last["measurements"]] == ["parts"]
    (a_run / "result.json").unlink()
    _kits(a_run, True, "a checker", "a detail", "out/census.json", [("items", 3, "count")])
    with pytest.raises(ValueError, match="already holds a validation"):
        _kits(a_run, False, "the verdict", "a part fails", "out/verdict.json", [("parts", 2, "count")])
    assert json.loads((a_run / "result.json").read_text())["validation"]["checker"] == "a checker"


def test_outside_a_run_the_kit_writes_out_s_result_where_the_old_writers_write_nothing(a_run, tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCH_RUN_DIR")
    out = tmp_path / "local"
    R.write(out / "bench.json", True, "a checker", "a detail", [("items", 3, "count")])
    assert not out.exists()
    kit.writer(out).measurement("items", 3, "count")
    assert [p.name for p in out.iterdir()] == ["result.json"] and not (a_run / "result.json").exists()
