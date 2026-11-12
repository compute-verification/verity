"""run_result.write: a benchmark's result.json under `research run`, and nothing outside one; and PoUW's two writers of a
run's result (it and `exact_kit.checks`) beside the harness kit's writer, which they retire onto once it writes what they
write."""
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


#: what PoUW's two writers put in a run's result and the kit's writer does not, so neither retires onto the kit until it
#: writes them
KIT_LEAVES_OUT = ("workload_fingerprint", "measurement_files", "artifacts")


def _kits(run_dir: Path, ok: bool, checker: str, detail: str, evidence: list[str], measurements) -> dict:
    w = kit.writer(run_dir / "out")
    w.validation("passed" if ok else "failed", checker, detail, evidence)
    for name, value, unit in measurements:
        w.measurement(name, value, unit)
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


def _the_same_but_for_the_named(theirs: dict, ours: dict) -> None:
    """Key for key the same result, but for `KIT_LEAVES_OUT` and each measurement's other keys in the kit's (its commit,
    run, interval, target, verdict, inputs and host stamp); a value given as an int stays one in the kit's, where the old
    writers write a float."""
    assert ours.keys() == theirs.keys() - set(KIT_LEAVES_OUT)
    assert {k: ours[k] for k in ours if k != "measurements"} == {k: theirs[k] for k in ours if k != "measurements"}
    assert [{k: m[k] for k in ("name", "value", "unit")} for m in ours["measurements"]] == theirs["measurements"]
    assert all(isinstance(m["value"], float) for m in theirs["measurements"])


def test_run_results_writer_and_the_kits_write_one_result_but_for_the_named_differences(a_run):
    measurements = [("items", 3, "count"), ("ratio", 1.5, "x")]
    theirs = _write(a_run, a_run / "out" / "bench.json")
    (a_run / "result.json").unlink()
    ours = _kits(a_run, True, "a checker", "a detail", ["out/bench.json"], measurements)
    _the_same_but_for_the_named(theirs, ours)
    assert [type(m["value"]) for m in ours["measurements"]] == [int, float]
    assert (theirs["workload_fingerprint"], theirs["measurement_files"], theirs["artifacts"]) == (
        {"source_sha": "abc123"}, ["out/bench.json"], [])


def test_exact_kits_writer_and_the_kits_write_one_result_but_for_the_named_differences(a_run, tmp_path):
    script = tmp_path / "probe.py"
    script.write_text("# a script the verdict names\n")
    checks = Checks(argv=["--k", "4"], script=str(script), out=io.StringIO())
    checks.check("a", True)
    checks.check("b", False)
    assert checks.finish() == 1
    theirs = json.loads((a_run / "result.json").read_text())
    (a_run / "result.json").unlink()
    counts = checks.verdict()["counts"]
    ours = _kits(a_run, False, "exact_kit.checks", checks.summary(), ["out/probe.verdict.json"],
                 [(name, value, "count") for name, value in counts.items()])
    _the_same_but_for_the_named(theirs, ours)
    assert theirs["workload_fingerprint"] == {"script": {"name": "probe.py", "sha256": checks.verdict()["script"]["sha256"]},
                                              "args": ["--k", "4"], "mutations_run": False, "source_sha": "abc123"}
    assert (theirs["measurement_files"], theirs["artifacts"]) == (["out/probe.verdict.json"], [])


def test_outside_a_run_the_kit_writes_out_s_result_where_the_old_writers_write_nothing(a_run, tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCH_RUN_DIR")
    out = tmp_path / "local"
    R.write(out / "bench.json", True, "a checker", "a detail", [("items", 3, "count")])
    assert not out.exists()
    kit.writer(out).measurement("items", 3, "count")
    assert [p.name for p in out.iterdir()] == ["result.json"] and not (a_run / "result.json").exists()
