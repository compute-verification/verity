"""run_result.write: a benchmark's result.json under `research run`, and nothing outside one."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
