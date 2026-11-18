from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import result

HERE = Path(__file__).resolve().parents[3] / "calibration" / "bandwidth"


def row(label: str, kernel: str, threads: int, gb_per_s: float, wrong: int = 0) -> dict:
    return {"label": label, "kernel": kernel, "threads": threads, "gb_per_s": gb_per_s, "checked": 4096 * threads,
            "wrong": wrong, "misplaced": 0}


def written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict]) -> tuple[int, dict]:
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    if rows:
        (tmp_path / "write_host.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    code = result.record(tmp_path, "a processor")
    return code, json.loads((tmp_path / "result.json").read_text())


def test_the_fastest_record_is_the_number_with_its_processor_and_cores(tmp_path, monkeypatch, capsys):
    rows = [row("cores1", "nt", 1, 27.5), row("cores8", "st", 8, 91.25), row("node0-all", "nt", 24, 80.0)]
    code, doc = written(tmp_path, monkeypatch, rows)
    m, = doc["measurements"]
    assert code == 0 and doc["validation"]["status"] == "passed" and doc["validation"]["evidence"] == ["write_host.jsonl"]
    assert (m["name"], m["value"], m["unit"], m["mark"], m["target"]) == (result.METRIC, 91.25, "GB a second", "measured", None)
    assert (m["cpu"], m["cores"], m["threads"], m["kernel"], m["set"]) == ("a processor", 24, 8, "st", "cores8")
    assert json.loads(capsys.readouterr().out)["value"] == 91.25


@pytest.mark.parametrize("rows", [[row("cores1", "nt", 1, 27.5), row("cores2", "nt", 2, 99.0, wrong=1)], []])
def test_a_wrong_byte_or_no_record_writes_no_number_and_exits_5(tmp_path, monkeypatch, rows):
    code, doc = written(tmp_path, monkeypatch, rows)
    assert code == 5 and doc["validation"]["status"] == "failed" and not doc["measurements"]


def test_the_metric_is_the_registered_one():
    metric, = tomllib.loads((HERE / "metrics.toml").read_text())["metric"]
    assert metric["id"] == result.METRIC and metric["unit"] == "GB a second"
    assert metric["command"].endswith("-- bash calibration/bandwidth/run.sh write_host")
    assert 'result.py" write_host "$O"' in (HERE / "run.sh").read_text()


@pytest.mark.parametrize("probe, first, rest", [("chip", "chip.json", ["ur.jsonl", "build.txt"]),
                                                ("write_gpu", "write.jsonl", ["build.txt", "rc.txt"])])
@pytest.mark.parametrize("rc, wrote, status", [(0, True, "passed"), (3, True, "failed"), (0, False, "failed")])
def test_a_device_probe_writes_its_programs_check_as_the_validation_and_no_number(tmp_path, monkeypatch, probe, first,
                                                                                 rest, rc, wrote, status):
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    for f in ([first] if wrote else []) + rest:
        (out / f).write_text("{}\n")
    assert result.record_probe(probe, out, rc) == 0
    doc = json.loads((tmp_path / "result.json").read_text())
    assert doc["measurements"] == [] and doc["validation"]["status"] == status
    assert doc["validation"]["evidence"] == ([f"out/{first}"] if wrote else [])
    assert doc["measurement_files"] == [f"out/{f}" for f in ([first] if wrote else []) + rest]
    assert doc["validation"]["detail"].startswith(f"{result.PROBES[probe][0]} exited {rc}: ")
