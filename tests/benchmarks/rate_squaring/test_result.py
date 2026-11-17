from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import result

HERE = Path(__file__).resolve().parents[3] / "benchmarks" / "rate_squaring"


def row(cc: str, which: str, g: int, threads: int, rate: float) -> dict:
    return {"cc": cc, "set": which, "bits": 512, "g": g, "threads": threads, "sq_per_s_sum": rate}


ROWS = [row("gcc", "one", 1, 1, 1.8e8), row("gcc", "cores", 1, 9, 1.6e9), row("clang", "mask", 2, 18, 1.75e9),
        row("gcc", "mask", 1, 18, 1.7e9)]


def written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict], errors: str = "") -> tuple[int, dict]:
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    if rows:
        (tmp_path / "time.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if errors:
        (tmp_path / "errors.txt").write_text(errors)
    (tmp_path / "gate-gcc-g1-k1.json").write_text('{"words": 384, "bad": 0, "ok": true}\n')
    code = result.record(tmp_path, "a processor")
    return code, json.loads((tmp_path / "result.json").read_text())


def test_the_fastest_timing_is_the_number_with_its_processor_and_cores(tmp_path, monkeypatch, capsys):
    code, doc = written(tmp_path, monkeypatch, ROWS)
    m, = doc["measurements"]
    assert code == 0 and doc["validation"]["status"] == "passed" and doc["validation"]["evidence"] == ["gate-gcc-g1-k1.json"]
    assert (m["name"], m["value"], m["unit"], m["mark"], m["target"]) == (result.METRIC, 1.75e9, "squarings a second", "measured", None)
    assert (m["cpu"], m["cores"], m["threads"], m["compiler"], m["vectors"], m["bits"]) == ("a processor", 9, 18, "clang", 2, 512)
    assert json.loads(capsys.readouterr().out)["value"] == 1.75e9


@pytest.mark.parametrize("rows, errors", [(ROWS, "clang g 2 failed its gate: not timed\n"), ([], "")])
def test_a_failed_gate_or_no_timing_writes_no_number_and_exits_5(tmp_path, monkeypatch, rows, errors):
    code, doc = written(tmp_path, monkeypatch, rows, errors)
    assert code == 5 and doc["validation"]["status"] == "failed" and not doc["measurements"]
    assert errors.strip() in doc["validation"]["detail"]


def test_the_metric_is_the_registered_one():
    metric, = tomllib.loads((HERE / "metrics.toml").read_text())["metric"]
    assert metric["id"] == result.METRIC and metric["unit"] == "squarings a second"
    assert "ifma.sh" in metric["command"] and "result.py" in (HERE / "ifma.sh").read_text()


def test_no_timing_on_one_thread_a_core_writes_no_number_and_exits_5(tmp_path, monkeypatch):
    code, doc = written(tmp_path, monkeypatch, [r for r in ROWS if r["set"] != "cores"])
    assert code == 5 and doc["validation"]["status"] == "failed" and not doc["measurements"]
    assert "no timing on one thread a core" in doc["validation"]["detail"]
