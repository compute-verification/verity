"""`verity bench overhead table`: the table, and under a research run its metrics."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest
from overhead_standins import REPO, Store, stored_points, timed
from overhead_standins import scheme as scheme
from overhead_standins import stamped as stamped
from verity_bench import kit
from verity_bench import overhead as O
from verity_bench import overhead_records as R

# ---- the command ------------------------------------------------------------------------------------------------------------------


def test_verity_bench_overhead_is_the_door_s_runner(tmp_path, monkeypatch, capsys):
    """`verity bench overhead` runs benchmarks/overhead/run.py, which is `verity-bench overhead`."""

    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "empty"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(Store(tmp_path).config))
    monkeypatch.setattr(sys, "argv", ["run.py", "table", "--json", "--local-only"])
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(REPO / "benchmarks" / "overhead" / "run.py"), run_name="__main__")
    assert done.value.code == 0
    table = json.loads(capsys.readouterr().out)
    assert table["rows"] == [] and table["census"]["world_flop_per_year"] == 9.36e30


def test_under_a_research_run_the_table_records_its_metrics_and_prints_the_same(tmp_path, monkeypatch, capsys, scheme):
    """Under a research run, `verity bench overhead table` writes run.py's metrics into the run's result through the kit:
    each the γ of record of its row, measured, with its interval, the scheme gate as its target, its run and points' art,
    and its note; with no γ of record, no value and why, and a failed validation naming it. What it prints is the same in
    a run as outside one, where it writes nothing."""

    s = Store(tmp_path)
    art = stored_points(tmp_path, s, "r20261104-020000-0001")
    run = timed(tmp_path, s, art, "pearl-plus", "r20261104-030000-0002")
    capsys.readouterr()
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "producer"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261114-000000-0000")
    runner = REPO / "benchmarks" / "overhead" / "run.py"
    metrics = runpy.run_path(str(runner))["METRICS"]
    assert metrics == {"benchmarks/pearl-plus-gamma-c-h200": "pearl-plus/h200/C",
                       "benchmarks/pearl-plus-gamma-l-h200": "pearl-plus/h200/L"}

    def table(within: Path | None = None) -> str:
        if within is None:
            monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
        else:
            monkeypatch.setenv("RESEARCH_RUN_DIR", str(within))
        monkeypatch.setattr(sys, "argv", ["run.py", "table", "--local-only"])
        with pytest.raises(SystemExit) as done:
            runpy.run_path(str(runner), run_name="__main__")
        assert done.value.code == 0
        return capsys.readouterr().out

    def result(within: Path) -> dict:
        return json.loads((within / kit.RESULT).read_text())

    before = tmp_path / "run-1"
    assert table(before) == table()
    got = result(before)
    assert [m["name"] for m in got["measurements"]] == list(metrics)
    for m, point in zip(got["measurements"], O.POINTS):
        assert m["value"] is None and m["mark"] == "measured" and m["target"] == O.GAMMA_MAX and m["verdict"] is None
        assert m["why"] == f"no γ of record: no labelled run times pearl-plus at h200's {point} (`run_of_record=overhead`)"
    assert got["validation"]["status"] == "failed" and got["validation"]["evidence"] == [R.TABLE]
    assert got["validation"]["detail"].startswith("0 of 2 metrics have a γ of record")

    assert R.record_run(s.store, run) == []
    after = tmp_path / "run-2"
    printed = table(after)
    assert printed == table() and "| pearl-plus | h200 | C |" in printed
    got, kept = result(after), json.loads((after / R.TABLE).read_text())
    rows = {r["point"]: r for r in kept["rows"]}
    assert got["workload_fingerprint"]["runs"] == [run] and got["measurement_files"] == [R.TABLE]
    for m, point in zip(got["measurements"], O.POINTS):
        g = rows[point]["gamma"]
        assert (m["value"], m["ci95"], m["unit"], m["mark"]) == (g["gamma"], g["ci95"], R.GAMMA_UNIT, "measured")
        assert m["verdict"] == "within target" and m["inputs"] == ["measured", run, art] and "why" not in m
        assert m["note"] == O.WITHOUT_Y and m["covers"] == O.covers()
    assert got["validation"] == {"status": "passed", "checker": "verity_bench.overhead_records.of_record",
                                 "detail": "2 of 2 metrics have a γ of record (the source check, `of_record`)",
                                 "evidence": [R.TABLE]}
