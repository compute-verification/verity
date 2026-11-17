"""`verity bench attester_load`: the stub arm end to end on a CPU at a fast schedule, its guard, the schedule's stated
figures and what it writes."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("attester_load_run", ROOT / "benchmarks/attester_load/run.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

#: the onsite application's schedule (`network_attester.W63K4`), as `scan_zk.py schedule --class w63k4` prints it
W63K4 = {"cell_bytes": 512, "slots": 4, "bucket_ms": 100, "delay": 27, "lanes": 4, "queue": 4, "window": 32,
         "ingress_cells": 47, "ingress_buckets": 12, "egress_cells": 5, "every": 16}
FAST = ["--bucket-ms", "50", "--delay", "4", "--paced-ms", "30", "--plant-ms", "10", "--slack-ms", "10", "--reps", "2",
        "--requests", "4"]


@pytest.fixture(scope="module")
def ran(tmp_path_factory) -> tuple[dict, Path, int]:
    out = tmp_path_factory.mktemp("out")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = R.main([*FAST, "--json", "--out", str(out)])
    return json.loads(buf.getvalue()), out, code


def test_the_default_schedule_is_the_application_s_and_states_its_capacity() -> None:
    s = R.schedule("w63k4", {})
    assert s == W63K4
    st = R.stated(s, 4, 50.0)
    assert st["lane_hold_buckets"] == 29 and st["capacity_per_s"] == pytest.approx(4 / 2.9)
    assert st["scheduled_first_ms"] == 2750 and st["scheduled_last_bucket"] == 28


def test_the_stub_arm_runs_end_to_end_and_its_guard_holds(ran) -> None:
    doc, _out, code = ran
    assert code == 0 and doc["model_arm"].startswith("The model arm was not run")
    (arm,) = doc["arms"]
    assert arm["member"] == "stub-cpu" and arm["egress"] == "id" and arm["card"] == "none"
    assert arm["guard"]["failed"] == []
    assert abs(arm["guard"]["planted_shift_ms"] - 10) <= R.PLANT_TOL_MS
    assert abs(arm["guard"]["direct_against_itself_ms"]) <= R.SELF_TOL_MS
    assert all(r["accepted"] and r["refused"] == 0 and r["n"] == 4 for r in arm["repetitions"])
    assert arm["full"]["accepted"] and arm["tick_lateness_ms"]["p99"] is not None
    assert 0 < arm["added_first_ms"] <= arm["through_first_ms"]
    assert arm["through_first_ms"] >= arm["scheduled_first_ms"]


def test_each_point_carries_the_schedule_and_its_scheduled_delays(ran) -> None:
    _doc, out, _code = ran
    got = json.loads((out / "result.json").read_text())["measurements"]
    assert sorted(m["name"] for m in got) == sorted([R.FIRST, R.LAST, R.TICK])
    for m in got:
        assert m["member"] == "stub-cpu" and m["unit"] == "ms" and m["mark"] == "measured"
        assert m["schedule"]["delay"] == 4 and m["schedule"]["bucket_ms"] == 50
        assert {"capacity_per_s", "scheduled_first_ms", "scheduled_last_bucket", "egress", "card", "model",
                "load_per_s"} <= set(m)


def test_the_guard_refuses_a_planted_delay_that_doesn_t_show_and_a_direct_arm_that_reads_off_zero() -> None:
    assert R.guard(30, 29.0, 0.4) == []
    assert len(R.guard(30, 2.0, 0.4)) == 1
    assert len(R.guard(30, 30.0, 12.0)) == 1
    assert len(R.guard(30, None, None)) == 2


def test_an_arm_whose_guard_failed_writes_no_number(ran, tmp_path) -> None:
    doc, _out, _code = ran
    failed = {**doc, "arms": [{**doc["arms"][0], "guard": {**doc["arms"][0]["guard"], "failed": ["planted"]}}]}
    assert R.write(failed, tmp_path) == 0
    assert not (tmp_path / "result.json").exists() or json.loads((tmp_path / "result.json").read_text()).get(
        "measurements", []) == []
