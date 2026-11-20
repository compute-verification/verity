"""`verity bench attester_load`: the schedule's stated figures, its guard and what it writes for a fixed arm's document.
No test runs an arm: an arm reads the machine's clock."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from verity_bench import kit


ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("attester_load_run", ROOT / "benchmarks/attester_load/run.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

#: the onsite application's schedule (`network_attester.W63K4`)
W63K4 = {"cell_bytes": 512, "slots": 4, "bucket_ms": 100, "delay": 27, "lanes": 4, "queue": 4, "window": 32,
         "ingress_cells": 47, "ingress_buckets": 12, "egress_cells": 5, "every": 16}
STAMP = {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": "test", "commit": None}


def arm_doc(failed: list[str]) -> dict:
    """One stub arm's document as `arm` returns it, at a fast schedule, with fixed readings."""
    s = W63K4 | {"bucket_ms": 50, "delay": 4}
    st = R.stated(s, 4, 30.0)
    arm = {"member": R.STUB, "egress": "id", "model": "fake", "class": "w63k4", "schedule": s, "tokens": 4,
           "prompt_ids": 32, "paced_ms": 30.0, "load_per_s": 0.8 * st["capacity_per_s"], "load_share": 0.8,
           "requests": 4, "reps": 2, **st, "card": "none", "endpoint": "network_attester_serve.py fake",
           "through_first_ms": 260.0, "direct_first_ms": 50.0, "through_last_ms": 320.0, "direct_last_ms": 120.0,
           "added_first_ms": 210.0, "added_first_ci95": [205.0, 215.0], "added_last_ms": 200.0,
           "added_last_ci95": [190.0, 210.0], "tick_lateness_ms": {"p50": 0.4, "p99": 1.5},
           "full": {"accepted": True, "refused": 0, "turned_away": 0}, "relay_cut": 0,
           "repetitions": [{"accepted": True, "refused": 0, "n": 4}] * 2,
           "guard": {"plant_ms": 10.0, "planted_shift_ms": 9.5, "direct_against_itself_ms": 0.3, "failed": failed}}
    return {"class": "w63k4", "arms": [arm], "model_arm": "The model arm was not run: no card."}


def written(monkeypatch, doc: dict, out: Path) -> tuple[int, list[dict]]:
    monkeypatch.setattr(kit, "host_stamp", lambda: STAMP)
    n = R.write(doc, out)
    result = out / "result.json"
    return n, json.loads(result.read_text())["measurements"] if result.exists() else []


def test_the_default_schedule_is_the_application_s() -> None:
    assert R.schedule("w63k4", {}) == W63K4
    assert R.schedule("w63k4", {"delay": 4, "bucket_ms": None}) == W63K4 | {"delay": 4}


def test_the_schedule_states_its_capacity() -> None:
    st = R.stated(W63K4, 4, 50.0)
    assert st["lane_hold_buckets"] == 29 and st["capacity_per_s"] == pytest.approx(4 / 2.9)
    assert st["scheduled_first_ms"] == 2750 and st["scheduled_last_bucket"] == 28


def test_the_guard_refuses_a_planted_delay_that_doesn_t_show_and_a_direct_arm_that_reads_off_zero() -> None:
    assert R.guard(30, 29.0, 0.4) == []
    assert len(R.guard(30, 2.0, 0.4)) == 1
    assert len(R.guard(30, 30.0, 12.0)) == 1
    assert len(R.guard(30, None, None)) == 2


def test_each_point_carries_the_schedule_and_its_scheduled_delays(monkeypatch, tmp_path) -> None:
    doc = arm_doc([])
    n, got = written(monkeypatch, doc, tmp_path)
    assert n == 1
    assert {m["name"]: m["value"] for m in got} == {R.FIRST: 210.0, R.LAST: 200.0, R.TICK: 1.5}
    for m in got:
        assert m["member"] == "stub-cpu" and m["unit"] == "ms" and m["mark"] == "measured"
        assert m["schedule"] == doc["arms"][0]["schedule"]
        assert {"capacity_per_s", "scheduled_first_ms", "scheduled_last_bucket", "egress", "card", "model",
                "load_per_s"} <= set(m)
        assert "repetitions" not in m


def test_an_arm_whose_guard_failed_writes_no_number(monkeypatch, tmp_path) -> None:
    n, got = written(monkeypatch, arm_doc(["planted"]), tmp_path)
    assert n == 0 and got == []
