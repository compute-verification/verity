"""`tools/verity/check/spans.py`: the frozen trace/v1 span shape, the root-open/children/root-close protocol over a run's
`spans.jsonl`, and `Steps` writing one span line per step record.  The shape is a contract the observability watcher
builds against (frozen 14 Oct 2026 19:13Z), so these tests pin its exact keys and value domains."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"repo_{name}", ROOT / "tools" / "verity" / "check" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SPANS = _load("spans")
FROZEN_KEYS = ["trace_id", "span_id", "parent_span_id", "name", "start_utc", "end_utc", "status", "attributes"]


def lines(out: Path) -> list[dict]:
    return [json.loads(l) for l in (out / "spans.jsonl").read_text().splitlines()]


@pytest.fixture(autouse=True)
def outside_a_scope(monkeypatch):
    """These tests may run inside a check in its own scope, whose memory would otherwise join the pinned attributes."""
    monkeypatch.delenv(SPANS.SCOPE_ENV, raising=False)


def test_a_check_in_its_own_scope_records_its_memory_and_a_shared_scope_records_none(tmp_path, monkeypatch):
    cg = tmp_path / "sys/fs/cgroup/user.slice/user-1001.slice/user@1001.service/app.slice/check-b-r1.scope"
    cg.mkdir(parents=True)
    (cg / "memory.peak").write_text("312400000000\n")
    (cg / "memory.current").write_text("120049999999\n")
    (tmp_path / "proc/self").mkdir(parents=True)
    (tmp_path / "proc/self/cgroup").write_text("0::/user.slice/user-1001.slice/user@1001.service/app.slice/check-b-r1.scope\n")
    assert SPANS.check_memory(tmp_path) == {}, "no scope named: nothing"
    monkeypatch.setenv(SPANS.SCOPE_ENV, "check-b-r1")
    assert SPANS.check_memory(tmp_path) == {"mem_peak_gb": 312.4, "mem_current_gb": 120.0}
    monkeypatch.setenv(SPANS.SCOPE_ENV, "check-b-r2")
    assert SPANS.check_memory(tmp_path) == {}, "a cgroup that isn't the named scope (a shared session) is other runs' too"
    monkeypatch.setenv(SPANS.SCOPE_ENV, "check-b-r1")
    monkeypatch.setattr(SPANS, "check_memory", lambda root=Path("/"): {"mem_peak_gb": 312.4, "mem_current_gb": 120.0})
    log = SPANS.SpanLog(tmp_path, trace_id="r-test")
    log.step({"step": "pytest", "status": "passed", "rc": 0, "seconds": 1.0, "ended_utc": "2026-10-14T20:00:00Z"})
    log.close("ok")
    got = lines(tmp_path)
    assert got[1]["attributes"]["mem_peak_gb"] == 312.4 and got[-1]["attributes"] == {"mem_peak_gb": 312.4, "mem_current_gb": 120.0}
    assert all(list(line) == FROZEN_KEYS for line in got)


def test_every_line_has_exactly_the_frozen_keys_and_flat_scalar_attributes(tmp_path):
    log = SPANS.SpanLog(tmp_path, trace_id="r-test")
    log.step({"step": "pytest", "status": "passed", "rc": 0, "log": "pytest.log", "seconds": 4.2,
              "ended_utc": "2026-10-14T20:00:00Z", "cache": {"hit": "abc123"}})
    log.close("ok")
    for line in lines(tmp_path):
        assert list(line) == FROZEN_KEYS
        assert all(isinstance(v, (str, int, float, bool)) for v in line["attributes"].values())


def test_the_root_opens_first_with_null_end_and_status_and_its_last_line_completes_it(tmp_path):
    log = SPANS.SpanLog(tmp_path, trace_id="r-test")
    log.step({"step": "pytest", "status": "passed", "rc": 0, "seconds": 1.0, "ended_utc": "2026-10-14T20:00:00Z"})
    log.close("failed")
    got = lines(tmp_path)
    first, last = got[0], got[-1]
    assert first["parent_span_id"] is None and first["name"] == "check"
    assert first["end_utc"] is None and first["status"] is None
    assert last["span_id"] == first["span_id"] and last["start_utc"] == first["start_utc"]
    assert last["status"] == "failed" and last["end_utc"] is not None
    log.close("ok")
    assert lines(tmp_path) == got, "close is once: a second close writes nothing"


def test_a_step_record_becomes_one_completed_child_with_the_contract_attributes(tmp_path):
    log = SPANS.SpanLog(tmp_path, trace_id="r-test")
    log.step({"step": "circuit-check", "status": "passed", "rc": 0, "log": "circuit-check.log", "seconds": 90.0,
              "ended_utc": "2026-10-14T20:01:30Z", "cache": {"stored": "key9"}})
    span = lines(tmp_path)[1]
    assert span["trace_id"] == "r-test" and span["parent_span_id"] == lines(tmp_path)[0]["span_id"]
    assert span["name"] == "circuit-check" and span["status"] == "ok"
    assert span["start_utc"] == "2026-10-14T20:00:00Z" and span["end_utc"] == "2026-10-14T20:01:30Z"
    assert span["attributes"] == {"step": "circuit-check", "rc": 0, "log": "circuit-check.log", "cache_key": "key9"}


def test_step_statuses_map_onto_the_three_span_statuses(tmp_path):
    log = SPANS.SpanLog(tmp_path, trace_id="r-test")
    for status in ("passed", "failed", "stopped", "skipped"):
        log.step({"step": status, "status": status, "seconds": 0.0, "ended_utc": "2026-10-14T20:00:00Z"})
    got = {s["name"]: s["status"] for s in lines(tmp_path)[1:]}
    assert got == {"passed": "ok", "failed": "failed", "stopped": "cancelled", "skipped": "cancelled"}


def test_the_trace_id_is_the_recorded_runs_id_and_local_otherwise(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261014-000000-abcd")
    assert SPANS.SpanLog(a).trace_id == "r20261014-000000-abcd"
    monkeypatch.delenv("RESEARCH_RUN_ID")
    assert SPANS.SpanLog(b).trace_id.startswith("local-")


def test_steps_mirrors_each_record_into_spans_jsonl(tmp_path):
    """`Steps.end` writes the step to `steps.jsonl` and its span beside it, so the two files never drift."""
    CHECK = _load("check")
    steps = CHECK.Steps(tmp_path, keep_going=True)
    steps.end({"step": "pytest", "status": "passed", "rc": 0, "seconds": 2.0})
    steps.end({"step": "lean-build", "status": "failed", "rc": 1, "seconds": 3.0})
    got = lines(tmp_path)
    assert [s["name"] for s in got] == ["check", "pytest", "lean-build"]
    assert [s["status"] for s in got[1:]] == ["ok", "failed"]
    recs = [json.loads(l) for l in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [r["step"] for r in recs] == [s["name"] for s in got[1:]]
