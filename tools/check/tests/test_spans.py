"""`tools/check/spans.py`: the frozen trace/v1 span shape, the root-open/children/root-close protocol over a run's
`spans.jsonl`, and `Steps` writing one span line per step record.  The shape is a frozen contract the observability watcher
builds against, so these tests pin its exact keys and value domains."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"repo_{name}", ROOT / "tools" / "check" / f"{name}.py")
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


def scope_files(cg: Path, usage_s: float, anon_gib: float, file_gib: float) -> None:
    cg.mkdir(parents=True, exist_ok=True)
    (cg / "cpu.stat").write_text(f"usage_usec {int(usage_s * 1e6)}\nuser_usec 1\n")
    (cg / "memory.stat").write_text(f"anon {int(anon_gib * 2 ** 30)}\nfile {int(file_gib * 2 ** 30)}\nshmem 0\n")


def test_a_scope_watch_keeps_the_busiest_minute_and_the_highest_anon_in_the_slot_record(tmp_path):
    """The figures outlive the scope in the run's slot.json, beside what slot.py wrote: cores over the busiest 60 s, not an
    average over a step, and `anon` at its highest, not at a step's end."""
    cg, record = tmp_path / "check-b-r1.scope", tmp_path / "slot.json"
    record.write_text(json.dumps({"state": "held", "slot": "b", "reserves_gib": 340}) + "\n")
    now = [0.0]
    watch = SPANS.ScopeWatch(cg, record, clock=lambda: now[0])
    for t, usage, anon, file in ((0, 0, 4, 1), (30, 60, 30, 2), (60, 600, 80, 9), (70, 610, 20, 70), (130, 1000, 10, 2)):
        now[0] = t
        scope_files(cg, usage, anon, file)
        watch.tick()
    (cg / "cpu.stat").unlink()
    watch.tick()
    watch.write()
    doc = json.loads(record.read_text())
    assert {k: doc[k] for k in ("state", "slot", "reserves_gib")} == {"state": "held", "slot": "b", "reserves_gib": 340}
    unread = doc["used"].pop("unread")
    assert doc["used"] == {"cores_max": 10.0, "anon_max_gib": 80.0, "samples": 5, "every_s": SPANS.WATCH_EVERY_S, "missed": 1}
    assert unread.startswith("FileNotFoundError: "), "a reading that failed says why in the record, and the check goes on"


def test_a_scope_watch_starts_only_in_the_checks_own_scope_inside_a_run(tmp_path, monkeypatch):
    cg = tmp_path / "sys/fs/cgroup/user.slice/user-1001.slice/user@1001.service/app.slice/check-b-r1.scope"
    scope_files(cg, 1, 1, 1)
    (tmp_path / "proc/self").mkdir(parents=True)
    (tmp_path / "proc/self/cgroup").write_text("0::/user.slice/user-1001.slice/user@1001.service/app.slice/check-b-r1.scope\n")
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setenv(SPANS.SCOPE_ENV, "check-b-r1")
    assert SPANS.ScopeWatch.start(tmp_path) is None, "outside a research run there is no slot record to keep it in"
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv(SPANS.SCOPE_ENV, "check-b-r2")
    assert SPANS.ScopeWatch.start(tmp_path) is None, "a cgroup that isn't the named scope is other runs' too"
    monkeypatch.setenv(SPANS.SCOPE_ENV, "check-b-r1")
    watch = SPANS.ScopeWatch.start(tmp_path)
    watch.stop()
    assert json.loads((tmp_path / "slot.json").read_text())["used"]["samples"] >= 1


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


def test_a_step_opens_its_span_as_it_starts_and_its_record_completes_the_same_span(tmp_path):
    """A running step is an open child (`research status RUN` shows it running, since its own start); its record completes
    that span_id from that start, and a step that never started (a skip) is still one completed line."""
    CHECK = _load("check")
    steps = CHECK.Steps(tmp_path, keep_going=True)
    with open(tmp_path / "x.log", "w") as log:
        p = steps.start("circuit-check", ["true"], tmp_path, {}, log)
        p.wait()
    opened = lines(tmp_path)[1]
    assert (opened["name"], opened["end_utc"], opened["status"], opened["attributes"]) == ("circuit-check", None, None,
                                                                                          {"step": "circuit-check"})
    steps.end({"step": "circuit-check", "status": "passed", "rc": 0, "seconds": 0.1})
    steps.end({"step": "lean-agreement", "status": "skipped", "seconds": 0.0})
    got = lines(tmp_path)
    assert [s["span_id"] for s in got[1:3]] == [opened["span_id"]] * 2 and got[2]["start_utc"] == opened["start_utc"]
    assert got[2]["status"] == "ok" and got[3]["name"] == "lean-agreement" and got[3]["span_id"] != opened["span_id"]
    assert all(list(line) == FROZEN_KEYS for line in got)
