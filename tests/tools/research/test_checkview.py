"""`research.checkview`: a running check's state and ETA from its own `spans.jsonl` and circuit-check progress, timed against
a finished check, and never a guess when the basis can't time it."""
from __future__ import annotations

import io
import json
import subprocess
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from research import checkview as CV

T0 = datetime(2026, 10, 28, 21, 0, tzinfo=UTC)


def at(minutes: float) -> str:
    return (T0 + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def span(sid: str, name: str, start: float, end: float | None = None, status: str | None = None, parent: str | None = "root") -> dict:
    return {"trace_id": "r", "span_id": sid, "parent_span_id": None if name == "check" else parent, "name": name,
            "start_utc": at(start), "end_utc": None if end is None else at(end), "status": status, "attributes": {}}


def write(d: Path, spans: list[dict], targets: list[tuple[str, float]] = (), failures: dict[str, int] | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / CV.SPANS).write_text("".join(json.dumps(s) + "\n" for s in spans))
    (d / CV.PROGRESS).write_text("".join(json.dumps({"kind": "definition", "target": t, "seconds": s,
                                                     "failures": (failures or {}).get(t, 0)}) + "\n" for t, s in targets))
    return d


def basis(tmp_path: Path) -> Path:
    return write(tmp_path / "r-basis", [span("root", "check", 0), span("a", "pytest", 0, 100, "ok"),
                                        span("b", "circuit-check", 0, 280, "ok"), span("root", "check", 0, 280, "ok")],
                 [("A", 10), ("B", 20), ("C", 30), ("D", 60)])


def test_a_running_circuit_check_is_timed_by_the_basiss_targets_left_at_its_own_pace(tmp_path):
    run = write(tmp_path / "r-run", [span("root", "check", 0), span("p", "pytest", 0), span("c", "circuit-check", 0),
                                     span("p", "pytest", 0, 90, "failed")], [("A", 10), ("B", 20)], {"B": 1})
    got = CV.view(run, T0 + timedelta(minutes=1), basis(tmp_path))
    assert got[0] == "ended   pytest failed in 1 h 30 m"
    # 30 target-seconds in 60 s: half pace; C and D are 90 target-seconds, so 180 s
    assert got[1] == ("running circuit-check for 1 min: 2 target(s) done, 1 with failures; 2 of the basis's 4 to go, "
                      "about 3 min at this run's pace")
    assert got[-1] == "ETA     about 3 min from now, by r-basis"


def test_without_a_basis_or_with_a_step_it_cant_time_the_eta_says_why_instead_of_guessing(tmp_path):
    run = write(tmp_path / "r-run", [span("root", "check", 0), span("c", "circuit-check", 0)], [("A", 10)])
    now = T0 + timedelta(minutes=1)
    assert CV.view(run, now)[-1] == "ETA     unknown: no finished check to time it against (`--basis RUN`)"
    assert "no pace yet" not in CV.view(run, now)[0]
    early = write(tmp_path / "r-early", [span("root", "check", 0), span("c", "circuit-check", 0)], [("A", 10)])
    b = basis(tmp_path)
    assert CV.view(early, now, b)[-1] == "ETA     unknown: the basis's pytest hasn't started here yet"
    odd = write(tmp_path / "r-odd", [span("root", "check", 0), span("p", "pytest", 0), span("x", "lean-audit", 0)])
    assert CV.view(odd, T0 + timedelta(minutes=30), b) == [
        "running pytest for 30 min: the basis took 1 h 40 m, about 70 min left",
        "running lean-audit for 30 min: no time for it in the basis",
        "ETA     unknown: the basis's circuit-check hasn't started here yet"]
    both = write(tmp_path / "r-both", [span("root", "check", 0), span("p", "pytest", 0), span("c", "circuit-check", 0),
                                       span("x", "lean-audit", 0)], [("A", 10)])
    assert CV.view(both, T0 + timedelta(minutes=1), b)[-1] == "ETA     unknown: a running step has no time in the basis"


def test_an_ended_check_says_so_and_a_run_that_isnt_a_check_shows_nothing(tmp_path):
    b = basis(tmp_path)
    assert CV.view(b, T0, None)[-1] == "check   ended ok at 2026-10-29T01:40:00Z after 4 h 40 m"
    assert CV.finished(b) and CV.is_check(b)
    (tmp_path / "plain").mkdir()
    assert CV.view(tmp_path / "plain", T0) == [] and not CV.is_check(tmp_path / "plain")


def test_the_machines_newest_finished_check_is_copied_as_the_basis_and_none_found_is_none(tmp_path):
    src = basis(tmp_path)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name in CV.CHECK_FILES:
            if (src / name).exists():
                tf.add(src / name, arcname=f"r20261028-064102-e0e7/{name}")
        tf.add(src / CV.SPANS, arcname="r20261028-064102-e0e7/secret.json")
    calls = []

    def read(ssh, argv, consume, timeout):
        calls.append(argv)
        consume(io.BytesIO(buf.getvalue()))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    rec = {"remote_run_dir": "/workspace/research/runs/r20261028-173829-d101"}
    got = CV.remote_basis(rec, ["ssh"], tmp_path / "cache", read)
    assert got == tmp_path / "cache" / "r20261028-064102-e0e7" and CV.finished(got)
    assert not (got / "secret.json").exists()
    assert "cd /workspace/research/runs" in calls[0][-1] and "r20261028-173829-d101" in calls[0][-1]

    nothing = io.BytesIO()
    tarfile.open(fileobj=nothing, mode="w").close()

    def empty(ssh, argv, consume, timeout):
        consume(io.BytesIO(nothing.getvalue()))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    assert CV.remote_basis(rec, ["ssh"], tmp_path / "cache2", empty) is None
    assert CV.remote_basis({}, ["ssh"], tmp_path / "cache3", read) is None
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / CV.SPANS).write_text(json.dumps(span("root", "check", 0)) + '\n{"trace_id": "r", "span')
    assert CV.is_check(tmp_path / "half") and not CV.finished(tmp_path / "half")   # a line still being written is left out
