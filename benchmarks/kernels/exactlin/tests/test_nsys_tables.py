"""share.py's and steps.py's analyses of an nsys SQLite export: a table nsys leaves out of a capture that holds none of
its events reads as no events, and any other database error raises."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import share
import steps

PID = 7 << 24
TABLES = {
    "StringIds": "id INTEGER, value TEXT",
    "NVTX_EVENTS": "start INTEGER, end INTEGER, text TEXT, globalTid INTEGER, textId INTEGER",
    "CUPTI_ACTIVITY_KIND_KERNEL": "demangledName INTEGER, globalPid INTEGER, correlationId INTEGER, start INTEGER, "
                                  "end INTEGER",
    "CUPTI_ACTIVITY_KIND_RUNTIME": "nameId INTEGER, correlationId INTEGER, globalTid INTEGER, start INTEGER, end INTEGER",
}


def export(path: Path, leave_out: tuple[str, ...] = (), extra: dict[str, str] | None = None) -> str:
    db = sqlite3.connect(path)
    for name, cols in {**TABLES, **(extra or {})}.items():
        if name not in leave_out:
            db.execute(f"CREATE TABLE {name} ({cols})")
    db.execute("INSERT INTO StringIds VALUES (1, 'k_gemm')")
    if "NVTX_EVENTS" not in leave_out:
        db.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, NULL)",
                       [(0, 100, f"{steps.PROF_RANGE}a/n1", PID), (200, 400, f"{steps.PROF_RANGE}a/n3", PID)])
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in leave_out:
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, ?, 0, ?, ?)",
                       [(PID, 10, 20), (PID, 210, 220), (PID, 230, 240), (PID, 250, 260)])
    db.commit()
    db.close()
    return str(path)


def test_steps_reads_a_capture_with_no_runtime_table_as_no_calls(tmp_path: Path) -> None:
    doc = steps.analyse(export(tmp_path / "t.sqlite", leave_out=("CUPTI_ACTIVITY_KIND_RUNTIME",)))
    arm = doc["arms"]["a"]
    assert arm["api_per_step"] == {} and arm["launches_per_step"] == 1.0


def test_steps_raises_on_a_capture_with_no_kernel_table(tmp_path: Path) -> None:
    with pytest.raises(sqlite3.OperationalError, match="CUPTI_ACTIVITY_KIND_KERNEL"):
        steps.analyse(export(tmp_path / "t.sqlite", leave_out=("CUPTI_ACTIVITY_KIND_KERNEL",)))


def test_share_reads_a_capture_with_no_memcpy_or_memset_table_as_none(tmp_path: Path) -> None:
    assert share.analyse(export(tmp_path / "t.sqlite"), {"config": {}}) == {}


def test_share_raises_on_a_memcpy_table_it_cannot_read(tmp_path: Path) -> None:
    path = export(tmp_path / "t.sqlite", extra={"CUPTI_ACTIVITY_KIND_MEMCPY": "start INTEGER"})
    with pytest.raises(sqlite3.OperationalError, match="correlationId"):
        share.analyse(path, {"config": {}})
