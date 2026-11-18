from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lean_stands_in


@pytest.fixture(autouse=True)
def _lean_stands_in(monkeypatch):
    """The Lean answers the memory verdicts don't have yet, stood in agreeing with the Python (`lean_stands_in`), so
    the suite's tests of what follows an accept run; a test of the undecided answer calls `lean_stands_in.unwritten`."""
    lean_stands_in.agree(monkeypatch)


@pytest.fixture(autouse=True)
def _no_run_result(monkeypatch):
    """No test writes into a research run's result (`tests/test_tests_write_no_measurement.py`)."""
    for name in ("RESEARCH_RUN_DIR", "OUT"):
        monkeypatch.delenv(name, raising=False)
