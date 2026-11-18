from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_run_result(monkeypatch):
    """Under `research run` the kit writes into the run's own result whatever directory its caller names
    (`tests/test_tests_write_no_measurement.py`)."""
    for name in ("RESEARCH_RUN_DIR", "OUT"):
        monkeypatch.delenv(name, raising=False)
