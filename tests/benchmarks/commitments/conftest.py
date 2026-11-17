from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_run_result(monkeypatch):
    for name in ("RESEARCH_RUN_DIR", "OUT"):
        monkeypatch.delenv(name, raising=False)
