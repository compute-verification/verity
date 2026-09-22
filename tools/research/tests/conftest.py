"""Every test gets its own evidence store: `research run` publishes an attempt at the end, and that must never land in the
developer's ~/.research/store."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_store(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path_factory.mktemp("store")))
    monkeypatch.delenv("RESEARCH_CAMPAIGN", raising=False)
