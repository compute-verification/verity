"""Every test gets its own evidence store: `research run` publishes an attempt at the end, and that must never land in the
developer's ~/.research/store."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_store(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path_factory.mktemp("store")))
    monkeypatch.delenv("RESEARCH_WRITE_THROUGH", raising=False)         # write-through is each test's explicit choice
    # ... and never the developer's ~/.research/store.toml remote (an S3 config without credentials in the test env would otherwise
    # be what every test sees).  Tests that want a remote point RESEARCH_STORE_CONFIG at their own file.
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path_factory.mktemp("cfg") / "absent-store.toml"))
    monkeypatch.delenv("RESEARCH_CAMPAIGN", raising=False)
    monkeypatch.setenv("RESEARCH_MACHINES_D", "none")      # the pod registry (pods/registry.py) is each test's explicit choice
    # a cloud VM's Cursor secrets stand in for the laptop's config files (pods/runpod.py, store/remote.py); no test may reach the
    # real account or bucket through them
    for k in ("RUNPOD_API_KEY", "RUNPOD_SSH_KEY_B64", "R2_ENDPOINT", "R2_BUCKET", "R2_ACCOUNT_ID"):
        monkeypatch.delenv(k, raising=False)
