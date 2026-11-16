from __future__ import annotations

import pytest

from verity_cli import run as R


@pytest.fixture(autouse=True)
def built_verifier(monkeypatch):
    monkeypatch.setattr(R, "build_verifier", lambda: None)
