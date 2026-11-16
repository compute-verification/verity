from __future__ import annotations

import pytest

from verity_cli import run as R

BUILD_VERIFIER = R.build_verifier


@pytest.fixture(autouse=True)
def built_verifier(monkeypatch):
    monkeypatch.setattr(R, "build_verifier", lambda: None)
