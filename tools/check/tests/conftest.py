"""Every test here starts from an environment without credentials, a recording run's variables, or what `check` sets for its
steps: inside a recorded check its own tests inherit all three (`check.env()`, #420)."""
from __future__ import annotations

import os

import pytest

CLEARED = ("AWS_", "R2_", "RESEARCH_")
SET_BY_CHECK = ("CHECK_STORE_CUSTODY", "VERITY_STORE_REQUIRED")


@pytest.fixture(autouse=True)
def _no_inherited_run(monkeypatch):
    for k in list(os.environ):
        if k.startswith(CLEARED) or k in SET_BY_CHECK:
            monkeypatch.delenv(k)
