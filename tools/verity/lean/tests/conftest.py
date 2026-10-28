"""Every test here starts from an environment without credentials, a recording run's variables, or what `check` sets for its
steps, as the check tool's tests do: the tests of `check.py`'s Lean steps inherit all three inside a recorded check.  And
with a step cache of its own (`CHECK_CACHE`), so a fake step's pass never lands under a real key of this machine's cache."""
from __future__ import annotations

import os

import pytest

CLEARED = ("AWS_", "R2_", "RESEARCH_", "CLOUD_AGENT_")
SET_BY_CHECK = ("CHECK_STORE_CUSTODY", "VERITY_STORE_REQUIRED", "VERITY_REGRESSION")


@pytest.fixture(autouse=True)
def _no_inherited_run(monkeypatch, tmp_path_factory):
    for k in list(os.environ):
        if k.startswith(CLEARED) or k in SET_BY_CHECK:
            monkeypatch.delenv(k)
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path_factory.mktemp("check-cache")))
