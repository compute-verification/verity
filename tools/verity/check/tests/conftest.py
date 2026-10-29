"""Every test here starts from an environment without credentials, a recording run's variables, or what `check` sets for its
steps: inside a recorded check its own tests inherit all three (`check.env()`).  And with a step cache of its own
(`CHECK_CACHE`): a fake step's pass stored under a real key (`check.STEP_KEYS`, which key the commit's tree) would be
reused by a later test, and by this machine's next real check."""
from __future__ import annotations

import os

import pytest

CLEARED = ("AWS_", "R2_", "RESEARCH_", "CLOUD_AGENT_")      # the last: or, R2_ cleared, the store reads a VM's shell that lost its secrets
SET_BY_CHECK = ("CHECK_STORE_CUSTODY", "VERITY_STORE_REQUIRED", "VERITY_REGRESSION")


@pytest.fixture(autouse=True)
def _no_inherited_run(monkeypatch, tmp_path_factory):
    for k in list(os.environ):
        if k.startswith(CLEARED) or k in SET_BY_CHECK:
            monkeypatch.delenv(k)
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path_factory.mktemp("check-cache")))
