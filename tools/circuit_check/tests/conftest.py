"""The suite's report: with ``CIRCUIT_CHECK_REPORT=path`` set, every target the suite checked is written there as one
`verity-circuit-check/v1` report when the session ends (CI uploads it)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from circuit_check import checks as CK


@pytest.fixture(scope="session")
def results():
    out: list[dict] = []
    yield out
    path = os.environ.get("CIRCUIT_CHECK_REPORT")
    if path and out:
        Path(path).write_text(json.dumps(CK.report(out, CK.Options()), indent=1, default=str) + "\n")
