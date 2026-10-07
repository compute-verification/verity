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


def pytest_collection_modifyitems(config, items):
    """The per-target tests are `circuit-check --all` as pytest items, which `check` runs through the CLI: they are collected
    only when `-m` names `circuit_suite`."""
    if "circuit_suite" in (config.getoption("markexpr") or ""):
        return
    dropped = [i for i in items if "circuit_suite" in i.keywords]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = [i for i in items if "circuit_suite" not in i.keywords]
