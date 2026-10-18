"""The suite's report: with ``CIRCUIT_CHECK_REPORT=path`` set, every target the suite checked is written there as one
`verity-circuit-check/v1` report when the session ends (CI uploads it).  In an xdist worker, the circuit-checks the tests start
keep to that worker's share of the suite's memory (`worker_slot_gb`)."""
from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest

from circuit_check import checks as CK
from circuit_check import cli

#: one xdist worker's share of the memory suites.py runs the suite in (`[tool.verity.tests] worker_gb`, which it divides by)
WORKER_GB = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["tool"]["verity"]["tests"]["worker_gb"]


def worker_slot_gb(env: Mapping[str, str], worker_gb: float = WORKER_GB) -> int | None:
    """The `cli.SLOT_MEM_ENV` a circuit-check started in this process budgets (`cli.budget_gb`, `cli.auto_jobs`): in an xdist
    worker, the worker's share ``worker_gb``, or the slot's own when that is smaller; outside one, None (the process has all of
    it).  Unset, each worker's circuit-check budgets the whole slot or cgroup, which every worker of the suite shares."""
    if not env.get("PYTEST_XDIST_WORKER"):
        return None
    slot = env.get(cli.SLOT_MEM_ENV, "")
    return min(int(slot), int(worker_gb)) if slot.isdigit() else int(worker_gb)


@pytest.fixture(scope="session", autouse=True)
def worker_share():
    with pytest.MonkeyPatch.context() as mp:
        if (gb := worker_slot_gb(os.environ)) is not None:
            mp.setenv(cli.SLOT_MEM_ENV, str(gb))
        yield


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
