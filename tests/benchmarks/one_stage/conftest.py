"""The suite's autouse fixture: no test writes a measurement into a research run's result; and its test primitives' Boolean bodies."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _stand_in() -> None:
    """The suite's test primitives' Boolean bodies (`tests/verity/circuits/stand_in.py`), registered with the Program's
    lookup as it loads, before any test module builds a Program of them."""
    if "stand_in" in sys.modules:
        return
    tests = next(p for p in Path(__file__).resolve().parents if p.name == "tests")
    spec = importlib.util.spec_from_file_location("stand_in", tests / "verity" / "circuits" / "stand_in.py")
    sys.modules["stand_in"] = mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


_stand_in()


@pytest.fixture(autouse=True)
def _stand_in_lookup():
    """The stand-in bodies' lookup again after each test: `door_walls` registers the catalog's (`_catalog`), which
    replaces the process's one lookup, and every later test on that worker would build its Programs without them."""
    from verity.circuits import program as PG
    saved = list(PG._BODY_LOOKUP)
    yield
    PG._BODY_LOOKUP[:] = saved


@pytest.fixture(autouse=True)
def _no_run_result(monkeypatch):
    """No test writes into a research run's result: under `research run` a test inherits `RESEARCH_RUN_DIR` and `OUT`,
    where the kit writes the run's measurements whatever directory its caller names, so each test starts with both
    unset, and a test that writes as a run sets its own (`tests/test_tests_write_no_measurement.py`)."""
    for name in ("RESEARCH_RUN_DIR", "OUT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _poow_named(monkeypatch):
    """`served_verdicts.sh` requires POUW named, and Pearl+ is the one scheme left: a test that runs it without POUW
    runs it under POUW=pearlplus, and one that sets POUW itself keeps its own."""
    import subprocess

    run = subprocess.run

    def named(argv, *a, **kw):
        env = kw.get("env")
        if isinstance(argv, (list, tuple)) and any(str(x).endswith("served_verdicts.sh") for x in argv) and isinstance(env, dict):
            kw["env"] = {"POUW": "pearlplus", **env}
        return run(argv, *a, **kw)

    monkeypatch.setattr(subprocess, "run", named)
