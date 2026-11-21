"""The suite's autouse fixture: no test writes a measurement into a research run's result; and the record check's test
primitives get Boolean bodies before collection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _stand_in() -> None:
    """The record check's test primitives' Boolean bodies (`tests/verity/circuits/stand_in.py`), registered with the
    Program's lookup as it loads, before any test module builds a Program of them."""
    if "stand_in" in sys.modules:
        return
    tests = next(p for p in Path(__file__).resolve().parents if p.name == "tests")
    spec = importlib.util.spec_from_file_location("stand_in", tests / "verity" / "circuits" / "stand_in.py")
    sys.modules["stand_in"] = mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


_stand_in()


@pytest.fixture(autouse=True)
def _no_run_result(monkeypatch):
    """No test writes into a research run's result: under `research run` a test inherits `RESEARCH_RUN_DIR` and `OUT`,
    where the kit writes the run's measurements whatever directory its caller names, so each test starts with both
    unset, and a test that writes as a run sets its own (`tests/test_tests_write_no_measurement.py`)."""
    for name in ("RESEARCH_RUN_DIR", "OUT"):
        monkeypatch.delenv(name, raising=False)
