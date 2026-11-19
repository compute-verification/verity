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


#: the registered compiler a test's Build of record binds
COMPILER = {"name": "verity/prove-path/capped-compiler/v0", "digest": "c0" * 64}


@pytest.fixture
def predicted_row():
    """Writes predict row's record (`verity_vllm.records.predicted_row`, as `aot.write_row` dumps it) into a Build dir with
    no manifest, binding `COMPILER` (its `compiler`); keyword arguments replace its fields. -> the dir."""
    import json

    from verity_vllm.records.predicted_row import FILE, PredictedRow

    def write(d: Path, **fields) -> Path:
        rec = {"row": "r", "config": "config.json", "code_version": "0" * 16, "requests": {"build_request_LP11_T8": "ab" * 64},
               "workload_digest": "f0" * 64, "lag": 0, "compiler": dict(COMPILER), "secs": 1.0} | fields
        d.mkdir(parents=True, exist_ok=True)
        (d / FILE).write_text(json.dumps(PredictedRow(**rec).dump()))
        return d
    write.compiler = COMPILER
    return write


@pytest.fixture(autouse=True)
def _no_run_result(monkeypatch):
    """No test writes into a research run's result: under `research run` a test inherits `RESEARCH_RUN_DIR` and `OUT`,
    where the kit writes the run's measurements whatever directory its caller names, so each test starts with both
    unset, and a test that writes as a run sets its own (`tests/test_tests_write_no_measurement.py`)."""
    for name in ("RESEARCH_RUN_DIR", "OUT"):
        monkeypatch.delenv(name, raising=False)
