import sys
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE.parents[2] / "benchmarks" / "prod_repro") not in sys.path:
    sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "prod_repro"))


@pytest.fixture(autouse=True)
def _no_run_result(monkeypatch):
    """No test writes into a research run's result: under `research run` a test inherits `RESEARCH_RUN_DIR` and `OUT`,
    where the kit writes the run's measurements whatever directory its caller names, so each test starts with both
    unset, and a test that writes as a run sets its own (`tests/test_tests_write_no_measurement.py`)."""
    for name in ("RESEARCH_RUN_DIR", "OUT"):
        monkeypatch.delenv(name, raising=False)
