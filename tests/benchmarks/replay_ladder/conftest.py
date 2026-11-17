import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE.parents[2] / "benchmarks" / "replay_ladder") not in sys.path:
    sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "replay_ladder"))


@pytest.fixture(autouse=True)
def _no_run_result(monkeypatch):
    for name in ("RESEARCH_RUN_DIR", "OUT"):
        monkeypatch.delenv(name, raising=False)
