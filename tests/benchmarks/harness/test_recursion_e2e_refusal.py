import runpy
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[3] / "benchmarks" / "recursion_e2e" / "run.py"


def test_the_old_recursion_benchmark_refuses_by_name():
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(RUNNER), run_name="__main__")
    assert done.value.code == "recursion_e2e: the old recursion's benchmark is gone, since the first version has no outer proof; the readings of record stand"
