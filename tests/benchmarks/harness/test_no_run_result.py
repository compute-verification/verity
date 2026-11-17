"""Under `research run`, the test that ran `benchmarks/exfiltration/run.py` with the run's environment wrote its fixture's
figures into the run's result, and the Metrics tab drew them as the metric's points. Run by a plain pytest with the
run's `RESEARCH_RUN_DIR` and `OUT` set, it writes nothing there; without the suite's conftest, which unsets them
(`tests/test_tests_write_no_measurement.py`), the kit refuses inside pytest the run the process started
under (`kit.refuse_the_started_run`), and the test fails, having written nothing."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WROTE = "test_exfiltration.py::test_verity_bench_exfiltration_is_the_commands_runner"


def under_a_run(tmp_path: Path, *flags: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    """A plain pytest of `WROTE` with a run's directory and its `OUT` set, and the files that appear in them."""
    run, out = tmp_path / "run", tmp_path / "out"
    run.mkdir(parents=True), out.mkdir(parents=True)
    env = {**os.environ, "RESEARCH_RUN_DIR": str(run), "OUT": str(out), "RESEARCH_RUN_ID": "r20261116-000000-0000"}
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist",
                           f"--basetemp={tmp_path / 'base'}", *flags, WROTE], cwd=HERE, env=env, capture_output=True,
                          text=True)
    return done, sorted(p.relative_to(tmp_path).as_posix() for d in (run, out) for p in d.rglob("*") if p.is_file())


@pytest.mark.slow
def test_a_test_run_under_a_research_run_writes_nothing_into_its_result(tmp_path):
    done, wrote = under_a_run(tmp_path / "with")
    assert done.returncode == 0 and wrote == [], done.stdout[-2000:] + done.stderr[-2000:]
    done, wrote = under_a_run(tmp_path / "without", "--noconftest")
    assert done.returncode == 1 and f"FAILED {WROTE}" in done.stdout and wrote == [], (done.stdout[-2000:]
                                                                                      + done.stderr[-2000:])
