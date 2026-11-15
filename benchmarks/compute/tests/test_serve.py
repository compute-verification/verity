"""`serve.sh`'s refusal of a run directory that already holds a record, before it touches the machine or writes a file:
a share read after a failed run would otherwise be an earlier run's number."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SERVE = Path(__file__).resolve().parents[1] / "serve.sh"


@pytest.mark.parametrize("held", ["steps.txt", "compute-run.json", "logs", "pieces", "record", "registered"])
def test_a_run_dir_holding_a_record_is_refused_before_any_step(tmp_path, held):
    out = tmp_path / "run" / "out"
    out.mkdir(parents=True)
    if "." in held:
        (out / held).write_text("an earlier run's\n")
    else:
        (out / held).mkdir()
    before = sorted(p.name for p in out.iterdir())
    got = subprocess.run(["bash", str(SERVE)], env={**os.environ, "RUN_DIR": str(tmp_path / "run")},
                         capture_output=True, text=True)
    assert got.returncode == 3 and got.stdout == ""
    assert f"{out} already holds a run's record ({out / held})" in got.stderr and "give RUN_DIR a new directory" in got.stderr
    assert sorted(p.name for p in out.iterdir()) == before
