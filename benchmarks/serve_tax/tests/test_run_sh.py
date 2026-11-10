"""`run.sh` refuses the boundary-rows arm under a research run's own directory. The arm's sink holds each class's prover-only
index and salts beside its public commit.json, all under OUT, and a research run publishes its own directory (the one its
runner left a job.json in): the arm runs only where RESEARCH_RUN_DIR names another directory. The other arms are not
refused, and nothing is written before the refusal."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

RUN_SH = Path(__file__).resolve().parents[1] / "run.sh"
#: where run.sh stops next in these stand-in runs, past the refusal: a model id it doesn't know
PAST = "serve-tax: MODEL_ID is"


def run(tmp: Path, arms: str, *, published: bool) -> tuple[subprocess.CompletedProcess, Path]:
    d = tmp / "run"
    d.mkdir()
    if published:
        (d / "job.json").write_text("{}")
    env = {"PATH": os.defpath, "HOME": str(tmp), "RESEARCH_RUN_DIR": str(d), "ARMS": arms, "VENV": str(tmp / "venv"), "MODEL_ID": "none"}
    return subprocess.run(["bash", str(RUN_SH)], env=env, cwd=tmp, stdin=subprocess.DEVNULL, capture_output=True, text=True), d


@pytest.mark.parametrize("arms", ["boundary-rows", "record boundary-rows compiled"])
def test_the_boundary_rows_arm_is_refused_in_a_run_dir_the_run_publishes(tmp_path: Path, arms: str):
    got, d = run(tmp_path, arms, published=True)
    assert got.returncode == 2 and PAST not in got.stdout
    assert "ARMS=boundary-rows" in got.stdout and "RESEARCH_RUN_DIR=DIR" in got.stdout and "commit.json" in got.stdout
    assert sorted(p.name for p in d.iterdir()) == ["job.json"], "nothing is written before the refusal"


@pytest.mark.parametrize("arms, published", [("boundary-rows", False), ("record compiled", True), ("boundary", True)])
def test_a_private_dir_takes_the_arm_and_the_other_arms_run_anywhere(tmp_path: Path, arms: str, published: bool):
    got, _ = run(tmp_path, arms, published=published)
    assert got.returncode == 2 and PAST in got.stdout and "ARMS=boundary-rows" not in got.stdout
