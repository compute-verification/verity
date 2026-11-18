"""run.sh takes CUDA 13.3 from CUDA_TK, CUDA or the run's machine's [properties] (benchmarks/kernels/machine.py), and where
there are none or their toolkit is unread refuses by name before it builds or leases anything. CPU only: gpu-lease and
nvidia-smi are stand-ins that say whether they ran."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
ONE_CARD = ROOT / "tests" / "benchmarks" / "harness" / "fixtures" / "machines" / "one-card.toml"


def _unread(inputs: Path) -> None:
    text = ONE_CARD.read_text()
    start = text.index("toolkit = { value = ")
    end = text.index(', read = "readlink', start)
    inputs.mkdir(parents=True)
    (inputs / "m.toml").write_text(text[:start] + 'toolkit = { value = "unread"' + text[end:])


@pytest.mark.parametrize(("machine", "says"), [
    ({}, "no CUDA toolkit 13.3: set CUDA to its directory, or run on a machine `research run --on` names"),
    ({"RESEARCH_MACHINE": "m"}, "machine 'm': its [properties] have toolkit unread"),
])
def test_no_toolkit_13_3_is_refused_by_name_before_any_build_or_lease(machine: dict, says: str, tmp_path: Path) -> None:
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for tool in ("gpu-lease", "nvidia-smi"):
        (bin_ / tool).write_text(f"#!/bin/sh\necho {tool} >> {tmp_path}/called\n")
        (bin_ / tool).chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA", "CUDA_TK", "RESEARCH_MACHINE", "GPU_LEASE_UUID")}
    env |= {"PATH": f"{bin_}:{env['PATH']}", "RESEARCH_RUN_DIR": str(tmp_path / "run"),
            "FLOCK_WORK": str(tmp_path / "work"), **machine}
    if machine:
        _unread(tmp_path / "run" / "inputs")
    run = subprocess.run(["bash", "benchmarks/kernels/wide_leaf/run.sh", "--on", "3"], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    assert run.returncode == 4 and "REFUSED: no CUDA 13.3 (machine.py, above)" in run.stdout
    assert says in run.stderr
    assert not (tmp_path / "called").exists() and not (tmp_path / "run" / "out" / "setup-run").exists()
