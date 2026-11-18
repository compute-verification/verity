"""run.sh takes CUDA 13.3 from CUDA_TK, CUDA or the machine's record (benchmarks/kernels/machine.py), and where there is
no record or its toolkit is unread refuses by name before it builds or leases anything. CPU only: gpu-lease and
nvidia-smi are stand-ins that say whether they ran."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
ONE_CARD = ROOT / "tests" / "benchmarks" / "harness" / "fixtures" / "machines" / "one-card.toml"


def _unread(directory: Path) -> Path:
    text = ONE_CARD.read_text()
    start = text.index("toolkit = { value = ")
    end = text.index(', read = "readlink', start)
    directory.mkdir()
    (directory / "m.toml").write_text(text[:start] + 'toolkit = { value = "unread"' + text[end:])
    return directory


@pytest.mark.parametrize(("machine", "says"), [
    ({}, "no CUDA toolkit 13.3: set CUDA to its directory, or MACHINE to the machine's name"),
    ({"MACHINE": "m"}, "machine 'm': its record ("),
])
def test_no_toolkit_13_3_is_refused_by_name_before_any_build_or_lease(machine: dict, says: str, tmp_path: Path) -> None:
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for tool in ("gpu-lease", "nvidia-smi"):
        (bin_ / tool).write_text(f"#!/bin/sh\necho {tool} >> {tmp_path}/called\n")
        (bin_ / tool).chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA", "CUDA_TK", "MACHINE", "MACHINES_D", "GPU_LEASE_UUID")}
    env |= {"PATH": f"{bin_}:{env['PATH']}", "RESEARCH_RUN_DIR": str(tmp_path / "run"),
            "FLOCK_WORK": str(tmp_path / "work"), **machine}
    if machine:
        env["MACHINES_D"] = str(_unread(tmp_path / "machines"))
    run = subprocess.run(["bash", "benchmarks/kernels/wide_leaf/run.sh", "--on", "3"], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    assert run.returncode == 4 and "REFUSED: no CUDA 13.3 (machine.py, above)" in run.stdout
    assert says in run.stderr and (not machine or "has toolkit unread" in run.stderr)
    assert not (tmp_path / "called").exists() and not (tmp_path / "run" / "out" / "setup-run").exists()
