"""exactlin's runners take CUDA 13.0 from CUDA or the run's machine's [properties] (benchmarks/kernels/machine.py), and with neither
refuse by name before they touch a card. CPU only: gpu-lease and nvidia-smi are stand-ins that say whether they ran."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
DIR = ROOT / "benchmarks" / "kernels" / "exactlin"


@pytest.mark.parametrize("script", ["kbench.sh", "served.sh", "share.sh"])
def test_a_runner_with_no_toolkit_refuses_by_name_before_any_card(script: str, tmp_path: Path) -> None:
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for tool in ("gpu-lease", "nvidia-smi"):
        (bin_ / tool).write_text(f"#!/bin/sh\necho {tool} >> {tmp_path}/called\n")
        (bin_ / tool).chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA", "RESEARCH_MACHINE", "GPU_LEASE_UUID")}
    env |= {"PATH": f"{bin_}:{env['PATH']}", "RESEARCH_RUN_DIR": str(tmp_path / "run"), "TMPDIR": str(tmp_path)}
    run = subprocess.run(["bash", str(DIR / script)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert run.returncode == 2
    assert "no CUDA toolkit 13.0: set CUDA to its directory, or run on a machine `research run --on` names" in run.stderr
    assert not (tmp_path / "called").exists()
