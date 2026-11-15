"""The device check of dc's five GPU scripts, run on a CPU with stand-ins for nvidia-smi and nvcc: on a card of another
compute capability a script stops with exit 2 and names what the device reports, and on its own card it goes on to the
build (which the stand-in nvcc refuses). The jobs themselves need an H200 or a B200."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DC = "benchmarks/pouw/dc"
#: (script, its arguments, the compute capability it needs, its exit when the build is refused)
JOBS = [("run.sh", ["sm90"], 9, 1), ("run.sh", ["sm100"], 10, 1), ("uni.sh", ["sm90"], 9, 1), ("uni.sh", ["sm100"], 10, 1),
        ("pipe.sh", [], 10, 1), ("ffill.sh", [], 9, 1), ("s8_mma.sh", ["sm90"], 9, 4), ("s8_mma.sh", ["sm100"], 10, 4)]


def _start(tmp_path: Path, script: str, args: list[str], reported: str) -> subprocess.CompletedProcess:
    """The script with a nvidia-smi that reports compute capability `reported` and a nvcc that compiles nothing."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text(
        '#!/bin/sh\ncase "$*" in *--query-gpu=*) echo "$STAND_IN_CC" ;; *) echo stand-in ;; esac\n')
    (bin_ / "nvcc").write_text('#!/bin/sh\n[ "$1" = --version ] && { echo "nvcc: stand-in"; exit 0; }\nexit 1\n')
    for tool in bin_.iterdir():
        tool.chmod(0o755)
    run_dir = tmp_path / "run"
    (run_dir / "out").mkdir(parents=True)
    env = {"PATH": f"{bin_}{os.pathsep}{os.defpath}", "NVCC": str(bin_ / "nvcc"), "STAND_IN_CC": reported,
           "OUT": str(run_dir / "out"), "RESEARCH_RUN_DIR": str(run_dir), "TMPDIR": str(tmp_path)}
    return subprocess.run(["bash", f"{DC}/{script}", *args], cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("script,args,cc,refused", JOBS, ids=[" ".join([s, *a]) for s, a, _, _ in JOBS])
def test_a_card_of_another_compute_capability_stops_the_script(tmp_path, script, args, cc, refused):
    other = "10.0" if cc == 9 else "9.0"
    done = _start(tmp_path, script, args, other)
    assert done.returncode == 2, done.stderr
    assert f"needs compute capability {cc}.x, the device reports '{other}'" in done.stderr


@pytest.mark.parametrize("script,args,cc,refused", JOBS, ids=[" ".join([s, *a]) for s, a, _, _ in JOBS])
def test_its_own_card_passes_the_check_and_reaches_the_build(tmp_path, script, args, cc, refused):
    done = _start(tmp_path, script, args, f"{cc}.0")
    assert done.returncode == refused, (done.stdout, done.stderr)
    assert "needs compute capability" not in done.stderr
