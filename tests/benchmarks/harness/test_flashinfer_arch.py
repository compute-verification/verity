"""flashinfer_arch.sh, which the network runners source on a held card, exports FlashInfer's build for the card's compute
capability, as a stand-in nvidia-smi gives it, unless one is set."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BASH = shutil.which("bash")


def _sourced(tmp_path: Path, cap: str, **env: str) -> subprocess.CompletedProcess:
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    for name in ("head", "tr"):
        if not (tools / name).exists():
            (tools / name).symlink_to(shutil.which(name))
    (tools / "nvidia-smi").write_text(f'#!{BASH}\necho "$*" > {tmp_path}/asked\necho " {cap}"\n')
    (tools / "nvidia-smi").chmod(0o755)
    return subprocess.run([BASH, "-c", '. benchmarks/harness/flashinfer_arch.sh && echo "$FLASHINFER_CUDA_ARCH_LIST"'],
                          cwd=ROOT, env={"PATH": str(tools), **env}, capture_output=True, text=True)


@pytest.mark.parametrize(("cap", "arch"), [("12.0", "12.0f"), ("10.0", "10.0a"), ("9.0", "9.0a")])
def test_each_capability_has_its_build(tmp_path: Path, cap: str, arch: str) -> None:
    ran = _sourced(tmp_path, cap, CUDA_VISIBLE_DEVICES="GPU-a,GPU-b")
    assert (ran.returncode, ran.stdout) == (0, f"{arch}\n"), ran.stderr
    assert (tmp_path / "asked").read_text().split()[:2] == ["-i", "GPU-a"]


def test_an_unknown_capability_is_refused_by_name(tmp_path: Path) -> None:
    ran = _sourced(tmp_path, "8.9")
    assert ran.returncode == 2 and ran.stdout == ""
    assert "compute capability '8.9'" in ran.stderr and "set FLASHINFER_CUDA_ARCH_LIST" in ran.stderr


def test_a_set_build_wins(tmp_path: Path) -> None:
    ran = _sourced(tmp_path, "8.9", FLASHINFER_CUDA_ARCH_LIST="8.9")
    assert (ran.returncode, ran.stdout) == (0, "8.9\n") and not (tmp_path / "asked").exists()
