"""`ops/cuda_headers.sh`: a toolkit without cuSPARSE's or cuSOLVER's headers gets them from the venv's NVIDIA wheels on
CPATH, with one line naming those directories; a header the toolkit has is left to it, and a header in neither place is
refused in one line naming it and the two places looked."""
import subprocess
from pathlib import Path

HELPER = Path(__file__).resolve().parents[2] / "verity_vllm" / "ops" / "cuda_headers.sh"


def _toolkit_and_wheels(tmp_path: Path, wheels: tuple[tuple[str, str], ...]) -> tuple[Path, Path]:
    toolkit, purelib = tmp_path / "cuda" / "include", tmp_path / "site-packages"
    toolkit.mkdir(parents=True)
    for header in ("cublas_v2.h", "cublasLt.h"):
        (toolkit / header).touch()
    for lib, header in wheels:
        (purelib / "nvidia" / lib / "include").mkdir(parents=True, exist_ok=True)
        (purelib / "nvidia" / lib / "include" / header).touch()
    py = tmp_path / "python"
    py.write_text(f"#!/bin/bash\necho {purelib}\n")
    py.chmod(0o755)
    return py, purelib


def _run(tmp_path: Path, script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], env={"CUDA_HOME": str(tmp_path / "cuda"), "PATH": "/usr/bin:/bin"},
                          capture_output=True, text=True)


def test_the_headers_a_toolkit_lacks_come_from_the_wheels_and_the_rest_from_the_toolkit(tmp_path):
    py, purelib = _toolkit_and_wheels(tmp_path, (("cusparse", "cusparse.h"), ("cusolver", "cusolverDn.h"),
                                                 ("cublas", "cublas_v2.h")))
    r = _run(tmp_path, f'. {HELPER} && cuda_headers {py} && cuda_headers {py} && echo "${{CPATH:-}}"')
    assert r.returncode == 0, r.stderr
    said, cpath = r.stdout.strip().splitlines()
    dirs = sorted(str(purelib / "nvidia" / lib / "include") for lib in ("cusparse", "cusolver"))
    assert sorted(cpath.split(":")) == dirs
    assert said.startswith("cuda_headers: ") and sorted(said.split(":")[-1].split()) == dirs


def test_a_header_in_neither_the_toolkit_nor_the_wheels_is_refused_in_one_line(tmp_path):
    py, purelib = _toolkit_and_wheels(tmp_path, (("cusolver", "cusolverDn.h"),))
    r = _run(tmp_path, f'. {HELPER} && cuda_headers {py}')
    assert r.returncode == 1
    assert r.stderr.splitlines() == [f"cuda_headers: cusparse.h is in neither {tmp_path / 'cuda'}/include nor the NVIDIA "
                                     f"wheels at {purelib}/nvidia/*/include"]


def test_an_interpreter_that_cannot_answer_leaves_cpath_alone(tmp_path):
    (tmp_path / "cuda" / "include").mkdir(parents=True)
    r = _run(tmp_path, f'. {HELPER} && cuda_headers {tmp_path / "no-python"} && echo "[${{CPATH:-}}]"')
    assert r.stdout.strip() == "[]"
