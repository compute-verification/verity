"""`ops/cuda_headers.sh`: a toolkit without cuSPARSE's or cuSOLVER's headers gets them from the venv's NVIDIA wheels on
CPATH, and a header the toolkit has is left to it."""
import subprocess
from pathlib import Path

HELPER = Path(__file__).resolve().parents[2] / "verity_vllm" / "ops" / "cuda_headers.sh"


def test_the_headers_a_toolkit_lacks_come_from_the_wheels_and_the_rest_from_the_toolkit(tmp_path):
    toolkit, purelib = tmp_path / "cuda" / "include", tmp_path / "site-packages"
    toolkit.mkdir(parents=True)
    (toolkit / "cublas_v2.h").touch()
    for lib, header in (("cusparse", "cusparse.h"), ("cusolver", "cusolverDn.h"), ("cublas", "cublas_v2.h")):
        (purelib / "nvidia" / lib / "include").mkdir(parents=True)
        (purelib / "nvidia" / lib / "include" / header).touch()
    py = tmp_path / "python"
    py.write_text(f"#!/bin/bash\necho {purelib}\n")
    py.chmod(0o755)
    script = f'. {HELPER} && cuda_headers {py} && cuda_headers {py} && echo "${{CPATH:-}}"'
    out = subprocess.run(["bash", "-c", script], env={"CUDA_HOME": str(tmp_path / "cuda"), "PATH": "/usr/bin:/bin"},
                         capture_output=True, text=True, check=True).stdout.strip()
    assert sorted(out.split(":")) == sorted(str(purelib / "nvidia" / lib / "include") for lib in ("cusparse", "cusolver"))


def test_an_interpreter_that_cannot_answer_leaves_cpath_alone(tmp_path):
    (tmp_path / "cuda" / "include").mkdir(parents=True)
    script = f'. {HELPER} && cuda_headers {tmp_path / "no-python"} && echo "[${{CPATH:-}}]"'
    out = subprocess.run(["bash", "-c", script], env={"CUDA_HOME": str(tmp_path / "cuda"), "PATH": "/usr/bin:/bin"},
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"
