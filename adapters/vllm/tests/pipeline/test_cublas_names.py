"""`row_driver.cublas_names`: a row's stages load cuBLAS by its unversioned name (flashinfer's autotune) from the venv's wheel, so a
stage on a node with a system CUDA maps one cuBLAS, the one the route pin (`driver.cublas_versions`) reads.

Two copies of the C library's zlib stand in for the venv's ``libcublas.so.12`` and a system CUDA's ``libcublas.so``: the loader only
needs an ELF to map, and ``/proc/self/maps`` names each copy by its path."""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from verity_vllm.check.replay.driver import _mapped_library
from verity_vllm.pipeline.row_driver import Stop, cublas_names

# a stage process: torch maps the venv's versioned cuBLAS by its path, then flashinfer's autotune loads the unversioned names
STAGE = """import ctypes, sys
ctypes.CDLL(sys.argv[1]); ctypes.CDLL(sys.argv[2])
ctypes.CDLL("libcublas.so"); ctypes.CDLL("libcublasLt.so")
sys.stdout.write(open("/proc/self/maps").read())
"""


def _elf() -> str:
    name = ctypes.util.find_library("z")
    if not name:
        pytest.skip("no zlib shared library to stand in for cuBLAS")
    ctypes.CDLL(name)
    return next(ln.split()[-1] for ln in open("/proc/self/maps") if os.path.basename(ln.split()[-1]).startswith("libz.so"))


def _node(tmp: Path) -> tuple[str, dict[str, str], Path]:
    """A venv with a cuBLAS wheel, and a system CUDA whose unversioned names are on the loader's path -> (py, env, the wheel's lib dir)."""
    elf = _elf()
    lib = tmp / "venv/lib/python3.12/site-packages/nvidia/cublas/lib"
    system = tmp / "cuda-13.1/lib64"
    for d, ver in ((lib, ".12"), (system, "")):
        d.mkdir(parents=True)
        for n in ("libcublas", "libcublasLt"):
            shutil.copy(elf, d / f"{n}.so{ver}")
    (tmp / "venv/bin").mkdir()
    py = tmp / "venv/bin/python"
    py.symlink_to(sys.executable)
    return str(py), {**os.environ, "LD_LIBRARY_PATH": str(system)}, lib


def _maps(tmp: Path, env: dict[str, str], lib: Path) -> Path:
    out = subprocess.run([sys.executable, "-c", STAGE, str(lib / "libcublas.so.12"), str(lib / "libcublasLt.so.12")], env=env,
                         capture_output=True, text=True, check=True).stdout
    p = tmp / "maps"
    p.write_text(out)
    return p


def test_a_stage_maps_the_venvs_cublas_alone(tmp_path):
    py, env, lib = _node(tmp_path)
    with pytest.raises(LookupError, match="2 libcublas.so"):
        _mapped_library("libcublas.so", str(_maps(tmp_path, env, lib)))
    d = cublas_names(env, py, str(tmp_path))
    assert env["LD_LIBRARY_PATH"] == f"{d}:{tmp_path}/cuda-13.1/lib64"
    assert sorted(os.listdir(d)) == ["libcublas.so", "libcublasLt.so"]
    maps = str(_maps(tmp_path, env, lib))
    assert _mapped_library("libcublas.so", maps) == os.path.realpath(lib / "libcublas.so.12")
    assert _mapped_library("libcublasLt.so", maps) == os.path.realpath(lib / "libcublasLt.so.12")


def test_every_stage_on_one_venv_gets_the_same_environment(tmp_path):
    py, env, _ = _node(tmp_path)
    d = cublas_names(env, py, str(tmp_path))
    first = env["LD_LIBRARY_PATH"]
    assert cublas_names(env, py, str(tmp_path)) == d and env["LD_LIBRARY_PATH"] == first
    other = dict(os.environ)
    assert cublas_names(other, py, str(tmp_path)) == d and other["LD_LIBRARY_PATH"].split(":")[0] == d


def test_a_venv_without_a_cublas_wheel_leaves_the_environment(tmp_path):
    env = {"LD_LIBRARY_PATH": "/usr/local/cuda/lib64"}
    assert cublas_names(env, str(tmp_path / "venv/bin/python"), str(tmp_path)) is None
    assert env == {"LD_LIBRARY_PATH": "/usr/local/cuda/lib64"}


def test_two_libraries_of_one_name_or_a_foreign_directory_are_refused(tmp_path):
    py, env, lib = _node(tmp_path)
    shutil.copy(lib / "libcublas.so.12", lib / "libcublas.so.13")
    with pytest.raises(ValueError, match="not one cuBLAS"):
        cublas_names(dict(env), py, str(tmp_path))
    (lib / "libcublas.so.13").unlink()
    d = cublas_names(dict(env), py, str(tmp_path))
    os.unlink(os.path.join(d, "libcublas.so"))
    os.symlink(f"{tmp_path}/cuda-13.1/lib64/libcublas.so", os.path.join(d, "libcublas.so"))
    with pytest.raises(ValueError, match="holds other than"):
        cublas_names(dict(env), py, str(tmp_path))


def test_a_rows_stages_get_the_names(tmp_path, monkeypatch, capsys):
    from tests.pipeline.test_row import _row
    py, _, lib = _node(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    r = _row(tmp_path, "--py", py)
    r.setup({})
    d = r.child_env["LD_LIBRARY_PATH"].split(":")[0]
    assert os.path.dirname(d) == str(tmp_path) and os.path.realpath(os.path.join(d, "libcublas.so")) == os.path.realpath(lib / "libcublas.so.12")
    shutil.copy(lib / "libcublasLt.so.12", lib / "libcublasLt.so.13")
    with pytest.raises(Stop) as e:
        _row(tmp_path, "--py", py).setup({})
    assert e.value.rc == 3 and "PRECHECK_FAIL_CUBLAS_NAMES" in capsys.readouterr().err
