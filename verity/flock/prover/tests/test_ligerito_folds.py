"""``cuda/ligerito_folds.cuh``, level 0's folds with their messages and the last out-of-domain glue fused in, word for word
against upstream's fold, message and glue kernels (``ligerito_f256.cuh``, ``introduce_glue.cuh``): every fold's outputs and
message, with the glue and without, from one block's worth of pairs to a grid that strides. ``ligerito_folds.cu`` runs them,
built by nvcc against a Flock b684b12 checkout's ``cuda-ghash`` (``VERITY_ML_FLOCK_CUDA_GHASH``) where a GPU is present."""
from __future__ import annotations

import ctypes
import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest

from verity import settings

SOURCE = Path(__file__).resolve().with_name("ligerito_folds.cu")
#: (log2 of level 0's words, folds): a partial block, one block, many, and a grid that strides from 2^22
CASES = ((4, 2), (5, 3), (9, 5), (14, 5), (20, 5), (23, 5))


@cache
def gpu_arch() -> str | None:
    """The first GPU's compute capability as nvcc's sm number (``"90"``), or None without nvcc or a GPU."""
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return None
    r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True,
                       check=False)
    caps = r.stdout.split() if r.returncode == 0 else []
    return caps[0].replace(".", "") if caps else None


GHASH = settings.VERITY_ML_FLOCK_CUDA_GHASH
needs_gpu = pytest.mark.skipif(
    gpu_arch() is None or not GHASH,
    reason="needs nvcc, a CUDA device and VERITY_ML_FLOCK_CUDA_GHASH (a Flock b684b12 checkout's cuda-ghash); about a minute",
)


@pytest.fixture(scope="module")
def lib(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    arch = gpu_arch()
    out = tmp_path_factory.mktemp("ligerito") / "ligerito_folds.so"
    r = subprocess.run(["nvcc", "-x", "cu", "-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared", f"-I{GHASH}",
                        f"-gencode=arch=compute_{arch},code=sm_{arch}", "-o", str(out), str(SOURCE)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"nvcc failed:\n{r.stderr[-4000:]}"
    dll = ctypes.CDLL(str(out))
    dll.lf_check.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.POINTER(ctypes.c_ulonglong)]
    dll.lf_check.restype = ctypes.c_int
    return dll


@pytest.mark.slow
@needs_gpu
def test_fused_folds_are_upstreams_word_for_word(lib: ctypes.CDLL) -> None:
    """Slow for its nvcc build: the fused kernels, which only a GPU runs."""
    bad = {}
    for log_n, folds in CASES:
        for glue in (0, 1):
            for seed in (1, 2):
                out = (ctypes.c_ulonglong * 2)()
                assert lib.lf_check(log_n, folds, glue, seed, out) == 0, (log_n, folds, glue, seed)
                if any(out):
                    bad[(log_n, folds, glue, seed)] = {"folds": out[0], "messages": out[1]}
    assert not bad, bad
