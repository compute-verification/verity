"""``cuda/zerocheck_fused.cuh``, the zerocheck's round 2 and the tail's first lookahead from the packed witness, word for word
against upstream's round 2 fold (``zerocheck_round2.cuh``) and first lookahead over its tables (``zerocheck_tail.cuh``):
both message sets and the lookahead's folded a and b, under every table mode and chunk width, on both of round 2's eq
paths (per point at small sizes, one eqhi entry per block at large ones) and with the lookahead striding its grid.
``zerocheck_fused.cu`` runs them, built by nvcc against a Flock b684b12 checkout's ``cuda-ghash``
(``VERITY_ML_FLOCK_CUDA_GHASH``) where a GPU is present."""
from __future__ import annotations

import ctypes
import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest

from verity import settings

SOURCE = Path(__file__).resolve().with_name("zerocheck_fused.cu")
#: log2 of round 2's rows: per-point eq through 16, one eqhi entry per block from 17, the lookahead's grid strided at 24
LOG_ROWS = (4, 5, 8, 12, 16, 17, 20, 24)
#: zerocheck_fused.cu's table codes: modes 0 and 1, and mode 2 in chunks of 4 to 7 bits
TABLES = (0, 1, 4, 5, 6, 7)


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
    out = tmp_path_factory.mktemp("zerocheck") / "zerocheck_fused.so"
    r = subprocess.run(["nvcc", "-x", "cu", "-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared", f"-I{GHASH}",
                        f"-gencode=arch=compute_{arch},code=sm_{arch}", "-o", str(out), str(SOURCE)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"nvcc failed:\n{r.stderr[-4000:]}"
    dll = ctypes.CDLL(str(out))
    dll.zf_check.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.POINTER(ctypes.c_longlong)]
    dll.zf_check.restype = ctypes.c_int
    return dll


@pytest.mark.slow
@needs_gpu
def test_the_packed_path_is_round_2_word_for_word(lib: ctypes.CDLL) -> None:
    """Slow for its nvcc build: the fused kernels, which only a GPU runs."""
    bad = {}
    for log_rows in LOG_ROWS:
        for table in TABLES:
            for seed in (1, 2):
                out = (ctypes.c_longlong * 3)()
                assert lib.zf_check(log_rows, table, seed, out) == 0, (log_rows, table, seed)
                if any(out):
                    bad[(log_rows, table, seed)] = {"round2": out[0], "lookahead": out[1], "folded": out[2]}
    assert not bad, bad
