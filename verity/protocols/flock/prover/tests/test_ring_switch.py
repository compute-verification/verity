"""``cuda/ring_switch.cuh``'s tensor-core ring switch against its split kernels, word for word: the fold's 128 sums
(``chunk_ring_fold_rows_bmma`` against ``chunk_ring_fold_rows_split``) and every combined row (``chunk_ring_combine_basis_bmma``
against ``chunk_ring_combine_basis_split``), on drawn witnesses, eq halves and weights, with the eq split at the prover's
d / 2 and away from it, from the smallest split each kernel takes to the fold's blocks sharing their b's across row ranges.
``ring_switch.cu`` runs them, built by nvcc against a Flock b684b12 checkout's ``cuda-ghash``
(``VERITY_ML_FLOCK_CUDA_GHASH``) where a GPU of sm_80 or later is present."""
from __future__ import annotations

import ctypes
import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest

from verity import settings

SOURCE = Path(__file__).resolve().with_name("ring_switch.cu")
#: (log2 of the rows, log2 of the eq's low half): the combine from k = 4 (an MMA tile's 16 rows), the fold from k = 8 (one
#: step's 256 rows) and past 11, where a block takes 2048 rows of the low half
COMBINE = ((8, 4), (9, 4), (12, 4), (12, 6), (12, 10), (16, 8), (20, 10), (21, 10))
FOLD = ((8, 8), (12, 8), (16, 8), (16, 11), (16, 12), (20, 10), (22, 11), (24, 12))


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
    gpu_arch() is None or int(gpu_arch() or 0) < 80 or not GHASH,
    reason="needs nvcc, a CUDA device of sm_80 or later and VERITY_ML_FLOCK_CUDA_GHASH (a Flock b684b12 checkout's cuda-ghash)",
)


@pytest.fixture(scope="module")
def lib(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    arch = gpu_arch()
    out = tmp_path_factory.mktemp("ring") / "ring_switch.so"
    r = subprocess.run(["nvcc", "-x", "cu", "-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared", f"-I{GHASH}",
                        f"-gencode=arch=compute_{arch},code=sm_{arch}", "-o", str(out), str(SOURCE)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"nvcc failed:\n{r.stderr[-4000:]}"
    dll = ctypes.CDLL(str(out))
    for f in (dll.rs_combine_check, dll.rs_fold_check):
        f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.POINTER(ctypes.c_longlong)]
        f.restype = ctypes.c_int
    return dll


@pytest.mark.slow
@needs_gpu
def test_the_tensor_core_ring_switch_is_the_split_one_word_for_word(lib: ctypes.CDLL) -> None:
    """Slow for its nvcc build: the ring switch's kernels, which only a GPU runs."""
    bad = {}
    for name, f, cases in (("combine", lib.rs_combine_check, COMBINE), ("fold", lib.rs_fold_check, FOLD)):
        for d, k in cases:
            for seed in (1, 2):
                out = (ctypes.c_longlong * 1)(-1)
                assert f(d, k, seed, out) == 0, (name, d, k, seed)
                if out[0]:
                    bad[(name, d, k, seed)] = out[0]
    assert not bad, bad
