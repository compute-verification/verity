"""``cuda/zc_round1.cuh``, the zerocheck's round 1 from the packed witness, word for word against upstream's table kernel
(``zerocheck_round1_cpustyle.cuh``): the 64 a·b sums and the 64 c sums, at sizes from one outer chunk to grids its warps
stride several times, from ``launch_zc_round1`` (``zc_round1_sm90`` on sm_90, ``zc_round1_zlin`` elsewhere) and from each
kernel it can take on this GPU; and the lincheck's z transpose (``cuda/zlin_transpose.cuh``) byte for byte against the
byte-at-a-time kernel, with nothing written past its end.
``zc_round1.cu`` runs them, built by nvcc against a Flock b684b12 checkout's ``cuda-ghash``
(``VERITY_ML_FLOCK_CUDA_GHASH``) where a GPU is present."""
from __future__ import annotations

import ctypes
import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest

from verity import settings

SOURCE = Path(__file__).resolve().with_name("zc_round1.cu")
#: outer chunks: one, a partial warp step's worth, and past a resident grid's warps (an H200's 132 SMs hold 3,168)
N_OUT = (1, 3, 64, 1000, 5003, 65543)
#: (k_log, stripes, blocks): one thread, a partial block, and grids that stride over several blocks' worth
ZT_CASES = ((7, 1, 4096), (7, 1000, 4096), (10, 333, 3), (14, 77, 5), (14, 4096, 4096))


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
    out = tmp_path_factory.mktemp("zc_round1") / "zc_round1.so"
    r = subprocess.run(["nvcc", "-x", "cu", "-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared", f"-I{GHASH}",
                        f"-gencode=arch=compute_{arch},code=sm_{arch}", "-o", str(out), str(SOURCE)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"nvcc failed:\n{r.stderr[-4000:]}"
    dll = ctypes.CDLL(str(out))
    dll.zr_check.argtypes = [ctypes.c_longlong, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
    dll.zr_check.restype = ctypes.c_int
    dll.zr_takes_sm90.restype = ctypes.c_int
    dll.zt_check.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_int, ctypes.c_uint64, ctypes.POINTER(ctypes.c_longlong)]
    dll.zt_check.restype = ctypes.c_int
    return dll


@pytest.mark.slow
@needs_gpu
def test_round_1_from_the_packed_witness_is_the_table_kernels_word_for_word(lib: ctypes.CDLL) -> None:
    """Slow for its nvcc build: round 1's kernels, which only a GPU runs."""
    assert bool(lib.zr_takes_sm90()) == (gpu_arch() == "90")
    paths = {"launch_zc_round1": 0, "zc_round1_zlin": 1, **({"zc_round1_sm90": 2} if lib.zr_takes_sm90() else {})}
    bad = {}
    for name, path in paths.items():
        for n_out in N_OUT:
            for seed in (1, 2):
                out = (ctypes.c_longlong * 2)()
                assert lib.zr_check(n_out, seed, path, out) == 0, (name, n_out, seed)
                if any(out):
                    bad[(name, n_out, seed)] = {"ab": out[0], "c": out[1]}
    assert not bad, bad


@pytest.mark.slow
@needs_gpu
def test_the_z_transpose_is_the_bytewise_kernels_byte_for_byte(lib: ctypes.CDLL) -> None:
    """Slow for its nvcc build (shared with round 1's test): the transpose, which only a GPU runs."""
    bad = {}
    for k_log, stripes, blocks in ZT_CASES:
        out = ctypes.c_longlong()
        assert lib.zt_check(k_log, stripes, blocks, 3, ctypes.byref(out)) == 0, (k_log, stripes, blocks)
        if out.value:
            bad[(k_log, stripes, blocks)] = out.value
    assert not bad, bad
