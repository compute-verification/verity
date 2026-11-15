"""``cuda/async_copies.cuh``, the provers' copies between the host and the default stream: queued reads come back in stream
order and only at ``hc_wait``, an upload takes its bytes at the call, staging wraps after a wait, a copy past ``HC_DIRECT``
stays in order with the queued ones, a symbol upload lands between the kernels around it, and ``hc_drop`` leaves the reads'
destinations alone. ``async_copies.cu`` runs them, built by nvcc where a GPU is present."""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import pytest

from test_zerocheck_fused import gpu_arch

SOURCE = Path(__file__).resolve().with_name("async_copies.cu")
CUDA = Path(__file__).resolve().parents[1] / "cuda"
CHECKS = ("stream order", "upload at the call", "staging wrap", "direct copies", "symbol uploads", "drop")


@pytest.mark.slow
@pytest.mark.skipif(gpu_arch() is None, reason="needs nvcc and a CUDA device")
def test_async_copies(tmp_path: Path) -> None:
    """Slow for its nvcc build: the copies only a GPU makes."""
    arch = gpu_arch()
    out = tmp_path / "async_copies.so"
    r = subprocess.run(["nvcc", "-x", "cu", "-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared", f"-I{CUDA}",
                        f"-gencode=arch=compute_{arch},code=sm_{arch}", "-o", str(out), str(SOURCE)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"nvcc failed:\n{r.stderr[-4000:]}"
    dll = ctypes.CDLL(str(out))
    dll.ac_check.restype = ctypes.c_int
    bad = dll.ac_check()
    assert bad >= 0, "cudaMalloc failed"
    assert bad == 0, [name for i, name in enumerate(CHECKS) if bad >> i & 1]
