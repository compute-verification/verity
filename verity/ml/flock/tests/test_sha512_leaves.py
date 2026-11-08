"""``cuda/sha512.cuh``'s leaf hashing on the device against hashlib's SHA-512: ``launch_merkle512``'s leaves and tree, its
staged kernel for every ``leaf_size % 16 == 0`` (every tail ``leaf_size % 128``, on both sides of the padding's one-block
limit, and level 0's leaves: ``(w + 4) * 16`` bytes under M1 and ``w * 16`` under M0), and its byte kernel, the path of
unaligned data. ``sha512_leaves.cu`` runs them, built by nvcc where a GPU is present."""
from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().with_name("sha512_leaves.cu")
#: every tail from none to 112 bytes past zero, one and two chunks, and level 0's leaves for w = 2^k lanes, k = 0..7
LEAF_SIZES = sorted({*range(16, 400, 16), *((2**k + 4) * 16 for k in range(8)), *(2**k * 16 for k in range(8))})
COUNTS = (1, 31, 33, 257)
PATHS = {"launch_merkle512": 0, "bytes": 1, "staged": 2}


@cache
def gpu_arch() -> str | None:
    """The first GPU's compute capability as nvcc's sm number (``"90"``), or None without nvcc or a GPU."""
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return None
    r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True,
                       check=False)
    caps = r.stdout.split() if r.returncode == 0 else []
    return caps[0].replace(".", "") if caps else None


@pytest.fixture(scope="module")
def lib(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    arch = gpu_arch()
    out = tmp_path_factory.mktemp("leaves") / "sha512_leaves.so"
    r = subprocess.run(["nvcc", "-x", "cu", "-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared",
                        f"-gencode=arch=compute_{arch},code=sm_{arch}", "-o", str(out), str(SOURCE)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"nvcc failed:\n{r.stderr[-4000:]}"
    dll = ctypes.CDLL(str(out))
    dll.flock_tree.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    dll.flock_tree.restype = ctypes.c_int
    return dll


def _run(lib: ctypes.CDLL, path: str, data: bytes, n: int, leaf_size: int, offset: int = 0) -> list[bytes]:
    digests = 2 * n - 1 if path == "launch_merkle512" else n
    out = ctypes.create_string_buffer(64 * digests)
    assert lib.flock_tree(PATHS[path], data, n, leaf_size, offset, out) == 0
    return [out.raw[64 * i:64 * i + 64] for i in range(digests)]


def _leaves(n: int, leaf_size: int) -> tuple[bytes, list[bytes]]:
    data = hashlib.shake_256(b"flock leaves %d %d" % (n, leaf_size)).digest(n * leaf_size)
    return data, [hashlib.sha512(data[i * leaf_size:(i + 1) * leaf_size]).digest() for i in range(n)]


def _tree(leaves: list[bytes]) -> list[bytes]:
    nodes, level = list(leaves), leaves
    while len(level) > 1:
        level = [hashlib.sha512(level[2 * j] + level[2 * j + 1]).digest() for j in range(len(level) // 2)]
        nodes += level
    return nodes


@pytest.mark.slow
@pytest.mark.skipif(gpu_arch() is None, reason="no nvcc or no CUDA device")
def test_every_leaf_path_is_sha512(lib: ctypes.CDLL) -> None:
    """Slow for its nvcc build: the device's leaf kernels, which only a GPU runs."""
    bad = {}
    for leaf_size in LEAF_SIZES:
        for n in COUNTS:
            data, want = _leaves(n, leaf_size)
            for path in PATHS:
                got = _run(lib, path, data, n, leaf_size)[:n]
                if k := sum(g != w for g, w in zip(got, want)):
                    bad[(path, leaf_size, n)] = k
    assert not bad, bad


@pytest.mark.slow
@pytest.mark.skipif(gpu_arch() is None, reason="no nvcc or no CUDA device")
@pytest.mark.parametrize("leaf_size", [80, 1024, 1088, 1136])
@pytest.mark.parametrize("offset", [0, 8])
def test_the_tree_is_sha512_aligned_or_not(lib: ctypes.CDLL, leaf_size: int, offset: int) -> None:
    """Slow for its nvcc build: the dispatcher on aligned data (the staged kernel) and unaligned (the byte kernel)."""
    data, want = _leaves(64, leaf_size)
    assert _run(lib, "launch_merkle512", data, 64, leaf_size, offset) == _tree(want)
