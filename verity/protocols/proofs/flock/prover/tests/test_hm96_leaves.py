"""``cuda/sha512.cuh``'s ``launch_hm96_finish`` (``hm96_mask_leaves``, then ``hm96_finish_leaves``), the device's hm96-sha512
tree leaves, byte for byte against
``verity.commitments.hm96.batch.tree_leaves``: at both block sizes its launches use (``launch_merkle512``'s 256 and
``flock_cuda_hm96_leaves``' 128), over a partial last block, from a base past leaf 0 (the digests below it untouched) and
across many blocks; and the block sizes ``hm96_tpb_fits`` refuses, which ``launch_merkle512`` aborts on. Whole salted trees
whose salts ``hm96_copy_ahead`` copied before their builds are the trees whose pieces copy theirs beside the build.
``hm96_leaves.cu`` runs it, built by nvcc where a GPU is present."""
from __future__ import annotations

import ctypes
import shutil
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from verity.commitments.hm96 import SHA512, batch

SOURCE = Path(__file__).resolve().with_name("hm96_leaves.cu")
#: (leaves, base): one leaf; a partial block; a base past leaf 0 inside a block and past one; many blocks
CASES = ((1, 0), (5, 0), (257, 0), (1000, 300), (4099, 1), ((1 << 13) + 3, 0))
TPB = (128, 256)
#: not a multiple of 32, or past HM96_TPB
REFUSED_TPB = (0, 16, 48, 288, 512)


@cache
def gpu_arch() -> str | None:
    """The first GPU's compute capability as nvcc's sm number (``"90"``), or None without nvcc or a GPU."""
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return None
    r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True,
                       check=False)
    caps = r.stdout.split() if r.returncode == 0 else []
    return caps[0].replace(".", "") if caps else None


needs_gpu = pytest.mark.skipif(gpu_arch() is None, reason="needs nvcc and a CUDA device")


@pytest.fixture(scope="module")
def lib(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    arch = gpu_arch()
    out = tmp_path_factory.mktemp("hm96") / "hm96_leaves.so"
    r = subprocess.run(["nvcc", "-x", "cu", "-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared",
                        f"-gencode=arch=compute_{arch},code=sm_{arch}", "-o", str(out), str(SOURCE)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"nvcc failed:\n{r.stderr[-4000:]}"
    dll = ctypes.CDLL(str(out))
    dll.hm96_leaves.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int,
                                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    dll.hm96_leaves.restype = ctypes.c_int
    dll.hm96_trees.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int, ctypes.c_char_p,
                               ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    dll.hm96_trees.restype = ctypes.c_int
    return dll


@pytest.mark.slow
@needs_gpu
def test_the_device_leaves_are_the_reference_leaves(lib: ctypes.CDLL) -> None:
    """Slow for its nvcc build: the leaves' kernel, which only a GPU runs."""
    key = SHA512.key_from_label(b"test_hm96_leaves")
    tables = b"".join(v.to_bytes(SHA512.digest_bytes, "little") for t in SHA512.byte_tables(key) for v in t)
    prefixes = SHA512.salt_prefix + SHA512.leaf_prefix(key)
    rng = np.random.default_rng(7)
    bad = {}
    for n, base in CASES:
        x = rng.integers(0, 256, 64 * n, np.uint8).tobytes()
        salts = rng.integers(0, 256, 192 * (n - base), np.uint8).tobytes()
        want = x[:64 * base] + b"".join(batch.tree_leaves(SHA512, key, x[64 * base:], salts))
        for tpb in TPB:
            out = ctypes.create_string_buffer(64 * n)
            assert lib.hm96_leaves(x, salts, n, base, tpb, prefixes, tables, out) == 0, (n, base, tpb)
            got = out.raw
            wrong = [i for i in range(n) if got[64 * i:64 * (i + 1)] != want[64 * i:64 * (i + 1)]]
            if wrong:
                bad[(n, base, tpb)] = wrong[:8]
    assert not bad, bad
    out = ctypes.create_string_buffer(64)
    taken = [tpb for tpb in REFUSED_TPB if lib.hm96_leaves(bytes(64), bytes(192), 1, 0, tpb, prefixes, tables, out) != 1]
    assert not taken, f"block sizes hm96_finish_leaves can't take, not refused: {taken}"


#: level 0's tree, then trees of two pieces, of less than one, and of exactly one (HM96_CHUNK leaves)
TREES = (1 << 10, 1 << 19, 1 << 9, 1 << 18)


@pytest.mark.slow
@needs_gpu
def test_trees_copied_ahead_are_the_trees_copied_beside(lib: ctypes.CDLL) -> None:
    """Slow for its nvcc build: launch_merkle512's salted trees, which only a GPU builds."""
    key = SHA512.key_from_label(b"test_hm96_trees")
    tables = b"".join(v.to_bytes(SHA512.digest_bytes, "little") for t in SHA512.byte_tables(key) for v in t)
    prefixes = SHA512.salt_prefix + SHA512.leaf_prefix(key)
    rng = np.random.default_rng(11)
    x = rng.integers(0, 256, 64 * max(TREES), np.uint8).tobytes()
    salts = rng.integers(0, 256, 192 * sum(TREES), np.uint8).tobytes()
    leaves = (ctypes.c_longlong * len(TREES))(*TREES)
    nodes = 64 * sum(2 * n - 1 for n in TREES)
    built = []
    for ahead in (0, 1):
        out = ctypes.create_string_buffer(nodes)
        assert lib.hm96_trees(x, 64, leaves, len(TREES), salts, ahead, prefixes, tables, out) == 0, ahead
        built.append(out.raw)
    beside, copied_ahead = built
    assert beside != bytes(nodes)
    wrong = [i for i in range(nodes // 64) if beside[64 * i:64 * (i + 1)] != copied_ahead[64 * i:64 * (i + 1)]]
    assert not wrong, f"{len(wrong)} nodes differ, first {wrong[:8]}"
