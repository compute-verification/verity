"""The row-table kernels of PoUW's `kernels/pouw_cuda/` (`row_table.cu`, `row_table_host.cpp`) built and loaded: each build
keyed by what it compiles (sources, the shared headers, flags and the compiler's version) under ``~/.cache/verity/row_table``
and compiled from a flat copy of those files, so the compiler is handed no path in the tree (`built`); the host's and a
device's libraries (`host_lib`, `device_lib`); the hm96-sha512 byte tables and prefixes the kernels hash with (`hm_tables`,
`leaf_prefixes`). A device's copy of a library, with its side streams, is the device adapter's, as are H1's device tables,
registered rows and receipts, which build on it. nvcc is the one on the path."""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import numpy as np
from verity.commitments import row_table as RT
from verity.commitments.cuda import ROW_COMMIT, SHA512
from verity.protocols.resource_accounting.compute.pouw import prover

__all__ = ["CFLAGS", "CUDA_DIR", "DEVICE_SOURCES", "HOST_SOURCES", "NVCC_FLAGS", "SEG_BYTES", "SHARED", "TAB_WORDS", "Args",
           "built", "device_lib", "hm_tables", "host_lib", "leaf_prefixes", "nvcc", "table_width", "u64_words"]

CUDA_DIR = Path(prover.__file__).resolve().parent / "kernels" / "pouw_cuda"
DEVICE_SOURCES = ("ncp2.cuh", "row_table.cuh", "row_table.cu")
HOST_SOURCES = ("ncp2.cuh", "row_table.cuh", "row_table_host.cpp")
#: the headers `ncp2.cuh` and `row_table.cuh` include by name from outside CUDA_DIR (the GPU's one SHA-512 and row commitment):
#: every build compiles a copy of them beside a copy of its sources (`built`) and is keyed by their bytes
SHARED = (SHA512, ROW_COMMIT)
NVCC_FLAGS = ("-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared")
CFLAGS = ("-O2", "-std=c++17", "-fPIC", "-shared")
BUILD_DIR = Path.home() / ".cache" / "verity" / "row_table"
#: words of a table's layout (`row_table.cuh`'s ``Tab``) and the bytes of a row segment
TAB_WORDS, SEG_BYTES = 14, 1024


class Args(ctypes.Structure):
    """`row_table.cuh`'s ``Args``: the row key's ChaCha20 words, the nonce (the engine step), the table count, the step's domain id."""
    _fields_ = [("key", ctypes.c_uint32 * 8), ("nonce", ctypes.c_uint32 * 3), ("T", ctypes.c_uint32), ("did", ctypes.c_uint64 * 8)]


def built(tool: str, sources: Sequence[str], flags: Sequence[str], name: str, headers: Sequence[Path] = ()) -> Path:
    """``sources``' last (of `CUDA_DIR`) compiled by ``tool`` with ``flags`` into ``name``, keyed by the sources, the
    ``headers`` it includes from elsewhere, the flags and the compiler's version.  It compiles a flat copy of the sources and
    headers in a fresh directory of the system's temporary one, each file under its base name and the last source named by it, so
    the compiler is handed no path in the tree, and ``sources`` and ``headers`` are every file the build reads: an include
    they leave out isn't found.  Two different files with one base name are refused (a file named twice is one file)."""
    files: dict[str, Path] = {}
    for p in (*(CUDA_DIR / s for s in sources), *headers):
        other = files.setdefault(p.name, p)
        if other.resolve() != p.resolve():
            raise ValueError(f"two files named {p.name} among a build's sources and headers, {other} and {p}: the copy it "
                             f"compiles holds one file of each name")
    h = hashlib.sha256()
    for s in sources:
        h.update(s.encode() + b"\0" + (CUDA_DIR / s).read_bytes() + b"\0")
    for p in headers:
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    version = subprocess.run([tool, "--version"], capture_output=True, text=True, check=True).stdout
    h.update(" ".join(flags).encode() + b"\0" + version.encode())
    lib = BUILD_DIR / h.hexdigest()[:12] / name
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_suffix(f".{os.getpid()}.tmp")
        stage = Path(tempfile.mkdtemp(prefix="verity-build."))
        try:
            for p in files.values():
                shutil.copy(p, stage)
            subprocess.run([tool, *flags, "-o", str(tmp), Path(sources[-1]).name], check=True, cwd=stage)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        tmp.replace(lib)
    return lib


def _build(tool: str, sources: Sequence[str], flags: Sequence[str], name: str) -> ctypes.CDLL:
    """The row tables' and registered rows' builds: `built` with the shared headers (`SHARED`), loaded."""
    return ctypes.CDLL(str(built(tool, sources, flags, name, SHARED)))


_u32, _u64 = (np.ctypeslib.ndpointer(t, flags="C_CONTIGUOUS") for t in (np.uint32, np.uint64))
_vp = ctypes.c_void_p


@cache
def host_lib() -> ctypes.CDLL:
    """`row_table_host.cpp` built with the system C++ compiler."""
    cxx = next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)
    if cxx is None:
        raise RuntimeError("no C++ compiler on the path")
    dll = _build(cxx, HOST_SOURCES, CFLAGS, "librow_table_host.so")
    dll.rt_step_host.argtypes = [_u64, _u32, ctypes.c_uint64, ctypes.c_uint32, _u64, _u64, ctypes.POINTER(Args), _u64]
    dll.rt_step_host.restype = ctypes.c_int
    return dll


def nvcc() -> str | None:
    """nvcc on the path."""
    return shutil.which("nvcc")


@cache
def device_lib(arch: str) -> ctypes.CDLL:
    """`row_table.cu` built with nvcc for ``sm_<arch>`` (``"120"``: the RTX PRO 6000)."""
    tool = nvcc()
    if tool is None:
        raise RuntimeError("no nvcc on the path")
    dll = _build(tool, DEVICE_SOURCES, (*NVCC_FLAGS, f"-gencode=arch=compute_{arch},code=sm_{arch}"), "librow_table_cuda.so")
    dll.rt_midstates.argtypes = [_vp, _vp, _vp]
    dll.rt_step.argtypes = [_vp, _vp, _vp, ctypes.c_uint64, ctypes.c_uint32, _vp, _vp, ctypes.POINTER(Args), *[_vp] * 5]
    dll.rt_midstates.restype = dll.rt_step.restype = ctypes.c_int
    return dll


def table_width(n: int) -> int:
    """The leaf count of an ``n``-row table's tree: ``n`` rounded up to a power of two (1 for ``n`` <= 1)."""
    return 1 << max(0, (n - 1).bit_length())


def u64_words(b: bytes) -> np.ndarray:
    """``b`` read as big-endian 64-bit words, the kernels' word order, as a uint64 array."""
    return np.frombuffer(b, dtype=">u8").astype(np.uint64)


@cache
def hm_tables() -> np.ndarray:
    """hm96-sha512's byte tables under the pinned key (`HidingLeaves`' tables), 192 x 256 entries of 8 words."""
    return u64_words(b"".join(t.to_bytes(RT.DIGEST_BYTES, "little") for table in RT.INST.byte_tables(RT.KEY) for t in table))


@cache
def leaf_prefixes() -> np.ndarray:
    """The salt prefix and the pinned key's leaf prefix the kernels hash each row's leaf under, as `u64_words`."""
    return u64_words(RT.INST.salt_prefix + RT.INST.leaf_prefix(RT.KEY))

