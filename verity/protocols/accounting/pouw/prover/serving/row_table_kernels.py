"""The row-table kernels of PoUW's `kernels/pouw_cuda/` (`row_table.cu`, `row_table_host.cpp`) built and loaded: each build
keyed by what it compiles (sources, the shared headers, flags and the compiler's version) under ``~/.cache/verity/row_table``
and compiled from a flat copy of those files, so the compiler is handed no path in the tree (`built`); the host's and a
device's libraries (`host_lib`, `device_lib`); the hm96-sha512 byte tables and prefixes the kernels hash with (`_hm`,
`_prefixes`); and one CUDA device's copy of a library with its side streams (`DeviceLibrary`). H1's device tables,
registered rows and receipts build on it, and so does the vLLM adapter's Commit (its ``commit.row_tables_device``). nvcc is
the one on the path."""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from verity.commitments import row_table as RT
from verity.commitments.cuda import ROW_COMMIT, SHA512
from verity.protocols.accounting.pouw import prover

__all__ = ["CFLAGS", "CUDA_DIR", "DEVICE_SOURCES", "HOST_SOURCES", "NVCC_FLAGS", "SEG_BYTES", "SHARED", "TAB_WORDS", "Args",
           "DeviceLibrary", "built", "device_lib", "host_lib", "nvcc", "side_stream"]

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


def _width(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def _words(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=">u8").astype(np.uint64)


@cache
def _hm() -> np.ndarray:
    """hm96-sha512's byte tables under the pinned key (`HidingLeaves`' tables), 192 x 256 entries of 8 words."""
    return _words(b"".join(t.to_bytes(RT.DIGEST_BYTES, "little") for table in RT.INST.byte_tables(RT.KEY) for t in table))


@cache
def _prefixes() -> np.ndarray:
    return _words(RT.INST.salt_prefix + RT.INST.leaf_prefix(RT.KEY))


# --- a device's library ---------------------------------------------------------------------------------------------------

class DeviceLibrary:
    """One CUDA device's build of a kernel library over hm96-sha512 (``lib(arch)``: this module's `device_lib`, or H1's
    `tables_device` and `registered_device`'s), its copies of the byte tables and midstates (the library's ``midstates``
    kernel over `_prefixes`), the side streams its kernels hash on (``stream`` the first, `next_stream` each in turn) and the
    identity rank order its tables share (`ranks`).  `of` builds one per library and device, once a process."""

    _of: dict[tuple[Callable[[str], ctypes.CDLL], int], DeviceLibrary] = {}

    def __init__(self, torch: Any, device: Any, lib: Callable[[str], ctypes.CDLL], midstates: str, streams: int) -> None:
        major, minor = torch.cuda.get_device_capability(device)
        self.lib = lib(f"{major}{minor}")
        self.streams, self.turn = [torch.cuda.Stream(device) for _ in range(streams)], 0
        self.stream = self.streams[0]
        with torch.cuda.stream(self.stream):
            self.hm = torch.from_numpy(_hm().view(np.int64)).to(device)
            pre = torch.from_numpy(_prefixes().view(np.int64)).to(device)
            self.ms = torch.empty(16, dtype=torch.int64, device=device)
            rc = getattr(self.lib, midstates)(self.stream.cuda_stream, pre.data_ptr(), self.ms.data_ptr())
        if rc:
            raise RuntimeError(f"{midstates}: CUDA error {rc}")
        self.stream.synchronize()
        self.perm = torch.arange(0, dtype=torch.int32, device=device)

    @classmethod
    def of(cls, torch: Any, index: int, lib: Callable[[str], ctypes.CDLL] = device_lib, midstates: str = "rt_midstates",
           streams: int = 1) -> DeviceLibrary:
        if (lib, index) not in cls._of:
            cls._of[lib, index] = cls(torch, torch.device("cuda", index), lib, midstates, streams)
        return cls._of[lib, index]

    def next_stream(self) -> Any:
        self.turn = (self.turn + 1) % len(self.streams)
        return self.streams[self.turn]

    def ranks(self, torch: Any, n: int) -> Any:
        """The identity rank order over at least ``n`` rows."""
        if len(self.perm) < n:
            self.perm = torch.arange(max(n, 2 * len(self.perm)), dtype=torch.int32, device=self.perm.device)
        return self.perm


def side_stream(device: Any = None) -> Any:
    """The side stream steps on ``device`` (default: the current CUDA device) hash on."""
    import torch

    return DeviceLibrary.of(torch, torch.cuda.current_device() if device is None else torch.device(device).index).stream
