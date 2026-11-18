"""The listed keyed stream (``verity.commitments.randomness``'s ``Key.block``) written as hm96 salts where a kernel reads
them: leaf i's 192 bytes are the stream's blocks 3i .. 3i + 2 at one index under one key. ::

    lib = host_lib(build=DIR)                     # the host twin; device_lib(DIR, arch) on a CUDA device
    fill(lib, out, key, ("y", call), first)       # out[i] = bytes [192 (first + i), 192 (first + i + 1)) of the stream

``out`` is ``(leaves, 192)`` uint8 and contiguous: a numpy array for the host twin, or on a CUDA device any tensor with
``data_ptr()`` (verity imports no tensor runtime), written on the caller's ``stream`` without synchronizing. The host
builds the block's head once a call from the module's own framing; the kernel takes the head's bytes and holds no framing
of its own. A head over ``HEAD_MAX`` bytes (an index whose frames pass 43 bytes) is refused, never truncated. The device
library's target is the caller's (``arch``, such as ``90a``, its card file's): this package can't load ``verity.device``.

Builds are cached by a hash of the sources, flags and compiler version under the caller's ``build`` directory; nvcc and
g++ are the ones on the path.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

from verity.commitments.cuda import SHA512
from verity.commitments.randomness import _STREAM, Index, Key, _frames

__all__ = ["HEAD_MAX", "LEAF_BYTES", "LEAVES_MAX", "Refused", "bind", "device_lib", "fill", "head", "host_lib"]

HERE = Path(__file__).resolve().parent
DEVICE_SOURCES = (SHA512, HERE / "stream.cuh", HERE / "stream.cu")
HOST_SOURCES = (SHA512, HERE / "stream.cuh", HERE / "stream_host.cpp")
NVCC_FLAGS = ("-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared", "-Xptxas", "-v")
CFLAGS = ("-O2", "-std=c++17", "-fPIC", "-shared")
#: stream.cuh's HEAD_MAX: head, counter, padding and length in one 128-byte chunk
HEAD_MAX = 128 - 8 - 1 - 16
LEAF_BYTES = 192
#: stream.cuh's LEAVES_MAX: first + leaves at most this, so the last block's counter is below 2^64
LEAVES_MAX = ((1 << 64) - 1) // 3
#: stream.cuh's REFUSED_*, by the status the library returns
REFUSALS = {-1: ("REFUSED_HEAD", f"the head is over {HEAD_MAX} bytes, so a block would take two compressions"),
            -2: ("REFUSED_RANGE", "the last leaf's counter is not below 2^64")}

vp, u32, u64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64


class Refused(ValueError):
    """A fill the library refused before writing anything, by its status's name."""

    def __init__(self, status: int) -> None:
        self.name, why = REFUSALS[status]
        super().__init__(f"{self.name}: {why}")


def _build(build: Path, tool: str, sources: Sequence[Path], flags: Sequence[str], name: str) -> ctypes.CDLL:
    h = hashlib.sha256()
    for s in sources:
        h.update(s.name.encode() + b"\0" + s.read_bytes() + b"\0")
    version = subprocess.run([tool, "--version"], capture_output=True, text=True, check=True).stdout
    h.update(" ".join(flags).encode() + b"\0" + version.encode())
    lib = Path(build) / h.hexdigest()[:12] / name
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_suffix(f".{os.getpid()}.tmp")
        r = subprocess.run([tool, *flags, "-o", str(tmp), str(sources[-1])], capture_output=True, text=True, check=False)
        lib.with_suffix(".log").write_text(r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError(f"{tool} failed:\n{r.stderr[-4000:]}")
        tmp.replace(lib)
    return bind(lib)


def bind(path: Path) -> ctypes.CDLL:
    """The library built at ``path``, its calls typed (a build of the sources by another builder, such as a test's)."""
    dll = ctypes.CDLL(str(path))
    dll.sxs_fill.argtypes, dll.sxs_fill.restype = [vp, ctypes.c_char_p, u32, u64, u64, vp], ctypes.c_int
    if hasattr(dll, "sxs_compressions"):
        dll.sxs_compressions.argtypes, dll.sxs_compressions.restype = [ctypes.c_int], u64
    dll.path = path  # type: ignore[attr-defined]
    return dll


@cache
def host_lib(build: Path) -> ctypes.CDLL:
    """``stream_host.cpp`` built with the system C++ compiler, cached under ``build``."""
    cxx = next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)
    if cxx is None:
        raise RuntimeError("no C++ compiler on the path")
    return _build(build, cxx, HOST_SOURCES, CFLAGS, "libsha512_stream_host.so")


@cache
def device_lib(build: Path, arch: str) -> ctypes.CDLL:
    """``stream.cu`` built with nvcc for ``sm_<arch>`` (``"90a"``: the H100 and H200), cached under ``build``."""
    tool = shutil.which("nvcc")
    if tool is None:
        raise RuntimeError("no nvcc on the path")
    return _build(build, tool, DEVICE_SOURCES, (*NVCC_FLAGS, f"-gencode=arch=compute_{arch},code=sm_{arch}"),
                  "libsha512_stream_cuda.so")


def head(key: Key, index: Index) -> bytes:
    """Every block's message at ``index`` but its counter: the stream's tag, the key and the index's frames."""
    return _STREAM + key.value + _frames(index)


def fill(lib: ctypes.CDLL, out: Any, key: Key, index: Index, first: int, *, stream: int = 0) -> Any:
    """out[i] = the stream's bytes [192 (first + i), 192 (first + i + 1)) at ``index``, for each of out's rows; returns out."""
    leaves = int(out.shape[0]) if len(out.shape) == 2 else -1
    if leaves < 0 or int(out.shape[1]) != LEAF_BYTES or str(out.dtype) not in ("uint8", "torch.uint8"):
        raise ValueError(f"out is (leaves, {LEAF_BYTES}) uint8, not {tuple(out.shape)} {out.dtype}")
    if hasattr(out, "data_ptr"):
        if not out.is_contiguous() or out.data_ptr() % 8:
            raise ValueError("out is contiguous at an address a multiple of 8")
        ptr = out.data_ptr()
    else:
        if not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError("out is a contiguous, writeable array")
        ptr = out.ctypes.data
    if isinstance(first, bool) or not isinstance(first, int) or first < 0:
        raise ValueError("first is a leaf, an int of at least 0")
    h = head(key, index)
    rc = lib.sxs_fill(stream, h, len(h), first, leaves, ptr) if first < 1 << 64 else -2
    if rc in REFUSALS:
        raise Refused(rc)
    if rc:
        raise RuntimeError(f"fill: CUDA error {rc}")
    return out
