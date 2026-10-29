"""`pouw_h1_registered.commit`'s hashing on the GPU: a registered value's ``b || c`` strings and every level of its
frame-v3-sha512 tree, computed by `protocol_options/pouw_cuda/registered_rows.cu` from the rows' words, the salts drawn on
the device (ChaCha20 under the run's key at the value's nonce, `audit.hm96_salt`) or given, under the domain id the host
computes (`registered.domain`).

`device_commit` copies the rows to the device in chunks of at most `CHUNK_BYTES` (rows already there, as a call's z/y
records are, it reads in place), hashes each row (its ``sha512/row/v2`` digest, salt, hm96-sha512 string, tree leaf and
frame leaf) and then the pads and levels, on a stream of its own, and returns the strings and levels as `Registered`
holds them (64-byte nodes, 128-byte strings, as bytes). `device_hash` launches the same and returns at once, its
`Hashing` reading them out once the stream is done; a public table's salts it derives on that stream, in the buffer it
hashes from, and rows written there (under `hashing`) it reads with no wait. `host_commit` runs
the same functions on the CPU (`registered_rows_host.cpp`, built with the system C++ compiler): the twin the CPU tests
hold to `pouw_h1_registered.commit`'s NumPy path on machines without a GPU. Builds share the Commit's row-table builder
(`row_tables_device._build`, keyed by sources, flags and compiler) and its per-device state (`row_tables_device.Device`);
nvcc is `pouw_device`'s (its `TOOLCHAIN`'s, or the one on the path).
"""
from __future__ import annotations

import ctypes
import shutil
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any

import numpy as np

from verity_vllm.commit import row_tables_device as RTD
from verity_vllm.protocols.pouw.h1.nodes import Nodes

__all__ = ["CHUNK_BYTES", "Hashing", "Pub", "Table", "available", "device_commit", "device_hash", "device_lib",
           "device_public_salts", "hashing", "host_commit", "host_lib", "host_public_salts", "nvcc", "pub", "table"]

DEVICE_SOURCES = ("ncp2.cuh", "row_table.cuh", "registered_rows.cuh", "registered_rows.cu")
HOST_SOURCES = ("ncp2.cuh", "row_table.cuh", "registered_rows.cuh", "registered_rows_host.cpp")
#: bytes of rows a chunk copies to the device
CHUNK_BYTES = 1 << 28
#: rows a value may have: row j's salt is ChaCha20 blocks 3j .. 3j + 2 under a 32-bit low counter word
MAX_ROWS = (1 << 32) // 3
#: a public salt head's words (`registered_rows.cuh`'s ``PUB_WORDS``)
PUB_WORDS = 40


class Table(ctypes.Structure):
    """`registered_rows.cuh`'s ``Table``."""
    _fields_ = [("rows", ctypes.c_uint64), ("salts", ctypes.c_uint64), ("n", ctypes.c_uint32), ("row_bytes", ctypes.c_uint32),
                ("W", ctypes.c_uint32), ("depth", ctypes.c_uint32), ("key", ctypes.c_uint32 * 8), ("nonce", ctypes.c_uint32 * 4),
                ("did", ctypes.c_uint64 * 8)]


def table(n: int, row_bytes: int, key: bytes, nonce: int, salted: bool, domain_id: bytes) -> Table:
    """The layout of a value of ``n`` rows of ``row_bytes`` bytes under ``domain_id``, its salts given (``salted``) or
    ChaCha20's under the 32-byte ``key`` at the u64 ``nonce``; the rows' and salts' addresses zero until bound."""
    if not 0 < n <= MAX_ROWS or row_bytes < 1 or len(domain_id) != 64:
        raise ValueError(f"{n} rows of {row_bytes} bytes under a {len(domain_id)}-byte domain id: not 1 to {MAX_ROWS} rows "
                         "under a 64-byte id")
    t = Table()
    t.n, t.row_bytes = n, row_bytes
    t.W = 1 << max(0, (n - 1).bit_length())
    t.depth = t.W.bit_length() - 1
    if not salted:
        if len(key) != 32 or not 0 <= nonce < 1 << 64:
            raise ValueError("ChaCha20's salts take a 32-byte key and a u64 nonce")
        t.key[:] = np.frombuffer(key, "<u4").tolist()
        t.nonce[:] = [0, nonce & 0xFFFFFFFF, nonce >> 32, 0]
    t.did[:] = np.frombuffer(domain_id, ">u8").tolist()
    return t


class Pub(ctypes.Structure):
    """`registered_rows.cuh`'s ``Pub``."""
    _fields_ = [("w", ctypes.c_uint64 * PUB_WORDS), ("len", ctypes.c_uint32), ("n", ctypes.c_uint32), ("first", ctypes.c_uint64)]


def pub(head: bytes, first: int, n: int) -> Pub:
    """The layout of ``n`` public salts from position ``first`` under the salt head ``head`` (at most 320 bytes)."""
    if len(head) > 8 * PUB_WORDS or not 0 < n < 1 << 32 or first < 0 or first + n > 1 << 64:
        raise ValueError(f"{n} public salts from {first} under a {len(head)}-byte head: at most {8 * PUB_WORDS} bytes, "
                         "1 to 2^32 - 1 salts, u64 positions")
    p = Pub()
    p.w[:] = np.frombuffer(head.ljust(8 * PUB_WORDS, b"\0"), ">u8").tolist()
    p.len, p.n, p.first = len(head), n, first
    return p


def nvcc() -> str | None:
    """`pouw_device`'s nvcc: its configured `TOOLCHAIN`'s, or nvcc on the path."""
    from verity_vllm.protocols.pouw import device as pouw_device

    return pouw_device.nvcc()


_vp, _u64 = ctypes.c_void_p, np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS")


@cache
def host_lib() -> ctypes.CDLL:
    """`registered_rows_host.cpp` built with the system C++ compiler."""
    cxx = next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)
    if cxx is None:
        raise RuntimeError("no C++ compiler on the path")
    dll = RTD._build(cxx, HOST_SOURCES, RTD.CFLAGS, "libregistered_rows_host.so")
    dll.rr_commit_host.argtypes = [ctypes.POINTER(Table), _u64, _u64, _u64, _u64]
    dll.rr_commit_host.restype = ctypes.c_int
    dll.rr_public_salts_host.argtypes = [ctypes.POINTER(Pub), _u64]
    dll.rr_public_salts_host.restype = ctypes.c_int
    return dll


@cache
def device_lib(arch: str) -> ctypes.CDLL:
    """`registered_rows.cu` built with nvcc for ``sm_<arch>`` (``"120"``: the RTX PRO 6000)."""
    tool = nvcc()
    if tool is None:
        raise RuntimeError("no nvcc (pouw_device.TOOLCHAIN, or on the path)")
    dll = RTD._build(tool, DEVICE_SOURCES, (*RTD.NVCC_FLAGS, f"-gencode=arch=compute_{arch},code=sm_{arch}"),
                     "libregistered_rows_cuda.so")
    dll.rr_midstates.argtypes = [_vp, _vp, _vp]
    dll.rr_leaves.argtypes = [_vp, ctypes.POINTER(Table), ctypes.c_uint32, ctypes.c_uint32, _vp, _vp, _vp, _vp]
    dll.rr_tree.argtypes = [_vp, ctypes.POINTER(Table), _vp]
    dll.rr_public_salts.argtypes = [_vp, ctypes.POINTER(Pub), _vp]
    dll.rr_midstates.restype = dll.rr_leaves.restype = dll.rr_tree.restype = dll.rr_public_salts.restype = ctypes.c_int
    return dll


def _rows(words: np.ndarray) -> np.ndarray:
    words = np.require(words, "<u2", ["C", "W"])
    return words.view(np.uint8).reshape(words.shape[0], 2 * words.shape[1])


def _salts(row_salts: bytes, n: int) -> np.ndarray:
    if len(row_salts) != 192 * n:
        raise ValueError(f"one 192-byte salt per row: {len(row_salts)} bytes for {n} rows")
    return np.frombuffer(row_salts, np.uint8).copy()


def _out(bcs: np.ndarray, levels: np.ndarray, W: int) -> tuple[Nodes, tuple[Nodes, ...]]:
    """The device's big-endian words as `Registered`'s ``bcs`` and ``levels``, each node's bytes made on read."""
    nodes = levels.byteswap().view(np.uint8)
    out, off, w = [], 0, W
    while True:
        out.append(Nodes(nodes[64 * off:64 * (off + w)], 64))
        off += w
        if w == 1:
            break
        w //= 2
    return Nodes(bcs.byteswap().view(np.uint8), 128), tuple(out)


def host_commit(words: np.ndarray, key: bytes, nonce: int, row_salts: bytes | None,
                domain_id: bytes) -> tuple[Nodes, tuple[Nodes, ...]]:
    """The ``b || c`` strings and tree levels of ``words`` ``(n, W)`` u16 by the kernels' functions on the CPU: the device
    path's twin."""
    rows = _rows(words)
    n = rows.shape[0]
    t = table(n, rows.shape[1], key, nonce, row_salts is not None, domain_id)
    salts = None if row_salts is None else _salts(row_salts, n)
    t.rows, t.salts = rows.ctypes.data, 0 if salts is None else salts.ctypes.data
    bcs, levels = np.empty(16 * n, np.uint64), np.empty(8 * (2 * t.W - 1), np.uint64)
    rc = host_lib().rr_commit_host(ctypes.byref(t), RTD._hm(), RTD._prefixes(), bcs, levels)
    if rc:
        raise ValueError(f"rr_commit_host refused the value's layout ({rc})")
    return _out(bcs, levels, t.W)


def host_public_salts(head: bytes, first: int, n: int) -> bytes:
    """`pouw_h1_registered.public_salts` by the kernels' function on the CPU: the device path's twin."""
    out = np.empty(24 * n, np.uint64)
    if host_lib().rr_public_salts_host(ctypes.byref(pub(head, first, n)), out):
        raise ValueError("rr_public_salts_host refused the salt head")
    return out.tobytes()


def _device(torch: Any) -> RTD.Device:
    return RTD.Device.of(torch, torch.cuda.current_device(), device_lib, "rr_midstates")


@cache
def available() -> bool:
    """Whether this process can hash on the GPU: torch with CUDA, and the kernels built for its current device (a warning
    says why not when CUDA is there)."""
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        _device(torch)
    except Exception as e:  # noqa: BLE001 - nvcc missing or a failed build leaves the NumPy path, said once
        warnings.warn(f"H1's registered values hash with NumPy: the GPU path doesn't build here ({type(e).__name__}: {e})",
                      RuntimeWarning, stacklevel=2)
        return False
    return True


class Hashing:
    """A value's hash launched on the device's stream (`device_hash`): ``ready()`` without blocking, ``result()`` its
    ``b || c`` strings and tree levels, waited for. It holds the rows, salts and scratch until read, so nothing the stream
    still reads goes back to the allocator."""

    def __init__(self, done: Any, bcs: Any, levels: Any, W: int, held: tuple[Any, ...]) -> None:
        self.done, self.bcs, self.levels, self.W, self.held = done, bcs, levels, W, held

    def ready(self) -> bool:
        return bool(self.done.query())

    def result(self) -> tuple[tuple[bytes, ...], tuple[tuple[bytes, ...], ...]]:
        self.done.synchronize()
        self.held = ()
        return _out(self.bcs.numpy().view(np.uint64), self.levels.numpy().view(np.uint64), self.W)


def device_commit(words: Any, key: bytes, nonce: int, row_salts: bytes | None,
                  domain_id: bytes) -> tuple[tuple[bytes, ...], tuple[tuple[bytes, ...], ...]]:
    """`device_hash`'s strings and levels, waited for."""
    return device_hash(words, key, nonce, row_salts, domain_id).result()


def device_hash(words: Any, key: bytes, nonce: int, row_salts: bytes | None, domain_id: bytes,
                public: tuple[bytes, int] | None = None) -> Hashing:
    """The ``b || c`` strings and tree levels of ``words`` ``(n, W)`` u16 hashed on the current CUDA device, on its own
    stream, nothing waiting on the host but a host ``words``' copy. ``words`` on the host are copied there; a u16 tensor
    on that device is read in place, after everything its current stream has enqueued (the stream that wrote it), and is
    held until the hash is done. ``public`` ``(head, first)``: the rows' salts are the public salts from position
    ``first`` under ``head`` (`device_public_salts`'), derived on the stream where the hash reads them, ``row_salts``
    unread."""
    import torch

    d = _device(torch)
    dev, st = d.hm.device, d.stream
    held = isinstance(words, torch.Tensor) and words.is_cuda
    if held and (words.device != dev or words.dtype != torch.uint16 or words.dim() != 2):
        raise ValueError(f"words {tuple(words.shape)} of {words.dtype} on {words.device}: not (n, W) u16 on {dev}")
    rows = words.contiguous().view(torch.uint8) if held else _rows(words)
    n, rb = (int(v) for v in rows.shape)
    t = table(n, rb, key, nonce, row_salts is not None or public is not None, domain_id)
    per = max(1, CHUNK_BYTES // rb)
    if held:
        written = torch.cuda.Event()
        written.record(torch.cuda.current_stream(dev))
        st.wait_event(written)
    with torch.cuda.stream(st):
        if public is not None:
            salts = _public_salts(d, pub(*public, n))
        else:
            salts = None if row_salts is None else torch.from_numpy(_salts(row_salts, n)).to(dev)
        t.salts = 0 if salts is None else salts.data_ptr()
        bcs = torch.empty(16 * n, dtype=torch.int64, device=dev)
        levels = torch.empty(8 * (2 * t.W - 1), dtype=torch.int64, device=dev)
        chunks = []
        for lo in range(0, n, per):
            hi = min(n, lo + per)
            chunk = rows[lo:hi] if held else torch.from_numpy(rows[lo:hi]).to(dev)
            chunks.append(chunk)
            t.rows = chunk.data_ptr()
            rc = d.lib.rr_leaves(st.cuda_stream, ctypes.byref(t), lo, hi - lo, d.hm.data_ptr(), d.ms.data_ptr(),
                                 bcs.data_ptr(), levels.data_ptr())
            if rc:
                raise RuntimeError(f"rr_leaves: CUDA error {rc}")
        rc = d.lib.rr_tree(st.cuda_stream, ctypes.byref(t), levels.data_ptr())
        if rc:
            raise RuntimeError(f"rr_tree: CUDA error {rc}")
        got_bcs = torch.empty(16 * n, dtype=torch.int64, pin_memory=True)
        got_levels = torch.empty(8 * (2 * t.W - 1), dtype=torch.int64, pin_memory=True)
        got_bcs.copy_(bcs, non_blocking=True)
        got_levels.copy_(levels, non_blocking=True)
        done = torch.cuda.Event()
        done.record(st)
    return Hashing(done, got_bcs, got_levels, t.W, (words, rows, salts, bcs, levels, chunks))


def _public_salts(d: Any, p: Pub) -> Any:
    import torch

    out = torch.empty(24 * p.n, dtype=torch.int64, device=d.hm.device)
    rc = d.lib.rr_public_salts(d.stream.cuda_stream, ctypes.byref(p), out.data_ptr())
    if rc:
        raise RuntimeError(f"rr_public_salts: CUDA error {rc}")
    return out


def device_public_salts(head: bytes, first: int, n: int) -> bytes:
    """`pouw_h1_registered.public_salts` on the current CUDA device, on its stream, returned once done."""
    import torch

    d = _device(torch)
    with torch.cuda.stream(d.stream):
        return _public_salts(d, pub(head, first, n)).cpu().numpy().tobytes()


@contextmanager
def hashing() -> Iterator[Any]:
    """The current CUDA device, the stream `device_hash` hashes on made current meanwhile: a tensor written there is read
    in place by a hash launched there after it, which waits on nothing else."""
    import torch

    d = _device(torch)
    with torch.cuda.stream(d.stream):
        yield d.hm.device
