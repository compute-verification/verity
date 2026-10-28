"""`pouw_h1_registered.commit`'s hashing on the GPU: a registered value's ``b || c`` strings and every level of its
frame-v3-sha512 tree, computed by `protocol_options/pouw_cuda/registered_rows.cu` from the rows' words, the salts drawn on
the device (ChaCha20 under the run's key at the value's nonce, `audit.hm96_salt`) or given, under the domain id the host
computes (`registered.domain`).

A call's z/y records are two tables' rows: its registered ``zy@<c>`` and its call z table (`pouw_h1_tables_device`, under
``hm96-sha512/row-seg/v1``). With ``also``, `device_hash` commits both from one read of each record
(`pouw_cuda/zy_commit.cu`, on `verity.ml.kernels`' ``row_commit.cuh``): each record's two row digests from the same loads,
each table's leaf, the value's tree as before and the call table's a level at a time, the call table's root `Hashing.also`.
`host_zy` is its twin on the CPU (`zy_commit_host.cpp`).

`device_commit` copies the rows to the device in chunks of at most `CHUNK_BYTES` (rows already there, as a call's z/y
records are, it reads in place), hashes each row (its ``sha512/row/v2`` digest, salt, hm96-sha512 string, tree leaf and
frame leaf) and then the pads and levels, on a stream of its own, and returns the strings and levels as `Registered`
holds them (64-byte nodes, 128-byte strings, as bytes). `device_hash` launches the same and returns at once, its
`Hashing` reading them out once the stream is done; a public table's salts it derives on that stream, in the buffer it
hashes from, and rows written there (under `hashing`) it reads with no wait. `host_commit` runs
the same functions on the CPU (`registered_rows_host.cpp`, built with the system C++ compiler): the twin the CPU tests
hold to `pouw_h1_registered.commit`'s NumPy path on machines without a GPU. Builds share the Commit's row-table builder
(`row_tables_device._build`, keyed by sources, flags and compiler); nvcc is `pouw_device`'s (its `TOOLCHAIN`'s,
or the one on the path).
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any

import numpy as np
from verity.ml.commitments import hm96
from verity.ml.commitments.rowleaf import (ROW_SEG_BYTES, sha512_row_seg_prefix, sha512_row_seg_segment_prefix,
                                           sha512_row_seg_segments, sha512_row_v2_prefix)
from verity.ml.kernels import ROW_COMMIT, SHA512

from verity_vllm.commit import row_tables_device as RTD

__all__ = ["CHUNK_BYTES", "Hashing", "Pub", "Table", "ZyArgs", "available", "device_commit", "device_hash", "device_lib",
           "device_public_salts", "hashing", "host_commit", "host_lib", "host_public_salts", "host_zy", "nvcc", "pub",
           "table", "zy_args", "zy_available", "zy_device_lib", "zy_fits", "zy_host_lib"]

DEVICE_SOURCES = ("ncp2.cuh", "row_table.cuh", "registered_rows.cuh", "registered_rows.cu")
HOST_SOURCES = ("ncp2.cuh", "row_table.cuh", "registered_rows.cuh", "registered_rows_host.cpp")
_ZY = ("ncp2.cuh", "row_table.cuh", "registered_rows.cuh", str(SHA512.resolve()), str(ROW_COMMIT.resolve()), "zy_commit.cuh")
ZY_DEVICE_SOURCES, ZY_HOST_SOURCES = (*_ZY, "zy_commit.cu"), (*_ZY, "zy_commit_host.cpp")
ZY_FLAGS = (f"-I{ROW_COMMIT.resolve().parent}",)
#: a record's segments `zy_commit.cuh` takes: rows of 1 to `ZY_SEGS` whole 1024-byte segments
ZY_SEGS = 8
#: the hm96 key both tables' leaves are under (`row_tables_device._hm`'s)
HM_KEY = hm96.SHA512.default_key
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


class ZyArgs(ctypes.Structure):
    """`zy_commit.cuh`'s ``Args``: the call table's then the value's domain id, the hm96 key, the prefix blocks, each
    table's salt key words and nonce, and the records' count and bytes."""
    _fields_ = [("did", (ctypes.c_uint8 * 64) * 2), ("hm_key", ctypes.c_uint8 * 256), ("pre_v2", ctypes.c_uint8 * 128),
                ("pre_seg_outer", ctypes.c_uint8 * 128), ("pre_seg", (ctypes.c_uint8 * 128) * ZY_SEGS),
                ("key", (ctypes.c_uint32 * 8) * 2), ("nonce", ctypes.c_uint64 * 2), ("n", ctypes.c_uint32),
                ("R", ctypes.c_uint32)]


def zy_fits(n: int, row_bytes: int) -> bool:
    """Whether `zy_commit.cuh` commits ``n`` rows of ``row_bytes``: 1 to `MAX_ROWS` rows of 1 to `ZY_SEGS` whole
    1024-byte segments."""
    return 0 < n <= MAX_ROWS and row_bytes % ROW_SEG_BYTES == 0 and 0 < row_bytes // ROW_SEG_BYTES <= ZY_SEGS


def zy_args(n: int, row_bytes: int, key: bytes, call: tuple[bytes, int], value: tuple[bytes, int]) -> ZyArgs:
    """The layout of ``n`` records of ``row_bytes`` committed as the call table ``call`` and the value ``value``, each
    ``(domain id, nonce)``, both salted by ChaCha20 under the 32-byte ``key``."""
    if not zy_fits(n, row_bytes) or len(key) != 32 or any(len(d) != 64 or not 0 <= s < 1 << 64 for d, s in (call, value)):
        raise ValueError(f"{n} records of {row_bytes} bytes under a {len(key)}-byte key: not 1 to {MAX_ROWS} rows of 1 to "
                         f"{ZY_SEGS} whole {ROW_SEG_BYTES}-byte segments under a 32-byte key, 64-byte ids, u64 nonces")
    bits, segs = 8 * row_bytes, sha512_row_seg_segments(8 * row_bytes)
    blocks = [sha512_row_v2_prefix(bits), sha512_row_seg_prefix(bits)]
    blocks += [sha512_row_seg_segment_prefix(j, len(segs), m) for j, m in enumerate(segs)]
    if len(HM_KEY) != 256 or any(len(b) != 128 for b in blocks):
        raise AssertionError("zy_commit.cuh takes a 256-byte hm96 key and one-block prefixes")
    a = ZyArgs()
    ctypes.memmove(a.did, call[0] + value[0], 128)
    ctypes.memmove(a.hm_key, HM_KEY, 256)
    ctypes.memmove(a.pre_v2, blocks[0], 128)
    ctypes.memmove(a.pre_seg_outer, blocks[1], 128)
    ctypes.memmove(a.pre_seg, b"".join(blocks[2:]), 128 * len(segs))
    words = np.frombuffer(key, "<u4").tolist()
    a.key[0][:], a.key[1][:] = words, words
    a.nonce[:] = [call[1], value[1]]
    a.n, a.R = n, row_bytes
    return a


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


def _zy_sizes(dll: ctypes.CDLL) -> ctypes.CDLL:
    dll.zy_args_bytes.restype = ctypes.c_uint64
    if dll.zy_args_bytes() != ctypes.sizeof(ZyArgs):
        raise RuntimeError(f"zy_commit.cuh's Args is {dll.zy_args_bytes()} bytes, ZyArgs {ctypes.sizeof(ZyArgs)}")
    return dll


@cache
def zy_host_lib() -> ctypes.CDLL:
    """`zy_commit_host.cpp` built with the system C++ compiler."""
    cxx = next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)
    if cxx is None:
        raise RuntimeError("no C++ compiler on the path")
    dll = _zy_sizes(RTD._build(cxx, ZY_HOST_SOURCES, (*RTD.CFLAGS, *ZY_FLAGS), "libzy_commit_host.so"))
    dll.zy_commit_host.argtypes = [ctypes.POINTER(ZyArgs), ctypes.POINTER(Table), _vp, _u64, _u64, _u64]
    dll.zy_commit_host.restype = ctypes.c_int
    return dll


@cache
def zy_device_lib(arch: str) -> ctypes.CDLL:
    """`zy_commit.cu` built with nvcc for ``sm_<arch>``."""
    tool = nvcc()
    if tool is None:
        raise RuntimeError("no nvcc (pouw_device.TOOLCHAIN, or on the path)")
    dll = _zy_sizes(RTD._build(tool, ZY_DEVICE_SOURCES,
                               (*RTD.NVCC_FLAGS, *ZY_FLAGS, f"-gencode=arch=compute_{arch},code=sm_{arch}"),
                               "libzy_commit_cuda.so"))
    dll.zy_plan_bytes.restype = ctypes.c_uint64
    dll.zy_plan.argtypes = [ctypes.POINTER(ZyArgs), _vp]
    dll.zy_nibbles.argtypes = [_vp, _vp]
    dll.zy_records.argtypes = [_vp, _vp, ctypes.c_uint32, _vp, _vp, _vp, _vp, _vp]
    dll.zy_call_tree.argtypes = [_vp, _vp, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, _vp]
    dll.zy_plan.restype = dll.zy_records.restype = dll.zy_call_tree.restype = ctypes.c_int
    dll.zy_nibbles.restype = None
    return dll


def _root(words: np.ndarray) -> bytes:
    return np.asarray(words).view(np.uint64).astype(">u8").tobytes()


def host_zy(words: np.ndarray, key: bytes, nonce: int, domain_id: bytes, also: tuple[bytes, int]
            ) -> tuple[bytes, tuple[bytes, ...], tuple[tuple[bytes, ...], ...]]:
    """`device_hash`'s ``also`` on the CPU, the device path's twin: the call table ``also`` ``(domain id, nonce)``'s root,
    then the value's ``b || c`` strings and tree levels, of ``words`` ``(n, W)`` u16."""
    rows = _rows(words)
    n, rb = rows.shape
    t = table(n, rb, key, nonce, False, domain_id)
    a = zy_args(n, rb, key, also, (domain_id, nonce))
    call, bcs, levels = (np.empty(k, np.uint64) for k in (8 * (2 * t.W - 1), 16 * n, 8 * (2 * t.W - 1)))
    rc = zy_host_lib().zy_commit_host(ctypes.byref(a), ctypes.byref(t), rows.ctypes.data, call, bcs, levels)
    if rc:
        raise ValueError(f"zy_commit_host refused the records' layout ({rc})")
    return (_root(call[-8:]), *_out(bcs, levels, t.W))


def _rows(words: np.ndarray) -> np.ndarray:
    words = np.require(words, "<u2", ["C", "W"])
    return words.view(np.uint8).reshape(words.shape[0], 2 * words.shape[1])


def _salts(row_salts: bytes, n: int) -> np.ndarray:
    if len(row_salts) != 192 * n:
        raise ValueError(f"one 192-byte salt per row: {len(row_salts)} bytes for {n} rows")
    return np.frombuffer(row_salts, np.uint8).copy()


def _out(bcs: np.ndarray, levels: np.ndarray, W: int) -> tuple[tuple[bytes, ...], tuple[tuple[bytes, ...], ...]]:
    """The device's big-endian words as `Registered`'s ``bcs`` and ``levels``."""
    nodes = levels.byteswap().view("V64").tolist()
    out, off, w = [], 0, W
    while True:
        out.append(tuple(nodes[off:off + w]))
        off += w
        if w == 1:
            break
        w //= 2
    return tuple(bcs.byteswap().view("V128").tolist()), tuple(out)


def host_commit(words: np.ndarray, key: bytes, nonce: int, row_salts: bytes | None,
                domain_id: bytes) -> tuple[tuple[bytes, ...], tuple[tuple[bytes, ...], ...]]:
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


class _Device:
    """One CUDA device's build, its copies of the byte tables and midstates, and the stream values are hashed on."""

    def __init__(self, torch: Any, device: Any) -> None:
        major, minor = torch.cuda.get_device_capability(device)
        self.lib = device_lib(f"{major}{minor}")
        self.stream = torch.cuda.Stream(device)
        with torch.cuda.stream(self.stream):
            self.hm = torch.from_numpy(RTD._hm().view(np.int64)).to(device)
            pre = torch.from_numpy(RTD._prefixes().view(np.int64)).to(device)
            self.ms = torch.empty(16, dtype=torch.int64, device=device)
            rc = self.lib.rr_midstates(self.stream.cuda_stream, pre.data_ptr(), self.ms.data_ptr())
        if rc:
            raise RuntimeError(f"rr_midstates: CUDA error {rc}")
        self.stream.synchronize()


_DEVICES: dict[int, _Device] = {}


def _device(torch: Any) -> _Device:
    index = torch.cuda.current_device()
    if index not in _DEVICES:
        _DEVICES[index] = _Device(torch, torch.device("cuda", index))
    return _DEVICES[index]


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


_ZY_LIBS: dict[int, tuple[ctypes.CDLL, Any, int]] = {}


def _zy(torch: Any) -> tuple[ctypes.CDLL, Any, int]:
    """`zy_commit.cu`'s build for the current device, the hm96 key's nibble table there and the plan's bytes: built once."""
    index = torch.cuda.current_device()
    if index not in _ZY_LIBS:
        d = _device(torch)
        major, minor = torch.cuda.get_device_capability(index)
        lib = zy_device_lib(f"{major}{minor}")
        key, tab = np.frombuffer(HM_KEY, np.uint8).copy(), np.empty(1 << 13, np.uint8)
        lib.zy_nibbles(key.ctypes.data, tab.ctypes.data)
        with torch.cuda.stream(d.stream):
            nib = torch.from_numpy(tab).to(d.hm.device)
        d.stream.synchronize()
        _ZY_LIBS[index] = (lib, nib, int(lib.zy_plan_bytes()))
    return _ZY_LIBS[index]


@cache
def zy_available() -> bool:
    """Whether `device_hash` takes ``also`` in this process: `available`, and `zy_commit.cu` built for the current device
    (a warning says why not)."""
    if not available():
        return False
    import torch

    try:
        _zy(torch)
    except (OSError, RuntimeError, AttributeError, subprocess.SubprocessError) as e:
        warnings.warn(f"H1's call z tables hash apart from their zy@<c> values: zy_commit.cu doesn't build here "
                      f"({type(e).__name__}: {e})", RuntimeWarning, stacklevel=2)
        return False
    return True


class Hashing:
    """A value's hash launched on the device's stream (`device_hash`): ``ready()`` without blocking, ``result()`` its
    ``b || c`` strings and tree levels, waited for, and with ``also`` `also` the other table's root. It holds the rows,
    salts and scratch until read, so nothing the stream still reads goes back to the allocator."""

    def __init__(self, done: Any, bcs: Any, levels: Any, W: int, held: tuple[Any, ...], also: Any = None) -> None:
        self.done, self.bcs, self.levels, self.W, self.held, self._also = done, bcs, levels, W, held, also

    def ready(self) -> bool:
        return bool(self.done.query())

    def result(self) -> tuple[tuple[bytes, ...], tuple[tuple[bytes, ...], ...]]:
        self.done.synchronize()
        self.held = ()
        return _out(self.bcs.numpy().view(np.uint64), self.levels.numpy().view(np.uint64), self.W)

    def also(self) -> bytes:
        """The root of the table `device_hash`'s ``also`` named, waited for."""
        if self._also is None:
            raise LookupError("this hash committed one table")
        self.done.synchronize()
        return _root(self._also.numpy())


def device_commit(words: Any, key: bytes, nonce: int, row_salts: bytes | None,
                  domain_id: bytes) -> tuple[tuple[bytes, ...], tuple[tuple[bytes, ...], ...]]:
    """`device_hash`'s strings and levels, waited for."""
    return device_hash(words, key, nonce, row_salts, domain_id).result()


def device_hash(words: Any, key: bytes, nonce: int, row_salts: bytes | None, domain_id: bytes,
                public: tuple[bytes, int] | None = None, also: tuple[bytes, int] | None = None) -> Hashing:
    """The ``b || c`` strings and tree levels of ``words`` ``(n, W)`` u16 hashed on the current CUDA device, on its own
    stream, nothing waiting on the host but a host ``words``' copy. ``words`` on the host are copied there; a u16 tensor
    on that device is read in place, after everything its current stream has enqueued (the stream that wrote it), and is
    held until the hash is done. ``public`` ``(head, first)``: the rows' salts are the public salts from position
    ``first`` under ``head`` (`device_public_salts`'), derived on the stream where the hash reads them, ``row_salts``
    unread. ``also`` ``(domain id, nonce)``: the same rows are a call's z table too (`pouw_h1_tables_device`'s
    ``hm96-sha512/row-seg/v1`` rows, salted under ``key`` at that nonce), committed from the same read of each row
    (`zy_commit.cu`), its root `Hashing.also`: rows on the device, of whole segments (`zy_fits`), 16-byte aligned, salted
    by ChaCha20 (no ``row_salts`` or ``public``)."""
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
    if also is not None and not (held and row_salts is None and public is None and zy_fits(n, rb)
                                 and rows.data_ptr() % 16 == 0):
        raise ValueError(f"also: {n} rows of {rb} bytes ({'on the device' if held else 'on the host'}, "
                         f"{'salted by ChaCha20' if row_salts is None and public is None else 'given salts'}): a call's "
                         "z table is committed beside its value only from 16-byte aligned device rows of whole "
                         "segments under ChaCha20's salts")
    if held:
        written = torch.cuda.Event()
        written.record(torch.cuda.current_stream(dev))
        st.wait_event(written)
    if also is not None:
        return _hash_both(torch, d, words, rows, t, key, nonce, domain_id, also)
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


def _hash_both(torch: Any, d: Any, words: Any, rows: Any, t: Table, key: bytes, nonce: int, domain_id: bytes,
               also: tuple[bytes, int]) -> Hashing:
    """`device_hash` with ``also``, its stream already behind the rows' writer."""
    lib, nib, plan_bytes = _zy(torch)
    dev, st, n = d.hm.device, d.stream, t.n
    a = zy_args(n, t.row_bytes, key, also, (domain_id, nonce))
    staged = torch.empty(plan_bytes, dtype=torch.uint8, pin_memory=True)
    if lib.zy_plan(ctypes.byref(a), staged.data_ptr()):
        raise ValueError("zy_plan refused the records' layout")
    nodes = 8 * (2 * t.W - 1)
    with torch.cuda.stream(st):
        plan = staged.to(dev, non_blocking=True)
        call, bcs, levels = (torch.empty(k, dtype=torch.int64, device=dev) for k in (nodes, 16 * n, nodes))
        rc = lib.zy_records(st.cuda_stream, plan.data_ptr(), n, nib.data_ptr(), rows.data_ptr(), call.data_ptr(),
                            bcs.data_ptr(), levels.data_ptr())
        if rc:
            raise RuntimeError(f"zy_records: CUDA error {rc}")
        rc = d.lib.rr_tree(st.cuda_stream, ctypes.byref(t), levels.data_ptr())
        if rc:
            raise RuntimeError(f"rr_tree: CUDA error {rc}")
        rc = lib.zy_call_tree(st.cuda_stream, plan.data_ptr(), n, t.W, t.depth, call.data_ptr())
        if rc:
            raise RuntimeError(f"zy_call_tree: CUDA error {rc}")
        got_bcs = torch.empty(16 * n, dtype=torch.int64, pin_memory=True)
        got_levels = torch.empty(nodes, dtype=torch.int64, pin_memory=True)
        got_root = torch.empty(8, dtype=torch.int64, pin_memory=True)
        got_bcs.copy_(bcs, non_blocking=True)
        got_levels.copy_(levels, non_blocking=True)
        got_root.copy_(call[-8:], non_blocking=True)
        done = torch.cuda.Event()
        done.record(st)
    return Hashing(done, got_bcs, got_levels, t.W, (words, rows, staged, plan, call, bcs, levels), got_root)


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
