"""`history.HistoryTable` on the GPU (`Table` with a ``device``) and its CPU twin (`Table` without one): a history table's tree
built, appended to and opened by `protocols/pouw/pouw_cuda/row_table.cuh`'s history functions, the device's through
`row_table.cu`'s ``rt_hist_*`` kernels and the twin's through `row_table_host.cpp`, the same build `row_tables_device` loads.

A `Table` reads as `history.HistoryTable` does where `commit.history_tables.Histories` reads one (its ``make``): ``rows``,
``salts``, ``bcs``, ``root``, ``path``, ``entry`` and ``append``, and holds the same root over the same rows under the same
salts. A row's salt is `history.HistoryTable`'s: 192 bytes fresh from the operating system at its append, or the one given,
and never one the table has used. No key derives a salt, so two builds of one table share none.
The tree is kept whole, (2 capacity - 1) nodes of 64 bytes; an append hashes its rows' leaves and the nodes above them.
"""
from __future__ import annotations

import ctypes
import os
from collections.abc import Sequence
from functools import cache
from typing import Any

import numpy as np

from verity.commitments import history as HT
from verity.commitments import hm96
from verity_vllm.commit import row_tables_device as D

__all__ = ["Hist", "Table"]

SALT_BYTES = hm96.SHA512.salt_bytes
#: a row's bytes at most (`row_table.cuh`'s ``HIST_MAX_SEGS`` segments)
MAX_ROW_BYTES = 8 * D.SEG_BYTES


class Hist(ctypes.Structure):
    """`row_table.cuh`'s ``Hist``: the domain id, the placeholder's tree leaf, the capacity, the row's bytes and the tree's
    depth."""
    _fields_ = [("did", ctypes.c_uint64 * 8), ("pad", ctypes.c_uint64 * 8), ("cap", ctypes.c_uint64), ("L", ctypes.c_uint32),
                ("depth", ctypes.c_uint32)]


_hp = ctypes.POINTER(Hist)


@cache
def _host() -> ctypes.CDLL:
    lib = D.host_lib()
    lib.rt_hist_init_host.argtypes = [_hp, D._u64]
    lib.rt_hist_append_host.argtypes = [_hp, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p, D._u64, D._u64, D._u64,
                                        D._u64, D._u32]
    lib.rt_hist_paths_host.argtypes = [_hp, D._u64, D._u32, ctypes.c_uint32, D._u64]
    for f in (lib.rt_hist_init_host, lib.rt_hist_append_host, lib.rt_hist_paths_host):
        f.restype = ctypes.c_int
    return lib


def _device(lib: ctypes.CDLL) -> ctypes.CDLL:
    vp = ctypes.c_void_p
    lib.rt_hist_init.argtypes = [vp, _hp, vp]
    lib.rt_hist_append.argtypes = [vp, _hp, ctypes.c_uint64, ctypes.c_uint32] + [vp] * 6
    lib.rt_hist_paths.argtypes = [vp, _hp, vp, vp, ctypes.c_uint32, vp]
    for f in (lib.rt_hist_init, lib.rt_hist_append, lib.rt_hist_paths):
        f.restype = ctypes.c_int
    return lib


def _bytes(words: np.ndarray, dtype: str) -> bytes:
    return np.asarray(words).astype(dtype).tobytes()


class Table:
    """Table ``value`` of ``program`` (`history.domain`) of ``words``-word rows at ``capacity`` leaves: its tree on the CUDA
    ``device`` (a torch device or index), or in host memory hashed by the CPU twin when ``device`` is None."""

    word_bits = 16
    schema = HT.SCHEMA

    def __init__(self, program: str, value: str, words: int, capacity: int = HT.CAPACITY, *, device: Any = None) -> None:
        if not HT.cap_ok(capacity) or capacity > 1 << 30:
            raise ValueError(f"a table's capacity is a power of two from 2 to 2^30, not {capacity!r}")
        if not HT.words_ok(words) or 2 * words > MAX_ROW_BYTES:
            raise ValueError(f"a device table's rows are 1 to {MAX_ROW_BYTES // 2} words, not {words!r}")
        self.program, self.value, self.words, self.leaves = program, value, words, capacity
        self.domain = HT.domain(program, value, words, capacity)
        self.rows: list[np.ndarray] = []
        self.salts: list[bytes] = []
        self.bcs: list[bytes] = []
        pad = hm96.SHA512.tree_leaf(hm96.SHA512.default_key, HT.placeholder(words))
        h = self._h = Hist()
        h.did[:] = D._words(self.domain.domain_id).tolist()
        h.pad[:] = D._words(pad).tolist()
        h.cap, h.L, h.depth = capacity, 2 * words, capacity.bit_length() - 1
        nodes = 8 * (2 * capacity - 1)
        if device is None:
            self._dev = None
            self._tree = np.zeros(nodes, dtype=np.uint64)
            self._check("rt_hist_init_host", _host().rt_hist_init_host(ctypes.byref(h), self._tree))
        else:
            import torch

            self._torch = torch
            dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
            self._dev = D.DeviceLibrary.of(torch, dev.index if dev.index is not None else torch.cuda.current_device())
            _device(self._dev.lib)
            with torch.cuda.stream(self._dev.stream):
                self._tree = torch.empty(nodes, dtype=torch.int64, device=self._dev.hm.device)
                self._check("rt_hist_init", self._dev.lib.rt_hist_init(self._dev.stream.cuda_stream, ctypes.byref(h),
                                                                       self._tree.data_ptr()))
            self._dev.stream.synchronize()

    def _check(self, fn: str, rc: int) -> None:
        if rc:
            raise RuntimeError(f"{fn} refused table {self.value} ({rc})")

    @property
    def root(self) -> bytes:
        top = self._tree[-8:]
        return _bytes(top.cpu().numpy().view(np.uint64) if self._dev is not None else top, ">u8")

    def append(self, rows: Any, salts: Sequence[bytes] | None = None) -> range:
        """Rows ``rows`` (``(k, words)`` 16-bit words: a numpy array, or a contiguous 16-bit tensor on the table's device) at
        the table's next ``k`` positions, each under its salt (default: fresh from the operating system), refused past the
        capacity or for a salt used before, as `history.HistoryTable.append` refuses them. -> their positions."""
        on_device = self._dev is not None and isinstance(rows, self._torch.Tensor) and rows.is_cuda
        shape = tuple(rows.shape)
        if len(shape) != 2 or shape[1] != self.words or (on_device and (rows.element_size() != 2 or not rows.is_contiguous())):
            raise ValueError(f"{self.value}: rows of {self.words} contiguous 16-bit words, not shape {shape}")
        at, k = len(self.rows), shape[0]
        if at + k > self.leaves:
            raise ValueError(f"{self.value}: {at + k} rows past the table's capacity of {self.leaves}")
        given = None if salts is None else [bytes(s) for s in salts]
        if given is not None and (len(given) != k or any(len(s) != SALT_BYTES for s in given)):
            raise ValueError(f"{self.value}: one {SALT_BYTES}-byte salt per row")
        if given is not None and (len(set(given)) != k or not set(given).isdisjoint(self.salts)):
            raise ValueError(f"{self.value}: a salt is never reused: equal salts on two rows reveal the XOR of their digests")
        if not k:
            return range(at, at)
        salt = os.urandom(SALT_BYTES * k) if given is None else b"".join(given)
        host = (rows.view(self._torch.int16).cpu().numpy().view(np.uint16) if on_device
                else np.ascontiguousarray(np.asarray(rows, dtype=np.uint16)))
        bcs, ys = np.zeros(16 * k, dtype=np.uint64), np.frombuffer(salt, dtype="<u4").astype(np.uint32)
        if self._dev is None:
            self._check("rt_hist_append_host", _host().rt_hist_append_host(
                ctypes.byref(self._h), at, k, host.ctypes.data, D._hm(), D._prefixes(), self._tree, bcs, ys))
        else:
            self._device_append(rows if on_device else None, host, at, k, bcs, ys)
        self.rows += [r.copy() for r in host]
        self.salts += [salt[SALT_BYTES * i:SALT_BYTES * (i + 1)] for i in range(k)]
        self.bcs += [_bytes(bcs[16 * i:16 * i + 16], ">u8") for i in range(k)]
        return range(at, at + k)

    def _device_append(self, rows: Any, host: np.ndarray, at: int, k: int, bcs: np.ndarray, ys: np.ndarray) -> None:
        torch, d = self._torch, self._dev
        written = torch.cuda.Event()
        written.record(torch.cuda.current_stream(d.hm.device))
        d.stream.wait_event(written)
        with torch.cuda.stream(d.stream):
            src = rows if rows is not None else torch.from_numpy(host.view(np.int16)).to(d.hm.device)
            out_bc = torch.empty(16 * k, dtype=torch.int64, device=d.hm.device)
            y = torch.from_numpy(ys.view(np.int32)).to(d.hm.device)
            self._check("rt_hist_append", d.lib.rt_hist_append(
                d.stream.cuda_stream, ctypes.byref(self._h), at, k, src.data_ptr(), d.hm.data_ptr(), d.ms.data_ptr(),
                self._tree.data_ptr(), out_bc.data_ptr(), y.data_ptr()))
            bcs[:] = out_bc.cpu().numpy().view(np.uint64)
        d.stream.synchronize()

    def paths(self, positions: Sequence[int]) -> bytes:
        """The paths of ``positions`` to the root, concatenated: each its siblings, leaf level first, 64 bytes each."""
        pos = np.asarray(positions, dtype=np.int64)
        if pos.ndim != 1 or not len(pos) or pos.min() < 0 or pos.max() >= self.leaves:
            raise ValueError(f"{self.value}: positions within the table's {self.leaves}")
        pos, n = pos.astype(np.uint32), len(pos)
        if self._dev is None:
            out = np.zeros(8 * n * self._h.depth, dtype=np.uint64)
            self._check("rt_hist_paths_host", _host().rt_hist_paths_host(ctypes.byref(self._h), self._tree, pos, n, out))
            return _bytes(out, ">u8")
        torch, d = self._torch, self._dev
        with torch.cuda.stream(d.stream):
            p = torch.from_numpy(pos.view(np.int32)).to(d.hm.device)
            out_d = torch.empty(8 * n * self._h.depth, dtype=torch.int64, device=d.hm.device)
            self._check("rt_hist_paths", d.lib.rt_hist_paths(d.stream.cuda_stream, ctypes.byref(self._h), self._tree.data_ptr(),
                                                             p.data_ptr(), n, out_d.data_ptr()))
            out = out_d.cpu().numpy().view(np.uint64)
        d.stream.synchronize()
        return _bytes(out, ">u8")

    def path(self, position: int) -> bytes:
        """Position ``position``'s path to the root (`history.HistoryTable.path`)."""
        return self.paths([position])

    def entry(self) -> dict[str, Any]:
        """What a pin names (`history.HistoryTable.entry`)."""
        return {"value": self.value, "root": self.root.hex(), "leaves": self.leaves, "words": self.words,
                "domain": self.domain.domain_id.hex()}
