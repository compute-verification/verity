"""`row_tables.step_roots` on the GPU: a step's row tables hashed by `protocol_options/pouw_cuda/row_table.cu` on a side stream,
behind an event on the serving stream, so the serving loop records the step's root when it is ready, without a host sync.

`step_roots` takes what `row_tables.step_roots` takes, with ``codes`` and ``scales`` CUDA tensors that hold until the result is
ready (a step's quantize outputs, in buffers the next step does not write), and returns a `Pending`: ``ready()`` asks the side
stream's event, ``result()`` is the `row_tables.StepRoots` the CPU path gives.  `record_ready` hands every leading ready step of
a queue to the loop's recorder (``LiveRecord.record``), in step order.  The step's root is what the challenger's coin for that
step waits on; the next step's compute does not wait on it.

The kernels compute each row's ``sha512/row-seg/v1`` digest, its salt from the step's ChaCha20 stream (`serving_rows.segment_salts`),
its ``hm96-sha512`` string and tree leaf and its ``frame-v3-sha512`` leaf, then each table's tree and the step's tree over their
roots (`row_table.step_tree`), with `ncp2.cuh`'s SHA-512.  The host computes the domain ids, the rows' rank order and the tables'
layout (`plan`).  `host_step` runs the same functions on the CPU (`row_table_host.cpp`, built with the system C++ compiler), the
twin the CPU tests hold to `row_tables.step_roots` on machines without a GPU.  Builds are keyed by what they compile (sources,
flags and the compiler's version) under ``~/.cache/verity/row_table``; nvcc is the one on the path.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.commitments import row_table as RT
from verity.ml.commitments.identity import tagged_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import domain_message

from . import row_tables as RTB

__all__ = ["DeviceLibrary", "Pending", "Plan", "device_lib", "host_lib", "host_step", "plan", "record_ready", "side_stream", "step_roots",
           "table_domain_id"]

CUDA_DIR = Path(__file__).resolve().parents[1] / "protocols" / "pouw" / "pouw_cuda"
DEVICE_SOURCES = ("ncp2.cuh", "row_table.cuh", "row_table.cu")
HOST_SOURCES = ("ncp2.cuh", "row_table.cuh", "row_table_host.cpp")
NVCC_FLAGS = ("-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared")
CFLAGS = ("-O2", "-std=c++17", "-fPIC", "-shared")
BUILD_DIR = Path.home() / ".cache" / "verity" / "row_table"
#: words of a table's layout (`row_table.cuh`'s ``Tab``) and the bytes of a row segment
TAB_WORDS, SEG_BYTES = 14, 1024


class Args(ctypes.Structure):
    """`row_table.cuh`'s ``Args``: the row key's ChaCha20 words, the nonce (the engine step), the table count, the step's domain id."""
    _fields_ = [("key", ctypes.c_uint32 * 8), ("nonce", ctypes.c_uint32 * 3), ("T", ctypes.c_uint32), ("did", ctypes.c_uint64 * 8)]


def _build(tool: str, sources: Sequence[str], flags: Sequence[str], name: str) -> ctypes.CDLL:
    h = hashlib.sha256()
    for s in sources:
        h.update(s.encode() + b"\0" + (CUDA_DIR / s).read_bytes() + b"\0")
    version = subprocess.run([tool, "--version"], capture_output=True, text=True, check=True).stdout
    h.update(" ".join(flags).encode() + b"\0" + version.encode())
    lib = BUILD_DIR / h.hexdigest()[:12] / name
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run([tool, *flags, "-o", str(tmp), str(CUDA_DIR / sources[-1])], check=True)
        tmp.replace(lib)
    return ctypes.CDLL(str(lib))


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
    dll.rt_step.argtypes = [_vp, _vp, _vp, ctypes.c_uint64, ctypes.c_uint32, _vp, _vp, ctypes.POINTER(Args)] + [_vp] * 5
    dll.rt_midstates.restype = dll.rt_step.restype = ctypes.c_int
    return dll


# --- the host's part: domains, rank order, layout ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _domain_frame(n: int) -> tuple[bytes, bytes]:
    """``domain_message`` of an ``n``-row table on either side of its 64-byte binding."""
    positions = RangeIndexedDomain(0, n).identity_sha512()
    a, b = (domain_message(bytes([v]) * 64, -1, positions, n, "sha512") for v in (0, 255))
    i = next(k for k in range(len(a)) if a[k] != b[k])
    if a[:i] != b[:i] or a[i + 64:] != b[i + 64:]:
        raise AssertionError("domain_message's binding is not one contiguous 64-byte part")
    return a[:i], a[i + 64:]


def table_domain_id(context: bytes, step: int, index: int, leaves: int) -> bytes:
    """``row_table.table_domain(context, step, index, leaves).domain_id``: the manifest's canonical JSON written out (keys
    sorted, ASCII values) and the domain message's frame cached by row count."""
    manifest = b'{"context":"%s","leaves":%d,"schema":"%s","step":%d,"table":%d}' % (
        context.hex().encode(), leaves, RT.SCHEMA.encode(), step, index)
    head, tail = _domain_frame(leaves)
    return hashlib.sha512(head + tagged_sha512(RT.TABLE_DOMAIN, manifest) + tail).digest()


def _width(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def _words(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=">u8").astype(np.uint64)


@dataclass(frozen=True)
class Plan:
    """A step's tables as the kernels take them: ``order`` (`row_tables.table_key`), ``tabs`` (T, 14) u64 (`row_table.cuh`'s
    ``Tab``, the codes and scales addresses zero until bound), ``perm`` each rank's batch row (tables in order), ``S`` segments,
    ``N`` rows, ``tree`` the tables' padded widths summed, and ``args``."""
    engine_step: int
    order: tuple[str, ...]
    tabs: np.ndarray
    perm: np.ndarray
    S: int
    N: int
    tree: int
    args: Args

    @property
    def step_scratch(self) -> int:
        """Nodes of the step tree's scratch: its leaves, then its levels."""
        return len(self.order) + _width(len(self.order))


def plan(context: bytes, engine_step: int, shapes: Mapping[str, tuple[Sequence[tuple[str, int]], int, int]], key: bytes) -> Plan:
    """The layout of a step whose tables are ``shapes[op_path] = (rows, B, K)`` (``rows[b]`` batch row ``b``'s (request, token
    row)), under ``context`` and the row key ``key``, as `row_tables.step_roots` orders, ranks, salts and domains them."""
    if len(context) != 64 or len(key) != 32 or not shapes:
        raise ValueError("a 64-byte context, a 32-byte row key and at least one table")
    order = tuple(sorted(shapes, key=RTB.table_key))
    layout: list[tuple[int, int, int, int, int, int]] = []
    dids: list[bytes] = []
    perms: list[list[int]] = []
    ranked: dict[int, list[int]] = {}  # a step's tables mostly share one rows list: rank it once
    row0 = seg0 = tree0 = 0
    for i, op in enumerate(order):
        ids, b, k = shapes[op]
        if b < 1 or k < 2 or k % 2 or len(ids) != b:
            raise ValueError(f"{op}: {b} rows of K = {k} with {len(ids)} row ids are not B >= 1 rows of an even K")
        if id(ids) not in ranked:
            keys = [(str(r), int(t)) for r, t in ids]
            if len(set(keys)) != len(keys):
                raise ValueError(f"{op}: a (request, token row) twice in one step")
            ranked[id(ids)] = sorted(range(b), key=keys.__getitem__)
        perms.append(ranked[id(ids)])
        s, w = -(-(k + 4) // SEG_BYTES), _width(b)
        layout.append((0, 0, k | b << 32, row0 | s << 32, seg0, w | tree0 << 32))
        dids.append(table_domain_id(context, engine_step, i, b))
        row0, seg0, tree0 = row0 + b, seg0 + b * s, tree0 + w
    tabs = np.concatenate([np.asarray(layout, dtype=np.uint64), _words(b"".join(dids)).reshape(-1, 8)], axis=1)
    args = Args()
    args.key[:] = np.frombuffer(key, dtype="<u4").tolist()
    args.nonce[:] = np.frombuffer(engine_step.to_bytes(12, "little"), dtype="<u4").tolist()
    args.T = len(order)
    args.did[:] = _words(RT.step_domain(context, engine_step, len(order)).domain_id).tolist()
    return Plan(engine_step, order, tabs, np.asarray([p for q in perms for p in q], dtype=np.uint32), seg0, row0, tree0, args)


@cache
def _hm() -> np.ndarray:
    """hm96-sha512's byte tables under the pinned key (`HidingLeaves`' tables), 192 x 256 entries of 8 words."""
    return _words(b"".join(t.to_bytes(RT.DIGEST_BYTES, "little") for table in RT.INST.byte_tables(RT.KEY) for t in table))


@cache
def _prefixes() -> np.ndarray:
    return _words(RT.INST.salt_prefix + RT.INST.leaf_prefix(RT.KEY))


def _roots(p: Plan, out: np.ndarray) -> RTB.StepRoots:
    raw = np.asarray(out).view(np.uint64).astype(">u8").tobytes()
    return RTB.StepRoots(p.engine_step, raw[:64], tuple((op, raw[64 * (i + 1):64 * (i + 2)]) for i, op in enumerate(p.order)))


def host_step(context: bytes, engine_step: int, outputs: Mapping[str, tuple[Sequence[tuple[str, int]], Any, Any]],
              key: bytes) -> RTB.StepRoots:
    """`row_tables.step_roots` by the kernels' functions on the CPU (`row_table_host.cpp`): the device path's twin."""
    held = {op: (np.ascontiguousarray(np.asarray(c, dtype=np.int8)), np.ascontiguousarray(np.asarray(s, dtype="<f4").reshape(-1)))
            for op, (_ids, c, s) in outputs.items()}
    for op, (c, s) in held.items():
        if c.ndim != 2 or s.shape != (c.shape[0],):
            raise ValueError(f"{op}: codes {c.shape} and scales {s.shape} are not (B, K) and (B,)")
    p = plan(context, engine_step, {op: (outputs[op][0], *held[op][0].shape) for op in outputs}, key)
    for i, op in enumerate(p.order):
        p.tabs[i, 0], p.tabs[i, 1] = held[op][0].ctypes.data, held[op][1].ctypes.data
    out = np.zeros(8 * (len(p.order) + 1), dtype=np.uint64)
    rc = host_lib().rt_step_host(p.tabs, p.perm, p.S, p.N, _hm(), _prefixes(), ctypes.byref(p.args), out)
    if rc:
        raise ValueError(f"rt_step_host refused the step's layout ({rc})")
    return _roots(p, out)


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


@dataclass
class Pending:
    """A step launched on the side stream: ``ready()`` without blocking, ``result()`` the step's `row_tables.StepRoots`.  It
    holds the step's scratch and its input tensors, so drop it only once ready (`record_ready` does): the allocator then has
    nothing back that the side stream still reads.  ``marks``: timing events around the kernels, when launched with
    ``timing``."""
    plan: Plan
    done: Any
    host: Any
    held: tuple[Any, ...]
    marks: tuple[Any, Any] | None = None

    @property
    def engine_step(self) -> int:
        return self.plan.engine_step

    def ready(self) -> bool:
        return bool(self.done.query())

    def result(self) -> RTB.StepRoots:
        self.done.synchronize()
        return _roots(self.plan, self.host.numpy())

    def kernel_ms(self) -> float:
        """The kernels' time on the device, once ready (``timing`` launches only)."""
        if self.marks is None:
            raise ValueError("launched without timing")
        self.done.synchronize()
        return float(self.marks[0].elapsed_time(self.marks[1]))


def step_roots(context: bytes, engine_step: int, outputs: Mapping[str, tuple[Sequence[tuple[str, int]], Any, Any]], key: bytes, *,
           stream: Any = None, timing: bool = False) -> Pending:
    """`row_tables.step_roots` of engine step ``engine_step``, hashed on ``stream`` (default: the device's side stream) after
    everything the current stream has enqueued so far, nothing waiting on the host: ``outputs[op_path] = (rows, codes, scales)``
    with ``codes`` (B, K) int8 and ``scales`` (B,) float32 CUDA tensors on one device."""
    import torch

    first = next(iter(outputs.values()))[1]
    dev = first.device
    for op, (_ids, c, s) in outputs.items():
        if c.device != dev or s.device != dev or c.dtype != torch.int8 or s.dtype != torch.float32 or c.dim() != 2 \
                or s.shape != (c.shape[0],) or not c.is_contiguous() or not s.is_contiguous():
            raise ValueError(f"{op}: codes and scales are contiguous (B, K) int8 and (B,) float32 tensors on {dev}")
    p = plan(context, engine_step, {op: (ids, int(c.shape[0]), int(c.shape[1])) for op, (ids, c, _s) in outputs.items()}, key)
    d = DeviceLibrary.of(torch, dev.index)
    side = stream or d.stream
    for i, op in enumerate(p.order):
        p.tabs[i, 0], p.tabs[i, 1] = outputs[op][1].data_ptr(), outputs[op][2].data_ptr()
    staged = torch.from_numpy(np.concatenate([p.tabs.reshape(-1).view(np.uint8), p.perm.view(np.uint8)])).pin_memory()
    written = torch.cuda.Event()
    written.record(torch.cuda.current_stream(dev))
    side.wait_event(written)
    t = len(p.order)
    with torch.cuda.stream(side):
        meta = staged.to(dev, non_blocking=True)
        dig, leaves, tree, step, out = (torch.empty(8 * n, dtype=torch.int64, device=dev)
                                        for n in (p.S, p.N, p.tree, p.step_scratch, t + 1))
        marks = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if timing else None
        if marks:
            marks[0].record(side)
        rc = d.lib.rt_step(side.cuda_stream, meta.data_ptr(), meta.data_ptr() + 8 * TAB_WORDS * t, p.S, p.N, d.hm.data_ptr(),
                           d.ms.data_ptr(), ctypes.byref(p.args), dig.data_ptr(), leaves.data_ptr(), tree.data_ptr(),
                           step.data_ptr(), out.data_ptr())
        if rc:
            raise RuntimeError(f"rt_step: CUDA error {rc}")
        if marks:
            marks[1].record(side)
        host = torch.empty(8 * (t + 1), dtype=torch.int64, pin_memory=True)
        host.copy_(out, non_blocking=True)
        done = torch.cuda.Event()
        done.record(side)
    inputs = tuple(x for _ids, c, s in outputs.values() for x in (c, s))
    return Pending(p, done, host, (staged, meta, dig, leaves, tree, step, out, *inputs), marks)


def record_ready(pending: deque[Pending], record: Callable[[RTB.StepRoots], Any]) -> int:
    """Hand each leading ready step of ``pending`` (in launch order) to ``record`` and drop it; how many.  Never blocks."""
    n = 0
    while pending and pending[0].ready():
        record(pending.popleft().result())
        n += 1
    return n
