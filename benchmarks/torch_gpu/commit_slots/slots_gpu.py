"""The serve-time commit call on the GPU (README.md is its specification, ``slots.py`` its CPU reference), on the
SHA-512 engine (``../sha512_rows/engine.py``): the engine hashes each group's rows into hm96 leaves and builds frame-v3
levels; this module lays them out as one subtree per slot.

``Plan(context, layout)`` holds what a run fixes (the engine, the unit and slot domains, each group's leaf table) on a device.
A step is committed as its groups' outputs arrive::

    sc = StepCommitter(plan, step, slots, salt_key)     # salt_key: slots.fresh_salt_key(), the prover's
    for tensors in each group's outputs:                 # e.g. a layer's ops, (slots, ...) CUDA tensors
        done = sc.add(tensors)                           # on the plan's side stream behind the current stream; overwrite
                                                         # the tensors in place only after the event `done`
    pending = sc.finish()                                # no host sync
    pending.result()                                     # slots.StepCommit(root, slot_roots), the reference's

or at once, ``commit(plan, step, salt_key, outputs, slots)``, the reference's ``slots.commit`` signature with the plan in place
of the context and layout. ``Plan(..., device="host")`` runs the same calls on the engine's host twin (g++), with numpy arrays
for tensors: what the CPU tests hold to the reference.

The plan holds each add's tensors until the side stream has read them (a thread drops them then) or, at the next add,
until the producer's stream waits for that read (``Plan.release``, ``LAG``), so a producer may free its tables once it
has added them: their memory goes back to its stream at the read, not at the side stream's tail, and at the HBM cap its
next allocations find it there without the allocator's retry (which frees the cache and waits on every stream).

With ``graphs=True`` (``StepCommitter``, ``commit``), each add is one CUDA graph launch: the group's passes, captured on its
first add and replayed with the tensors' addresses and the salt key as data, so the same kernels give the same roots while
the host enqueues a handful of calls an add, not the thousands a step that otherwise wait on a full launch queue.

A group is one launch of the engine's row kernel a pass, so groups of several layers (``regroup``) fill the GPU better at
small slot counts. Grouping is the prover's plumbing: the layout's digest, the salts and every root stay the same.

A step holds no slot's leaves: each slot's tree is its frontier, the left sibling a level still waiting for its right one
((slots, depth + 1, 8) words, 1.5 KB a slot at six million units). ``add`` folds a group's unit leaves into the frontiers
a level at a time, and ``finish`` closes them with the subtrees of pads alone, which the Plan computes once. A group goes
through in passes of as many slots as keep its temporaries (row digests, the run of slot-tree leaves, the leaves of the
units of several rows) within the Plan's ``temp_bytes``.

A row-slot of a one-row proof block costs the engine nine compressions (one of them over a constant schedule), three
ChaCha20 chunks and the hm96 mask: its row digest (one), hm96's c (two) and h (two), its frame leaf in the proof block's domain (one), the
unit's slot-tree leaf (one, from the unit's midstate, ``frame_table``, set up once an add), all in the leaf kernel's thread,
and about one slot-tree node (two).
"""
from __future__ import annotations

import collections
import contextlib
import importlib.util
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import slots as S

from verity.ml.commitments.merkle import merkle_depth

__all__ = ["LAG", "TEMP_BYTES", "E", "Pending", "Plan", "SmPartition", "StepCommitter", "build", "commit", "regroup"]

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "sha512_rows" / "engine.py"
LAG = 1               # adds a plan holds unread tensors for before their producer waits for the read: 1 keeps no more
                      # tables alive at the producer's next allocation than it holds itself (at the HBM cap, 2 ran out)
TEMP_BYTES = 1 << 30
TEMP_BUFFERS = 4      # a pass's live (rows, slots) digest buffers at most: row digests, the run, the rest's leaves, a level
PAD_CHUNK = 16        # pads alone are hashed 2^16 at a time


def _load_engine() -> Any:
    name = "sha512_rows_engine"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ENGINE)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


E = _load_engine()


def _arch(major: int, minor: int) -> str:
    return f"{major}{minor}" + ("a" if major in (9, 10, 12) else "")


def build(capability: str) -> list[str]:
    """The engine's device and host builds for compute capability ``capability`` (e.g. ``"9.0"``), with no GPU needed: what
    a job builds before it takes a GPU lease."""
    major, minor = (int(x) for x in capability.split("."))
    return [str(E.device_lib(_arch(major, minor)).path), str(E.host_lib().path)]


class _Engine(E.Engine):
    """The engine with its uploads staged through pinned memory and copied asynchronously, so a call on the side stream never
    waits for the work queued ahead of it there."""

    def _up(self, a: np.ndarray) -> Any:
        if self.host:
            return super()._up(a)
        staged = self.torch.from_numpy(np.ascontiguousarray(a).view(np.uint8).reshape(-1).copy()).pin_memory()
        return staged.to(self.device, non_blocking=True)


@dataclass(frozen=True)
class _Shape:
    """A group's rows and units as arrays, the same for every group the layout repeats (its layers): each row's parts
    (``a`` then ``b``, ``b_len`` 0 for a one-part row) and its unit and rank, the engine's row spec with byte offsets in
    place of addresses (``row_setup`` done), the row kernel's order, and each row's ``dest`` (the leaf kernel's): its unit
    when the unit is that row alone, else the units plus its index among the ``rest``, the rows of the units of n > 1 rows,
    which ``multi`` gives as n: (those units, their first index in the rest, how many)."""
    rows: int
    units: int
    unit_of_row: np.ndarray
    rank: np.ndarray
    src: np.ndarray
    off: np.ndarray
    extent: np.ndarray
    spec: Any
    order: Any
    nbytes: tuple[int, ...]
    dest: Any
    rest: int
    multi: dict[int, tuple[Any, int, int]]


@dataclass(frozen=True)
class _Group:
    shape: _Shape
    leaf_table: Any
    dids: Any
    first_unit: int


@dataclass(frozen=True)
class _Graph:
    """A group's add captured as one CUDA graph: it reads its tensors' addresses and slot strides from ``addr`` ((2,
    tensors) int64 on the device, filled before each replay), and salts under the key ``rekey`` sets in its ``n`` leaf
    launches ``nodes``."""
    graph: Any
    addr: Any
    nodes: Any
    n: int


class SmPartition:
    """``sms`` of a CUDA device's SMs as a green context (the driver's ``cuGreenCtxCreate``, through the bindings torch's
    ``GreenContext`` uses), and the rest as another, made on first use: a stream of either runs its kernels on its SMs
    alone, while the primary context's streams still run on every SM. The split rounds ``sms`` up to the device's
    granularity; ``sms`` and ``rest`` are the counts it gave."""

    def __init__(self, sms: int, device: Any) -> None:
        import torch
        from torch.cuda._utils import _check_cuda_bindings, _cuda_bindings_driver

        self.torch, self.ok, self.drv = torch, _check_cuda_bindings, _cuda_bindings_driver
        self.index = torch.device(device).index
        self.index = torch.cuda.current_device() if self.index is None else self.index
        torch.cuda.init()
        drv, ok = self.drv, self.ok
        self.dev = ok(drv.cuDeviceGet(self.index))
        whole = ok(drv.cuDeviceGetDevResource(self.dev, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
        if not 0 < sms < whole.sm.smCount:
            raise ValueError(f"a partition takes 1 .. {whole.sm.smCount - 1} of the device's {whole.sm.smCount} SMs, not {sms}")
        groups, n, self._rest = ok(drv.cuDevSmResourceSplitByCount(1, whole, 0, sms))
        if n != 1:
            raise RuntimeError(f"the device's SMs split into {n} groups of {sms}, not one")
        self.sms, self.rest, self.device_sms = int(groups[0].sm.smCount), int(self._rest.sm.smCount), int(whole.sm.smCount)
        self._ctx, self._rest_ctx = self._green(groups[0]), None
        self._streams: dict[tuple[bool, int], Any] = {}

    def _green(self, resource: Any) -> Any:
        drv, ok = self.drv, self.ok
        desc = ok(drv.cuDevResourceGenerateDesc([resource], 1))
        return ok(drv.cuGreenCtxCreate(desc, self.dev, drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))

    def _stream(self, rest: bool, priority: int) -> Any:
        least, greatest = self.ok(self.drv.cuCtxGetStreamPriorityRange())
        priority = min(max(priority, greatest), least)
        if (rest, priority) not in self._streams:
            if rest and self._rest_ctx is None:
                self._rest_ctx = self._green(self._rest)
            ctx = self._rest_ctx if rest else self._ctx
            s = self.ok(self.drv.cuGreenCtxStreamCreate(ctx, self.drv.CUstream_flags.CU_STREAM_NON_BLOCKING, priority))
            self._streams[(rest, priority)] = (s, self.torch.cuda.ExternalStream(int(s), self.index))
        return self._streams[(rest, priority)][1]

    def stream(self, priority: int = 0) -> Any:
        """The stream on the partition's ``sms`` SMs at ``priority`` (torch's, lower first, clamped to the device's)."""
        return self._stream(False, priority)

    def rest_stream(self, priority: int = 0) -> Any:
        """The stream on the other ``rest`` SMs at ``priority``."""
        return self._stream(True, priority)

    def close(self) -> None:
        """Destroys the partition's streams and green contexts, each of which holds device memory, once their work is done:
        after this no tensor may still wait on one of its streams, and the caller has freed (``torch.cuda.empty_cache``)
        what was allocated on them."""
        for s, ext in self._streams.values():
            ext.synchronize()
            self.ok(self.drv.cuStreamDestroy(s))
        self._streams = {}
        for ctx in (self._ctx, self._rest_ctx):
            if ctx is not None:
                self.ok(self.drv.cuGreenCtxDestroy(ctx))
        self._ctx = self._rest_ctx = None

    @contextlib.contextmanager
    def current(self) -> Any:
        """The partition's green context as this thread's current context, the one the runtime resolves a kernel in: a
        graph captured on ``stream`` takes kernel node parameters only from it (the primary context's are refused)."""
        ctx = self.ok(self.drv.cuCtxFromGreenCtx(self._ctx))
        self.ok(self.drv.cuCtxPushCurrent(ctx))
        try:
            yield
        finally:
            self.ok(self.drv.cuCtxPopCurrent())


class _Held:
    """A plan's added tensors not yet dropped, oldest first, each with its read (an event on the side stream) and the stream
    that made them. On a CUDA plan (``torch``), a thread of its own, started at its first add and then waiting for the
    next when none are held, waits on the oldest read and drops each add's tensors as soon as its read has passed, so a
    producer that has freed them finds their memory back at its next allocation, the next add or not."""

    def __init__(self, torch: Any = None, device: Any = None) -> None:
        self.entries: collections.deque[tuple[Any, Any, tuple[Any, ...]]] = collections.deque()
        self.lock = threading.Lock()
        self.added, self.drained = threading.Condition(self.lock), threading.Condition(self.lock)
        self.torch, self.device = torch, device
        self.thread: threading.Thread | None = None

    def hold(self, read: Any, producer: Any, tensors: Sequence[Any]) -> None:
        with self.lock:
            self.entries.append((read, producer, tuple(tensors)))
            if self.torch is None:
                return
            if self.thread is None:
                self.thread = threading.Thread(target=self._reap, name="slots_gpu-release", daemon=True)
                self.thread.start()
            self.added.notify()

    def release(self, keep: int | None = None) -> None:
        with self.lock:
            while self.entries:
                read, producer = self.entries[0][:2]
                if not read.query():
                    if keep is None or len(self.entries) <= keep:
                        break
                    producer.wait_event(read)
                self.entries.popleft()
            if not self.entries:
                self.drained.notify_all()

    def _reap(self) -> None:
        # Waiting on and querying events while another thread captures a CUDA graph is refused unless that capture is
        # thread-local, as the plan's own are (Plan.graph).
        try:
            self.torch.cuda.set_device(self.device)
            while True:
                with self.lock:
                    self.added.wait_for(lambda: self.entries)
                    read = self.entries[0][0]
                read.synchronize()
                self.release()
        except BaseException:
            with self.lock:
                self.thread = None
            raise


class Plan:
    """What a run's ``context`` and ``layout`` fix, on ``device`` (a CUDA device, or ``"host"`` for the engine's host twin).
    Its work is linear in the layout's units: each unit's domain id is hashed once (``slots.unit_domain_ids``), each group
    the layout repeats is laid out once, and the pads that close a slot's tree are hashed once. A step's passes keep their
    temporaries within ``temp_bytes``, and run the leaf kernel at ``leaf_minb`` CTAs an SM (the engine's ``minb``, 0 for the
    build's default; every setting commits the same bytes).

    With ``sms``, the side stream is on an ``SmPartition`` of that many SMs (``partition``; ``sms`` becomes the count it
    gave), and the leaf kernel's persistent grid fills those alone: the commit then shares the device with work on the
    other SMs instead of taking turns with it for all of them. The bytes are the same at every ``sms``."""

    def __init__(self, context: bytes, layout: S.CommitShape, device: Any = "cuda", arch: str | None = None,
                 temp_bytes: int = TEMP_BYTES, leaf_minb: int = 0, sms: int = 0) -> None:
        self.context, self.layout, self.temp_bytes, self.leaf_minb = context, layout, temp_bytes, leaf_minb
        self.engine = _Engine(device, arch)
        self.host, self.torch, self.device = self.engine.host, self.engine.torch, self.engine.device
        self.unit_dids = E.words(S.unit_domain_ids(context, layout))
        self.slot_did = S.slot_domain(context, layout).domain_id
        self.depth = merkle_depth(layout.unit_count)
        self.partition = SmPartition(sms, self.device) if sms and not self.host else None
        self.sms = self.partition.sms if self.partition else 0
        if self.host:
            self.stream = None
        else:
            self.stream = self.partition.stream() if self.partition else self.torch.cuda.Stream(self.device)
        self._graphs: dict[tuple[int, ...], _Graph] = {}
        self._pool: Any = None
        self._frontiers: dict[int, Any] = {}
        self._open: StepCommitter | None = None
        self._held = _Held() if self.host else _Held(self.torch, self.device)
        shapes: dict[int, _Shape] = {}
        with self.side():
            self.slot_dw = self.engine._wordsof([self.slot_did])
            self.engine._schema(E.ROOT_SCHEMA)
            self.groups = []
            u0 = 0
            for i, g in enumerate(layout.groups):
                if id(g) not in shapes:
                    shapes[id(g)] = self._shape(g)
                self.groups.append(self._group(i, u0, shapes[id(g)]))
                u0 += len(g.units)
            self.mids = [self.engine.node_mids([self.slot_did], k) for k in range(self.depth)]
            self.pad0, self.pads_right = self._closing()
        if not self.host:
            self.stream.synchronize()
            self.torch.cuda.synchronize(self.device)

    def side(self) -> Any:
        """The plan's side stream as the current stream (nothing on the host)."""
        return contextlib.nullcontext() if self.host else self.torch.cuda.stream(self.stream)

    def hold(self, read: Any, producer: Any, tensors: Sequence[Any]) -> None:
        """Holds an add's ``tensors`` until the side stream has read them (``read``, an event there); ``producer`` is the
        stream that made them. A CUDA plan drops them as soon as the read has passed."""
        self._held.hold(read, producer, tensors)

    def release(self, keep: int | None = None) -> None:
        """Drops the held tensors whose read has passed, oldest first, and with ``keep`` all but the last ``keep`` adds' too,
        each once its producer's stream waits for its read. A tensor its producer has freed then goes back to that stream's
        pool after its read, in the stream's order, with no event for the allocator to wait on (``record_stream``'s would
        be the side stream's tail when the producer frees it)."""
        self._held.release(keep)

    def put(self, a: np.ndarray) -> Any:
        """An index or word array on the plan's device (int64 there)."""
        return a if self.host else self.torch.from_numpy(np.ascontiguousarray(a).view(np.int64)).to(self.device)

    def _shape(self, g: S.Group) -> _Shape:
        n = np.fromiter((len(u) for u in g.units), dtype=np.int64, count=len(g.units))
        first = np.concatenate(([0], np.cumsum(n)[:-1])).astype(np.int64)
        rows = [r for u in g.units for r in u]
        if any(len(r.parts) > 2 for r in rows):
            raise ValueError("a row on the device has at most two parts")
        G = len(rows)
        flat = [(p.source, p.offset, p.nbytes) for r in rows for p in (*r.parts, *[S.Part(0, 0, 0)] * (2 - len(r.parts)))]
        cols = np.array(flat, dtype=np.int64).reshape(G, 2, 3) if G else np.zeros((0, 2, 3), dtype=np.int64)
        src, off, ln = cols[:, :, 0].T.copy(), cols[:, :, 1].T.copy(), cols[:, :, 2].T.copy()
        spec = np.zeros(G, dtype=E.ROW_SPEC)
        spec["a"], spec["b"], spec["a_len"], spec["b_len"] = off[0], off[1], ln[0], ln[1]
        spec["nbytes"] = ln[0] + ln[1]
        spec = self.engine._up(spec)
        self.engine._ok(self.engine.lib.sxr_row_setup(self.engine._stream(), self.engine._ptr(spec), G), "row setup")
        nbytes = ln[0] + ln[1]
        order = np.argsort(-((nbytes + 17 + 127) // 128), kind="stable").astype(np.uint32)
        extent = np.zeros(len(g.sources), dtype=np.int64)
        for k in range(2):
            has = ln[k] > 0
            np.maximum.at(extent, src[k][has], off[k][has] + ln[k][has])
        dest = np.zeros(G, dtype=np.int64)
        ones = np.flatnonzero(n == 1)
        dest[first[ones]] = ones
        multi, rest = {}, 0
        for m in sorted(set(n.tolist()) - {1}):
            units = np.flatnonzero(n == m)
            dest[(first[units][:, None] + np.arange(m)).reshape(-1)] = len(n) + rest + np.arange(len(units) * m)
            multi[m] = (self.put(units), rest, len(units))
            rest += len(units) * m
        return _Shape(G, len(n), np.repeat(np.arange(len(n)), n), np.arange(G) - np.repeat(first, n), self.put(src), off, extent,
                      spec, self.engine._up(order), tuple(nbytes.tolist()), self.engine.u32s(dest), rest, multi)

    def _group(self, i: int, u0: int, shape: _Shape) -> _Group:
        dids = self.unit_dids[u0:u0 + shape.units]
        salt_row = self.layout.group_row_base[i] + np.arange(shape.rows, dtype=np.int64)
        lt = self.engine.leaf_table(dids, shape.unit_of_row, shape.rank, None, salt_row)
        return _Group(shape, lt, self.put(dids), u0)

    def addresses(self, shape: _Shape, tensors: Sequence[Any]) -> tuple[list[int], list[int]]:
        """Each of ``tensors``' address and slot stride, each part of a group of ``shape`` checked to lie in its tensor's
        slot rows."""
        slots = int(tensors[0].shape[0])
        base, stride, width = [], [], []
        for t in tensors:
            if int(t.shape[0]) != slots:
                raise ValueError("every tensor has the same slots")
            if self.host:
                a = np.asarray(t)
                base.append(a.ctypes.data)
                stride.append(a.strides[0] if slots > 1 else a[0].nbytes)
                width.append(a[0].nbytes)
            else:
                base.append(t.data_ptr())
                stride.append(t.stride(0) * t.element_size())
                width.append(t[0].numel() * t.element_size())
        if np.any(shape.extent > np.asarray(width, dtype=np.int64)):
            raise ValueError(f"a part is outside its tensor's rows {width}")
        return base, stride

    def rows(self, shape: _Shape, tensors: Sequence[Any], addr: Any = None) -> Any:
        """The engine's row table for a group of ``shape`` over ``tensors``: its spec with each part's tensor address added
        (the engine's ``rows``, without a Python step a row). With ``addr`` ((2, tensors) int64 on the device), the spec
        takes the addresses and strides from there when it is computed, not from ``tensors``: a captured graph's."""
        slots = int(tensors[0].shape[0])
        base, stride = self.addresses(shape, tensors)
        if self.host:
            spec = shape.spec.copy()
            b, s = np.asarray(base, dtype=np.uint64), np.asarray(stride, dtype=np.uint64)
            two = shape.spec["b_len"] > 0
            spec["a"] = b[shape.src[0]] + shape.off[0].astype(np.uint64)
            spec["a_stride"] = s[shape.src[0]]
            spec["b"] = np.where(two, b[shape.src[1]] + shape.off[1].astype(np.uint64), 0)
            spec["b_stride"] = np.where(two, s[shape.src[1]], 0)
            spec["a16"] = (spec["a"] % 16 == 0) & (spec["a_stride"] % 16 == 0)
        else:
            torch = self.torch
            spec = shape.spec.clone()
            w = spec.view(torch.int64).reshape(shape.rows, -1)
            if addr is None:
                b, s = (self.engine._up(np.asarray(v, dtype=np.int64)).view(torch.int64) for v in (base, stride))
            else:
                b, s = addr[0], addr[1]
            src = shape.src
            two = (w[:, 4] >> 32) > 0
            w[:, 0] = b[src[0]] + w[:, 0]
            w[:, 1] = s[src[0]]
            w[:, 2] = torch.where(two, b[src[1]] + w[:, 2], 0)
            w[:, 3] = torch.where(two, s[src[1]], 0)
            a16 = ((w[:, 0] % 16 == 0) & (w[:, 1] % 16 == 0)).to(torch.int64)
            w[:, 5] = (w[:, 5] & 0xFFFFFFFF) | (a16 << 32)
        return E.RowTable(spec, shape.rows, slots, shape.nbytes, tuple(tensors), shape.order)

    def multi_units(self, grp: _Group, frames: Any, rest: Any, run: Any) -> None:
        """The slot-tree leaves of the units of several rows into ``run`` (S, units, 8): each unit's root over its rows'
        leaves (``rest``, (shape.rest, S, 8), as the leaf kernel left them), then its frame leaf under ``frames``; the units of
        each row count are one call each."""
        S_ = int(run.shape[0])
        for m, (units, r0, n) in grp.shape.multi.items():
            x = rest[r0:r0 + n * m].reshape(n, m, S_, 8)
            x = np.ascontiguousarray(x.transpose(0, 2, 1, 3)) if self.host else x.permute(0, 2, 1, 3).contiguous()
            dids = grp.dids[units]
            dids = np.repeat(dids, S_, axis=0) if self.host else dids.repeat_interleave(S_, dim=0)
            roots = self.engine.root(x.reshape(-1, m, 8), dids)
            self.engine.frame_units(frames, units, roots.reshape(n, S_, 8), run)

    def graph(self, sc: StepCommitter, grp: _Group, tensors: Sequence[Any]) -> _Graph:
        """Group ``sc.k``'s add at ``sc``'s slots and this plan's pass size, leaf kernel and SMs, as a CUDA graph: captured
        (not run) on its first call, on the side stream, into the one memory pool the plan's graphs share (they run one
        after another on that stream), reading its tensors' addresses from its ``addr``. A graph keeps the leaf grid it
        was captured with, so ``sms`` is in its key."""
        key = (sc.k, sc.slots, self.temp_bytes, self.leaf_minb, self.sms)
        if key not in self._graphs:
            torch = self.torch
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            addr = torch.zeros((2, len(tensors)), dtype=torch.int64, device=self.device)
            g = torch.cuda.CUDAGraph(keep_graph=True)
            g.capture_begin(pool=self._pool, capture_error_mode="thread_local")
            sc.capturing = True
            try:
                sc._passes(grp, self.rows(grp.shape, tensors, addr), False)
            finally:
                sc.capturing = False
                g.capture_end()
            g.instantiate()
            n = -(-sc.slots // self.per_pass(grp.shape, sc.slots))
            self._graphs[key] = _Graph(g, addr, self.engine.graph_leaf_nodes(g.raw_cuda_graph(), n), n)
        return self._graphs[key]

    def empty(self, *shape: int) -> Any:
        """Digest words, (*shape, 8)."""
        if self.host:
            return np.zeros((*shape, 8), dtype=np.uint64)
        return self.torch.empty((*shape, 8), dtype=self.torch.int64, device=self.device)

    def per_slot(self, rows: Any) -> Any:
        """(n, S, 8) leaves as (S, n, 8), contiguous."""
        return np.ascontiguousarray(rows.transpose(1, 0, 2)) if self.host else rows.permute(1, 0, 2).contiguous()

    def per_pass(self, shape: _Shape, slots: int) -> int:
        """How many of ``slots`` slots a pass over a group of ``shape`` takes, its temporaries within ``temp_bytes``."""
        return max(1, min(slots, self.temp_bytes // (TEMP_BUFFERS * 64 * max(1, shape.rows))))

    def slots_of(self, table: Any, s0: int, s1: int) -> Any:
        """``table`` (bound to whole (slots, ...) tensors) for slots s0 .. s1 - 1 alone: each part's address moved on s0
        slots (a one-part row's b and b_stride are 0)."""
        if s0 == 0 and s1 == table.slots:
            return table
        if self.host:
            spec = table.spec.copy()
            spec["a"] += np.uint64(s0) * spec["a_stride"]
            spec["b"] += np.uint64(s0) * spec["b_stride"]
        else:
            spec = table.spec.clone()
            w = spec.view(self.torch.int64).reshape(table.G, -1)
            w[:, 0] += s0 * w[:, 1]
            w[:, 2] += s0 * w[:, 3]
        return E.RowTable(spec, table.G, s1 - s0, table.nbytes, table.tensors, table.order)

    def nodes(self, children: Any, level: int, first: int, pre: Any = None) -> Any:
        """(T, w, 8): the slot trees' nodes (level, first + i) over (T, 2w, 8) children, or over ``pre`` (T, 8) and then
        (T, 2w - 1, 8) children; read in place."""
        return self.engine.level(children, [self.slot_did], level, first, self.mids[level], pre)

    def _cat(self, parts: Sequence[Any], axis: int) -> Any:
        return np.concatenate(parts, axis=axis) if self.host else self.torch.cat(list(parts), dim=axis)

    def fold(self, frontier: Any, first: int, run: Any) -> None:
        """Fold ``run`` (T, m, 8), leaves ``first`` .. ``first + m - 1`` of T slot trees, into their ``frontier`` (T, depth + 1,
        8), in place. A frontier over the first n leaves holds at level k, for each set bit k of n, the node (k, (n >> k) - 1):
        a left child still waiting for its right sibling. A run at level k starting at an odd index takes that node in front,
        and one ending at an odd index leaves its last node there (after the level's nodes have read the one in front).
        Nothing is copied but that last node."""
        a, b, k = first, first + int(run.shape[1]), 0
        while a < b:
            pre = keep = None
            if a & 1:
                pre = frontier[:, k]
                a -= 1
            if b & 1:
                keep = run[:, -1]
                run = run[:, :-1]
                b -= 1
            up = self.nodes(run, k, a >> 1, pre) if a < b else run
            if keep is not None:
                frontier[:, k] = keep
            run = up
            a, b, k = a >> 1, b >> 1, k + 1

    def close(self, frontier: Any) -> Any:
        """(T, 8): the roots of T slot trees whose ``frontier`` holds every unit, padded as ``merkle.tree_levels`` pads: rank n
        (n the unit count) is pad(n), and at each level the node holding it pairs with the frontier's node on its left or
        with the subtree of pads alone on its right."""
        n, T = self.layout.unit_count, int(frontier.shape[0])
        if self.pad0 is None:
            return np.ascontiguousarray(frontier[:, self.depth]) if self.host else frontier[:, self.depth].contiguous()
        cur = np.repeat(self.pad0, T, axis=0) if self.host else self.pad0.expand(T, 8).contiguous()
        for k in range(self.depth):
            i = n >> k
            if i & 1:
                pair = (frontier[:, k], cur)
            else:
                right = np.repeat(self.pads_right[k], T, axis=0) if self.host else self.pads_right[k].expand(T, 8)
                pair = (cur, right)
            pair = np.stack(pair, axis=1) if self.host else self.torch.stack(pair, dim=1)
            cur = self.nodes(pair, k, i >> 1).reshape(T, 8)
        return cur

    def _closing(self) -> tuple[Any, tuple[Any, ...]]:
        """pad(n), and for each level k below the depth where rank n's node is a left child, its right sibling's digest: the
        subtree of pads alone at (k, (n >> k) + 1). Nothing when n is a power of two (no pads)."""
        n = self.layout.unit_count
        if n == 1 << self.depth:
            return None, ()
        pad0 = self.engine.pads([self.slot_did], 1, n, 1).reshape(1, 8)
        return pad0, tuple(None if (n >> k) & 1 else self._pads_root(k, (n >> k) + 1) for k in range(self.depth))

    def _pads_root(self, k: int, j: int) -> Any:
        """(1, 8): node (k, j) of a slot tree over pads alone, ranks j 2^k .. (j + 1) 2^k - 1."""
        if k <= PAD_CHUNK:
            nodes, lo = self.engine.pads([self.slot_did], 1, j << k, 1 << k), 0
        else:
            parts = [self._pads_root(PAD_CHUNK, (j << (k - PAD_CHUNK)) + q) for q in range(1 << (k - PAD_CHUNK))]
            nodes, lo = self._cat(parts, 0).reshape(1, -1, 8), PAD_CHUNK
        for lv in range(lo, k):
            nodes = self.nodes(nodes, lv, j << (k - lv - 1))
        return nodes.reshape(1, 8)


class Pending:
    """A finished step's commitment in flight: ``result()`` waits for it (``ready()`` doesn't), and then drops the tensors
    its plan holds that are read."""

    def __init__(self, event: Any, root: Any, slot_roots: Any, plan: Plan | None = None) -> None:
        self.event, self._root, self._slot_roots, self._plan = event, root, slot_roots, plan

    def ready(self) -> bool:
        return self.event is None or self.event.query()

    def result(self) -> S.StepCommit:
        if self.event is not None:
            self.event.synchronize()
            if self._plan is not None:
                self._plan.release()
        return S.StepCommit(E.digests(self._root)[0], tuple(E.digests(self._slot_roots)))


class StepCommitter:
    """Step ``step`` of ``slots`` slots under ``plan``, its salts under ``salt_key`` (32 bytes, the prover's): ``add`` each
    group's outputs in the layout's order, then ``finish``. On a CUDA plan every call goes on the plan's side stream after
    what the current stream has enqueued, and the plan holds each add's tensors until they are read (``Plan.release``),
    those of the add ``lag`` adds back no longer: the stream current when the step was made (``producer``, the tensors'
    own) waits for their read instead, as `e2e_h200.commit`'s loop waits ``LAG`` (2) groups back. A producer that drops
    each group's tables once it has made the next group's (e2e_h200's driver) has their memory back as soon as the side
    stream has read them, and at the default lag of 1 by its next add at the latest. With ``marks`` (a
    list), each phase appends ``(name, event)``, its end on the side stream (timing only). It holds each slot's frontier
    (``frontier``, (slots, depth + 1, 8)) and a pass's temporaries, never a slot's leaves.

    With ``graphs`` (a CUDA plan), each add is one CUDA graph launch: the group's passes, captured on the first add of that
    group at these slots, pass size and leaf kernel (``Plan.graph``), and replayed with the tensors' addresses and the salt
    key as data. The kernels and their arguments are the same, so the roots are; the host enqueues a handful of calls an add
    instead of a pass's dozens, so it no longer waits on the launch queue. The frontier is then the plan's, so a plan has
    one such step open at a time, its adds and finish in order. Its ``marks`` are one ``("add", event)`` an add, and
    ``finish``'s."""

    def __init__(self, plan: Plan, step: int, slots: int, salt_key: bytes, marks: list[tuple[str, Any]] | None = None,
                 graphs: bool = False, lag: int = LAG) -> None:
        if len(salt_key) != 32:
            raise ValueError("a salt key is 32 bytes")
        if graphs and plan.host:
            raise ValueError("a step on graphs is on a CUDA plan")
        if lag < 0:
            raise ValueError("a lag is 0 or more adds")
        self.step_did = S.step_domain(plan.context, plan.layout, step, slots).domain_id
        self.plan, self.slots, self.k, self.marks, self.graphs, self.lag = plan, slots, 0, marks, graphs, lag
        self.producer = None if plan.host else plan.torch.cuda.current_stream(plan.device)
        self.capturing = False
        self.salts = E.SaltKey(salt_key)
        if graphs:
            if plan._open is not None and plan._open.k:
                raise ValueError("the plan's step on graphs is open: finish it first")
            plan._open = self
        with plan.side():
            if not graphs:
                self.frontier = plan.empty(slots, plan.depth + 1)
            else:
                if slots not in plan._frontiers:
                    plan._frontiers[slots] = plan.empty(slots, plan.depth + 1)
                self.frontier = plan._frontiers[slots]

    def _mark(self, name: str) -> None:
        if self.marks is not None and not self.plan.host and not self.capturing:
            ev = self.plan.torch.cuda.Event(enable_timing=True)
            ev.record(self.plan.stream)
            self.marks.append((name, ev))

    def add(self, tensors: Sequence[Any]) -> Any:
        """The next group's outputs: tensor i of shape (slots, ...), each slot's bytes contiguous, ``group.sources[i]`` of them.
        Returns an event on the side stream after which the tensors may be overwritten (a producer that reuses its buffers
        waits on it; one that frees them needn't), or None on the host."""
        plan, layout = self.plan, self.plan.layout
        if self.k >= len(layout.groups):
            raise ValueError("every group is in")
        g, grp = layout.groups[self.k], plan.groups[self.k]
        if len(tensors) != len(g.sources):
            raise ValueError(f"group {self.k} takes {len(g.sources)} tensors, not {len(tensors)}")
        for t, n in zip(tensors, g.sources, strict=True):
            self._check(t, n)
        if not plan.host:
            plan.stream.wait_stream(plan.torch.cuda.current_stream(plan.device))
        if self.graphs:
            done = self._replay(grp, tensors)
        else:
            with plan.side():
                done = self._passes(grp, plan.rows(grp.shape, tensors), True)
        self.k += 1
        if done is not None:
            plan.hold(done, self.producer, tensors)
            plan.release(self.lag)
        return done

    def _passes(self, grp: _Group, table: Any, signal: bool) -> Any:
        """The group's passes over ``table``, each folded into the frontier; with ``signal``, the event after which its
        tensors are read (``_leaves``)."""
        plan, done = self.plan, None
        frames = plan.engine.frame_table(plan.slot_dw, grp.first_unit, grp.shape.units)
        per = plan.per_pass(grp.shape, self.slots)
        for s0 in range(0, self.slots, per):
            s1 = min(self.slots, s0 + per)
            run, done = self._leaves(grp, frames, plan.slots_of(table, s0, s1), s0, signal and s1 == self.slots)
            plan.fold(self.frontier[s0:s1], grp.first_unit, run)
            del run
            self._mark("fold")
        return done

    def _replay(self, grp: _Group, tensors: Sequence[Any]) -> Any:
        """The group's add as its graph's launch (captured first if this is its first): the tensors' addresses into the
        graph's ``addr``, the step's salt key into its leaf launches, then the launch; the event after it."""
        plan, torch = self.plan, self.plan.torch
        base, stride = plan.addresses(grp.shape, tensors)
        with plan.side():
            gr = plan.graph(self, grp, tensors)
            gr.addr.copy_(torch.tensor([base, stride], dtype=torch.int64).pin_memory(), non_blocking=True)
            with plan.partition.current() if plan.partition else contextlib.nullcontext():
                plan.engine.rekey(gr.graph.raw_cuda_graph_exec(), gr.nodes, gr.n, self.salts.key)
            gr.graph.replay()
            done = torch.cuda.Event(blocking=True)
            done.record(plan.stream)
        self._mark("add")
        return done

    def _leaves(self, grp: _Group, frames: Any, table: Any, s0: int, last: bool) -> tuple[Any, Any]:
        """(slots in the pass, units, 8): the group's units' slot-tree leaves in slots s0 .. of ``table``, those of one-row
        units from the leaf kernel itself; and on the pass over the last slots, the event after which the group's tensors
        are read."""
        plan, eng, shape, done = self.plan, self.plan.engine, grp.shape, None
        x = eng.row_digests(table)
        self._mark("rows")
        if last and not plan.host:
            done = plan.torch.cuda.Event(blocking=True)
            done.record(plan.stream)
        S_ = table.slots
        run, rest = plan.empty(S_, shape.units), plan.empty(shape.rest, S_)
        eng.leaves(grp.leaf_table, x, E.SaltKey(self.salts.key, self.salts.nonce0 + s0), rest, plan.leaf_minb, frames=frames,
                   dest=shape.dest, run=run, sms=plan.sms)
        del x
        self._mark("leaves")
        if shape.multi:
            plan.multi_units(grp, frames, rest, run)
        del rest
        self._mark("units")
        return run, done

    def _check(self, t: Any, n: int) -> None:
        if self.plan.host:
            a = np.asarray(t)
            if a.ndim < 1 or a.shape[0] != self.slots or not a[0].flags.c_contiguous or a[0].nbytes != n:
                raise ValueError(f"a tensor is (slots={self.slots}, ...) with {n} contiguous bytes a slot")
            return
        if t.device != self.plan.device or t.shape[0] != self.slots or not t[0].is_contiguous() \
                or t[0].numel() * t.element_size() != n:
            raise ValueError(f"a tensor is (slots={self.slots}, ...) on {self.plan.device} with {n} contiguous bytes a slot")

    def finish(self) -> Pending:
        plan = self.plan
        if self.k != len(plan.layout.groups):
            raise ValueError(f"{self.k} of {len(plan.layout.groups)} groups are in")
        if plan._open is self:
            plan._open = None
        eng = plan.engine
        with plan.side():
            slot_roots = plan.close(self.frontier)
            self._mark("slot_trees")
            root = eng.root(eng.frame_leaves(slot_roots.reshape(1, self.slots, 8), [self.step_did]), [self.step_did])
            self._mark("step_tree")
            if plan.host:
                return Pending(None, root, slot_roots)
            torch = plan.torch
            r = torch.empty(root.shape, dtype=torch.int64, pin_memory=True)
            sr = torch.empty(slot_roots.shape, dtype=torch.int64, pin_memory=True)
            r.copy_(root, non_blocking=True)
            sr.copy_(slot_roots, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(plan.stream)
        return Pending(ev, r, sr, plan)


def commit(plan: Plan, step: int, salt_key: bytes, outputs: Sequence[Sequence[Any]], slots: int,
           graphs: bool = False) -> S.StepCommit:
    """``slots.commit`` on ``plan``'s device: every group's outputs at once, waiting for the result."""
    sc = StepCommitter(plan, step, slots, salt_key, graphs=graphs)
    for tensors in outputs:
        sc.add(tensors)
    return sc.finish().result()


def regroup(layout: S.CommitShape, k: int) -> S.CommitShape:
    """``layout`` with every ``k`` consecutive groups as one, its tensors theirs in order: the same units and rows in the same
    order, so the same digest and roots, in a fewer launches."""
    if k < 1:
        raise ValueError("a group holds at least one of the layout's groups")
    groups = []
    for i in range(0, len(layout.groups), k):
        sources: list[int] = []
        units: list[tuple[S.Row, ...]] = []
        for g in layout.groups[i:i + k]:
            base = len(sources)
            sources += g.sources
            units += [tuple(S.Row(tuple(S.Part(base + p.source, p.offset, p.nbytes) for p in r.parts)) for r in u)
                      for u in g.units]
        groups.append(S.Group(tuple(sources), tuple(units)))
    return S.CommitShape(layout.name, tuple(groups))
