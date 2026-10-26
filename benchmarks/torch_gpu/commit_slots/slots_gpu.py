"""The serve-time commit call on the GPU (README.md is its specification, ``slots.py`` its CPU reference), on HASH-20's
SHA-512 engine (``../sha512_rows/engine.py``): the engine hashes each group's rows into hm96 leaves and builds frame-v3
levels; this module lays them out as one subtree per slot.

``Plan(context, layout)`` holds what a run fixes (the engine, the unit and slot domains, each group's leaf table) on a device.
A step is committed as its groups' outputs arrive::

    sc = StepCommitter(plan, step, slots, salt_key)     # salt_key: slots.fresh_salt_key(), the prover's
    for tensors in each group's outputs:                 # e.g. a layer's ops, (slots, ...) CUDA tensors
        done = sc.add(tensors)                           # on the plan's side stream behind the current stream; reuse the
                                                         # tensors' memory only after the event `done`
    pending = sc.finish()                                # no host sync
    pending.result()                                     # slots.StepCommit(root, slot_roots), the reference's

or at once, ``commit(plan, step, salt_key, outputs, slots)``, the reference's ``slots.commit`` signature with the plan in place
of the context and layout. ``Plan(..., device="host")`` runs the same calls on the engine's host twin (g++), with numpy arrays
for tensors: what the CPU tests hold to the reference.

A group is one launch of the engine's row kernel, so groups of several layers (``regroup``) fill the GPU better at small slot
counts. Grouping is the prover's plumbing: the layout's digest, the salts and every root stay the same.
"""
from __future__ import annotations

import contextlib
import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import slots as S

__all__ = ["E", "Pending", "Plan", "StepCommitter", "build", "commit", "regroup"]

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "sha512_rows" / "engine.py"


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
    place of addresses (``row_setup`` done), the row kernel's order, and for each row count n > 1 its units and their
    first rows."""
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
    single: Any
    single_rows: Any
    multi: dict[int, tuple[Any, Any]]


@dataclass(frozen=True)
class _Group:
    shape: _Shape
    leaf_table: Any
    dids: Any
    first_unit: int


class Plan:
    """What a run's ``context`` and ``layout`` fix, on ``device`` (a CUDA device, or ``"host"`` for the engine's host twin).
    Its work is linear in the layout's units: each unit's domain id is hashed once (``slots.unit_domain_ids``), and each
    group the layout repeats is laid out once."""

    def __init__(self, context: bytes, layout: S.Layout, device: Any = "cuda", arch: str | None = None) -> None:
        self.context, self.layout = context, layout
        self.engine = _Engine(device, arch)
        self.host, self.torch, self.device = self.engine.host, self.engine.torch, self.engine.device
        self.unit_dids = E.words(S.unit_domain_ids(context, layout))
        self.slot_did = S.slot_domain(context, layout).domain_id
        self.stream = None if self.host else self.torch.cuda.Stream(self.device)
        shapes: dict[int, _Shape] = {}
        with self.side():
            self.groups = []
            u0 = 0
            for i, g in enumerate(layout.groups):
                if id(g) not in shapes:
                    shapes[id(g)] = self._shape(g)
                self.groups.append(self._group(i, u0, shapes[id(g)]))
                u0 += len(g.units)
        if not self.host:
            self.torch.cuda.synchronize(self.device)

    def side(self) -> Any:
        """The plan's side stream as the current stream (nothing on the host)."""
        return contextlib.nullcontext() if self.host else self.torch.cuda.stream(self.stream)

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
        multi = {}
        for m in sorted(set(n.tolist()) - {1}):
            units = np.flatnonzero(n == m)
            multi[m] = (self.put(units), self.put(first[units][:, None] + np.arange(m)))
        ones = np.flatnonzero(n == 1)
        return _Shape(G, len(n), np.repeat(np.arange(len(n)), n), np.arange(G) - np.repeat(first, n), self.put(src), off, extent,
                      spec, self.engine._up(order), tuple(nbytes.tolist()), self.put(ones), self.put(first[ones]), multi)

    def _group(self, i: int, u0: int, shape: _Shape) -> _Group:
        dids = self.unit_dids[u0:u0 + shape.units]
        salt_row = self.layout.group_row_base[i] + np.arange(shape.rows, dtype=np.int64)
        lt = self.engine.leaf_table(dids, shape.unit_of_row, shape.rank, None, salt_row)
        return _Group(shape, lt, self.put(dids), u0)

    def rows(self, shape: _Shape, tensors: Sequence[Any]) -> Any:
        """The engine's row table for a group of ``shape`` over ``tensors``: its spec with each part's tensor address added
        (the engine's ``rows``, without a Python step a row)."""
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
            b = torch.tensor(base, dtype=torch.int64, device=self.device)
            s = torch.tensor(stride, dtype=torch.int64, device=self.device)
            src = shape.src
            two = (w[:, 4] >> 32) > 0
            w[:, 0] = b[src[0]] + w[:, 0]
            w[:, 1] = s[src[0]]
            w[:, 2] = torch.where(two, b[src[1]] + w[:, 2], 0)
            w[:, 3] = torch.where(two, s[src[1]], 0)
            a16 = ((w[:, 0] % 16 == 0) & (w[:, 1] % 16 == 0)).to(torch.int64)
            w[:, 5] = (w[:, 5] & 0xFFFFFFFF) | (a16 << 32)
        return E.RowTable(spec, shape.rows, slots, shape.nbytes, tuple(tensors), shape.order)

    def unit_roots(self, grp: _Group, leaves: Any) -> Any:
        """(units, S, 8): each unit's root over its rows' (G, S, 8) leaves; a one-row unit's root is its leaf, and the units
        of each larger row count are one call."""
        shape, S_ = grp.shape, int(leaves.shape[1])
        if shape.units == shape.rows:
            return leaves
        roots = self.empty(shape.units, S_)
        roots[shape.single] = leaves[shape.single_rows]
        for m, (units, rows) in shape.multi.items():
            x = leaves[rows]
            x = x.transpose(0, 2, 1, 3) if self.host else x.permute(0, 2, 1, 3)
            dids = grp.dids[units]
            dids = np.repeat(dids, S_, axis=0) if self.host else dids.repeat_interleave(S_, dim=0)
            r = self.engine.root(x.reshape(-1, m, 8) if self.host else x.reshape(-1, m, 8).contiguous(), dids)
            roots[units] = r.reshape(len(units), S_, 8)
        return roots

    def empty(self, *shape: int) -> Any:
        """Digest words, (*shape, 8)."""
        if self.host:
            return np.zeros((*shape, 8), dtype=np.uint64)
        return self.torch.empty((*shape, 8), dtype=self.torch.int64, device=self.device)

    def per_slot(self, rows: Any) -> Any:
        """(n, S, 8) leaves as (S, n, 8), contiguous."""
        return np.ascontiguousarray(rows.transpose(1, 0, 2)) if self.host else rows.permute(1, 0, 2).contiguous()


class Pending:
    """A finished step's commitment in flight: ``result()`` waits for it (``ready()`` doesn't)."""

    def __init__(self, event: Any, root: Any, slot_roots: Any) -> None:
        self.event, self._root, self._slot_roots = event, root, slot_roots

    def ready(self) -> bool:
        return self.event is None or self.event.query()

    def result(self) -> S.StepCommit:
        if self.event is not None:
            self.event.synchronize()
        return S.StepCommit(E.digests(self._root)[0], tuple(E.digests(self._slot_roots)))


class StepCommitter:
    """Step ``step`` of ``slots`` slots under ``plan``, its salts under ``salt_key`` (32 bytes, the prover's): ``add`` each
    group's outputs in the layout's order, then ``finish``. On a CUDA plan every call goes on the plan's side stream after
    what the current stream has enqueued, and each added tensor is held for that stream (``record_stream``). With ``marks``
    (a list), each phase appends ``(name, event)``, its end on the side stream (timing only)."""

    def __init__(self, plan: Plan, step: int, slots: int, salt_key: bytes, marks: list[tuple[str, Any]] | None = None) -> None:
        if len(salt_key) != 32:
            raise ValueError("a salt key is 32 bytes")
        self.step_did = S.step_domain(plan.context, plan.layout, step, slots).domain_id
        self.plan, self.slots, self.k, self.marks = plan, slots, 0, marks
        self.salts = E.SaltKey(salt_key)
        with plan.side():
            self.slot_leaves = plan.empty(slots, plan.layout.unit_count)

    def _mark(self, name: str) -> None:
        if self.marks is not None and not self.plan.host:
            ev = self.plan.torch.cuda.Event(enable_timing=True)
            ev.record(self.plan.stream)
            self.marks.append((name, ev))

    def add(self, tensors: Sequence[Any]) -> Any:
        """The next group's outputs: tensor i of shape (slots, ...), each slot's bytes contiguous, ``group.sources[i]`` of them.
        Returns an event on the side stream after which the tensors may be overwritten (a producer that reuses its buffers
        waits on it), or None on the host."""
        plan, layout = self.plan, self.plan.layout
        if self.k >= len(layout.groups):
            raise ValueError("every group is in")
        g, grp = layout.groups[self.k], plan.groups[self.k]
        if len(tensors) != len(g.sources):
            raise ValueError(f"group {self.k} takes {len(g.sources)} tensors, not {len(tensors)}")
        for t, n in zip(tensors, g.sources, strict=True):
            self._check(t, n)
        eng, done = plan.engine, None
        if not plan.host:
            plan.stream.wait_stream(plan.torch.cuda.current_stream(plan.device))
        with plan.side():
            x = eng.row_digests(plan.rows(grp.shape, tensors))
            self._mark("rows")
            if not plan.host:
                done = plan.torch.cuda.Event()
                done.record(plan.stream)
            leaves = eng.leaves(grp.leaf_table, x, self.salts)
            self._mark("leaves")
            u0, u1 = grp.first_unit, grp.first_unit + grp.shape.units
            self.slot_leaves[:, u0:u1] = eng.frame_leaves(plan.per_slot(plan.unit_roots(grp, leaves)), [plan.slot_did], u0)
            self._mark("units")
        self.k += 1
        return done

    def _check(self, t: Any, n: int) -> None:
        if self.plan.host:
            a = np.asarray(t)
            if a.ndim < 1 or a.shape[0] != self.slots or not a[0].flags.c_contiguous or a[0].nbytes != n:
                raise ValueError(f"a tensor is (slots={self.slots}, ...) with {n} contiguous bytes a slot")
            return
        if t.device != self.plan.device or t.shape[0] != self.slots or not t[0].is_contiguous() \
                or t[0].numel() * t.element_size() != n:
            raise ValueError(f"a tensor is (slots={self.slots}, ...) on {self.plan.device} with {n} contiguous bytes a slot")
        t.record_stream(self.plan.stream)

    def finish(self) -> Pending:
        plan = self.plan
        if self.k != len(plan.layout.groups):
            raise ValueError(f"{self.k} of {len(plan.layout.groups)} groups are in")
        eng = plan.engine
        with plan.side():
            slot_roots = eng.root(self.slot_leaves, [plan.slot_did])
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
        return Pending(ev, r, sr)


def commit(plan: Plan, step: int, salt_key: bytes, outputs: Sequence[Sequence[Any]], slots: int) -> S.StepCommit:
    """``slots.commit`` on ``plan``'s device: every group's outputs at once, waiting for the result."""
    sc = StepCommitter(plan, step, slots, salt_key)
    for tensors in outputs:
        sc.add(tensors)
    return sc.finish().result()


def regroup(layout: S.Layout, k: int) -> S.Layout:
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
    return S.Layout(layout.name, tuple(groups))
