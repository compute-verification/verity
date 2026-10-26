"""The serve-time commit call on the GPU (README.md is its specification, ``slots.py`` its CPU reference).

``Plan(context, layout)`` holds what a run fixes (the unit and slot domains, their pad tails, the rows' first-block midstates and
hm96's table) on a device. A step is committed as its groups' outputs arrive::

    sc = StepCommitter(plan, step, slots, salt_key)     # salt_key: slots.fresh_salt_key(), the prover's
    for tensors in each group's outputs:                 # e.g. a layer's ops, (slots, ...) CUDA tensors
        done = sc.add(tensors)                           # on the plan's side stream behind the current stream; reuse the
                                                         # tensors' memory only after the event `done`
    pending = sc.finish()                                # no host sync
    pending.result()                                     # slots.StepCommit(root, slot_roots), the reference's

or at once, ``commit(plan, step, salt_key, outputs, slots)``, the reference's ``slots.commit`` signature with the plan in place
of the context and layout. ``Plan(..., device="host")`` runs the same calls on the host twin (``commit_slots_host.cpp``, g++),
with numpy arrays for tensors: what the CPU tests hold to the reference.

Builds are cached by a hash of the sources, flags and compiler version under ``~/.cache/verity/commit_slots``; nvcc and g++ are
the ones on the path.
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

import numpy as np

import slots as S
from verity.ml.commitments.merkle import merkle_depth
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix

__all__ = ["Pending", "Plan", "StepCommitter", "commit", "device_lib", "host_lib", "t_table"]

HERE = Path(__file__).resolve().parent
SHA512 = HERE.parents[1] / "pouw" / "commit_gpu" / "sha512.cuh"
DEVICE_SOURCES = (SHA512, HERE / "commit_slots.cuh", HERE / "commit_slots.cu")
HOST_SOURCES = (SHA512, HERE / "commit_slots.cuh", HERE / "commit_slots_host.cpp")
NVCC_FLAGS = ("-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared", "-lineinfo")
CFLAGS = ("-O2", "-std=c++17", "-fPIC", "-shared")
BUILD_DIR = Path.home() / ".cache" / "verity" / "commit_slots"
MAX_SRC, SLOT_LV, UNIT_LV = 24, 17, 7

u32, u64, vp = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_void_p


class Ctx(ctypes.Structure):
    _fields_ = [("T", vp), ("unit_did", vp), ("unit_tail", vp), ("tail_lv", vp), ("tail_node", vp), ("slot_did", u64 * 8),
                ("ms_salt", u64 * 8), ("ms_leaf", u64 * 8), ("U", u32), ("slot_depth", u32), ("slot_tail_off", u32),
                ("slot_tail_n", u32)]


class Step(ctypes.Structure):
    _fields_ = [("key", u32 * 8), ("step_did", u64 * 8), ("front", vp), ("leaves", vp), ("rl", vp), ("slot_root", vp),
                ("lv_a", vp), ("lv_b", vp), ("root", vp), ("S", u32), ("W", u32), ("depth", u32), ("pad_", u32)]


class Grp(ctypes.Structure):
    _fields_ = [("rows", vp), ("units", vp), ("src", vp * MAX_SRC), ("stride", u64 * MAX_SRC), ("nrows", u32), ("nunits", u32),
                ("unit_base", u32), ("row_base", u32)]


def _build(tool: str, sources: Sequence[Path], flags: Sequence[str], name: str) -> ctypes.CDLL:
    h = hashlib.sha256()
    for s in sources:
        h.update(s.name.encode() + b"\0" + s.read_bytes() + b"\0")
    version = subprocess.run([tool, "--version"], capture_output=True, text=True, check=True).stdout
    h.update(" ".join(flags).encode() + b"\0" + version.encode())
    lib = BUILD_DIR / h.hexdigest()[:12] / name
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run([tool, *flags, "-o", str(tmp), str(sources[-1])], check=True)
        tmp.replace(lib)
    return _abi(ctypes.CDLL(str(lib)))


def _abi(dll: ctypes.CDLL) -> ctypes.CDLL:
    dll.cs_midstate.argtypes, dll.cs_midstate.restype = [vp, vp], None
    dll.cs_tail.argtypes, dll.cs_tail.restype = [vp, u64, ctypes.c_int, vp, vp], ctypes.c_int
    for f in (dll.cs_rows, dll.cs_units, dll.cs_fold):
        f.argtypes, f.restype = [vp, ctypes.POINTER(Ctx), ctypes.POINTER(Step), ctypes.POINTER(Grp)], ctypes.c_int
    dll.cs_finish.argtypes, dll.cs_finish.restype = [vp, ctypes.POINTER(Ctx), ctypes.POINTER(Step)], ctypes.c_int
    return dll


@cache
def host_lib() -> ctypes.CDLL:
    """``commit_slots_host.cpp`` built with the system C++ compiler."""
    cxx = next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)
    if cxx is None:
        raise RuntimeError("no C++ compiler on the path")
    return _build(cxx, HOST_SOURCES, CFLAGS, "libcommit_slots_host.so")


@cache
def device_lib(arch: str) -> ctypes.CDLL:
    """``commit_slots.cu`` built with nvcc for ``sm_<arch>`` (``"90"``: the H200)."""
    tool = shutil.which("nvcc")
    if tool is None:
        raise RuntimeError("no nvcc on the path")
    return _build(tool, DEVICE_SOURCES, (*NVCC_FLAGS, f"-gencode=arch=compute_{arch},code=sm_{arch}"), "libcommit_slots_cuda.so")


@cache
def t_table() -> bytes:
    """hm96-sha512's Hankel table under the pinned key: T_v = xor of (k >> s) over the set bits s of v (k the key, a little-endian
    integer), 256 bytes little-endian each, so bytes [p, p + 64) of T_v are ``byte_tables(key)[p][v]``."""
    k = int.from_bytes(S.KEY, "little")
    rows = [0] * 256
    for v in range(1, 256):
        rows[v] = rows[v & (v - 1)] ^ (k >> ((v & -v).bit_length() - 1))
    return b"".join(r.to_bytes(256, "little") for r in rows)


def _words(digest: bytes) -> np.ndarray:
    return np.frombuffer(digest, dtype=">u8").astype(np.uint64)


def _digests(words: np.ndarray) -> list[bytes]:
    raw = np.ascontiguousarray(words, dtype=np.uint64).reshape(-1, 8).astype(">u8").tobytes()
    return [raw[64 * i:64 * (i + 1)] for i in range(len(raw) // 64)]


class Plan:
    """What a run's ``context`` and ``layout`` fix, on ``device`` (a CUDA device, or ``"host"`` for the host twin)."""

    def __init__(self, context: bytes, layout: S.Layout, device: Any = "cuda", arch: str | None = None) -> None:
        self.context, self.layout, self.host = context, layout, device == "host"
        if self.host:
            self.torch, self.device, self.lib = None, "host", host_lib()
        else:
            import torch
            self.torch, self.device = torch, torch.device(device)
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            if arch is None:
                major, minor = torch.cuda.get_device_capability(self.device)
                arch = f"{major}{minor}"
            self.lib = device_lib(arch)
            self.stream = torch.cuda.Stream(self.device)
        U = layout.unit_count
        if U >= 1 << (SLOT_LV - 1):
            raise ValueError(f"a slot has fewer than {1 << (SLOT_LV - 1)} units")
        unit_did = np.stack([_words(S.unit_domain(context, layout, u).domain_id) for u in range(U)])
        slot_dom = S.slot_domain(context, layout)
        lv: list[np.ndarray] = []
        nodes: list[np.ndarray] = []
        unit_tail = np.zeros((U, 2), np.uint32)
        at = 0
        for u, rows in enumerate(layout.units):
            if len(rows) > 1 << UNIT_LV:
                raise ValueError(f"a unit has at most {1 << UNIT_LV} rows")
            a, b = self._tail(unit_did[u], len(rows))
            unit_tail[u] = (at, len(a))
            lv.append(a), nodes.append(b)
            at += len(a)
        self.slot_depth = merkle_depth(U)
        a, b = self._tail(_words(slot_dom.domain_id), U)
        slot_tail = (at, len(a))
        lv.append(a), nodes.append(b)
        tail_lv = np.concatenate(lv + [np.zeros(1, np.uint32)])
        tail_node = np.concatenate(nodes + [np.zeros((1, 8), np.uint64)])
        self.groups: dict[S.Group, tuple[Any, Any, int, int]] = {}
        for g in layout.groups:
            if g not in self.groups:
                self.groups[g] = self._group(g)
        self.held = [self._put(np.frombuffer(t_table(), np.uint8)), self._put(unit_did), self._put(unit_tail),
                     self._put(tail_lv), self._put(tail_node)]
        c = Ctx()
        c.T, c.unit_did, c.unit_tail, c.tail_lv, c.tail_node = (self._ptr(t) for t in self.held)
        c.slot_did[:] = [int(w) for w in _words(slot_dom.domain_id)]
        c.ms_salt[:] = self._midstate(S.INST.salt_prefix)
        c.ms_leaf[:] = self._midstate(S.INST.leaf_prefix(S.KEY))
        c.U, c.slot_depth, c.slot_tail_off, c.slot_tail_n = U, self.slot_depth, *slot_tail
        self.ctx = c
        self._work: dict[int, dict[str, Any]] = {}

    def _midstate(self, block: bytes) -> list[int]:
        out = np.zeros(8, np.uint64)
        self.lib.cs_midstate(np.frombuffer(block, np.uint8).ctypes.data, out.ctypes.data)
        return [int(w) for w in out]

    def _tail(self, did: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        depth = merkle_depth(n)
        cap = max(1, depth)
        lv, nodes = np.zeros(cap, np.uint32), np.zeros((cap, 8), np.uint64)
        did = np.ascontiguousarray(did, np.uint64)
        k = self.lib.cs_tail(did.ctypes.data, n, depth, lv.ctypes.data, nodes.ctypes.data)
        return lv[:k], nodes[:k]

    def _group(self, g: S.Group) -> tuple[Any, Any, int, int]:
        if len(g.sources) > MAX_SRC:
            raise ValueError(f"a group takes at most {MAX_SRC} tensors")
        rows = np.zeros((sum(len(u) for u in g.units), 32), np.uint32)
        units = np.zeros((len(g.units), 4), np.uint32)
        i = 0
        for k, unit in enumerate(g.units):
            units[k] = (i, len(unit), merkle_depth(len(unit)), 0)
            for r, row in enumerate(unit):
                if len(row.parts) > 2:
                    raise ValueError("a row on the device has at most two parts")
                a, b = row.parts[0], (row.parts[1] if len(row.parts) > 1 else S.Part(0, 0, 0))
                rows[i, :10] = (k, r, i, row.nbytes, a.source, a.offset, a.nbytes, b.source, b.offset, b.nbytes)
                rows[i].view(np.uint64)[8:] = self._midstate(sha512_row_v2_prefix(row.bits))
                i += 1
        order = np.argsort(-rows[:, 3].astype(np.int64), kind="stable")
        return self._put(np.ascontiguousarray(rows[order])), self._put(units), len(rows), len(units)

    def _put(self, a: np.ndarray) -> Any:
        a = np.ascontiguousarray(a)
        if self.host:
            return a
        return self.torch.from_numpy(a.view(np.uint8).copy()).to(self.device)

    def _ptr(self, t: Any) -> int:
        return t.ctypes.data if self.host else t.data_ptr()

    def _zeros(self, n_words: int) -> Any:
        if self.host:
            return np.zeros(n_words, np.uint64)
        return self.torch.zeros(n_words, dtype=self.torch.int64, device=self.device)

    def work(self, slots: int) -> dict[str, Any]:
        """The scratch a step of ``slots`` slots uses (kept per slot count: one step at a time per plan)."""
        if slots not in self._work:
            W = 1 << merkle_depth(slots)
            rows = max(v[2] for v in self.groups.values())
            units = max(v[3] for v in self.groups.values())
            self._work[slots] = {"front": self._zeros(slots * SLOT_LV * 8), "leaves": self._zeros(rows * slots * 8),
                                 "rl": self._zeros(units * slots * 8), "slot_root": self._zeros(slots * 8),
                                 "lv_a": self._zeros(W * 8), "lv_b": self._zeros(W * 8), "root": self._zeros(8)}
        return self._work[slots]


class Pending:
    """A finished step's commitment in flight: ``result()`` waits for it (``ready()`` doesn't)."""

    def __init__(self, event: Any, root: Any, slot_roots: Any) -> None:
        self.event, self._root, self._slot_roots = event, root, slot_roots

    def ready(self) -> bool:
        return self.event is None or self.event.query()

    def result(self) -> S.StepCommit:
        if self.event is not None:
            self.event.synchronize()
        root = self._root if isinstance(self._root, np.ndarray) else self._root.numpy()
        sr = self._slot_roots if isinstance(self._slot_roots, np.ndarray) else self._slot_roots.numpy()
        return S.StepCommit(_digests(root.view(np.uint64))[0], tuple(_digests(sr.view(np.uint64))))


class StepCommitter:
    """Step ``step`` of ``slots`` slots under ``plan``, its salts under ``salt_key`` (32 bytes, the prover's): ``add`` each
    group's outputs in the layout's order, then ``finish``. On a CUDA plan every launch goes on the plan's side stream after
    what the current stream has enqueued, and each added tensor is held for that stream (``record_stream``). With ``marks``
    (a list), each launch appends ``(name, event)``, its end on the side stream (timing only)."""

    def __init__(self, plan: Plan, step: int, slots: int, salt_key: bytes, marks: list[tuple[str, Any]] | None = None) -> None:
        if len(salt_key) != 32:
            raise ValueError("a salt key is 32 bytes")
        self.plan, self.slots, self.k, self.marks = plan, slots, 0, marks
        w = plan.work(slots)
        st = Step()
        st.key[:] = [int(x) for x in np.frombuffer(salt_key, "<u4")]
        st.step_did[:] = [int(x) for x in _words(S.step_domain(plan.context, plan.layout, step, slots).domain_id)]
        st.front, st.leaves, st.rl, st.slot_root, st.lv_a, st.lv_b, st.root = (plan._ptr(w[k]) for k in (
            "front", "leaves", "rl", "slot_root", "lv_a", "lv_b", "root"))
        st.S, st.depth = slots, merkle_depth(slots)
        st.W = 1 << st.depth
        self.st, self.w = st, w

    def _call(self, fn: Any, *args: Any) -> None:
        if self.plan.host:
            rc = fn(None, *args)
        else:
            torch = self.plan.torch
            with torch.cuda.device(self.plan.device):
                rc = fn(self.plan.stream.cuda_stream, *args)
                if self.marks is not None:
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record(self.plan.stream)
                    self.marks.append((fn.__name__, ev))
        if rc:
            raise RuntimeError(f"{fn.__name__} failed: CUDA error {rc}")

    def add(self, tensors: Sequence[Any]) -> Any:
        """The next group's outputs: tensor i of shape (slots, ...), each slot's bytes contiguous, ``group.sources[i]`` of them.
        Returns an event on the side stream after which the tensors may be overwritten (a producer that reuses its buffers
        waits on it), or None on the host."""
        layout = self.plan.layout
        if self.k >= len(layout.groups):
            raise ValueError("every group is in")
        g = layout.groups[self.k]
        if len(tensors) != len(g.sources):
            raise ValueError(f"group {self.k} takes {len(g.sources)} tensors, not {len(tensors)}")
        rows, units, gr_rows, gr_units = self.plan.groups[g]
        gr = Grp()
        gr.rows, gr.units = self.plan._ptr(rows), self.plan._ptr(units)
        gr.nrows, gr.nunits = gr_rows, gr_units
        gr.unit_base, gr.row_base = layout.unit_base[self.k], layout.row_base[layout.unit_base[self.k]]
        if not self.plan.host:
            torch = self.plan.torch
            self.plan.stream.wait_stream(torch.cuda.current_stream(self.plan.device))
        for i, (t, n) in enumerate(zip(tensors, g.sources, strict=True)):
            gr.src[i], gr.stride[i] = self._source(t, n)
        self._call(self.plan.lib.cs_rows, ctypes.byref(self.plan.ctx), ctypes.byref(self.st), ctypes.byref(gr))
        self._call(self.plan.lib.cs_units, ctypes.byref(self.plan.ctx), ctypes.byref(self.st), ctypes.byref(gr))
        self._call(self.plan.lib.cs_fold, ctypes.byref(self.plan.ctx), ctypes.byref(self.st), ctypes.byref(gr))
        self.k += 1
        if self.plan.host:
            return None
        done = self.plan.torch.cuda.Event()
        done.record(self.plan.stream)
        return done

    def _source(self, t: Any, n: int) -> tuple[int, int]:
        if self.plan.host:
            a = np.asarray(t)
            if a.ndim < 1 or a.shape[0] != self.slots or not a[0].flags.c_contiguous or a[0].nbytes != n:
                raise ValueError(f"a tensor is (slots={self.slots}, ...) with {n} contiguous bytes a slot")
            return a.ctypes.data, a.strides[0]
        if t.device != self.plan.device or t.shape[0] != self.slots or not t[0].is_contiguous() \
                or t[0].numel() * t.element_size() != n:
            raise ValueError(f"a tensor is (slots={self.slots}, ...) on {self.plan.device} with {n} contiguous bytes a slot")
        t.record_stream(self.plan.stream)
        return t.data_ptr(), t.stride(0) * t.element_size()

    def finish(self) -> Pending:
        if self.k != len(self.plan.layout.groups):
            raise ValueError(f"{self.k} of {len(self.plan.layout.groups)} groups are in")
        self._call(self.plan.lib.cs_finish, ctypes.byref(self.plan.ctx), ctypes.byref(self.st))
        if self.plan.host:
            return Pending(None, self.w["root"].copy(), self.w["slot_root"].copy())
        torch = self.plan.torch
        with torch.cuda.stream(self.plan.stream):
            root = torch.empty(8, dtype=torch.int64, pin_memory=True)
            sr = torch.empty(self.slots * 8, dtype=torch.int64, pin_memory=True)
            root.copy_(self.w["root"], non_blocking=True)
            sr.copy_(self.w["slot_root"], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.plan.stream)
        return Pending(ev, root, sr)


def commit(plan: Plan, step: int, salt_key: bytes, outputs: Sequence[Sequence[Any]], slots: int) -> S.StepCommit:
    """``slots.commit`` on ``plan``'s device: every group's outputs at once, waiting for the result."""
    sc = StepCommitter(plan, step, slots, salt_key)
    for tensors in outputs:
        sc.add(tensors)
    return sc.finish().result()
