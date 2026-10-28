"""The remote bandwidth session over the node's `Store`, inside the prover's process (the Commit's, or the serve
driver's): `StoreBackend` is the backend `benchmarks/remote_session`'s `ProverSide` drives, the role `device.Device`
and `CpuBackend` play there, and `prove` connects it to a verifier at ``HOST:PORT`` (the row's ``REMOTE_VERIFIER``,
`config.CommitConfig.remote_verifier`).

~~~text
hello         the node, N = the store's blocks (`Store.layout`) and the payload root of every block's payload as the
              store holds it now (`Store.payload`: weights, the KV pool's pages as filler, host filler)
registration  the verifier's; the backend encodes the payloads under its key (`audit.Store`, the driver's reference
              encoding, or kv.cu on the GPU into the same host mirror: `DeviceEncoded`); the hello's layout is each
              region's kind, tier and blocks (`StoreBackend.layout`)
session       the store records its memory map against `Registration.digest` under the session's lifetime cap L
              (`Store.record_memory_map`), one session a store: the map's indices are one session's
pass          before each pass `Serving.advance` runs the engine's next step s against the store (its frees, allocs
              and `kv_append`, which closes the step before's memory-map line with its re-keys for the pass's window),
              then appends the blocks step s completed and still holds (`MemoryMap.complete`) as the audit's append
              (committed by its digests alone, `audit.Store.append`), and applies each closed line's releases and
              writes (`memory_map.Replay`, the map's).  `ProverSide` sends each append after the root before the
              pass, checks the reply's ``first`` against the append's j and hands its coin c_a to `encode_append`,
              which encodes the blocks under b_j = codec.binding(d_j, j, key, c_a) (`audit.Store.encode`, the CPU
              reference of kv.cu's and paged.cu's per-block offset, or `DeviceEncoded.encode`, which hands b_j to
              kv.cu) and tells the map the encode has landed: deferred residency (remote_bandwidth's ``PROTOCOL.md``
              §4) records the blocks' writes in the line open then, the step before the first whose pass reads them
              encoded, so a block step s completed joins the plans a step or more late (one pass in flight, so its
              coin is back before step s + 1 runs: resident from s + 1) and is read as plaintext,
              uncredited, until then; then the pass reads every live block in store order; its
              ticket carries the step (`step_of`), which the root's message names, and the head of the appends so
              far, which its seed chains (`audit.chained`: unchanged in a session without appends)
openings      as the driver's reference: the block, its receipt leaf, and its payload's salt, path and d_j
~~~

``engine(s)`` runs engine step s against the store as the binder does (`verity_vllm.protocols.remote_bandwidth.binder`): a request's pages
freed after the step before (a finished request's) or in this step's schedule (a preempted one's), allocs, then
`kv_append` (or `kv_write_reference`, which writes the KV first) and `end_step` once the step's KV is in place.  A
CPU test passes a scripted engine; with vLLM, the binder on the engine, with ``passes=False``, numbers the steps.  The
Commit's session is `CommitSession`: the binder on its engine from the build, the session in lockstep with the
Commit's instrumented run (`Lockstep`: each pass's engine step is the run's next step, so the memory map's steps are
the steps the Commit commits), its store's encode on the GPU by kv.cu (`KvDevice`).

`benchmarks/remote_session` is not a package: `session_module` loads its ``prover.py`` (and ``verifier.py``, for a
local verifier) from the source tree (`config.REMOTE_SESSION`).
"""
from __future__ import annotations

import importlib.util
import json
import os
import queue
import socket
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from verity.core.protocols.remote_bandwidth import audit, codec as rcodec, receipts as rb
from verity_vllm import config
from verity_vllm.protocols.remote_bandwidth import memory_map as store_map
from verity_vllm.protocols.remote_bandwidth.registry import Store


def address(value: str) -> tuple[str, int]:
    """``HOST:PORT``, the verifier's listening address."""
    host, sep, port = str(value).rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 1 << 16:
        raise ValueError(f"a remote verifier is HOST:PORT, not {value!r}")
    return host, int(port)


def session_module(name: str, directory: str | Path = config.REMOTE_SESSION) -> ModuleType:
    """``benchmarks/remote_session``'s module ``name`` (``prover``, ``verifier``, ``wire``), loaded once from its file,
    as its own scripts import it."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(directory) / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"no remote session module {name!r} in {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


class Phases:
    """The host's time in each phase of a pass interval, from one pass's end to the next's (`close`), for the session's
    record: ``wall`` the interval, each phase timed by `time`, and what the phases leave of ``wall`` (the appends'
    round trips, the root's send) as ``other``."""

    def __init__(self) -> None:
        self.open: dict[str, int] = defaultdict(int)
        self.closed: list[dict[str, int]] = []
        self.since: int | None = None

    @contextmanager
    def time(self, name: str):
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.open[name] += time.perf_counter_ns() - t0

    def close(self) -> None:
        now = time.perf_counter_ns()
        if self.since is not None:
            self.closed.append({**self.open, "wall": now - self.since,
                                "other": now - self.since - sum(self.open.values())})
        self.open.clear()
        self.since = now

    def summary(self) -> dict[str, dict[str, float]]:
        """Each phase's p50, p99, max and mean over the intervals, in ms (0 in an interval it didn't run in)."""
        out = {}
        for k in sorted({k for x in self.closed for k in x}):
            ms = sorted(x.get(k, 0) / 1e6 for x in self.closed)
            at = lambda q: round(ms[min(len(ms) - 1, int(q * len(ms)))], 3)  # noqa: E731
            out[k] = {"p50": at(0.5), "p99": at(0.99), "max": at(1.0), "mean": round(sum(ms) / len(ms), 3)}
        return out


class Serving:
    """The serving node's steps as a session sees them, `ProverSide`'s serving trace in its ``serve`` modes: the
    memory map's appends as the audit's (`appends`, each an `audit.Append`, and `appended`, their indices), made as
    each step's blocks complete, its closed lines' releases (`released`), and `plan`, every block the record holds
    live, registered and written: an appended block joins it once its write is recorded, after its encode has landed
    (deferred residency; `encoded`).  A step that runs past a boundary ``moves`` its pass into the next window
    (`ProverSide`), so the map re-keys a window ahead (`MemoryMap.ahead`), which needs a lifetime cap of at least 2
    windows: at 1, every line would re-key every live block, those the line before re-keyed among them."""

    def __init__(self, backend: StoreBackend, registration: audit.Registration, lifetime: int) -> None:
        if type(lifetime) is not int or lifetime < 2:
            raise ValueError(f"a serving session's lifetime cap is at least 2 windows, not {lifetime!r}: its map "
                             "re-keys a window ahead, so at 1 every line re-keys every live block; give the verifier "
                             "--lifetime 2 or more, or leave it to its default under the hello's deferred residency")
        self.backend, self.store, self.lifetime = backend, backend.store, lifetime
        self._events: list[tuple] = []
        self.map = self.store.record_memory_map(
            backend.path, registration.digest, lifetime, lambda change: self._events.append(("line", change)),
            lambda first, slots, digests, payloads: self._events.append(("append", first, slots, digests, payloads)))
        self.moves, self._early = True, False
        self.map.ahead = 1
        self.step, self.steps, self.first = -1, 0, None
        self.live = np.ones(registration.blocks, bool)
        self._plan: rb.Plan | None = None
        self.appends: list[audit.Append] = []
        self.released: list[int] = []
        self.appended: list[int] = []
        self.digests: list[bytes] = []
        self.made: list[int] = []
        self.landed: list[int] = []

    @property
    def rekeyed(self) -> int:
        return self.map.rekeyed

    def ready(self) -> None:
        """Before the session arms its first window: wait for a run in lockstep to be at its first step
        (`Lockstep.arrived`), then run that step, pass 0's (`advance`), so the first window's clock starts at pass 0's
        appends and read and not at its step, as a pass moved past a boundary starts at its read (`ProverSide`)."""
        getattr(self.backend.engine, "arrived", lambda: None)()
        self.advance(None)
        self._early = True

    def advance(self, window: int | None) -> int:
        """The engine's next step s, run against the store up to its pass (in ``window``, the store's open one; None:
        before the first, from `ready`), its completed blocks appended once it has returned (`MemoryMap.complete`):
        each append made as the audit's, and each line closed applied as its releases and its writes.  Returns s: the
        session's first step takes the number the engine gives it (a binder on an engine that served before the
        session numbers on from there), and each later one the next.  Pass 0's step, run by `ready`, is taken
        as it stands."""
        if self._early:
            self._early = False
            return self.step
        if window is None:
            if self.store.window_open or self.store.window != -1:
                raise RuntimeError(f"a step before the first window, and window {self.store.window} has begun")
        elif not self.store.window_open or self.store.window != window:
            raise RuntimeError(f"the pass is in window {window}, the store's open window is {self.store.window}")
        s, phases = self.step + 1, self.backend.phases
        with phases.time("engine"):
            self.backend.engine(s)
        if self.map.last < s or (self.steps and self.map.last != s):
            raise RuntimeError(f"the engine's step after {self.step} opens one memory-map line (kv_append({s}))")
        s = self.map.last
        if not self.steps:
            self.first = s
        with phases.time("appends"):
            self._apply(s)
        self.step, self.steps = s, self.steps + 1
        return s

    def _apply(self, s: int) -> None:
        self.map.complete()
        for kind, *ev in self._events:
            if kind == "append":
                first, slots, digests, payloads = ev
                a = self.backend.append(payloads, self.backend.append_salts(first, len(payloads)), slots)
                if a.first != first or a.digests != digests:
                    raise AssertionError("the store's append is not its memory map's: a completed block's bytes "
                                         "changed between its digest and its append")
                self.appends.append(a)
                self.appended.extend(range(a.first, a.first + a.count))
                self.digests.extend(a.digests)
                self.made.append(s)
                continue
            (change,) = ev
            if change.digests != tuple(self.digests[change.first - self.map.header.blocks:][:change.count]):
                raise AssertionError(f"step {change.step}'s writes from {change.first} are not the blocks appended there")
            gone = [j for f, c in change.released for j in range(f, f + c)]
            self.released.extend(gone)
            if len(self.live) < change.first + change.count:
                self.live = np.concatenate([self.live, np.zeros(change.first + change.count - len(self.live), bool)])
            self.live[gone] = False
            self.live[change.first:change.first + change.count] = True
            self._plan = None
        self._events.clear()

    def encoded(self, first: int) -> None:
        """The append from ``first`` is encoded: the record writes its blocks in the line open now, at the step the
        trace last advanced to, so they join the plan from the next step's pass on."""
        self.map.encoded(first)
        self.landed.append(self.step)

    def deferral(self) -> dict[str, Any]:
        """The appends made, those whose encodes landed, how many steps after its append's each landed (its blocks
        resident from the step after that), and each append's ``[step made, step landed (None: not yet), first,
        count]``: public structure, as the verifier sees each append between the roots that name their steps."""
        late: dict[int, int] = {}
        for m, w in zip(self.made, self.landed):
            late[w - m] = late.get(w - m, 0) + 1
        landed = self.landed + [None] * (len(self.made) - len(self.landed))
        return {"appends": len(self.made), "encoded": len(self.landed), "steps_late": dict(sorted(late.items())),
                "waiting": self.map.lines.waiting if self.store.memory_map is self.map else None,
                "encodes": [[m, w, a.first, a.count] for m, w, a in zip(self.made, landed, self.appends)]}

    def plan(self) -> rb.Plan:
        """Every live block, kept until a closed line changes them or an append extends the store."""
        if self._plan is None or self._plan.blocks != self.map.extent:
            with self.backend.phases.time("plan"):
                edge = np.flatnonzero(self.live[1:] != self.live[:-1]) + 1
                cuts = np.concatenate([[0] if self.live[0] else [], edge, [len(self.live)] if self.live[-1] else []])
                runs = np.stack([cuts[0::2], cuts[1::2] - cuts[0::2]], axis=1).astype(np.int64)
                self._plan = rb.Plan.from_array(runs, self.map.extent)
        return self._plan


class StoreBackend:
    """`ProverSide`'s backend over ``store``, whose engine steps ``engine(s)`` runs (the module's docstring).  Its
    passes are the driver's reference over its `audit.Store` (`receipts.receipter` over each plan's blocks, its tree
    `receipts.tree`), hiding salts from ``randbytes``; ``path`` is the memory map's file (None: kept in memory).  With
    ``device`` (`KvDevice`) the store's blocks are encoded on the GPU (`DeviceEncoded`).  An appended block is resident
    once its encode lands (`deferred`, which the hello says), so the verifier refuses it a lifetime of one window."""

    deferred = True

    def __init__(self, store: Store, engine: Callable[[int], None], *, randbytes: Callable[[int], bytes] = os.urandom,
                 path: str | Path | None = None, read_bytes_per_s: float = 1e12, device: KvDevice | None = None) -> None:
        self.store, self.engine, self.path, self.randbytes, self.device = store, engine, path, randbytes, device
        self.blocks, self.read_bytes_per_s = store.layout().blocks, read_bytes_per_s
        self.dropped: set[int] = set()
        self.gates: dict = {}
        self.trace: Serving | None = None
        self.previous: bytes | None = None
        self.phases = Phases()
        self.step_ns = 0

    def layout(self) -> list[dict]:
        """The hello's public layout: each registered region's kind, tier and blocks, in store order."""
        return [{"kind": r.kind, "tier": r.tier, "first": r.first, "blocks": r.blocks}
                for r in self.store.layout().regions]

    def payload_root(self, node: bytes) -> bytes:
        self.payloads = [self.store.payload(x) for x in range(self.blocks)]
        self.salts = [self.randbytes(rb.SALT_BYTES) for _ in range(self.blocks)]
        return audit.payload_tree(node, self.payloads, self.salts).commitment.root

    def encode(self, registration: audit.Registration) -> dict:
        if any(self.store.payload(x) != m for x, m in enumerate(self.payloads)):
            raise RuntimeError("the store changed between its hello and its registration")
        self.registration = registration
        if self.device is None:
            self.audit = audit.Store(registration, self.payloads, self.salts)
        else:
            self.audit = DeviceEncoded(registration, self.payloads, self.salts, self.device)
            self.gates["device_encode"] = self.audit.gates
        return {"encode_bytes_per_s": 1.0}

    def begin_session(self) -> None:
        if self.trace is not None:
            raise RuntimeError("a store serves one session: its memory map's indices are the session's")

    def serving(self, registration: audit.Registration, lifetime: int) -> Serving:
        self.trace = Serving(self, registration, lifetime)
        return self.trace

    def append_salts(self, first: int, count: int) -> list[bytes]:
        """The payload salts of an append's blocks ``first`` …, hiding each digest in its tree."""
        return [self.randbytes(rb.SALT_BYTES) for _ in range(count)]

    def append(self, payloads, salts, slots=()) -> audit.Append:
        return self.audit.append(payloads, salts)

    def encode_append(self, first: int, coin: bytes) -> None:
        self.audit.encode(first, coin)
        if self.trace is not None:
            self.trace.encoded(first)

    def holds(self, slot: int, j: int, digest: bytes) -> bool:
        """Whether ``slot`` holds bytes of the payload of digest d_j (`store_map.agreement`'s check of an appended block)."""
        return rcodec.digest(self.store.payload(slot)) == digest

    def calibrate(self) -> dict:
        return self.with_step({"pass_ns_max": 0, "gap_ns_max": 0, "read_bytes_per_s": self.read_bytes_per_s})

    def with_step(self, cal: dict) -> dict:
        """``cal`` with the engine step a paced pass waits on (``step_ns``) added to each of its pass times, so the
        verifier's allowance covers the step and the read: a paced pass's step runs inside its coin's window."""
        if not self.step_ns:
            return cal
        out = {**cal, "step_ns": self.step_ns}
        for k in ("pass_ns_max", "pass_ns_median"):
            if k in cal:
                out[k] = cal[k] + self.step_ns
        if "pass_ns" in cal:
            out["pass_ns"] = {q: v + self.step_ns for q, v in cal["pass_ns"].items()}
        return out

    def arm(self, dropped) -> None:
        self.dropped = {int(j) for j in dropped}

    def begin_window(self, challenge: bytes, previous: bytes | None) -> None:
        """The session's next window, the store's too (the first chained from the registration its map names)."""
        if self.store.window_open:
            self.store.end_window()
        self.store.begin_window(challenge, self.registration.digest if self.store.window < 0 else None)
        self.challenge = challenge
        if previous is not None:
            self.previous = previous

    def launch(self, i: int, after: dict | None, plan: rb.Plan | None = None) -> dict:
        """Pass i of the window over ``plan`` (None: every registered block) at the step the trace last advanced to,
        chained from ``after``'s root (None: the window's previous root) and the head of every append so far
        (`audit.Store.head`, all sent with this pass's root or before it), run when awaited."""
        return {"challenge": self.challenge, "i": i, "after": after, "previous": self.previous, "root": None,
                "plan": plan, "step": None if self.trace is None else self.trace.step, "head": self.audit.head}

    def step_of(self, ticket: dict) -> int | None:
        return ticket["step"]

    def wait(self, ticket: dict):
        after = ticket["after"]
        prev = audit.chained(ticket["previous"] if after is None else after["root"], ticket["head"])
        root, handle = self.run_pass(rb.seed(ticket["challenge"], ticket["i"], prev), ticket["plan"])
        ticket["root"] = root
        return root, handle

    def run_pass(self, seed: bytes, plan: rb.Plan | None = None):
        plan = plan or rb.Plan.whole(self.blocks)
        if waiting := [j for f, c in plan.runs for j in range(f, f + c) if self.audit.blocks[j] is None][:4]:
            raise RuntimeError(f"the plan holds blocks {waiting}, appended and not yet encoded: their writes are "
                               "recorded once their encodes land (deferred residency)")
        reg = self.registration
        r = rb.receipter(reg.receipt, seed, reg.rows)
        ys = tuple(r(j, bytes(rcodec.BLOCK_BYTES) if j in self.dropped else self.audit.blocks[j]) for j in plan)
        salts = tuple(self.randbytes(rb.SALT_BYTES) for _ in range(rb.leaves(plan.length)))
        tree = rb.tree(seed, plan, ys, salts)
        return tree.commitment.root, (ys, salts, tree, plan)

    def open_window(self, items) -> list[list[tuple]]:
        return [self.open(handle, positions) for handle, positions in items]

    def open(self, handle, positions: list[int]) -> list[tuple]:
        ys, salts, tree, plan = handle
        out = []
        for p in positions:
            c, j = p // rb.PER_LEAF, plan.block(p)
            leaf = b"".join(ys[rb.PER_LEAF * c:rb.PER_LEAF * (c + 1)])
            out.append((self.audit.blocks[j], leaf, salts[c], tree.open(c).path, *self.audit.committed(j)))
        return out


class KvDevice:
    """The store's encode on the GPU by kv.cu (``benchmarks/word_codec/kv.cu``, a build at the codec's field and mask
    loaded as `kv_bench.Lib`): blocks under the seed's key by ``kv_encode``; an append's under their bound keys and its
    coin by ``kv_encode_bound``, each block's offset b_j(c_a) = `codec.binding`(d_j, j, key, c_a) handed to the device
    by ``paged.offsets`` (`config.VLLM_ENCODED`), so the host computes only d_j and b_j."""

    CHUNK = 1 << 14

    def __init__(self, path: str | Path, codec: rcodec.Codec = rcodec.DEFAULT) -> None:
        import torch

        self.torch, self.codec = torch, codec
        self.paged = session_module("paged", config.VLLM_ENCODED)
        self.lib, self.field = self.paged.KB.Lib(codec.field.name, str(path)), self.paged.K.field(codec.field.name)
        if self.lib.mask != codec.masked or self.lib.kmax < codec.rounds:
            raise ValueError(f"{path} is kv.cu at {self.field.name}, masked={self.lib.mask}, at most {self.lib.kmax} "
                             f"rounds: not {codec.name}")
        if self.paged.K.token_layer_words(self.field) != codec.field.block_words:
            raise ValueError("kv.cu's bound offsets are a token-layer's: build it with a token-layer of one 4 KiB block")

    def set_key(self, key: rcodec.Key, lib: Any = None) -> None:
        """``key`` on kv.cu's build, or on ``lib`` (another build at the codec's field, paged.cu's)."""
        k = self.paged.K.key(self.field.name, key.seed, key.codec.rounds, mask=key.codec.masked)
        if (k.c, k.tau, k.mask) != (key.c, key.tau, key.mask):
            raise RuntimeError("kv.py's key is not the registration's codec key")
        self.paged.KB.set_key(self.lib if lib is None else lib, k)

    def encode(self, payloads: Sequence[bytes], first: int, offsets: Sequence[int] | None = None) -> list[bytes]:
        """Blocks ``first`` … of ``payloads``: under the seed's key, or each under its bound offset."""
        torch, w, out = self.torch, self.codec.field.block_words, []
        stream = torch.cuda.current_stream().cuda_stream
        for u in range(0, len(payloads), self.CHUNK):
            part = payloads[u:u + self.CHUNK]
            m = torch.frombuffer(bytearray(b"".join(part)), dtype=torch.uint8).cuda()
            e = torch.empty_like(m)
            n, base, k = len(part) * w, (first + u) * w, self.codec.rounds
            if offsets is None:
                self.lib.call("kv_encode", m.data_ptr(), e.data_ptr(), n, base, k, 0, stream)
            else:
                b = self.paged.offsets(torch, list(offsets[u:u + self.CHUNK]), self.field)
                self.lib.call("kv_encode_bound", m.data_ptr(), e.data_ptr(), n, base, k, b.data_ptr(), 0, stream)
            raw = e.cpu().numpy().tobytes()
            out.extend(raw[i * rcodec.BLOCK_BYTES:(i + 1) * rcodec.BLOCK_BYTES] for i in range(len(part)))
        return out

    def encode_in_place(self, data: Any, first: int, offsets: Sequence[int] | None = None) -> None:
        """Blocks ``first`` … held contiguously on the GPU at ``data`` (a uint8 tensor of whole blocks), encoded where
        they are: under the seed's key, or each under its bound offset."""
        torch, w, k = self.torch, self.codec.field.block_words, self.codec.rounds
        stream, n = torch.cuda.current_stream().cuda_stream, data.numel() // rcodec.BLOCK_BYTES
        if not data.is_contiguous() or data.numel() != n * rcodec.BLOCK_BYTES:
            raise ValueError("an in-place encode is of whole blocks held contiguously")
        for u in range(0, n, self.CHUNK):
            ptr, c = data.data_ptr() + u * rcodec.BLOCK_BYTES, min(self.CHUNK, n - u)
            if offsets is None:
                self.lib.call("kv_encode", ptr, ptr, c * w, (first + u) * w, k, 0, stream)
            else:
                b = self.paged.offsets(torch, list(offsets[u:u + c]), self.field)
                self.lib.call("kv_encode_bound", ptr, ptr, c * w, (first + u) * w, k, b.data_ptr(), 0, stream)

    def decode(self, src: int, dst: int, blocks: int, first: int) -> None:
        """``blocks`` blocks at device address ``src``, the store's blocks ``first`` … encoded under the seed's key,
        decoded to ``dst`` (which may be ``src``) on the current stream."""
        w, stream = self.codec.field.block_words, self.torch.cuda.current_stream().cuda_stream
        self.lib.call("kv_decode", src, dst, blocks * w, first * w, self.codec.rounds, 0, None, stream)


class DeviceEncoded(audit.Store):
    """`audit.Store` with its encodes on a `KvDevice`: the registered blocks at once, an append's once its coin
    arrives.  Before the store takes them, the device's blocks are checked against `codec.encode`: the registration's
    first and last, and each append's first (`gates`)."""

    def __init__(self, registration: audit.Registration, payloads: Sequence[bytes], salts: Sequence[bytes],
                 device: KvDevice) -> None:
        if len(payloads) != registration.blocks:
            raise ValueError(f"the registration is of {registration.blocks} blocks")
        self.registration, self.device = registration, device
        self.payloads, self.salts = tuple(payloads), tuple(salts)
        self.payload_tree = audit.payload_tree(registration.node, self.payloads, self.salts)
        if self.payload_tree.commitment.root != registration.payload_root:
            raise ValueError("these payloads are not the registered ones")
        device.set_key(registration.key)
        encoded = device.encode(self.payloads, 0)
        for j in sorted({0, len(encoded) - 1}):
            self._gate(encoded[j], self.payloads[j], j)
        self.blocks: list[bytes | None] = list(encoded)
        self.appends: list[audit.Append] = []
        self.pending: dict[int, tuple[bytes, ...]] = {}
        self.gates = {"registered": len(encoded), "appends": 0, "blocks_checked": len({0, len(encoded) - 1})}

    def _gate(self, block: bytes, payload: bytes, j: int, coin: bytes | None = None) -> None:
        if block != rcodec.encode(payload, j, self.registration.key, bound=coin is not None, coin=coin):
            raise RuntimeError(f"kv.cu's block {j} is not codec.encode's")

    def encode(self, first: int, coin: bytes) -> None:
        payloads = self.pending.get(first)
        if payloads is None:
            raise ValueError(f"no append from block {first} awaits its coin")
        digests = next(a.digests for a in reversed(self.appends) if a.first == first)
        k = self.registration.key
        encoded = self.device.encode(payloads, first, [rcodec.binding(d, first + u, k, coin) for u, d in enumerate(digests)])
        self._gate(encoded[0], payloads[0], first, coin)
        del self.pending[first]
        self.blocks[first:first + len(encoded)] = encoded
        self.gates["appends"] += 1
        self.gates["blocks_checked"] += 1


class Refused(RuntimeError):
    """The verifier refused the store's registration, answering its hello with ``bye``: a node has one live
    registration (`audit.Registry`), which only the verifier closes.  ``why`` is the verifier's reason."""

    def __init__(self, verifier: str, why: str) -> None:
        super().__init__(f"the verifier at {verifier} refused the store's registration: {why}")
        self.verifier, self.why = verifier, why


def prove(backend: StoreBackend, verifier: str, plan: str = "serve", *, node: bytes = b"verity-vllm/store",
          codec: rcodec.Codec = rcodec.DEFAULT, log: Callable[[str], None] = print, **kw: Any) -> list[dict]:
    """The prover's side of a remote session against the verifier at ``verifier`` (``HOST:PORT``): register the
    store, then run each session of ``plan`` (`ProverSide`'s modes: ``serve``, or ``servedrop:K``, which withholds the
    first K blocks appended) as the verifier starts it.  Returns each session's result; raises `Refused` with the
    verifier's reason if it refuses the registration.  A prover that fails drops the connection, so the verifier sees
    the session end there."""
    host, port = address(verifier)
    prover, wire = session_module("prover"), session_module("wire")
    conn = wire.connect(host, port)
    try:
        p = prover.ProverSide(conn, backend, node, prover.parse_plan(plan), log=log, **kw)
        try:
            p.register(codec=codec)
        except wire.Closed as e:
            if str(e).startswith("refused"):
                raise Refused(verifier, str(e)) from None
            raise
        out = []
        for mode, fraction in p.plan:
            head, _, _ = p.expect("session")
            out.append(p.session(head, mode, fraction))
        return out
    except BaseException:
        try:
            conn.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn.sock.close()
        raise


class Lockstep:
    """A run's engine steps as a session's: the session's backend, on the session's thread, asks for its next engine
    step (``lockstep(s)``, the backend's ``engine``) and waits; the run, on its own thread, takes each ask in `step`,
    runs its engine step and hands back.  So the session's steps are the run's first steps, one a pass, and one thread
    at a time touches the store.  The session waits for the run to be at its first step (`arrived`) and runs it, pass
    0's, before it arms its first window (`Serving.ready`), so the window's clock starts at pass 0's read and not at
    the run's setup or its prefill.  Once the session
    has ended (`close`), the run's later steps are unaudited (``unaudited``); a run that ends while the session still
    asks for steps fails the session (`finish`)."""

    def __init__(self) -> None:
        self._asks: queue.Queue = queue.Queue()
        self._done: queue.Queue = queue.Queue()
        self._ask: int | None = None
        self._begun, self._arrived = threading.Event(), threading.Event()
        self.ended = False
        self.audited = self.unaudited = 0

    def __call__(self, s: int) -> None:
        self._asks.put(s)
        why = self._done.get()
        if why is not None:
            raise RuntimeError(why)

    def arrived(self) -> None:
        """The session's, registered and about to arm its first window: let the run begin (`begun`) and wait until it
        is at its first step, or done."""
        self._begun.set()
        self._arrived.wait()

    def begun(self) -> None:
        """Wait until the session waits for the run's first step (`arrived`), or has ended."""
        self._begun.wait()

    def close(self) -> None:
        self._begun.set()
        self._asks.put(None)

    def ready(self) -> bool:
        """Wait until the session asks for a step (True) or has ended (False)."""
        if not self.ended and self._ask is None:
            self._ask = self._asks.get()
            self.ended = self._ask is None
        return not self.ended

    def step(self, run: Callable[[], Any]) -> Any:
        """The run's next engine step ``run()``: the session's, if it asks for one."""
        self._arrived.set()
        if not self.ready():
            self.unaudited += 1
            return run()
        self._ask = None
        try:
            out = run()
        except BaseException as e:
            self._done.put(f"the run's engine step raised {type(e).__name__}: {e}")
            raise
        self.audited += 1
        self._done.put(None)
        return out

    def finish(self) -> None:
        """The run is done: a session that asks for another step fails."""
        self._arrived.set()
        if self.ready():
            self._ask = None
            self._done.put(f"the run ended after {self.audited} steps, before the session's last pass")


class CommitSession:
    """The Commit's remote bandwidth session (``--remote-verifier HOST:PORT``, `pipeline.commit`), on its engine:

    ~~~text
    engine_args  before the build: one GPU, synchronous scheduling, no prefix caching (refusing anything else), the KV
                 pool at ``kv_pages`` pages (None: the engine's own, at the row's gpu_memory_utilization)
    target       before the build: the row's target declares synchronous scheduling (its Program's LAG 0)
    install      before the build: ``binder.Binder(store, passes=False, kv=False)`` on vLLM's model runner, block pool
                 and scheduler, registering weight tensors up to ``weight_blocks`` blocks.  The KV pool stays outside
                 the registration (served writes do, and live KV is uncredited for chat),
                 so no step writes a registered block and every pass reads every one
    check        after the build: the engine core in this process, scheduling synchronous, no KV pool registered; then
                 ``filler_gib`` GiB of zeros on the engine's GPU registered as the store's filler (`Binder.register_filler`)
    begin        before the Commit's first instrumented run: the session on its own thread, registered before the run
                 goes on, and its first window armed once the run's first step has run (`Serving.ready`).  Its
                 backend is ``backend_of(session, path)``'s (the store read where the engine holds it,
                 `in_place.InPlaceBackend.for_commit`, kv.cu at ``kv_lib`` encoding it in place) or, without one, a host mirror encoded by kv.cu at ``kv_lib`` (`KvDevice`) or
                 on the CPU
    (the run)    each of its engine steps through the returned `Lockstep`: a pass's step while the session lasts, so the
                 memory map, recorded against the registration's digest beside the Commit (``out``), is of the run's
                 steps from its first (a header and no line: no step writes the store); the steps after it unaudited
    end          after the run: the session done (failed if it still asked for a step), its record
                 ``remote_session.json`` with the map's agreement with the store (`store_map.agreement`), taken before the run
                 went on, each pass's ``[window, pass, step]`` as the verifier's record has its steps
                 (``pass_steps``, each a step of the map), and the map stopped there (`store.memory_map.end`)
    ~~~

    A store serves one session, so ``plan`` is one mode, ``serve`` (``servedrop:K`` withholds blocks appended, and
    this store appends none), and the session audits one run: with more than one pair, the later instrumented runs are
    committed and unaudited.  The control arm is never audited.  A verifier that paces the session (``--paced 1:…``: a
    live coin a pass) has each pass's engine step run inside its coin's window, so ``step_ms``, the step's time, goes
    into the calibration's pass times (`StoreBackend.with_step`) and the allowance covers it."""

    def __init__(self, verifier: str, *, plan: str = "serve", kv_pages: int | None = 32,
                 weight_blocks: int | None = None, kv_lib: str | None = None,
                 backend_of: Callable[[CommitSession, Path | None], StoreBackend] | None = None,
                 out: str | Path | None = None, node: str | None = None, codec: rcodec.Codec = rcodec.DEFAULT,
                 step_ms: float = 0.0, filler_gib: float = 0.0, log: Callable[[str], None] = print) -> None:
        address(verifier)
        if not step_ms >= 0 or not filler_gib >= 0:
            raise ValueError("the engine step a paced pass waits on, and the filler, are nonnegative")
        node = f"verity-vllm/{socket.gethostname()}" if node is None else node
        if not 1 <= len(node.encode()) <= audit.NODE_BYTES:
            raise ValueError(f"the session's node id {node!r} is not 1 to {audit.NODE_BYTES} bytes")
        if "," in plan:
            raise ValueError("a store serves one session: --remote-plan is one mode")
        if session_module("prover").parse_plan(plan)[0][0] != "serve":
            raise ValueError(f"the Commit's session is serve, not {plan!r}: its store (weights and filler) appends nothing")
        if kv_pages is not None and kv_pages < 1:
            raise ValueError("the session's KV pool is at least a page (None: the engine's own)")
        if backend_of is not None and kv_lib is None:
            raise ValueError("a store read in place is encoded in place by kv.cu: give kv_lib with backend_of")
        self.verifier, self.plan, self.kv_pages, self.weight_blocks = verifier, plan, kv_pages, weight_blocks
        self.kv_lib, self.backend_of, self.step_ms, self.filler_gib = kv_lib, backend_of, step_ms, filler_gib
        self.out, self.node, self.codec, self.log = None if out is None else Path(out), node, codec, log
        self.store, self.binder = Store(), None
        self.lockstep: Lockstep | None = None
        self.backend: StoreBackend | None = None
        self.record: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None

    def engine_args(self, engine_args: dict[str, Any]) -> dict[str, Any]:
        why = []
        if int(engine_args.get("tensor_parallel_size") or 1) != 1 or int(engine_args.get("pipeline_parallel_size") or 1) != 1:
            why.append("parallelism (a store is one GPU's: a TP Commit's session is not wired yet)")
        if engine_args.get("async_scheduling"):
            why.append("async scheduling (a pass follows each step's schedule and its output in turn)")
        if engine_args.get("enable_prefix_caching"):
            why.append("prefix caching (a prefix hit brings a page back with its KV)")
        if why:
            raise SystemExit("--remote-verifier refuses " + "; ".join(why))
        pool = {} if self.kv_pages is None else {"num_gpu_blocks_override": self.kv_pages}
        return {**engine_args, "async_scheduling": False, "enable_prefix_caching": False, **pool}

    def target(self, target: Any) -> None:
        """Before the build: the row's target declares synchronous scheduling, so its Program is the one the engine
        runs (a target that leaves it unset is specialized for the scheduler vLLM resolves, async, LAG 1)."""
        got = getattr(target, "async_scheduling", None)
        if got is not False:
            raise SystemExit(f"--remote-verifier: the row's target declares async_scheduling {got!r}, and the session's "
                             "engine schedules synchronously: declare \"async_scheduling\": false in the workload's target")

    def install(self) -> None:
        from verity_vllm.protocols.remote_bandwidth.binder import Binder

        self.binder = Binder(self.store, passes=False, weight_blocks=self.weight_blocks, kv=False).install()

    def before_build(self, ekw: dict[str, Any]) -> None:
        """`engine_args`, `target` and `install` on the Commit's engine build arguments (``engine_args``, ``target``)."""
        ekw["engine_args"] = self.engine_args(ekw["engine_args"])
        self.target(ekw["target"])
        self.install()

    def check(self, llm: Any, runner: Any) -> dict[str, Any]:
        vc, why = llm.llm_engine.vllm_config, []
        if getattr(getattr(llm.llm_engine, "engine_core", None), "engine_core", None) is None:
            why.append("the engine core in another process")
        if vc.scheduler_config.async_scheduling:
            why.append("async scheduling resolved on")
        if vc.cache_config.enable_prefix_caching:
            why.append("prefix caching on")
        if vc.parallel_config.tensor_parallel_size != 1:
            why.append(f"tensor parallel size {vc.parallel_config.tensor_parallel_size}")
        if "kv" in {r.kind for r in self.store.layout().regions}:
            why.append("a KV pool registered (served writes stay outside the registration)")
        if why:
            raise SystemExit("--remote-verifier: the engine is not one the session serves: " + "; ".join(why))
        if self.filler_gib:
            self.binder.register_filler(int(self.filler_gib * (1 << 30)), runner.device)
        stats = self.binder.stats()
        self.log(f"[delta] remote session: the store binder on the engine, {stats}")
        return stats

    def begin(self) -> Lockstep:
        if self.lockstep is not None:
            raise RuntimeError("a store serves one session: the Commit's session audits one run")
        self.lockstep = Lockstep()
        path = None if self.out is None else self.out / "memory_map.jsonl"
        if self.backend_of is not None:
            self.backend = self.backend_of(self, path)
        else:
            device = None if self.kv_lib is None else KvDevice(self.kv_lib, self.codec)
            self.backend = StoreBackend(self.store, self.lockstep, path=path, device=device)
        self.backend.step_ns = round(self.step_ms * 1e6)
        self._thread = threading.Thread(target=self._session, name="remote-session", daemon=True)
        self._thread.start()
        self.lockstep.begun()
        return self.lockstep

    def _session(self) -> None:
        backend, error, result, agree = self.backend, None, {}, {}
        try:
            (result,) = prove(backend, self.verifier, self.plan, node=self.node.encode(), codec=self.codec, log=self.log,
                              warmup_ns=round(getattr(backend, "warmup_ms", 0.0) * 1e6))
            agree = store_map.agreement(backend)
        except Exception as e:  # noqa: BLE001  -- recorded: a session that fails is rejected, and the Commit's verdict still written
            error = f"{type(e).__name__}: {e}"
        try:
            getattr(backend, "close", lambda: None)()
            trace = backend.trace
            steps = result.get("pass_steps") or []
            self.record = {
                "schema": "verity-vllm/remote-session/v0", "verifier": self.verifier, "plan": self.plan,
                "accepted": error is None and bool(result.get("accepted")), "error": error, "prover": result,
                "store": self.binder.stats() if self.binder is not None else None, "layout": backend.layout(),
                "kv_pages": self.kv_pages, "weight_blocks": self.weight_blocks, "step_ms": self.step_ms,
                "filler_gib": self.filler_gib,
                "encode": ("kv.cu in place" if self.backend_of else "kv.cu") if backend.device is not None else "cpu",
                "gates": backend.gates, "device": getattr(backend, "session_stats", dict)(),
                "memory_map": None if trace is None else {
                    "path": None if backend.path is None else str(backend.path),
                    "registration": backend.registration.digest.hex(), "first_step": trace.first, "steps": trace.steps,
                    "rekeyed": trace.rekeyed, "deferral": trace.deferral(), "pass_steps": steps,
                    "pass_steps_mapped": trace.first is not None and all(
                        trace.first <= x <= trace.step for *_, x in steps),
                    **agree}}
            store_map.end(self.store)
        finally:
            self.lockstep.close()

    def end(self) -> dict[str, Any]:
        self.lockstep.finish()
        self._thread.join()
        rec = self.record if self.record is not None else {
            "schema": "verity-vllm/remote-session/v0", "verifier": self.verifier, "plan": self.plan, "accepted": False,
            "error": "the session's record was not written (its thread raised)", "memory_map": None, "encode": None}
        lock = self.lockstep
        rec["run"] = {"steps": lock.audited + lock.unaudited, "audited": lock.audited, "unaudited": lock.unaudited}
        self.record = rec
        if self.out is not None:
            (self.out / "remote_session.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
        return rec

