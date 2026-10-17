"""The remote bandwidth session over the node's `Store`, inside the prover's process (the Commit's, or the serve
driver's): `StoreBackend` is the backend `benchmarks/remote_session`'s `ProverSide` drives, the role `device.Device`
and `CpuBackend` play there, and `prove` connects it to a verifier at ``HOST:PORT`` (the row's ``REMOTE_VERIFIER``,
`config.CommitConfig.remote_verifier`).

~~~text
hello         the node, N = the store's blocks (`Store.layout`) and the payload root of every block's payload as the
              store holds it now (`Store.payload`: weights, the KV pool's pages as filler, host filler)
registration  the verifier's; the backend encodes the payloads under its key (`audit.Store`, the driver's reference
              encoding, which the device's in-place codec replaces on a GPU)
session       the store records its memory map against `Registration.digest` under the session's lifetime cap L
              (`Store.record_memory_map`), one session a store: the map's indices are one session's
pass          before each pass `Serving.advance` runs the engine's next step against the store (its frees, allocs and
              `kv_append`, which closes the step before's memory-map line with its re-keys for the pass's window), turns
              each closed line into the audit's append (each written block's payload encoded under its bound key) and
              releases (`memory_map.Replay`, the map's), and the pass reads every live block in store order; its
              ticket carries the step (`step_of`), which the root's message names, and the head of the appends so
              far, which its seed chains (`audit.chained`: unchanged in a session without appends)
openings      as the driver's reference: the block, its receipt leaf, and its payload's salt, path and d_j
~~~

``engine(s)`` runs engine step s against the store as the binder does (`verity_vllm.store.binder`): a request's pages
freed after the step before (a finished request's) or in this step's schedule (a preempted one's), allocs, then
`kv_append` (or `kv_write_reference`, which writes the KV first) and `end_step` once the step's KV is in place.  A
CPU test passes a scripted engine; with vLLM, `serve` passes ``LLMEngine.step`` and the binder on the engine, with
``passes=False``, numbers the steps.  The Commit refuses ``--remote-verifier`` (`unwired`) until it puts a binder on its engine.

`benchmarks/remote_session` is not a package: `session_module` loads its ``prover.py`` (and ``verifier.py``, for a
local verifier) from the source tree (`config.REMOTE_SESSION`).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from verity.core.protocols.remote_bandwidth import audit, codec as rcodec, receipts as rb
from verity_vllm import config
from verity_vllm.store.registry import Store


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


class Serving:
    """The serving node's steps as a session sees them, `ProverSide`'s serving trace in its ``serve`` modes: the
    memory map's closed lines as the audit's appends (`appends`, (root, count) each, and `appended`, their indices) and
    releases (`released`), and `plan`, every live block, registered and appended."""

    def __init__(self, backend: StoreBackend, registration: audit.Registration, lifetime: int) -> None:
        self.backend, self.store, self.lifetime = backend, backend.store, lifetime
        self._lines: list[tuple[Any, list[bytes]]] = []
        self.map = self.store.record_memory_map(backend.path, registration.digest, lifetime,
                                                lambda change, payloads: self._lines.append((change, payloads)))
        self.step = -1
        self.live = set(range(registration.blocks))
        self.appends: list[tuple[bytes, int]] = []
        self.released: list[int] = []
        self.appended: list[int] = []

    @property
    def steps(self) -> int:
        return self.step + 1

    @property
    def rekeyed(self) -> int:
        return self.map.rekeyed

    def advance(self, window: int) -> int:
        """The engine's next step s, run against the store up to its pass (in ``window``, the store's open one): each
        line it closed applied as the audit's append and releases.  Returns s."""
        if not self.store.window_open or self.store.window != window:
            raise RuntimeError(f"the pass is in window {window}, the store's open window is {self.store.window}")
        s = self.step + 1
        self.backend.engine(s)
        if self.map.last != s:
            raise RuntimeError(f"engine step {s} opens its memory-map line (kv_append({s}))")
        for change, payloads in self._lines:
            if change.count:
                a = self.backend.append(payloads, [self.backend.randbytes(rb.SALT_BYTES) for _ in payloads])
                if a.first != change.first or a.digests != change.digests:
                    raise AssertionError("the store's append is not its memory map's: a written block's bytes changed "
                                         "between its digest and its line's close")
                self.appends.append((a.root, a.count))
                self.appended.extend(range(a.first, a.first + a.count))
            gone = [j for f, c in change.released for j in range(f, f + c)]
            self.released.extend(gone)
            self.live.difference_update(gone)
            self.live.update(range(change.first, change.first + change.count))
        self._lines.clear()
        self.step = s
        return s

    def plan(self) -> rb.Plan:
        out: list[list[int]] = []
        for j in sorted(self.live):
            if out and out[-1][0] + out[-1][1] == j:
                out[-1][1] += 1
            else:
                out.append([j, 1])
        return rb.Plan(tuple((f, c) for f, c in out), self.map.extent)


class StoreBackend:
    """`ProverSide`'s backend over ``store``, whose engine steps ``engine(s)`` runs (the module's docstring).  Its
    passes are the driver's reference over its `audit.Store` (`receipts.receipter` over each plan's blocks, its tree
    `receipts.tree`), hiding salts from ``randbytes``; ``path`` is the memory map's file (None: kept in memory)."""

    def __init__(self, store: Store, engine: Callable[[int], None], *, randbytes: Callable[[int], bytes] = os.urandom,
                 path: str | Path | None = None, read_bytes_per_s: float = 1e12) -> None:
        self.store, self.engine, self.path, self.randbytes = store, engine, path, randbytes
        self.blocks, self.read_bytes_per_s = store.layout().blocks, read_bytes_per_s
        self.dropped: set[int] = set()
        self.gates: dict = {}
        self.trace: Serving | None = None
        self.previous: bytes | None = None

    def payload_root(self, node: bytes) -> bytes:
        self.payloads = [self.store.payload(x) for x in range(self.blocks)]
        self.salts = [self.randbytes(rb.SALT_BYTES) for _ in range(self.blocks)]
        return audit.payload_tree(node, self.payloads, self.salts).commitment.root

    def encode(self, registration: audit.Registration) -> dict:
        if any(self.store.payload(x) != m for x, m in enumerate(self.payloads)):
            raise RuntimeError("the store changed between its hello and its registration")
        self.registration = registration
        self.audit = audit.Store(registration, self.payloads, self.salts)
        return {"encode_bytes_per_s": 1.0}

    def begin_session(self) -> None:
        if self.trace is not None:
            raise RuntimeError("a store serves one session: its memory map's indices are the session's")

    def serving(self, registration: audit.Registration, lifetime: int) -> Serving:
        self.trace = Serving(self, registration, lifetime)
        return self.trace

    def append(self, payloads, salts) -> audit.Append:
        return self.audit.append(payloads, salts)

    def calibrate(self) -> dict:
        return {"pass_ns_max": 0, "gap_ns_max": 0, "read_bytes_per_s": self.read_bytes_per_s}

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
    verifier's reason if it refuses the registration."""
    host, port = address(verifier)
    prover, wire = session_module("prover"), session_module("wire")
    p = prover.ProverSide(wire.connect(host, port), backend, node, prover.parse_plan(plan), log=log, **kw)
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


def serve(verifier: str, store: Store, step: Callable[[], Any], **kw: Any) -> list[dict]:
    """The Commit's hook (``--remote-verifier HOST:PORT``): `prove` over ``store``, with ``binder.Binder(store,
    passes=False)`` on the engine (the session runs the passes and windows), each pass after one engine step
    ``step()`` (``LLMEngine.step``), which the binder numbers and records."""
    return prove(StoreBackend(store, lambda s: step()), verifier, **kw)


def unwired(verifier: str) -> str:
    """The Commit's refusal of ``--remote-verifier`` until it puts a `binder.Binder` on its engine (the GPU Commit's
    work): without one it has no `Store` to `serve`."""
    address(verifier)
    return (f"--remote-verifier {verifier}: this Commit puts no store binder (verity_vllm.store.binder) on its engine, so "
            "it has no Store to serve the remote bandwidth session over (verity_vllm.store.remote.serve); run the "
            "session from a serve driver that does")
