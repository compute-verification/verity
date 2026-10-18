"""The prover's process: registers a store with the verifier, then runs chained passes over it in each session, sends
every root as its pass ends, and opens what the verifier draws.

A backend holds the store and makes the passes: `device.Device` on the GPUs (the store sharded across them, each pass
every shard's sketch-sha512 receipts and the one tree over the whole store), or `CpuBackend`, the driver's own
reference (`audit.Store`, `receipts`), which the suite runs over a loopback. Either queues a pass (`launch`, chained
from an earlier pass's ticket) before the one ahead of it is awaited (`wait`), so the device never idles between
passes. A boundary message (the next window's coin, or the close) is taken at the first root that reaches the host
after it: that root is sent as its window's last if its pass started (the root before it reached the host) before the
message arrived, and dropped otherwise, and the pass queued behind it is dropped too; the next window's first pass
chains from the last root sent. Each session runs under a mode of the prover's plan, which the verifier never sees:

~~~text
honest          every block held and read
skip:F          a fraction F of the blocks dropped from the store before the session (chosen afresh each session,
                zeroed where they were): each pass's receipts for them are of what is there, so wrong, and an opening
                of one carries the true block, as rebuilding it from its payload after the roots would give
kv:P            honest, beside a serving node's KV writes (`ServingTrace`): before every --kv-every'th pass but the
                session's first, a memory-map/v1 step writes P pages and releases the blocks they overwrite or free,
                and a step re-keys each block before a pass of the window the session's lifetime cap L makes it stale
                in; every pass reads every live block, registered and appended
kvdrop:P        kv:P, but the appended pages are not held: each pass's receipts for them are of zeros, and an
                opening of one carries the true page, as skip's do
serve           honest, beside the backend's own serving (`backend.serving`: a node's `Store`, whose engine steps
                write, free and re-key its KV, `verity_vllm.store.remote`): before each pass the backend's engine runs
                its next step, and the pass reads every live block at it
servedrop:K     serve, but the first K blocks appended (1 by default) are not held, as kvdrop's
~~~

With a serving trace, the appends a pass's step made go out after the root before it and before the pass is computed,
each its own message (its tree's root and size, `Session.append`) answered with its coin, under which the backend
encodes its blocks (`audit.Store.encode`); then the root's message carries the blocks released since the last root
sent, as runs, with those among them the trace withdrew (`withdrawn`, freed before their encode landed, so never
written: `Session.release(released, withdrawn)`), and the pass's plan as a change from the last one sent
(`Session.receive(..., change=...)`).  The hello says whether the backend defers an appended block's residency to its
encode (`deferred`, a backend's attribute, False if it has none), which the verifier refuses under a lifetime of one
window.  A pass's seed binds every append sent before its root (`audit.chained`), so a backend that appends
takes the chain over its appends when the pass is launched (`audit.Store.head`).  A pass whose appends went out is
sent, never dropped at a boundary, since a coin no root follows rejects the session; one dropped before its appends
went out sends nothing, and the next pass sends its appends and changes too.

Every root's message names the step its pass covers (`Session.receive(..., step=...)`, PROTOCOL.md §8 "The pass and
its step"): the backend's `step_of(ticket)` if it has one, which a serving backend takes from its engine, and otherwise
the count of the session's passes launched before it. A serving trace's step before pass n is recorded as step n − 1,
so pass n reads the store that step left.

    python prover.py --connect HOST:PORT --out OUT --plan honest,skip:3e-5 [device options, `device.py`]
    python prover.py --connect HOST:PORT --out OUT --plan kv:4,kvdrop:4 --cpu 256 --codec p512/k2
                                                        the driver's reference as the backend, no GPU
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import queue
import random
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import audit, coins, memory_map, receipts as rb  # noqa: E402
from verity.core.protocols.remote_bandwidth import codec as rcodec  # noqa: E402

import wire  # noqa: E402

PAYLOAD_LABEL = hashlib.sha512(b"verity/remote-session/payload-salts/v0").digest()
OPENINGS_PART = 128
"""Openings a frame: a window's go out in parts, so a root waits for one part's send, not the window's."""
DROPPED_LISTED = 4096
"""A session's result lists its dropped blocks up to this many, and counts them always."""
BOUNDARIES = ("coin", "close", "end")
MODES = ("honest", "skip", "kv", "kvdrop", "serve", "servedrop")


@dataclass
class _Pass:
    t: int
    i: int
    ticket: object
    start: int | None
    step: int = 0
    discard: bool = False
    plan: rb.Plan | None = None
    upto: tuple[int, int, int] = (0, 0, 0)
    flushed: bool = False


def hits(dropped: np.ndarray, positions) -> np.ndarray:
    """Which of `positions` are in `dropped` (sorted)."""
    pos = np.asarray(positions, dtype=np.int64)
    if not len(dropped) or not len(pos):
        return np.zeros(len(pos), dtype=bool)
    k = np.searchsorted(dropped, pos)
    return dropped[np.minimum(k, len(dropped) - 1)] == pos


def parse_plan(plan: str) -> list[tuple[str, float]]:
    out = []
    for part in plan.split(","):
        mode, _, f = part.partition(":")
        if mode not in MODES or (mode in ("kv", "kvdrop") and (not f.isdigit() or int(f) < 1)) or (
                mode == "serve" and f) or (mode == "servedrop" and f and (not f.isdigit() or int(f) < 1)):
            raise ValueError(f"a session's mode is honest, skip:F, kv:P, kvdrop:P (P pages a step), serve or "
                             f"servedrop:K (K blocks withheld), not {part!r}")
        out.append((mode, float(f) if f else 1.0 if mode == "servedrop" else 0.0))
    return out


def payload(rng: random.Random) -> bytes:
    """A random payload block whose last word is below any codec's p."""
    return rng.randbytes(rcodec.BLOCK_BYTES - 16) + bytes(16)


def runs_of(blocks) -> tuple[tuple[int, int], ...]:
    """Sorted distinct store indices as a plan's canonical runs (first, count)."""
    out: list[list[int]] = []
    for j in blocks:
        if out and out[-1][0] + out[-1][1] == j:
            out[-1][1] += 1
        else:
            out.append([j, 1])
    return tuple((f, c) for f, c in out)


def plan_change(before: rb.Plan, after: rb.Plan) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """`after` as `before.changed(added, removed, ...)` takes it: (added, removed) runs."""
    a, b = set(before), set(after)
    return runs_of(sorted(b - a)), runs_of(sorted(a - b))


class ServingTrace:
    """A serving node's KV writes beside its registered store, as memory-map/v1 steps (`memory_map`).  Step s writes
    `pages` blocks of fresh payloads round-robin over a pool of slots: the store's last `pool` registered blocks (a KV
    pool's pages, registered as filler) and `pool // 4` slots of memory outside the store.  Every third step first frees
    the `pages // 2` slots after its last write, so a later write lands in a slot holding nothing.  A block written in
    window a is stale from window a + `lifetime` (`audit.Params.lifetime`), and the step before a pass of a window
    that would hold a stale block re-keys it: writes its payload again in its slot, at the store's next fresh index.
    `memory_map.Replay` turns each step into the audit's append, which the backend's store commits and, once the
    verifier's coin for it arrives, encodes under the pages' bound keys, and its release; `plan` is every live block in
    store order."""

    def __init__(self, backend, registration: audit.Registration, pages: int, pool: int, rng: random.Random,
                 lifetime: int = audit.LIFETIME) -> None:
        N = registration.blocks
        pool = max(1, min(pool, N))
        self.slots = list(range(N - pool, N)) + list(range(N, N + max(1, pool // 4)))
        if pages + pages // 2 > len(self.slots):
            raise ValueError(f"a step of {pages} pages frees and writes more than the pool's {len(self.slots)} slots")
        self.backend, self.registration, self.pages, self.rng = backend, registration, pages, rng
        self.lifetime = lifetime
        self.replay = memory_map.Replay(memory_map.Header(registration.digest, N), registration)
        self.steps, self.cursor, self.rekeyed = 0, 0, 0
        self.live = set(range(N))
        self.held: dict[int, tuple[int, bytes]] = {}
        self.appends: list[audit.Append] = []
        self.released: list[int] = []
        self.withdrawn: list[int] = []
        self.appended: list[int] = []

    def stale(self, window: int) -> list[int]:
        """The slots holding a block written `lifetime` windows or more before `window`, by its index."""
        old = [x for x, (a, _) in self.held.items() if a <= window - self.lifetime]
        return sorted(old, key=self.replay.holds)

    def step(self, s: int, window: int, pages: bool = True) -> memory_map.Change:
        """The trace's next step in `window`, recorded as memory-map step `s`: its `pages` writes if `pages`, and the
        stale blocks' re-keys."""
        n, k = len(self.slots), (self.pages // 2 if pages and self.steps % 3 == 2 else 0)
        freed = [x for x in (self.slots[(self.cursor + u) % n] for u in range(k)) if self.replay.holds(x) is not None]
        slots = [self.slots[(self.cursor + k + u) % n] for u in range(self.pages)] if pages else []
        if pages:
            self.cursor = (self.cursor + k + self.pages) % n
            self.steps += 1
        payloads = [payload(self.rng) for _ in slots]
        rekey = [x for x in self.stale(window) if x not in slots and x not in freed]
        slots, payloads = slots + rekey, payloads + [self.held[x][1] for x in rekey]
        self.rekeyed += len(rekey)
        first = self.replay.extent
        written = tuple(memory_map.Written(x, first + u, rcodec.digest(m)) for u, (x, m) in enumerate(zip(slots, payloads)))
        change = self.replay.apply(memory_map.Step(s, tuple(freed), written))
        for x in freed:
            self.held.pop(x, None)
        self.held.update((x, (window, m)) for x, m in zip(slots, payloads))
        a = self.backend.append(payloads, [self.rng.randbytes(rb.SALT_BYTES) for _ in payloads])
        if a.first != change.first or a.digests != change.digests:
            raise AssertionError("the store's append is not the memory map's")
        self.appends.append(a)
        self.appended.extend(range(a.first, a.first + a.count))
        gone = [j for f, c in change.released for j in range(f, f + c)]
        self.released.extend(gone)
        self.withdrawn.extend(j for f, c in change.withdrawn for j in range(f, f + c))
        self.live.difference_update(gone)
        self.live.update(range(a.first, a.first + a.count))
        return change

    def plan(self) -> rb.Plan:
        return rb.Plan(runs_of(sorted(self.live)), self.replay.extent)


class CpuBackend:
    """The driver's reference as a backend: a store of `blocks` random payload blocks under the registration's codec,
    every pass `receipts.receipter` over its plan's blocks (every registered block unless the pass names a plan), its
    tree `receipts.tree`, salts from `entropy`; blocks appended after the codec seed (`append`, committed, then
    `encode_append` under the append's coin) are the session's."""

    def __init__(self, blocks: int, rng: random.Random, read_bytes_per_s: float = 1e12) -> None:
        self.blocks, self.rng, self.read_bytes_per_s = blocks, rng, read_bytes_per_s
        self.payloads = [payload(rng) for _ in range(blocks)]
        self.salts = [rng.randbytes(rb.SALT_BYTES) for _ in range(blocks)]
        self.dropped: set[int] = set()
        self.gates: dict = {}

    def payload_root(self, node: bytes) -> bytes:
        return audit.payload_tree(node, self.payloads, self.salts).commitment.root

    def encode(self, registration: audit.Registration) -> dict:
        self.store = audit.Store(registration, self.payloads, self.salts)
        self.plan = rb.Plan.whole(self.blocks)
        return {"encode_bytes_per_s": 1.0}

    def begin_session(self) -> None:
        """A session's appends start again from N."""
        del self.store.blocks[self.blocks:]
        self.store.appends.clear()
        self.store.pending.clear()

    def append(self, payloads, salts) -> audit.Append:
        return self.store.append(payloads, salts)

    def encode_append(self, first: int, coin: bytes) -> None:
        self.store.encode(first, coin)

    def calibrate(self) -> dict:
        return {"pass_ns_max": 0, "gap_ns_max": 0, "read_bytes_per_s": self.read_bytes_per_s}

    def arm(self, dropped) -> None:
        self.dropped = {int(j) for j in dropped}

    def begin_window(self, challenge: bytes, previous: bytes | None) -> None:
        self.challenge = challenge
        if previous is not None:
            self.previous = previous

    def launch(self, i: int, after: dict | None, plan: rb.Plan | None = None) -> dict:
        """Pass i of the window over `plan` (None: every registered block), chained from `after`'s root (None: the
        window's previous root) and the store's appends so far, every one of which goes out before this pass's root
        (`audit.chained`), run when awaited."""
        return {"challenge": self.challenge, "i": i, "after": after, "previous": self.previous, "root": None,
                "plan": plan, "head": self.store.head}

    def wait(self, ticket: dict):
        after = ticket["after"]
        prev = audit.chained(ticket["previous"] if after is None else after["root"], ticket["head"])
        root, handle = self.run_pass(rb.seed(ticket["challenge"], ticket["i"], prev), ticket["plan"])
        ticket["root"] = root
        return root, handle

    def run_pass(self, seed: bytes, plan: rb.Plan | None = None):
        plan = plan or self.plan
        r = rb.receipter(self.store.registration.receipt, seed, self.store.registration.rows)
        ys = tuple(r(j, bytes(rcodec.BLOCK_BYTES) if j in self.dropped else self.store.blocks[j]) for j in plan)
        salts = tuple(self.rng.randbytes(rb.SALT_BYTES) for _ in range(rb.leaves(plan.length)))
        tree = rb.tree(seed, plan, ys, salts)
        return tree.commitment.root, (ys, salts, tree, plan)

    def open_window(self, items) -> list[list[tuple]]:
        return [self.open(handle, positions) for handle, positions in items]

    def open(self, handle, positions: list[int]) -> list[tuple]:
        """Each position's opening fields: the block, its receipt leaf, salt and path, and its payload's salt, path
        and digest d_j (empty for a registered block)."""
        ys, salts, tree, plan = handle
        out = []
        for p in positions:
            c, j = p // rb.PER_LEAF, plan.block(p)
            leaf = b"".join(ys[rb.PER_LEAF * c:rb.PER_LEAF * (c + 1)])
            out.append((self.store.blocks[j], leaf, salts[c], tree.open(c).path, *self.store.committed(j)))
        return out


class ProverSide:
    def __init__(self, conn: wire.Conn, backend, node: bytes, plan: list[tuple[str, float]], *, seed: int = 0,
                 defer_openings: bool = True, warmup_ns: int = 0, kv_every: int = 1, kv_pool: int = 64,
                 log=print) -> None:
        self.conn, self.backend, self.node, self.plan, self.log = conn, backend, node, plan, log
        self.defer_openings, self.warmup_ns = defer_openings, warmup_ns
        self.kv_every, self.kv_pool = kv_every, kv_pool
        self.rng = random.Random(seed)
        self.inbox: queue.Queue = queue.Queue()
        self.jobs: queue.Queue = queue.Queue()
        self.kept: dict[tuple[int, int], object] = {}
        self.plans: dict[tuple[int, int], rb.Plan] = {}
        self.dropped = np.empty(0, dtype=np.int64)
        self.lock = threading.Lock()
        self.open_enabled = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._open, daemon=True).start()

    def _read(self) -> None:
        try:
            while True:
                head, body, stamp = self.conn.recv()
                if head["type"] == "ping":
                    self.conn.send({"type": "pong", "k": head["k"]})
                elif head["type"] == "draws":
                    self.jobs.put((head["t"], [tuple(d) for d in head["draws"]]))
                else:
                    self.inbox.put((head, body, stamp))
        except (wire.Closed, OSError) as e:
            self.inbox.put(({"type": "lost", "why": str(e)}, b"", 0))

    def _open(self) -> None:
        while True:
            t, draws = self.jobs.get()
            # Opening work rereads the store and reconstructs receipt subtrees.  Keep it off the timed pass phase:
            # the verifier may check after all roots are final, and this also makes the measured pass span contain
            # passes alone rather than a prover-local choice of when to prepare the already-drawn openings.
            if self.defer_openings:
                self.open_enabled.wait()
            c0 = time.perf_counter_ns()
            by_pass = defaultdict(list)
            for k, (i, p) in enumerate(draws):
                by_pass[i].append((k, p))
            out: list = [None] * len(draws)
            passes = sorted(by_pass.items())
            with self.lock:
                handles = [self.kept[(t, i)] for i, _ in passes]
            c1 = time.perf_counter_ns()
            fields = self.backend.open_window([(h, [p for _, p in items]) for h, (_, items) in zip(handles, passes)])
            c2 = time.perf_counter_ns()
            for j, ((i, items), got) in enumerate(zip(passes, fields)):
                for (k, p), f in zip(items, got):
                    out[k] = audit.Opening(i, p, f[0], f[1], f[2], tuple(f[3]), f[4], tuple(f[5]), b"",
                                           f[6] if len(f) > 6 else b"")
                fields[j] = None
            del fields
            parts = [out[k:k + OPENINGS_PART] for k in range(0, len(out), OPENINGS_PART)] or [[]]
            n = len(out)
            del out
            c3, sent = time.perf_counter_ns(), 0
            for k in range(len(parts)):
                body = wire.encode_openings(parts[k])
                self.conn.send({"type": "openings", "t": t, "part": k, "parts": len(parts), "n": len(parts[k])}, body)
                parts[k] = None
                sent += len(body)
                del body
            with self.lock:
                for key in [key for key in self.kept if key[0] == t]:
                    del self.kept[key]
            self.opened.append({"t": t, "n": n, "bytes": sent, "ns": time.perf_counter_ns() - c0,
                                "backend_ns": c2 - c1, "send_ns": time.perf_counter_ns() - c3,
                                "on_dropped": int(hits(self.dropped, [self.block(t, i, p) for i, p in draws]).sum())})
            self.jobs.task_done()

    def block(self, t: int, i: int, p: int) -> int:
        """The block at position p of pass i of window t: p itself unless the pass's plan was sent."""
        plan = self.plans.get((t, i))
        return p if plan is None else plan.block(p)

    def expect(self, kind: str):
        while True:
            head, body, stamp = self.inbox.get()
            if head["type"] == kind:
                return head, body, stamp
            if head["type"] in ("lost", "bye"):
                raise wire.Closed(head.get("why", "bye"))

    def register(self, codec: rcodec.Codec = rcodec.DEFAULT, receipt: str = rb.DEFAULT, rows: int | None = None) -> dict:
        root = self.backend.payload_root(self.node)
        layout = self.backend.layout() if hasattr(self.backend, "layout") else None
        self.conn.send({"type": "hello", "node": self.node.hex(), "blocks": self.backend.blocks,
                        "payload_root": root.hex(), "codec": codec.name, "receipt": receipt, "rows": rows,
                        "layout": layout, "deferred": bool(getattr(self.backend, "deferred", False))})
        head, _, _ = self.expect("registration")
        self.registration = wire.registration_of(head["registration"])
        if self.registration.payload_root != root or self.registration.blocks != self.backend.blocks:
            raise ValueError("the registration is not of this store")
        out = self.backend.encode(self.registration)
        out.update(self.backend.calibrate())
        self.conn.send({"type": "ready", "calibration": out})
        return out

    def session(self, head: dict, mode: str, fraction: float) -> dict:
        k, W = head["k"], head["windows"]
        commitment = wire.commitment_of(head["commitment"])
        if commitment.registration != self.registration.digest or commitment.count != 2 * W:
            raise ValueError("the session's coins are not for this registration")
        N = self.backend.blocks
        getattr(self.backend, "begin_session", lambda: None)()
        trace = None
        if mode in ("kv", "kvdrop"):
            if not hasattr(self.backend, "append"):
                raise ValueError(f"a {mode} session needs a backend that appends blocks after the codec seed (--cpu)")
            trace = ServingTrace(self.backend, self.registration, int(fraction), self.kv_pool,
                                 random.Random(self.rng.getrandbits(64)), head.get("lifetime", audit.LIFETIME))
        elif mode in ("serve", "servedrop"):
            if not hasattr(self.backend, "serving"):
                raise ValueError(f"a {mode} session needs a backend that serves (a Store's: verity_vllm.store.remote)")
            trace = self.backend.serving(self.registration, head.get("lifetime", audit.LIFETIME))
        if mode == "skip":
            pick = np.random.default_rng(self.rng.getrandbits(64))
            dropped = np.sort(pick.choice(N, size=round(fraction * N), replace=False)).astype(np.int64)
        else:
            dropped = np.empty(0, dtype=np.int64)
        self.backend.arm(dropped)
        self.plans = {}
        # Untimed passes bring the GPU to its boost clocks and warm its caches before the window clock starts: the
        # window's released_ns is the verifier's when it sends the first coin, after this `armed`, so the first
        # window does not pay the first passes' ramp (`device.DeviceStore.warmup`).
        warmup = getattr(self.backend, "warmup", None)
        self.warmup = warmup(self.warmup_ns) if warmup is not None and self.warmup_ns else None
        self.dropped, self.opened = dropped, []
        if self.defer_openings:
            self.open_enabled.clear()
        self.conn.send({"type": "armed", "k": k})
        head, _, _ = self.expect("coin")
        t = 0

        def take(coin) -> bytes:
            reason = coins.coin_reason(commitment, coin, t)
            if reason is not None:
                raise ValueError(reason)
            return coin.coin

        backend, clock = self.backend, self.conn.clock
        step_of = getattr(backend, "step_of", None)
        launched, sent_plan, sent_released, sent_withdrawn, sent_appends = 0, rb.Plan.whole(N), 0, 0, 0

        def launch(t: int, i: int, after, start: int | None) -> _Pass:
            """Pass i of window t; with a serving trace, after the trace's step if one falls due (the backend's next
            engine step, serving), over every live block."""
            nonlocal launched
            n, launched = launched, launched + 1
            if trace is None:
                ticket = backend.launch(i, after)
                return _Pass(t, i, ticket, start, step=step_of(ticket) if step_of is not None else n)
            if mode in ("serve", "servedrop"):
                trace.advance(t)
                if mode == "servedrop":
                    backend.arm(trace.appended[:int(fraction)])
            else:
                due = n > 0 and n % self.kv_every == 0
                if due or (n > 0 and trace.stale(t)):
                    trace.step(n - 1, t, due)
                    if mode == "kvdrop":
                        backend.arm(trace.appended)
            plan = trace.plan()
            ticket = backend.launch(i, after, plan)
            return _Pass(t, i, ticket, start, step=step_of(ticket) if step_of is not None else n, plan=plan,
                         upto=(len(trace.appends), len(trace.released), len(getattr(trace, "withdrawn", ()))))

        backend.begin_window(take(wire.coin_of(head["coin"])), self.registration.digest)
        inflight = deque([launch(0, 0, None, clock())])
        pending: deque = deque()
        passes, last, nxt, stopping, ended, discarded = [], None, 1, False, None, 0

        def send(x: _Pass, root: bytes, handle, seen: int) -> None:
            nonlocal sent_plan, sent_released, sent_withdrawn
            with self.lock:
                self.kept[(x.t, x.i)] = handle
            msg = {"type": "root", "t": x.t, "i": x.i, "root": root.hex(), "step": x.step}
            if x.plan is not None:
                if x.upto[1] > sent_released:
                    msg["released"] = [list(r) for r in runs_of(sorted(trace.released[sent_released:x.upto[1]]))]
                if x.upto[2] > sent_withdrawn:
                    msg["withdrawn"] = [list(r) for r in runs_of(sorted(trace.withdrawn[sent_withdrawn:x.upto[2]]))]
                msg["change"] = [[list(r) for r in side] for side in plan_change(sent_plan, x.plan)]
                self.plans[(x.t, x.i)] = x.plan
                sent_plan, sent_released, sent_withdrawn = x.plan, x.upto[1], x.upto[2]
            self.conn.send(msg)
            passes.append((x.t, x.i, x.start, seen, clock()))

        def flush(x: _Pass) -> dict | None:
            """x's appends not yet sent, each sent and answered with its coin, under which the backend encodes its
            blocks: None, or the session's `end` if it came first."""
            nonlocal sent_appends
            for a in trace.appends[sent_appends:x.upto[0]]:
                self.conn.send({"type": "append", "root": a.root.hex(), "count": a.count})
                sent_appends, x.flushed = sent_appends + 1, True
                reply = self._appended(pending)
                if reply["type"] == "end":
                    return reply
                if reply["first"] != a.first:
                    raise ValueError(f"the verifier placed the append from block {a.first} at {reply['first']}")
                backend.encode_append(a.first, bytes.fromhex(reply["coin"]))
            return None

        while inflight:
            if not stopping and len(inflight) < 2:
                inflight.append(launch(t, nxt, inflight[-1].ticket, None))
                nxt += 1
            x = inflight.popleft()
            unsent = x.plan is not None and x.upto[0] > sent_appends
            if unsent and not x.discard:
                self._drain(pending)
                if not pending or (pending[0][0]["type"] != "end" and pending[0][2] >= x.start):
                    if (gone := flush(x)) is not None:
                        ended, stopping = gone, True
                        discarded += 1 + len(inflight)
                        break
                    unsent = False
            if unsent:
                if inflight:
                    inflight[0].start = clock()
            else:
                root, handle = backend.wait(x.ticket)
                seen = clock()
                if inflight:
                    inflight[0].start = seen
            self._drain(pending)
            if x.discard:
                discarded += 1
                continue
            if not pending:
                send(x, root, handle, seen)
                last = x
                continue
            msg, _, stamp = pending.popleft()
            for p in inflight:
                p.discard = True
            if msg["type"] == "end":
                ended, stopping = msg, True
                discarded += 1
                continue
            if stamp < x.start and not x.flushed:
                discarded += 1
            else:
                send(x, root, handle, seen)
                last = x
            if msg["type"] == "close":
                stopping = True
                continue
            t += 1
            backend.begin_window(take(wire.coin_of(msg["coin"])), None)
            inflight.append(launch(t, 0, last.ticket if last else None, None if inflight else clock()))
            nxt = 1
        if mode in ("kvdrop", "servedrop"):
            held = trace.appended if mode == "kvdrop" else trace.appended[:int(fraction)]
            dropped = self.dropped = np.array(sorted(held), dtype=np.int64)
        self.conn.send({"type": "closed", "k": k})
        self.open_enabled.set()
        if ended is None:
            ended = next((m for m, _, _ in pending if m["type"] == "end"), None) or self.expect("end")[0]
        self.jobs.join()
        device = getattr(backend, "session_stats", lambda: {})()
        backend.arm(set())
        with self.lock:
            self.kept.clear()
        self.conn.send({"type": "ended", "k": k})
        per_window = defaultdict(int)
        for tt, *_ in passes:
            per_window[tt] += 1
        failure = ended["failure"]
        at = failure.get("at") if failure else None
        out = {"k": k, "mode": mode, "fraction": fraction,
               "dropped": dropped.tolist() if len(dropped) <= DROPPED_LISTED else None, "dropped_count": len(dropped),
               "failure_on_dropped": bool(hits(dropped, [self.block(failure["t"], *at)])[0]) if at else None,
               "accepted": ended["accepted"], "failure": failure, "passes": [per_window[tt] for tt in range(W)],
               "pass_host_ns": [seen - start for _, _, start, seen, _ in passes],
               "gap_host_ns": [sent - seen for _, _, _, seen, sent in passes], "discarded": discarded,
               "opened": self.opened, "warmup": self.warmup, "device": device}
        if trace is not None:
            sent = [self.plans[tt, i] for tt, i, *_ in passes]
            out["kv"] = {"pages": getattr(trace, "pages", None), "every": self.kv_every,
                         "pool_slots": len(getattr(trace, "slots", ())),
                         "steps": trace.steps, "rekeyed": trace.rekeyed, "lifetime": trace.lifetime,
                         "appended": len(trace.appended), "released": len(trace.released),
                         "sent_appended": sum(a.count for a in trace.appends[:sent_appends]),
                         "sent_released": sent_released,
                         "pass_blocks": [p.length for p in sent],
                         "pass_appended": [sum(max(0, f + c - max(f, N)) for f, c in p.runs) for p in sent]}
        return out

    def _appended(self, pending: deque) -> dict:
        """The verifier's reply to the append just sent, `appended`, or the session's `end` if it came first; the
        boundary messages that arrive before it go to `pending`, in order."""
        while True:
            item = self.inbox.get()
            kind = item[0]["type"]
            if kind in ("lost", "bye"):
                raise wire.Closed(item[0].get("why", "bye"))
            if kind == "appended":
                return item[0]
            if kind in BOUNDARIES:
                pending.append(item)
                if kind == "end":
                    return item[0]

    def _drain(self, pending: deque) -> None:
        """Move the boundary messages that have arrived into `pending`, in order."""
        while True:
            try:
                item = self.inbox.get_nowait()
            except queue.Empty:
                return
            kind = item[0]["type"]
            if kind in ("lost", "bye"):
                raise wire.Closed(item[0].get("why", "bye"))
            if kind in BOUNDARIES:
                pending.append(item)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connect", default="127.0.0.1:7781")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan", default="honest")
    ap.add_argument("--node", default="vy-nebius/rtx-pro-6000/0")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--overlap-openings", action="store_true",
                    help="benchmark the old behavior: prepare each window's openings while later timed passes run")
    ap.add_argument("--warmup-ms", type=float, default=2500.0,
                    help="untimed passes before each session's first window, to reach boost clocks before the window "
                         "clock starts (0: none; the CPU reference backend ignores it)")
    ap.add_argument("--cpu", type=int, default=0, metavar="BLOCKS",
                    help="the driver's reference (`CpuBackend`) over a store of BLOCKS blocks, no GPU and none of "
                         "device.py's options")
    ap.add_argument("--codec", default=rcodec.DEFAULT.name, help="the registration's codec, with --cpu")
    ap.add_argument("--kv-every", type=int, default=1, help="a kv session's passes per memory-map step")
    ap.add_argument("--kv-pool", type=int, default=64, help="a kv session's registered slots the KV pool writes over")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--cpu", type=int, default=0)
    if not pre.parse_known_args()[0].cpu:
        import device

        device.add_arguments(ap)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.setswitchinterval(2e-4)
    if a.cpu:
        backend = CpuBackend(a.cpu, random.Random(a.seed))
    else:
        backend = device.from_arguments(a, log=lambda m: print(m, flush=True))
    host, port = a.connect.rsplit(":", 1)
    conn = wire.connect(host, int(port))
    p = ProverSide(conn, backend, a.node.encode(), parse_plan(a.plan), seed=a.seed,
                   defer_openings=not a.overlap_openings, warmup_ns=round(a.warmup_ms * 1e6), kv_every=a.kv_every,
                   kv_pool=a.kv_pool, log=lambda m: print(m, flush=True))
    cal = p.register(codec=wire.codec_named(a.codec)) if a.cpu else p.register()
    print(f"prover: registered, calibration {json.dumps(cal)}", flush=True)
    stalls = getattr(backend, "calibration_trace", None)
    (a.out / "prover.json").write_text(json.dumps({"calibration": cal, "gates": backend.gates,
                                                   "calibration_trace": stalls}, indent=1))
    results = []
    for mode, fraction in p.plan:
        try:
            head, _, _ = p.expect("session")
        except wire.Closed:
            break
        # A collection (or a window's openings freed at once) holds the GIL for tens of ms, long enough to make a
        # boundary pass's root late: no collector during a session, and the opener frees what it sent part by part.
        gc.collect()
        gc.freeze()
        gc.disable()
        try:
            r = p.session(head, mode, fraction)
        finally:
            gc.enable()
            gc.unfreeze()
        results.append(r)
        kv = r.get("kv")
        print(f"prover: session {r['k']} ({mode}:{fraction}): accepted={r['accepted']} passes={r['passes']}"
              + (f" appended={kv['appended']} released={kv['released']} failure={r['failure']}"
                 f" on_dropped={r['failure_on_dropped']}" if kv else ""), flush=True)
        (a.out / "prover.json").write_text(json.dumps(
            {"schema": "verity/remote-session/prover/v0", "registration": wire.registration_json(p.registration),
             "calibration": cal, "gates": backend.gates, "calibration_trace": stalls, "sessions": results,
             "wire": conn.totals()}, indent=1))
    # Tearing down a whole-node store (every pass handle in host memory, every GPU's mapped segments) kept the process
    # alive 12 minutes past its last output, to the end of its GPU lease (r20261015-171223-cc99).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
