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
kvoverwrite:P   kv:P with serving ahead of the passes (--ahead), but its writes skip the free fence: a write lands in
                a slot whose block a pass still has to read, that pass's receipt of it is of the new bytes, and an
                opening of it carries the true block
~~~

With serving ahead of the passes (`Ahead`: --ahead K for a kv session, or a serving backend's own), serving runs on
its own thread, at most K steps past the pass reading, and the session takes each pass's step from it: the pass at s,
once serving has handed step s − 1's record, to which a kv session adds the re-keys the pass's window needs and the
store's indices (`ServingTrace.take`).  Each block is read in its slot (`CpuBackend.write`, `place`), so a write that
lands in a slot while a pass still has to read the block it held changes that pass's receipts: serving writes into a
slot whose block was released at step r (freed at r, or overwritten by the write itself at r) only once the pass at
r has read (`Ahead.reuse`, the free fence).  One pass is then in flight at a time, not two: the next pass's step waits
on serving, whose write may wait on this pass's read.  Without serving ahead the trace's step is the session's own,
before each pass, and a block lands in its slot when it is encoded.

With a serving trace, the appends a pass's step made go out after the root before it and before the pass is computed,
each its own message (its tree's root and size, `Session.append`) answered with its coin, under which the backend
encodes its blocks (`audit.Store.encode`); then the root's message carries the blocks released since the last root
sent, as runs (`Session.release`), and the pass's plan as a change from the last one sent (`Session.receive(...,
change=...)`).  A pass's seed binds every append sent before its root (`audit.chained`), so a backend that appends
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
    python prover.py ... --plan kv:4 --cpu 256 --ahead 4 --kv-headroom 5
                                                        serving up to 4 steps past the passes, each slot reused 5 steps
                                                        after its free
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
from collections.abc import Callable
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
MODES = ("honest", "skip", "kv", "kvdrop", "kvoverwrite", "serve", "servedrop")
TRACED = ("kv", "kvdrop", "kvoverwrite")


@dataclass
class _Pass:
    t: int
    i: int
    ticket: object
    start: int | None
    step: int = 0
    discard: bool = False
    plan: rb.Plan | None = None
    upto: tuple[int, int] = (0, 0)
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
        if mode not in MODES or (mode in TRACED and (not f.isdigit() or int(f) < 1)) or (
                mode == "serve" and f) or (mode == "servedrop" and f and (not f.isdigit() or int(f) < 1)):
            raise ValueError(f"a session's mode is honest, skip:F, kv:P, kvdrop:P, kvoverwrite:P (P pages a step), "
                             f"serve or servedrop:K (K blocks withheld), not {part!r}")
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


class Ahead:
    """Serving's steps beside the session's passes, serving at most `k` steps past the pass reading: the hook that takes
    the audit off serving's critical path.  R is the last step whose pass has read (-1 before the first), and the
    session's next pass is at step R + 1, one pass a serving step.

    serving's thread (the run's)
      begin(s)     before step s: waits until s ≤ R + 1 + k; False once the session has ended (`end`), and serving's
                   steps from s on are unaudited
      reuse(r)     before a write into memory whose block was released at step r (freed there, or overwritten by this
                   very write, r = s): waits until R ≥ r, the free fence, so every pass that holds the block has read
                   it in place first
      hand(s, x)   once step s has run, its record x, which the session takes for the pass at s + 1
      finish()     serving has no more steps: the passes after its last cover idle steps
      fail(e)      serving raised e: the session's next `take` raises
    the session's thread
      take(s)      the record of step s − 1 (None for one serving never ran), once serving has handed it or finished
      passed(s)    the pass at s has read, or never will (dropped at a boundary before it read)
      end()        the session is over: serving runs free

    With `fill`, `take` also waits until serving waits on the passes (in `begin` or `reuse`) or has finished, so each
    pass reads with serving as far ahead as the bound and the fence let it, whatever the threads' timing: one schedule
    a run, the furthest, for tests.  `leads` is, for each pass, how many steps past its own serving had begun when it
    finished reading (at most k); `waits` counts serving's waits on the bound and on the fence."""

    def __init__(self, k: int, fill: bool = False) -> None:
        if type(k) is not int or k < 0:
            raise ValueError(f"serving runs k ≥ 0 steps past the passes, not {k!r}")
        self.k, self.fill = k, fill
        self.read, self.begun, self.last = -1, -1, -1
        self.handed: dict[int, object] = {}
        self.waiting: Callable[[], bool] | None = None
        self.over = self.done = False
        self.error: BaseException | None = None
        self.leads: list[int] = []
        self.waits = {"bound": 0, "fence": 0}
        self.cv = threading.Condition()

    def _hold(self, on: str, ready: Callable[[], bool]) -> None:
        """Under the lock: wait until `ready` or the session's end, counting the wait on `on` as it starts."""
        if self.over or ready():
            return
        self.waits[on] += 1
        self.waiting = ready
        self.cv.notify_all()
        self.cv.wait_for(lambda: self.over or ready())
        self.waiting = None

    def begin(self, s: int) -> bool:
        with self.cv:
            self._hold("bound", lambda: s <= self.read + 1 + self.k)
            if self.over:
                return False
            self.begun = s
            return True

    def reuse(self, r: int) -> None:
        with self.cv:
            self._hold("fence", lambda: r <= self.read)

    def hand(self, s: int, record) -> None:
        with self.cv:
            if not self.over:
                self.handed[s], self.last = record, s
                self.cv.notify_all()

    def finish(self) -> None:
        with self.cv:
            self.done = True
            self.cv.notify_all()

    def fail(self, error: BaseException) -> None:
        with self.cv:
            self.error, self.done = error, True
            self.cv.notify_all()

    def take(self, s: int):
        with self.cv:
            self.cv.wait_for(lambda: self.done or (self.last >= s - 1 and (
                not self.fill or (self.waiting is not None and not self.waiting()))))
            if self.error is not None:
                raise RuntimeError(f"serving raised before step {s}'s pass") from self.error
            return self.handed.pop(s - 1, None)

    def passed(self, s: int) -> None:
        with self.cv:
            if s > self.read:
                self.leads.append(self.begun - s)
                self.read = s
                self.cv.notify_all()

    def end(self) -> None:
        with self.cv:
            self.over = True
            self.cv.notify_all()

    def stats(self) -> dict:
        return {"k": self.k, "fill": self.fill, "steps": self.begun + 1, "leads": self.leads,
                "bound_waits": self.waits["bound"], "fence_waits": self.waits["fence"]}


class ServingTrace:
    """A serving node's KV writes beside its registered store, as memory-map/v1 steps (`memory_map`).  Step s writes
    `pages` blocks of fresh payloads round-robin over a pool of slots: the store's last `pool` registered blocks (a KV
    pool's pages, registered as filler) and `pool // 4` slots of memory outside the store.  Every third step first frees
    the `pages // 2` slots after its last write, so a later write lands in a slot holding nothing; or, with `headroom`
    h, every step frees the slots the step h steps on writes, so each write lands in a slot freed h steps before (an
    allocator that keeps h steps of frees before it reuses one).  A block written in window a is stale from window
    a + `lifetime` (`audit.Params.lifetime`), and the step before a pass of a window that would hold a stale block
    re-keys it: writes its payload again in its slot, at the store's next fresh index.  `memory_map.Replay` turns each
    step into the audit's append, which the backend's store commits and, once the verifier's coin for it arrives,
    encodes under the pages' bound keys, and its release; `plan` is every live block in store order.

    A step is serving's half (`serve`: its frees and writes, by slot) and then the session's (`record`: the re-keys,
    the indices, the append), which `step` runs together before the session's pass.  With serving ahead of the passes
    (`ahead`, an `Ahead`), `run` is serving on its own thread and `take` the session's half at each pass."""

    def __init__(self, backend, registration: audit.Registration, pages: int, pool: int, rng: random.Random,
                 lifetime: int = audit.LIFETIME, headroom: int = 0) -> None:
        N = registration.blocks
        pool = max(1, min(pool, N))
        self.slots = list(range(N - pool, N)) + list(range(N, N + max(1, pool // 4)))
        if pages + pages // 2 > len(self.slots) or (headroom + 1) * pages > len(self.slots):
            raise ValueError(f"a step of {pages} pages, its slots kept {headroom} steps after their free, frees and "
                             f"writes more than the pool's {len(self.slots)} slots")
        self.backend, self.registration, self.pages, self.rng = backend, registration, pages, rng
        self.salts = random.Random(rng.getrandbits(64))
        self.lifetime, self.headroom = lifetime, headroom
        self.replay = memory_map.Replay(memory_map.Header(registration.digest, N), registration)
        self.steps, self.served, self.cursor, self.rekeyed = 0, 0, 0, 0
        self.full = set(range(N - pool, N))
        self.freed_at: dict[int, int] = {}
        self.ahead: Ahead | None = None
        self.taken = -1
        self.live = set(range(N))
        self.held: dict[int, tuple[int, bytes]] = {}
        self.appends: list[audit.Append] = []
        self.released: list[int] = []
        self.appended: list[int] = []

    def stale(self, window: int) -> list[int]:
        """The slots holding a block written `lifetime` windows or more before `window`, by its index."""
        old = [x for x, (a, _) in self.held.items() if a <= window - self.lifetime]
        return sorted(old, key=self.replay.holds)

    def step(self, s: int, window: int, pages: bool = True) -> memory_map.Change | None:
        """The trace's next step in `window`, recorded as memory-map step `s`: its `pages` writes if `pages`, and the
        stale blocks' re-keys."""
        return self.record(s, window, *self.serve(s, pages))

    def serve(self, s: int, pages: bool = True, fence: Callable[[int], None] | None = None):
        """Serving's step s: the slots it frees, then its writes as (slot, payload), if it writes `pages`.  With
        `fence`, each write lands in its slot as the step runs (`backend.write`), after fence(r) for the step r at
        which the block its slot held was released; without, the block lands when the session encodes it."""
        if not pages:
            return (), ()
        n, P = len(self.slots), self.pages
        if self.headroom:
            write = [self.slots[(self.cursor + u) % n] for u in range(P)]
            freed = [x for x in (self.slots[(self.cursor + self.headroom * P + u) % n] for u in range(P))
                     if x in self.full and x not in write]
            self.cursor = (self.cursor + P) % n
        else:
            k = P // 2 if self.served % 3 == 2 else 0
            freed = [x for x in (self.slots[(self.cursor + u) % n] for u in range(k)) if x in self.full]
            write = [self.slots[(self.cursor + k + u) % n] for u in range(P)]
            self.cursor = (self.cursor + k + P) % n
        self.served += 1
        self.full.difference_update(freed)
        self.freed_at.update((x, s) for x in freed)
        writes = tuple((x, payload(self.rng)) for x in write)
        for x, m in writes:
            if fence is not None:
                released = s if x in self.full else self.freed_at.get(x)
                if released is not None:
                    fence(released)
                self.backend.write(x, m, s)
            self.full.add(x)
        return tuple(freed), writes

    def record(self, s: int, window: int, freed, writes) -> memory_map.Change | None:
        """Step s's record, the session's, in `window`: serving's frees and writes, then the re-keys of the blocks stale
        in `window`, at the store's next fresh indices, appended to the backend's store and placed in their slots;
        None for a step that changes nothing."""
        slots, payloads = [x for x, _ in writes], [m for _, m in writes]
        rekey = [x for x in self.stale(window) if x not in slots and x not in freed]
        if not (freed or slots or rekey):
            return None
        self.steps += bool(slots)
        slots, payloads = slots + rekey, payloads + [self.held[x][1] for x in rekey]
        self.rekeyed += len(rekey)
        first = self.replay.extent
        written = tuple(memory_map.Written(x, first + u, rcodec.digest(m)) for u, (x, m) in enumerate(zip(slots, payloads)))
        change = self.replay.apply(memory_map.Step(s, tuple(freed), written))
        for x in freed:
            self.held.pop(x, None)
        self.held.update((x, (window, m)) for x, m in zip(slots, payloads))
        if slots:
            a = self.backend.append(payloads, [self.salts.randbytes(rb.SALT_BYTES) for _ in payloads])
            if a.first != change.first or a.digests != change.digests:
                raise AssertionError("the store's append is not the memory map's")
            getattr(self.backend, "place", lambda first, slots, step: None)(a.first, slots, s)
            self.appends.append(a)
            self.appended.extend(range(a.first, a.first + a.count))
            self.live.update(range(a.first, a.first + a.count))
        gone = [j for f, c in change.released for j in range(f, f + c)]
        self.released.extend(gone)
        self.live.difference_update(gone)
        return change

    def run(self, ahead: Ahead, every: int, fenced: bool = True) -> None:
        """Serving on its own thread, until the session ends: step s writes pages when s + 1 is a multiple of `every`
        (the step before every `every`'th pass), each write past `ahead`'s free fence unless not `fenced`."""
        fence = ahead.reuse if fenced else (lambda r: None)
        try:
            s = 0
            while ahead.begin(s):
                ahead.hand(s, self.serve(s, (s + 1) % every == 0, fence))
                s += 1
        except BaseException as e:
            ahead.fail(e)

    def take(self, window: int) -> int:
        """The step of the session's next pass, in `window`: one pass a serving step, the pass at s once serving has
        handed step s − 1 (`Ahead.take`), whose record, with the re-keys `window` needs, is applied first."""
        s = self.taken + 1
        got = self.ahead.take(s)
        if s > 0:
            self.record(s - 1, window, *(got or ((), ())))
        self.taken = s
        return s

    def plan(self) -> rb.Plan:
        return rb.Plan(runs_of(sorted(self.live)), self.replay.extent)


class CpuBackend:
    """The driver's reference as a backend: a store of `blocks` random payload blocks under the registration's codec,
    every pass `receipts.receipter` over its plan's blocks (every registered block unless the pass names a plan), its
    tree `receipts.tree`, salts from `entropy`; blocks appended after the codec seed (`append`, committed, then
    `encode_append` under the append's coin) are the session's.  A pass reads each block in its slot (`mem`): block j
    of the registered store in slot j, an appended one in the slot `place` gave it, put there encoded by
    `encode_append`; serving's `write` puts a payload in a slot as it runs, whatever block a pass may still have to
    read there.  A block's encode lands only in a slot nothing has been written into since the step that placed it:
    a block freed before its coin arrived (its pass dropped at a boundary) is never put over serving's later write.
    `misread(ticket)` is the pass's blocks whose slot held other bytes than the block when it read."""

    def __init__(self, blocks: int, rng: random.Random, read_bytes_per_s: float = 1e12) -> None:
        self.blocks, self.rng, self.read_bytes_per_s = blocks, rng, read_bytes_per_s
        self.payloads = [payload(rng) for _ in range(blocks)]
        self.salts = [rng.randbytes(rb.SALT_BYTES) for _ in range(blocks)]
        self.dropped: set[int] = set()
        self.gates: dict = {}
        self.misreads: tuple[int, ...] = ()
        self.slot_lock = threading.Lock()

    def payload_root(self, node: bytes) -> bytes:
        return audit.payload_tree(node, self.payloads, self.salts).commitment.root

    def encode(self, registration: audit.Registration) -> dict:
        self.store = audit.Store(registration, self.payloads, self.salts)
        self.plan = rb.Plan.whole(self.blocks)
        self._slots()
        return {"encode_bytes_per_s": 1.0}

    def _slots(self) -> None:
        self.mem, self.where, self.placed, self.wrote = dict(enumerate(self.store.blocks)), {}, {}, {}

    def begin_session(self) -> None:
        """A session's appends start again from N, over the registered store as encoded: the KV pool's filler is put
        back where the last session's pages overwrote it."""
        del self.store.blocks[self.blocks:]
        self.store.appends.clear()
        self.store.pending.clear()
        self._slots()

    def append(self, payloads, salts) -> audit.Append:
        return self.store.append(payloads, salts)

    def place(self, first: int, slots, step: int) -> None:
        """The slots of the blocks step `step` appended from `first`, in order."""
        with self.slot_lock:
            for j, x in enumerate(slots, first):
                self.where[j], self.placed[j] = x, step

    def write(self, slot: int, block: bytes, step: int) -> None:
        with self.slot_lock:
            self.mem[slot], self.wrote[slot] = block, step

    def encode_append(self, first: int, coin: bytes) -> None:
        count = len(self.store.pending[first])
        self.store.encode(first, coin)
        with self.slot_lock:
            for j in range(first, first + count):
                x = self.where.get(j, j)
                if self.wrote.get(x, -1) <= self.placed.get(j, self.wrote.get(x, -1)):
                    self.mem[x] = self.store.blocks[j]

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
        ticket["root"], ticket["misread"] = root, self.misreads
        return root, handle

    def misread(self, ticket: dict) -> tuple[int, ...]:
        return ticket.get("misread", ())

    def run_pass(self, seed: bytes, plan: rb.Plan | None = None):
        plan = plan or self.plan
        r = rb.receipter(self.store.registration.receipt, seed, self.store.registration.rows)
        zero = bytes(rcodec.BLOCK_BYTES)
        held = [zero if j in self.dropped else self.mem.get(self.where.get(j, j), zero) for j in plan]
        self.misreads = tuple(j for j, e in zip(plan, held) if j not in self.dropped and e != self.store.blocks[j])
        ys = tuple(r(j, e) for j, e in zip(plan, held))
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
                 kv_headroom: int = 0, ahead: int | None = None, fill: bool = False, log=print) -> None:
        """`ahead`: a kv session's serving runs on its own thread, at most that many steps past the passes (`Ahead`,
        `fill` its test schedule); `kv_headroom`: its trace's steps between a slot's free and its reuse."""
        self.conn, self.backend, self.node, self.plan, self.log = conn, backend, node, plan, log
        self.defer_openings, self.warmup_ns = defer_openings, warmup_ns
        self.kv_every, self.kv_pool, self.kv_headroom = kv_every, kv_pool, kv_headroom
        self.ahead, self.fill = ahead, fill
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
                        "layout": layout})
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
        trace = serving = None
        if mode in TRACED:
            if not hasattr(self.backend, "append"):
                raise ValueError(f"a {mode} session needs a backend that appends blocks after the codec seed (--cpu)")
            if mode == "kvoverwrite" and self.ahead is None:
                raise ValueError("a kvoverwrite session's serving runs ahead of its passes (ahead=K)")
            trace = ServingTrace(self.backend, self.registration, int(fraction), self.kv_pool,
                                 random.Random(self.rng.getrandbits(64)), head.get("lifetime", audit.LIFETIME),
                                 self.kv_headroom)
            if self.ahead is not None:
                trace.ahead = Ahead(self.ahead, self.fill)
                serving = threading.Thread(target=trace.run, args=(trace.ahead, self.kv_every, mode != "kvoverwrite"),
                                           daemon=True)
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
        launched, sent_plan, sent_released, sent_appends = 0, rb.Plan.whole(N), 0, 0
        ahead: Ahead | None = getattr(trace, "ahead", None)
        self.misread = {}

        def launch(t: int, i: int, after, start: int | None) -> _Pass:
            """Pass i of window t; with a serving trace, after the trace's step if one falls due (the backend's next
            engine step, serving), over every live block; with serving ahead, at the step it takes from serving, and
            started once it has.  Then one pass is in flight at a time: the next one's step waits on serving, whose
            write may wait on this one's read."""
            nonlocal launched
            n, launched = launched, launched + 1
            if trace is None:
                ticket = backend.launch(i, after)
                return _Pass(t, i, ticket, start, step=step_of(ticket) if step_of is not None else n)
            if ahead is not None:
                n = trace.take(t)
                if mode == "kvdrop":
                    backend.arm(trace.appended)
            elif mode in ("serve", "servedrop"):
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
            return _Pass(t, i, ticket, clock() if ahead is not None else start,
                         step=step_of(ticket) if step_of is not None else n, plan=plan,
                         upto=(len(trace.appends), len(trace.released)))

        backend.begin_window(take(wire.coin_of(head["coin"])), self.registration.digest)
        pending: deque = deque()
        passes, last, nxt, stopping, ended, discarded = [], None, 1, False, None, 0
        depth = 1 if ahead is not None else 2
        misread = getattr(backend, "misread", None)

        def send(x: _Pass, root: bytes, handle, seen: int) -> None:
            nonlocal sent_plan, sent_released
            with self.lock:
                self.kept[(x.t, x.i)] = handle
            msg = {"type": "root", "t": x.t, "i": x.i, "root": root.hex(), "step": x.step}
            if x.plan is not None:
                if x.upto[1] > sent_released:
                    msg["released"] = [list(r) for r in runs_of(sorted(trace.released[sent_released:x.upto[1]]))]
                msg["change"] = [[list(r) for r in side] for side in plan_change(sent_plan, x.plan)]
                self.plans[(x.t, x.i)] = x.plan
                sent_plan, sent_released = x.plan, x.upto[1]
            if misread is not None and (got := misread(x.ticket)):
                self.misread[(x.t, x.i)] = got
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

        if serving is not None:
            serving.start()
        try:
            inflight = deque([launch(0, 0, None, clock())])
            while inflight:
                if not stopping and len(inflight) < depth:
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
                if ahead is not None:
                    ahead.passed(x.step)
                self._drain(pending)
                if x.discard:
                    discarded += 1
                    continue
                if not pending:
                    send(x, root, handle, seen)
                    last = x
                    if depth == 1 and not stopping:
                        inflight.append(launch(t, nxt, x.ticket, None))
                        nxt += 1
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
        finally:
            if ahead is not None:
                ahead.end()
            if serving is not None:
                serving.join()
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
        if ahead is not None:
            out["ahead"] = {**ahead.stats(), "headroom": getattr(trace, "headroom", None)}
        out["misread"] = [[tt, i, j] for (tt, i), js in sorted(self.misread.items()) for j in js]
        out["failure_on_misread"] = (self.block(failure["t"], *at) in self.misread.get((failure["t"], at[0]), ())
                                     if at else None)
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
    ap.add_argument("--ahead", type=int, default=None, metavar="K",
                    help="a kv session's serving on its own thread, at most K steps past the passes (default: none, "
                         "each step before its pass)")
    ap.add_argument("--kv-headroom", type=int, default=0, metavar="H",
                    help="a kv session's steps between a slot's free and its reuse (0: every third step frees)")
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
                   kv_pool=a.kv_pool, kv_headroom=a.kv_headroom, ahead=a.ahead, log=lambda m: print(m, flush=True))
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
