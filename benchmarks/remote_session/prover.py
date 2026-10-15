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
~~~

    python prover.py --connect HOST:PORT --out OUT --plan honest,skip:3e-5 [device options, `device.py`]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
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

from verity.core.protocols.remote_bandwidth import audit, coins, receipts as rb  # noqa: E402
from verity.core.protocols.remote_bandwidth import codec as rcodec  # noqa: E402

import wire  # noqa: E402

PAYLOAD_LABEL = hashlib.sha512(b"verity/remote-session/payload-salts/v0").digest()
OPENINGS_PART = 128
"""Openings a frame: a window's go out in parts, so a root waits for one part's send, not the window's."""
DROPPED_LISTED = 4096
"""A session's result lists its dropped blocks up to this many, and counts them always."""
BOUNDARIES = ("coin", "close", "end")


@dataclass
class _Pass:
    t: int
    i: int
    ticket: object
    start: int | None
    discard: bool = False


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
        if mode not in ("honest", "skip"):
            raise ValueError(f"a session's mode is honest or skip:F, not {part!r}")
        out.append((mode, float(f) if f else 0.0))
    return out


class CpuBackend:
    """The driver's reference as a backend: a store of `blocks` random payload blocks under the registration's codec,
    every pass `receipts.receipter` over every block, its tree `receipts.tree`, salts from `entropy`."""

    def __init__(self, blocks: int, rng: random.Random, read_bytes_per_s: float = 1e12) -> None:
        self.blocks, self.rng, self.read_bytes_per_s = blocks, rng, read_bytes_per_s
        self.payloads = [bytes(rng.getrandbits(8) for _ in range(rcodec.BLOCK_BYTES - 16)) + bytes(16)
                         for _ in range(blocks)]
        self.salts = [rng.randbytes(rb.SALT_BYTES) for _ in range(blocks)]
        self.dropped: set[int] = set()

    def payload_root(self, node: bytes) -> bytes:
        return audit.payload_tree(node, self.payloads, self.salts).commitment.root

    def encode(self, registration: audit.Registration) -> dict:
        self.store = audit.Store(registration, self.payloads, self.salts)
        self.plan = rb.Plan.whole(self.blocks)
        return {"encode_bytes_per_s": 1.0}

    def calibrate(self) -> dict:
        return {"pass_ns_max": 0, "gap_ns_max": 0, "read_bytes_per_s": self.read_bytes_per_s}

    def arm(self, dropped) -> None:
        self.dropped = {int(j) for j in dropped}

    def begin_window(self, challenge: bytes, previous: bytes | None) -> None:
        self.challenge = challenge
        if previous is not None:
            self.previous = previous

    def launch(self, i: int, after: dict | None) -> dict:
        """Pass i of the window, chained from `after`'s root (None: the window's previous root), run when awaited."""
        return {"challenge": self.challenge, "i": i, "after": after, "previous": self.previous, "root": None}

    def wait(self, ticket: dict):
        after = ticket["after"]
        prev = ticket["previous"] if after is None else after["root"]
        root, handle = self.run_pass(rb.seed(ticket["challenge"], ticket["i"], prev))
        ticket["root"] = root
        return root, handle

    def run_pass(self, seed: bytes):
        r = rb.receipter(self.store.registration.receipt, seed, self.store.registration.rows)
        ys = tuple(r(j, bytes(rcodec.BLOCK_BYTES) if j in self.dropped else self.store.blocks[j]) for j in self.plan)
        salts = tuple(self.rng.randbytes(rb.SALT_BYTES) for _ in range(rb.leaves(self.blocks)))
        tree = rb.tree(seed, self.plan, ys, salts)
        return tree.commitment.root, (ys, salts, tree)

    def open_window(self, items) -> list[list[tuple]]:
        return [self.open(handle, positions) for handle, positions in items]

    def open(self, handle, positions: list[int]) -> list[tuple]:
        ys, salts, tree = handle
        out = []
        for p in positions:
            c = p // rb.PER_LEAF
            leaf = b"".join(ys[rb.PER_LEAF * c:rb.PER_LEAF * (c + 1)])
            out.append((self.store.blocks[p], leaf, salts[c], tree.open(c).path, self.store.salts[p],
                        self.store.payload_tree.open(p).path))
        return out


class ProverSide:
    def __init__(self, conn: wire.Conn, backend, node: bytes, plan: list[tuple[str, float]], *, seed: int = 0,
                 defer_openings: bool = True, warmup_ns: int = 0, log=print) -> None:
        self.conn, self.backend, self.node, self.plan, self.log = conn, backend, node, plan, log
        self.defer_openings, self.warmup_ns = defer_openings, warmup_ns
        self.rng = random.Random(seed)
        self.inbox: queue.Queue = queue.Queue()
        self.jobs: queue.Queue = queue.Queue()
        self.kept: dict[tuple[int, int], object] = {}
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
                    out[k] = audit.Opening(i, p, f[0], f[1], f[2], tuple(f[3]), f[4], tuple(f[5]), b"")
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
                                "on_dropped": int(hits(self.dropped, [p for _, p in draws]).sum())})
            self.jobs.task_done()

    def expect(self, kind: str):
        while True:
            head, body, stamp = self.inbox.get()
            if head["type"] == kind:
                return head, body, stamp
            if head["type"] in ("lost", "bye"):
                raise wire.Closed(head.get("why", "bye"))

    def register(self, codec: rcodec.Codec = rcodec.DEFAULT, receipt: str = rb.DEFAULT, rows: int | None = None) -> dict:
        root = self.backend.payload_root(self.node)
        self.conn.send({"type": "hello", "node": self.node.hex(), "blocks": self.backend.blocks,
                        "payload_root": root.hex(), "codec": codec.name, "receipt": receipt, "rows": rows})
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
        if mode == "skip":
            pick = np.random.default_rng(self.rng.getrandbits(64))
            dropped = np.sort(pick.choice(N, size=round(fraction * N), replace=False)).astype(np.int64)
        else:
            dropped = np.empty(0, dtype=np.int64)
        self.backend.arm(dropped)
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
        backend.begin_window(take(wire.coin_of(head["coin"])), self.registration.digest)
        inflight = deque([_Pass(0, 0, backend.launch(0, None), clock())])
        pending: deque = deque()
        passes, last, nxt, stopping, ended, discarded = [], None, 1, False, None, 0

        def send(x: _Pass, root: bytes, handle, seen: int) -> None:
            with self.lock:
                self.kept[(x.t, x.i)] = handle
            self.conn.send({"type": "root", "t": x.t, "i": x.i, "root": root.hex()})
            passes.append((x.t, x.i, x.start, seen, clock()))

        while inflight:
            if not stopping and len(inflight) < 2:
                inflight.append(_Pass(t, nxt, backend.launch(nxt, inflight[-1].ticket), None))
                nxt += 1
            x = inflight.popleft()
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
            if stamp < x.start:
                discarded += 1
            else:
                send(x, root, handle, seen)
                last = x
            if msg["type"] == "close":
                stopping = True
                continue
            t += 1
            backend.begin_window(take(wire.coin_of(msg["coin"])), None)
            inflight.append(_Pass(t, 0, backend.launch(0, last.ticket if last else None), None if inflight else clock()))
            nxt = 1
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
        return {"k": k, "mode": mode, "fraction": fraction,
                "dropped": dropped.tolist() if len(dropped) <= DROPPED_LISTED else None, "dropped_count": len(dropped),
                "failure_on_dropped": bool(hits(dropped, [at[1]])[0]) if at else None,
                "accepted": ended["accepted"], "failure": failure, "passes": [per_window[tt] for tt in range(W)],
                "pass_host_ns": [seen - start for _, _, start, seen, _ in passes],
                "gap_host_ns": [sent - seen for _, _, _, seen, sent in passes], "discarded": discarded,
                "opened": self.opened, "warmup": self.warmup, "device": device}

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
    import device

    device.add_arguments(ap)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.setswitchinterval(2e-4)
    backend = device.from_arguments(a, log=lambda m: print(m, flush=True))
    host, port = a.connect.rsplit(":", 1)
    conn = wire.connect(host, int(port))
    p = ProverSide(conn, backend, a.node.encode(), parse_plan(a.plan), seed=a.seed,
                   defer_openings=not a.overlap_openings, warmup_ns=round(a.warmup_ms * 1e6),
                   log=lambda m: print(m, flush=True))
    cal = p.register()
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
        print(f"prover: session {r['k']} ({mode}:{fraction}): accepted={r['accepted']} passes={r['passes']}", flush=True)
        (a.out / "prover.json").write_text(json.dumps(
            {"schema": "verity/remote-session/prover/v0", "registration": wire.registration_json(p.registration),
             "calibration": cal, "gates": backend.gates, "calibration_trace": stalls, "sessions": results,
             "wire": conn.totals()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
