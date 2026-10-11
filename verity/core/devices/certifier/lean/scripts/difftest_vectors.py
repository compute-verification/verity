"""The difftest's data: event traces of the active certifier, and what `active.Proxy` does on each, for `CertifierDevice/Difftest.lean`.

    python scripts/difftest_vectors.py <verity checkout> --out DIR [--count N]

Writes `DIR/difftest/instances.jsonl`, one instance per line as `CertifierDevice.Difftest.evaluate` reads it (the proxy's
configuration, a trace of events, the path that made it and, for an honest run, the Program's frames), and
`DIR/difftest/python.jsonl`, the same line's results from the checkout's real `active.Proxy` driven through the trace:
every action it took, in order (each link write, server connection, server write with whether it waited for its drain,
and published record with its grid's parameters), and at the end its declared syncs, its shaper's hand times, its log
of egress frames read and its two counts. The Lean audit runs `CertifierDevice/DifftestMain.lean` on both files (`runs`
in `lean-audit.json`, which pins their SHA-256), and `verity/core/protocols/certifier/tests/
test_network_certifier_device_lean_difftest.py` checks that this script still writes them.

**Driving the proxy event by event.** The proxy runs on its own asyncio loop, unchanged, in a world this script holds:
its `clock` reads the trace's clock, and its `sleep` parks the calling ticker (egress or ingress, told apart by the
task's coroutine) until the trace's next tick of that grid releases it, whatever the delay; the link is an in-memory
reader and writer whose backlog the tick event sets; `asyncio.open_connection` is replaced by one that records the
connection and waits for the trace's `connected` or `refused`; a server connection's reader gives the trace's bytes, and
its writer records each write and holds a drain until the trace's `delivered`; `asyncio.to_thread` (the proxy's
executor shutdown) runs its call inline. After each event the loop runs until no callback is ready (`loop._ready`
empty, with no timer, socket or thread in play), so each event's effects, and the order of the
actions its tasks take, are a function of the trace alone. An event the proxy isn't waiting for (bytes on a connection
not open, a second `connected`) is dropped, as `step` drops it.

**The traces.** Each run has an honest base: sessions on random streams, each a request of one or two frames written on
the link inside an ingress bucket, released by the ingress grid's FIFO, its connection opened and its request delivered
within the slack, and a server that answers with events numbered in the Program's canonical order (by ready bucket, or
by request, so a frame may be read before the one ahead of it), its `id:` spelled one of several ways, each read inside
the egress bucket before the one it is due by, sometimes in two reads, under declared syncs that cover it; and every
tick of both grids, each a little after its boundary. A path then perturbs it (`PATHS`): a late tick or a stall of
several buckets, a link backlog at or over one or four buckets, a refused connection, a request forwarded late (past
the slack, or only past its anchor's egress bucket) or never, a queue past Q on either grid, a sync outside Σ_sync,
valid syncs that don't cover the server (frames ready after their eligibility, each handed late, which faults the
window it is handed in `coverage-never-established`), the five malformed inputs (an event
without a position, a position seen twice (also while the queue holds exactly Q), an event longer than a frame, an
event on a stream with no released request, bytes left at a stream's end), link bytes that aren't frames (partial,
padding, unknown kinds, a length past the frame, an early end), the online prover in place of the given syncs (with or
without a prior history, and an envelope outside Σ_sync), the prover with an ingress tick read early, so that the
first event of the request it releases is read before the request's anchor bucket (the prover refuses the first choice
that reads it, which faults the egress window declaring it `monitor-unavailable`), or several at once. The prover
chooses each window's sync `lag` buckets before the window (1, T − 1, or between), on `active.InlineExecutor`, so every
choice is ready by its tick; each tick that starts a choice wakes late within the slack, after some of its bucket's
response reads, which the choice has read but mustn't see. Every reading is integer nanoseconds at least `MARGIN` from every grid boundary and every
boundary plus the slack, where Python's float seconds and Lean's integers could part.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import verity.core.protocols.certifier
from verity.core.protocols.certifier import online
from verity.core.protocols.certifier.active import DATA, LAST, PAD, InlineExecutor, Proxy, encode_frame
from verity.core.protocols.certifier.grid import GridParams, Reason, Record
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, expected_grids

COUNT = 1500
MS = 1_000_000
MARGIN = 10_000
RHOS = [Fraction(1), Fraction(2), Fraction(1, 2), Fraction(1, 3), Fraction(3, 2)]
PATHS = ["honest", "honest", "honest", "late-tick", "stall", "backlog", "overrun", "refused", "late-forward",
         "undelivered", "overflow-egress", "overflow-ingress", "advice", "no-id", "duplicate", "duplicate-at-capacity",
         "long", "no-request", "trailing", "slow-server", "link-garbage", "prover", "prover", "prover",
         "early-tick", "chaos", "chaos"]
PERTURBATIONS = [p for p in dict.fromkeys(PATHS)
                 if p not in ("honest", "chaos", "prover", "early-tick", "duplicate-at-capacity")]


# --- the world the proxy runs in ------------------------------------------------------------------------------------

class Harness:
    """The proxy's clock, sleep, link and server connections, each held until the trace moves it."""

    def __init__(self) -> None:
        self.now = 0
        self.backlog = 0
        self.actions: list[list] = []
        self.ticks: dict[str, asyncio.Future] = {}
        self.link: asyncio.Queue = asyncio.Queue()
        self.connecting: dict[int, asyncio.Future] = {}
        self.reads: dict[int, asyncio.Queue] = {}
        self.drains: dict[int, asyncio.Future] = {}
        self.proxy: Proxy | None = None

    def clock(self) -> float:
        return self.now / 1e9

    async def sleep(self, delay: float) -> None:
        grid = {"_egress": "egress", "_ingress": "ingress"}[asyncio.current_task().get_coro().__name__]
        future = asyncio.get_running_loop().create_future()
        self.ticks[grid] = future
        await future

    async def open_connection(self, host, port):
        task = asyncio.current_task()
        s = next(s for s, st in self.proxy.streams.items() if st.task is task)
        self.actions.append(["connect", s])
        future = asyncio.get_running_loop().create_future()
        self.connecting[s] = future
        return await future

    def publish(self, record: Record, params: GridParams) -> None:
        self.actions.append(["publish", record_json(record), params_json(params)])


class LinkReader:
    def __init__(self, h: Harness) -> None:
        self.h = h

    async def read(self, n: int) -> bytes:
        return await self.h.link.get()


class LinkWriter:
    def __init__(self, h: Harness) -> None:
        self.h, self.transport = h, self

    def get_write_buffer_size(self) -> int:
        return self.h.backlog

    def writelines(self, chunks) -> None:
        self.h.actions.append(["link", b"".join(bytes(c) for c in chunks).hex()])

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass


class UpstreamReader:
    def __init__(self, h: Harness, s: int) -> None:
        self.h, self.s = h, s

    async def read(self, n: int) -> bytes:
        return await self.h.reads[self.s].get()


class UpstreamWriter:
    def __init__(self, h: Harness, s: int) -> None:
        self.h, self.s, self.transport = h, s, self

    def set_write_buffer_limits(self, high=None, low=None) -> None:
        pass

    def write(self, data: bytes) -> None:
        self.h.actions.append(["upstream", self.s, bytes(data).hex(), False])

    async def drain(self) -> None:
        last = self.h.actions[-1]
        assert last[:2] == ["upstream", self.s], "a drain follows its write"
        last[3] = True
        future = asyncio.get_running_loop().create_future()
        self.h.drains[self.s] = future
        await future

    def close(self) -> None:
        pass


async def settle() -> None:
    """Run the loop until no callback is ready: every task waits on the trace again."""
    loop = asyncio.get_running_loop()
    for _ in range(100_000):
        await asyncio.sleep(0)
        if not loop._ready:  # noqa: SLF001 -- no timer, socket or thread in play: an empty ready queue is quiescence
            return
    raise RuntimeError("the proxy never settles")


def apply(h: Harness, ev: list) -> None:
    kind = ev[0]
    if kind == "tick":
        _, grid, h.now, h.backlog = ev
        if (f := h.ticks.pop(grid, None)) is not None and not f.done():
            f.set_result(None)
        return
    if kind == "link":
        _, h.now, data = ev
        h.link.put_nowait(bytes.fromhex(data))
        return
    _, s, h.now, *rest = ev
    if kind in ("connected", "refused"):
        if (f := h.connecting.pop(s, None)) is not None and not f.done():
            if kind == "connected":
                h.reads[s] = asyncio.Queue()
                f.set_result((UpstreamReader(h, s), UpstreamWriter(h, s)))
            else:
                f.set_exception(ConnectionRefusedError(f"stream {s} refused"))
    elif kind == "upstream":
        if (q := h.reads.get(s)) is not None:
            q.put_nowait(bytes.fromhex(rest[0]))
    elif kind == "delivered":
        if (f := h.drains.pop(s, None)) is not None and not f.done():
            f.set_result(None)
    else:
        raise ValueError(f"no event {kind!r}")


# --- the configuration and the results ------------------------------------------------------------------------------

def grid_params(p: dict) -> GridParams:
    return GridParams(buckets=p["T"], allocation=p["r"], frame_bytes=p["B"], bucket_ms=p["tau"], queue_capacity=p["Q"])


def params_json(p: GridParams) -> dict:
    return {"T": p.buckets, "r": p.allocation, "B": p.frame_bytes, "tau": p.bucket_ms, "Q": p.queue_capacity}


def record_json(r: Record) -> dict:
    return {"link": r.link, "window": r.window, "grid": [[None if x is None else x.hex() for x in row] for row in r.grid],
            "status": None if r.status.reason is None else r.status.reason.value}


def sync_of(x: list) -> ClockSync:
    return ClockSync(x[0], Fraction(x[1], x[2]))


def sync_json(s: ClockSync) -> list:
    return [s.delta, s.rho.numerator, s.rho.denominator]


def sync_set_of(x: dict) -> SyncSet:
    return SyncSet(tuple(x["deltas"]), tuple(Fraction(n, d) for n, d in x["rhos"]))


def prover_of(x: dict | None, sync_set: SyncSet, buckets: int) -> online.Prover | None:
    if x is None:
        return None
    k, margin, fallback = x["rule"]
    prior = [online.Past(p["window"], sync_of(p["sync"]), p["frames"], p["late"],
                         tuple(None if f is None else (f[0], f[1]) for f in p["frontier"])) for p in x["prior"]]
    return online.Prover(sync_set, buckets, online.cover(k, margin, fallback), sync_of(x["envelope"]), prior=prior,
                         lag=x["lag"])


async def drive(h: Harness, inst: dict) -> dict:
    c = inst["config"]
    egress, ingress, sync_set = grid_params(c["egress"]), grid_params(c["ingress"]), sync_set_of(c["sync_set"])
    proxy = h.proxy = Proxy(("upstream", 0), egress, ingress, sync_set, [sync_of(s) for s in c["syncs"]],
                            origin=c["origin"] / 1e9, slack=c["slack"] / 1e9, clock=h.clock, sleep=h.sleep,
                            publish=h.publish, egress_link=c["egress_link"], ingress_link=c["ingress_link"],
                            prover=prover_of(c["prover"], sync_set, egress.buckets), windows=c["windows"],
                            executor=InlineExecutor())
    running = asyncio.create_task(proxy.run(LinkReader(h), LinkWriter(h)))
    await settle()
    for ev in inst["events"]:
        apply(h, ev)
        await settle()
    if not running.done():
        raise RuntimeError("the trace leaves ticks unrun")
    running.result()
    for st in proxy.streams.values():
        if st.task is not None and not st.task.cancelled() and st.task.exception() is not None:
            raise st.task.exception()
    return {"actions": h.actions, "syncs": [sync_json(s) for s in proxy.shaper.syncs], "hands": proxy.shaper.hands,
            "frames": [list(f) for f in proxy.frames], "link_frames_read": proxy.link_frames_read,
            "requests_released": proxy.requests_released}


async def inline_to_thread(fn, /, *args, **kwargs):
    return fn(*args, **kwargs)


def python_results(inst: dict) -> dict:
    """`drive` on a fresh loop, with `asyncio.open_connection` and `asyncio.to_thread` (which the proxy calls) the
    harness's: a real thread would finish when the host gets to it, after `settle` has taken the loop as quiet."""
    saved = asyncio.open_connection, asyncio.to_thread
    h = Harness()
    asyncio.open_connection, asyncio.to_thread = h.open_connection, inline_to_thread
    try:
        return asyncio.run(drive(h, inst))
    finally:
        asyncio.open_connection, asyncio.to_thread = saved


# --- the traces -----------------------------------------------------------------------------------------------------

@dataclass
class Session:
    stream: int
    send: int  # the ingress bucket inside which its request is written on the link
    parts: int  # 1: a LAST frame; 2: DATA, then LAST
    delays: list[int]  # each response frame's ready bucket after the anchor
    head: bool = False  # an HTTP response head before its first event
    ids: bytes = b"id: %d\n"  # how its events spell their position
    data_only: bool = False  # its request has no LAST frame, so it is never released
    refused: bool = False
    trailing: bytes = b""  # bytes after its last event
    replace: dict[int, bytes] = field(default_factory=dict)  # a response frame's bytes by read time, in place of its own
    release: list[int] = field(default_factory=list)  # the ingress tick that releases each request frame
    anchor: int = 0
    connected: int = 0
    delivered: int = 0
    times: list[int] = field(default_factory=list)  # when each response frame is read
    ready: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    end: int = 0


class Trace:
    """One run: the configuration, the sessions and the events, kept apart until `events` merges them by time."""

    def __init__(self, rng: random.Random, path: str) -> None:
        self.rng, self.path = rng, path
        self.tau = 100
        self.T = rng.choice([3, 4, 5])
        self.r = rng.choice([1, 2, 3])
        self.B = rng.choice([48, 64, 96])
        self.tau_in, self.T_in = (self.tau, self.T) if rng.random() < 0.6 else (self.tau // 2, 2 * self.T)
        self.r_in = rng.choice([1, 2])
        self.B_in = rng.choice([32, 48])
        self.Q = self.Q_in = 1 << 16
        self.W = rng.choice([2, 3, 4])
        self.origin = rng.choice([0, 0, 2_000_000_000, 123_456_789])
        self.slack = rng.choice([25 * MS, 10 * MS, 60 * MS])
        self.sessions: list[Session] = []
        self.extra: list[tuple[int, int, list]] = []  # (time, order, event) beside the sessions'
        self.tick_late: dict[tuple[str, int], int] = {}
        self.backlog: dict[int, int] = {}
        self.syncs: list[ClockSync] = []
        self.deltas: list[int] = []
        self.rhos: list[Fraction] = []
        self.prover: dict | None = None
        self.windows = self.W

    # times

    def safe(self, t: int) -> bool:
        for tau in (self.tau, self.tau_in):
            for base in (0, self.slack):
                d = (t - self.origin - base) % (tau * MS)
                if min(d, tau * MS - d) < MARGIN:
                    return False
        return True

    def at(self, t: int) -> int:
        """`t`, or the first safe reading after it."""
        while not self.safe(t):
            t += MARGIN
        return t

    def boundary(self, tau: int, t: int) -> int:
        return self.origin + t * tau * MS

    def within(self, tau: int, bucket: int, lo: float = 0.05, hi: float = 0.9) -> int:
        """A reading inside bucket `bucket` of length `tau`."""
        return self.at(self.boundary(tau, bucket) + int(self.rng.uniform(lo, hi) * tau * MS))

    def tick_times(self, grid: str) -> dict[int, int]:
        """Each tick's reading: a little after its boundary, or as late as `tick_late` says, and never before the one
        before it, since a ticker wakes in order."""
        tau, T = (self.tau, self.T) if grid == "egress" else (self.tau_in, self.T_in)
        out, prev = {}, 0
        for t in range(self.W * T):
            late = self.tick_late.get((grid, t), self.rng.randrange(2 * MARGIN, 3 * MS))
            out[t] = prev = self.at(max(self.boundary(tau, t) + late, prev + 3 * MARGIN))
        return out

    # the honest base

    def build(self) -> None:
        rng = self.rng
        horizon_in = self.W * self.T_in
        streams = rng.sample([1, 2, 3, 7, 9, 300, 65536, 2 ** 32 - 1], rng.randint(1, 5))
        for s in streams:
            n = rng.randint(1, 4)
            delays = sorted(rng.randint(1, 3) for _ in range(n))
            send = rng.randrange(max(1, horizon_in - 2 * self.T_in // self.W - 1))
            ids = rng.choice([b"id: %d\n", b"id: %d\n", b"id:%d\n", b"id:\t%d \r\n", b"data: -\nid: %d\n",
                              b"id: %d\nid: 9\n"])
            self.sessions.append(Session(s, send, rng.choice([1, 1, 2]), delays, head=rng.random() < 0.3, ids=ids))
        self.sessions.sort(key=lambda x: x.send)
        self.ticks_in = self.tick_times("ingress")
        self.release()
        for x in self.sessions:
            first = self.ticks_in.get(x.release[0], self.boundary(self.tau_in, horizon_in))
            last = self.ticks_in.get(x.release[-1], self.boundary(self.tau_in, horizon_in))
            x.connected = self.at(first + rng.randrange(2 * MARGIN, MS))
            x.delivered = self.at(max(x.connected, last) + rng.randrange(2 * MARGIN, MS))
            t = x.delivered
            for d in x.delays:
                bucket = x.anchor + d - 1
                t = max(t + 3 * MARGIN, self.within(self.tau, bucket))
                while t > self.boundary(self.tau, bucket + 1) - 3 * MARGIN:
                    bucket += 1
                    t = max(t, self.within(self.tau, bucket))
                x.times.append(t)
                x.ready.append(bucket + 1)
            x.end = self.at(x.times[-1] + rng.randrange(2 * MARGIN, 2 * MS))
        by_ready = rng.random() < 0.6  # else by request, so a frame may be read before the one ahead of it
        order = sorted(((x.ready[k] if by_ready else 0, x.anchor, x.stream, k, i) for i, x in enumerate(self.sessions)
                        for k in range(len(x.delays))))
        for pos, (_, _, _, k, i) in enumerate(order):
            self.sessions[i].positions.append(pos)
        self.sessions_order = order
        self.choose_syncs()

    def release(self) -> None:
        """The ingress FIFO: each request frame arrives at the bucket after the one it is written in, and tick t releases
        up to r_in of those arrived by t, in order."""
        queue = [(x.send + 1, i, j) for i, x in enumerate(self.sessions) for j in range(x.parts)]
        queue.sort()
        t, k = 0, 0
        out: dict[tuple[int, int], int] = {}
        while k < len(queue):
            n = 0
            while k < len(queue) and queue[k][0] <= t and n < self.r_in:
                out[queue[k][1:]] = t
                k, n = k + 1, n + 1
            t += 1
        for i, x in enumerate(self.sessions):
            x.release = [out[(i, j)] for j in range(x.parts)]
            x.anchor = x.release[-1] * self.tau_in // self.tau

    def choose_syncs(self) -> None:
        rng = self.rng
        need = [0] * self.W
        for x in self.sessions:
            w = x.anchor // self.T
            if w < self.W:
                need[w] = max(need[w], max(r - x.anchor for r in x.ready))
        self.deltas = sorted(set(need) | set(rng.sample(range(8), rng.randint(1, 3))))
        self.rhos = rng.sample(RHOS, rng.randint(1, 3))
        self.syncs = [ClockSync(min(d for d in self.deltas if d >= need[w]), rng.choice(self.rhos)) for w in range(self.W)]

    # the events

    def request_frames(self, x: Session) -> list[bytes]:
        body = f"GET /{x.stream}".encode()[: self.B_in - 9]
        kinds = [DATA] * x.parts if x.data_only else [DATA, LAST] if x.parts == 2 else [LAST]
        return [encode_frame(k, x.stream, body[: self.B_in - 9 - (j == 0)], self.B_in) for j, k in enumerate(kinds)]

    def response(self, x: Session, k: int) -> bytes:
        head = b"HTTP/1.1 200 OK\r\n\r\n" if x.head and k == 0 else b""
        event = head + x.ids % x.positions[k] + b"data: %d-%d\n\n" % (x.stream, k)
        return event if 9 + len(event) <= self.B else b"id:%d\n\n" % x.positions[k]

    def events(self) -> list[list]:
        ev: list[tuple[int, int, list]] = list(self.extra)
        seq = iter(range(1, 1 << 30))
        for grid, ticks in (("egress", self.tick_times("egress")), ("ingress", self.ticks_in)):
            for t, now in ticks.items():
                ev.append((now, next(seq), ["tick", grid, now, self.backlog.get(t, 0) if grid == "egress" else 0]))
        sends: dict[int, list[bytes]] = {}
        for x in self.sessions:
            sends.setdefault(x.send, []).extend(self.request_frames(x))
        for b, frames in sorted(sends.items()):
            data = b"".join(frames)
            cuts = sorted(self.rng.sample(range(1, len(data)), min(len(data) - 1, self.rng.choice([0, 0, 1, 2]))))
            t = self.within(self.tau_in, b, 0.1, 0.5)
            for lo, hi in zip([0, *cuts], [*cuts, len(data)]):
                ev.append((t, next(seq), ["link", t, data[lo:hi].hex()]))
                t = self.at(t + 3 * MARGIN)
        for x in self.sessions:
            ev.append((x.connected, next(seq), ["refused" if x.refused else "connected", x.stream, x.connected]))
            if x.delivered is not None:
                ev.append((x.delivered, next(seq), ["delivered", x.stream, x.delivered]))
            for k, t in enumerate(x.times):
                data = x.replace.get(t, self.response(x, k))
                if len(data) > 4 and self.rng.random() < 0.2:  # one event in two reads at one reading
                    cut = self.rng.randrange(1, len(data))
                    ev.append((t, next(seq), ["upstream", x.stream, t, data[:cut].hex()]))
                    data = data[cut:]
                ev.append((t, next(seq), ["upstream", x.stream, t, data.hex()]))
            if x.trailing:
                ev.append((x.end, next(seq), ["upstream", x.stream, x.end, x.trailing.hex()]))
            end = self.at(x.end + 2 * MARGIN)
            ev.append((end, next(seq), ["upstream", x.stream, end, ""]))
        ev.sort(key=lambda e: (e[0], e[1]))
        return [e for _, _, e in ev]

    def config(self) -> dict:
        return {"egress": {"T": self.T, "r": self.r, "B": self.B, "tau": self.tau, "Q": self.Q},
                "ingress": {"T": self.T_in, "r": self.r_in, "B": self.B_in, "tau": self.tau_in, "Q": self.Q_in},
                "sync_set": {"deltas": self.deltas, "rhos": [[r.numerator, r.denominator] for r in self.rhos]},
                "syncs": [sync_json(s) for s in self.syncs], "windows": self.windows, "origin": self.origin,
                "slack": self.slack, "prover": self.prover, "egress_link": "responses", "ingress_link": "requests"}

    def program(self) -> list[list]:
        frames = []
        for _, _, _, k, i in self.sessions_order:
            x = self.sessions[i]
            frames.append([encode_frame(DATA, x.stream, self.response(x, k), self.B).hex(), x.anchor, k])
        return frames

    # the paths

    def perturb(self, path: str) -> None:
        rng, xs = self.rng, self.sessions
        x = rng.choice(xs)
        if path == "late-tick":
            grid = rng.choice(["egress", "ingress"])
            T = self.T if grid == "egress" else self.T_in
            self.tick_late[(grid, rng.randrange(self.W * T))] = self.slack + rng.randrange(MS, 20 * MS)
            if grid == "ingress":
                self.ticks_in = self.tick_times("ingress")
        elif path == "stall":
            grid = rng.choice(["egress", "ingress"])
            tau, T = (self.tau, self.T) if grid == "egress" else (self.tau_in, self.T_in)
            t0 = rng.randrange(self.W * T - 2)
            until = self.boundary(tau, t0) + int(rng.uniform(1.1, 2.6) * tau * MS)
            for t in range(t0, self.W * T):
                if self.boundary(tau, t) < until:
                    self.tick_late[(grid, t)] = until - self.boundary(tau, t) + (t - t0) * 3 * MARGIN
            if grid == "ingress":
                self.ticks_in = self.tick_times("ingress")
        elif path in ("backlog", "overrun"):
            bucket = self.r * self.B
            for _ in range(rng.randint(1, 2)):
                bound = bucket if path == "backlog" else 4 * bucket
                self.backlog[rng.randrange(self.W * self.T)] = bound + rng.choice([0, 1, 1 + rng.randrange(bucket)])
        elif path == "refused":
            x.refused = True
        elif path == "late-forward":
            by_slack = self.origin + x.release[-1] * self.tau_in * MS + self.slack
            by_bucket = self.origin + (x.anchor + 1) * self.tau * MS
            deadline = min(by_slack, by_bucket)
            if by_bucket + 2 * MS < by_slack and rng.random() < 0.6:  # late only by the end of its anchor's bucket
                x.delivered = self.at(by_bucket + rng.randrange(MS, by_slack - by_bucket - MS))
            elif rng.random() < 0.5:
                x.delivered = self.at(deadline + rng.randrange(MS, 30 * MS))
            else:
                x.connected = self.at(deadline + rng.randrange(MS, 30 * MS))
                x.delivered = self.at(x.connected + rng.randrange(2 * MARGIN, MS))
            x.times = [max(t, self.at(x.delivered + (k + 1) * 3 * MARGIN)) for k, t in enumerate(x.times)]
            x.end = self.at(max(x.end, x.times[-1] + 2 * MARGIN))
        elif path == "undelivered":
            x.delivered = None
            if rng.random() < 0.5:
                x.times, x.end = [], self.at(self.boundary(self.tau, self.W * self.T) - 5 * MS)
        elif path == "overflow-egress":
            self.Q = rng.randint(1, 3)
        elif path == "overflow-ingress":  # a burst of request frames inside one bucket, past Q and the allocation
            self.Q_in = rng.randint(1, 2)
            b = rng.randrange(self.W * self.T_in - 1)
            t = self.within(self.tau_in, b, 0.5, 0.9)
            burst = encode_frame(DATA, x.stream, b"more", self.B_in) * (self.Q_in + self.r_in + 1)
            self.extra.append((t, 0, ["link", t, burst.hex()]))
        elif path == "duplicate-at-capacity":  # a duplicate of a queued frame while the queue holds exactly Q
            reads = sorted((t, i, k) for i, y in enumerate(xs) for k, t in enumerate(y.times))
            bucket = lambda t: (t - self.origin) // (self.tau * MS)
            q = 2 if len(reads) > 1 and bucket(reads[0][0]) == bucket(reads[1][0]) and rng.random() < 0.5 else 1
            t, i, _ = reads[q - 1]
            _, j, k = rng.choice(reads[:q])
            self.Q = q
            dup = self.at(t + MARGIN)
            data = b"id: %d\ndata: dup\n\n" % xs[j].positions[k]
            self.extra.append((dup, 0, ["upstream", xs[i].stream, dup, data.hex()]))
        elif path == "early-tick":  # the request's tick read early, and its first event read before its anchor's bucket
            x = min(xs, key=lambda y: y.positions[0] if y.positions else 1 << 60)
            t = x.release[-1]
            target = self.boundary(self.tau, x.anchor)
            if target != self.boundary(self.tau_in, t) or not x.times:
                return
            self.ticks_in[t] = early = self.at(target - int(0.3 * self.tau_in * MS))
            if len(x.release) == 1 or x.release[0] == t:
                x.connected = self.at(early + rng.randrange(2 * MARGIN, MS))
            x.delivered = self.at(max(x.connected, early) + rng.randrange(2 * MARGIN, MS))
            x.times[0] = self.at(x.delivered + rng.randrange(3 * MARGIN, 2 * MS))
            # the choice that reads it, and refuses, is window ⌈(anchor + lag)/T⌉'s: one inside the run where a lag allows
            inside = [g for g in range(1, self.T) if -(-(x.anchor + g) // self.T) < self.W]
            if self.prover is not None and inside and self.prover["lag"] not in inside:
                self.prover["lag"] = rng.choice(inside)
        elif path == "slow-server":  # valid syncs that don't cover it: frames ready after their eligibility
            self.syncs = [ClockSync(min(self.deltas), max(self.rhos)) for _ in range(self.W)]
        elif path == "advice":
            w = rng.randrange(self.W)
            s = self.syncs[w]
            self.syncs[w] = ClockSync(max(self.deltas) + 1, s.rho) if rng.random() < 0.5 else \
                ClockSync(s.delta, next(r for r in RHOS + [Fraction(5, 2)] if r not in self.rhos))
        elif path in ("no-id", "duplicate", "long"):
            if not x.times:
                return
            k = rng.randrange(len(x.times))
            if path == "no-id":
                data = rng.choice([b"data: %d\n\n" % k, b"id:\ndata: %d\n\n" % k, b"id: x%d\n\n" % k, b"id: -%d\n\n" % k,
                                   b"ide: %d\n\n" % x.positions[k], b"data: 1\nid: \nid: %d\n\n" % x.positions[k]])
            elif path == "duplicate":
                other = rng.choice(xs)
                data = b"id: %d\ndata: dup\n\n" % rng.choice(other.positions)
            else:
                data = b"id: %d\ndata: " % x.positions[k] + b"x" * self.B + b"\n\n"
            x.replace = {x.times[k]: data}
        elif path == "no-request":
            x.parts = 1 if rng.random() < 0.5 else 2
            x.data_only = True
        elif path == "trailing":
            x.trailing = rng.choice([b"id: 99\ndata: cut", b"\n", b"x"])
        elif path == "link-garbage":
            for _ in range(rng.randint(1, 3)):
                kind = rng.randrange(5)
                b = rng.randrange(self.W * self.T_in)
                t = self.within(self.tau_in, b, 0.5, 0.9)
                if kind == 0:
                    data = encode_frame(PAD, 0, b"", self.B_in)
                elif kind == 1:
                    data = encode_frame(7, rng.choice(xs).stream, b"junk", self.B_in)
                elif kind == 2:
                    frame = bytearray(encode_frame(LAST, rng.choice(xs).stream, b"GET /long", self.B_in))
                    frame[5:9] = (self.B_in * 2).to_bytes(4, "big")
                    data = bytes(frame)
                elif kind == 3:
                    data = rng.randbytes(rng.randrange(1, self.B_in))
                else:
                    data = b""
                self.extra.append((t, 0, ["link", t, data.hex()]))
        else:
            raise ValueError(path)

    def configure_prover(self) -> None:
        rng = self.rng
        lag = rng.choice([1, self.T - 1, rng.randrange(1, self.T)])
        k, margin, fallback = rng.choice([(1, 0, False), (1, 0, False), (2, 1, False), (2, 0, True), (3, 1, False),
                                         (1, 1, True)])
        envelope = ClockSync(max(self.deltas), min(self.rhos)) if rng.random() < 0.85 else \
            ClockSync(max(self.deltas) + 1, min(self.rhos))
        prior = []
        if rng.random() < 0.3:
            for u in range(rng.randint(1, 2)):
                frontier, alive = [], True
                d = rng.choice(self.deltas)
                for _ in self.rhos:  # a larger ρ needs at least the same δ, and a None is followed only by Nones
                    alive = alive and rng.random() < 0.8
                    frontier.append([d, rng.randrange(40)] if alive else None)
                    d = rng.choice([x for x in self.deltas if x >= d])
                prior.append({"window": u, "sync": sync_json(rng.choice(self.syncs)), "frames": rng.randint(1, 4),
                              "late": rng.random() < 0.3, "frontier": frontier})
        self.prover = {"rule": [k, margin, fallback], "envelope": sync_json(envelope), "lag": lag, "prior": prior}
        self.windows = self.W
        self.syncs = []
        self.read_before_choices(lag)

    def read_before_choices(self, lag: int) -> None:
        """Each tick w·T − lag that starts a choice wakes late within the slack, and some of its bucket's response reads
        move before it: ready at w·T − lag + 1, read by the choice's start, and not ready by the bucket it reads to."""
        rng = self.rng
        for w in range(1, self.W):
            t = w * self.T - lag
            if ("egress", t) in self.tick_late:  # a perturbation's late tick or stall
                continue
            late = rng.randrange(self.slack // 2, self.slack - 2 * MARGIN)
            self.tick_late[("egress", t)] = late
            lo, hi = self.boundary(self.tau, t), self.boundary(self.tau, t) + late - 3 * MARGIN
            for x in self.sessions:
                for k, s in enumerate(x.times):
                    prev = x.times[k - 1] if k else x.delivered
                    if prev is None or not lo <= s < self.boundary(self.tau, t + 1) or rng.random() < 0.4:
                        continue
                    new = self.at(max(lo, prev) + rng.randrange(3 * MARGIN, 6 * MARGIN))
                    if new < min(hi, s):
                        x.times[k] = new
                        if s in x.replace:
                            x.replace[new] = x.replace.pop(s)


def instance(seed: int) -> dict:
    rng = random.Random(seed)
    path = PATHS[seed % len(PATHS)]
    t = Trace(rng, path)
    t.build()
    if path == "chaos":
        for p in rng.sample(PERTURBATIONS, rng.randint(2, 3)):
            t.perturb(p)
        if rng.random() < 0.3:
            t.configure_prover()
    elif path == "prover":
        t.configure_prover()
    elif path == "early-tick":
        t.configure_prover()
        t.perturb(path)
    elif path != "honest":
        t.perturb(path)
    return {"path": path, "config": t.config(), "events": t.events(),
            "program": t.program() if path == "honest" else None}


def check_honest(inst: dict, results: dict) -> None:
    """An honest run is accepted, and its egress records are the Program's grids under the syncs it declared."""
    c = inst["config"]
    egress = grid_params(c["egress"])
    records = [a[1] for a in results["actions"] if a[0] == "publish"]
    if any(r["status"] is not None for r in records):
        raise AssertionError(f"an honest run faulted: {[(r['link'], r['window'], r['status']) for r in records]}")
    frames = [LogicalFrame(bytes.fromhex(p), a, i) for p, a, i in inst["program"]]
    want = expected_grids(frames, [sync_of(s) for s in results["syncs"]], egress)
    got = [r["grid"] for r in records if r["link"] == c["egress_link"]]
    if got != [[[None if x is None else x.hex() for x in row] for row in g] for g in want]:
        raise AssertionError("an honest run's egress grids aren't the Program's")


def line(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """Both files for instances 0 … count − 1 under ``out/difftest``."""
    d = out / "difftest"
    d.mkdir(parents=True, exist_ok=True)
    paths = {"difftest/instances.jsonl": d / "instances.jsonl", "difftest/python.jsonl": d / "python.jsonl"}
    with open(paths["difftest/instances.jsonl"], "w") as fi, open(paths["difftest/python.jsonl"], "w") as fp:
        for seed in range(count):
            inst = instance(seed)
            results = python_results(inst)
            if inst["program"] is not None:
                check_honest(inst, results)
            fi.write(line(inst))
            fp.write(line(results))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.core.protocols.certifier is tested")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    here = Path(verity.core.protocols.certifier.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.core.protocols.certifier comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
