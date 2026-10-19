"""An active certifier on a live byte stream: an asyncio TCP proxy that runs the reference's certifiers on one boundary link.

The proxy is the I/O shell around the reference. It owns a monotonic clock and a ticker per grid, socket reads and
writes, and the framing of byte streams into B-byte frames. Everything that decides a grid is the reference's:
`schedule.eligible` and `schedule.run_shaper`'s handing rule, and `grid.Certifier` on egress (at r) and on ingress (at r_in).
Its records are `grid.Record`s, published raw and through `commitment.commit`, which `audit.Audit.check_link` and
`check_ingress_link` read.

**Topology.** One TCP connection is the boundary link, the wire, between the proxy and the outside peer. Inside, the
proxy opens one TCP connection per stream to the server, and writes it the stream's request as the ingress certifier
releases it. Every frame on the link is B bytes: a header (kind u8, stream u32, length u32), the payload, and zero
fill. Padding is B zero bytes (kind `PAD`); `DATA` is a payload, and `LAST` the last frame of a request.

**Egress.** The server's bytes on a stream are cut after every `\\n\\n`, the end of a server-sent event, so a frame's
content is fixed by the bytes and never by how they were read; an HTTP response head goes with the first event. The
event's `id:` field is its position in the Program's canonical order on the link. A frame's anchor is the egress bucket
in which the ingress certifier released its stream's `LAST` frame (the request is recorded there), and its logical index
is its position in its stream. A frame read at time s is ready at bucket ⌊s/τ_b⌋ + 1, the first that starts after it.
At bucket t the shaper hands, in canonical order, each frame whose `max(h(k), ready(k), last)` is at most t, where
`h(k) = max(eligible(k), h(k−1))` under the declared sync of its anchor's window (`run_shaper`'s rule). `Certifier.tick`
then releases min(queue, r) of them, left-packed, and the proxy writes those and r − n padding frames. A frame handed at
a bucket after h(k) is late: the declared sync didn't cover it, and the window it is handed in fails closed.

**Ingress.** A frame read from the link at time s is enqueued at bucket ⌈s/τ_in⌉ (`calibration.size_ingress_ms`'s
model) and released by the ingress `Certifier` at r_in into its stream's server connection.

**The salt** (`Proxy(salt=…)`, off by default) is `salt_bits` bits (512 by default) per PoUW window from the
certifier's own randomness (`fresh_salt`), released into the block only. PoUW window p is the `pouw_window` = k
certifier windows p·k … p·k + k − 1 (k = 1 by default). At p's first ingress tick, t_s, the certifier draws p's salt
and enqueues its own salt frame ahead of that bucket's arrivals (`Bucketer(first=…)`): kind `SALT` on `SALT_STREAM`,
its payload u64be p followed by `PLACEHOLDER`, 64 zero bytes. The frame and its slot are fixed by the window and the
traffic, so every record is a function of the traffic alone and none depends on the salt. When the ingress certifier
releases that frame (the `Bucketer` knows it by its place in the queue, so a link frame of the same bytes is never taken
for it), the proxy writes the salt frame with the salt in place of the placeholder (the header, p and the salt, no fill)
on a server connection of its own, which it never reads: nothing the server writes there reaches egress. The egress
side never sees the salt.

The certifier's record cells are its grids and one sealed slot per PoUW window, `_salts[p]`, which holds the salt and
is opened only by `reveal`. So the record's open cells (`Accounting.Attribution.openCells`) are the grids, a function
of the traffic, and what the certifier sends into the block is a function of its record. `reveal(p, registration)`
takes the PoUW window's registration as the proof service returned it (`sampled_proofs.service.Registration`) and fails
closed without it: it refuses unless the receipt is that registration's (`one_stage.registration.receipt`) and
`registered` reads p off the registration, while p's last ingress record is unpublished (t_d), and for a window whose
salt was never drawn. `check_salts` is the audit's read of it: p's first raw ingress record holds exactly one salt
frame of p, the placeholder's, and the revealed salt has `salt_bits` bits.

**Back from maintenance** (`Proxy(start=k, position=p)`, off by default): a certifier that was down serves from window k
of the trusted clock on. Its first tick of each grid is the window's first bucket, its first records are window k's, and
the first egress frame it takes is the Program's at position p, the frames before it being those of the windows before.
It carries nothing from before: a window it was down for has no record. The windows before k are another proxy's, the
same certifier's before it went down (`windows`, which ends a proxy's run early), or none.

**Advice.** The proxy runs the windows of its declared syncs, one per window, which the prover gives before the run,
or which an online prover (`online.Prover`, with a lag of 1 … T − 1 buckets) declares at each window's first tick. A
frame anchored past them is never handed, nor is any frame behind it (`schedule.handing_buckets`). The online
choice for window w costs hundreds of ms on a live configuration, against a slack of tens, so it is never made on a
tick: right after tick w·T − lag the proxy hands what it reads (`Prover.snapshot`, the frames ready by then and the
earlier declarations) to a worker process, off its event loop and its GIL, and the tick at w·T declares the result
if it is done, else the envelope (`LaggedChoices`). Window 0's is started before the first tick. The envelope is in
Σ_sync, so an unready choice changes nothing the audit checks or charges: a declaration is advice, checked only for
membership in Σ_sync, and the bound charges log2 #Σ_sync bits per window whatever the value.

**Faults fail closed** through `Certifier.fault`: the rest of the window is padding, the queue is dropped, and the status
names the first reason (W6). Since the Program's grid carries a queue into the next window, an egress window that
closes faulted while its queue would still hold frames faults the next one too, with the same reason. They are:
- a late tick (`monitor-unavailable`): the ticker woke more than `slack` after its boundary. A bucket it missed is
  still emitted, as padding, so every window keeps its T rows and no bucket slips into the next;
- a declaration the online prover refuses (`monitor-unavailable`, `online.Refused`): a frame read for the window's
  choice, at tick w·T − lag, is ready no later than its anchor's bucket, which an ingress tick read before its boundary
  makes. The egress window that declares is faulted, and its sync is the choice made once those frames are read;
- an overrun: the link still holds a whole bucket unwritten at a tick (`delivery-refused`), or the queue or the
  shaper's buffer passes Q (`overflow`); a server that refuses a stream's connection is `delivery-refused` on ingress;
- a late forward (`delivery-refused`): a released request not written to its server connection by `slack` after its
  release tick, nor by the end of its anchor's egress bucket. The server answers it late, so its events, and those
  the server numbers after them, aren't the Program's. Its ingress window is faulted if still open, and every egress
  window from the check, at an egress tick before handing, until the stream has ended upstream and every frame that
  arrived before its end is handed;
- an advice violation: a window's declared sync outside Σ_sync (`coverage-never-established`), checked at the window's
  start; the grid still follows the declared sync, as the audit's expected grid does;
- a late frame (`coverage-never-established`): a frame handed at a bucket t after its h(k), because its bytes were
  ready after h(k), a frame before it was handed late (`last`), or it was read after bucket h(k)'s tick. The sync
  declared for its anchor's window didn't cover the response, so the Program's grid holds the frame earlier than the
  certifier could: the window of bucket t is faulted, and the frame is dropped with the queue. A window closed before
  the frame was handed can't be faulted any more; its record is the Program's on the frames handed by its close;
- a malformed frame (`malformed`): an event longer than a frame, without a position or with one already seen, on a
  stream with no released request, or bytes left unframed at the stream's end.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import math
import multiprocessing
import os
import resource
import secrets
import struct
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import dataclass, field

from verity.core.protocols.certifier.audit import Audit, Verdict
from verity.core.protocols.certifier.grid import (
    GridParams,
    Reason,
    Record,
    Row,
    Certifier,
    row_count,
)
from verity.core.protocols.certifier.online import Prover, Refused
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, eligible
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.core.protocols.sampled_proofs.service import Registration

HEADER = struct.Struct(">BII")
PAD, DATA, LAST, SALT = 0, 1, 2, 3
EVENT_END = b"\n\n"
SALT_STREAM = 0xFFFF_FFFF
SALT_BITS = 512
SALT_BYTES = SALT_BITS // 8
PLACEHOLDER = bytes(64)
SALT_FRAME_BYTES = HEADER.size + 8 + len(PLACEHOLDER)


def encode_frame(kind: int, stream: int, payload: bytes, frame_bytes: int) -> bytes:
    """One B-byte link frame: the header, the payload, then zero fill."""
    if HEADER.size + len(payload) > frame_bytes:
        raise ValueError(f"a {len(payload)}-byte payload doesn't fit a {frame_bytes}-byte frame")
    return HEADER.pack(kind, stream, len(payload)) + payload + bytes(frame_bytes - HEADER.size - len(payload))


def decode_frame(frame: bytes) -> tuple[int, int, bytes]:
    """(kind, stream, payload) of a link frame; padding is (`PAD`, 0, b"")."""
    kind, stream, n = HEADER.unpack_from(frame)
    return kind, stream, bytes(frame[HEADER.size : HEADER.size + n])


def fresh_salt(window: int, nbytes: int = SALT_BYTES) -> bytes:
    """PoUW window `window`'s salt, `nbytes` bytes of the certifier's own randomness. A source of coins handed to the
    certifier has the same signature."""
    return secrets.token_bytes(nbytes)


def salt_frame(window: int, value: bytes, frame_bytes: int) -> bytes:
    """PoUW window `window`'s salt frame, carrying `value`: `PLACEHOLDER` in the grid, the salt on the server's
    connection."""
    return encode_frame(SALT, SALT_STREAM, window.to_bytes(8, "big") + value, frame_bytes)


def served_window(registration: Registration) -> int | None:
    """The window a registration's provenance names when it is exactly `{"kind": "served", "window": w}` (the network
    certifier's own, `service.LinkWindows.provenance`), else None."""
    window = registration.record.get("window") if isinstance(registration.record, Mapping) else None
    if isinstance(window, Mapping) and set(window) == {"kind", "window"} and window["kind"] == "served" and \
            type(window["window"]) is int:
        return window["window"]
    return None


def event_position(event: bytes) -> int | None:
    """The event's `id:` field, its canonical position on the link, or None."""
    for line in event.split(b"\n"):
        if line.startswith(b"id:"):
            value = line[3:].strip()
            return int(value) if value.isdigit() else None
    return None


class Events:
    """Cuts a byte stream after every `\\n\\n`. The cuts depend only on the bytes, not on how they were read."""

    def __init__(self) -> None:
        self.buffer = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        self.buffer += data
        out = []
        while (i := self.buffer.find(EVENT_END)) >= 0:
            out.append(bytes(self.buffer[: i + len(EVENT_END)]))
            del self.buffer[: i + len(EVENT_END)]
        return out


class Shaper:
    """The honest shaper and the egress certifier on one link, fed frames as they arrive (no I/O).

    `arrive` takes a frame at its canonical position with its ready bucket; `tick` hands what is due at the current
    bucket by `run_shaper`'s rule, ticks the certifier and returns its row, with the window's record at its last bucket.
    Fed every frame at its ready bucket, `hands` is `run_shaper`'s hand times, and the records are `run_shaper`'s while
    no frame is late. A frame handed after its h(k) faults the window it is handed in `coverage-never-established`.

    `queued` is what the certifier's queue would hold had no fault dropped it. A window that closes faulted with frames
    still queued faults the next one too, with the same reason: the Program's grid carries them over.

    With a `prover`, the syncs after the given ones are its: it sees every frame the shaper takes, and declares each of
    the first `windows` windows' syncs at the window's first tick, before anything is handed. The proxy's is
    `LaggedChoices`, whose `declare` only takes a choice already made; an `online.Prover` chooses at the tick. A
    declaration it refuses (`online.Refused`) faults the window `monitor-unavailable`, and the next one, on the frames
    it has read, is the window's.

    With `start`, it begins at window `start`'s first bucket, its certifier at that window, and the first frame it
    takes is the one at `position`: a shaper back from maintenance, which carries nothing from before."""

    def __init__(self, certifier: Certifier, syncs: Sequence[ClockSync], sync_set: SyncSet,
                 prover: Prover | LaggedChoices | None = None, windows: int | None = None, start: int = 0,
                 position: int = 0) -> None:
        self.certifier, self.syncs, self.sync_set = certifier, list(syncs), sync_set
        self.prover, self.windows = prover, len(self.syncs) if windows is None else windows
        if start:
            certifier.window = start
        self.pending: dict[int, tuple[LogicalFrame, int]] = {}
        self.next = position
        self.h = self.last = self.queued = 0
        self.t = start * certifier.params.buckets
        self.hands: list[int] = []
        self._check_advice()

    def _check_advice(self) -> None:
        w = self.certifier.window
        if w < len(self.syncs) and self.syncs[w] not in self.sync_set:
            self.certifier.fault(Reason.COVERAGE_NEVER_ESTABLISHED)

    def arrive(self, position: int, frame: LogicalFrame, ready: int) -> None:
        if position < self.next or position in self.pending:
            self.certifier.fault(Reason.MALFORMED)
        elif len(self.pending) >= self.certifier.params.queue_capacity:
            self.certifier.fault(Reason.OVERFLOW)
        else:
            self.pending[position] = (frame, ready)
            if self.prover is not None:
                self.prover.arrive(0, position, frame, ready)

    def tick(self) -> tuple[Row, Record | None]:
        T = self.certifier.params.buckets
        if self.prover is not None and self.t % T == 0 and self.t // T == len(self.syncs) < self.windows:
            try:
                sync = self.prover.declare()
            except Refused:
                self.certifier.fault(Reason.MONITOR_UNAVAILABLE)
                sync = self.prover.declare()
            self.syncs.append(sync)
            self._check_advice()
        while self.next in self.pending:
            frame, ready = self.pending[self.next]
            w = frame.anchor // T
            if w >= len(self.syncs):
                break
            h = max(eligible(frame, self.syncs[w]), self.h)
            hand = max(h, ready, self.last)
            if hand > self.t:
                break
            del self.pending[self.next]
            if self.t > h:
                self.certifier.fault(Reason.COVERAGE_NEVER_ESTABLISHED)
            self.h, self.last, self.next = h, hand, self.next + 1
            self.hands.append(hand)
            self.certifier.enqueue(frame.payload)
            self.queued += 1
        row = self.certifier.tick()
        self.queued -= min(self.queued, self.certifier.params.allocation)
        self.t += 1
        record = None
        if self.t % T == 0:
            record = self.certifier.close_window()
            if self.queued and not record.status.complete:
                self.certifier.fault(record.status.reason)
            self._check_advice()
        return row, record


class Bucketer:
    """The ingress certifier on one link, fed frames as they arrive (no I/O): a frame arriving at bucket time s is
    `arrive`d with ⌈s⌉ and enqueued at that tick; `tick` returns the released row, with the record at a window's end.
    With `first`, each window's first tick enqueues the certifier's own frame `first(window)`, if it gives one, ahead of
    that bucket's arrivals. The bucketer follows it by its place in the queue, never by its bytes: `own` maps each slot
    of the row last released that holds one to its window, and one a fault drops is never released. With `start`, it
    begins at window `start`'s first bucket, its certifier at that window."""

    def __init__(self, certifier: Certifier, first: Callable[[int], bytes | None] | None = None,
                 start: int = 0) -> None:
        self.certifier = certifier
        self.first = first
        if start:
            certifier.window = start
        self.waiting: deque[tuple[int, bytes]] = deque()
        self.t = start * certifier.params.buckets
        self.own: dict[int, int] = {}
        self._own: list[tuple[int, int]] = []  # (place in the queue, window) of each own frame queued

    def arrive(self, frame: bytes, bucket: int) -> None:
        self.waiting.append((bucket, frame))

    def tick(self) -> tuple[Row, Record | None]:
        c, T = self.certifier, self.certifier.params.buckets
        if self.first is not None and self.t % T == 0 and (frame := self.first(self.t // T)) is not None:
            self._own.append((c.queued, self.t // T))
            c.enqueue(frame)
        while self.waiting and self.waiting[0][0] <= self.t:
            c.enqueue(self.waiting.popleft()[1])
        row = c.tick()
        if self._own:
            n = row_count(row)
            # a fault empties the queue, and its window enqueues nothing after it
            kept = [(place, w) for place, w in self._own if place < c.queued + n]
            self.own = {place: w for place, w in kept if place < n}
            self._own = [(place - n, w) for place, w in kept if place >= n]
        elif self.own:
            self.own = {}
        self.t += 1
        return row, c.close_window() if self.t % T == 0 else None


def _choose(snapshot: Prover) -> tuple[ClockSync, float]:
    t0 = time.perf_counter()
    sync = snapshot.choose()
    return sync, round((time.perf_counter() - t0) * 1000, 1)


def reserve_fds(limit: int = 1 << 16) -> None:
    """Grow this process's fd table now, to its open-files limit or `limit` if less. Linux grows it on demand, and once
    the process has a second thread (an executor's, a publisher's) each growth waits for an RCU grace period, tens of
    ms, inside the `socket()` that needed it: live, the first burst of upstream connections then missed its slack."""
    soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    top = (limit if soft == resource.RLIM_INFINITY else min(soft, limit)) - 1
    fd = os.open(os.devnull, os.O_RDONLY)
    try:
        if top > fd:
            os.dup2(fd, top)
            os.close(top)
    finally:
        os.close(fd)


def worker() -> ProcessPoolExecutor:
    """The proxy's worker for its online choices: one spawned process, started before it returns, so that its
    start-up imports don't share window 0 with the proxy's forwards. A thread would hold the proxy's GIL."""
    pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    pool.submit(int).result()
    return pool


class InlineExecutor(Executor):
    """An executor that runs each call at `submit`, for a driver on a `VirtualClock`, where no tick waits for it: every
    choice is ready."""

    def submit(self, fn, /, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as err:
            future.set_exception(err)
        return future


class LaggedChoices:
    """The online prover's declarations off the proxy's tick (no I/O). `start()`, right after tick w·T − lag (window
    0's before the first tick), submits window w's choice on `prover.snapshot()` to `executor`; `declare()` at tick w·T
    declares it if it is done, else the envelope, without waiting. `log` has the lag, the windows declared the envelope
    for want of a choice (`unready`) and each choice's time in ms (`choose_ms`)."""

    def __init__(self, prover: Prover, executor: Executor) -> None:
        if not 0 < prover.lag < prover.T:
            raise ValueError("the proxy's online prover chooses off its tick: a lag of 1 … T − 1 buckets")
        self.prover, self.executor = prover, executor
        self._next: tuple[int, Future] | None = None
        self._refused: int | None = None
        self.log: dict = {"lag": prover.lag, "unready": [], "choose_ms": []}

    def arrive(self, link: int, position: int, frame: LogicalFrame, ready: int) -> None:
        self.prover.arrive(link, position, frame, ready)

    def start(self) -> None:
        w = len(self.prover.syncs)
        try:
            snapshot = self.prover.snapshot()
        except Refused:
            self._refused = w
            snapshot = self.prover.snapshot()
        self._next = (w, self.executor.submit(_choose, snapshot))

    def declare(self) -> ClockSync:
        """Window w's declaration; `Refused` once if its snapshot was, and then the choice made on what it read."""
        w, sync = len(self.prover.syncs), None
        if self._refused == w:
            self._refused = None
            raise Refused("a frame is ready only after its anchor's bucket")
        if self._next is not None and self._next[0] == w and self._next[1].done():
            sync, ms = self._next[1].result()
            self.log["choose_ms"].append(ms)
        else:
            self.log["unready"].append(w)
        self._next = None
        return self.prover.declare(self.prover.envelope if sync is None else sync)


class VirtualClock:
    """A clock its caller moves, for runs driven event by event (tests, and replays without the host's timer):
    `clock()` reads it, `sleep` waits until `advance` passes the deadline."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self._waiters: list[tuple[float, int, asyncio.Future]] = []
        self._seq = itertools.count()

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        if delay <= 0:
            await asyncio.sleep(0)
            return
        future = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (self.now + delay, next(self._seq), future))
        await future

    def advance(self, to: float) -> None:
        self.now = max(self.now, to)
        while self._waiters and self._waiters[0][0] <= self.now:
            future = heapq.heappop(self._waiters)[2]
            if not future.done():
                future.set_result(None)


@dataclass
class _Stream:
    anchor: int | None = None
    count: int = 0
    out: asyncio.Queue = field(default_factory=asyncio.Queue)
    events: Events = field(default_factory=Events)
    task: asyncio.Task | None = None
    window: int | None = None  # the ingress window of its release
    deadline: float | None = None  # when its request must be written upstream
    delivered: float | None = None  # when it was
    ended: int | None = None  # the ready bucket of its end upstream


class Proxy:
    """The active certifier on one link: `run(reader, writer)` serves the link connection from `origin`, the clock's time
    of bucket 0, for the windows of `syncs`, or `windows` of them declared by an online `prover`, then closes it.

    - the prover's choices (`LaggedChoices`) run on `executor`, by default a `worker()` process, which the proxy shuts
      down when `run` ends; on a `VirtualClock`, pass `InlineExecutor()`;
    - `publish(record, params)` receives every window's record, egress and ingress, as it closes (by default they are
      kept in `records`);
    - `frames` logs every egress frame read, as (position, stream, index, anchor, ready), and `shaper.hands` its hand
      bucket by position; `lateness` holds each tick's lateness in seconds, per grid; `link_frames_read` and
      `requests_released` count the ingress frames read and the requests released to the server;
    - with `salt`, a source of one salt of `salt_bits` bits per PoUW window of `pouw_window` certifier windows, such as
      `fresh_salt`, the ingress grid carries each PoUW window's salt frame at its first bucket, the server receives
      the salt on its own connection (`salts_released` and `salts_delivered` count them), and `reveal(p,
      registration)` gives the verifier PoUW window p's salt once p is registered (`registered` reads the window off
      a registration, by default `served_window`) and its last ingress record is published;
    - with `start` and `position`, it is back from maintenance: it serves windows `start` … `windows` − 1, its first
      egress frame the Program's at `position`;
    - `until(predicate)` waits until the proxy's state satisfies it, for a caller that drives a `VirtualClock`."""

    def __init__(self, upstream: tuple[str, int], egress: GridParams, ingress: GridParams, sync_set: SyncSet,
                 syncs: Sequence[ClockSync], *, origin: float, slack: float = 0.025,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 publish: Callable[[Record, GridParams], None] | None = None,
                 egress_link: str = "responses", ingress_link: str = "requests",
                 prover: Prover | None = None, windows: int | None = None, executor: Executor | None = None,
                 salt: Callable[[int, int], bytes] | None = None, salt_bits: int = SALT_BITS, pouw_window: int = 1,
                 registered: Callable[[Registration], int | None] = served_window, start: int = 0,
                 position: int = 0) -> None:
        if (egress.buckets * egress.bucket_ms) != (ingress.buckets * ingress.bucket_ms):
            raise ValueError("the egress and ingress grids have windows of the same length")
        if prover is not None and syncs:
            raise ValueError("an online prover declares every window's sync")
        if salt is not None and ingress.frame_bytes < SALT_FRAME_BYTES:
            raise ValueError(f"a salt frame needs ingress frames of at least {SALT_FRAME_BYTES} bytes")
        if not (type(salt_bits) is int and salt_bits > 0 and salt_bits % 8 == 0):
            raise ValueError(f"a salt is a positive whole number of bytes, not {salt_bits!r} bits")
        if not (type(pouw_window) is int and pouw_window >= 1):
            raise ValueError(f"a PoUW window is a positive number of certifier windows, not {pouw_window!r}")
        self.upstream, self.egress, self.ingress = upstream, egress, ingress
        self.origin, self.slack, self.clock, self.sleep = origin, slack, clock, sleep
        self.windows = len(syncs) if windows is None else windows
        if not 0 <= start <= self.windows or position < 0:
            raise ValueError(f"a proxy starts at one of windows 0 … {self.windows}, at a position ≥ 0")
        if start and prover is not None:
            raise ValueError("an online prover declares from window 0 on: a proxy back from maintenance takes syncs")
        self.start = start
        reserve_fds()
        self.choices = None if prover is None else LaggedChoices(prover, executor or worker())
        self.shaper = Shaper(Certifier(egress_link, egress), syncs, sync_set, self.choices, self.windows, start,
                             position)
        self.salt, self.salt_bits, self.pouw_window, self.registered = salt, salt_bits, pouw_window, registered
        self._salts: dict[int, bytes] = {}
        self._salt_out: asyncio.Queue | None = None
        self._salt_task: asyncio.Task | None = None
        self.salts_released = self.salts_delivered = 0
        self.bucketer = Bucketer(Certifier(ingress_link, ingress), None if salt is None else self._draw_salt, start)
        self.records: list[Record] = []
        self.publish = publish or (lambda record, params: self.records.append(record))
        self.streams: dict[int, _Stream] = {}
        self.frames: list[tuple[int, int, int, int, int]] = []
        self.lateness: dict[str, list[float]] = {"egress": [], "ingress": []}
        self.link_frames_read = self.requests_released = 0
        self._released: list[_Stream] = []
        self._late: list[_Stream] = []
        self._changed = asyncio.Event()

    async def until(self, predicate: Callable[[Proxy], bool]) -> None:
        while not predicate(self):
            self._changed.clear()
            await self._changed.wait()

    async def run(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        link = asyncio.create_task(self._read_link(reader))
        try:
            await asyncio.gather(self._egress(writer), self._ingress())
        finally:
            link.cancel()
            tasks = [st.task for st in self.streams.values() if st.task] + [t for t in (self._salt_task,) if t]
            for task in tasks:
                task.cancel()
            await asyncio.gather(link, *tasks, return_exceptions=True)
            writer.close()
            if self.choices is not None:
                await asyncio.to_thread(self.choices.executor.shutdown, True, cancel_futures=True)
            self._changed.set()

    def _bucket_ready(self, now: float) -> int:
        return math.floor((now - self.origin) * 1000 / self.egress.bucket_ms) + 1

    async def _ticks(self, params: GridParams, lateness: list[float], certifier: Certifier):
        tau = params.bucket_ms / 1000
        for t in range(self.start * params.buckets, self.windows * params.buckets):
            deadline = self.origin + t * tau
            await self.sleep(deadline - self.clock())
            late = self.clock() - deadline
            lateness.append(late)
            if late > self.slack:
                certifier.fault(Reason.MONITOR_UNAVAILABLE)
            yield t

    async def _egress(self, writer: asyncio.StreamWriter) -> None:
        p = self.egress
        bucket_bytes = p.allocation * p.frame_bytes
        padding = memoryview(bytes(bucket_bytes))
        choices = self.choices
        if choices is not None and self.windows:
            choices.start()
        async for t in self._ticks(p, self.lateness["egress"], self.shaper.certifier):
            backlog = writer.transport.get_write_buffer_size()
            if backlog > bucket_bytes:
                self.shaper.certifier.fault(Reason.DELIVERY_REFUSED)
            self._check_forwarding()
            row, record = self.shaper.tick()
            n = row_count(row)
            if backlog <= 4 * bucket_bytes:
                writer.writelines([*row[:n], padding[: (p.allocation - n) * p.frame_bytes]])
            if choices is not None and (t + choices.prover.lag) % p.buckets == 0 and \
                    (t + choices.prover.lag) // p.buckets < self.windows:
                choices.start()
            if record is not None:
                self.publish(record, p)
            self._changed.set()
        await writer.drain()

    def _check_forwarding(self) -> None:
        """Fault every window a late forward can reach: a released request not written upstream by its deadline makes
        its ingress window delivery-refused, if still open, and the egress windows too until the stream has ended and
        every frame that arrived before its end is handed, since the server's events can't be the Program's before."""
        now, waiting = self.clock(), []
        for st in self._released:
            if st.delivered is None and now < st.deadline:
                waiting.append(st)
            elif st.delivered is None or st.delivered > st.deadline:
                if self.bucketer.certifier.window == st.window:
                    self.bucketer.certifier.fault(Reason.DELIVERY_REFUSED)
                self._late.append(st)
        self._released = waiting
        self._late = [st for st in self._late if not self._settled(st)]
        if self._late:
            self.shaper.certifier.fault(Reason.DELIVERY_REFUSED)

    def _settled(self, st: _Stream) -> bool:
        return st.ended is not None and self.shaper.t >= st.ended and \
            all(ready > st.ended for _, ready in self.shaper.pending.values())

    async def _ingress(self) -> None:
        p = self.ingress
        tau, tau_e = p.bucket_ms / 1000, self.egress.bucket_ms / 1000
        async for t in self._ticks(p, self.lateness["ingress"], self.bucketer.certifier):
            row, record = self.bucketer.tick()
            anchor = t * p.bucket_ms // self.egress.bucket_ms
            own = self.bucketer.own
            for i, frame in enumerate(row[: row_count(row)]):
                if i in own:
                    self._release_salt(own[i] // self.pouw_window)
                    continue
                kind, s, payload = decode_frame(frame)
                if kind not in (DATA, LAST):
                    continue
                st = self._stream(s)
                st.out.put_nowait((payload, kind == LAST))
                if kind == LAST and st.anchor is None:
                    st.anchor, st.window = anchor, t // p.buckets
                    st.deadline = self.origin + min(t * tau + self.slack, (anchor + 1) * tau_e)
                    self._released.append(st)
                    self.requests_released += 1
            if record is not None:
                self.publish(record, p)
            self._changed.set()

    def _draw_salt(self, window: int) -> bytes | None:
        """At a PoUW window's first certifier window, draw its salt and give the certifier's own frame, the
        placeholder's."""
        if window % self.pouw_window:
            return None
        p, n = window // self.pouw_window, self.salt_bits // 8
        salt = self.salt(p, n)
        if len(salt) != n:
            raise ValueError(f"the salt source gave {len(salt)} bytes, not {n}")
        self._salts[p] = salt
        return salt_frame(p, PLACEHOLDER, self.ingress.frame_bytes)

    def _release_salt(self, p: int) -> None:
        """Queue PoUW window p's salt for the server, its frame being released. A frame of kind `SALT` read from the
        link is never the certifier's, and is left alone like any frame of another kind."""
        self.salts_released += 1
        if self._salt_out is None:
            self._salt_out = asyncio.Queue()
            self._salt_task = asyncio.create_task(self._salt_upstream(self._salt_out))
        salt = self._salts[p]
        self._salt_out.put_nowait(salt_frame(p, salt, HEADER.size + 8 + len(salt)))

    async def _salt_upstream(self, out: asyncio.Queue) -> None:
        """The salt stream's server connection: written, never read, so that nothing on it reaches egress."""
        try:
            _, writer = await asyncio.open_connection(*self.upstream)
        except OSError:
            self.bucketer.certifier.fault(Reason.DELIVERY_REFUSED)
            return
        try:
            while True:
                writer.write(await out.get())
                await writer.drain()
                self.salts_delivered += 1
                self._changed.set()
        finally:
            writer.close()

    def reveal(self, window: int, registration: Registration | None) -> bytes:
        """PoUW window `window`'s salt, for the verifier, given the window's registration as the proof service returned
        it. It fails closed (`RuntimeError`): without a registration whose receipt is its own and off which `registered`
        reads `window`, while the window's last ingress record is unpublished (t_d), and for a window whose salt wasn't
        drawn (one this proxy was down for at its start)."""
        if self.salt is None:
            raise ValueError("the salt is off")
        if not self._registers(window, registration):
            raise RuntimeError(f"no registration of window {window} with its receipt, so its salt isn't revealed")
        if (window + 1) * self.pouw_window > self.bucketer.certifier.window:
            raise RuntimeError(f"window {window}'s ingress records aren't all published, so its salt isn't revealed")
        if window not in self._salts:
            raise RuntimeError(f"window {window}'s salt wasn't drawn")
        return self._salts[window]

    def _registers(self, window: int, registration: Registration | None) -> bool:
        if not isinstance(registration, Registration) or not isinstance(registration.receipt, Mapping):
            return False
        try:
            digest = R.digest(registration.record)
        except (AttributeError, TypeError, ValueError):
            return False
        receipt = registration.receipt
        return receipt.get("format") == R.RECEIPT and receipt.get("registration") == digest == registration.digest \
            and self.registered(registration) == window

    async def _read_link(self, reader: asyncio.StreamReader) -> None:
        size, buffer = self.ingress.frame_bytes, bytearray()
        while data := await reader.read(1 << 16):
            bucket = math.ceil((self.clock() - self.origin) * 1000 / self.ingress.bucket_ms)
            buffer += data
            whole = len(buffer) - len(buffer) % size
            for i in range(0, whole, size):
                self.bucketer.arrive(bytes(buffer[i : i + size]), bucket)
                self.link_frames_read += 1
            del buffer[:whole]
            self._changed.set()

    def _stream(self, s: int) -> _Stream:
        if s not in self.streams:
            st = self.streams[s] = _Stream()
            st.task = asyncio.create_task(self._upstream(s, st))
        return self.streams[s]

    async def _upstream(self, s: int, st: _Stream) -> None:
        try:
            reader, writer = await asyncio.open_connection(*self.upstream)
        except OSError:
            self.bucketer.certifier.fault(Reason.DELIVERY_REFUSED)
            st.ended = self._bucket_ready(self.clock())
            return
        writer.transport.set_write_buffer_limits(0)

        async def pump() -> None:
            while True:
                payload, last = await st.out.get()
                writer.write(payload)
                if last and st.delivered is None:
                    await writer.drain()
                    st.delivered = self.clock()

        pumping = asyncio.create_task(pump())
        try:
            while data := await reader.read(1 << 16):
                ready = self._bucket_ready(self.clock())
                for event in st.events.feed(data):
                    self._egress_event(s, st, event, ready)
                self._changed.set()
            if st.events.buffer:
                self.shaper.certifier.fault(Reason.MALFORMED)
        finally:
            st.ended = self._bucket_ready(self.clock())
            pumping.cancel()
            writer.close()

    def _egress_event(self, s: int, st: _Stream, event: bytes, ready: int) -> None:
        position = event_position(event)
        if position is None or st.anchor is None or HEADER.size + len(event) > self.egress.frame_bytes:
            self.shaper.certifier.fault(Reason.MALFORMED)
            return
        frame = LogicalFrame(encode_frame(DATA, s, event, self.egress.frame_bytes), st.anchor, st.count)
        self.frames.append((position, s, st.count, st.anchor, ready))
        st.count += 1
        self.shaper.arrive(position, frame, ready)


def check_salts(audit: Audit, link: str, records: Sequence[Record], windows: int, salts: Mapping[int, bytes], *,
                pouw_window: int = 1, salt_bits: int = SALT_BITS) -> list[Verdict]:
    """The audit's read of the salt on ingress link `link` for certifier windows 0 … windows − 1, given the salts
    `reveal` gave by PoUW window of `pouw_window` certifier windows: PoUW window p, at its first certifier window
    w = p·`pouw_window`, is accepted if w has exactly one raw record, which holds exactly one salt frame of p, carrying
    `PLACEHOLDER`, and p's revealed salt has `salt_bits` bits. Anything else, a missing salt included, is a `salt`
    violation at w. The verdicts join `audit`'s. A committed or certified record doesn't show its frames, so a proof
    opens the salt frame there."""
    by_window: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        if not isinstance(r, Record):
            raise ValueError("the salt check reads raw records")
        if r.link == link:
            by_window[r.window].append(r)
    out = []
    for w in range(0, windows, pouw_window):
        p = w // pouw_window
        found, salt, prefix = by_window.get(w, []), salts.get(p), p.to_bytes(8, "big")
        accepted = False
        if len(found) == 1 and salt is not None and 8 * len(salt) == salt_bits:
            held = [x for row in found[0].grid for k, s, x in map(decode_frame, row[: row_count(row)])
                    if k == SALT and s == SALT_STREAM and x[:8] == prefix]
            accepted = held == [prefix + PLACEHOLDER]
        out.append(Verdict(link, w, None if accepted else "salt"))
    audit.verdicts.extend(out)
    return out
