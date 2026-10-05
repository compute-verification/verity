"""An active warden on a live byte stream: an asyncio TCP proxy that runs the reference's wardens on one boundary link.

The proxy is the I/O shell around the reference. It owns a monotonic clock and a ticker per grid, socket reads and
writes, and the framing of byte streams into B-byte frames. Everything that decides a grid is the reference's:
`schedule.eligible` and `schedule.run_shaper`'s handing rule, and `grid.Warden` on egress (at r) and on ingress (at r_in).
Its records are `grid.Record`s, published raw and through `commitment.commit`, which `audit.Audit.check_link` and
`check_ingress_link` read.

**Topology.** One TCP connection is the boundary link, the wire, between the proxy and the outside peer. Inside, the
proxy opens one TCP connection per stream to the server, and writes it the stream's request as the ingress warden
releases it. Every frame on the link is B bytes: a header (kind u8, stream u32, length u32), the payload, and zero
fill. Padding is B zero bytes (kind `PAD`); `DATA` is a payload, and `LAST` the last frame of a request.

**Egress.** The server's bytes on a stream are cut after every `\\n\\n`, the end of a server-sent event, so a frame's
content is fixed by the bytes and never by how they were read; an HTTP response head goes with the first event. The
event's `id:` field is its position in the Program's canonical order on the link. A frame's anchor is the egress bucket
in which the ingress warden released its stream's `LAST` frame (the request is recorded there), and its logical index
is its position in its stream. A frame read at time s is ready at bucket ⌊s/τ_b⌋ + 1, the first that starts after it.
At bucket t the shaper hands, in canonical order, each frame whose `max(h(k), ready(k), last)` is at most t, where
`h(k) = max(eligible(k), h(k−1))` under the declared sync of its anchor's window (`run_shaper`'s rule). `Warden.tick`
then releases min(queue, r) of them, left-packed, and the proxy writes those and r − n padding frames.

**Ingress.** A frame read from the link at time s is enqueued at bucket ⌈s/τ_in⌉ (`calibration.size_ingress_ms`'s
model) and released by the ingress `Warden` at r_in into its stream's server connection.

**Advice.** The proxy runs the windows of its declared syncs, one per window, which the prover gives before the run.
A frame anchored past them is never handed, nor is any frame behind it (`schedule.handing_buckets`).

**Faults fail closed** through `Warden.fault`: the rest of the window is padding, the queue is dropped, and the status
names the first reason (W6). They are:
- a late tick (`monitor-unavailable`): the ticker woke more than `slack` after its boundary. A bucket it missed is
  still emitted, as padding, so every window keeps its T rows and no bucket slips into the next;
- an overrun: the link still holds a whole bucket unwritten at a tick (`delivery-refused`), or the queue or the
  shaper's buffer passes Q (`overflow`); a server that refuses a stream's connection is `delivery-refused` on ingress;
- an advice violation: a window's declared sync outside Σ_sync (`coverage-never-established`), checked at the window's
  start; the grid still follows the declared sync, as the audit's expected grid does;
- a malformed frame (`malformed`): an event longer than a frame, without a position or with one already seen, on a
  stream with no released request, or bytes left unframed at the stream's end.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import math
import struct
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from verity.protocols.accounting.communication.warden.grid import (
    GridParams,
    Reason,
    Record,
    Row,
    Warden,
    row_count,
)
from verity.protocols.accounting.communication.warden.schedule import ClockSync, LogicalFrame, SyncSet, eligible

HEADER = struct.Struct(">BII")
PAD, DATA, LAST = 0, 1, 2
EVENT_END = b"\n\n"


def encode_frame(kind: int, stream: int, payload: bytes, frame_bytes: int) -> bytes:
    """One B-byte link frame: the header, the payload, then zero fill."""
    if HEADER.size + len(payload) > frame_bytes:
        raise ValueError(f"a {len(payload)}-byte payload doesn't fit a {frame_bytes}-byte frame")
    return HEADER.pack(kind, stream, len(payload)) + payload + bytes(frame_bytes - HEADER.size - len(payload))


def decode_frame(frame: bytes) -> tuple[int, int, bytes]:
    """(kind, stream, payload) of a link frame; padding is (`PAD`, 0, b"")."""
    kind, stream, n = HEADER.unpack_from(frame)
    return kind, stream, bytes(frame[HEADER.size : HEADER.size + n])


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
    """The honest shaper and the egress warden on one link, fed frames as they arrive (no I/O).

    `arrive` takes a frame at its canonical position with its ready bucket; `tick` hands what is due at the current
    bucket by `run_shaper`'s rule, ticks the warden and returns its row, with the window's record at its last bucket.
    Fed every frame at its ready bucket, the records equal `run_shaper`'s and `hands` its hand times."""

    def __init__(self, warden: Warden, syncs: Sequence[ClockSync], sync_set: SyncSet) -> None:
        self.warden, self.syncs, self.sync_set = warden, list(syncs), sync_set
        self.pending: dict[int, tuple[LogicalFrame, int]] = {}
        self.next = 0
        self.h = self.last = self.t = 0
        self.hands: list[int] = []
        self._check_advice()

    def _check_advice(self) -> None:
        w = self.warden.window
        if w < len(self.syncs) and self.syncs[w] not in self.sync_set:
            self.warden.fault(Reason.COVERAGE_NEVER_ESTABLISHED)

    def arrive(self, position: int, frame: LogicalFrame, ready: int) -> None:
        if position < self.next or position in self.pending:
            self.warden.fault(Reason.MALFORMED)
        elif len(self.pending) >= self.warden.params.queue_capacity:
            self.warden.fault(Reason.OVERFLOW)
        else:
            self.pending[position] = (frame, ready)

    def tick(self) -> tuple[Row, Record | None]:
        T = self.warden.params.buckets
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
            self.h, self.last, self.next = h, hand, self.next + 1
            self.hands.append(hand)
            self.warden.enqueue(frame.payload)
        row = self.warden.tick()
        self.t += 1
        record = None
        if self.t % T == 0:
            record = self.warden.close_window()
            self._check_advice()
        return row, record


class Bucketer:
    """The ingress warden on one link, fed frames as they arrive (no I/O): a frame arriving at bucket time s is
    `arrive`d with ⌈s⌉ and enqueued at that tick; `tick` returns the released row, with the record at a window's end."""

    def __init__(self, warden: Warden) -> None:
        self.warden = warden
        self.waiting: deque[tuple[int, bytes]] = deque()
        self.t = 0

    def arrive(self, frame: bytes, bucket: int) -> None:
        self.waiting.append((bucket, frame))

    def tick(self) -> tuple[Row, Record | None]:
        while self.waiting and self.waiting[0][0] <= self.t:
            self.warden.enqueue(self.waiting.popleft()[1])
        row = self.warden.tick()
        self.t += 1
        return row, self.warden.close_window() if self.t % self.warden.params.buckets == 0 else None


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


class Proxy:
    """The active warden on one link: `run(reader, writer)` serves the link connection from `origin`, the clock's time
    of bucket 0, for the windows of `syncs`, then closes it.

    - `publish(record, params)` receives every window's record, egress and ingress, as it closes (by default they are
      kept in `records`);
    - `frames` logs every egress frame read, as (position, stream, index, anchor, ready), and `shaper.hands` its hand
      bucket by position; `lateness` holds each tick's lateness in seconds, per grid; `link_frames_read` and
      `requests_released` count the ingress frames read and the requests released to the server;
    - `until(predicate)` waits until the proxy's state satisfies it, for a caller that drives a `VirtualClock`."""

    def __init__(self, upstream: tuple[str, int], egress: GridParams, ingress: GridParams, sync_set: SyncSet,
                 syncs: Sequence[ClockSync], *, origin: float, slack: float = 0.025,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 publish: Callable[[Record, GridParams], None] | None = None,
                 egress_link: str = "responses", ingress_link: str = "requests") -> None:
        if (egress.buckets * egress.bucket_ms) != (ingress.buckets * ingress.bucket_ms):
            raise ValueError("the egress and ingress grids have windows of the same length")
        self.upstream, self.egress, self.ingress = upstream, egress, ingress
        self.origin, self.slack, self.clock, self.sleep = origin, slack, clock, sleep
        self.windows = len(syncs)
        self.shaper = Shaper(Warden(egress_link, egress), syncs, sync_set)
        self.bucketer = Bucketer(Warden(ingress_link, ingress))
        self.records: list[Record] = []
        self.publish = publish or (lambda record, params: self.records.append(record))
        self.streams: dict[int, _Stream] = {}
        self.frames: list[tuple[int, int, int, int, int]] = []
        self.lateness: dict[str, list[float]] = {"egress": [], "ingress": []}
        self.link_frames_read = self.requests_released = 0
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
            for st in self.streams.values():
                if st.task is not None:
                    st.task.cancel()
            await asyncio.gather(link, *(st.task for st in self.streams.values() if st.task), return_exceptions=True)
            writer.close()
            self._changed.set()

    def _bucket_ready(self, now: float) -> int:
        return math.floor((now - self.origin) * 1000 / self.egress.bucket_ms) + 1

    async def _ticks(self, params: GridParams, lateness: list[float], warden: Warden):
        tau = params.bucket_ms / 1000
        for t in range(self.windows * params.buckets):
            deadline = self.origin + t * tau
            await self.sleep(deadline - self.clock())
            late = self.clock() - deadline
            lateness.append(late)
            if late > self.slack:
                warden.fault(Reason.MONITOR_UNAVAILABLE)
            yield t

    async def _egress(self, writer: asyncio.StreamWriter) -> None:
        p = self.egress
        bucket_bytes = p.allocation * p.frame_bytes
        padding = memoryview(bytes(bucket_bytes))
        async for _ in self._ticks(p, self.lateness["egress"], self.shaper.warden):
            backlog = writer.transport.get_write_buffer_size()
            if backlog > bucket_bytes:
                self.shaper.warden.fault(Reason.DELIVERY_REFUSED)
            row, record = self.shaper.tick()
            n = row_count(row)
            if backlog <= 4 * bucket_bytes:
                writer.writelines([*row[:n], padding[: (p.allocation - n) * p.frame_bytes]])
            if record is not None:
                self.publish(record, p)
            self._changed.set()
        await writer.drain()

    async def _ingress(self) -> None:
        p = self.ingress
        async for t in self._ticks(p, self.lateness["ingress"], self.bucketer.warden):
            row, record = self.bucketer.tick()
            anchor = t * p.bucket_ms // self.egress.bucket_ms
            for frame in row[: row_count(row)]:
                kind, s, payload = decode_frame(frame)
                if kind not in (DATA, LAST):
                    continue
                st = self._stream(s)
                st.out.put_nowait(payload)
                if kind == LAST and st.anchor is None:
                    st.anchor = anchor
                    self.requests_released += 1
            if record is not None:
                self.publish(record, p)
            self._changed.set()

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
            self.bucketer.warden.fault(Reason.DELIVERY_REFUSED)
            return

        async def pump() -> None:
            while True:
                writer.write(await st.out.get())

        pumping = asyncio.create_task(pump())
        try:
            while data := await reader.read(1 << 16):
                ready = self._bucket_ready(self.clock())
                for event in st.events.feed(data):
                    self._egress_event(s, st, event, ready)
                self._changed.set()
            if st.events.buffer:
                self.shaper.warden.fault(Reason.MALFORMED)
        finally:
            pumping.cancel()
            writer.close()

    def _egress_event(self, s: int, st: _Stream, event: bytes, ready: int) -> None:
        position = event_position(event)
        if position is None or st.anchor is None or HEADER.size + len(event) > self.egress.frame_bytes:
            self.shaper.warden.fault(Reason.MALFORMED)
            return
        frame = LogicalFrame(encode_frame(DATA, s, event, self.egress.frame_bytes), st.anchor, st.count)
        self.frames.append((position, s, st.count, st.anchor, ready))
        st.count += 1
        self.shaper.arrive(position, frame, ready)
