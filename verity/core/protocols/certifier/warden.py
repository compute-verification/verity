"""The active warden: the network certifier as a relay that knows bytes only and retransmits everything on a fixed
schedule, so it rebuilds the framing of every connection it carries.

The warden runs on an auditor host (`trusted-host`; co-located with the server for the demo, a named
assumption). It terminates each client's TCP connection, opens one to the server for it, and forwards each direction's
bytes unchanged in content: no tokenizer, API, request or stop string is the warden's. The one addition is the release
order on the server's copy of an HTTP request (`release`): what the warden decides is the framing and the timing, both
public functions of the schedule (`WardenParams`), and it certifies every cell it sends with the certifier's grid
(`grid.Certifier`) and certificates (`certificate.Issuer`, in the runner).

**Cells.** Every cell is B = `cell_bytes` bytes, `active.encode_frame(kind, ordinal, payload)`: the header (kind u8,
the connection's ordinal u32, the payload's length u32), at most P = B − 9 payload bytes, zero fill. `DATA` carries
exactly P bytes, `END` the last 0 … P of a direction, `PAD` none (marked padding), `FAIL` none (the connection failed
closed, for whatever reason), and `REFUSED` none (a connection the warden doesn't take). Both directions are cut
the same way: byte k of a direction is in its cell ⌊k/P⌋, so a cell's content is fixed by the bytes and never by how
they were read. The client speaks cells both ways through a thin, workload-blind shim (`cells`, `Shim`); the server
gets the ingress payloads as bytes (co-located in the demo; deployed, its own shim strips the cells as the client's
does).

**Admission.** The k-th connection to arrive, from 0, has ordinal k + 1, as the active certifier's `Requests` streams
it: ordinal 0 is no connection's. At most `lanes` connections are served at once; one that arrives with every lane
taken waits in a FIFO queue of at most `queue`, else it gets one `REFUSED` cell, on ordinal 0, and is closed.
The queue's head is admitted at the first bucket with a free lane, and holds that lane, r cells a bucket each way,
until its last bucket. Nothing admitted is dropped or reordered. A refusal is traffic past what the window's Programs
can hold (`every`: a timestep of `every` buckets has `program_slots` = `lanes` Program slots, the scan's
`ingress_scan.Class`), so the warden certifies no record of a window in which it refused a connection: `withhold`
gets that window's three records in place of `publish`, and the period check refuses a link-window missing any
record's certificate.

**The schedule.** Connection q, admitted at bucket s on the warden's own schedule, carries r = `slots` cells a bucket
each way from s on:
- ingress: the client's bytes as `DATA` cells, as soon as a whole cell's bytes have arrived, then `END` once the
  client's `END` cell (or its end of stream) has arrived, `PAD` in a slot with neither; the bucket of the `END` is a,
  and ingress has no cells after it. Ingress must end within `ingress_buckets` buckets and `ingress_cells` cells. Its
  release never waits on the server: the server gets a connection's ingress whole once its `END` is released, as
  `release.with_release` puts it, N the count of `END`s the warden released before (so the order of the `END` cells
  in the ingress records) and the run the warden's `run_id`; what it releases before then, or before the server's
  connection opens, is held for the server (at most the `ingress_cells` cells' bytes the ingress bound allows; past
  them the connection fails closed, and the server gets none of it);
- egress: `PAD` in every slot of the buckets before s + D, D = `delay`; then egress cell i is in bucket
  s + D + ⌊i/r⌋, slot i mod r: `DATA` with the server's bytes iP … (i+1)P − 1 if it has sent them, else `END` with its
  remaining 0 … P bytes if it has ended, else `FAIL` (late): the cell's deadline passed. The rest of the bucket of
  an `END` or a `FAIL` is `PAD`, and the connection ends with that bucket. Egress holds at most `egress_cells` cells.
Egress is based on the admission bucket s, never on the ingress's end, the server's connect or its response: nothing
the server does moves s + D. So the egress cells on a connection, their sizes and their buckets are a function of (s,
the number of egress cells, the failure's cell) and the public parameters alone, s of the arrivals and the admission
rule; the ingress cells are a function of s and the client's bytes.

**Fail closed.** A connection fails at the first of: an egress cell whose bytes aren't there at its deadline (late),
a server that hasn't ended by the last of `egress_cells` cells (limit), the server's connection failing (upstream),
ingress past its bounds (ingress), a client cell of another kind than `DATA` or `END` (malformed). Its `FAIL` cell is
in the slot of the next egress cell, and it is in the egress record. Every `FAIL` is the same bytes: the reason is a
frame of the warden's `fails` record (`reason_frame`), witness the client never sees. The server's three reasons show
only at an egress slot, so the server, which controls whether and where a connection fails, chooses one of at most
(cells + 1) outcomes; the client's two fail at once, and are the client's own doing. A tick that runs more than
`slack` past its boundary faults every grid's window (`monitor-unavailable`), as the active certifier's do.

**The pacer** (`Warden(paced=…)`, off by default): a bucket written on the loop leaves when its tick runs, and the
loop reads every server, so when a server writes within a bucket, which the prover chooses, moves when the bucket
reaches the clients. With `paced`, seconds past `slack`, each tick hands its bucket, every live connection's cells as a
part for that connection's socket, to a process of its own (`pacer.Pacer`), which releases it at `origin` + t·τ +
`paced` on the host's monotonic clock, a time fixed before the bucket is made; each connection's socket reaches the
pacer at its admission, and its last bucket's part closes the pacer's copy. A bucket that reached the pacer after its
time, or whose release started more than `slack` after it, faults every grid's window `monitor-unavailable`, as a late
tick does. A window's three records are published once every bucket of it that was handed is released, or by the
next window's first tick, faulted `monitor-unavailable` if one isn't by then. A client that reads slowly holds only its
own bytes in the pacer, and one more than `WRITE_BACKLOG` behind is closed, as on the loop.

**The records.** Each direction's grid has r·`lanes` slots a bucket of B-byte frames, and its frames are exactly the
cells the warden sent that way: every connection's r cells of the bucket, in ordinal order, padding included. So the
records are the wire, and each window's frames root (`certificate.frames_root`) attests every byte on it. The third,
`fails`, has `lanes` slots a bucket and one frame a failed connection, its reason, in the bucket of its `FAIL`
(`WardenParams.fails_grid`). `audit` is the read of the records in the clear: every connection's cells are the
schedule's, every `FAIL` has its one reason, and no connection failed.

`PROTOCOL.md`'s "The active warden" has the leak bound. The module imports only the standard library and `verity`.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import secrets
import time
from collections import Counter, defaultdict, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from typing import Any

from verity.ml.commitments.identity import canonical_json_bytes, tagged_sha512
from verity.core.protocols.certifier.active import (DATA, HEADER, PAD, RUN_ID_BYTES, decode_frame, encode_frame,
                                                   precise_sleep)
from verity.core.protocols.certifier.grid import Certifier, GridParams, Reason, Record, Row, Status, row_count
from verity.core.protocols.certifier.pacer import Pacer, Report
from verity.core.protocols.certifier.release import with_release

PARAMETERS_TAG = "verity-network-warden/parameters/v3"
#: the warden's cell kinds, beside `active`'s PAD and DATA
END, FAIL, REFUSED = 0x20, 0x21, 0x22
KIND_NAMES = {PAD: "pad", DATA: "data", END: "end", FAIL: "fail", REFUSED: "refused"}
#: why a connection failed: the server's three, then the client's two; only the `fails` record holds them
REASONS = ("late", "limit", "upstream", "ingress", "malformed")
#: the bytes the warden reads from a server at once
READ_BYTES = 1 << 16
#: the most a client connection may hold unwritten before the warden stops writing to it
WRITE_BACKLOG = 1 << 22


@dataclass(frozen=True)
class WardenParams:
    """The schedule's public parameters, fixed before the run (`WardenParams.digest` binds them)."""

    cell_bytes: int = 256  # B
    slots: int = 4  # r: a connection's cells a bucket, each way
    bucket_ms: int = 100  # τ
    delay: int = 10  # D: buckets from admission to the first egress cell's
    lanes: int = 16  # connections served at once
    queue: int = 64  # connections waiting for a lane, past which REFUSED
    window: int = 600  # T: buckets a record covers
    ingress_cells: int = 64
    ingress_buckets: int = 50
    egress_cells: int = 4096
    every: int = 1  # a timestep's buckets: the Programs' boundaries are every `every` buckets of a window

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} is a positive integer, not {value!r}")
        if self.cell_bytes <= HEADER.size + 1:
            raise ValueError(f"a cell holds its {HEADER.size}-byte header and a byte")
        if self.window % self.every:
            raise ValueError(f"every ({self.every}) divides the window ({self.window}): a window is whole timesteps")

    @property
    def program_slots(self) -> int:
        """A timestep's Program slots, one a lane (`ingress_scan.Class.SLOTS`, for a class that serves the warden)."""
        return self.lanes

    @property
    def window_programs(self) -> int:
        """A window's Program slots: `window / every` timesteps of `program_slots`."""
        return self.window // self.every * self.program_slots

    @property
    def payload(self) -> int:
        """P: the bytes a `DATA` cell carries."""
        return self.cell_bytes - HEADER.size

    @property
    def grid(self) -> GridParams:
        """Each direction's record: r·lanes cells a bucket of B bytes."""
        return GridParams(self.window, self.lanes * self.slots, self.cell_bytes, self.bucket_ms)

    @property
    def fails_grid(self) -> GridParams:
        """The `fails` record: at most one reason a connection, so `lanes` a bucket."""
        return GridParams(self.window, self.lanes, self.cell_bytes, self.bucket_ms)

    @cached_property
    def digest(self) -> bytes:
        """The 64 bytes each record head's `parameters` carries."""
        return tagged_sha512(PARAMETERS_TAG, canonical_json_bytes(asdict(self) | {"kinds": sorted(KIND_NAMES),
                                                                                  "reasons": list(REASONS)}))


def cell(kind: int, ordinal: int, payload: bytes, cell_bytes: int) -> bytes:
    return encode_frame(kind, ordinal, payload, cell_bytes)


def read_cell(data: bytes, cell_bytes: int) -> tuple[int, int, bytes]:
    """(kind, ordinal, payload) of a cell the warden makes, or ValueError: its kind, length and zero fill exact."""
    if len(data) != cell_bytes:
        raise ValueError(f"a cell is {cell_bytes} bytes, not {len(data)}")
    kind, ordinal, payload = decode_frame(data)
    size = len(payload)
    if kind not in KIND_NAMES or HEADER.unpack_from(data)[2] != size or \
            cell(kind, ordinal, payload, cell_bytes) != data:
        raise ValueError("not a warden cell")
    if (kind == DATA and size != cell_bytes - HEADER.size) or (kind in (PAD, FAIL, REFUSED) and size):
        raise ValueError(f"a {KIND_NAMES[kind]} cell of {size} bytes")
    return kind, ordinal, payload


def reason_frame(ordinal: int, reason: str, cell_bytes: int) -> bytes:
    """A frame of the `fails` record: why connection `ordinal` failed, as one byte, its index in `REASONS`."""
    return encode_frame(FAIL, ordinal, bytes([REASONS.index(reason)]), cell_bytes)


def read_reason(data: bytes, cell_bytes: int) -> tuple[int, str]:
    """(ordinal, reason) of a `reason_frame`, or ValueError."""
    if len(data) != cell_bytes:
        raise ValueError(f"a reason is {cell_bytes} bytes, not {len(data)}")
    kind, ordinal, payload = decode_frame(data)
    if kind != FAIL or len(payload) != 1 or payload[0] >= len(REASONS) or \
            reason_frame(ordinal, REASONS[payload[0]], cell_bytes) != data:
        raise ValueError("not a reason frame")
    return ordinal, REASONS[payload[0]]


def cells(data: bytes, cell_bytes: int, ordinal: int = 0) -> bytes:
    """A direction's bytes as cells: `DATA` cells of P bytes, then one `END` with the rest (the client shim's sending
    half, and the warden's own cut)."""
    p = cell_bytes - HEADER.size
    whole = (len(data) - 1) // p if data else 0
    out = [cell(DATA, ordinal, data[i * p:(i + 1) * p], cell_bytes) for i in range(whole)]
    return b"".join(out) + cell(END, ordinal, data[whole * p:], cell_bytes)


@dataclass
class Shim:
    """The client's thin shim, reading half: every cell the warden sends on one connection, until it closes, each kept
    with the time it arrived; `data` is the `DATA` and `END` payloads, `outcome` "end", "fail" or "refused", or "cut"
    when the connection closed before any of those, or inside a cell."""

    cell_bytes: int
    clock: Callable[[], float] = time.monotonic
    data: bytearray = field(default_factory=bytearray)
    arrivals: list[tuple[float, int]] = field(default_factory=list)
    outcome: str | None = None
    ordinal: int | None = None

    async def read(self, reader: asyncio.StreamReader) -> Shim:
        while True:
            try:
                raw = await reader.readexactly(self.cell_bytes)
            except asyncio.IncompleteReadError as e:
                if e.partial or self.outcome is None:
                    self.outcome = "cut"
                return self
            kind, ordinal, payload = read_cell(raw, self.cell_bytes)
            self.arrivals.append((self.clock(), kind))
            self.ordinal = ordinal
            if self.outcome is not None:
                if kind != PAD:
                    self.outcome = "cut"
                continue
            if kind in (DATA, END):
                self.data += payload
            if kind in (END, FAIL, REFUSED):
                self.outcome = KIND_NAMES[kind]


@dataclass(eq=False)
class _Connection:
    ordinal: int
    reader: asyncio.StreamReader
    writer: Any
    done: asyncio.Future
    arrived: int
    admitted: int | None = None
    pad: bytes = b""  # its PAD cell, made once (`_pad`): a window's grids hold one a connection, not one a slot
    server: Any = None
    ingress: bytearray = field(default_factory=bytearray)  # received, not yet released
    held: bytearray = field(default_factory=bytearray)  # released, held for the server until the END's release
    unsent: bytearray = field(default_factory=bytearray)  # released whole before the server's connection opened
    release: int | None = None  # N: its END's place in the warden's release order
    ingress_ended: bool = False  # the client's END has been received
    ingress_end: int | None = None  # a: the bucket the warden released ingress's END
    ingress_cells: int = 0
    egress: bytearray = field(default_factory=bytearray)  # the server's bytes, not yet released
    egress_ended: bool = False
    egress_cells: int = 0  # egress slots past s + D, released
    problem: str | None = None  # the client's: failed at the next slot
    upstream: str | None = None  # the server connection's: failed at the first egress slot without a whole cell
    failed: str | None = None  # one of REASONS
    failed_at: int | None = None  # the egress cell of the FAIL
    ended: int | None = None  # the bucket of the connection's last cells
    bytes_in: int = 0
    bytes_out: int = 0
    tasks: list[asyncio.Task] = field(default_factory=list)


def _pad(c: _Connection, cell_bytes: int) -> bytes:
    """c's PAD cell, made at its first use and kept, however c was admitted."""
    if not c.pad:
        c.pad = cell(PAD, c.ordinal, b"", cell_bytes)
    return c.pad


class Warden:
    """The active warden (the module docstring): `handle(reader, writer)` serves one client connection, `run()` ticks
    the schedule from `origin`, the clock's time of bucket 0, for `windows` record windows, then cuts whatever is still
    served or queued. `connect()` opens a server connection, a (reader, writer) pair.

    - `publish(record, params)` receives each window's records, ingress, egress, then fails, as they close (by
      default they are kept in `records`), and `withhold(record, params)` in its place those of a window in which a
      connection was refused (`refused_windows`; by default kept in `withheld`): nothing of that window is to be
      certified; `log` has one entry a connection once it ends, `refused` the ordinal-less refusals' count,
      `lateness` each tick's in seconds;
    - `release(link, row)`, when given, receives each grid's row as it ticks, ingress, egress, then fails, before
      that tick's `publish`: a publisher in another process builds each record from them as it fills, so that the
      tick never hands it a whole window at once;
    - `requests_released` counts the `END`s released, each connection's N its count before (`log`'s `release`), and
      `run_id`, `RUN_ID_BYTES` drawn from `run_nonce` at construction, names the run on the server's side;
    - with `paced`, seconds past `slack` (off by default), each tick hands its bucket to the pacer (`pacer.Pacer`;
      the module docstring's "The pacer"), which releases it `paced` after its boundary on the host's monotonic clock,
      `origin`'s clock then, to each connection's socket (`writer.get_extra_info("socket")`); `releases` holds each
      bucket's `pacer.Report` once its writes end;
    - `until(predicate)` waits until the warden's state satisfies it, for a caller that drives a `VirtualClock`."""

    def __init__(self, params: WardenParams, connect: Callable[[], Awaitable[tuple[Any, Any]]], *, origin: float,
                 windows: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = precise_sleep,
                 publish: Callable[[Record, GridParams], None] | None = None,
                 withhold: Callable[[Record, GridParams], None] | None = None,
                 release: Callable[[str, Row], None] | None = None, slack: float = 0.025,
                 egress_link: str = "responses", ingress_link: str = "requests", fails_link: str = "fails",
                 run_nonce: Callable[[int], bytes] = secrets.token_bytes, paced: float | None = None) -> None:
        if paced is not None and not (isinstance(paced, (int, float)) and slack < paced < params.bucket_ms / 1000):
            raise ValueError(f"a paced bucket leaves past the slack and before the next boundary, so that a tick in "
                             f"time hands it before its time, not {paced!r} s after its boundary")
        self.paced = paced
        self.releases: list[Report] = []
        self._pacer: Pacer | None = None
        self._unreleased: Counter[int] = Counter()
        self._held: dict[int, tuple[Callable[[Record, GridParams], None], list[tuple[Record, GridParams]]]] = {}
        self.run_id = run_nonce(RUN_ID_BYTES)
        if not (isinstance(self.run_id, bytes) and len(self.run_id) == RUN_ID_BYTES):
            raise ValueError(f"a run id is {RUN_ID_BYTES} bytes")
        self.requests_released = 0
        self.params, self.connect = params, connect
        self.origin, self.windows, self.clock, self.sleep, self.slack = origin, windows, clock, sleep, slack
        self.egress = Certifier(egress_link, params.grid)
        self.ingress = Certifier(ingress_link, params.grid)
        self.fails = Certifier(fails_link, params.fails_grid)
        self.records: list[Record] = []
        self.publish = publish or (lambda record, grid: self.records.append(record))
        self.withheld: list[Record] = []
        self.withhold = withhold or (lambda record, grid: self.withheld.append(record))
        self.release = release
        self.queue: deque[_Connection] = deque()
        self.active: list[_Connection] = []
        self.ordinals = 0
        self.refused = 0
        self.refused_windows: set[int] = set()
        self.log: list[dict] = []
        self.lateness: list[float] = []
        self.ticks = 0
        self._changed = asyncio.Event()

    async def until(self, predicate: Callable[[Warden], bool]) -> None:
        while not predicate(self):
            self._changed.clear()
            await self._changed.wait()

    def _bucket(self) -> int:
        return math.ceil((self.clock() - self.origin) * 1000 / self.params.bucket_ms)

    async def handle(self, reader: asyncio.StreamReader, writer: Any) -> None:
        if len(self.queue) >= self.params.queue:
            self.refused += 1
            self.refused_windows.add(self.ticks // self.params.window)
            writer.write(cell(REFUSED, 0, b"", self.params.cell_bytes))
            writer.close()
            self._changed.set()
            return
        connection = _Connection(self.ordinals + 1, reader, writer, asyncio.get_running_loop().create_future(),
                                 self._bucket())
        self.ordinals += 1
        self.queue.append(connection)
        self._changed.set()
        try:
            await connection.done
        finally:
            writer.close()

    async def run(self) -> None:
        tau = self.params.bucket_ms / 1000
        reports = None
        if self.paced is not None:
            self._pacer = await Pacer.start(backlog=WRITE_BACKLOG)
            reports = asyncio.create_task(self._reports(self._pacer))
        try:
            for t in range(self.windows * self.params.window):
                deadline = self.origin + t * tau
                await self.sleep(deadline - self.clock())
                late = self.clock() - deadline
                self.lateness.append(late)
                if late > self.slack:
                    for grid in (self.ingress, self.egress, self.fails):
                        grid.fault(Reason.MONITOR_UNAVAILABLE)
                self._tick(t)
                self.ticks += 1
                self._changed.set()
            if self._pacer is not None:
                await self._pacer.close()
                await reports
        finally:
            for connection in [*self.active, *self.queue]:
                self._end(connection, cut=True)
            self.active, self.queue = [], deque()
            if self._pacer is not None:
                reports.cancel()
                await self._pacer.kill()
                for w in sorted(self._held):
                    self._emit(w, Reason.MONITOR_UNAVAILABLE)
            self._changed.set()

    def _tick(self, t: int) -> None:
        T = self.params.window
        if t % T == 0 and t // T - 1 in self._held:
            self._emit(t // T - 1, Reason.MONITOR_UNAVAILABLE)
        while self.queue and len(self.active) < self.params.lanes:
            self._admit(self.queue.popleft(), t)
        parts = []
        for c in self.active:
            if c.ingress_end is None and c.failed is None:
                for frame in self._ingress(c, t):
                    self.ingress.enqueue(frame)
            out = self._egress(c, t)
            for frame in out:
                self.egress.enqueue(frame)
            if self._pacer is None:
                self._write(c, b"".join(out))
            else:
                parts.append((c.ordinal, b"".join(out), c.ended is not None))
        if parts:
            self._pacer.hand(t, self.origin + t * self.params.bucket_ms / 1000 + self.paced, parts)
            self._unreleased[t // T] += 1
        for c in self.active:
            if c.ended is not None:
                self._end(c)
        self.active = [c for c in self.active if c.ended is None]
        for grid in (self.ingress, self.egress, self.fails):
            row = grid.tick()
            if self.release is not None:
                self.release(grid.link, row)
        if (t + 1) % T == 0:
            w = t // T
            out = self.withhold if w in self.refused_windows else self.publish
            records = [(self.ingress.close_window(), self.params.grid), (self.egress.close_window(), self.params.grid),
                       (self.fails.close_window(), self.params.fails_grid)]
            self._held[w] = (out, records)
            if not self._unreleased[w]:
                self._emit(w)

    def _emit(self, w: int, reason: Reason | None = None) -> None:
        """Window w's held records, each with `reason` as its status unless it has one: a window records its first
        fault."""
        out, records = self._held.pop(w)
        for record, grid in records:
            out(record if reason is None or not record.status.complete else replace(record, status=Status(reason)),
                grid)

    async def _reports(self, pacer: Pacer) -> None:
        async for report in pacer.reports():
            self._report(report)

    def _report(self, report: Report) -> None:
        """A bucket's release or its writes as the pacer reports them. A bucket that reached the pacer after its time,
        or whose release started more than `slack` after it, faults every grid's window `monitor-unavailable`: the open
        window's certifiers, or the records held for a closed one, published once every bucket of it that was handed is
        released. A connection's writes fault nothing, as on the loop: a client that doesn't read holds its own."""
        if report.end is not None:
            self.releases.append(report)
            return
        w = report.t // self.params.window
        if report.arrived > 0 or report.start > self.slack:
            if w in self._held:
                out, records = self._held[w]
                self._held[w] = (out, [(r if not r.status.complete else
                                        replace(r, status=Status(Reason.MONITOR_UNAVAILABLE)), g) for r, g in records])
            elif w == self.ingress.window:
                for grid in (self.ingress, self.egress, self.fails):
                    grid.fault(Reason.MONITOR_UNAVAILABLE)
        self._unreleased[w] -= 1
        if not self._unreleased[w] and w in self._held:
            self._emit(w)
        self._changed.set()

    def _admit(self, c: _Connection, t: int) -> None:
        c.admitted = t
        _pad(c, self.params.cell_bytes)
        self.active.append(c)
        if self._pacer is not None:
            self._pacer.add(c.ordinal, c.writer.get_extra_info("socket").fileno())
        c.tasks = [asyncio.create_task(self._client(c)), asyncio.create_task(self._server(c))]

    async def _client(self, c: _Connection) -> None:
        """Read the client's cells into `c.ingress` until its `END` (or its end of stream), at most `ingress_cells`
        cells' bytes. Only the payload of each passes: the warden makes every cell it sends."""
        B, p = self.params.cell_bytes, self.params.payload
        try:
            while not c.ingress_ended:
                try:
                    raw = await c.reader.readexactly(B)
                except asyncio.IncompleteReadError as e:
                    if e.partial:
                        c.problem = c.problem or "malformed"
                    c.ingress_ended = True
                    break
                kind, _, n = HEADER.unpack_from(raw)
                if kind not in (DATA, END) or n > p:
                    c.problem = c.problem or "malformed"
                    return
                c.ingress += raw[HEADER.size:HEADER.size + n]
                c.bytes_in += n
                c.ingress_ended = kind == END
                if c.bytes_in > self.params.ingress_cells * p:
                    c.problem = c.problem or "ingress"
                    return
                self._changed.set()
        except OSError:
            c.problem = c.problem or "malformed"
        finally:
            self._changed.set()

    async def _server(self, c: _Connection) -> None:
        """Open the server connection and read its bytes into `c.egress` until it ends. Past `egress_cells` cells'
        bytes it stops reading: those can't be sent, and the slot of the cell past the last fails (limit)."""
        cap = self.params.egress_cells * self.params.payload
        try:
            c.server = await self.connect()
            if c.unsent:
                c.server[1].write(bytes(c.unsent))
                c.unsent.clear()
            self._changed.set()
            reader = c.server[0]
            while c.bytes_out <= cap:
                data = await reader.read(READ_BYTES)
                if not data:
                    c.egress_ended = True
                    return
                c.egress += data
                c.bytes_out += len(data)
                self._changed.set()
        except OSError as e:
            c.upstream = f"{type(e).__name__}: {e}"
        finally:
            self._changed.set()

    def _ingress(self, c: _Connection, t: int) -> list[bytes]:
        """Connection c's r ingress cells at bucket t, their payloads held for the server until the `END`'s release,
        then written whole with the release order, or kept until its connection opens (at most the ingress bound's
        bytes): the release never waits on the server."""
        B, p, r = self.params.cell_bytes, self.params.payload, self.params.slots
        out, sent, pad = [], [], _pad(c, B)
        for _ in range(r):
            if c.ingress_end is not None:
                out.append(pad)
            elif len(c.ingress) > p or (len(c.ingress) == p and not c.ingress_ended):
                sent.append(bytes(c.ingress[:p]))
                del c.ingress[:p]
                out.append(cell(DATA, c.ordinal, sent[-1], B))
            elif c.ingress_ended:
                sent.append(bytes(c.ingress))
                c.ingress.clear()
                out.append(cell(END, c.ordinal, sent[-1], B))
                c.ingress_end = t
            else:
                out.append(pad)
        c.ingress_cells += len(sent)
        if c.ingress_end is None and t - c.admitted + 1 >= self.params.ingress_buckets:
            c.problem = c.problem or "ingress"
        c.held += b"".join(sent)
        if c.ingress_end == t:
            c.release, self.requests_released = self.requests_released, self.requests_released + 1
            whole = with_release(bytes(c.held), c.release, self.run_id)
            c.held.clear()
            if c.server is None:
                c.unsent += whole
            elif not c.server[1].is_closing():
                c.server[1].write(whole)
        return out

    def _egress(self, c: _Connection, t: int) -> list[bytes]:
        """Connection c's r egress cells at bucket t (the module docstring's schedule), from s + D on. A client's
        problem fails the connection at once; the server's only at an egress slot, so where it fails is one of the
        egress cells. The reason goes to the `fails` record, never into the cell."""
        B, p, r, D = self.params.cell_bytes, self.params.payload, self.params.slots, self.params.delay
        out, pad = [], _pad(c, B)
        for _ in range(r):
            if c.ended is not None:
                out.append(pad)
                continue
            failed = None
            if c.problem is not None:
                failed = c.problem
            elif t < c.admitted + D:
                out.append(pad)
                continue
            else:
                whole = len(c.egress) > p or (len(c.egress) == p and not c.egress_ended)
                if c.egress_ended and not whole:
                    out.append(cell(END, c.ordinal, bytes(c.egress), B))
                    c.egress.clear()
                    c.egress_cells += 1
                    c.ended = t
                    continue
                if c.egress_cells + 1 >= self.params.egress_cells:
                    failed = "limit"
                elif whole:
                    out.append(cell(DATA, c.ordinal, bytes(c.egress[:p]), B))
                    del c.egress[:p]
                    c.egress_cells += 1
                    continue
                else:
                    failed = "upstream" if c.upstream else "late"
            c.failed, c.failed_at, c.ended = failed, c.egress_cells, t
            out.append(cell(FAIL, c.ordinal, b"", B))
            self.fails.enqueue(reason_frame(c.ordinal, failed, B))
        return out

    def _write(self, c: _Connection, data: bytes) -> None:
        writer = c.writer
        if writer.is_closing():
            return
        if writer.transport.get_write_buffer_size() > WRITE_BACKLOG:
            writer.close()
            return
        writer.write(data)

    def _end(self, c: _Connection, cut: bool = False) -> None:
        for task in c.tasks:
            task.cancel()
        if c.server is not None:
            c.server[1].close()
        c.writer.close()
        if not c.done.done():
            c.done.set_result(None)
        self.log.append({"ordinal": c.ordinal, "arrived": c.arrived, "admitted": c.admitted,
                         "ingress_end": c.ingress_end, "release": c.release, "ended": c.ended,
                         "ingress_cells": c.ingress_cells,
                         "egress_cells": c.egress_cells, "bytes_in": c.bytes_in, "bytes_out": c.bytes_out,
                         "failed": c.failed, "failed_at": c.failed_at, "upstream": c.upstream, "cut": cut})


def audit(params: WardenParams, ingress: Sequence[Record], egress: Sequence[Record], fails: Sequence[Record]) -> dict:
    """The read of a run's records in the clear (a rehearsal's; a deployment's is N2's, in ZK). Every cell is a warden
    cell; in every bucket of each direction each live connection has r cells, in ordinal order, and admissions are in
    ordinal order; each connection's ingress is data and padding, then one `END` and padding to its bucket a; and its
    egress is exactly the schedule's for (its admission s, its egress cells): padding through bucket s + D − 1 (up to
    its `FAIL`, for a client's failure before then), then `DATA` in every slot, then one `END` or `FAIL`, then padding
    to the bucket's end. Every `FAIL` has one reason in the `fails` record, in its bucket, and every reason its `FAIL`.
    A window is accepted when its three records are complete, its cells are the schedule's and no connection failed in
    it. Each connection's entry has its buckets, cells, outcome, its failure's reason and the SHA-256 of each
    direction's bytes."""
    T, r, D = params.window, params.slots, params.delay
    verdicts: dict[int, list[str]] = defaultdict(list)
    for record in [*ingress, *egress, *fails]:
        if not record.status.complete:
            verdicts[record.window].append(f"{record.link}: {record.status.reason.value}")
    reasons: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for record in fails:
        for b, row in enumerate(record.grid):
            for f in row[:row_count(row)]:
                try:
                    o, why = read_reason(f, params.cell_bytes)
                except ValueError as e:
                    verdicts[record.window].append(f"fails bucket {record.window * T + b}: {e}")
                    continue
                reasons[o].append((record.window * T + b, why))
    per: dict[int, dict[str, dict[int, list[tuple[int, bytes]]]]] = defaultdict(lambda: {"ingress": {}, "egress": {}})
    total = {"ingress": 0, "egress": 0}
    for name, records in (("ingress", ingress), ("egress", egress)):
        for record in records:
            for b, row in enumerate(record.grid):
                t = record.window * T + b
                try:
                    got = [read_cell(f, params.cell_bytes) for f in row[:row_count(row)]]
                except ValueError as e:
                    verdicts[record.window].append(f"{name} bucket {t}: {e}")
                    continue
                total[name] += len(got)
                order = [o for _, o, _ in got[::r]]
                if len(got) % r or got != sorted(got, key=lambda c: c[1]) or len(set(order)) != len(order) or \
                        any(o != got[i * r + j][1] for i, o in enumerate(order) for j in range(r)):
                    verdicts[record.window].append(f"{name} bucket {t}: not r cells a connection in ordinal order")
                for kind, o, payload in got:
                    per[o][name].setdefault(t, []).append((kind, payload))
    connections, last_admitted = [], -1
    for n, o in enumerate(sorted(per)):
        ing, eg = per[o]["ingress"], per[o]["egress"]
        s = min([*ing, *eg])
        problems = [f"connection {o} where {n + 1} is next"] if o != n + 1 else []
        if s < last_admitted:
            problems.append(f"admitted at {s}, before connection {o - 1}")
        last_admitted = s
        if sorted(ing) != list(range(s, s + len(ing))) or sorted(eg) != list(range(s, s + len(eg))):
            problems.append("its buckets aren't consecutive from its admission")
        ing_cells = [c for t in sorted(ing) for c in ing[t]]
        eg_cells = [c for t in sorted(eg) for c in eg[t]]
        kinds = [k for k, _ in ing_cells]
        a = None
        if END in kinds:
            e = kinds.index(END)
            a = s + e // r
            if any(k not in (DATA, PAD) for k in kinds[:e]) or any(k != PAD for k in kinds[e + 1:]) or \
                    len(kinds) != (a - s + 1) * r:
                problems.append("its ingress isn't data and padding, then one END and padding to its bucket")
        elif any(k not in (DATA, PAD) for k in kinds):
            problems.append("its ingress isn't data and padding")
        kinds = [k for k, _ in eg_cells]
        live = [k for k in kinds if k != PAD]
        failed = failed_at = None
        if live:
            last = live[-1]
            lead = kinds.index(FAIL) if live == [FAIL] and kinds.index(FAIL) < D * r else D * r
            expected = [PAD] * lead + [DATA] * (len(live) - 1) + [last]
            expected += [PAD] * (-len(expected) % r)
            if last not in (END, FAIL) or kinds != expected:
                problems.append("its egress isn't padding to s + D, data in every slot, one END or FAIL, padding")
        got = reasons.pop(o, [])
        if FAIL in kinds:
            failed_at = live.index(FAIL)
            if [t for t, _ in got] == [s + kinds.index(FAIL) // r]:
                failed = got[0][1]
            else:
                failed = "fail"
                problems.append("its FAIL hasn't one reason, in its bucket")
        elif got:
            problems.append("it has a reason and no FAIL")
        data_in = b"".join(p for k, p in ing_cells if k in (DATA, END))
        data_out = b"".join(p for k, p in eg_cells if k in (DATA, END))
        ended = max(eg) if live and live[-1] in (END, FAIL) else None
        for problem in problems:
            verdicts[s // T].append(f"connection {o}: {problem}")
        if failed:
            verdicts[max(eg) // T].append(f"connection {o} failed closed ({failed}) at egress cell {failed_at}")
        connections.append({"ordinal": o, "admitted": s, "ingress_end": a, "ended": ended,
                            "egress_cells": len(live), "ingress_bytes": len(data_in), "egress_bytes": len(data_out),
                            "failed": failed, "failed_at": failed_at,
                            "ingress_sha256": hashlib.sha256(data_in).hexdigest(),
                            "egress_sha256": hashlib.sha256(data_out).hexdigest()})
    for o, got in reasons.items():
        for t, _ in got:
            verdicts[t // T].append(f"a reason for connection {o}, which has no cells")
    windows = sorted({rec.window for rec in [*ingress, *egress, *fails]})
    out = [{"window": w, "accepted": not verdicts.get(w), "problems": verdicts.get(w, [])} for w in windows]
    return {"accepted": bool(out) and all(w["accepted"] for w in out), "windows": out, "connections": connections,
            "cells": total}
