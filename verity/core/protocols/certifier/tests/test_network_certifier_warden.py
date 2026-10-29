"""The active warden (`warden`), a relay that knows bytes only: driven tick by tick on a `VirtualClock`, its clients
in-memory connections speaking cells and its server an in-memory one whose bytes the test schedules, it releases honest
traffic exactly on its schedule, fails a late byte closed, queues past its lanes then refuses (and withholds that
window's records from `publish`), passes both directions'
bytes unchanged (the server's copy of a request adding only its place in the release order), and puts the same number
of same-size cells in the same buckets however the server times its bytes."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import socket

import pytest

from verity.core.protocols.certifier.active import DATA, HEADER, PAD, VirtualClock, encode_frame
from verity.core.protocols.certifier.grid import Reason
from verity.core.protocols.certifier.pacer import Report
from verity.core.protocols.certifier.release import release_of, with_release
from verity.core.protocols.certifier.warden import (END, FAIL, KIND_NAMES, REASONS, REFUSED, Shim, Warden,
                                                    WardenParams, audit, cell, cells, read_cell, read_reason,
                                                    reason_frame)

PARAMS = WardenParams(cell_bytes=64, slots=2, bucket_ms=100, delay=3, lanes=2, queue=2, window=24, ingress_cells=8,
                      ingress_buckets=6, egress_cells=12)
B, P, R, D = PARAMS.cell_bytes, PARAMS.payload, PARAMS.slots, PARAMS.delay
TAU = PARAMS.bucket_ms / 1000
EOF = None


def request(n: int, tag: int = 0) -> bytes:
    """A request of n bytes ending in its newline, every byte value in it."""
    return bytes((tag + i * 37) % 256 for i in range(n - 1)).replace(b"\n", b"\x0b") + b"\n"


class Server:
    """An in-memory server: connection k (in connect order, which is admission order) answers once it has received a
    newline, with `answers[k]`, a list of (buckets after the request's bucket, bytes or EOF); `release(t)` feeds what is
    due by bucket t. `refuse` fails the connects listed."""

    def __init__(self, answers: list[list], refuse: tuple[int, ...] = (), hold: dict[int, int] | None = None) -> None:
        self.answers, self.refuse, self.hold = answers, refuse, hold or {}
        self.conns: list[ServerConn] = []
        self.held: dict[int, asyncio.Event] = {}

    async def connect(self):
        k = len(self.conns)
        conn = ServerConn(self.answers[k] if k < len(self.answers) else [])
        self.conns.append(conn)
        if k in self.hold:
            self.held[k] = asyncio.Event()
            await self.held[k].wait()
        if k in self.refuse:
            raise ConnectionRefusedError("refused")
        return conn.reader, conn

    def release(self, t: int) -> None:
        for k, opened in list(self.held.items()):
            if t >= self.hold[k]:
                opened.set()
                del self.held[k]
        for conn in self.conns:
            conn.release(t)


class ServerConn:
    def __init__(self, answer: list) -> None:
        self.answer, self.reader, self.received = list(answer), asyncio.StreamReader(), bytearray()
        self.at: int | None = None  # the bucket its request ended in
        self.fed, self.eof, self.closed = 0, False, False

    def write(self, data: bytes) -> None:
        self.received += data
        if self.at is None and b"\n" in data:
            self.at = -1

    def release(self, t: int) -> None:
        if self.at == -1:
            self.at = t - 1
        while self.at is not None and self.answer and self.at + self.answer[0][0] <= t and not self.closed:
            _, data = self.answer.pop(0)
            if data is EOF:
                self.eof = True
                self.reader.feed_eof()
            else:
                self.fed += len(data)
                self.reader.feed_data(data)

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True


class Client:
    """One client connection: its cells in its reader, and every write with the bucket it was made in."""

    def __init__(self, warden: Warden, data: bytes | tuple[bytes, bool]) -> None:
        self.warden, self.writes, self.closed = warden, [], False
        self.transport = self
        data, eof = data if isinstance(data, tuple) else (data, True)
        self.reader = asyncio.StreamReader()
        self.reader.feed_data(data)
        if eof:
            self.reader.feed_eof()
        self.task = asyncio.create_task(warden.handle(self.reader, self))

    def write(self, data: bytes) -> None:
        assert not self.closed
        self.writes.append((self.warden.ticks, bytes(data)))

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    def get_write_buffer_size(self) -> int:
        return 0

    def cells(self) -> list[tuple[int, int, bytes]]:
        """(bucket, kind, payload) of every cell written, each exactly B bytes."""
        out = []
        for t, data in self.writes:
            assert len(data) % B == 0
            for i in range(0, len(data), B):
                kind, _, payload = read_cell(data[i:i + B], B)
                out.append((t, kind, payload))
        return out

    def shim(self) -> Shim:
        async def read() -> Shim:
            reader = asyncio.StreamReader()
            reader.feed_data(b"".join(d for _, d in self.writes))
            reader.feed_eof()
            return await Shim(B).read(reader)
        return asyncio.run(read())


class SocketClient(Client):
    """A client whose connection's warden end is a socket, the one a paced warden's pacer writes to, and whose other
    end the test reads (`received`) once the run is over."""

    def __init__(self, warden: Warden, data: bytes | tuple[bytes, bool]) -> None:
        self.ours, self.peer = socket.socketpair()
        self.ours.setblocking(False)
        super().__init__(warden, data)

    def get_extra_info(self, name: str):
        return self.ours if name == "socket" else None

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.ours.close()

    def received(self) -> bytes:
        out = bytearray()
        while chunk := self.peer.recv(1 << 16):
            out += chunk
        return bytes(out)


def settled(warden: Warden, server: Server) -> bool:
    cap = PARAMS.egress_cells * P
    if sum(c.server is None and c.upstream is None for c in warden.active) != len(server.held):
        return False
    for c in warden.active:
        if c.problem is None and not c.ingress_ended and c.reader._buffer:
            return False
        if c.server is not None:
            conn = c.server[1]
            if c.bytes_out <= cap and (c.bytes_out != conn.fed or c.egress_ended != conn.eof):
                return False
    return True


async def drive(server: Server, arrivals: dict[int, list[bytes]], *, windows: int = 1, late: dict | None = None,
                params: WardenParams = PARAMS, client: type[Client] = Client, **hooks):
    """Connect `arrivals[t]` (each a client's raw cell bytes) just before bucket t's tick, let the server answer what is
    due and the warden read it, then tick, and with `paced` wait for the pacer to report the bucket's release, for
    `windows` windows; `late[t]` puts tick t that many seconds late. `hooks` are the warden's `publish`, `release` and
    `paced`."""
    clock = VirtualClock(-TAU / 2)
    warden = Warden(params, server.connect, origin=0.0, windows=windows, clock=clock, sleep=clock.sleep, **hooks)
    running = asyncio.create_task(warden.run())
    clients: list[Client] = []
    for t in range(windows * params.window):
        for data in arrivals.get(t, ()):
            clients.append(client(warden, data))
            count = len(clients)
            await warden.until(lambda w, n=count: w.ordinals + w.refused == n)
        server.release(t)
        await warden.until(lambda w: settled(w, server))
        clock.advance(t * TAU + (late or {}).get(t, 0))
        await warden.until(lambda w, t=t: w.ticks > t)
        await warden.until(lambda w: not any(w._unreleased.values()))
    await running
    await asyncio.gather(*(c.task for c in clients))
    return warden, clients


def run(answers, arrivals, **kw):
    server = answers if isinstance(answers, Server) else Server(answers)
    warden, clients = asyncio.run(drive(server, arrivals, **kw))
    return warden, clients, server


def records(warden: Warden):
    return tuple([r for r in warden.records if r.link == link] for link in ("requests", "responses", "fails"))


def reasons(warden: Warden) -> list[tuple[int, int, str]]:
    """(bucket, ordinal, reason) of every frame of the `fails` record."""
    return [(r.window * PARAMS.window + b, *read_reason(f, B)) for r in records(warden)[2]
            for b, row in enumerate(r.grid) for f in row if f is not None]


def expected_egress(s: int, n: int, last: int = END) -> list[tuple[int, int]]:
    """(bucket, kind) of a connection's egress cells: admitted at s, n cells from s + D."""
    kinds = [PAD] * (D * R) + [DATA] * (n - 1) + [last]
    kinds += [PAD] * (-len(kinds) % R)
    return [(s + i // R, k) for i, k in enumerate(kinds)]


# --- the schedule ----------------------------------------------------------------------------------------------------


def test_honest_traffic_is_released_exactly_on_schedule_passes_bytes_unchanged_and_is_audited():
    """Four connections, requests of one cell and of several, responses empty, of exactly whole cells and of a part
    cell: each client's cells are B bytes, r a bucket from its admission, padding until s + D, then its response's
    cells in every slot and its END, and padding to the bucket's end; the server got each request and each client each
    response, byte for byte; the audit of the records accepts every window and reads back the same bytes."""
    reqs = [request(10, 1), request(3 * P + 5, 2), request(P, 3), request(1, 4)]
    resps = [bytes(range(256)) * 2, b"", bytes(3 * P), b"z" * (P + 1)]
    answers = [[(0, resp), (0, EOF)] for resp in resps]
    warden, clients, server = run(answers, {1: [cells(reqs[0], B)], 2: [cells(reqs[1], B)],
                                            3: [cells(reqs[2], B), cells(reqs[3], B)]}, windows=2)
    log = {e["ordinal"] - 1: e for e in warden.log}
    assert [server.conns[k].received for k in range(4)] == reqs
    for k, client in enumerate(clients):
        e = log[k]
        shim = client.shim()
        assert (bytes(shim.data), shim.outcome) == (resps[k], "end")
        n = max(1, -(-len(resps[k]) // P))
        assert [(t, kind) for t, kind, _ in client.cells()] == expected_egress(e["admitted"], n)
        # nothing is released at admission: the client's cells are read from then on
        assert e["ingress_end"] - e["admitted"] == 1 + (-(-len(reqs[k]) // P) - 1) // R
    # FIFO: 2 takes the first lane to free (1's), 3 the next (0's)
    assert [log[k]["admitted"] for k in range(4)] == [1, 2, log[1]["ended"] + 1, log[0]["ended"] + 1]
    assert max(warden.lateness) == 0
    verdict = audit(PARAMS, *records(warden))
    assert verdict["accepted"], verdict["windows"]
    assert [(c["ordinal"], c["admitted"], c["ingress_end"], c["ended"]) for c in verdict["connections"]] == \
        [(k + 1, log[k]["admitted"], log[k]["ingress_end"], log[k]["ended"]) for k in range(4)]
    assert [c["egress_sha256"] for c in verdict["connections"]] == [hashlib.sha256(r).hexdigest() for r in resps]
    assert [c["ingress_sha256"] for c in verdict["connections"]] == [hashlib.sha256(r).hexdigest() for r in reqs]
    assert verdict["cells"]["egress"] == sum(len(c.cells()) for c in clients)


def test_the_records_are_the_wire():
    """The egress record's frames are exactly the cells the clients got, in ordinal order a bucket."""
    warden, clients, _ = run([[(0, b"hello"), (0, EOF)], [(0, b"x" * 100), (0, EOF)]],
                             {0: [cells(request(5), B), cells(request(70), B)]})
    _, egress, _ = records(warden)
    rows = [f for r in egress for row in r.grid for f in row if f is not None]
    by_bucket = {}
    for k, client in enumerate(clients):
        for t, data in client.writes:
            by_bucket.setdefault(t, []).append(data)
    assert b"".join(rows) == b"".join(b"".join(by_bucket[t]) for t in sorted(by_bucket))


def test_the_rows_released_before_each_publish_are_its_records_grid():
    """`release` gets each grid's row as it ticks, before the window's `publish`: a publisher in another process
    rebuilds every record, bucket for bucket, from the rows alone."""
    events = []
    run([[(0, b"hello"), (0, EOF)], [(0, b"x" * 100), (0, EOF)]], {0: [cells(request(5), B), cells(request(70), B)]},
        windows=2, release=lambda link, row: events.append((link, row)), publish=lambda r, grid: events.append(r))
    pending, published = {}, []
    for e in events:
        if isinstance(e, tuple):
            pending.setdefault(e[0], []).append(e[1])
        else:
            assert e.grid == tuple(pending.pop(e.link)), (e.link, e.window)
            published.append((e.link, e.window))
    assert published == [(link, w) for w in range(2) for link in ("requests", "responses", "fails")] and not pending


# --- fail closed -----------------------------------------------------------------------------------------------------


def test_a_late_byte_fails_its_connection_closed_at_its_cell():
    """The server sends two cells' bytes and a part, then the rest far too late: the client gets the two DATA cells,
    then a FAIL in the slot of the third, padding to the bucket's end, and the connection closes; the late bytes never
    reach it, and the audit refuses the window, naming the cell and the reason the `fails` record holds."""
    answer = [(0, b"a" * (2 * P + 7)), (40, b"b" * 100), (40, EOF)]
    warden, (client,), _ = run([answer], {1: [cells(request(5), B)]})
    (e,) = warden.log
    assert (e["failed"], e["failed_at"]) == ("late", 2)
    shim = client.shim()
    assert (bytes(shim.data), shim.outcome) == (b"a" * (2 * P), "fail")
    assert [(t, k) for t, k, _ in client.cells()] == expected_egress(1, 3, FAIL)
    assert reasons(warden) == [(e["ended"], 1, "late")]
    verdict = audit(PARAMS, *records(warden))
    assert not verdict["accepted"]
    assert verdict["windows"][0]["problems"] == ["connection 1 failed closed (late) at egress cell 2"]
    assert verdict["connections"][0]["failed"] == "late"


def test_a_server_that_never_answers_fails_at_the_first_egress_slot():
    warden, (client,), _ = run([[]], {0: [cells(request(5), B)]})
    (e,) = warden.log
    assert (e["failed"], e["failed_at"], e["ended"]) == ("late", 0, e["admitted"] + D)
    assert client.shim().outcome == "fail"


def test_a_server_failure_shows_only_at_an_egress_slot():
    """A server whose connection is refused fails the connection at s + D, slot 0, like a server that is late there:
    the server chooses whether and at which egress cell a connection fails, and nothing else: the FAIL cell is the
    same bytes whatever the reason, which only the `fails` record holds."""
    server = Server([[]], refuse=(0,))
    warden, (client,), _ = run(server, {0: [cells(request(5), B)]})
    (e,) = warden.log
    assert (e["failed"], e["failed_at"], e["ended"]) == ("upstream", 0, e["admitted"] + D)
    assert [(t, k) for t, k, _ in client.cells()] == expected_egress(0, 1, FAIL)
    silent_warden, (silent,), _ = run([[]], {0: [cells(request(5), B)]})
    assert client.writes == silent.writes and client.shim().outcome == "fail"
    assert records(warden)[1] == records(silent_warden)[1]
    assert (reasons(warden), reasons(silent_warden)) == ([(D, 1, "upstream")], [(D, 1, "late")])


def test_a_server_slow_to_accept_its_connection_moves_nothing():
    """Ingress is released on schedule whether or not the server's connection has opened, and what was released
    before it opened is written once it does; egress starts at s + D whenever it opens. A server that accepts late,
    by its first egress cell's deadline, leaves the wire byte for byte as it was, and one that accepts past it fails
    at egress cell 0."""
    req, resp = request(3 * P + 5), b"r" * (2 * P + 3)
    wires = []
    for hold in ({}, {0: 2}, {0: 3}):
        warden, (client,), server = run(Server([[(0, resp), (0, EOF)]], hold=hold), {1: [cells(req, B)]})
        assert server.conns[0].received == req and client.shim().data == resp
        assert (warden.log[0]["admitted"], warden.log[0]["ingress_end"]) == (1, 3)
        wires.append(client.writes)
    assert wires[0] == wires[1] == wires[2]
    warden, (client,), _ = run(Server([[(0, resp), (0, EOF)]], hold={0: 1 + D + 1}), {1: [cells(req, B)]})
    assert (warden.log[0]["failed"], warden.log[0]["failed_at"]) == ("late", 0)
    assert client.shim().outcome == "fail"


def test_egress_past_its_cells_fails_at_the_last_and_the_rest_never_reaches_the_client():
    answer = [(0, b"q" * (PARAMS.egress_cells * P + 500)), (0, EOF)]
    warden, (client,), _ = run([answer], {0: [cells(request(5), B)]})
    (e,) = warden.log
    assert (e["failed"], e["failed_at"]) == ("limit", PARAMS.egress_cells - 1)
    assert bytes(client.shim().data) == b"q" * ((PARAMS.egress_cells - 1) * P)


def test_a_response_of_the_most_cells_ends():
    resp = b"q" * ((PARAMS.egress_cells - 1) * P + 3)
    warden, (client,), _ = run([[(0, resp), (0, EOF)]], {0: [cells(request(5), B)]})
    assert warden.log[0]["failed"] is None and bytes(client.shim().data) == resp


def test_a_client_cell_of_another_kind_or_ingress_past_its_bounds_fails_closed():
    """Each fails at once, at the next egress slot. With D at least `ingress_buckets`, a client that never ends fails
    on its ingress bound; with a shorter D it would fail at s + D, its server not having had its request."""
    params = dataclasses.replace(PARAMS, delay=PARAMS.ingress_buckets)
    bad = cell(DATA, 0, b"a" * P, B) + encode_frame(PAD, 0, b"", B)
    long = cells(b"a" * (PARAMS.ingress_cells * P + 1), B)
    slow = (cell(DATA, 0, b"a" * P, B), False)  # never ends, nor closes
    warden, clients, server = run([[(0, b"x"), (0, EOF)]] * 3, {0: [bad, long], 1: [slow]}, params=params)
    assert sorted((e["ordinal"], e["failed"]) for e in warden.log) == [(1, "malformed"), (2, "ingress"),
                                                                        (3, "ingress")]
    assert all(c.shim().outcome == "fail" for c in clients)
    assert sorted((o, why) for _, o, why in reasons(warden)) == [(1, "malformed"), (2, "ingress"), (3, "ingress")]
    assert all(b"x" not in bytes(c.shim().data) for c in clients)
    slow_log = next(e for e in warden.log if e["ordinal"] == 3)
    assert slow_log["ended"] - slow_log["admitted"] + 1 == PARAMS.ingress_buckets


def test_a_late_tick_faults_both_grids_window_and_the_audit_refuses_it():
    warden, _, _ = run([[(0, b"hi"), (0, EOF)]], {1: [cells(request(5), B)]}, windows=2, late={5: 2 * 0.025})
    statuses = {(r.link, r.window): r.status.reason for r in warden.records}
    assert statuses == {(link, w): Reason.MONITOR_UNAVAILABLE if w == 0 else None
                        for link in ("requests", "responses", "fails") for w in (0, 1)}
    assert [w["accepted"] for w in audit(PARAMS, *records(warden))["windows"]] == [False, True]


def test_a_paced_wardens_pacer_writes_each_client_its_cells_and_a_bucket_it_got_late_faults_every_grid():
    """With `paced`, each connection's cells reach its socket from the pacer, the bytes the loop writes unpaced. On the
    injected clock a bucket's time (origin 0, on the host's monotonic clock) has long passed when its tick hands it, so
    each reaches the pacer late, and the report of a window's first bucket handed faults the three grids' window
    (`monitor-unavailable`) from the next bucket on: each grid is the unpaced run's up to that bucket, then padding. Each
    window's records are published as its buckets' releases are in. Each connection's last part closes the pacer's
    copy of its socket, so each client reads to its end."""
    answers = [[(0, b"hello"), (0, EOF)], [(0, b"x" * 100), (0, EOF)]]
    arrivals = {0: [cells(request(5), B), cells(request(70), B)], 30: [cells(request(9), B)]}
    plain, plain_clients, _ = run([*answers, [(0, b"late"), (0, EOF)]], arrivals, windows=2)
    paced, clients, _ = run([*answers, [(0, b"late"), (0, EOF)]], arrivals, windows=2, paced=0.05,
                            client=SocketClient)
    assert [c.received() for c in clients] == [b"".join(d for _, d in c.writes) for c in plain_clients]
    assert [(r.link, r.window, r.status.reason) for r in paced.records] == \
        [(link, w, Reason.MONITOR_UNAVAILABLE) for w in (0, 1) for link in ("requests", "responses", "fails")]
    live = sorted({t for c in plain_clients for t, _ in c.writes})
    T = PARAMS.window

    def faulted(record):
        first = min(t for t in live if t // T == record.window) % T
        return record.grid[:first + 1] + ((None,) * len(record.grid[0]),) * (T - first - 1)

    assert [(r.link, r.window, r.grid) for r in paced.records] == \
        [(r.link, r.window, faulted(r)) for r in plain.records]
    assert [r.t for r in paced.releases] == live and all(r.written for r in paced.releases)
    with pytest.raises(ValueError, match="past the slack"):
        Warden(PARAMS, Server([]).connect, origin=0.0, windows=1, paced=TAU)


def test_a_warden_on_the_hosts_clock_is_paced_and_a_release_unwritten_faults_every_grid():
    """Only a warden on an injected clock writes each bucket on its loop: on the host's clock it must have `paced`.
    A release the pacer reports on time and written faults nothing; one it reports unwritten, a connection's socket
    not yet the pacer's so that its cells are on no wire, faults the three grids' window `monitor-unavailable`, as a
    bucket it got late does."""
    with pytest.raises(ValueError, match="host's clock is paced"):
        Warden(PARAMS, Server([]).connect, origin=0.0, windows=1)
    clock = VirtualClock(0.0)
    warden = Warden(PARAMS, Server([]).connect, origin=0.0, windows=1, clock=clock, sleep=clock.sleep)
    grids = (warden.ingress, warden.egress, warden.fails)
    warden._unreleased[0] = 2
    warden._report(Report(0, -0.04, 0.001, None, True))
    assert [g.faulted for g in grids] == [None] * 3
    warden._report(Report(1, -0.04, 0.001, None, False))
    assert [g.faulted for g in grids] == [Reason.MONITOR_UNAVAILABLE] * 3 and not warden._unreleased[0]


def test_the_audit_refuses_a_record_whose_cells_are_not_the_schedules():
    warden, _, _ = run([[(0, b"x" * 200), (0, EOF)]], {0: [cells(request(5), B)]})
    ingress, egress, fails = records(warden)
    (record,) = egress
    rows = [list(row) for row in record.grid]
    t = next(t for t, row in enumerate(rows) if any(f is not None and f[0] == DATA for f in row))
    rows[t + 1], rows[t] = rows[t], rows[t + 1]  # a bucket's cells a bucket late
    moved = dataclasses.replace(record, grid=tuple(tuple(row) for row in rows))
    verdict = audit(PARAMS, ingress, [moved], fails)
    assert not verdict["accepted"] and "egress isn't padding to s + D" in verdict["windows"][0]["problems"][0]


# --- admission -------------------------------------------------------------------------------------------------------


def test_over_capacity_queues_fifo_then_refuses():
    """Two lanes and a queue of two: two connections take the lanes, and of three more the next bucket two wait and
    are admitted in order as lanes free, and the third gets one REFUSED cell and takes no ordinal."""
    answers = [[(0, b"r%d" % k), (0, EOF)] for k in range(4)]
    warden, clients, server = run(answers, {1: [cells(request(5, k), B) for k in range(2)],
                                            2: [cells(request(5, k), B) for k in range(2, 5)]})
    assert warden.refused == 1 and warden.ordinals == 4
    refused = clients[4]
    assert [(k, len(p)) for _, k, p in refused.cells()] == [(REFUSED, 0)] and refused.shim().outcome == "refused"
    log = sorted(warden.log, key=lambda e: e["ordinal"])
    assert [e["admitted"] for e in log] == [1, 1, log[0]["ended"] + 1, log[1]["ended"] + 1]
    assert [bytes(c.shim().data) for c in clients[:4]] == [b"r0", b"r1", b"r2", b"r3"]
    assert [conn.received for conn in server.conns] == [request(5, k) for k in range(4)]
    assert warden.refused_windows == {0} and not warden.records
    assert [(r.link, r.window) for r in warden.withheld] == [("requests", 0), ("responses", 0), ("fails", 0)]


def test_a_window_with_a_refusal_goes_to_withhold_whole():
    """The warden certifies no record of a window in which it refused a connection (traffic past its Program slots):
    of two windows, the second's refusal, just before its second bucket, gives its three records to `withhold` in place
    of `publish`, and the first's three are published."""
    T = PARAMS.window
    answers = [[(0, b"r%d" % k), (0, EOF)] for k in range(4)]
    warden, clients, _ = run(answers, {T: [cells(request(5, k), B) for k in range(2)],
                                       T + 1: [cells(request(5, k), B) for k in range(2, 5)]}, windows=2)
    assert warden.refused == 1 and warden.refused_windows == {1}
    assert [(r.link, r.window) for r in warden.records] == [("requests", 0), ("responses", 0), ("fails", 0)]
    assert [(r.link, r.window) for r in warden.withheld] == [("requests", 1), ("responses", 1), ("fails", 1)]
    assert [bytes(c.shim().data) for c in clients[:4]] == [b"r0", b"r1", b"r2", b"r3"]


def test_every_divides_the_window():
    assert (dataclasses.replace(PARAMS, every=8).program_slots, dataclasses.replace(PARAMS, every=8).window_programs) \
        == (PARAMS.lanes, 3 * PARAMS.lanes)
    with pytest.raises(ValueError, match="divides the window"):
        dataclasses.replace(PARAMS, every=5)


# --- the server's timing never reaches the wire ----------------------------------------------------------------------


def test_the_wire_is_a_function_of_admission_and_egress_length_alone():
    """The same request and the same number of egress cells from servers that time and chunk their bytes differently,
    and send different bytes: every client gets the same cells, B bytes each, of the same kinds, in the same buckets.
    Padding and data cells are the same size, so a cell's size says nothing."""
    resp = bytes(range(256)) * 2
    other = b"\0" * (len(resp) - 1) + b"\xff"
    # 37 bytes three times a bucket from the request's bucket: ahead of the R·P = 110 a bucket drained from s + D
    trickle = [(i // 111, resp[i:i + 37]) for i in range(0, len(resp), 37)] + [(len(resp) // 111, EOF)]
    burst = [(0, resp[:3]), (D - 1, resp[3:]), (D - 1, EOF)]
    wires = []
    for answer in ([(0, resp), (0, EOF)], trickle, burst, [(0, other), (0, EOF)]):
        warden, (client,), _ = run([answer], {2: [cells(request(30), B)]})
        assert warden.log[0]["failed"] is None, answer
        wires.append([(t, k, len(c)) for t, k, c in client.cells()])
        assert {len(d) % B for _, d in client.writes} == {0}
    assert all(w == wires[0] for w in wires)
    assert {len(cell(k, 0, b"", B)) for k in (PAD, END, REFUSED)} == {len(cell(DATA, 0, b"a" * P, B))} == {B}


def test_cells_cut_bytes_the_same_way_however_they_arrive():
    for n in (0, 1, P - 1, P, P + 1, 3 * P):
        data = bytes(i % 251 for i in range(n))
        got = cells(data, B, 7)
        parsed = [read_cell(got[i:i + B], B) for i in range(0, len(got), B)]
        assert [k for k, _, _ in parsed] == [DATA] * (len(parsed) - 1) + [END]
        assert b"".join(p for _, _, p in parsed) == data and {o for _, o, _ in parsed} == {7}
        assert len(parsed) == max(1, -(-n // P))


def test_read_cell_refuses_what_the_warden_would_not_make():
    good = cell(DATA, 3, b"a" * P, B)
    assert read_cell(good, B) == (DATA, 3, b"a" * P)
    end = cell(END, 3, b"a", B)
    for bad in (good[:-1], end[:-1] + b"\x01", cell(DATA, 3, b"a", B), encode_frame(9, 3, b"", B),
                cell(PAD, 3, b"x", B), cell(FAIL, 3, b"\x01", B), reason_frame(3, "late", B),
                HEADER.pack(END, 3, P + 1) + bytes(B - HEADER.size)):
        with pytest.raises(ValueError):
            read_cell(bad, B)
    assert all(read_reason(reason_frame(5, why, B), B) == (5, why) for why in REASONS)
    for bad in (cell(FAIL, 5, b"", B), encode_frame(FAIL, 5, bytes([len(REASONS)]), B), end):
        with pytest.raises(ValueError):
            read_reason(bad, B)


# --- the release order -----------------------------------------------------------------------------------------------


def http(body: bytes, extra: bytes = b"") -> bytes:
    return b"POST /v1/completions HTTP/1.1\r\nhost: s\r\n%scontent-length: %d\r\n\r\n%s" % (extra, len(body), body)


def test_the_server_gets_each_request_whole_with_its_place_among_the_signed_ends():
    """Two requests admitted in one bucket, the first two buckets long and the second one, so their `END`s are
    released in the other order: each reaches the server once, whole, at its `END`'s release, with
    `x-verity-release: N`, N its `END`'s index among the `END` cells of the signed ingress records, and the warden's
    run id; a client's own release header is dropped there, while the cells, the records and the audit hold the
    client's bytes as sent, which the audit accepts."""
    reqs = [http(b'{"prompt":"%s"}' % (b"a" * 140)), http(b"{}", b"X-Verity-Release: 0\r\n")]
    assert [-(-len(r) // P) for r in reqs] == [2 * R, R]
    run_id = bytes(range(100, 132))
    warden, clients, server = run([[(0, b"r0"), (0, EOF)], [(0, b"r1"), (0, EOF)]],
                                  {1: [cells(reqs[0], B), cells(reqs[1], B)]},
                                  run_nonce=lambda n: run_id[:n])
    ingress, _, _ = records(warden)
    ends = [read_cell(f, B)[1] for r in ingress for row in r.grid for f in row
            if f is not None and read_cell(f, B)[0] == END]
    assert ends == [2, 1] and warden.run_id == run_id and warden.requests_released == 2
    log = {e["ordinal"]: e for e in warden.log}
    assert log[1]["admitted"] == log[2]["admitted"] == 1 and log[2]["ingress_end"] < log[1]["ingress_end"]
    for k, req in enumerate(reqs):
        assert server.conns[k].received == with_release(req, ends.index(k + 1), run_id) == \
            with_release(req, log[k + 1]["release"], run_id)
    assert release_of(server.conns[1].received) == (0, run_id) and b"X-Verity" not in server.conns[1].received
    verdict = audit(PARAMS, *records(warden))
    assert verdict["accepted"], verdict["windows"]
    assert [c["ingress_sha256"] for c in verdict["connections"]] == [hashlib.sha256(r).hexdigest() for r in reqs]
    assert [bytes(c.shim().data) for c in clients] == [b"r0", b"r1"]


def test_a_connection_that_fails_before_its_end_is_never_numbered_and_its_server_gets_nothing():
    """A client that never ends fails at s + D with nothing sent to its server, and takes no place in the order: the
    next request's N is 0. Each warden draws its own run id."""
    slow = (cell(DATA, 0, b"a" * P, B), False)
    warden, _, server = run([[], [(0, b"r"), (0, EOF)]], {0: [slow], 4: [cells(http(b"{}"), B)]})
    log = {e["ordinal"]: e for e in warden.log}
    assert (log[1]["failed"], log[1]["release"], log[2]["failed"], log[2]["release"]) == ("late", None, None, 0)
    assert server.conns[0].received == b"" and release_of(server.conns[1].received) == (0, warden.run_id)
    other, _, _ = run([[(0, b"r"), (0, EOF)]], {0: [cells(http(b"{}"), B)]})
    assert len(warden.run_id) == len(other.run_id) == 32 and warden.run_id != other.run_id


def test_the_parameters_digest_binds_every_parameter():
    fields = [f.name for f in dataclasses.fields(WardenParams)]
    digests = {PARAMS.digest} | {dataclasses.replace(PARAMS, **{f: getattr(PARAMS, f) + 1}).digest for f in fields}
    assert len(digests) == len(fields) + 1 and all(len(d) == 64 for d in digests)
    assert set(KIND_NAMES) == {PAD, DATA, END, FAIL, REFUSED} and len(set(REASONS)) == len(REASONS)
