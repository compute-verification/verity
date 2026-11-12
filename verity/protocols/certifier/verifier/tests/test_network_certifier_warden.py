"""The active network certifier (`warden`): driven tick by tick on a `VirtualClock`, its clients in-memory connections
speaking cells and its server an in-memory one whose bytes the test schedules, it releases honest traffic exactly on its
schedule, fails a late byte closed, queues past its lanes then refuses (and withholds that window's records from
`publish`), passes both directions' bytes unchanged (the server's copy of a request adding only its place in the release
order), and puts the same number of same-size cells in the same buckets however the server times its bytes. The
schedule's tests relay any bytes, in the bytes egress form (`PARAMS`); in the id form (`GATED`) it releases an honest
id-form server's cells and fails any other at its first cell (format), none of whose bytes leaves."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import socket

import pytest

from verity.protocols.certifier.verifier import active, io_tie as IT
from verity.protocols.certifier.verifier.active import DATA, HEADER, PAD, VirtualClock, encode_frame
from verity.protocols.certifier.verifier.grid import Reason
from verity.protocols.certifier.verifier.release import read_watermark, release_of, watermark_head, with_release
from verity.protocols.certifier.verifier.warden import (END, FAIL, INTEGER_FIELDS, KIND_NAMES, PARAMETERS_TAG, REASONS,
                                                    REFUSED, WRITE_BACKLOG, Shim, Warden, WardenParams, audit, cell,
                                                    cells, read_cell, read_reason, reason_frame)
from verity.commitments.identity import canonical_json_bytes, tagged_sha512

PARAMS = WardenParams(cell_bytes=64, slots=2, bucket_ms=100, delay=3, lanes=2, queue=2, window=24, ingress_cells=8,
                      ingress_buckets=6, egress_cells=12, egress="bytes")
B, P, R, D = PARAMS.cell_bytes, PARAMS.payload, PARAMS.slots, PARAMS.delay
#: the id egress form at cells that hold an id event: PARAMS' schedule otherwise
GATED = dataclasses.replace(PARAMS, cell_bytes=256, egress="id")
GB, GP = GATED.cell_bytes, GATED.payload
TAU = PARAMS.bucket_ms / 1000
EOF = None


def request(n: int, tag: int = 0) -> bytes:
    """A request of n bytes ending in its newline, every byte value in it."""
    return bytes((tag + i * 37) % 256 for i in range(n - 1)).replace(b"\n", b"\x0b") + b"\n"


class Server:
    """An in-memory server: connection k (in connect order, which is admission order) answers once it has received a
    newline, with `answers[k]`, a list of (buckets after the request's bucket, bytes or EOF); `release(t)` feeds what is
    due by bucket t. `refuse` fails the connects listed, `hold[k]` opens connection k only at bucket `hold[k]`,
    `close_at[k]` closes it at that bucket, and `flush_at[k]` keeps its transport's buffer unemptied, and so its request
    unread, until then."""

    def __init__(self, answers: list[list], refuse: tuple[int, ...] = (), hold: dict[int, int] | None = None,
                 close_at: dict[int, int] | None = None, flush_at: dict[int, int] | None = None) -> None:
        self.answers, self.refuse, self.hold = answers, refuse, hold or {}
        self.close_at, self.flush_at = close_at or {}, flush_at or {}
        self.conns: list[ServerConn] = []
        self.held: dict[int, asyncio.Event] = {}

    async def connect(self):
        k = len(self.conns)
        conn = ServerConn(self.answers[k] if k < len(self.answers) else [])
        conn.unflushed = int(k in self.flush_at)
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
        for k, conn in enumerate(self.conns):
            conn.closed |= t >= self.close_at.get(k, t + 1)
            conn.unflushed *= t < self.flush_at.get(k, t + 1)
            conn.release(t)


class ServerConn:
    def __init__(self, answer: list) -> None:
        self.answer, self.reader, self.received = list(answer), asyncio.StreamReader(), bytearray()
        self.at: int | None = None  # the bucket its request ended in
        self.fed, self.eof, self.closed = 0, False, False
        self.transport, self.unflushed = self, 0

    def get_write_buffer_size(self) -> int:
        return self.unflushed

    def write(self, data: bytes) -> None:
        self.received += data
        if self.at is None and b"\n" in data:
            self.at = -1

    def release(self, t: int) -> None:
        if self.at == -1 and not self.unflushed:
            self.at = t - 1
        while self.at is not None and self.at >= 0 and self.answer and self.at + self.answer[0][0] <= t \
                and not self.closed:
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
        out, B = [], self.warden.params.cell_bytes
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
            return await Shim(self.warden.params.cell_bytes).read(reader)
        return asyncio.run(read())


class SocketClient(Client):
    """A client whose connection's network certifier end is a socket, the one a paced network certifier's pacer writes
    to, and whose other end the test reads (`received`) once the run is over."""

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
    cap = warden.params.egress_cells * warden.params.payload
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
    due and the network certifier read it, then tick, and with `paced` wait for the pacer to report the bucket's
    release, for `windows` windows; `late[t]` puts tick t that many seconds late. `hooks` are the network certifier's
    `publish`, `release` and `paced`."""
    clock = VirtualClock(-TAU / 2)
    warden = Warden(params, server.connect, origin=0.0, windows=windows, clock=clock, sleep=clock.sleep, **hooks)
    running = asyncio.create_task(warden.run())
    clients: list[Client] = []
    for t in range(windows * params.window):
        for data in arrivals.get(t, ()):
            clients.append(client(warden, data))
            count = len(clients)
            await warden.until(lambda w, n=count: w.ordinals + w.refused + w.turned_away == n)
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
    p = warden.params
    return [(r.window * p.window + b, *read_reason(f, p.cell_bytes)) for r in records(warden)[2]
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


# --- the egress form -------------------------------------------------------------------------------------------------


def id_answer(ids: list[int]) -> bytes:
    """An honest id-form response: one `io_tie.id_event` of P bytes an output id, then `io_tie.TRAILER`."""
    return b"".join(IT.id_event(x, GP) for x in ids) + IT.TRAILER


def live_cells(client: Client) -> list[bytes]:
    """The client's cells other than padding, as bytes."""
    data = b"".join(d for _, d in client.writes)
    return [c for c in (data[i:i + GB] for i in range(0, len(data), GB)) if c[0] != PAD]


def test_an_honest_id_form_server_leaves_one_event_a_cell_and_its_end_holds_the_trailer():
    """In the id form an honest server's response, however its writes split it, leaves as one `io_tie.data_cell` an
    output id on the connection's ordinal, then the `io_tie.end_cell`, each in its slot of the schedule; a response of
    no outputs is its `END` alone. The bytes form puts out the same cells, and the audit accepts the records."""
    ids = [0, 7, 42, 1 << 31, (1 << 32) - 1]
    body = id_answer(ids)
    answers = [[(0, body[:100]), (0, body[100:600]), (1, body[600:]), (1, EOF)], [(0, IT.TRAILER), (0, EOF)]]
    arrivals = {1: [cells(request(5), GB), cells(request(9, 1), GB)]}
    warden, clients, _ = run(answers, arrivals, params=GATED)
    assert [(e["failed"], e["egress_cells"]) for e in sorted(warden.log, key=lambda e: e["ordinal"])] == \
        [(None, len(ids) + 1), (None, 1)]
    for q, (client, outputs) in enumerate(zip(clients, (ids, [])), 1):
        assert live_cells(client) == [IT.data_cell(x, q, GB) for x in outputs] + [IT.end_cell(q, GB)]
        assert [(t, k) for t, k, _ in client.cells()] == expected_egress(1, len(outputs) + 1)
    verdict = audit(GATED, *records(warden))
    assert verdict["accepted"], verdict["windows"]
    _, plain, _ = run(answers, arrivals, params=dataclasses.replace(GATED, egress="bytes"))
    assert [c.writes for c in plain] == [c.writes for c in clients]


def departures() -> dict[str, tuple[bytes, int]]:
    """Responses out of the id form, each with the egress cell it first leaves the form at."""
    e = [IT.id_event(x, GP) for x in (3, 4, 5)]
    ordinal = active.id_event(1, 4)
    spaced = e[1].replace(b'"token_id":', b'"token_id": ')
    named = b'"id":"cmpl-1",'
    error = b'data: {"error":{"message":"overloaded","type":"server_error","code":503}}\n\n'
    return {
        "an HTTP head": (b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n" + b"".join(e) + IT.TRAILER, 0),
        "an HTTP error": (b"HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\n\r\n", 0),
        "a created time": (e[0] + e[1].replace(b'"created":0', b'"created":1') + e[2] + IT.TRAILER, 1),
        "a response id": (e[0] + e[1][:7] + named + e[1][7:-2 - len(named)] + b"\n\n" + e[2] + IT.TRAILER, 1),
        "the ordinal event": (e[0] + ordinal[:-2].ljust(GP - 2) + b"\n\n" + e[2] + IT.TRAILER, 1),
        "a non-canonical id": (e[0] + e[1].replace(b"         4", b"0000000004") + e[2] + IT.TRAILER, 1),
        "extra spaces": (e[0] + spaced[:-3] + b"\n\n" + e[2] + IT.TRAILER, 1),
        "a chunk split across cells": (e[0] + IT.id_event(4, GP + 1) + e[2] + IT.TRAILER, 1),
        "an error body": (e[0] + error + IT.TRAILER, 1),
        "no trailer": (e[0] + e[1], 1),
        "bytes past the trailer": (b"".join(e) + IT.TRAILER + b"x", 3),
    }


@pytest.mark.parametrize("name", sorted(departures()))
def test_each_departure_from_the_id_form_fails_format_at_its_cell_and_none_of_its_bytes_leaves(name):
    """In the id form a response that leaves the form fails its connection at the first cell out of it (format), whose
    bytes never leave: the client gets the response's cells before it, each the id form's, then one `FAIL` in that
    cell's slot and padding, byte for byte what a server late at that cell gives it, so the server picks one of the
    same (cells + 1) outcomes and the leak bound's failure charge is unchanged. The `fails` record holds the reason and
    the audit names it. The bytes form relays the same response whole."""
    response, k = departures()[name]
    assert len(response) >= k * GP and all(IT.parse_cell(cell(DATA, 1, response[i * GP:(i + 1) * GP], GB))
                                           for i in range(k))
    warden, (client,), _ = run([[(0, response), (0, EOF)]], {1: [cells(request(5), GB)]}, params=GATED)
    (e,) = warden.log
    assert (e["failed"], e["failed_at"]) == ("format", k)
    assert live_cells(client) == [cell(DATA, 1, response[i * GP:(i + 1) * GP], GB) for i in range(k)] + \
        [cell(FAIL, 1, b"", GB)]
    assert [(t, kind) for t, kind, _ in client.cells()] == expected_egress(1, k + 1, FAIL)
    shim = client.shim()
    assert (bytes(shim.data), shim.outcome) == (response[:k * GP], "fail")
    assert reasons(warden) == [(e["ended"], 1, "format")]
    egress = [f for r in records(warden)[1] for row in r.grid for f in row if f is not None]
    assert response[k * GP:(k + 1) * GP] not in b"".join(egress)
    verdict = audit(GATED, *records(warden))
    assert verdict["windows"][0]["problems"] == [f"connection 1 failed closed (format) at egress cell {k}"]

    _, (late,), _ = run([[(0, response[:k * GP]), (40, response[k * GP:]), (40, EOF)]], {1: [cells(request(5), GB)]},
                        params=GATED)
    assert late.writes == client.writes

    _, (plain,), _ = run([[(0, response), (0, EOF)]], {1: [cells(request(5), GB)]},
                         params=dataclasses.replace(GATED, egress="bytes"))
    assert (bytes(plain.shim().data), plain.shim().outcome) == (response, "end")


def test_the_id_form_releases_no_cell_smaller_than_an_event():
    """A cell of fewer than `io_tie.CELL_USED` bytes holds no id event, so in the id form at such a B every connection
    fails at its first egress cell, the trailer alone included."""
    assert B < IT.CELL_USED <= GB
    warden, (client,), _ = run([[(0, IT.TRAILER), (0, EOF)]], {1: [cells(request(5), B)]},
                               params=dataclasses.replace(PARAMS, egress="id"))
    assert [(e["failed"], e["failed_at"]) for e in warden.log] == [("format", 0)]


def test_the_audit_refuses_an_egress_cell_out_of_the_id_form():
    """A record whose egress holds a `DATA` cell `io_tie.parse_cell` doesn't read is refused in the id form, which the
    network certifier never sends, and read as any bytes in the bytes form."""
    warden, _, _ = run([[(0, id_answer([1, 2])), (0, EOF)]], {1: [cells(request(5), GB)]}, params=GATED)
    ingress, (record,), fails = records(warden)
    t, j = next((t, j) for t, row in enumerate(record.grid) for j, f in enumerate(row)
                if f is not None and f[0] == DATA)
    row = list(record.grid[t])
    row[j] = cell(DATA, 1, b"x" * GP, GB)
    forged = dataclasses.replace(record, grid=record.grid[:t] + (tuple(row),) + record.grid[t + 1:])
    assert audit(GATED, ingress, [record], fails)["accepted"]
    verdict = audit(GATED, ingress, [forged], fails)
    assert verdict["windows"][0]["problems"] == [f"egress bucket {t}: a cell out of the id egress form"]
    assert audit(dataclasses.replace(GATED, egress="bytes"), ingress, [forged], fails)["accepted"]


# --- the response cap ------------------------------------------------------------------------------------------------

#: GATED with a response cap of four tokens
CAPPED = dataclasses.replace(GATED, max_tokens_cap=4)


def id_req(max_tokens: int) -> bytes:
    """An id request in the network certifier's cells (offset 0), of two cells at GB."""
    return active.id_request(model="m", user="u", prompt=[1, 2, 3], max_tokens=max_tokens, frame_bytes=GB, offset=0)


def capped(request_bytes: bytes, response: bytes, params: WardenParams = CAPPED):
    warden, (client,), server = run([[(0, response), (0, EOF)]], {1: [cells(request_bytes, GB)]}, params=params)
    (e,) = warden.log
    return warden, client, server, e


def test_a_request_at_the_cap_passes_and_its_response_is_released():
    """Under a cap of four a request of `max_tokens` four reaches the server, and its response of four ids, or fewer,
    leaves as the id form's cells; the log and the audit name the request's `max_tokens` and the schedule's cap, and
    the audit accepts."""
    for ids in ([1, 2, 3, 4], [9]):
        warden, client, server, e = capped(id_req(4), id_answer(ids))
        assert (e["failed"], e["max_tokens"], e["egress_cells"]) == (None, 4, len(ids) + 1)
        assert server.conns[0].received and live_cells(client) == [IT.data_cell(x, 1, GB) for x in ids] + \
            [IT.end_cell(1, GB)]
        verdict = audit(CAPPED, *records(warden))
        assert verdict["accepted"], verdict["windows"]
        assert verdict["max_tokens_cap"] == 4 and [c["max_tokens"] for c in verdict["connections"]] == [4]
    assert active.id_request_max_tokens(id_req(4)) == 4 and len(id_req(4)) > GP


def refused(request_bytes: bytes) -> None:
    """The request is refused under CAPPED: none of its bytes is cut into a cell or reaches the server, the client gets
    one `FAIL` at once (in the bucket its first cell's bytes are read in, the one after its admission, before s + D),
    the `fails` record holds `cap`, and the audit names it."""
    warden, client, server, e = capped(request_bytes, id_answer([1]))
    assert (e["failed"], e["failed_at"], e["ended"], e["release"]) == ("cap", 0, e["admitted"] + 1, None)
    assert server.conns[0].received == b"" and live_cells(client) == [cell(FAIL, 1, b"", GB)]
    assert client.shim().outcome == "fail"
    ingress = [f for r in records(warden)[0] for row in r.grid for f in row if f is not None]
    assert ingress and all(f[0] == PAD for f in ingress)
    assert reasons(warden) == [(e["ended"], 1, "cap")]
    verdict = audit(CAPPED, *records(warden))
    assert verdict["windows"][0]["problems"] == ["connection 1 failed closed (cap) at egress cell 0"]
    assert [c["max_tokens"] for c in verdict["connections"]] == [None]


def test_a_request_past_the_cap_is_refused_at_ingress_with_its_own_reason_never_clamped():
    """cap + 1 is refused (cap): the network certifier never rewrites the field to the cap, it sends nothing."""
    refused(id_req(5))
    refused(id_req((1 << 32) - 1))


def malformed_requests() -> dict[str, bytes]:
    """Requests whose `max_tokens` the network certifier can't read as `active.id_request` writes one."""
    good, at = id_req(4), active.MAX_TOKENS_AT
    field = lambda f: good[:at] + f + good[at + active.ID_DIGITS:]  # noqa: E731
    return {
        "a leading zero": field(b"0000000004"), "a sign": field(b"        -4"), "not digits": field(b"      four"),
        "past a u32": field(b"4294967296"), "no digits": field(b" " * active.ID_DIGITS),
        "another key": good.replace(b'{"max_tokens":', b'{"max_tokenz":'), "not an id request": request(300),
        "a short request": good[:at + 4], "no request": b"",
        "a content-length that isn't a u32": good.replace(b"content-length:", b"content-length:x", 1)[:len(good)],
    }


@pytest.mark.parametrize("name", sorted(malformed_requests()))
def test_a_request_whose_max_tokens_field_does_not_parse_is_refused(name):
    """A request whose field isn't a u32 at its place is refused (cap) as one past the cap is."""
    request_bytes = malformed_requests()[name]
    assert active.id_request_max_tokens(request_bytes[:GP]) is None
    refused(request_bytes)


def test_a_response_past_its_requests_max_tokens_fails_limit_at_that_cell():
    """Under the cap a server is held to its request's `max_tokens`: a third id on a request of two fails the
    connection (limit) at egress cell 2, none of that id's bytes leaving, and a request of zero gets its END alone or
    fails at cell 0. Without a cap the same response is released whole."""
    for max_tokens, ids, outcome in ((2, [1, 2, 3], ("limit", 2)), (2, [1, 2], (None, None)), (0, [], (None, None)),
                                     (0, [7], ("limit", 0))):
        warden, client, _, e = capped(id_req(max_tokens), id_answer(ids))
        assert (e["failed"], e["failed_at"]) == outcome
        sent = ids[:max_tokens]
        assert live_cells(client) == [IT.data_cell(x, 1, GB) for x in sent] + \
            [cell(FAIL, 1, b"", GB) if outcome[0] else IT.end_cell(1, GB)]
        verdict = audit(CAPPED, *records(warden))
        assert verdict["accepted"] == (outcome[0] is None)
        if outcome[0]:
            assert reasons(warden) == [(e["ended"], 1, "limit")]
    _, client, _, e = capped(id_req(2), id_answer([1, 2, 3]), params=GATED)
    assert (e["failed"], e["max_tokens"]) == (None, None) and len(live_cells(client)) == 4


def test_the_audit_holds_released_requests_and_responses_to_the_cap():
    """The audit under a cap reads each request's `max_tokens` from its cells: records an uncapped run made, with a
    request past the cap released or a response past its request's `max_tokens`, are refused; a `cap` reason under a
    schedule without a cap is not the schedule's."""
    warden, _, _, _ = capped(id_req(5), id_answer([1]), params=GATED)
    verdict = audit(CAPPED, *records(warden))
    assert verdict["windows"][0]["problems"] == ["connection 1: its request's max_tokens (5) is past the cap, and was "
                                                 "released"]
    warden, _, _, _ = capped(id_req(2), id_answer([1, 2, 3]), params=GATED)
    verdict = audit(CAPPED, *records(warden))
    assert verdict["windows"][0]["problems"] == ["connection 1: its response has 3 DATA cells, past its max_tokens (2)"]
    warden, _, _, _ = capped(id_req(5), id_answer([1]))
    assert "fails bucket 2: cap, not this schedule's" in audit(GATED, *records(warden))["windows"][0]["problems"]


def test_the_cap_is_a_u32_below_egress_cells_in_the_id_form():
    """A cap is a positive u32 below `egress_cells` (a capped response's END is its egress cell `max_tokens`), only in
    the id egress form, at cells that hold an id request's `max_tokens` field."""
    for bad in (0, -1, True, 1.0, "4", GATED.egress_cells, 1 << 32):
        with pytest.raises(ValueError, match="max_tokens_cap is a u32"):
            dataclasses.replace(GATED, max_tokens_cap=bad)
    with pytest.raises(ValueError, match="egress form is id"):
        dataclasses.replace(GATED, egress="bytes", max_tokens_cap=4)
    with pytest.raises(ValueError, match="doesn't hold an id request's max_tokens"):
        dataclasses.replace(PARAMS, egress="id", max_tokens_cap=4)
    assert dataclasses.replace(GATED, max_tokens_cap=GATED.egress_cells - 1).max_tokens_cap == 11


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
    """The network certifier certifies no record of a window in which it refused a connection (traffic past its Program
    slots): of two windows, the second's refusal, just before its second bucket, gives its three records to `withhold`
    in place of `publish`, and the first's three are published."""
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


def signed_ends(warden: Warden) -> list[tuple[int, int]]:
    """(bucket, ordinal) of every `END` cell of the signed ingress records, in record order."""
    return [(r.window * PARAMS.window + b, read_cell(f, B)[1]) for r in records(warden)[0]
            for b, row in enumerate(r.grid) for f in row if f is not None and read_cell(f, B)[0] == END]


def test_the_server_gets_each_request_whole_with_its_place_among_the_signed_ends():
    """Two requests admitted in one bucket, the first two buckets long and the second one, so their `END`s are
    released in the other order: each reaches the server once, whole, at its `END`'s release, with
    `x-verity-release: N`, N its `END`'s index among the `END` cells of the signed ingress records, the network
    certifier's run id and `x-verity-release-bucket: a`, the bucket of that `END`; a client's own release headers are
    dropped there, while the cells, the records and the audit hold the client's bytes as sent, which the audit
    accepts."""
    reqs = [http(b'{"prompt":"%s"}' % (b"a" * 112), b"x-verity-release-bucket: 0\r\n"),
            http(b"{}", b"X-Verity-Release: 0\r\n")]
    assert [-(-len(r) // P) for r in reqs] == [2 * R, R]
    run_id = bytes(range(100, 132))
    warden, clients, server = run([[(0, b"r0"), (0, EOF)], [(0, b"r1"), (0, EOF)]],
                                  {1: [cells(reqs[0], B), cells(reqs[1], B)]},
                                  run_nonce=lambda n: run_id[:n])
    ends = signed_ends(warden)
    assert [o for _, o in ends] == [2, 1] and warden.run_id == run_id and warden.requests_released == 2
    log = {e["ordinal"]: e for e in warden.log}
    assert log[1]["admitted"] == log[2]["admitted"] == 1 and log[2]["ingress_end"] < log[1]["ingress_end"]
    for k, req in enumerate(reqs):
        n = [o for _, o in ends].index(k + 1)
        assert server.conns[k].received == with_release(req, n, run_id, ends[n][0]) == \
            with_release(req, log[k + 1]["release"], run_id, log[k + 1]["ingress_end"])
    assert release_of(server.conns[1].received) == (0, run_id, log[2]["ingress_end"]) == (0, run_id, ends[0][0])
    assert b"X-Verity" not in server.conns[1].received
    assert release_of(server.conns[0].received) == (1, run_id, log[1]["ingress_end"]) == (1, run_id, ends[1][0])
    assert server.conns[0].received.count(b"x-verity-release-bucket") == 1
    verdict = audit(PARAMS, *records(warden))
    assert verdict["accepted"], verdict["windows"]
    assert [c["ingress_sha256"] for c in verdict["connections"]] == [hashlib.sha256(r).hexdigest() for r in reqs]
    assert [bytes(c.shim().data) for c in clients] == [b"r0", b"r1"]


def test_a_connection_that_fails_before_its_end_is_never_numbered_and_its_server_gets_nothing():
    """A client that never ends fails at s + D with nothing sent to its server, and takes no place in the order: the
    next request's N is 0. Each network certifier samples its own run id."""
    slow = (cell(DATA, 0, b"a" * P, B), False)
    warden, _, server = run([[], [(0, b"r"), (0, EOF)]], {0: [slow], 4: [cells(http(b"{}"), B)]})
    log = {e["ordinal"]: e for e in warden.log}
    assert (log[1]["failed"], log[1]["release"], log[2]["failed"], log[2]["release"]) == ("late", None, None, 0)
    assert server.conns[0].received == b"" and \
        release_of(server.conns[1].received) == (0, warden.run_id, log[2]["ingress_end"])
    other, _, _ = run([[(0, b"r"), (0, EOF)]], {0: [cells(http(b"{}"), B)]})
    assert len(warden.run_id) == len(other.run_id) == 32 and warden.run_id != other.run_id


# --- the watermark ---------------------------------------------------------------------------------------------------


class Ordered(Server):
    """`Server` whose connections' writes and the watermark stream's go in one list, `events`, in the order the network
    certifier made them: ("request", connection, bytes) and ("watermark", bytes), with ("bucket", t) where bucket t
    became due. Its `watermark` connect opens once bucket `opens` is due (at once without it), or is refused, and its
    transport reports `buffered` bytes unwritten."""

    def __init__(self, answers, *, opens: int | None = None, refuse_watermark: bool = False, buffered: int = 0,
                 **kw) -> None:
        super().__init__(answers, **kw)
        self.events: list[tuple] = []
        self.opens, self.refuse_watermark, self.buffered = opens, refuse_watermark, buffered
        self.opened: asyncio.Event | None = None
        self.transport, self.closed = self, False

    async def connect(self):
        k = len(self.conns)
        reader, conn = await super().connect()
        write = conn.write

        def logged(data: bytes) -> None:
            self.events.append(("request", k, bytes(data)))
            write(data)

        conn.write = logged
        return reader, conn

    async def watermark(self):
        self.opened = asyncio.Event()
        if self.opens is not None:
            await self.opened.wait()
        if self.refuse_watermark:
            raise ConnectionRefusedError("refused")
        return asyncio.StreamReader(), self

    def release(self, t: int) -> None:
        self.events.append(("bucket", t))
        if self.opened is not None and self.opens is not None and t >= self.opens:
            self.opened.set()
        super().release(t)

    def write(self, data: bytes) -> None:
        assert not self.closed
        self.events.append(("watermark", bytes(data)))

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    def get_write_buffer_size(self) -> int:
        return self.buffered


def watermark_lines(data: bytes, run_id: bytes) -> tuple[list[bytes], bool]:
    """The lines of one write to the watermark stream (whole chunks, after the head when it opens the stream), and
    whether it ends the stream."""
    if data.startswith(b"POST "):
        assert data.startswith(watermark_head(run_id))
        data = data[len(watermark_head(run_id)):]
    lines = []
    while data:
        size, _, data = data.partition(b"\r\n")
        k = int(size, 16)
        if k == 0:
            assert data == b"\r\n"
            return lines, True
        assert data[k:k + 2] == b"\r\n"
        lines.append(data[:k])
        data = data[k + 2:]
    return lines, False


def tick_engine(events: list[tuple], run_id: bytes) -> tuple[dict[int, list[int]], dict[int, int], list, bool, list]:
    """An engine that admits at logical time T exactly the releases the watermark closes by bucket T, reading the
    events in order: line T once it has been read and every release N < n_T not named lost on a line ≤ T has
    arrived, the releases n_{T−1} … n_T − 1 not named lost, in order, each admitted at a T no earlier than its bucket;
    a release named lost never arrives after its line. Returns what it admitted at each T, each release's bucket, the
    lines, whether the stream ended and the lines read before every release they count had arrived or been named
    lost."""
    buckets: dict[int, int] = {}
    lines: list[tuple[int, int, tuple[int, ...]]] = []
    lost: set[int] = set()
    admitted: dict[int, list[int]] = {}
    ended, early = False, []
    for event in events:
        if event[0] == "request":
            n, run, a = release_of(event[2])
            assert run == run_id and n not in buckets and n not in lost and a is not None
            buckets[n] = a
        elif event[0] == "watermark":
            assert not ended
            got, ended = watermark_lines(event[1], run_id)
            for line in got:
                lines.append(read_watermark(line, lines[-1] if lines else None))
                lost.update(lines[-1][2])
                if not all(n in buckets or n in lost for n in range(lines[-1][1])):
                    early.append(lines[-1][0])
        while len(admitted) < len(lines) and all(n in buckets or n in lost for n in range(lines[len(admitted)][1])):
            T, n, _ = lines[len(admitted)]
            admitted[T] = [k for k in range(lines[T - 1][1] if T else 0, n) if k not in lost]
            assert all(buckets[k] <= T for k in admitted[T]), (T, admitted[T], buckets)
    return admitted, buckets, lines, ended, early


def tick_traffic() -> tuple[list, dict[int, list[bytes]], dict[int, int]]:
    """Five requests: two admitted at bucket 1, their `END`s at 2 and 3; one admitted at 8, its `END` at 9, whose
    server connection opens only before bucket 10, holding bucket 9's watermark line; and two admitted at 14 whose
    `END`s share bucket 15. The answers, the arrivals and the server's held connects."""
    one = cells(http(b"{}"), B)
    assert len(one) == R * B and len(cells(http(b'{"p":"%s"}' % (b"a" * 120)), B)) == 2 * R * B
    arrivals = {1: [cells(http(b'{"p":"%s"}' % (b"a" * 120)), B), one], 8: [one], 14: [one, one]}
    return [[(0, b"r%d" % k), (0, EOF)] for k in range(5)], arrivals, {2: 10}


def positions(events: list[tuple], run_id: bytes) -> dict[tuple[str, int], int]:
    """The index in `events` of each request, ("request", N), each watermark line, ("line", t), and each bucket's
    becoming due, ("bucket", t)."""
    position, seen = {}, 0
    for i, event in enumerate(events):
        if event[0] == "watermark":
            for _ in watermark_lines(event[1], run_id)[0]:
                position[("line", seen)] = i
                seen += 1
        elif event[0] == "request":
            position[("request", release_of(event[2])[0])] = i
        else:
            position[event] = i
    return position


def test_the_watermark_closes_each_bucket_after_its_releases_and_only_advances():
    """On one stream the network certifier sends its head, then one line `t n` a tick, t = 0 … the run's last bucket,
    then the terminating chunk: bucket t closed with n `END`s released through it, so the releases of buckets ≤ t are
    exactly N < n, as the signed ingress records' `END` cells say, and t steps by 1 and n never decreases. Each request
    reaches the server at its bucket's tick or later, never before, and before its bucket's line: a request whose server
    connection opens late holds its bucket's line, and the later ones, until it is written. So an engine that admits at
    T exactly what the watermark closes by bucket T, waiting for every N < n_T, never reads a line before what it
    counts, and admits each request at its bucket, two in one bucket in their order, and nothing in an empty bucket."""
    answers, arrivals, hold = tick_traffic()
    run_id = bytes(range(32))
    server = Ordered(answers, hold=hold)
    warden, clients, _ = run(server, arrivals, watermark=server.watermark, run_nonce=lambda n: run_id[:n])
    ends = signed_ends(warden)
    assert [t for t, _ in ends] == [2, 3, 9, 15, 15] and [o for _, o in ends] == [2, 1, 3, 4, 5]
    admitted, buckets, lines, ended, early = tick_engine(server.events, run_id)
    assert ended and server.closed and warden.watermark_error is None and early == []
    assert buckets == dict(enumerate(t for t, _ in ends))
    assert lines == [(t, sum(a <= t for a, _ in ends), ()) for t in range(PARAMS.window)]
    assert warden.watermarked == lines[-1][:2] == (PARAMS.window - 1, 5) and warden.watermark_lost == []
    assert {T: ns for T, ns in admitted.items() if ns} == {2: [0], 3: [1], 9: [2], 15: [3, 4]}
    assert sorted(admitted) == list(range(PARAMS.window))
    position = positions(server.events, run_id)
    assert len([k for k in position if k[0] == "line"]) == PARAMS.window
    for n, a in buckets.items():
        assert position[("bucket", a)] < position[("request", n)] < position[("line", a)]
    assert position[("bucket", 10)] < position[("request", 2)] < position[("line", 9)] < position[("line", 10)]
    assert max(warden.lateness) == 0
    assert [bytes(c.shim().data) for c in clients] == [b"r0", b"r1", b"r2", b"r3", b"r4"]
    verdict = audit(PARAMS, *records(warden))
    assert verdict["accepted"], verdict["windows"]


def test_a_release_its_server_connection_loses_is_counted_and_named_on_its_buckets_line():
    """A release whose server connection is closing when it is released (N1's, closed at bucket 2), or whose connect
    fails after it (N3's, refused at bucket 16), is never written, and its bucket's line names it after n: line 3 is
    `3 2 1`, and line 15, held until N3's connect fails, is `15 5 3`. n stays the signed `END`s' count and N their
    order, so the engine admits every release but those, waits on neither, and none arrives; every tick runs on time,
    and the records, the clients' cells and the server's bytes are the run's without the watermark."""
    answers, arrivals, _ = tick_traffic()
    run_id = bytes(range(32))
    lossy = {"hold": {2: 10, 3: 16}, "refuse": (3,), "close_at": {0: 2}}
    server = Ordered(answers, **lossy)
    warden, clients, _ = run(server, arrivals, watermark=server.watermark, run_nonce=lambda n: run_id[:n])
    ends = signed_ends(warden)
    assert [t for t, _ in ends] == [2, 3, 9, 15, 15]
    admitted, buckets, lines, ended, early = tick_engine(server.events, run_id)
    assert ended and warden.watermark_error is None and early == [] and max(warden.lateness) == 0
    lost = {3: (1,), 15: (3,)}
    assert lines == [(t, sum(a <= t for a, _ in ends), lost.get(t, ())) for t in range(PARAMS.window)]
    assert warden.watermark_lost == [1, 3] and warden.watermarked == (PARAMS.window - 1, 5)
    assert {T: ns for T, ns in admitted.items() if ns} == {2: [0], 9: [2], 15: [4]}
    assert sorted(admitted) == list(range(PARAMS.window)) and sorted(buckets) == [0, 2, 4]
    assert server.conns[0].received == server.conns[3].received == b""
    position = positions(server.events, run_id)
    assert position[("bucket", 16)] < position[("line", 15)]
    plain = Ordered(answers, **lossy)
    without, plain_clients, _ = run(plain, arrivals, run_nonce=lambda n: run_id[:n])
    assert [(r.link, r.window, r.grid, r.status) for r in warden.records] == \
        [(r.link, r.window, r.grid, r.status) for r in without.records]
    assert [c.writes for c in clients] == [c.writes for c in plain_clients]
    assert [c.received for c in server.conns] == [c.received for c in plain.conns]


def test_a_release_still_in_its_transports_buffer_holds_its_line_until_it_is_written_or_lost():
    """N2's server connection opens at bucket 10 and its transport holds the request unwritten. Emptied at bucket 11,
    lines 9 and 10 wait for it, never a tick, and go out in order at tick 11, after it, so the engine admits N2 at 9 as
    before. Never emptied while the connection lives (it fails at s + D = 11, the request unread), line 9 names N2
    lost, and the engine skips it although its bytes were handed over. No line is read before the releases it counts."""
    answers, arrivals, hold = tick_traffic()
    run_id = bytes(range(32))
    for flushed, lost in ((11, []), (12, [2])):
        server = Ordered(answers, hold=hold, flush_at={2: flushed})
        warden, clients, _ = run(server, arrivals, watermark=server.watermark, run_nonce=lambda n: run_id[:n])
        admitted, buckets, lines, ended, early = tick_engine(server.events, run_id)
        assert ended and warden.watermark_error is None and early == [] and max(warden.lateness) == 0
        assert warden.watermark_lost == lost and lines[9] == (9, 3, tuple(lost)) and len(lines) == PARAMS.window
        assert {T: ns for T, ns in admitted.items() if ns} == {2: [0], 3: [1], **({} if lost else {9: [2]}), 15: [3, 4]}
        assert sorted(buckets) == [0, 1, 2, 3, 4]
        position = positions(server.events, run_id)
        assert position[("line", 8)] < position[("bucket", 9)]
        assert position[("request", 2)] < position[("bucket", 11)] < position[("line", 9)] < position[("line", 10)] < \
            position[("line", 11)] < position[("bucket", 12)]
        assert [bytes(c.shim().data) for c in clients] == [b"r0", b"r1", b"" if lost else b"r2", b"r3", b"r4"]


def test_the_watermark_moves_nothing_certified_and_never_waits_on_the_server():
    """The same traffic without the watermark, with it, with its connect refused, with it never open by the run's end,
    and with a server that has stopped reading it: every tick runs on time and every record, every client's cells and
    every request the server gets are the same. A refused or unopened stream sends nothing, and one whose transport
    holds more than `WRITE_BACKLOG` unwritten is closed, each saying why; the open one ends with the run."""
    answers, arrivals, hold = tick_traffic()

    def served(watermark: bool = True, **kw):
        server = Ordered(answers, hold=hold, **kw)
        hooks = {"watermark": server.watermark} if watermark else {}
        warden, clients, _ = run(server, arrivals, run_nonce=bytes, **hooks)
        wire = ([(r.link, r.window, r.grid, r.status) for r in warden.records], [c.writes for c in clients],
                [c.received for c in server.conns])
        assert max(warden.lateness) == 0 and audit(PARAMS, *records(warden))["accepted"]
        return warden, server, wire

    plain = served(watermark=False)
    assert plain[0].watermarked is None and not [e for e in plain[1].events if e[0] == "watermark"]
    cases = {"open": {}, "refused": {"refuse_watermark": True}, "unopened": {"opens": 10 ** 6},
             "backlog": {"buffered": WRITE_BACKLOG + 1}}
    for name, kw in cases.items():
        warden, server, wire = served(**kw)
        assert wire == plain[2], name
        assert warden.watermarked == (PARAMS.window - 1, 5), name
        sent = [e for e in server.events if e[0] == "watermark"]
        assert warden.watermark_error == {"open": None, "refused": "ConnectionRefusedError: refused",
                                          "unopened": "unopened", "backlog": "backlog"}[name]
        assert bool(sent) == (name == "open") and server.closed == (name in ("open", "backlog")), name


def test_the_parameters_digest_binds_every_parameter():
    """Each integer parameter, the egress form and the response cap move the digest, as do the cell kinds and the
    reasons it binds; a schedule without a cap hashes neither it nor its reason, so its digest is the period check's
    (`NetworkCheck.Period.Schedule.digest`, over the integer fields and the first six reasons). An egress form other
    than id or bytes is refused."""
    moved = [dataclasses.replace(PARAMS, **{f: getattr(PARAMS, f) + 1}) for f in INTEGER_FIELDS]
    moved += [dataclasses.replace(PARAMS, egress="id"), CAPPED, dataclasses.replace(CAPPED, max_tokens_cap=5)]
    digests = {PARAMS.digest, GATED.digest} | {p.digest for p in moved}
    assert len(digests) == len(moved) + 2 and all(len(d) == 64 for d in digests)
    assert {*INTEGER_FIELDS, "egress", "max_tokens_cap", "lengths", "content"} == {
        f.name for f in dataclasses.fields(WardenParams)}
    assert set(KIND_NAMES) == {PAD, DATA, END, FAIL, REFUSED} and len(set(REASONS)) == len(REASONS)
    assert WardenParams().egress == "id" and WardenParams().max_tokens_cap is None and REASONS[-2:] == ("format", "cap")
    assert GATED.reasons == REASONS[:-1] and CAPPED.reasons == REASONS
    for p in (PARAMS, GATED, WardenParams()):
        values = {f: getattr(p, f) for f in (*INTEGER_FIELDS, "egress")}
        assert p.digest == tagged_sha512(PARAMETERS_TAG, canonical_json_bytes(
            values | {"kinds": sorted(KIND_NAMES), "reasons": list(REASONS[:-1])}))
    for bad in ("", "text", "ID", 1, None):
        with pytest.raises(ValueError, match="egress is one of id, bytes"):
            WardenParams(egress=bad)
