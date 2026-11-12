"""Where the active network certifier reads the server's bytes: only on a connection it opened when it admitted that
client's connection. `connect`, which opens a server connection, is called once for each connection the network
certifier admits, after the tick that admits it and never before, and never for a connection it does not admit: one it
refuses at its arrival, the queue full, or one still queued when the run ends. With the watermark on it holds one more
connection to the server, the watermark stream, opened at the run's start with no admission, which it writes and never
reads.

That the watermark stream changes no record and no client's cell is held beside the network certifier's other tests,
by `test_the_watermark_moves_nothing_certified_and_never_waits_on_the_server`."""

from __future__ import annotations

import asyncio

import pytest

from verity.protocols.certifier.verifier.warden import REFUSED, cells
from verity.protocols.certifier.verifier.tests.test_network_certifier_warden import (
    B, EOF, P, PARAMS, Client, Ordered, Server, request, run, tick_engine, tick_traffic)


class Admissions(Server):
    """`Server` that keeps, for each `connect`, the ticks the network certifier had run and the bucket it had admitted
    the call's connection at, None for one not admitted (queued, or not arrived). The network certifier connects in
    admission order, which is ordinal order, so call k is for the connection of ordinal k + 1. `client` is `drive`'s
    `client`: through it the server knows its network certifier before the first client connects."""

    def __init__(self, answers: list[list], **kw) -> None:
        super().__init__(answers, **kw)
        self.certifier = None
        self.calls: list[tuple[int | None, int | None]] = []

    def client(self, certifier, data) -> Client:
        self.certifier = certifier
        return Client(certifier, data)

    async def connect(self):
        ordinal, certifier = len(self.calls) + 1, self.certifier
        live = [] if certifier is None else [*certifier.active, *certifier.queue]
        admitted = [c.admitted for c in live if c.ordinal == ordinal]
        self.calls.append((None if certifier is None else certifier.ticks, admitted[0] if admitted else None))
        return await super().connect()


class Unread:
    """The reader half of a stream the network certifier must not read: any use of it, a `StreamReader`'s `read`,
    `readline`, `readexactly`, `readuntil`, `at_eof` or any other attribute, and iteration, is kept in `touched` and
    raises. A test reads `touched`: what a task of the network certifier raises fails no test by itself."""

    def __init__(self) -> None:
        self.touched: list[str] = []

    def __getattr__(self, name: str):
        self.touched.append(name)
        raise AssertionError(f"the stream's reader half was used: {name}")

    def __aiter__(self):
        return self.__getattr__("__aiter__")

    def __anext__(self):
        return self.__getattr__("__anext__")


class Streamed(Ordered):
    """`Ordered` whose watermark stream's reader half is an `Unread`, `unread`. `opened_after` holds, for each time the
    network certifier opened the stream, how many grid rows it had released by then (`row`, its `release`)."""

    def __init__(self, answers: list[list], **kw) -> None:
        super().__init__(answers, **kw)
        self.unread, self.rows = Unread(), 0
        self.opened_after: list[int] = []

    def row(self, link: str, row) -> None:
        self.rows += 1

    async def watermark(self):
        self.opened_after.append(self.rows)
        _, writer = await super().watermark()
        return self.unread, writer


# --- `connect`: once for each admitted connection, after its admission -----------------------------------------------


def test_an_admitted_connection_gets_one_connect_after_the_tick_that_admits_it_and_none_before():
    """Two connections take the two lanes at bucket 0, and a third that arrives before bucket 1 waits in the queue for
    a lane until bucket 4. `connect` is called once for each, after the tick that admits it and before the next, the
    connection admitted when it is called: the third has none through the ticks of buckets 1 to 3, while it is queued.
    Each client's response is the bytes of its own server connection."""
    server = Admissions([[(0, b"r%d" % k), (0, EOF)] for k in range(3)])
    certifier, clients, _ = run(server, {0: [cells(request(5, k), B) for k in range(2)], 1: [cells(request(5, 2), B)]},
                                client=server.client)
    admitted = [e["admitted"] for e in sorted(certifier.log, key=lambda e: e["ordinal"])]
    assert admitted == [0, 0, 4]
    assert server.calls == [(s + 1, s) for s in admitted] == [(1, 0), (1, 0), (5, 4)]
    assert [conn.received for conn in server.conns] == [request(5, k) for k in range(3)]
    assert [bytes(c.shim().data) for c in clients] == [b"r0", b"r1", b"r2"]


# --- `connect`: never for a connection that is not admitted ----------------------------------------------------------


def test_a_connection_refused_at_its_arrival_gets_no_connect_and_no_byte_of_the_server():
    """Both lanes taken and the queue full, a fifth connection is refused as it arrives: it gets one `REFUSED` cell,
    on ordinal 0 and with no payload, and is closed. `connect` is called for the four admitted and for no other through
    the run's end, so the answer the server has for a fifth connection is never asked for and reaches no client."""
    server = Admissions([[(0, b"r%d" % k), (0, EOF)] for k in range(5)])
    certifier, clients, _ = run(server, {1: [cells(request(5, k), B) for k in range(2)],
                                         2: [cells(request(5, k), B) for k in range(2, 5)]}, client=server.client)
    assert (certifier.refused, certifier.ordinals, certifier.ticks) == (1, 4, PARAMS.window)
    admitted = [e["admitted"] for e in sorted(certifier.log, key=lambda e: e["ordinal"])]
    assert len(admitted) == 4 and None not in admitted
    assert server.calls == [(s + 1, s) for s in admitted] and len(server.conns) == 4
    refused = clients[4]
    assert [(kind, payload) for _, kind, payload in refused.cells()] == [(REFUSED, b"")] and refused.closed
    assert (refused.shim().outcome, refused.shim().ordinal, bytes(refused.shim().data)) == ("refused", 0, b"")
    assert [bytes(c.shim().data) for c in clients[:4]] == [b"r0", b"r1", b"r2", b"r3"]


def test_a_connection_still_queued_when_the_run_ends_gets_no_connect_and_no_byte_of_the_server():
    """Two connections admitted at bucket 15 hold the two lanes through bucket 23, the run's last, each with a response
    of eleven cells, so two more that arrive before bucket 16 wait in the queue through every later tick and are cut
    there, never admitted, when the run ends. `connect` is called for the first two and for no other: the two queued
    are written no cell, and the network certifier has read nothing of the server for them."""
    response = b"z" * (10 * P + 1)
    server = Admissions([[(0, response), (0, EOF)] for _ in range(4)])
    certifier, clients, _ = run(server, {15: [cells(request(5, k), B) for k in range(2)],
                                         16: [cells(request(5, k), B) for k in range(2, 4)]}, client=server.client)
    log = sorted(certifier.log, key=lambda e: e["ordinal"])
    assert (certifier.refused, certifier.ordinals, certifier.ticks) == (0, 4, PARAMS.window) and PARAMS.window == 24
    assert [(e["admitted"], e["ended"], e["failed"], e["cut"]) for e in log] == \
        [(15, 23, None, False)] * 2 + [(None, None, None, True)] * 2
    assert server.calls == [(16, 15), (16, 15)] and len(server.conns) == 2
    assert [c.writes for c in clients[2:]] == [[], []]
    assert [(e["bytes_out"], e["egress_cells"], e["upstream"]) for e in log[2:]] == [(0, 0, None)] * 2
    assert [bytes(c.shim().data) for c in clients[:2]] == [response] * 2


# --- the watermark stream: opened with no admission, written and never read -------------------------------------------


def test_the_watermark_stream_is_opened_at_the_runs_start_with_no_admission_and_no_connect():
    """A run with the watermark on and no client at all: the network certifier opens the watermark stream once, before
    its first tick releases a row, and writes it its head, a line a tick and its end. It admits no connection, and
    `connect` is never called."""
    run_id = bytes(range(32))
    server = Streamed([])
    certifier, clients, _ = run(server, {}, watermark=server.watermark, release=server.row,
                                run_nonce=lambda n: run_id[:n])
    assert server.unread.touched == []
    assert server.opened_after == [0] and server.rows == 3 * PARAMS.window
    assert clients == [] and (certifier.ordinals, certifier.refused, certifier.log) == (0, 0, [])
    assert server.conns == []
    _, _, lines, ended, _ = tick_engine(server.events, run_id)
    assert lines == [(t, 0, ()) for t in range(PARAMS.window)] and ended and server.closed
    assert certifier.watermark_error is None


def test_the_watermark_stream_is_written_and_never_read():
    """Five requests released over a run with the watermark on (`tick_traffic`'s, each server connection opening at
    once): the network certifier writes the stream its head, a line a tick and its end, and uses the stream's reader
    half in no way, where a `StreamReader`'s every method and iteration is kept (`Unread`). Each response a client
    gets was read on the connection `connect` opened for it, one a client."""
    answers, arrivals, _ = tick_traffic()
    run_id = bytes(range(32))
    server = Streamed(answers)
    certifier, clients, _ = run(server, arrivals, watermark=server.watermark, release=server.row,
                                run_nonce=lambda n: run_id[:n])
    assert server.unread.touched == [] and server.opened_after == [0]
    _, _, lines, ended, early = tick_engine(server.events, run_id)
    assert [t for t, _, _ in lines] == list(range(PARAMS.window)) and ended and server.closed and early == []
    assert certifier.watermarked == (PARAMS.window - 1, 5) and certifier.watermark_error is None
    assert len(server.conns) == len(clients) == 5
    assert [bytes(c.shim().data) for c in clients] == [b"r0", b"r1", b"r2", b"r3", b"r4"]


def test_every_use_of_a_stream_reader_is_kept_by_the_unread_reader_and_raises():
    """The guard of the two tests above is real: each public name of `asyncio.StreamReader`, its read methods among
    them, and each half of iteration, used on an `Unread`, is kept in `touched` and raises."""
    names = [name for name in dir(asyncio.StreamReader) if not name.startswith("_")]
    assert {"read", "readline", "readexactly", "readuntil", "at_eof"} <= set(names)
    for name in names:
        probe = Unread()
        with pytest.raises(AssertionError, match=name):
            getattr(probe, name)
        assert probe.touched == [name]
    for name in ("__aiter__", "__anext__"):
        probe = Unread()
        with pytest.raises(AssertionError, match=name):
            getattr(probe, name)()
        assert probe.touched == [name]
