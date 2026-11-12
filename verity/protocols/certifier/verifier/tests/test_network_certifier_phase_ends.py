"""The lengths rows' form 2 (`warden`'s "The lengths rows", `connection_rows`' "The word that no head comes"), driven
tick by tick on `test_network_certifier_warden`'s harness: every execution phase of a run ends once, in its hand or
without releases, the run's last at its last close. The form is a value of its own digest, and a schedule at form 1
leaves what it left. The network certifier admits no connection that may end past the run, and turns away an arrival
in the run's last buckets, by its tick count and the schedule alone, with one constant cell on the next tick and no
refusal; it hands an execution phase only where it and the ones beside it were published complete, so no hand holds
a byte of one that wasn't; a run that ends early ends the execution phases it has not; and the word that an
execution phase has no head is signed for that execution phase, link, session and schedule, and verifies as nothing
else."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import socket

import pytest

import signers
from verity.commitments.identity import tagged_sha512
from verity.protocols.certifier.verifier import connection_rows as CR
from verity.protocols.certifier.verifier.active import VirtualClock
from verity.protocols.certifier.verifier.certificate import Issuer
from verity.protocols.certifier.verifier.tests.test_network_certifier_content_tables import (
    CON, GONE, HELD, LEN_DIGEST, LINK, leaves, leaving, mixed, registered)
from verity.protocols.certifier.verifier.tests.test_network_certifier_lengths import (
    B, LEN, PLAIN, D, R, T, answer, id_req, traffic)
from verity.protocols.certifier.verifier.tests.test_network_certifier_warden import (
    TAU, Client, Server, SocketClient, records, run)
from verity.protocols.certifier.verifier.warden import REFUSED, LENGTHS_FORMS, Warden, WardenParams, cell, cells

#: the lengths tests' schedule at form 2, and the content tables tests' classes at it, each its two sizes
TWO = dataclasses.replace(LEN, lengths=2)
TWO_CON = HELD
SPAN = TWO.span
#: pinned, as `LEN_DIGEST` is: the content tables tests' schedule's digest at form 1, and `leaves` of two runs at
#: form 1, the content tables tests' traffic and the lengths tests' with its fourth tick of the second execution
#: phase late. Form 1 leaves these whatever other forms there are
CON_DIGEST = ("515a1584b683fde5d5af88641e5e894e51168b5726d6d261bedcfd0d741f9e97"
              "c79a0fe6e593b3984b5890d1c7f49d405736444f5a71ad15fadbdc8167ab8280")
CON_RUN = ("e654968a5568465737c2286818ecfc3a171f856ef0c44c67eecbed14c266f9fd"
           "24f9bb588bbe0f34c9fd7e62da8584d513c1307022901995eff2be5a3159fdbb")
LATE_RUN = ("5b80530f5b725d8f9f6068dc5db9becb00ecd2ea186889c31573378b806425f9"
            "a2e5998a32a4dd617d60868359836eb3bd888cfa586c41d13e9ce008f602eb46")
#: pinned with them: `registered` of the content tables tests' traffic at form 1, and of the lengths tests' without
#: content rows, the registrations a verifier reads
CON_REGISTERED = ("c969de114e4808370eb22eb9082ff4d290c742459fe1b9cab183a258b1d6268d"
                  "9691e51ccbfb79420456efeb5f76d5669a8e9c9bffb8c3662b71538faeb3797a")
LEN_REGISTERED = ("97a5a38d9ed904b94947425638fb0fd7c092d128f8940ecae204a25bff540fdf"
                  "4d2ed4043989d3d0b0a5930a550ba7c6e925ea78855b0955705c8519b28d5fba")
#: the one cell of a turn-away, a refusal's bytes
AWAY = cell(REFUSED, 0, b"", B)
#: a run of two execution phases at the lengths tests' schedule: under form 2, its first bucket that admits nothing
RUN = 2
FIRST = RUN * T - SPAN
#: the buckets of five arrivals from that bucket on: three at once, more than the queue holds, and one in the run's
#: last bucket
LATE = (FIRST, FIRST + 1, FIRST + 1, FIRST + 1, RUN * T - 1)
#: what the lanes and the queue hold when those buckets begin: the arrivals before them (`through`), on two lanes and
#: a queue of two, and whether the admitted ones answer the longest response, so hold their lanes to the run's last
#: bucket but one, or the shortest, so free them `D` buckets on with the queue as it was
STARTS = ((0, True), (1, True), (2, True), (3, True), (4, True), (4, False), (5, True))
#: pinned, as the runs above are: `left` of form 1's run at each of `STARTS`, in order, hashed together
FORM_1_LATE = ("115ee1bd14ba2473588110c114c4c6d6d49c99e83a133171a2c4324d7afc88d1"
               "4e903e44ce765310b28ae5ad0c0551d6a3637623ea446574b6ba0ac27a1b5df5")


def ends(device: Warden) -> list[tuple[int, bool]]:
    """Each execution phase the device ended, in the order it ended them, and whether it handed it."""
    return [(w, strides is not None) for w, strides in device.lengths]


def handed(device: Warden) -> dict:
    return {w: strides for w, strides in device.lengths if strides is not None}


def order_of(device: Warden) -> dict[int, int]:
    """Each released connection's place in the release order, by its ordinal, as the device's log has it."""
    return {e["ordinal"]: e["release"] for e in device.log if e["release"] is not None}


# --- the form ------------------------------------------------------------------------------------------------------


def test_form_2_is_a_value_of_its_own_digest_and_a_schedule_at_form_1_leaves_what_it_left(tmp_path):
    """The form is 1 or 2. Form 2 has the lengths and fails tables and the bounds of form 1 and another digest, and
    its content tables have form 1's names and row sizes; a schedule at form 1, with content rows or without, has
    the digest it had, and its run leaves the records, the certificates, the cells' heads, the hands, the
    connection rows' heads and their registrations it left, byte for byte: with content rows, and with a late tick,
    whose execution phase and the one before it it doesn't hand."""
    assert LENGTHS_FORMS == (1, 2)
    assert [(p.lengths, p.closes) for p in (PLAIN, LEN, TWO)] == [(None, False), (1, False), (2, True)]
    assert LEN.digest.hex() == LEN_DIGEST and CON.digest.hex() == CON_DIGEST
    assert len({PLAIN.digest, LEN.digest, TWO.digest, CON.digest, TWO_CON.digest}) == 5
    assert CR.tables_of(TWO) == CR.tables_of(LEN) == CR.tables_of(TWO_CON)[:2] == CR.tables_of(CON)[:2]
    assert [(name, size) for name, _, size in CR.tables_of(TWO_CON)] == \
        [(name, size) for name, _, size in CR.tables_of(CON)]
    assert (TWO.stride_releases, TWO.span) == (LEN.stride_releases, LEN.span)
    assert SPAN == D + (LEN.egress_cells - 1) // R
    assert WardenParams(**json.loads(json.dumps(dataclasses.asdict(TWO_CON)))) == TWO_CON
    for bad in (0, 3, True, "2", 2.0):
        with pytest.raises(ValueError, match="lengths is 1 or 2"):
            dataclasses.replace(PLAIN, lengths=bad)
    with pytest.raises(ValueError, match="ends within a window"):
        dataclasses.replace(TWO, delay=13)
    _, arrivals, answers = mixed()
    device, _, _ = run(answers, arrivals, params=CON, windows=3, client=leaving(GONE))
    for name in ("content", "content-registered", "late", "lengths-registered"):
        (tmp_path / name).mkdir()
    assert leaves(device, CON, tmp_path / "content") == CON_RUN
    assert registered(device, CON, tmp_path / "content-registered") == CON_REGISTERED
    _, arrivals, answers = traffic()
    device, _, _ = run(answers, arrivals, params=LEN, windows=5, late={T + 3: 0.05})
    assert [w for w, _ in device.lengths] == [2, 3] and leaves(device, LEN, tmp_path / "late") == LATE_RUN
    device, _, _ = run(answers, arrivals, params=LEN, windows=3)
    assert registered(device, LEN, tmp_path / "lengths-registered") == LEN_REGISTERED


# --- the admission -------------------------------------------------------------------------------------------------


def test_a_connection_that_may_end_past_the_run_is_never_admitted():
    """Form 2 admits a connection only at a bucket whose `span` buckets on are inside the run. One that arrives for
    the last such bucket is admitted and, its response the longest the schedule carries, ends in the run's last
    bucket. One that arrives a bucket later is turned away, though its response would have ended in time: it takes
    no ordinal, the server never sees it, it has no place in the release order, and it is no refusal. Form 1 admits
    it, and it ends."""
    W = 2
    last = W * T - 1 - SPAN
    arrivals = {3: [cells(id_req(1, 3), B)], last: [cells(id_req(2, 3), B)], last + 1: [cells(id_req(3, 3), B)]}
    answers = [answer((1, 2)), answer(range(8)), answer(())]
    two, clients, server = run(answers, arrivals, params=TWO, windows=W)
    log = {e["ordinal"]: e for e in two.log}
    assert [log[k]["admitted"] for k in sorted(log)] == [3, last] and two.ordinals == 2
    assert (log[2]["ended"], log[2]["egress_cells"], log[2]["failed"], log[2]["cut"]) == (W * T - 1, 9, None, False)
    assert clients[2].writes == [(last + 1, AWAY)] and clients[2].closed and two.turned_away == 1
    assert len(server.conns) == 2 and two.requests_released == 2
    assert (two.refused, two.refused_windows) == (0, set())
    assert all(r.status.complete for link in records(two) for r in link)
    one, _, server = run(answers, arrivals, params=LEN, windows=W)
    log = {e["ordinal"]: e for e in one.log}
    assert (log[3]["admitted"], log[3]["ended"], log[3]["cut"]) == (last + 1, last + 1 + D, False)
    assert len(server.conns) == 3 and one.requests_released == 3 and one.turned_away == 0


def last_phase_most(params: WardenParams) -> tuple[int, int]:
    """The most connections form 2 admits in a run's last execution phase, and the most releases that execution phase
    holds. It admits in its first A = T − `span` buckets, and a lane admits every D + 1 buckets at the most. A lane
    may also release there a connection admitted in the `ingress_buckets` − 1 buckets before it, which holds the lane
    through its admission's bucket + D, so the lane next admits D + 2 − `ingress_buckets` buckets in at the earliest."""
    A, L = params.window - params.span, params.delay + 1
    admitted = -(-A // L)
    after = max(0, -(-(A - (params.delay + 2 - params.ingress_buckets)) // L))
    return params.lanes * admitted, params.lanes * max(admitted, 1 + after)


def test_a_runs_last_execution_phase_holds_at_most_what_its_first_buckets_admit():
    """`last_phase_most` on the device, at two schedules: each lane filled at every bucket it frees, the last
    execution phase admits and releases exactly that many, and the connection that arrives next is turned away.
    At the first schedule a lane's two admissions are the most it releases there. At the second, each lane releases
    a connection admitted in the bucket before the execution phase, and then admits one more. Both are handed whole
    at the run's last close."""
    one, more, two_cells = cells(id_req(1, 3), B), cells(id_req(2, 3), B), cells(id_req(3, 40), B)
    assert (len(one), len(two_cells)) == (B, 2 * B)
    busy = dataclasses.replace(TWO, ingress_buckets=2, egress_cells=11)
    assert (last_phase_most(TWO), last_phase_most(busy)) == ((4, 4), (2, 4))
    for params, arrivals, n in ((TWO, {T: [one, one], T + 6: [one, one], T + 7: [more]}, 5),
                                (busy, {T - 1: [two_cells, two_cells], T + 5: [one, one], T + 6: [more]}, 5)):
        device, _, _ = run([answer(())] * n, arrivals, params=params, windows=2)
        admitted = [e["admitted"] for e in sorted(device.log, key=lambda e: e["ordinal"])]
        assert sum(a >= T for a in admitted) == last_phase_most(params)[0]
        assert (len(admitted), device.turned_away) == (n - 1, 1) and max(admitted) == 2 * T - 1 - params.span
        assert ends(device) == [(0, True), (1, True)]
        last = [r for stride in handed(device)[1] for r in stride]
        assert len(last) == last_phase_most(params)[1] == device.requests_released
        assert handed(device) == CR.of_records(params, *records(device)[:2])
    # the measured schedule's shape: a connection's span most of an execution phase, and ingress as long as the delay
    assert last_phase_most(dataclasses.replace(TWO, window=64, delay=4, ingress_buckets=4, egress_cells=101,
                                               lanes=3)) == (3 * 2, 3 * 3)


# --- the turn-away -------------------------------------------------------------------------------------------------


class Stamped(Client):
    """A client that keeps the device's clock at each write to it, at its stream's end and at its close."""

    def __init__(self, warden: Warden, data) -> None:
        self.at, self.ended_at, self.closed_at = [], None, None
        super().__init__(warden, data)

    def write(self, data: bytes) -> None:
        self.at.append(self.warden.clock())
        super().write(data)

    def write_eof(self) -> None:
        self.ended_at = self.warden.clock()
        super().write_eof()

    def close(self) -> None:
        if self.closed_at is None:
            self.closed_at = self.warden.clock()
        super().close()


def through(params: WardenParams, before: int, long: bool, **kw):
    """A run of `RUN` execution phases with `before` arrivals before `FIRST`, the first two, which take the lanes,
    two buckets before it and the rest in the bucket before it, and `LATE`'s after them; the first two connections
    admitted answer the longest response when `long`, every other the shortest. Returns the device, the clients of
    the arrivals before, `LATE`'s, and the server."""
    arrivals = {FIRST - 2: [cells(id_req(j + 1, 3), B) for j in range(before)][:2],
                FIRST - 1: [cells(id_req(j + 1, 3), B) for j in range(before)][2:]}
    for j, t in enumerate(LATE):
        arrivals.setdefault(t, []).append(cells(id_req(10 + j, 3), B))
    answers = [answer(range(8) if long else ())] * 2 + [answer(())] * (before + len(LATE))
    device, clients, server = run(answers, arrivals, params=params, windows=RUN, **kw)
    return device, clients[:before], clients[before:], server


def left(device: Warden, clients: list) -> str:
    """A digest of what a run left: each client's writes with their buckets and whether it was closed, the log, the
    counts, each record published or withheld, and the hands."""
    doc = ([(c.writes, c.closed) for c in clients], sorted(device.log, key=lambda e: e["ordinal"]),
           (device.ordinals, device.refused, sorted(device.refused_windows), device.requests_released),
           [(r.link, r.window, r.status.reason and r.status.reason.value, r.grid)
            for r in (*device.records, *device.withheld)], device.lengths)
    return hashlib.sha512(repr(doc).encode()).hexdigest()


def test_an_arrival_in_the_runs_last_buckets_is_turned_away_whatever_the_queue_and_the_lanes_hold():
    """Under form 2, at each of `STARTS`: every lane free, one or both taken, one queued, the queue full with the
    lanes held to the run's end or freed inside those buckets, and one arrival more than the queue holds. `LATE`'s
    five arrivals, and no other, are turned away in every one: the set reads the tick count and the schedule, and
    nothing the queue or the lanes hold moves it. Each gets one cell, the same bytes every time, and its stream's
    end in its own bucket, and the close; none takes an ordinal, is queued or is a refusal, so no record is withheld
    for them and both execution phases are published complete and handed. A connection queued when those buckets
    begin waits, though a lane frees, and is cut at the run's end with no cell. The last arrival before them with
    the queue full is refused, as it was, and its execution phase's records are withheld."""
    room = TWO.lanes + TWO.queue
    seen = set()
    for before, long in STARTS:
        device, early, late, server = through(TWO, before, long)
        assert [(c.writes, c.ended, c.closed) for c in late] == [([(t, AWAY)], t, True) for t in LATE]
        seen |= {data for c in late for _, data in c.writes}
        taken, held = min(before, TWO.lanes), min(before, room)
        assert (device.turned_away, device.ordinals, len(server.conns)) == (len(LATE), held, taken)
        log = sorted(device.log, key=lambda e: e["ordinal"])
        assert [(e["admitted"], e["ended"], e["cut"]) for e in log] == \
            [(FIRST - 2, FIRST - 2 + (SPAN if long else D), False)] * taken + [(None, None, True)] * (held - taken)
        assert all(c.writes == [] and c.closed for c in early[taken:held])
        if before <= room:
            assert (device.refused, device.refused_windows, device.withheld) == (0, set(), [])
            assert all(r.status.complete for link in records(device) for r in link)
            assert ends(device) == [(0, True), (1, True)]
            assert handed(device) == CR.of_records(TWO, *records(device)[:2])
        else:
            assert (device.refused, device.refused_windows) == (1, {1}) and early[room].writes == [(FIRST - 1, AWAY)]
            assert [r.window for r in device.withheld] == [1] * 3 and ends(device) == [(0, False), (1, False)]
    assert seen == {AWAY} and len(AWAY) == B


def test_a_turn_aways_cell_leaves_on_the_next_tick_through_the_pacer_and_never_at_its_arrival():
    """An arrival is handled before its bucket's tick, while the clock still reads the tick before. A refusal is
    written and closed then. A turn-away is neither until the clock reads its bucket's boundary: the tick writes its
    cell and ends its stream, and the link, its client's stream already ended, is closed. With the pacer the loop
    writes it nothing: each turn-away's cell is a part of its bucket, which the pacer writes to its socket, ending
    the stream, a bucket of its own where no connection is live, and the clients read what the unpaced run wrote
    them; one queued and cut reads nothing."""
    device, early, late, _ = through(TWO, 5, True, client=Stamped)
    refused, earlier = early[TWO.lanes + TWO.queue], (FIRST - 2) * TAU
    assert (refused.writes, refused.at, refused.closed_at) == ([(FIRST - 1, AWAY)], [earlier], earlier)
    assert [(c.at, c.ended_at, c.closed_at) for c in late] == [([t * TAU], t * TAU, t * TAU) for t in LATE]
    for before in (0, 4):
        _, plain_early, plain_late, _ = through(TWO, before, True)
        paced, early, late, _ = through(TWO, before, True, paced=0.05, client=SocketClient)
        assert all(c.writes == [] for c in (*early, *late)) and paced.turned_away == len(LATE)
        assert [c.received() for c in late] == [AWAY] * len(LATE)
        got = [c.received() for c in early]
        assert got == [b"".join(data for _, data in c.writes) for c in plain_early]
        assert got[TWO.lanes:] == [b""] * (before - TWO.lanes if before else 0)
        assert [b"".join(data for _, data in c.writes) for c in plain_late] == [AWAY] * len(LATE)
        assert set(LATE) <= {r.t for r in paced.releases} and all(r.written for r in paced.releases)
        if not before:
            assert [r.t for r in paced.releases] == sorted(set(LATE))


class Ticking:
    """A device at form 2 for a run of one execution phase, on a clock the test drives a tick at a time."""

    def __init__(self, server: Server, **hooks) -> None:
        self.clock = VirtualClock(-TAU / 2)
        self.device = Warden(TWO, server.connect, origin=0.0, windows=1, clock=self.clock, sleep=self.clock.sleep,
                             **hooks)
        self.running = asyncio.create_task(self.device.run())

    async def tick(self, *buckets: int) -> None:
        """Each of `buckets`' ticks, and with the pacer its bucket's release."""
        for t in buckets:
            self.clock.advance(t * TAU)
            await self.device.until(lambda w: w.ticks > t and not any(w._unreleased.values()))


def test_a_turned_away_link_stays_open_to_its_clients_end_of_stream_and_what_it_sends_is_dropped():
    """The tick that releases a turn-away's cell ends the device's stream on the link and leaves the link open. The
    device reads what the client still sends and drops it, closes the link at the client's end of stream, or when
    the client resets it, which it counts, and closes at the run's end one whose client never ends its stream. None
    of those bytes reaches a server, takes an ordinal or is in a record, and the execution phase is published
    complete and handed."""
    async def main():
        server = Server([])
        run = Ticking(server)
        device, first = run.device, T - SPAN
        await run.tick(*range(first))
        clients = [Client(device, (cells(id_req(j, 3), B), False)) for j in (1, 2, 3)]

        def links() -> list:
            return [(list(c.writes), c.ended, c.closed, len(c.reader._buffer)) for c in clients]

        await device.until(lambda w: w.turned_away + w.ordinals + w.refused == 3)
        seen = [links()]
        await run.tick(first)
        seen.append(links())
        for c in clients:
            c.reader.feed_data(bytes(5 * B))
        await run.tick(first + 1)
        seen.append(links())
        clients[0].reader.feed_eof()
        clients[2].reader.set_exception(ConnectionResetError("reset by its client"))
        await run.tick(first + 2)
        closed = [[c.closed for c in clients]]
        await run.tick(*range(first + 3, T))
        await run.running
        closed.append([c.closed for c in clients])
        return device, server, clients, seen, closed

    device, server, clients, seen, closed = asyncio.run(main())
    held = ([(T - SPAN, AWAY)], T - SPAN, False, 0)
    assert seen == [[([], None, False, B)] * 3, [held] * 3, [held] * 3]
    assert closed == [[True, False, True], [True, True, True]] and device.turned_away_reset == 1
    assert all(c.writes == [(T - SPAN, AWAY)] for c in clients)
    assert (device.turned_away, device.ordinals, device.refused, len(server.conns)) == (3, 0, 0, 0)
    assert device.withheld == [] and all(r.status.complete for link in records(device) for r in link)
    assert ends(device) == [(0, True)] and handed(device) == CR.of_records(TWO, *records(device)[:2])
    assert not any(frame is not None for link in records(device) for r in link for row in r.grid for frame in row)


#: Linux's TCP states of a socket that has sent its stream's end and still reads its peer's, FIN-WAIT-1 and
#: FIN-WAIT-2 (`tcp_info`'s first byte)
ENDED_AND_OPEN = (4, 5)


async def on_loopback(paced: float | None, more: bytes):
    """A run of one execution phase behind a loopback listener. A client connects in the first bucket that admits
    nothing and writes its whole request, which the device has not read when its tick comes; once that bucket is
    released it reads to its stream's end, where a reset raises, then writes `more` and closes. Returns the device,
    the TCP state of the device's end of the link at the release, and what the client read."""
    run = Ticking(Server([]), paced=paced)
    device, first = run.device, T - SPAN
    listener = await asyncio.start_server(device.handle, "127.0.0.1", 0)
    try:
        await run.tick(*range(first))
        reader, writer = await asyncio.open_connection(*listener.sockets[0].getsockname()[:2])
        writer.write(cells(id_req(1, 40), B))
        await writer.drain()
        await device.until(lambda w: w.turned_away + w.ordinals + w.refused == 1)
        await run.tick(first)
        (link,) = device._leaving
        state = link.get_extra_info("socket").getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 1)[0]
        got = await reader.read() if state in ENDED_AND_OPEN else None
        writer.write(more)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        await run.tick(*range(first + 1, T))
        await run.running
    finally:
        listener.close()
        run.running.cancel()
    return device, state, got


def test_a_turned_away_client_on_a_socket_reads_its_cell_and_the_streams_end_and_never_a_reset():
    """Over TCP, a link closed with bytes of its client's unread is reset, and the client may lose what it has not
    read yet. The turned-away client here has written its whole request, of two cells, and the device has read
    none of it at its tick. With the loop writing and with the pacer, the device's end of the link has sent its
    stream's end at the bucket's release and is still open; the client reads the one cell and then the stream's
    end, never a reset, and none either when it sends more after."""
    assert len(cells(id_req(1, 40), B)) == 2 * B
    for paced in (None, 0.05):
        for more in (b"", bytes(3 * B)):
            device, state, got = asyncio.run(on_loopback(paced, more))
            assert state in ENDED_AND_OPEN and got == AWAY
            assert (device.turned_away, device.ordinals, device.refused) == (1, 0, 0)


def test_an_arrival_with_no_tick_left_is_closed_with_no_cell():
    """A turn-away leaves only on a tick. An arrival turned away in a bucket whose tick never runs, the run cancelled
    first, is held until then and closed with no cell, and so is one that arrives once the run is over; both are
    counted."""
    async def main():
        clock = VirtualClock(-TAU / 2)
        device = Warden(TWO, Server([]).connect, origin=0.0, windows=1, clock=clock, sleep=clock.sleep)
        running = asyncio.create_task(device.run())
        for t in range(T - 1):
            clock.advance(t * TAU)
            await device.until(lambda w, t=t: w.ticks > t)
        held = Client(device, cells(id_req(1, 3), B))
        await device.until(lambda w: w.turned_away + w.ordinals == 1)
        assert (device.turned_away, held.writes, held.closed) == (1, [], False)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        after = Client(device, cells(id_req(2, 3), B))
        await device.until(lambda w: w.turned_away + w.ordinals == 2)
        return device, held, after

    device, held, after = asyncio.run(main())
    assert (device.turned_away, device.ordinals, device.refused, device.ticks) == (2, 0, 0, T - 1)
    assert [(c.writes, c.closed) for c in (held, after)] == [([], True)] * 2


def test_a_load_that_never_stops_leaves_every_execution_phase_complete_and_ended_in_its_head():
    """An arrival every third bucket from a run's first to its last, a rate two lanes carry with nothing queued, at
    two schedules: each arrival before the first bucket that admits nothing is admitted in its bucket and ends, and
    each one after it is turned away. Nothing is refused, queued at the run's end or cut; every execution phase's
    three records are published complete, none withheld, and every one is handed, the last at the run's last close,
    what the records say."""
    busy = dataclasses.replace(TWO, ingress_buckets=2, egress_cells=11)
    for params, windows in ((TWO, 3), (busy, 2)):
        first = windows * T - params.span
        every = list(range(0, windows * T, 3))
        n = sum(t < first for t in every)
        device, clients, _ = run([answer((t,)) for t in every[:n]], {t: [cells(id_req(t, 3), B)] for t in every},
                                 params=params, windows=windows)
        assert 0 < n < len(every) and every[-1] >= windows * T - 3
        assert (device.ordinals, device.turned_away, device.refused) == (n, len(every) - n, 0)
        assert [(e["admitted"], e["cut"], e["failed"]) for e in sorted(device.log, key=lambda e: e["ordinal"])] == \
            [(t, False, None) for t in every[:n]]
        assert [c.writes for c in clients[n:]] == [[(t, AWAY)] for t in every[n:]]
        assert device.withheld == [] and all(r.status.complete for link in records(device) for r in link)
        assert ends(device) == [(w, True) for w in range(windows)]
        assert handed(device) == CR.of_records(params, *records(device)[:2])
        assert sum(len(stride) for strides in handed(device).values() for stride in strides) == n


def test_form_1_at_the_same_arrivals_leaves_what_it_left():
    """Form 1 turns nothing away. At each of `STARTS` it admits, queues, refuses and cuts `LATE`'s arrivals as it
    did, and the runs leave the cells, the log, the records and the hands they left, byte for byte."""
    all_left = hashlib.sha512()
    for before, long in STARTS:
        device, early, late, _ = through(LEN, before, long)
        assert device.turned_away == 0 and device.ordinals + device.refused == before + len(LATE)
        all_left.update(bytes.fromhex(left(device, early + late)))
    assert all_left.hexdigest() == FORM_1_LATE


# --- the hand ------------------------------------------------------------------------------------------------------


def test_every_execution_phase_ends_once_in_order_and_the_runs_last_at_its_last_close():
    """Form 2 ends execution phase w once execution phase w + 1's records are published, and the run's last with its
    own records: three execution phases, each handed, in order. The last one's hand holds each release admitted in
    it, each ended, the second in the run's last bucket; every hand is what the records say; the device then holds
    no release; and form 1 hands the first two as form 2 does and the last never."""
    reqs, arrivals, answers = traffic()
    last = 3 * T - 1 - SPAN
    arrivals |= {2 * T: [cells(id_req(6, 40), B)], last: [cells(id_req(7, 3), B)]}
    answers += [answer((4,)), answer(range(8))]
    calls: list = []
    two, _, _ = run(answers, arrivals, params=TWO, windows=3,
                    publish=lambda record, grid: calls.append((record.link, record.window)),
                    lengths=lambda window, strides: calls.append(("lengths", window, strides)))
    assert [e[:2] for e in calls] == [
        ("requests", 0), ("responses", 0), ("fails", 0), ("requests", 1), ("responses", 1), ("fails", 1),
        ("lengths", 0), ("requests", 2), ("responses", 2), ("fails", 2), ("lengths", 1), ("lengths", 2)]
    got = {e[1]: e[2] for e in calls if e[0] == "lengths"}
    assert [(r.order, r.request_bytes, r.outputs, r.failed) for stride in got[2] for r in stride] == [
        (5, len(id_req(6, 40)), 1, False), (6, len(id_req(7, 3)), 8, False)]
    log = {e["ordinal"]: e for e in two.log}
    assert [(log[k]["admitted"], log[k]["ended"]) for k in (6, 7)] == [(2 * T, 2 * T + D), (last, 3 * T - 1)]
    assert two._released == {} and all(not e["cut"] for e in two.log)
    kept, _, _ = run(answers, arrivals, params=TWO, windows=3)
    assert handed(kept) == got == CR.of_records(TWO, *records(kept)[:2]) and ends(kept) == [(w, True) for w in range(3)]
    one, _, _ = run(answers, arrivals, params=LEN, windows=3)
    assert dict(one.lengths) == {0: got[0], 1: got[1]}
    alone, _, _ = run([answer((1,))], {0: [cells(reqs[0], B)]}, params=TWO, windows=1)
    assert ends(alone) == [(0, True)] and handed(alone) == CR.of_records(TWO, *records(alone)[:2])


def straddling():
    """The lengths tests' traffic, then a request of three cells whose first two leave in the third execution
    phase's last bucket and whose `END` in the fourth's first, and one in each of the fifth and the sixth."""
    _, arrivals, answers = traffic()
    arrivals |= {3 * T - 2: [cells(id_req(6, 100), B)], 4 * T + 2: [cells(id_req(7, 3), B)],
                 5 * T + 1: [cells(id_req(8, 40), B)]}
    return arrivals, answers + [answer((7, 8)), answer((9,)), answer((1, 2, 3))]


def test_an_execution_phase_is_handed_only_where_it_and_the_ones_beside_it_were_published_complete():
    """A tick past its slack faults the third of six execution phases: under form 2 it, the one before it and the one
    after it end without releases, and the other three are handed, each what the records say. The one after it
    holds a release whose request's first cells left in the faulted one: form 1 hands it, with bytes the records
    don't hold. A fault in the run's first execution phase or in its last takes the one beside it, a refusal takes
    its execution phase and the one after, and a run of one execution phase is handed where it is whole."""
    arrivals, answers = straddling()
    late = {2 * T + 3: 0.05}
    two, _, _ = run(answers, arrivals, params=TWO, windows=6, late=late)
    assert [r.status.complete for r in records(two)[0]] == [True, True, False, True, True, True]
    assert ends(two) == [(0, True), (1, False), (2, False), (3, False), (4, True), (5, True)]
    reading = CR.of_records(TWO, *records(two)[:2], order_of(two))
    assert handed(two) == {w: reading[w] for w in (0, 4, 5)} and sum(map(len, handed(two)[5])) == 1
    one, _, _ = run(answers, arrivals, params=LEN, windows=6, late=late)
    assert [w for w, _ in one.lengths] == [0, 3, 4]
    (straddler,) = [r for stride in dict(one.lengths)[3] for r in stride]
    (read,) = [r for stride in CR.of_records(LEN, *records(one)[:2], order_of(one))[3] for r in stride]
    assert (straddler.order, read.order) == (5, 5)
    assert straddler.request_bytes == len(id_req(6, 100)) > read.request_bytes
    log = {e["ordinal"]: e for e in one.log}
    assert (log[6]["admitted"], log[6]["ingress_end"]) == (3 * T - 2, 3 * T)
    for windows, tick, without in ((3, 3, [0, 1]), (3, 2 * T + 3, [1, 2]), (1, 3, [0]), (2, T, [0, 1])):
        device, _, _ = run(answers, arrivals, params=TWO, windows=windows, late={tick: 0.05})
        assert ends(device) == [(w, w not in without) for w in range(windows)]
    crowd = {0: [cells(id_req(1, 3), B)] * 3}
    refused, _, _ = run([answer((1,))] * 2, crowd, params=TWO, windows=4)
    assert (refused.refused, refused.refused_windows) == (1, {0})
    assert ends(refused) == [(0, False), (1, False), (2, True), (3, True)]


def test_the_records_count_the_release_order_only_up_to_one_that_isnt_complete():
    """`of_records` past a faulted execution phase: the records have lost a release's `END`, so their count of the
    later ones is short. With the device's log of the run the reading has each one's place, and holds the log to
    the records' count where they still count; without it, it reads no execution phase past the faulted one."""
    _, arrivals, answers = traffic()
    arrivals = {t + 5: made for t, made in arrivals.items()} | {2 * T + 1: [cells(id_req(6, 3), B)]}
    answers += [answer((5,))]
    device, _, _ = run(answers, arrivals, params=TWO, windows=4, late={3: 0.05})
    ingress, egress, _ = records(device)
    assert [r.status.complete for r in ingress] == [False, True, True, True]
    order = order_of(device)
    assert ends(device) == [(0, False), (1, False), (2, True), (3, True)]
    assert [r.order for stride in handed(device)[2] for r in stride] == [5] and order[6] == 5
    assert handed(device) == {w: CR.of_records(TWO, ingress, egress, order)[w] for w in (2, 3)}
    assert sorted(CR.of_records(TWO, ingress, egress)) == [0]
    whole, _, _ = run(answers, arrivals, params=TWO, windows=4)
    assert CR.of_records(TWO, *records(whole)[:2], order_of(whole)) == CR.of_records(TWO, *records(whole)[:2])
    for wrong in ({}, order_of(whole) | {1: 1, 2: 0}):
        with pytest.raises(ValueError, match="of the run's log"):
            CR.of_records(TWO, *records(whole)[:2], wrong)


async def stopped(params: WardenParams, windows: int, ticks: int, **hooks) -> Warden:
    """A run of `windows` execution phases with no traffic, cancelled after `ticks` ticks, or ended by what its
    hooks raise."""
    clock = VirtualClock(-TAU / 2)
    device = Warden(params, Server([]).connect, origin=0.0, windows=windows, clock=clock, sleep=clock.sleep, **hooks)
    running = asyncio.create_task(device.run())
    for t in range(ticks):
        clock.advance(t * TAU)
        await device.until(lambda w, t=t: w.ticks > t or len(w.lengths) == windows)
    running.cancel()
    with pytest.raises((asyncio.CancelledError, RuntimeError)):
        await running
    return device


def test_a_run_that_ends_early_ends_the_execution_phases_it_has_not():
    """A run of four execution phases cancelled in its third has handed the first; the second, whose next one's
    records never came, the third and the fourth, which it never closed, end without releases, in order. A run whose
    publisher fails at the second's close has ended none: all four end without releases. Form 1 ends none of them."""
    device = asyncio.run(stopped(TWO, 4, 2 * T + 3))
    assert device.ticks == 2 * T + 3 and ends(device) == [(0, True), (1, False), (2, False), (3, False)]

    def failing(record, grid) -> None:
        if record.window == 1:
            raise RuntimeError("the publisher failed")

    device = asyncio.run(stopped(TWO, 4, 4 * T, publish=failing))
    assert device.ticks == 2 * T - 1 and ends(device) == [(w, False) for w in range(4)]
    assert asyncio.run(stopped(LEN, 4, 2 * T + 3)).lengths == [(0, ((),) * (T // LEN.every))]


# --- the word ------------------------------------------------------------------------------------------------------


def test_the_word_that_no_head_comes_is_signed_for_its_execution_phase_link_session_and_schedule(tmp_path):
    """The word is one certificate of the link's egress in the execution phase, over a digest of the link, the
    execution phase and the schedule's parameters under a tag of its own; its record reads back and is refused for
    another execution phase, link, session, schedule or key, with a field changed or added. A head's certificate is
    no word and the word's is no head's, and a schedule at form 1 has no such word."""
    issuer = Issuer(signers.signer(tmp_path), signers.SESSION)
    key = issuer.signer.public_key()
    cert = CR.attest_no_head(issuer, LINK, 3, TWO)
    assert cert.digest == CR.no_head_digest(LINK, 3, TWO) == tagged_sha512(
        CR.NO_HEAD_TAG, len(LINK).to_bytes(2, "big") + LINK.encode() + (3).to_bytes(8, "big") + TWO.digest)
    assert cert.verify(key, signers.SESSION) and signers.used(issuer.signer) == [0]
    doc = json.loads(json.dumps(CR.no_head(cert, TWO)))
    assert sorted(doc) == ["certificate", "format", "link", "parameters", "window"]
    CR.check_no_head(doc, key, signers.SESSION, TWO, LINK, 3)
    (tmp_path / "other").mkdir()
    other = signers.signer(tmp_path / "other", b"another").public_key()
    for args, why in (((key, signers.SESSION, TWO, LINK, 4), "the word's window"),
                      ((key, signers.SESSION, TWO, "requests", 3), "the word's link"),
                      ((key, signers.SESSION, TWO_CON, LINK, 3), "the word's parameters"),
                      ((key, signers.OTHER_SESSION, TWO, LINK, 3), "doesn't verify"),
                      ((other, signers.SESSION, TWO, LINK, 3), "doesn't verify")):
        with pytest.raises(ValueError, match=why):
            CR.check_no_head(doc, *args)
    signed = doc["certificate"]
    for changed, why in ((doc | {"format": CR.REGISTRATION}, "the word's format"),
                         (doc | {"tables": []}, "and no other"),
                         ({k: v for k, v in doc.items() if k != "certificate"}, "and no other"),
                         (doc | {"certificate": signed | {"digest": "00" * 64}}, "doesn't sign that"),
                         (doc | {"certificate": signed | {"signature": "zz"}}, "doesn't read"),
                         (doc | {"certificate": signed | {"signature": signed["signature"][:-2] + "00"}},
                          "doesn't verify")):
        with pytest.raises(ValueError, match=why):
            CR.check_no_head(changed, key, signers.SESSION, TWO, LINK, 3)
    rows, head = CR.attest(issuer, LINK, 3, ((),) * CR.strides_of(TWO), TWO)
    registration = CR.registration(rows.head, head, TWO)
    CR.check_registration(registration, key, signers.SESSION, TWO, LINK, 3)
    with pytest.raises(ValueError, match="doesn't sign that"):
        CR.check_no_head(CR.no_head(head, TWO), key, signers.SESSION, TWO, LINK, 3)
    with pytest.raises(ValueError, match="doesn't sign its head"):
        CR.check_registration(registration | {"certificate": signed}, key, signers.SESSION, TWO, LINK, 3)
    assert len({cert.digest, head.digest, CR.no_head_digest(LINK, 4, TWO), CR.no_head_digest("requests", 3, TWO),
                CR.no_head_digest(LINK, 3, TWO_CON)}) == 5
    for params in (LEN, PLAIN):
        with pytest.raises(ValueError, match="lengths is 2"):
            CR.no_head_digest(LINK, 3, params)
