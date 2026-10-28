"""Adversarial cases against the active warden (`warden`, the byte-only relay). The warden closes the egress base and
the fail reason as channels, and the first two cases and the regression after them pin that.

The warden's own tests show the honest path: the egress cells, their sizes and buckets are a function of (admission s,
the egress length, the fail cell) and the public parameters, and the server's timing never reaches the wire. These show
where a *malicious server* still gets bits to the client beyond the stated v3 bound (ε·egress bits at δ, plus log2 of
the schedule, plus the fail-closed term log2(cells+1) a failed connection), and confirm which channels close.

Each case states the bits it bounds.

- **the egress base (CLOSED).** Formerly egress started at a + D, a the bucket the warden released the client's
  ingress END, which waited on the server's connect: the server shifted a by delaying its accept, ~log2(ingress_buckets)
  bit a connection. Egress now starts at s + D, s the admission bucket, and nothing the server does moves s.
- **the fail reason (CLOSED).** Formerly a FAIL cell carried a reason byte the client's shim read, so at one fail
  cell the server still picked late vs upstream, ≥1 bit a fail beyond the fail-closed term. Every FAIL is now the same
  bytes; the reason is in the warden's `fails` record, witness.
- **the fail cell (CHARGED).** The server picks which egress cell a connection fails at — the fail-closed term.
- **data vs padding (CLOSED to timing).** For a fixed (s, egress length) the cell kinds and buckets don't move with
  the server's send timing: 0 bit from timing.
- **content (tie-bound).** The client reads the relayed bytes unchanged, so the content is pinned only by the
  all-frames tie (selection All) and batch invariance; two responses of one length give one wire shape.
- **the lane (COUPLED).** A fail ends the connection at its fail bucket, freeing the lane early, so the fail cell also
  shifts the next queued connection's admission — no entropy beyond the charged fail term, but it reaches another
  client.
"""

from __future__ import annotations

import asyncio

from verity.core.protocols.certifier.active import PAD, VirtualClock
from verity.core.protocols.certifier.warden import DATA, FAIL, Warden, audit, cells
from verity.core.protocols.certifier.tests.test_network_certifier_warden import (
    B, D, EOF, P, PARAMS, R, TAU, Client, Server, ServerConn, expected_egress, reasons, records, request, run)


def _verdict(warden):
    return audit(PARAMS, *records(warden))


# --- the egress base: s + D, whatever the server's connect (CLOSED) ---------------------------------------------------


def test_the_servers_connect_delay_doesnt_move_the_egress_base_from_the_admission_bucket():
    """Formerly the egress schedule started at a + D, a the bucket the warden released the client's END, and the
    warden released nothing until the server's connection was open: a server that connected at bucket 1 or 4 set a to
    1 or 4, and the client read it from the padding before the first live cell. Now egress starts at s + D (s = 0
    here) whatever the connect: a server that connects at 1 or 2 gives the client the same bytes in the same cells, and
    one that connects at 4, past its first cell's deadline, fails there (the charged fail term), still at s + D."""

    async def go(connect_at: int):
        clock = VirtualClock(-TAU / 2)
        gate = asyncio.Event()

        async def connect():
            await gate.wait()
            conn = ServerConn([])
            conn.reader.feed_data(b"hi")
            conn.reader.feed_eof()
            conn.fed, conn.eof = len(b"hi"), True  # so `settle` sees the read complete
            return conn.reader, conn

        warden = Warden(PARAMS, connect, origin=0.0, windows=1, clock=clock, sleep=clock.sleep)
        running = asyncio.create_task(warden.run())
        client = Client(warden, cells(request(5), B))  # one-cell request, fully buffered at once
        await warden.until(lambda w: w.ordinals == 1)

        def settle(w):  # unlike the harness's, this does not block on the server connecting
            for c in w.active:
                if c.problem is None and not c.ingress_ended and c.reader._buffer:
                    return False
                if c.server is not None and (c.bytes_out != c.server[1].fed or c.egress_ended != c.server[1].eof):
                    return False
            return True

        for t in range(PARAMS.window):
            if t == connect_at:
                gate.set()
                await warden.until(lambda w: all(c.server is not None for c in w.active))
            await warden.until(settle)
            clock.advance(t * TAU)
            await warden.until(lambda w, t=t: w.ticks > t)
        await running
        await client.task
        return warden, client

    (one_w, one_c), (two_w, two_c), (late_w, late_c) = (asyncio.run(go(t)) for t in (1, 2, 4))
    assert {w.log[0]["admitted"] for w in (one_w, two_w, late_w)} == {0}  # the same admission s
    assert {w.log[0]["ingress_end"] for w in (one_w, two_w, late_w)} == {1}  # read from s on, whatever the connect
    assert one_c.writes == two_c.writes and bytes(one_c.shim().data) == b"hi"  # byte for byte, bucket for bucket
    first_live = lambda c: next(t for t, k, _ in c.cells() if k != PAD)
    assert first_live(one_c) == first_live(late_c) == 0 + D  # the base is s + D
    assert (late_w.log[0]["failed"], late_w.log[0]["failed_at"]) == ("late", 0)


def test_regression_two_server_accept_delays_give_byte_identical_client_visible_cells():
    """The harness's server accepting its connection at once, a bucket late and two buckets late (each before the
    first egress cell's deadline at s + D): the client's cells are the same bytes in the same buckets, and so are both
    wire records; the server gets the same request. Formerly each delay moved the egress base."""
    req, resp = request(5), b"r" * (2 * P + 3)
    runs = [run(Server([[(0, resp), (0, EOF)]], hold=hold), {0: [cells(req, B)]}) for hold in ({}, {0: 1}, {0: 2})]
    (w0, (c0,), s0), *others = runs
    assert c0.shim().outcome == "end" and bytes(c0.shim().data) == resp and s0.conns[0].received == req
    for w, (c,), server in others:
        assert c.writes == c0.writes
        assert records(w)[:2] == records(w0)[:2]
        assert server.conns[0].received == req


# --- the fail reason: one FAIL cell, the reason only in the fails record (CLOSED) -------------------------------------


def test_at_one_fail_cell_the_reason_is_not_on_the_wire():
    """Two connections that both fail at egress cell 0: one whose server connects but never sends (late), one whose
    server connection is refused (upstream). Formerly the FAIL cell carried the reason byte and the shim read
    "late" vs "upstream". Now the clients get the same bytes, a FAIL with no payload, and the reason is only in the
    warden's `fails` record, witness the client never sees; the audit reads it there."""
    late_w, (late_c,), _ = run([[]], {0: [cells(request(5), B)]})  # connects, never answers: late at cell 0
    up_w, (up_c,), _ = run(Server([[]], refuse=(0,)), {0: [cells(request(5), B)]})  # refused: upstream at cell 0
    assert late_w.log[0]["failed_at"] == up_w.log[0]["failed_at"] == 0  # same fail cell
    assert late_c.writes == up_c.writes  # the same bytes on the wire
    assert late_c.shim().outcome == up_c.shim().outcome == "fail"
    assert [p for _, k, p in late_c.cells() if k == FAIL] == [b""]
    assert records(late_w)[1] == records(up_w)[1]  # the same egress record
    assert ([why for _, _, why in reasons(late_w)], [why for _, _, why in reasons(up_w)]) == (["late"], ["upstream"])
    assert [c["failed"] for c in _verdict(late_w)["connections"]] == ["late"]
    assert not _verdict(late_w)["accepted"] and not _verdict(up_w)["accepted"]


# --- the fail cell: the server's choice, the charged term -------------------------------------------------------------


def test_the_server_chooses_which_egress_cell_a_connection_fails_at():
    """Sending exactly i whole cells' bytes then withholding fails the connection late at cell i. Three choices give
    three distinct client outputs (i DATA cells then a FAIL), each the same schedule up to the fail, and each refused.
    This is the fail-closed term: one of cells+1 outcomes a connection, log2(cells+1) bit (≈12 at egress_cells=4096)."""
    seen = {}
    for i in (0, 2, 5):
        warden, (client,), _ = run([[(0, b"a" * (i * P))]], {0: [cells(request(5), B)]})
        assert warden.log[0]["failed"] == "late" and warden.log[0]["failed_at"] == i
        assert [(t, k) for t, k, _ in client.cells()] == expected_egress(0, i + 1, FAIL)  # i DATA, then FAIL
        assert bytes(client.shim().data) == b"a" * (i * P)  # the client reads i cells, then the fail
        seen[i] = b"".join(d for _, d in client.writes)
        assert not _verdict(warden)["accepted"]
    assert len(set(seen.values())) == 3  # distinct outcomes


# --- data vs padding: fixed by (s, length), not the server's send timing (CLOSED to timing) --------------------------


def test_the_cell_kinds_and_buckets_dont_move_with_the_servers_send_timing():
    """The same response over the same s from servers that send it at once, in a trickle and in a late burst (all by
    their deadlines): every client gets the same (bucket, kind) for every cell, so |T| and which cells are data vs
    padding carry 0 bit of the server's timing. Only the DATA payloads (the content) can differ."""
    resp = bytes(range(256)) * 2
    whole = (len(resp) - 1) // P
    atonce = [(0, resp), (0, EOF)]
    trickle = [(i // 111, resp[i:i + 37]) for i in range(0, len(resp), 37)] + [(len(resp) // 111, EOF)]
    burst = [(0, resp[:3]), (D - 1, resp[3:]), (D - 1, EOF)]
    shapes = []
    for answer in (atonce, trickle, burst):
        warden, (client,), _ = run([answer], {2: [cells(request(30), B)]})
        assert warden.log[0]["failed"] is None, answer
        shapes.append([(t, k) for t, k, _ in client.cells()])
        assert bytes(client.shim().data) == resp
    assert all(s == shapes[0] for s in shapes)  # 0 bit from timing
    assert sum(k == DATA for _, k in shapes[0]) == whole  # the data cells are the egress length, not the timing


# --- content: the client reads the relayed bytes, pinned only by the tie ---------------------------------------------


def test_the_client_reads_the_relayed_bytes_so_content_is_pinned_only_by_the_all_frames_tie():
    """Two responses of the same length give the client different bytes in exactly the same (bucket, kind) cells, and
    the audit accepts both (it checks the schedule, not which bytes). So the content is bound only by the all-frames
    egress tie at selection All; if kernels aren't batch invariant the server chooses these bytes by batching, so the
    bound's content term (ε·egress bits at δ) rests on batch-invariant kernels (a named assumption)."""
    one = bytes(range(256))
    two = bytes((255 - x) for x in range(256))
    w1, (c1,), _ = run([[(0, one), (0, EOF)]], {0: [cells(request(5), B)]})
    w2, (c2,), _ = run([[(0, two), (0, EOF)]], {0: [cells(request(5), B)]})
    assert [(t, k) for t, k, _ in c1.cells()] == [(t, k) for t, k, _ in c2.cells()]  # same schedule
    assert bytes(c1.shim().data) == one and bytes(c2.shim().data) == two and one != two  # different content
    assert _verdict(w1)["accepted"] and _verdict(w2)["accepted"]  # the audit accepts either content


# --- the lane: a fail frees it early, coupling the next admission (within the charged term) --------------------------


def test_a_fail_ends_the_connection_at_its_cell_so_it_frees_its_lane_early_and_shifts_the_next_admission():
    """Unlike a connection held to a fixed end, a byte-cell connection ends at its fail bucket, so where connection 1
    fails changes when its lane frees and so when the queued connection is admitted. No entropy beyond the fail cell
    already charged, but the fail reaches a different client through its admission timing. Cheapest fix (if this matters
    for a targeted leak): hold a failed lane to the admitting connection's scheduled end."""
    # two lanes fill at bucket 1; a long-running connection 2 keeps its lane, so the queued connection 3 takes 1's
    arrivals = {1: [cells(request(5), B), cells(request(5), B)], 2: [cells(request(5), B)]}
    long = [(0, b"z" * ((PARAMS.egress_cells - 2) * P)), (0, EOF)]
    early, _, _ = run([[(0, b"a" * (0 * P))], long, [(0, b"ok"), (0, EOF)]], arrivals)  # 1 fails at cell 0
    later, _, _ = run([[(0, b"a" * (4 * P))], long, [(0, b"ok"), (0, EOF)]], arrivals)  # 1 fails at cell 4
    s0 = early.log[0]["admitted"]
    assert early.log[0]["failed_at"] == 0 and later.log[0]["failed_at"] == 4
    assert early.log[0]["ended"] == s0 + D and later.log[0]["ended"] == s0 + D + 4 // R  # the fail bucket
    admit = lambda w: {e["ordinal"]: e["admitted"] for e in w.log}[3]
    assert admit(early) != admit(later)  # the fail cell shifts the next connection's admission
