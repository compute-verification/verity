"""Adversarial cases against the active warden (`warden`, the byte-only relay), for the v3 red team (REDTEAM-ADVICE of
network-accounting). Aimed at the byte-cell warden (`cursor/warden-061a` @316f3292b6).

The warden's own tests show the honest path: the cells, their sizes and buckets are a function of (admission s, ingress
end a, the egress length, the fail cell) and the public parameters, and the server's timing never reaches the wire.
These show where a *malicious server* still gets bits to the client beyond the stated v3 bound (ε·egress bits at δ, plus
log2 of the schedule, plus the fail-closed term log2(cells+1) a failed connection), and confirm which channels close.

Each case states the bits it bounds. The two channels this red team reported OPEN on @316f3292b6 are now CLOSED by
WARDEN's fix (`cursor/warden-061a` @220b476e65, "ingress never waits on the server's connect … one FAIL code on the
wire"); their cases below are rewritten to lock the fix.

- **the egress base (CLOSED @220b476e65).** The warden releases the client's ingress on its own schedule and buffers
  what it would send before the server connects (`unsent`), so a — the bucket of the client's END, where the egress
  base a + D is read — no longer waits on the server's accept. A server slow to connect can't move a.
- **the fail reason (CLOSED @220b476e65).** A server failure is one `server` code on the wire whatever the reason
  (late/limit/upstream, `SERVER_FAILS`); the reason stays in the warden's log. At one fail cell a late and an upstream
  failure read the same outcome, so the reason carries 0 bit on the wire.
- **the fail cell (CHARGED).** The server picks which egress cell a connection fails at — the fail-closed term.
- **data vs padding (CLOSED to timing).** For a fixed (a, egress length) the cell kinds and buckets don't move with
  the server's send timing: 0 bit from timing.
- **content (tie-bound).** The client reads the relayed bytes unchanged, so the content is pinned only by the
  all-frames tie (selection All) and batch invariance; two responses of one length give one wire shape.
- **the lane (COUPLED).** A fail ends the connection at its fail bucket, freeing the lane early, so the fail cell also
  shifts the next queued connection's admission — no entropy beyond the charged fail term, but it reaches another client.
"""

from __future__ import annotations

import asyncio

from verity.core.protocols.certifier.active import PAD, VirtualClock
from verity.core.protocols.certifier.warden import (DATA, END, FAIL, FAILS, Shim, Warden, audit, cells, read_cell)
from verity.core.protocols.certifier.tests.test_network_certifier_warden import (
    B, D, EOF, P, PARAMS, R, TAU, Client, Server, ServerConn, expected_egress, records, request, run)


def _verdict(warden):
    return audit(PARAMS, *records(warden))


# --- the egress base: a no longer waits on the server's connect (CLOSED @220b476e65) ---------------------------------


def test_the_server_cannot_shift_the_egress_base_by_delaying_its_connect():
    """FIXED (@220b476e65). The warden releases the client's ingress on its own schedule and buffers what it would send
    before the server connects, so a — the bucket of the client's END, where the egress base a + D is read — no longer
    waits on the server. With the client's bytes all in from bucket 0, two very different connect times give the SAME
    admission s and the SAME a, so the egress base is the warden's, not the server's: the ~log2(ingress_buckets)
    bit/connection channel this red team reported on @316f3292b6 is closed."""

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

    early_w, early_c = asyncio.run(go(1))
    late_w, late_c = asyncio.run(go(4))
    assert early_w.log[0]["admitted"] == late_w.log[0]["admitted"] == 0  # same admission s
    assert early_w.log[0]["ingress_end"] == late_w.log[0]["ingress_end"]  # a does NOT move with the connect (fixed)
    assert bytes(early_c.shim().data) == bytes(late_c.shim().data) == b"hi"  # same bytes delivered
    # the egress base a + D is the same for both: the first live cell doesn't move with the connect
    first_live = lambda c: next(t for t, k, _ in c.cells() if k != PAD)
    assert first_live(early_c) == first_live(late_c)


# --- the fail reason: one `server` code on the wire, the reason in the log only (CLOSED @220b476e65) ------------------


def test_at_one_fail_cell_the_server_can_no_longer_pick_the_reason_on_the_wire():
    """FIXED (@220b476e65). A server failure is one `server` code on the wire whatever the reason, and the reason stays
    in the warden's log. Two connections that both fail at egress cell 0 — one whose server connects but never sends
    (late), one whose server connection is refused (upstream) — now read the SAME outcome at the client, so the ≥1
    bit/fail reason channel this red team reported on @316f3292b6 is closed. The warden's own log still records which
    reason it was, for the record."""
    late_w, (late_c,), _ = run([[]], {0: [cells(request(5), B)]})  # connects, never answers: late at cell 0
    up_w, (up_c,), _ = run(Server([[]], refuse=(0,)), {0: [cells(request(5), B)]})  # refused: upstream at cell 0
    assert late_w.log[0]["failed_at"] == up_w.log[0]["failed_at"] == 0  # same fail cell
    assert [(t, k) for t, k, _ in late_c.cells()] == [(t, k) for t, k, _ in up_c.cells()]  # identical layout
    assert late_c.shim().outcome == up_c.shim().outcome == "server"  # one code on the wire: indistinguishable
    late_fail = next(p for _, k, p in late_c.cells() if k == FAIL)
    up_fail = next(p for _, k, p in up_c.cells() if k == FAIL)
    assert late_fail == up_fail == bytes([FAILS["server"]])  # the same FAIL byte
    assert late_w.log[0]["failed"] == "late" and up_w.log[0]["failed"] == "upstream"  # the reason is in the log only
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
        a = warden.log[0]["ingress_end"]
        assert [(t, k) for t, k, _ in client.cells()] == expected_egress(0, a, i + 1, FAIL)  # i DATA, then FAIL
        assert bytes(client.shim().data) == b"a" * (i * P)  # the client reads i cells, then the fail
        seen[i] = b"".join(d for _, d in client.writes)
        assert not _verdict(warden)["accepted"]
    assert len(set(seen.values())) == 3  # distinct outcomes


# --- data vs padding: fixed by (a, length), not the server's send timing (CLOSED to timing) --------------------------


def test_the_cell_kinds_and_buckets_dont_move_with_the_servers_send_timing():
    """The same response over the same a from servers that send it at once, in a trickle and in a late burst (all by
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
    """Unlike a connection held to a fixed end, a byte-cell connection ends at its fail bucket, so where connection 0
    fails changes when its lane frees and so when the queued connection is admitted. No entropy beyond the fail cell
    already charged, but the fail reaches a different client through its admission timing. Cheapest fix (if this matters
    for a targeted leak): hold a failed lane to the admitting connection's scheduled end."""
    # two lanes fill at bucket 1; a long-running connection 1 keeps its lane, so the queued connection 2 takes 0's
    arrivals = {1: [cells(request(5), B), cells(request(5), B)], 2: [cells(request(5), B)]}
    long = [(0, b"z" * ((PARAMS.egress_cells - 2) * P)), (0, EOF)]
    early, _, _ = run([[(0, b"a" * (0 * P))], long, [(0, b"ok"), (0, EOF)]], arrivals)  # 0 fails at cell 0
    later, _, _ = run([[(0, b"a" * (4 * P))], long, [(0, b"ok"), (0, EOF)]], arrivals)  # 0 fails at cell 4
    a0 = early.log[0]["ingress_end"]
    assert early.log[0]["failed_at"] == 0 and later.log[0]["failed_at"] == 4
    assert early.log[0]["ended"] == a0 + D and later.log[0]["ended"] == a0 + D + 4 // R  # the fail bucket
    admit = lambda w: {e["ordinal"]: e["admitted"] for e in w.log}[2]
    assert admit(early) != admit(later)  # the fail cell shifts the next connection's admission
