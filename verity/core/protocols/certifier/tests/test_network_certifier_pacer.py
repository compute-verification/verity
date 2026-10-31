"""The egress pacer (`pacer`): a process of its own that releases each bucket it is handed to its links' sockets, in
order, and reports each bucket's release and then its writes; a link that refuses a write gets nothing more. A link
slow to read holds its own bytes and nothing else: never a later bucket's release, nor another link's bytes. Every
bucket here is handed a time long past on the host's monotonic clock, so the pacer releases it at once and the test
reads no clock."""

import asyncio
import socket

from verity.core.protocols.certifier.pacer import Pacer


async def _paced(buckets: list[tuple[int, bytes]], refuse: bool = False):
    """Run a pacer on one end of a socket pair, hand it `buckets` (bucket t's time t ms after the clock's start), and
    return its exit, its reports, `behind` just after the last hand at two times, and the pair's other end. With
    `refuse`, the other end is closed first."""
    link, peer = socket.socketpair()
    link.setblocking(False)
    pacer = await Pacer.start({0: link.fileno()})
    link.close()
    if refuse:
        peer.close()
    reports = []

    async def collect() -> None:
        async for report in pacer.reports():
            reports.append(report)

    collecting = asyncio.create_task(collect())
    for t, data in buckets:
        pacer.hand(t, t / 1000, [(0, data, False)])
    behind = pacer.behind(-1.0), pacer.behind(1.5 / 1000)
    code = await pacer.close()
    await collecting
    return code, reports, behind, peer


def _read(peer: socket.socket) -> bytes:
    peer.setblocking(True)
    out = bytearray()
    while chunk := peer.recv(1 << 16):
        out += chunk
    return bytes(out)


def _pair(buffer: int | None = None) -> tuple[socket.socket, socket.socket]:
    link, peer = socket.socketpair()
    if buffer is not None:
        link.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer)
        peer.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer)
    link.setblocking(False)
    return link, peer


def test_the_pacer_writes_every_bucket_in_order_and_reports_each():
    """The link carries every bucket's bytes in order and nothing else; each bucket is reported released, then
    written, in order: it reached the pacer after its time, so its release started then, and the link took it. Until
    its writes are reported, a bucket whose time has come is behind."""
    buckets = [(t, bytes([t]) * (512 + t)) for t in range(4)]
    code, reports, behind, peer = asyncio.run(_paced(buckets))
    assert code == 0 and behind == (0, 2)
    assert _read(peer) == b"".join(data for _, data in buckets)
    released = [r for r in reports if r.end is None]
    written = [r for r in reports if r.end is not None]
    assert [r.t for r in released] == [r.t for r in written] == [0, 1, 2, 3]
    assert all(0 < r.arrived <= r.start for r in released)
    assert all(r.written and 0 < r.arrived <= r.start <= r.end for r in written)


def test_a_link_that_refuses_a_write_leaves_that_bucket_and_every_later_one_unwritten():
    code, reports, _, _ = asyncio.run(_paced([(t, bytes(64)) for t in range(3)], refuse=True))
    assert code == 0
    assert [(r.t, r.written) for r in reports if r.end is not None] == [(0, False), (1, False), (2, False)]


def test_a_reader_that_does_not_read_holds_no_later_bucket_s_release():
    """Bucket 0 is far more than the link's buffers hold and its reader doesn't read: bucket 1 is still released, and
    neither is written until the reader reads; then the link carries both, in order."""

    async def run():
        link, peer = _pair(4096)
        pacer = await Pacer.start({0: link.fileno()})
        link.close()
        reports = pacer.reports()
        big, small = bytes(range(256)) * 4096, b"next"
        pacer.hand(0, 0.0, [(0, big, False)])
        pacer.hand(1, 0.001, [(0, small, False)])
        released = [await anext(reports), await anext(reports)]
        held = pacer.behind(1.0)
        reading = asyncio.create_task(asyncio.to_thread(_read, peer))
        written = [await anext(reports), await anext(reports)]
        code = await pacer.close()
        return code, released, held, written, await reading, big + small

    code, released, held, written, got, sent = asyncio.run(run())
    assert code == 0 and held == 2
    assert [(r.t, r.end) for r in released] == [(0, None), (1, None)]
    assert [(r.t, r.written) for r in written] == [(0, True), (1, True)]
    assert got == sent


def test_each_link_holds_only_its_own_bytes_and_a_last_part_closes_its_link():
    """Two links passed after the start: one doesn't read, the other gets every part of its own at once, and its
    bucket's last part closes it, so its reader reaches its end while the first link still holds its bytes."""

    async def run():
        pacer = await Pacer.start()
        (slow, slow_peer), (fast, fast_peer) = _pair(4096), _pair()
        pacer.add(1, slow.fileno())
        pacer.add(2, fast.fileno())
        slow.close()
        fast.close()
        reports = pacer.reports()
        big = bytes(1 << 20)
        pacer.hand(0, 0.0, [(1, big, False), (2, b"a" * 300, False)])
        pacer.hand(1, 0.001, [(1, b"x", False), (2, b"b" * 200, True)])
        released = [await anext(reports), await anext(reports)]
        fast_got = await asyncio.to_thread(_read, fast_peer)
        reading = asyncio.create_task(asyncio.to_thread(_read, slow_peer))
        written = [await anext(reports), await anext(reports)]
        await pacer.close()
        return released, fast_got, written, await reading, big + b"x"

    released, fast_got, written, slow_got, slow_sent = asyncio.run(run())
    assert [r.t for r in released] == [0, 1] and all(r.end is None for r in released)
    assert fast_got == b"a" * 300 + b"b" * 200
    assert [(r.t, r.written) for r in written] == [(0, True), (1, True)]
    assert slow_got == slow_sent


def test_a_link_passed_just_before_its_first_bucket_holds_no_release():
    """Each link is passed just before the bucket that first writes to it, as the warden passes a client's connection
    when it admits it: its socket reaches the pacer ahead of the bucket, which may be read first and released at once,
    so the pacer takes what its channel holds at the release. Every bucket is released and written, and each link
    carries its own."""

    async def run():
        pacer = await Pacer.start()
        pairs = [_pair() for _ in range(8)]
        reports = pacer.reports()
        for key, (link, _) in enumerate(pairs):
            pacer.add(key, link.fileno())
            link.close()
            pacer.hand(key, key / 1000, [(key, bytes([key]) * 100, True)])
        got = [await anext(reports) for _ in range(2 * len(pairs))]
        await pacer.close()
        return got, [_read(peer) for _, peer in pairs]

    got, read = asyncio.run(run())
    assert sorted((r.t, r.end is None) for r in got) == [(t, end) for t in range(8) for end in (False, True)]
    assert all(r.written for r in got)
    assert read == [bytes([key]) * 100 for key in range(8)]


def test_a_link_that_never_reaches_the_pacer_faults_each_release_naming_it_and_holds_no_other():
    """Bucket 0 names link 7, whose socket the pacer never got: the pacer doesn't wait for it, and reports bucket 0's
    release unwritten (a fault) and its writes unwritten, while link 0 still gets its part and bucket 1 is released
    and written. Link 7 is lost for good: its socket, passed late, is closed, and bucket 2, naming it again, faults
    too; link 0 carries its three parts."""

    async def run():
        (link, peer), (late, late_peer) = _pair(), _pair()
        pacer = await Pacer.start({0: link.fileno()})
        link.close()
        reports = pacer.reports()
        pacer.hand(0, 0.0, [(0, b"a", False), (7, b"lost", False)])
        pacer.hand(1, 0.001, [(0, b"b", False)])
        got = [await anext(reports) for _ in range(4)]
        pacer.add(7, late.fileno())
        late.close()
        pacer.hand(2, 0.002, [(7, b"x", False), (0, b"c", True)])
        got += [await anext(reports) for _ in range(2)]
        code = await pacer.close()
        return code, got, _read(peer), _read(late_peer)

    code, got, read, late_read = asyncio.run(run())
    assert code == 0
    assert [(r.t, r.written) for r in got if r.end is None] == [(0, False), (1, True), (2, False)]
    assert [(r.t, r.written) for r in got if r.end is not None] == [(0, False), (1, True), (2, False)]
    assert read == b"abc" and late_read == b""


def test_a_link_past_the_backlog_is_closed_and_its_parts_are_unwritten():
    """A pacer started with a backlog closes a link that holds more than it unwritten: the bucket that finds it so,
    and the parts it still held, are unwritten, and the other link's are written."""

    async def run():
        pacer = await Pacer.start(backlog=1 << 16)
        (slow, slow_peer), (fast, fast_peer) = _pair(4096), _pair()
        pacer.add(1, slow.fileno())
        pacer.add(2, fast.fileno())
        slow.close()
        fast.close()
        reports = pacer.reports()
        pacer.hand(0, 0.0, [(1, bytes(1 << 20), False), (2, b"a", False)])
        pacer.hand(1, 0.001, [(1, b"x", False), (2, b"b", True)])
        got = [await anext(reports) for _ in range(4)]
        await pacer.close()
        return got, _read(fast_peer), _read(slow_peer)

    got, fast_got, slow_got = asyncio.run(run())
    assert [(r.t, r.end is None, r.written) for r in got] == [(0, True, True), (1, True, True), (0, False, False),
                                                                (1, False, False)]
    assert fast_got == b"ab" and len(slow_got) < 1 << 20
