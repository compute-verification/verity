"""The egress pacer (`pacer`): a process of its own that writes each bucket it is handed to the link's socket, in order,
and reports each; once the link refuses a write it writes nothing more. Every bucket here is handed a time long past
on the host's monotonic clock, so the pacer writes it at once and the test reads no clock."""

import asyncio
import socket

from verity.core.protocols.certifier.pacer import Pacer


async def _paced(buckets: list[tuple[int, bytes]], refuse: bool = False):
    """Run a pacer on one end of a socket pair, hand it `buckets` (bucket t's time t ms after the clock's start), and
    return its exit, its reports, `behind` just after the last hand at two times, and the pair's other end. With
    `refuse`, the other end is closed first."""
    link, peer = socket.socketpair()
    link.setblocking(False)
    pacer = await Pacer.start(link.fileno())
    link.close()
    if refuse:
        peer.close()
    reports = []

    async def collect() -> None:
        async for report in pacer.reports():
            reports.append(report)

    collecting = asyncio.create_task(collect())
    for t, data in buckets:
        pacer.hand(t, t / 1000, [data[:8], memoryview(data)[8:]])
    behind = pacer.behind(-1.0), pacer.behind(1.5 / 1000)
    code = await pacer.close()
    await collecting
    return code, reports, behind, peer


def _read(peer: socket.socket) -> bytes:
    out = bytearray()
    while chunk := peer.recv(1 << 16):
        out += chunk
    return bytes(out)


def test_the_pacer_writes_every_bucket_in_order_and_reports_each():
    """The link carries every bucket's bytes in order and nothing else; each report names its bucket, in order, and
    says it reached the pacer after its time, so its write started then, and the link took it. Until its reports come
    in, the buckets whose time has come are behind."""
    buckets = [(t, bytes([t]) * (512 + t)) for t in range(4)]
    code, reports, behind, peer = asyncio.run(_paced(buckets))
    assert code == 0 and behind == (0, 2)
    assert _read(peer) == b"".join(data for _, data in buckets)
    assert [r.t for r in reports] == [0, 1, 2, 3]
    assert all(r.written and 0 < r.arrived <= r.start <= r.end for r in reports)


def test_a_link_that_refuses_a_write_leaves_that_bucket_and_every_later_one_unwritten():
    code, reports, _, _ = asyncio.run(_paced([(t, bytes(64)) for t in range(3)], refuse=True))
    assert code == 0 and [(r.t, r.written) for r in reports] == [(0, False), (1, False), (2, False)]
