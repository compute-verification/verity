"""The suite's capture fixture, `capture_flows.pcap`, and the packets its tests build other captures from. Run by hand,
it writes the file beside it:

    python benchmarks/network_traces/tests/fixtures/capture_flows.py

Every address is a documentation address, every hardware address is zero and a packet's time stamp is its index, so
the file is the same wherever it is written. It keeps at most each packet's first `KEPT` bytes, as a capture taken
for the reader does: every header and, where the headers are shorter, the first bytes of its content. It holds, their
packets interleaved:
- two connections from the certifier to the server: on the first the request's second segment is sent twice and the
  response's last two segments arrive out of order; on the second the request's first segment is sent again together
  with the bytes after it;
- the server's flows beside those, with a third machine: a TCP connection, a UDP exchange and an ICMP echo;
- a connection between two machines that are neither."""

from __future__ import annotations

import ipaddress
import struct
from pathlib import Path

SERVER, CERTIFIER, BESIDE = ("192.0.2.10", 8000), "198.51.100.7", "203.0.113.5"
FOREIGN = ("203.0.113.20", "203.0.113.21")
SERVER6, CERTIFIER6 = ("2001:db8::10", 8000), "2001:db8::7"
#: what the fixture keeps of each packet: the reader's `SNAPSHOT`
KEPT = 130
FIN, SYN, RST, PSH, ACK = 0x01, 0x02, 0x04, 0x08, 0x10
ICMP, TCP, UDP = 1, 6, 17
ETHERNET, RAW, LINUX_SLL, LINUX_SLL2 = 1, 101, 113, 276


def data(n: int) -> bytes:
    return bytes(i % 251 for i in range(n))


def tcp(sport: int, dport: int, seq: int, flags: int, payload: bytes = b"", ack: int = 0) -> bytes:
    return struct.pack("!HHIIBBHHH", sport, dport, seq & 0xFFFFFFFF, ack & 0xFFFFFFFF, 5 << 4, flags, 0xFFFF, 0,
                       0) + payload


def udp(sport: int, dport: int, payload: bytes) -> bytes:
    return struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload


def ip(src: str, dst: str, protocol: int, body: bytes, *, fragment: int = 0, above: int | None = None,
       extensions: bytes = b"") -> bytes:
    """An IP packet, IPv4 or IPv6 by its addresses: `fragment` is IPv4's flags and offset field, `above` the length
    its header states for `body` where the packet is cut short of it, and `extensions` IPv6 extension headers, the
    first's type 60 and the last's next header `protocol`."""
    a, b, size = ipaddress.ip_address(src), ipaddress.ip_address(dst), len(body) if above is None else above
    if a.version == 4:
        return struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + size, 0, fragment, 64, protocol, 0, a.packed, b.packed) + body
    return struct.pack("!IHBB16s16s", 6 << 28, len(extensions) + size, 60 if extensions else protocol, 64, a.packed,
                       b.packed) + extensions + body


def ethernet(packet: bytes, vlans: tuple[int, ...] = (), kind: int | None = None) -> bytes:
    """An Ethernet frame of an IP packet (or of `kind`), under the VLAN tags `vlans`, outermost first."""
    kind = kind or (0x0800 if packet[0] >> 4 == 4 else 0x86DD)
    return bytes(12) + b"".join(struct.pack("!HH", 0x8100, v) for v in vlans) + struct.pack("!H", kind) + packet


def cooked(packet: bytes, version: int = 1) -> bytes:
    """A Linux cooked capture's frame of an IP packet."""
    kind = 0x0800 if packet[0] >> 4 == 4 else 0x86DD
    return (bytes(14) + struct.pack("!H", kind) if version == 1 else struct.pack("!H", kind) + bytes(18)) + packet


def pcap(frames: list, *, link: int = ETHERNET, order: str = "<", nanos: bool = False, snapshot: int = 262144) -> bytes:
    """A capture file of `frames`, each cut to `snapshot` bytes: a frame is its bytes, or (the bytes kept, its length
    on the wire)."""
    out = [struct.pack(order + "IHHiIII", 0xA1B23C4D if nanos else 0xA1B2C3D4, 2, 4, 0, 0, snapshot, link)]
    for i, frame in enumerate(frames):
        kept, wire = frame if isinstance(frame, tuple) else (frame, len(frame))
        out.append(struct.pack(order + "IIII", i, 0, min(len(kept), snapshot), wire) + kept[:snapshot])
    return b"".join(out)


class Connection:
    """A TCP connection's IP packets in the order they are sent, `ends` its two (address, port), the first the side
    that opens it."""

    def __init__(self, opener: tuple[str, int], listener: tuple[str, int], isn: tuple[int, int] = (1000, 5000)) -> None:
        self.ends, self.next, self.packets = (opener, listener), list(isn), []

    def packet(self, side: int, seq: int, flags: int, payload: bytes = b"") -> bytes:
        (src, sport), (dst, dport) = self.ends[side], self.ends[1 - side]
        self.packets.append(ip(src, dst, TCP, tcp(sport, dport, seq, flags, payload, self.next[1 - side])))
        return self.packets[-1]

    def open(self) -> Connection:
        for side, flags in ((0, SYN), (1, SYN | ACK)):
            self.packet(side, self.next[side], flags)
            self.next[side] += 1
        self.packet(0, self.next[0], ACK)
        return self

    def send(self, side: int, payload: bytes, *, at: int | None = None, acked: bool = True) -> bytes:
        """One segment of `side`'s bytes, at its next sequence number or at `at`, then the other side's ACK when
        `acked`; returns the segment's packet."""
        seq = self.next[side] if at is None else at
        packet = self.packet(side, seq, ACK | PSH, payload)
        self.next[side] = max(self.next[side], seq + len(payload))
        if acked:
            self.packet(1 - side, self.next[1 - side], ACK)
        return packet

    def close(self) -> Connection:
        for side in (1, 0):
            self.packet(side, self.next[side], FIN | ACK)
            self.next[side] += 1
        self.packet(1, self.next[1], ACK)
        return self


def exchange(opener: tuple[str, int], listener: tuple[str, int], sent: bytes, answered: bytes, *,
             isn: tuple[int, int] = (1000, 5000), mss: int = 536) -> list[bytes]:
    """A whole connection's packets: `opener` sends `sent`, `listener` answers `answered`, in segments of `mss`."""
    c = Connection(opener, listener, isn).open()
    for side, stream in ((0, sent), (1, answered)):
        for i in range(0, len(stream), mss):
            c.send(side, stream[i:i + mss])
    return c.close().packets


def interleaved(*flows: list[bytes]) -> list[bytes]:
    """The flows' packets in turn, one of each at a time, each flow's in its own order."""
    return [flow[i] for i in range(max(map(len, flows))) for flow in flows if i < len(flow)]


def first() -> list[bytes]:
    """The certifier's first connection: a request of 300 bytes whose second segment is sent twice, and a response of
    700 whose last two segments arrive out of order."""
    c = Connection((CERTIFIER, 40001), SERVER).open()
    c.send(0, data(200))
    again = c.next[0]
    c.send(0, data(100), acked=False)
    c.send(0, data(100), at=again)
    c.send(1, data(300))
    c.send(1, data(100), at=c.next[1] + 300, acked=False)
    c.send(1, data(300), at=c.next[1] - 400)
    return c.close().packets


def second() -> list[bytes]:
    """The certifier's second connection: a request of 120 bytes whose first 50 are sent again together with the 70
    after them, and a response of 64."""
    c = Connection((CERTIFIER, 40002), SERVER, (77000, 88000)).open()
    start = c.next[0]
    c.send(0, data(50), acked=False)
    c.send(0, data(120), at=start)
    c.send(1, data(64))
    return c.close().packets


def beside() -> list[list[bytes]]:
    """The server's flows with a third machine: a TCP connection it opens (90 bytes out, 400 back), a UDP exchange
    (40 out, 120 back) and an ICMP echo and its reply (64 bytes above the IP header each way)."""
    echo = struct.pack("!BBHHH", 8, 0, 0, 1, 1) + data(56)
    return [exchange((SERVER[0], 50000), (BESIDE, 443), data(90), data(400), isn=(300, 400)),
            [ip(SERVER[0], BESIDE, UDP, udp(53535, 53, data(40))),
             ip(BESIDE, SERVER[0], UDP, udp(53, 53535, data(120)))],
            [ip(SERVER[0], BESIDE, ICMP, echo), ip(BESIDE, SERVER[0], ICMP, bytes(1) + echo[1:])]]


def foreign() -> list[bytes]:
    """A connection between two machines that are neither the server nor the certifier: four packets, 10 bytes sent."""
    c = Connection((FOREIGN[0], 1234), (FOREIGN[1], 80)).open()
    c.send(0, data(10), acked=False)
    return c.packets


def capture() -> bytes:
    """The fixture's bytes."""
    return pcap([ethernet(p) for p in interleaved(first(), second(), *beside(), foreign())], snapshot=KEPT)


if __name__ == "__main__":
    Path(__file__).with_suffix(".pcap").write_bytes(capture())
