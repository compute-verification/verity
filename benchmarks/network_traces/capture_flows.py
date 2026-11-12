"""capture-flows: a packet capture summed by flow and direction, and held against the network certifier's record of the
same run.

A run through the network certifier (`warden_serve.py serve`) leaves the certifier's own count of what it sent the
server and what the server sent it. A capture taken beside the run, by a machine that is neither the server nor the
certifier, counts the same bytes a second time. This reads the capture, sums it, says for the flows between the
server and the certifier whether the two counts agree in each direction, and lists every other flow that has an
address it was given on one side with its bytes. It is a measurement: it says "agrees" or the difference in bytes, and
gives no protocol's verdict.

**The capture** is a file as `tcpdump -w` writes one (pcap: either byte order, microseconds or nanoseconds), of link
type Ethernet (VLAN tags skipped), Linux cooked (v1, v2) or raw IP, holding IPv4 or IPv6. Lengths are read from the
IP and TCP headers, never from how much of a packet was kept, so a capture that keeps only each packet's first bytes
(`-s`) sums the same. Several files are one capture: an optical splitter's two directions, or a rotated capture's
parts in order. A flow is the two addresses, the two ports and the protocol, and each direction of it has
- `packets` and `wire_bytes`: every frame, as long as it was on the wire (headers, ACKs, retransmissions; no preamble
  or checksum trailer);
- `payload_bytes`: the bytes above the transport header, a retransmitted segment's each time it was sent;
- for TCP, `stream_bytes`: the payload with each sequence number counted once, which is what the receiver's
  application reads; `holes`, the sequence numbers between the stream's first and last that no kept packet carries
  (packets the capture missed); and whether its `syn`, its `fin` and a `rst` are in the capture. Sequence numbers
  wrap at 2^32, and each is read as the position nearest the data packet before it, so a stream is counted across
  the wrap, and past 2^32 bytes, while no gap between two kept packets reaches 2^31. A second SYN with another
  sequence number on the same addresses and ports starts another flow.

**What it refuses**, each by name and with the packet it met it in: pcapng (with the conversion); a link type
outside the list above; an IP fragment of a flow it reads, whose ports and length one fragment doesn't hold (nothing
is reassembled); an encapsulation, between any addresses, since the flows read here could be inside it (MPLS, PPPoE,
MACsec, provider bridging; IP in IP, GRE, ESP, AH, EtherIP, L2TP; VXLAN, Geneve and GTP-U by their UDP ports); a
packet kept too short for its headers, with how many of its bytes this reads (its frame header, its IP header and
the first 14 of TCP's or 4 of UDP's), or, where the bytes kept end before the header that says so is whole, with that
header; a file that ends inside a packet.

**Which bytes are compared.** The certifier opens one connection to the server for each client connection it admits,
and one more for the watermark stream. Of each, the record gives two counts, both of the byte stream (what a `read`
returns and a `write` is handed), so the capture's matching sum is `stream_bytes`:
- server to certifier: the bytes of the connection's `DATA` and `END` cells in the signed egress record. The
  certifier cuts the server's stream into cells in order and cuts an `END` only after the server's end of stream, so
  on a connection that ended in `END` this is every byte the server sent;
- certifier to server: the length of the request as the server gets it at its release, `release.with_release` of the
  signed ingress record's request under its place in the release order, the run's id (`deployment.json`) and its
  release bucket; nothing for a request that was never released. A request's place in the release order is its `END`
  cell's among the `END` cells of the ingress records, in the records' order, and its release bucket that cell's:
  the place is a decimal header of the request, so its length follows it. The watermark stream's is its head, a chunk
  a tick and its end, t and n read from the ingress record; the lost releases and whether it stopped are
  `warden.json`'s, which nothing signs.
A header, an ACK without data, a SYN and a FIN are in `wire_bytes` alone. A retransmitted segment is on the wire twice
and in the stream once. A packet lost before the capture point is seen only as its retransmission, and one lost after
it twice; neither moves `stream_bytes`. So on a connection that ended in `END`, in a capture that holds the flow's SYN
and FIN and no hole, the two counts are equal in each direction.

**Where only a bound holds**, the difference is printed as a bound and never as agreement:
- the capture's count is a floor in a direction where a flow has a hole or lacks its SYN or FIN: with holes alone
  the bytes on the wire are at most the holes more, else unbounded above;
- the record's count is a bound for a connection that ended in `FAIL` or was cut, or whose release was lost: the
  certifier stopped reading, so the server sent at least the record's bytes, and the server was sent at most the
  record's; likewise the watermark stream once it stopped.
A direction says "agrees" only where both counts are exact and equal, "differs by N bytes" where both are exact, and
otherwise the range the difference (the capture's less the record's) lies in. What the server answers on the
watermark's connection is in no record (the certifier never reads it). Where that connection pairs with a flow, the
answer is given and is in no sum. Where it pairs with none, the answer can't be told from the server's other bytes, so
it stays in the sum from the server, and that direction is a bound: its difference may be lower by as many bytes as
the capture counts from the server on the largest flow in no pair.

**Pairing.** No record holds a port, so the flows and the connections are held against each other as multisets of
their two counts: a connection pairs with a flow of exactly its counts that the capture holds whole, the watermark's
connection by its bytes to the server alone; then a connection whose counts are bounds, with a flow inside them. The
result counts the pairs and what stays unpaired on each side, and says the multisets agree only when every connection
and every flow is in a pair of equal counts. Two connections of equal counts are not told apart, and nothing here
says which flow is which connection.

**Whose flows it reads.** Only those with an address it was given on one side: `--server` and `--certifier`, each
`ADDR` or `ADDR:PORT` (`[ADDR]:PORT` for IPv6), as often as the machine has addresses. A port tells the server from
the certifier where they share an address. A capture on a shared link holds other users' traffic, so it is filtered
to our own addresses when it is taken, and a file that still holds a flow between two addresses that weren't given,
or a frame that isn't IP, is refused, unless `--skip-foreign`, which counts what it skipped and reads nothing else of
it. No address is printed or written: the table and the result name the server, the certifier and `other-1`,
`other-2`, … in the order the capture first shows them.

    python capture_flows.py --capture FILE [--capture FILE ...] --server ADDR[:PORT] [--certifier ADDR[:PORT]]
        [--run DIR] [--skip-foreign] --out RESULT.json

Without `--run` (a `warden_serve.py serve` run directory) it sums and lists only. It imports the standard library
alone until it reads a run."""

from __future__ import annotations

import argparse
import bisect
import ipaddress
import json
import struct
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

#: a capture file's first four bytes, and the byte order of what follows
MAGICS = {b"\xa1\xb2\xc3\xd4": ">", b"\xd4\xc3\xb2\xa1": "<", b"\xa1\xb2\x3c\x4d": ">", b"\x4d\x3c\xb2\xa1": "<"}
PCAPNG = b"\x0a\x0d\x0d\x0a"
ETHERNET, RAW, LINUX_SLL, LINUX_SLL2 = 1, 101, 113, 276
IPV4, IPV6, VLAN = 0x0800, 0x86DD, (0x8100, 0x88A8, 0x9100)
TCP, UDP = 6, 17
FIN, SYN, RST = 0x01, 0x02, 0x04
PROTOCOLS = {1: "icmp", TCP: "tcp", UDP: "udp", 58: "icmp6"}
#: the IPv6 headers walked between the fixed header and the transport's: hop-by-hop, routing, fragment, destination
#: options
EXTENSIONS, FRAGMENT = (0, 43, 44, 60), 44
#: encapsulations, refused by name: by the frame's type, by the IP protocol, by a UDP port
TUNNEL_FRAMES = {0x8847: "MPLS", 0x8848: "MPLS", 0x8864: "PPPoE", 0x88E5: "MACsec", 0x88E7: "provider bridging"}
TUNNEL_PROTOCOLS = {4: "IP in IP", 41: "IPv6 in IP", 47: "GRE", 50: "ESP", 51: "AH", 97: "EtherIP", 115: "L2TP",
                    137: "MPLS in IP"}
TUNNEL_PORTS = {4789: "VXLAN", 8472: "VXLAN", 6081: "Geneve", 2152: "GTP-U"}
#: the least a capture keeps of each packet for every header here to be whole: Ethernet with two VLAN tags, IPv6 with
#: a fragment header, TCP with every option. A packet whose headers are shorter keeps the first bytes of its content
#: in the rest. The README's command for a link captures at it, and nothing here reads it: a packet is refused by
#: what this reads of it, never by this length
SNAPSHOT = 14 + 8 + 40 + 8 + 60


class Refused(Exception):
    """A capture or a run this doesn't read, with what to do about it."""


class _Unread(Exception):
    """A packet this doesn't read, with what it is."""


#: how the refusal of a packet kept too short ends: it names no length for every capture
LENGTHS = ('No one length suits every capture, since a length past a packet\'s headers keeps the first bytes of its '
           'content: see "Taking the capture" (a link) and "On the server\'s loopback" in '
           'benchmarks/network_traces/README.md')


def _cut(kept: int, header: str) -> _Unread:
    """A packet whose `kept` bytes end before `header` is whole, the header that says how many of its bytes this
    reads."""
    return _Unread(f"kept too short for its headers: the {kept} bytes kept of it end before its {header} is whole, so "
                   f"this can't tell how many it reads. Capture again keeping more of each packet. {LENGTHS}")


def _needs(kept: int, frame: int, at: int, transport: str, first: int) -> _Unread:
    """A packet of which `kept` bytes are kept and this reads `at + first`: a frame header of `frame` bytes, its IP
    header up to `at` and the first `first` bytes of `transport`'s."""
    headers = (f"{frame} of its frame header, " if frame else "") + f"{at - frame} of its IP header"
    return _Unread(f"kept too short for its headers: this reads its first {at + first} bytes ({headers} and "
                   f"{transport}'s first {first}) and {kept} are kept. Capture again keeping at least that many of "
                   f"each packet; a packet with longer headers needs more. {LENGTHS}")


def packets(path: Path, facts: dict) -> Iterator[tuple[int, int, bytes]]:
    """Each packet of a capture file as (its link type, its length on the wire, the bytes kept of it); `facts` gets
    the file's link type and snapshot length."""
    with open(path, "rb") as fh:
        head = fh.read(24)
        if head[:4] == PCAPNG:
            raise Refused(f"{path.name} is pcapng, which this doesn't read: `tcpdump -r {path.name} -w OUT.pcap` "
                          f"rewrites it as pcap")
        order = MAGICS.get(head[:4])
        if order is None or len(head) < 24:
            raise Refused(f"{path.name} isn't a capture file as tcpdump writes one (pcap)")
        snapshot, link = struct.unpack(order + "II", head[16:])
        link &= 0x0FFFFFFF
        if link not in (ETHERNET, RAW, LINUX_SLL, LINUX_SLL2):
            raise Refused(f"{path.name} has link type {link}: this reads Ethernet (1), raw IP (101) and Linux cooked "
                          f"(113, 276) captures")
        facts.update(link_type=link, snapshot=snapshot)
        record, n = struct.Struct(order + "IIII"), 0
        while raw := fh.read(16):
            kept = record.unpack(raw)[2] if len(raw) == 16 else 0
            data = fh.read(kept) if len(raw) == 16 else b""
            if len(raw) < 16 or len(data) < kept:
                raise Refused(f"{path.name} ends inside its packet {n}: a capture stopped while writing; `tcpdump -r "
                              f"{path.name} -w OUT.pcap` keeps its whole packets")
            n += 1
            yield link, record.unpack(raw)[3], data


def read_packet(link: int, wire: int, data: bytes) -> tuple | None:
    """A packet as (source address, destination address, protocol, source port, destination port, payload bytes,
    sequence number, TCP flags, whether it is a fragment), the addresses packed; ports are None for a protocol
    without them and for a fragment, the sequence number None but for TCP. None for a frame that isn't IP; `_Unread`
    for an encapsulation and a packet kept too short."""
    if link == ETHERNET:
        if len(data) < 14:
            raise _cut(len(data), "Ethernet header")
        at, kind = 14, int.from_bytes(data[12:14], "big")
        while kind in VLAN:
            if len(data) < at + 4:
                raise _cut(len(data), "Ethernet header")
            kind, at = int.from_bytes(data[at + 2:at + 4], "big"), at + 4
    elif link in (LINUX_SLL, LINUX_SLL2):
        at = 16 if link == LINUX_SLL else 20
        if len(data) < at:
            raise _cut(len(data), "Linux cooked header")
        kind = int.from_bytes(data[14:16] if link == LINUX_SLL else data[0:2], "big")
    else:
        at, kind = 0, {4: IPV4, 6: IPV6}.get(data[0] >> 4 if data else 0)
    if kind in TUNNEL_FRAMES:
        raise _Unread(f"{TUNNEL_FRAMES[kind]}, an encapsulation this doesn't read")
    frame, fragment = at, 0
    if kind == IPV4:
        if len(data) < at + 20:
            raise _cut(len(data), "IPv4 header")
        header = (data[at] & 15) * 4
        # a segment the sender's kernel had not yet cut for the wire can be kept with a total length of 0
        above = (int.from_bytes(data[at + 2:at + 4], "big") or wire - at) - header
        fragment = int.from_bytes(data[at + 6:at + 8], "big") & 0x3FFF
        protocol, src, dst, at = data[at + 9], data[at + 12:at + 16], data[at + 16:at + 20], at + header
    elif kind == IPV6:
        if len(data) < at + 40:
            raise _cut(len(data), "IPv6 header")
        above = int.from_bytes(data[at + 4:at + 6], "big") or wire - at - 40
        protocol, src, dst, at = data[at + 6], data[at + 8:at + 24], data[at + 24:at + 40], at + 40
        while protocol in EXTENSIONS and not fragment:
            if len(data) < at + 8:
                raise _cut(len(data), "IPv6 extension header")
            size = 8 if protocol == FRAGMENT else (data[at + 1] + 1) * 8
            if protocol == FRAGMENT:
                fragment = int.from_bytes(data[at + 2:at + 4], "big") & 0xFFF9
            protocol, at, above = data[at], at + size, above - size
    else:
        return None
    if protocol in TUNNEL_PROTOCOLS:
        raise _Unread(f"{TUNNEL_PROTOCOLS[protocol]}, an encapsulation this doesn't read")
    if fragment:
        return src, dst, protocol, None, None, max(above, 0), None, 0, True
    if protocol == TCP:
        if len(data) < at + 14:
            raise _needs(len(data), frame, at, "TCP", 14)
        sport, dport, seq = struct.unpack_from("!HHI", data, at)
        return src, dst, protocol, sport, dport, max(above - (data[at + 12] >> 4) * 4, 0), seq, data[at + 13], False
    if protocol == UDP:
        if len(data) < at + 4:
            raise _needs(len(data), frame, at, "UDP", 4)
        sport, dport = struct.unpack_from("!HH", data, at)
        tunnel = TUNNEL_PORTS.get(dport) or TUNNEL_PORTS.get(sport)
        if tunnel:
            raise _Unread(f"{tunnel} by its UDP port, an encapsulation this doesn't read")
        return src, dst, protocol, sport, dport, max(above - 8, 0), None, 0, False
    return src, dst, protocol, None, None, max(above, 0), None, 0, False


class Stream:
    """One direction of a flow: its sums, and for TCP the sequence numbers its kept packets carry, as disjoint spans
    in order (`starts`, `ends`) of positions unwrapped from the 32-bit numbers."""

    def __init__(self) -> None:
        self.packets = self.wire = self.payload = 0
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.syn: int | None = None  # the SYN's sequence number, as sent
        self.first: int | None = None  # the stream's first byte's position, where the SYN is in the capture
        self.end: int | None = None  # the position past its last byte, where the FIN is
        self.at: int | None = None  # the position past the last data packet: what the next number is unwrapped against
        self.rst = False

    def add(self, wire: int, payload: int, seq: int | None, flags: int) -> None:
        self.packets, self.wire, self.payload = self.packets + 1, self.wire + wire, self.payload + payload
        if seq is None:
            return
        if flags & RST:  # a reset's sequence number needn't be the stream's
            self.rst = True
            return
        at = seq if self.at is None else self.at + ((seq - self.at + 0x80000000) & 0xFFFFFFFF) - 0x80000000
        if flags & SYN:
            self.syn, at = seq, at + 1
            self.first = at
        if payload > 0:
            self.cover(at, at + payload)
        if flags & FIN:
            self.end = at + payload
        if payload > 0 or flags & (SYN | FIN):
            self.at = at + payload

    def cover(self, start: int, end: int) -> None:
        starts, ends = self.starts, self.ends
        if self.first is not None:  # a keep-alive's byte before the stream's first is no byte of it
            start = max(start, self.first)
        if start >= end:
            return
        if ends and start == ends[-1]:
            ends[-1] = end
            return
        i, j = bisect.bisect_left(ends, start), bisect.bisect_right(starts, end)
        if i < j:
            start, end = min(start, starts[i]), max(end, ends[j - 1])
        starts[i:j], ends[i:j] = [start], [end]

    def sums(self, tcp: bool) -> dict:
        out = {"packets": self.packets, "wire_bytes": self.wire, "payload_bytes": self.payload}
        if not tcp:
            return out
        covered = sum(e - s for s, e in zip(self.starts, self.ends))
        marks = [x for x in (self.first, self.end, *self.starts[:1], *self.ends[-1:]) if x is not None]
        return out | {"stream_bytes": covered, "holes": max(marks) - min(marks) - covered if marks else 0,
                      "syn": self.syn is not None, "fin": self.end is not None, "rst": self.rst}


def endpoint(text: str) -> tuple[bytes, int | None]:
    """`ADDR` or `ADDR:PORT` (`[ADDR]:PORT` for IPv6) as (the packed address, the port or None)."""
    host, port = text, None
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        port = rest[1:] if rest.startswith(":") else None
    elif text.count(":") == 1:
        host, _, port = text.partition(":")
    try:
        return ipaddress.ip_address(host).packed, None if port is None else int(port)
    except ValueError:
        raise argparse.ArgumentTypeError("an address is ADDR or ADDR:PORT, [ADDR]:PORT for IPv6") from None


def _at(side: tuple[bytes, int | None], given: list[tuple[bytes, int | None]]) -> bool:
    """Whether `side` is one of `given`: its address, and its port where one was given."""
    return any(side[0] == address and port in (None, side[1]) for address, port in given)


def _has(side: tuple[bytes, int | None], given: list[tuple[bytes, int | None]]) -> bool:
    """Whether `side` has the address of one of `given`, on whatever port."""
    return any(side[0] == address for address, _ in given)


def read_flows(paths: list[Path], servers: list[tuple[bytes, int | None]],
               certifiers: list[tuple[bytes, int | None]] = (), skip_foreign: bool = False) -> dict:
    """The capture in `paths` summed by flow and direction (the module docstring): what the files are, what was
    skipped, the flows between the server and the certifier (`compared`), and every other flow with a given address
    on one side (`others`). `Refused` for what the module docstring's "What it refuses" lists, a flow whose two sides
    are each the server's and the certifier's, and, unless `skip_foreign`, a file holding a foreign flow or a frame
    that isn't IP."""
    files, flows = [], {}
    foreign: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    not_ip = [0, 0]
    for path in paths:
        facts = {"packets": 0}
        files.append(facts)
        for link, wire, data in packets(Path(path), facts):
            facts["packets"] += 1
            try:
                got = read_packet(link, wire, data)
            except _Unread as e:
                raise Refused(f"{Path(path).name}'s packet {facts['packets'] - 1} is {e}") from None
            if got is None:
                not_ip[0], not_ip[1] = not_ip[0] + 1, not_ip[1] + wire
                continue
            src, dst, protocol, sport, dport, payload, seq, flags, fragment = got
            a, b = (src, sport), (dst, dport)
            forward = (src, sport or 0) <= (dst, dport or 0)
            key = (protocol, a, b) if forward else (protocol, b, a)
            if not any(_has(x, servers) or _has(x, certifiers) for x in (a, b)):
                foreign[key][0], foreign[key][1] = foreign[key][0] + 1, foreign[key][1] + wire
                continue
            if fragment:
                raise Refused(f"{Path(path).name}'s packet {facts['packets'] - 1} is an IP fragment of a flow with a "
                              f"given address: one fragment doesn't hold the flow's ports and length, and this "
                              f"reassembles nothing")
            way = flows.setdefault(key, ([], []))[0 if forward else 1]
            if not way or (flags & SYN and way[-1].syn != seq):
                way.append(Stream())
            way[-1].add(wire, payload, seq, flags)
    skipped = {"foreign": {"flows": len(foreign), "packets": sum(n for n, _ in foreign.values()),
                           "wire_bytes": sum(w for _, w in foreign.values())},
               "not_ip": {"packets": not_ip[0], "wire_bytes": not_ip[1]}}
    if not skip_foreign and (foreign or not_ip[0]):
        raise Refused(f"the capture holds {len(foreign)} flow(s) ({skipped['foreign']['packets']} packets) between "
                      f"addresses that weren't given and {not_ip[0]} frame(s) that aren't IP: filter it to our own "
                      f"addresses when it is taken, or --skip-foreign counts and skips them")
    compared, others, names = [], [], {}

    def name(side: tuple[bytes, int | None], role: str | None) -> str:
        if role is None:
            role = names.setdefault(side[0], f"other-{len(names) + 1}")
        return role if side[1] is None else f"{role}:{side[1]}"

    for (protocol, a, b), (ab, ba) in flows.items():
        sides = (a, b)
        # the side that is the server's where the other is the certifier's
        ways = [w for w in (0, 1) if _at(sides[w], servers) and _at(sides[1 - w], certifiers)]
        if len(ways) == 2:
            raise Refused("a flow's two sides are each the server's and the certifier's, so its directions can't be "
                          "told apart: give the server's port (--server ADDR:PORT)")
        tcp = protocol == TCP
        roles = ["server" if _at(x, servers) else "certifier" if _at(x, certifiers) else "server" if _has(x, servers)
                 else "certifier" if _has(x, certifiers) else None for x in sides]
        given = roles.index("server") if "server" in roles else roles.index("certifier")
        for i in range(max(len(ab), len(ba))):
            streams = [s[i] if i < len(s) else Stream() for s in (ab, ba)]
            if ways and tcp:
                server = ways[0]
                compared.append({"server": name(sides[server], "server"),
                                 "certifier": name(sides[1 - server], "certifier"),
                                 "to_server": streams[1 - server].sums(True),
                                 "from_server": streams[server].sums(True)})
            else:
                others.append({"of": name(sides[given], roles[given]),
                               "peer": name(sides[1 - given], roles[1 - given]),
                               "protocol": PROTOCOLS.get(protocol, str(protocol)),
                               "sent": streams[given].sums(tcp), "received": streams[1 - given].sums(tcp)})
    return {"files": files, "skipped": skipped, "compared": compared, "others": others}


def record_counts(run: Path) -> dict:
    """What the network certifier's record of the run in `run` (`warden_serve.py serve`'s directory) says each of its
    connections to the server carried (the module docstring's "Which bytes are compared"): each admitted connection's
    `release` (its place in the release order, None if it was never released), `to_server` and `from_server`, whether
    they are `exact`, and `why` not (then the first is a ceiling and the second a floor); the watermark stream's where
    `warden.json` has one, its `release` and `from_server` None; and how many of the records it read, each execution
    phase's ingress, egress and reasons, are certified under the deployment's key."""
    import warden_serve as ws
    from verity.protocols.certifier.verifier.active import DATA
    from verity.protocols.certifier.verifier.certificate import record_head
    from verity.protocols.certifier.verifier.grid import row_count
    from verity.protocols.certifier.verifier.release import WATERMARK_END, watermark_chunk, watermark_head, with_release
    from verity.protocols.certifier.verifier.warden import END, WardenParams, audit, read_cell

    dep = json.loads((run / "deployment.json").read_text())
    params = WardenParams(**dep["params"])
    run_id = bytes.fromhex(dep["run_id"]) if dep.get("run_id") else None
    records = ws.read_records(run, params)
    key, session = ws.cs.read_public(dep["key"]), bytes.fromhex(dep["session"])
    certified, _ = ws.read_certified(run / "certified.jsonl")
    windows = 0
    for name, (link, direction) in ws.CERTIFIED.items():
        grid = params.fails_grid if name == ws.FAILS else params.grid
        for record in records[name]:
            c = certified.get((link, direction, record.window))
            windows += bool(c is not None and c.verify(key, session) and c.head == record_head(
                link, direction, record.window, record.grid, record.status, grid, params.digest))
    requests: dict[int, bytearray] = defaultdict(bytearray)
    # each released connection's (N, a): its `END` cell's place among the ingress records' `END` cells, in the
    # records' order, and that cell's bucket
    released: dict[int, tuple[int, int]] = {}
    ends = 0
    for record in sorted(records[ws.INGRESS], key=lambda r: r.window):
        for b, row in enumerate(record.grid):
            for frame in row[:row_count(row)]:
                try:
                    kind, ordinal, payload = read_cell(frame, params.cell_bytes)
                except ValueError as e:
                    raise Refused(f"{run.name}'s ingress record holds a frame that isn't a cell: {e}") from None
                if kind in (DATA, END):
                    requests[ordinal] += payload
                if kind == END:
                    released.setdefault(ordinal, (ends, record.window * params.window + b))
                    ends += 1
    connections = audit(params, records[ws.INGRESS], records[ws.EGRESS], records[ws.FAILS])["connections"]
    log = json.loads((run / "warden.json").read_text()) if (run / "warden.json").exists() else {}
    mark = log.get("watermark") or {}
    lost = set(mark.get("lost") or ())
    out = []
    for c in connections:
        o = c["ordinal"]
        n, a = released.get(o, (None, None))
        why = [] if c["failed"] is None and c["ended"] is not None else \
            [f"failed ({c['failed']})" if c["failed"] else "not ended"]
        if n in lost:
            why.append("its release was lost")
        out.append({"connection": o, "release": n, "exact": not why, "why": ", ".join(why) or None,
                    "to_server": 0 if n is None else len(with_release(bytes(requests[o]), n, run_id, a)),
                    "from_server": c["egress_bytes"]})
    if mark.get("last"):
        buckets, at = [a for _, a in sorted(released.values())], defaultdict(list)
        for n, a in sorted(released.values()):
            if n in lost:
                at[a].append(n)
        size = len(watermark_head(run_id)) + sum(len(watermark_chunk(t, bisect.bisect_right(buckets, t), at[t]))
                                                 for t in range(mark["last"][0] + 1))
        stopped = mark.get("error")
        out.append({"connection": "watermark", "release": None, "exact": stopped is None,
                    "why": stopped and f"stopped ({stopped})",
                    "to_server": size + (0 if stopped else len(WATERMARK_END)), "from_server": None})
    return {"phase_records": {"certified": windows, "of": len(ws.CERTIFIED) * dep["windows"]}, "connections": out}


def _signed(n: int) -> str:
    return f"{n:+,d}" if n else "0"


def says(low: int | None, high: int | None) -> str:
    """What a direction says of its difference, the capture's bytes less the record's, which is at least `low` and at
    most `high` (None: no bound that way): "agrees" or "differs by N bytes" where both counts are exact; where one is
    a bound, "differs" with the least the capture has more or less than the record allows, else "bound" with the
    range, which holds 0."""
    if low is not None and low == high:
        return f"differs by {_signed(low)} bytes" if low else "agrees"
    if low is not None and low > 0:
        return f"differs: the capture has at least {low:,} bytes more"
    if high is not None and high < 0:
        return f"differs: the capture has at least {-high:,} bytes less"
    return "bound: difference " + ("not bounded" if low is None and high is None else
                                   f"at most {_signed(high)} bytes" if low is None else
                                   f"at least {_signed(low)} bytes" if high is None else
                                   f"{_signed(low)} to {_signed(high)} bytes")


def compare(flows: list[dict], record: dict) -> dict:
    """The capture's flows between the server and the certifier held against the record's connections
    (`record_counts`), as the module docstring's "Where only a bound holds" and "Pairing" have it: each direction's
    sums, the range its difference lies in (`difference`: its least and its most, None where it has none) and what it
    `says`; the pairs, and what stays unpaired on each side. The server's answer on the watermark's connection is in
    no sum where that connection pairs with a flow (`watermark_answer_bytes`); where it pairs with none, the answer
    stays in the sum from the server, at most `watermark_answer_in_sum` bytes of it, and the direction's least
    difference is lower by as much."""
    def held_whole(f: dict) -> bool:
        return all(f[w]["syn"] and f[w]["fin"] and not f[w]["holes"] for w in ("to_server", "from_server"))

    whole = [f for f in flows if held_whole(f)]
    left = [(f["to_server"]["stream_bytes"], f["from_server"]["stream_bytes"]) for f in whole]
    answer = None

    def take(c: dict, fits) -> bool:
        nonlocal answer
        got = next((x for x in left if fits(c, *x)), None)
        if got is None:
            return False
        left.remove(got)
        if c["from_server"] is None:
            answer = got[1]
        return True

    # the watermark's connection has one count, so it pairs last, with a flow no other connection's two counts took
    connections = sorted(record["connections"], key=lambda c: c["from_server"] is None)
    rest = [c for c in connections
            if not take(c, lambda c, t, f: t == c["to_server"] and c["from_server"] in (None, f))]
    # a connection whose counts are bounds, with a flow inside them
    unpaired = [c for c in rest if c["exact"] or not take(
        c, lambda c, t, f: t <= c["to_server"] and (c["from_server"] is None or f >= c["from_server"]))]
    bounded = [c for c in connections if not c["exact"]]
    # the watermark's connection in no pair: its flow, if the capture holds it, is in no pair, or has the counts of
    # one that is (a least difference is given only where every other pair is of equal counts), so the sum from the
    # server holds at most the largest count of those of the server's answer on it
    stray = 0
    if any(c["from_server"] is None for c in unpaired):
        stray = max([f for _, f in left] + [f["from_server"]["stream_bytes"] for f in flows if not held_whole(f)],
                    default=0)
    out = {}
    for way in ("to_server", "from_server"):
        capture = {k: sum(f[way][k] for f in flows)
                   for k in ("packets", "wire_bytes", "payload_bytes", "stream_bytes", "holes")}
        capture |= {"flows": len(flows), "without_syn": sum(not f[way]["syn"] for f in flows),
                    "without_fin": sum(not f[way]["fin"] for f in flows), "reset": sum(f[way]["rst"] for f in flows)}
        counted = capture["stream_bytes"] - ((answer or 0) if way == "from_server" else 0)
        most = None if capture["without_syn"] or capture["without_fin"] else counted + capture["holes"]
        stated = sum(c[way] or 0 for c in connections)
        if way == "to_server":  # a bounded connection's is a ceiling
            floor, ceiling = stated - sum(c[way] for c in bounded), stated
        else:  # and a floor; the watermark's has no count from the server, and its flow's answer is in no sum
            floor, ceiling = stated, None if any(c[way] is not None for c in bounded) else stated
        low, high = None if ceiling is None else counted - ceiling, None if most is None else most - floor
        if way == "from_server" and low is not None:
            low -= stray
        out[way] = {"capture": capture, "record_bytes": stated, "difference": [low, high], "says": says(low, high)}
    keys = ("connection", "why", "to_server", "from_server")
    pairs, inside = len(connections) - len(rest), len(rest) - len(unpaired)
    return out | {"pairing": {"connections": len(connections), "flows": len(flows), "pairs": pairs,
                              "pairs_inside_bounds": inside, "unpaired_connections": len(unpaired),
                              "unpaired_flows": len(flows) - pairs - inside, "partial_flows": len(flows) - len(whole),
                              "says": "agrees" if pairs == len(connections) == len(flows) else
                              f"{len(rest)} connection(s) and {len(flows) - pairs} flow(s) have no pair of equal "
                              f"counts"},
                  "bounded": [{k: c[k] for k in keys} for c in bounded],
                  "unpaired": [{k: c[k] for k in keys} for c in unpaired],
                  "watermark_answer_bytes": answer, "watermark_answer_in_sum": stray}


def _row(cells: list, widths: list[int]) -> str:
    return "  ".join(f"{c:,}".rjust(w) if isinstance(c, int) else str(c).ljust(w)
                     for c, w in zip(cells, widths)).rstrip()


def _head(names: list[str], widths: list[int]) -> str:
    return "  ".join(h.rjust(w) if i else h.ljust(w) for i, (h, w) in enumerate(zip(names, widths)))


def table(result: dict) -> str:
    """The result as text: what was read and skipped; each direction between the server and the certifier, its sums
    and, with a run, the record's bytes and what the direction says; the pairing; then every other flow, a line each:
    what its given side sent (out) and received (in)."""
    files, skipped = result["capture"]["files"], result["capture"]["skipped"]
    lines = [f"capture: {sum(f['packets'] for f in files):,} packets in {len(files)} file(s); skipped "
             f"{skipped['foreign']['packets']:,} packets of {skipped['foreign']['flows']:,} foreign flow(s) and "
             f"{skipped['not_ip']['packets']:,} frame(s) that aren't IP"]
    widths = [20, 6, 9, 14, 14, 14, 9, 14]
    names = ["", "flows", "packets", "wire bytes", "payload bytes", "stream bytes", "holes", "record bytes"]
    flows, compared = result["between"], result.get("compared")
    if compared is not None:
        w = result["record"]["phase_records"]
        lines.append(f"record: {compared['pairing']['connections']:,} connection(s) to the server; {w['certified']} "
                     f"of its execution phases' {w['of']} records certified")
    lines += ["", _head(names if compared else names[:7], widths) + ("  says" if compared else "")]
    for way, label in (("to_server", "certifier to server"), ("from_server", "server to certifier")):
        row = [label, len(flows), *(sum(f[way][k] for f in flows)
                                    for k in ("packets", "wire_bytes", "payload_bytes", "stream_bytes", "holes"))]
        lines.append(_row(row, widths) if compared is None else
                     _row([*row, compared[way]["record_bytes"]], widths) + "  " + compared[way]["says"])
        part = [f"{n:,} flow(s) {text}" for n, text in ((sum(not f[way]["syn"] for f in flows), "without their SYN"),
                                                        (sum(not f[way]["fin"] for f in flows), "without their FIN"),
                                                        (sum(f[way]["rst"] for f in flows), "reset")) if n]
        if part:
            lines.append(f"    the capture holds {', '.join(part)}")
    if compared is not None:
        p = compared["pairing"]
        lines.append(f"pairing, a connection's two counts with a flow's: {p['pairs']:,} pair(s) of equal counts, "
                     f"{p['pairs_inside_bounds']:,} inside a connection's bounds; unpaired, "
                     f"{p['unpaired_connections']:,} of {p['connections']:,} connection(s) and "
                     f"{p['unpaired_flows']:,} of {p['flows']:,} flow(s), {p['partial_flows']:,} of them held in "
                     f"part: the multisets " + ("agree" if p["says"] == "agrees" else "differ"))
        lines += [f"    connection {c['connection']} is a bound, {c['why']}: at most {c['to_server']:,} bytes to the "
                  f"server" + ("" if c["from_server"] is None else f", at least {c['from_server']:,} from it")
                  for c in compared["bounded"]]
        if compared["watermark_answer_bytes"] is not None:
            lines.append(f"the server's answer on the watermark's connection: {compared['watermark_answer_bytes']:,} "
                         f"bytes, in no record and no sum above")
        elif any(c["from_server"] is None for c in compared["unpaired"]):
            lines.append("the watermark's connection is in no pair: the server's answer on it is in no record, and "
                         f"the sum from the server holds at most {compared['watermark_answer_in_sum']:,} bytes of it")
    others = result["others"]
    widths = [max([36, *(len(f"{f['of']} with {f['peer']}, {f['protocol']}") for f in others)]), 11, 14, 14, 11, 14, 14]
    lines += ["", _head(["other flows with a given address", "packets out", "wire bytes out", "payload out",
                         "packets in", "wire bytes in", "payload in"], widths)
              if others else "no other flow has a given address on one side"]
    lines += [_row([f"{f['of']} with {f['peer']}, {f['protocol']}",
                    *(f[way][k] for way in ("sent", "received") for k in ("packets", "wire_bytes", "payload_bytes"))],
                   widths) for f in others]
    return "\n".join(lines)


def read(paths: list[Path], servers: list, certifiers: list = (), run: Path | None = None,
         skip_foreign: bool = False) -> dict:
    """The result (the module docstring): the capture summed, and with `run` held against its record."""
    flows = read_flows(paths, servers, certifiers, skip_foreign)
    result = {"capture": {"files": flows["files"], "skipped": flows["skipped"]}, "between": flows["compared"],
              "others": flows["others"]}
    if run is not None:
        record = record_counts(run)
        result |= {"record": record, "compared": compare(flows["compared"], record)}
    return result


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="capture_flows.py", description=" ".join(__doc__.split("\n\n")[0].split()))
    ap.add_argument("--capture", action="append", required=True, type=Path, metavar="FILE",
                    help="a capture file as tcpdump writes one; several are one capture, read in this order")
    ap.add_argument("--server", action="append", required=True, type=endpoint, metavar="ADDR[:PORT]",
                    help="an address of the server's, with its port where the certifier shares the address")
    ap.add_argument("--certifier", action="append", default=[], type=endpoint, metavar="ADDR[:PORT]",
                    help="an address of the network certifier's")
    ap.add_argument("--run", type=Path, metavar="DIR", help="the run's directory, as `warden_serve.py serve` wrote it")
    ap.add_argument("--skip-foreign", action="store_true",
                    help="count and skip flows between addresses that weren't given, and frames that aren't IP")
    ap.add_argument("--out", required=True, type=Path, metavar="FILE", help="the result, as JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.run is not None and not args.certifier:
        parser().error("--run compares the flows between the server and the certifier: give --certifier")
    try:
        result = read(args.capture, args.server, args.certifier, args.run, args.skip_foreign)
    except Refused as e:
        print(f"capture-flows: refused: {e}", file=sys.stderr)
        return 2
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(table(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
