"""The lab's Program: the workload's messages as the network certifier's sessions and server-sent events, and N3 over the
records the certifiers publish.

On the mock datacenter (`../mockdc.py`) every block is a server behind the certifier's proxy on its uplink, and the client
on the outside host is the only peer of every link (`lab_run`). The proxy carries requests in and server-sent events out,
so φ's messages are recast:

- **Sessions.** A block's inbound messages read at one tick are one request: session j of block X, named `X.j`, is stream
  j + 1 on X's link, in read-tick order. A block that reads nothing, or sends a message read no later than its first read,
  starts with a session that brings nothing.
- **Work.** Session j evaluates the block's gates of the ticks from its read tick up to the next session's (the gates before
  the first read go with session 0), then answers with the block's outbound messages whose last value is ready by then
  (`Message.ready_tick`); each is read at a later tick than the session's.
- **Requests.** A request is the session's index and its payload's length (`>II`), then zero bytes that make every
  request of the block the same number of frames, F_in, the least power of two that holds its longest, then its
  messages' bytes (`framing.encode`) in stream order. On the link it is the certifier's canonical framing
  (`active.request_frames`): its length and bytes in `DATA` frames on stream 0, the last zero-filled, so the k-th
  request on a link is session k − 1's.
- **Events.** Each outbound message's bytes are cut into chunks of `active.TOKEN_BYTES` (the last may be shorter), each
  chunk one canonical event (`active.event`) after the line `id: <position>`, which the proxy reads and doesn't
  record. A session's events are then padded with empty ones (`active.event(b"")`, of no message) to the block's most,
  F, so every response of a block is F events (`fixlen`): with F_in, the served profile N2 derives a link's egress
  counts from (`verity.core.protocols.certifier.served`). The positions number a block's events 0 … n − 1 in session,
  message and chunk order, the padding last, the canonical order the certifier's shaper hands in, so the k-th event
  released on a link is position k.

A message from block X to block Y therefore crosses X's certifier in an event of X's response, and Y's in Y's request: the
client relays it. Every byte on every link is a function of the placement and the committed values.

**N3** (`check`), in the clear like `n3.check`, whose first two steps it shares: the placement registers and is causal, each
unit's values register, and every crossing value opens at its source unit. Then, per block: the k-th frame the certifier
released on the block's egress is the canonical event of the chunk the opened values declare at position k, and none
is missing; and its ingress frames, cut into requests by their lengths, are the requests the opened values declare, in
session order. When a frame crossed is the certifier's audit's to check, not N3's. The codes are `n3.check`'s
`program`, `placement`, `hardware` and `values`, and per link (`BLOCK:responses`, `BLOCK:requests`): `malformed` (not
a whole `DATA` frame on stream 0, or on egress not a canonical event), `fill` (a frame whose fill isn't zero),
`unexplained` (an event or request past the Program's), `content` (an event or request that isn't the declared one)
and `missing` (a declared event or request that wasn't recorded).
"""
from __future__ import annotations

import bisect
import struct
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from verity.core.protocols.certifier.active import (
    CANONICAL_STREAM,
    DATA,
    HEADER,
    LENGTH,
    TOKEN_BYTES,
    Requests,
    decode_frame,
    encode_frame,
    event,
    event_data,
    event_placement,
    request_frame_count,
)
from verity.core.protocols.certifier.grid import Record, row_count
from verity.ml.circuits.codec import program_digest

from . import framing as F
from . import model as M
from . import n3
from . import placement as P
from .hardware import OUTSIDE, Hardware

REQUEST = struct.Struct(">II")
EGRESS, INGRESS = "responses", "requests"


@dataclass(frozen=True)
class Session:
    block: str
    index: int
    read_tick: int  # -1 for a start session, which reads nothing
    inbound: tuple[F.Message, ...]
    outbound: tuple[F.Message, ...]
    gates: tuple[int, ...]  # the block's computed gates it evaluates, in tick order

    @property
    def name(self) -> str:
        return f"{self.block}.{self.index}"

    @property
    def stream(self) -> int:
        return self.index + 1


class Lab:
    """The lab's Program for a placed workload: each block's sessions, its events in position order as (session index,
    message, chunk index), message None for a padding event, and its (F, F_in) (`fixlen`)."""

    def __init__(self, model: M.Model, hw: Hardware, placement: P.Placement, messages: Sequence[F.Message],
                 frame_bytes: int) -> None:
        self.model, self.hw, self.frame_bytes = model, hw, frame_bytes
        self.chunk = TOKEN_BYTES
        event_placement(frame_bytes)
        self.messages = {m.stream: m for m in messages}
        self.blocks = [u for u in hw.unit_names() if u != OUTSIDE]
        self.held = {u: placement.at(hw.leaves_of(u)) for u in hw.unit_names()}
        self.sessions = {b: self._sessions(b, placement.tick_of) for b in self.blocks}
        self.events: dict[str, list[tuple[int, F.Message | None, int]]] = {}
        self._first: dict[str, list[int]] = {}
        self.fixlen: dict[str, tuple[int, int]] = {}
        for b in self.blocks:
            chunks = [[(s.index, m, j) for m in s.outbound for j in range(self.chunks(m))] for s in self.sessions[b]]
            most = max([1, *map(len, chunks)])
            longest = max([1] + [request_frame_count(self._length(s), frame_bytes) for s in self.sessions[b]])
            self.fixlen[b] = (most, 1 << (longest - 1).bit_length())
            self.events[b], self._first[b] = [], []
            for s, own in zip(self.sessions[b], chunks):
                self._first[b].append(len(self.events[b]))
                self.events[b] += own + [(s.index, None, j) for j in range(len(own), most)]
            self._first[b].append(len(self.events[b]))

    @classmethod
    def of(cls, ctx, frame_bytes: int) -> "Lab":
        """The lab's Program of a run's `block.Context`."""
        return cls(ctx.model, ctx.hw, ctx.placement, ctx.phi.messages, frame_bytes)

    def _sessions(self, block: str, tick: Sequence[int]) -> list[Session]:
        inbound: dict[int, list[F.Message]] = defaultdict(list)
        for m in self.messages.values():
            if m.dst == block:
                inbound[m.read_tick].append(m)
        ticks = sorted(inbound)
        if not ticks or any(m.src == block and m.read_tick <= ticks[0] for m in self.messages.values()):
            ticks = [-1] + ticks

        def session_of(t: int) -> int:
            return max(0, bisect.bisect_right(ticks, t) - 1)

        outbound: dict[int, list[F.Message]] = defaultdict(list)
        for m in self.messages.values():
            if m.src == block:
                outbound[session_of(m.ready_tick)].append(m)
        gates: dict[int, list[int]] = defaultdict(list)
        kind = self.model.kind
        for g in sorted(self.held[block], key=lambda g: tick[g]):
            if kind[g] == "computed":
                gates[session_of(tick[g])].append(g)
        return [Session(block, j, t, tuple(inbound.get(t, ())), tuple(outbound[j]), tuple(gates[j]))
                for j, t in enumerate(ticks)]

    def chunks(self, m: F.Message) -> int:
        return -(-((m.bits + 7) // 8) // self.chunk)

    def positions(self, block: str, session: int) -> range:
        return range(self._first[block][session], self._first[block][session + 1])

    def request(self, s: Session, payloads: dict[int, bytes]) -> bytes:
        """Session s's request, from its messages' bytes by stream."""
        body = bytes(self.request_pad(s)) + b"".join(payloads[m.stream] for m in s.inbound)
        return REQUEST.pack(s.index, len(body)) + body

    def _length(self, s: Session) -> int:
        return REQUEST.size + sum((m.bits + 7) // 8 for m in s.inbound)

    def request_pad(self, s: Session) -> int:
        """The zero bytes after session s's request head that make it its block's F_in frames."""
        n, room = self.fixlen[s.block][1], self.frame_bytes - HEADER.size
        return max(0, (n - 1) * room - LENGTH.size + 1 - self._length(s)) if n > 1 else 0

    def request_length(self, s: Session) -> int:
        return self._length(s) + self.request_pad(s)

    def request_frames(self, s: Session) -> int:
        """The link frames session s's request takes (`active.request_frames`): its block's F_in."""
        return request_frame_count(self.request_length(s), self.frame_bytes)

    def payload(self, block: str, position: int, payloads: dict[int, bytes]) -> bytes:
        """The bytes of the message whose chunk the event at `position` carries, empty for a padding event."""
        m = self.events[block][position][1]
        return b"" if m is None else payloads[m.stream]

    def token(self, block: str, position: int, payload: bytes) -> bytes:
        """The chunk the event at `position` on the block's link carries, from its message's bytes: none for a padding
        event."""
        _, m, j = self.events[block][position]
        return b"" if m is None else payload[j * self.chunk:(j + 1) * self.chunk]

    def event(self, block: str, position: int, payload: bytes) -> bytes:
        """The event at `position` as the block's server writes it: its `id:` line, then the canonical event of its
        chunk, which is what the proxy frames."""
        return b"id: %d\n" % position + event(self.token(block, position, payload))


def released(records: Sequence[Record]) -> Iterator[bytes]:
    """The frames a link's records released, in window and bucket order."""
    for rec in sorted(records, key=lambda r: r.window):
        for row in rec.grid:
            yield from row[:row_count(row)]


def _frame(f: bytes, B: int) -> tuple[str, bytes | None]:
    """A recorded frame's payload, or why it isn't a whole `DATA` frame on `CANONICAL_STREAM` with zero fill."""
    if len(f) != B or HEADER.unpack_from(f)[:2] != (DATA, CANONICAL_STREAM) or HEADER.size + HEADER.unpack_from(f)[2] > B:
        return "malformed", None
    _, _, payload = decode_frame(f)
    if encode_frame(DATA, CANONICAL_STREAM, payload, B) != f:
        return "fill", None
    return "", payload


def check(public: dict, witness: dict, records: dict[str, Sequence[Record]]) -> n3.Report:
    """N3 over the certifiers' raw records, `records[block]` holding both of the block's links' (`EGRESS`, `INGRESS`)."""
    report = n3.Report()
    model = M.build(public.get("workload", "mlp"), public["spec"])
    if program_digest(model.program) != public["program"]:
        report.add("program", "", "the public digest isn't the spec's Program's")
    hw = Hardware.from_json(public["hardware"])
    placement = n3.placement_of(public, witness, model, hw, report)
    if placement is None:
        return report
    edges = P.crossings(model, placement, hw)
    lab = Lab(model, hw, placement, F.messages(edges), public["frame_bytes"])
    opened = n3.open_values(public, witness, placement, hw, edges, model.value_bits, report)
    payloads = {s: F.encode(m, opened) for s, m in lab.messages.items() if all(e.value in opened for e in m.edges)}
    report.stats.update(messages=len(lab.messages), sessions=sum(len(ss) for ss in lab.sessions.values()),
                        events=sum(len(es) for es in lab.events.values()))
    for b in lab.blocks:
        recs = records.get(b, [])
        _egress(lab, b, [r for r in recs if r.link == EGRESS], payloads, report)
        _ingress(lab, b, [r for r in recs if r.link == INGRESS], payloads, report)
    return report


def _egress(lab: Lab, block: str, records: Sequence[Record], payloads: dict[int, bytes], report: n3.Report) -> None:
    where, declared, n = f"{block}:{EGRESS}", lab.events[block], 0
    for p, f in enumerate(released(records)):
        report.stats["frames"] += 1
        n += 1
        why, payload = _frame(f, lab.frame_bytes)
        if not why and event_data(payload) is None:
            why = "malformed"
        if why:
            report.add(why, where, f"frame {report.stats['frames'] - 1} recorded")
            continue
        if p >= len(declared):
            report.add("unexplained", where, f"an event at position {p}, past the Program's {len(declared)}")
            continue
        _, m, _ = declared[p]
        if (m is None or m.stream in payloads) and payload != event(lab.token(block, p, lab.payload(block, p, payloads))):
            report.add("content", where, f"the event at position {p} ({'padding' if m is None else f'message {m.stream}'}) "
                                         "isn't the declared one")
            continue
        report.stats["explained"] += 1
    if n < len(declared):
        report.add("missing", where, f"{len(declared) - n} declared events, the first at position {n}")


def _ingress(lab: Lab, block: str, records: Sequence[Record], payloads: dict[int, bytes], report: n3.Report) -> None:
    where, room = f"{block}:{INGRESS}", lab.frame_bytes - HEADER.size
    sessions = {s.stream: s for s in lab.sessions[block]}
    split, got, ended = Requests(), defaultdict(bytes), set()
    for f in released(records):
        report.stats["frames"] += 1
        why, payload = _frame(f, lab.frame_bytes)
        if not why and len(payload) != room:
            why = "malformed"
        if why:
            report.add(why, where, f"frame {report.stats['frames'] - 1} recorded")
            continue
        used = LENGTH.size * (not split.left)
        stream, data, end = split.feed(payload)
        if end and any(payload[used + len(data):]):
            report.add("fill", where, f"the fill after the request on stream {stream} isn't zero")
        got[stream] += data
        if end:
            ended.add(stream)
    for stream in sorted(got.keys() - sessions.keys()):
        report.add("unexplained", where, f"a request on stream {stream}, past the Program's {len(sessions)} sessions")
    for stream, s in sessions.items():
        if stream not in ended:
            report.add("missing", where, f"the request of session {s.name}")
        elif all(m.stream in payloads for m in s.inbound) and got[stream] != lab.request(s, payloads):
            report.add("content", where, f"the request of session {s.name} isn't the declared one")
        else:
            report.stats["requests"] += 1
