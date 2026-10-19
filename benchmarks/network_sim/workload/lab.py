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
- **Requests.** A request is the session's index and its payload's length (`>II`), then its messages' bytes
  (`framing.encode`) in stream order.
- **Events.** Each outbound message's bytes are cut into chunks of `chunk` bytes (the last may be shorter), each chunk one
  event: `id: <position>`, `event: <message stream>`, `data: <the chunk in base64>`. The positions number a block's events
  0 … n − 1 in session, message and chunk order, the canonical order the certifier's shaper hands in. An event and its
  frame header fit one frame.

A message from block X to block Y therefore crosses X's certifier in an event of X's response, and Y's in Y's request: the
client relays it. Every byte on every link is a function of the placement and the committed values.

**N3** (`check`), in the clear like `n3.check`, whose first two steps it shares: the placement registers and is causal, each
unit's values register, and every crossing value opens at its source unit. Then, per block: every frame the certifier
released on the block's egress is a `DATA` frame of its session's stream carrying the event the opened values declare at
its position, each position once and none missing; and on its ingress each stream's frames, up to its `LAST`, are the
request the opened values declare for that session. When a frame crossed is the certifier's audit's to check, not N3's.
The codes are `n3.check`'s `program`, `placement`, `hardware` and `values`, and per link (`BLOCK:responses`,
`BLOCK:requests`): `malformed` (not a whole frame of the link's kinds, or an event without a position), `fill` (a frame
whose fill isn't zero), `unexplained` (a stream or position the Program doesn't have), `extra` (a position, or a request's
frame after its `LAST`, seen again), `content` (an event or request that isn't the declared one) and `missing` (a declared
event or request that wasn't recorded).
"""
from __future__ import annotations

import base64
import binascii
import bisect
import struct
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from verity.core.protocols.certifier.active import DATA, HEADER, LAST, decode_frame, encode_frame, event_position
from verity.core.protocols.certifier.grid import Record, row_count
from verity.ml.circuits.codec import program_digest

from . import framing as F
from . import model as M
from . import n3
from . import placement as P
from .hardware import OUTSIDE, Hardware

REQUEST = struct.Struct(">II")
EVENT_ROOM = 41  # the event's fields without its data: "id: ", "\nevent: ", "\ndata: ", "\n\n" and two 10-digit numbers
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
    """The lab's Program for a placed workload: each block's sessions, and its events in position order as (session index,
    message, chunk index)."""

    def __init__(self, model: M.Model, hw: Hardware, placement: P.Placement, messages: Sequence[F.Message],
                 frame_bytes: int) -> None:
        self.model, self.hw, self.frame_bytes = model, hw, frame_bytes
        self.chunk = 3 * ((frame_bytes - HEADER.size - EVENT_ROOM) // 4)
        if self.chunk < 3:
            raise ValueError(f"a {frame_bytes}-byte frame has no room for an event")
        self.messages = {m.stream: m for m in messages}
        self.blocks = [u for u in hw.unit_names() if u != OUTSIDE]
        self.held = {u: placement.at(hw.leaves_of(u)) for u in hw.unit_names()}
        self.sessions = {b: self._sessions(b, placement.tick_of) for b in self.blocks}
        self.events: dict[str, list[tuple[int, F.Message, int]]] = {}
        self._first: dict[str, list[int]] = {}
        for b in self.blocks:
            self.events[b], self._first[b] = [], []
            for s in self.sessions[b]:
                self._first[b].append(len(self.events[b]))
                self.events[b] += [(s.index, m, j) for m in s.outbound for j in range(self.chunks(m))]
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
        body = b"".join(payloads[m.stream] for m in s.inbound)
        return REQUEST.pack(s.index, len(body)) + body

    def request_length(self, s: Session) -> int:
        return REQUEST.size + sum((m.bits + 7) // 8 for m in s.inbound)

    def event(self, block: str, position: int, payload: bytes) -> bytes:
        """The event at `position` on the block's link, from its message's bytes."""
        _, m, j = self.events[block][position]
        chunk = payload[j * self.chunk:(j + 1) * self.chunk]
        return b"id: %d\nevent: %d\ndata: %s\n\n" % (position, m.stream, base64.b64encode(chunk))


def parse(event: bytes) -> tuple[int, bytes] | None:
    """(position, chunk) of an event, or None when it has no position or no data in base64."""
    p = event_position(event)
    data = [line[6:] for line in event.split(b"\n") if line.startswith(b"data: ")]
    if p is None or len(data) != 1:
        return None
    try:
        return p, base64.b64decode(data[0], validate=True)
    except binascii.Error:
        return None


def released(records: Sequence[Record]) -> Iterator[bytes]:
    """The frames a link's records released, in window and bucket order."""
    for rec in sorted(records, key=lambda r: r.window):
        for row in rec.grid:
            yield from row[:row_count(row)]


def _frame(f: bytes, B: int, kinds: tuple[int, ...]) -> tuple[str, tuple[int, int, bytes] | None]:
    """A recorded frame's (kind, stream, payload), or why it isn't a whole frame of `kinds` with zero fill."""
    if len(f) != B or HEADER.unpack_from(f)[0] not in kinds or HEADER.size + HEADER.unpack_from(f)[2] > B:
        return "malformed", None
    kind, stream, payload = decode_frame(f)
    if encode_frame(kind, stream, payload, B) != f:
        return "fill", None
    return "", (kind, stream, payload)


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
    where, declared, seen = f"{block}:{EGRESS}", lab.events[block], set()
    for f in released(records):
        report.stats["frames"] += 1
        why, parts = _frame(f, lab.frame_bytes, (DATA,))
        p = None if parts is None else event_position(parts[2])
        if parts is not None and p is None:
            why = "malformed"
        if why:
            report.add(why, where, f"frame {report.stats['frames'] - 1} recorded")
            continue
        if p >= len(declared):
            report.add("unexplained", where, f"an event at position {p}, past the Program's {len(declared)}")
            continue
        if p in seen:
            report.add("extra", where, f"position {p} again")
            continue
        seen.add(p)
        j, m, _ = declared[p]
        want = lab.event(block, p, payloads[m.stream]) if m.stream in payloads else None
        if parts[1] != j + 1 or (want is not None and parts[2] != want):
            report.add("content", where, f"the event at position {p} (message {m.stream}) isn't the declared one")
            continue
        report.stats["explained"] += 1
    if missing := [p for p in range(len(declared)) if p not in seen]:
        report.add("missing", where, f"{len(missing)} declared events, the first at position {missing[0]}")


def _ingress(lab: Lab, block: str, records: Sequence[Record], payloads: dict[int, bytes], report: n3.Report) -> None:
    where = f"{block}:{INGRESS}"
    sessions = {s.stream: s for s in lab.sessions[block]}
    got: dict[int, bytes] = defaultdict(bytes)
    ended: set[int] = set()
    for f in released(records):
        report.stats["frames"] += 1
        why, parts = _frame(f, lab.frame_bytes, (DATA, LAST))
        if why:
            report.add(why, where, f"frame {report.stats['frames'] - 1} recorded")
            continue
        kind, stream, payload = parts
        if stream not in sessions:
            report.add("unexplained", where, f"a frame of stream {stream}, which no session has")
        elif stream in ended:
            report.add("extra", where, f"a frame of stream {stream} after its LAST")
        else:
            got[stream] += payload
            if kind == LAST:
                ended.add(stream)
    for stream, s in sessions.items():
        if stream not in ended:
            report.add("missing", where, f"the request of session {s.name}")
        elif all(m.stream in payloads for m in s.inbound) and got[stream] != lab.request(s, payloads):
            report.add("content", where, f"the request of session {s.name} isn't the declared one")
        else:
            report.stats["requests"] += 1
