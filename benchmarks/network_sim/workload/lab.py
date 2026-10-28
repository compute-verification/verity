"""The lab's Program: the workload's messages as the network certifier's sessions and server-sent events.

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
"""
from __future__ import annotations

import bisect
import struct
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from verity.core.protocols.certifier.active import (
    HEADER,
    LENGTH,
    TOKEN_BYTES,
    event,
    event_placement,
    request_frame_count,
)

from . import framing as F
from . import model as M
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

