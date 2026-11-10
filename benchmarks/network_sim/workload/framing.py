"""The framing φ: crossing values to positions in messages, messages to B-byte frames, frames to the links that carry them.

**Messages.** The crossing edges from unit U to unit V read at one tick are one message. Messages are numbered in
canonical order (read tick, source unit, destination unit), and message i is φ's stream i + 1, a name the wire doesn't
carry. Inside a message, edges are in value order and each value's bits are packed one after another, least significant
first, from bit 0 of byte 0; `positions` gives each value's bit offset, so its bytes are [offset // 8, ⌈(offset + width)
/ 8⌉). The last byte's unused bits are 0.

**Frames.** A message's bytes are `verity.core.protocols.certifier.active.data_frames`', the canonical framing: `DATA`
frames on `CANONICAL_STREAM` of B − 9 bytes each, the last zero-filled, so every frame's header is one constant and a
frame says nothing of the message it belongs to. φ, which is public, fixes where each message starts and ends: a
connection carries its messages in send order, each in `frame_count` frames.

**Declared traffic D.** Each connection (source unit, destination unit) opens with one `HELLO` frame, the `data_frames`
of the two units' indices (u16 each, big-endian) in `DeviceGraph.unit_names` order: a function of public data, so it
carries nothing. A frame therefore has room for at least `HELLO.size` payload bytes.

**Links and buckets.** A message crosses the certified unit boundaries on the route from its first edge's source leaf to its
destination leaf (`DeviceGraph.watched`), and on each its frames are counted in the bucket of the message's read tick, at the
reader's tick as `Placed.readOn` counts what a link delivers. A source sends its messages in send order (ready tick, then
stream), and a connection's `HELLO` goes ahead of, and is counted with, the first message sent on it (`openers`).

**The occupancy profile** (`occupancy`) is, per certified link direction and bucket, the number of frames: public, because it
is a property of the placement, and it fixes each certifier's grid.
"""
from __future__ import annotations

import struct
from collections import defaultdict
from dataclasses import dataclass

from verity.core.protocols.certifier.active import HEADER, data_frames, request_placement

from .hardware import DeviceGraph
from .placement import Edge

HELLO = struct.Struct(">HH")


@dataclass(frozen=True)
class Message:
    stream: int
    src: str
    dst: str
    read_tick: int
    edges: tuple[Edge, ...]

    @property
    def bits(self) -> int:
        return sum(e.width for e in self.edges)

    @property
    def nbytes(self) -> int:
        return (self.bits + 7) // 8

    @property
    def ready_tick(self) -> int:
        """The tick at which its last value is produced: it can be sent from then on."""
        return max(e.tick for e in self.edges)

    def positions(self) -> dict[int, int]:
        out, o = {}, 0
        for e in self.edges:
            out[e.value] = o
            o += e.width
        return out


def messages(edges: list[Edge]) -> list[Message]:
    groups: dict[tuple[int, str, str], list[Edge]] = defaultdict(list)
    for e in sorted(edges):
        groups[(e.read_tick, e.src, e.dst)].append(e)
    return [Message(i + 1, src, dst, t, tuple(es)) for i, ((t, src, dst), es) in enumerate(sorted(groups.items()))]


def encode(m: Message, value: dict[int, int]) -> bytes:
    """The message's bytes, from each crossing value's bits."""
    acc, o = 0, 0
    for e in m.edges:
        acc |= (value[e.value] & ((1 << e.width) - 1)) << o
        o += e.width
    return acc.to_bytes((o + 7) // 8, "little")


def decode(m: Message, payload: bytes) -> dict[int, int]:
    """Each crossing value at its position in the message's bytes."""
    acc, out, o = int.from_bytes(payload, "little"), {}, 0
    for e in m.edges:
        out[e.value] = (acc >> o) & ((1 << e.width) - 1)
        o += e.width
    return out


def cut(data: bytes, frame_bytes: int) -> list[bytes]:
    """`data_frames` of the bytes, frame by frame."""
    out = data_frames(data, frame_bytes)
    return [out[i:i + frame_bytes] for i in range(0, len(out), frame_bytes)]


def head(frame_bytes: int) -> bytes:
    """Every frame's header."""
    return request_placement(frame_bytes)[2]


def room(frame_bytes: int) -> int:
    return frame_bytes - HEADER.size


def frame_count(m: Message, frame_bytes: int) -> int:
    return max(1, -(-m.nbytes // room(frame_bytes)))


def hello_data(hw: DeviceGraph, src: str, dst: str) -> bytes:
    names = hw.unit_names()
    return HELLO.pack(names.index(src), names.index(dst))


def hello(hw: DeviceGraph, src: str, dst: str, frame_bytes: int) -> bytes:
    (f,) = cut(hello_data(hw, src, dst), frame_bytes)
    return f


def read_hello(hw: DeviceGraph, frame: bytes) -> tuple[str, str] | None:
    """(source unit, destination unit) of a `HELLO` frame, or None."""
    names = hw.unit_names()
    if len(frame) < HEADER.size + HELLO.size:
        return None
    s, d = HELLO.unpack_from(frame, HEADER.size)
    if s >= len(names) or d >= len(names) or frame != hello(hw, names[s], names[d], len(frame)):
        return None
    return names[s], names[d]


@dataclass(frozen=True)
class Slot:
    """One frame φ puts on a link direction: in `bucket`, a connection's `hello` or frame `index` of message `stream`."""

    bucket: int
    kind: str
    stream: int
    index: int
    src: str
    dst: str


@dataclass(frozen=True)
class Phi:
    hw: DeviceGraph
    frame_bytes: int
    messages: tuple[Message, ...]

    def send_order(self) -> list[Message]:
        return sorted(self.messages, key=lambda m: (m.ready_tick, m.stream))

    def connection(self, src: str, dst: str) -> list[Message]:
        """The messages the connection carries after its `HELLO`, in send order."""
        return [m for m in self.send_order() if (m.src, m.dst) == (src, dst)]

    def openers(self) -> dict[tuple[str, str], Message]:
        """Per connection (source unit, destination unit), the first message sent on it."""
        out: dict[tuple[str, str], Message] = {}
        for m in self.send_order():
            out.setdefault((m.src, m.dst), m)
        return out

    def route(self, m: Message) -> list[str]:
        """The certified link directions message m crosses, by `DeviceGraph.key`."""
        e = m.edges[0]
        return [self.hw.key(link, to) for link, to in self.hw.watched(e.src_leaf, e.dst_leaf)]

    def slots(self) -> dict[str, list[Slot]]:
        """Per certified link direction, the frames φ puts on it, in send order."""
        out: dict[str, list[Slot]] = defaultdict(list)
        openers = {m.stream for m in self.openers().values()}
        for m in self.send_order():
            b = self.hw.bucket(m.read_tick)
            for key in self.route(m):
                if m.stream in openers:
                    out[key].append(Slot(b, "hello", 0, 0, m.src, m.dst))
                out[key] += [Slot(b, "data", m.stream, j, m.src, m.dst) for j in range(frame_count(m, self.frame_bytes))]
        return dict(out)

    def occupancy(self) -> dict[str, list[int]]:
        """The public occupancy profile: per certified link direction, frames per bucket from bucket 0."""
        out = {}
        for key, slots in sorted(self.slots().items()):
            counts = [0] * (max(s.bucket for s in slots) + 1)
            for s in slots:
                counts[s.bucket] += 1
            out[key] = counts
        return out

    def frames(self, m: Message, value: dict[int, int]) -> list[bytes]:
        return cut(encode(m, value), self.frame_bytes)


def phi(hw: DeviceGraph, edges: list[Edge], frame_bytes: int) -> Phi:
    if frame_bytes < HEADER.size + HELLO.size:
        raise ValueError(f"a frame is at least {HEADER.size + HELLO.size} bytes, to hold a HELLO; not {frame_bytes}")
    return Phi(hw, frame_bytes, tuple(messages(edges)))
