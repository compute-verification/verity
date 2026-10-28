"""Channels: ordered pipes of frames between two parties, untrusted by both
(`note:memory-accounting/20261026T2305Z-draft-deployment-api-v3` §2, §4.4). The runtime holds one end of a pipe per party
and channel:

- `MemoryPipe`: in one process, for tests and for the parties of one machine; in a simulated run each frame arrives
  after its channel's latency in `VirtualTime`;
- `TcpPipe` (`dial`, `TcpListen`): a TCP stream, for live runs across processes and machines;
- `Playback`: the transcript reader. It plays a recorded party's side of one channel: what it sent arrives, and what
  the live party sends must be what it received, or the run stops at the first divergence (`Refused('divergence')`);
- `Tee`: the transcript writer, which keeps every frame through any other pipe in a transcript of its own;
- `Adversary`: drops, delays, reorders or corrupts what one end sends, each by its seed.

A pipe never parses: the runtime parses what `take` returns with the wire (`verity.core.wire`), as adversarial. A stream
that ends inside a frame, or announces one past its bounds, yields the bytes it has as the frame, which the wire refuses
by name, and the pipe ends there. A stream that fails (a reset, a send to a peer gone, a connect reset or refused)
keeps what failed as `ended`, which the runtime records: in the `Closed` it tells the party, so in its transcript, and in
the run's result.
"""

from __future__ import annotations

import bisect
import random
import select
import socket
from collections import deque
from typing import Protocol

from verity.core.wire import HEADER, MAX_HEAD, FrameReader, Refused

from .sources import VirtualTime
from .transcript import Transcript


class Pipe(Protocol):
    def send(self, frame: bytes) -> None: ...

    def take(self) -> bytes | None:
        """The next frame that has arrived, else None."""

    def due(self) -> int | None:
        """When the next frame in flight arrives, in virtual time; None when none is in flight."""

    def fileno(self) -> int | None: ...

    @property
    def closed(self) -> bool:
        """Nothing more will arrive."""

    @property
    def ended(self) -> str | None:
        """What failed on the stream, the first time it did (`"ConnectionResetError on receiving: ..."`); None while
        nothing has, which a stream its peer ended cleanly keeps."""

    def close(self) -> None: ...


class MemoryPipe:
    """One end of an in-process pipe: what `send` sends arrives at the other end `latency_ns` later in `time`."""

    ended = None

    def __init__(self, time: VirtualTime, latency_ns: int = 0) -> None:
        self.time, self.latency_ns = time, latency_ns
        self.peer: MemoryPipe | None = None
        self._inbox: list[tuple[int, int, bytes]] = []
        self._seq = 0
        self._closed = False

    @classmethod
    def pair(cls, time: VirtualTime, latency_ns: int = 0, back_ns: int | None = None) -> tuple[MemoryPipe, MemoryPipe]:
        """Two ends: what the first sends takes `latency_ns`, what the second sends `back_ns` (by default the same)."""
        a, b = cls(time, latency_ns), cls(time, latency_ns if back_ns is None else back_ns)
        a.peer, b.peer = b, a
        return a, b

    def send(self, frame: bytes, delay_ns: int = 0) -> None:
        if self._closed:
            raise OSError("this end of the pipe is closed")
        p = self.peer
        if p is None or p._closed:
            return
        p._seq += 1
        bisect.insort(p._inbox, (self.time.now + self.latency_ns + delay_ns, p._seq, bytes(frame)))

    def take(self) -> bytes | None:
        if self._inbox and self._inbox[0][0] <= self.time.now:
            return self._inbox.pop(0)[2]
        return None

    def due(self) -> int | None:
        return self._inbox[0][0] if self._inbox else None

    def fileno(self) -> int | None:
        return None

    @property
    def closed(self) -> bool:
        return not self._inbox and (self._closed or self.peer is None or self.peer._closed)

    def close(self) -> None:
        self._closed = True
        self._inbox.clear()


class TcpPipe:
    """A frame stream over a connected socket. `take` reads what has arrived without blocking. A send that fails is kept
    as `ended`, not raised: the peer is gone, and the stream still yields what it sent before it went."""

    def __init__(self, sock: socket.socket, max_head: int = MAX_HEAD, max_body: int = 1 << 32) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setblocking(True)
        self.sock, self._reader, self._eof, self._closed = sock, FrameReader(max_head, max_body), False, False
        self.ended: str | None = None

    def _failed(self, doing: str, e: OSError) -> None:
        if self.ended is None:
            self.ended = f"{type(e).__name__} on {doing}: {e}"

    def send(self, frame: bytes) -> None:
        if self._closed:
            raise OSError("this end of the stream is closed")
        try:
            self.sock.sendall(frame)
        except OSError as e:
            self._failed("sending", e)

    def _fill(self) -> None:
        while not self._eof and select.select([self.sock], [], [], 0)[0]:
            try:
                data = self.sock.recv(1 << 20)
            except OSError as e:
                self._failed("receiving", e)
                data = b""
            if not data:
                self._eof = True
                return
            self._reader.feed(data)

    def take(self) -> bytes | None:
        if self._closed:
            return None
        self._fill()
        try:
            f = self._reader.next()
        except Refused:
            return self._cut(HEADER.size)
        if f is None and self._eof and self._reader.pending:
            return self._cut(self._reader.pending)
        return f

    def _cut(self, n: int) -> bytes:
        """The stream's end: its next `n` bytes as the last frame, which the wire refuses."""
        out = bytes(self._reader._buf[:n])
        self._reader._buf.clear()
        self._eof = True
        return out

    def due(self) -> int | None:
        return None

    def fileno(self) -> int | None:
        return None if self._closed or self._eof else self.sock.fileno()

    @property
    def closed(self) -> bool:
        return self._closed or (self._eof and not self._reader.pending)

    def close(self) -> None:
        """End this side of the stream after what it has sent, and read off what has arrived, so the peer gets every
        frame sent before it and an end, not a reset."""
        if not self._closed:
            self._closed = True
            try:
                self.sock.shutdown(socket.SHUT_WR)
            except OSError as e:
                self._failed("closing", e)
            self._fill()
            self.sock.close()


class EndedPipe:
    """A stream that failed before it carried a frame: it sends and yields nothing, and is closed with what failed as
    `ended`."""

    def __init__(self, ended: str) -> None:
        self.ended: str | None = ended

    def send(self, frame: bytes) -> None:
        pass

    def take(self) -> bytes | None:
        return None

    def due(self) -> int | None:
        return None

    def fileno(self) -> int | None:
        return None

    @property
    def closed(self) -> bool:
        return True

    def close(self) -> None:
        pass


def dial(address: str, max_head: int = MAX_HEAD, max_body: int = 1 << 32) -> TcpPipe | EndedPipe:
    """A `TcpPipe` to `HOST:PORT`, or an `EndedPipe` when the connect fails (a peer that resets the connection before
    the dialer resumes from the handshake makes a blocking connect raise `ConnectionResetError`)."""
    host, sep, port = address.rpartition(":")
    if not sep or not port.isdigit():
        raise ValueError(f"an address is HOST:PORT, not {address!r}")
    try:
        sock = socket.create_connection((host, int(port)))
    except OSError as e:
        return EndedPipe(f"{type(e).__name__} on connecting: {e}")
    return TcpPipe(sock, max_head, max_body)


class TcpListen:
    """A `TcpPipe` that listens on `HOST:PORT` (port 0: any) for its one connection. Frames sent before it arrives are
    held, then sent in order."""

    def __init__(self, address: str = "127.0.0.1:0", max_head: int = MAX_HEAD, max_body: int = 1 << 32) -> None:
        host, _, port = address.rpartition(":")
        self.server = socket.create_server((host, int(port)))
        self.address = "%s:%d" % self.server.getsockname()[:2]
        self.pipe: TcpPipe | None = None
        self._held: list[bytes] = []
        self._bounds = (max_head, max_body)
        self._shut = False

    def _accept(self) -> TcpPipe | None:
        if self._shut:
            return None
        if self.pipe is None:
            if not select.select([self.server], [], [], 0)[0]:
                return None
            self.server.setblocking(False)
            conn, _ = self.server.accept()
            self.server.close()
            self.pipe = TcpPipe(conn, *self._bounds)
            for f in self._held:
                self.pipe.send(f)
            self._held.clear()
        return self.pipe

    def send(self, frame: bytes) -> None:
        if self._accept() is None:
            self._held.append(bytes(frame))
        else:
            self.pipe.send(frame)

    def take(self) -> bytes | None:
        p = self._accept()
        return None if p is None else p.take()

    def due(self) -> int | None:
        return None

    def fileno(self) -> int | None:
        if self._shut:
            return None
        return self.server.fileno() if self.pipe is None else self.pipe.fileno()

    @property
    def closed(self) -> bool:
        return self._shut or (self.pipe is not None and self.pipe.closed)

    @property
    def ended(self) -> str | None:
        return None if self.pipe is None else self.pipe.ended

    def peer_gone(self) -> None:
        """The process that was to connect has ended: what it left in the backlog is still read, and with nothing
        there the listener closes."""
        self._accept()
        if self.pipe is None:
            self.close()

    def close(self) -> None:
        """Before its connection: stop listening, and nothing will arrive."""
        if self.pipe is None:
            self._shut = True
            self._held.clear()
            self.server.close()
        else:
            self.pipe.close()


class Playback:
    """The transcript reader: `recorded`'s side of its channel `channel` (its peer's name), played back. Its `out` frames
    arrive here in order; each frame sent here must be its next `in` frame, else `send` raises `Refused('divergence')`.
    The recorded handshake is not played: the live party's end runs none. A transcript that kept only the digest of a
    frame it sent on the channel can't play it (`Refused('elided')`)."""

    ended = None

    def __init__(self, recorded: Transcript, channel: str) -> None:
        self.recorded, self.channel = recorded.party, channel
        self.entries = [e for e in recorded.on(channel) if e.dir in ("in", "out")]
        if any(e.elided and e.dir == "out" for e in self.entries):
            raise Refused("elided", f"{self.recorded}'s transcript keeps only the digest of a frame it sent on {channel}")
        self.i = 0

    def send(self, frame: bytes) -> None:
        e = self.entries[self.i] if self.i < len(self.entries) else None
        if e is None or e.dir != "in" or not e.holds(frame):
            had = "nothing more" if e is None else f"{'sent' if e.dir == 'out' else 'received'} entry {e.n}"
            raise Refused("divergence", f"{self.recorded}'s transcript has {had} on {self.channel} where its peer now sends "
                                        f"{len(frame)} other bytes")
        self.i += 1

    def take(self) -> bytes | None:
        if self.i < len(self.entries) and self.entries[self.i].dir == "out":
            self.i += 1
            return self.entries[self.i - 1].data
        return None

    def due(self) -> int | None:
        return None

    def fileno(self) -> int | None:
        return None

    @property
    def closed(self) -> bool:
        return self.i >= len(self.entries)

    def close(self) -> None:
        self.i = len(self.entries)


class Tee:
    """The transcript writer: `pipe`, with every frame sent through it and taken from it appended to `transcript` (`out`
    and `in` on `channel`, of kind `kind`)."""

    def __init__(self, pipe: Pipe, transcript: Transcript, channel: str, kind: str = "private") -> None:
        self.pipe, self.transcript, self.channel, self.kind = pipe, transcript, channel, kind

    def send(self, frame: bytes) -> None:
        self.transcript.append("out", self.channel, self.kind, frame)
        self.pipe.send(frame)

    def take(self) -> bytes | None:
        f = self.pipe.take()
        if f is not None:
            self.transcript.append("in", self.channel, self.kind, f)
        return f

    def due(self) -> int | None:
        return self.pipe.due()

    def fileno(self) -> int | None:
        return self.pipe.fileno()

    @property
    def closed(self) -> bool:
        return self.pipe.closed

    @property
    def ended(self) -> str | None:
        return self.pipe.ended

    def close(self) -> None:
        self.pipe.close()


class Adversary:
    """`pipe` with what it sends tampered with, from its frame numbered `after` on (0 counts the handshake's), each frame
    by `seed`'s draws in turn: dropped (`drop`), one byte flipped (`corrupt`), held back until the next frame has gone
    (`reorder`, or until the run has nothing else to do), or delayed by `delay_ns` (`delay`, on a `MemoryPipe`). `events`
    lists what it did to which frame."""

    def __init__(self, pipe: Pipe, seed: int, *, drop: float = 0.0, corrupt: float = 0.0, reorder: float = 0.0,
                 delay: float = 0.0, delay_ns: int = 0, after: int = 0) -> None:
        self.pipe, self.rng = pipe, random.Random(seed)
        self.p = {"drop": drop, "corrupt": corrupt, "reorder": reorder, "delay": delay}
        self.delay_ns, self.after = delay_ns, after
        self.held: deque[bytes] = deque()
        self.events: list[tuple[int, str]] = []
        self._n = 0

    def send(self, frame: bytes) -> None:
        n, self._n = self._n, self._n + 1
        frame = bytes(frame)
        if n < self.after:
            self.pipe.send(frame)
            return
        draw = {k: self.rng.random() for k in ("drop", "corrupt", "reorder", "delay")}
        if draw["drop"] < self.p["drop"]:
            self.events.append((n, "drop"))
            return
        if draw["corrupt"] < self.p["corrupt"] and frame:
            i = self.rng.randrange(len(frame))
            frame = frame[:i] + bytes([frame[i] ^ (1 + self.rng.randrange(255))]) + frame[i + 1:]
            self.events.append((n, f"corrupt byte {i}"))
        if draw["reorder"] < self.p["reorder"]:
            self.events.append((n, "reorder"))
            self.held.append(frame)
            return
        if draw["delay"] < self.p["delay"]:
            self.events.append((n, f"delay {self.delay_ns} ns"))
            self.pipe.send(frame, self.delay_ns)  # type: ignore[call-arg]
        else:
            self.pipe.send(frame)
        while self.held:
            self.pipe.send(self.held.popleft())

    def idle(self) -> bool:
        """Release what is held, once the run has nothing else to do; whether anything was."""
        released = bool(self.held)
        while self.held:
            self.pipe.send(self.held.popleft())
        return released

    def take(self) -> bytes | None:
        return self.pipe.take()

    def due(self) -> int | None:
        return self.pipe.due()

    def fileno(self) -> int | None:
        return self.pipe.fileno()

    @property
    def closed(self) -> bool:
        return self.pipe.closed

    @property
    def ended(self) -> str | None:
        return self.pipe.ended

    def close(self) -> None:
        self.pipe.close()


__all__ = ["Adversary", "MemoryPipe", "Pipe", "Playback", "Tee", "TcpListen", "TcpPipe", "dial"]
