"""The egress pacer: a process of its own that writes each egress bucket to its links at a time fixed before the bucket
is made, so that nothing the handing process's event loop does within a bucket, and no reader of a link, moves when the
bucket leaves.

The loop that runs a grid's ticks also reads the link and every server connection, so when the server writes, which
the prover chooses, moves when the tick runs, and a bucket written on the loop leaves when its tick does. A thread of the
same process would wait on the same interpreter lock. The pacer waits on neither: the proxy (`active.Proxy(paced=…)`)
or the warden (`warden.Warden(paced=…)`) hands it each bucket's bytes with its time once the tick has made them
(`Pacer.hand`), and it releases them at that time on the host's monotonic clock, waiting in `poll` until `SPIN_S`
before it and spinning on the clock for the rest. A bucket is parts, each for one link, a socket the pacer holds by its
key: the proxy's one link, or each of the warden's client connections (`Pacer.start`'s links, then each `Pacer.add`
passes over a Unix socket of its own).

A release puts each part behind whatever its link still holds and writes what the link takes without waiting for it:
each link keeps its own pending bytes, written as `poll` finds it ready, so a reader slow on one link holds neither a
later bucket's release nor another link's bytes, and the pacer reads its input whenever it isn't spinning. It reports
each bucket twice (`Report`): at its release, how long after its time the bucket reached the pacer and its release
started; then, once every part is written or its link refused it, when its last byte was written and whether every
link took its part. `Pacer.behind` counts the buckets whose time has come and whose bytes aren't all written, as a
transport's backlog does on the loop. The pacer still shares its host's scheduler: at an ordinary priority, a task
that wakes on its CPU near a bucket's time, the handing process's own reads among them, can hold the release back by a
scheduler slice, which a real-time class or a core of its own on the auditor's host takes away.

The input is each bucket as its index, its time and its parts' count (`HEAD`), then each part as its link's key, its
length and whether the link closes once it is written (`PART`), then its bytes. The output is a ready byte (`READY`),
then `REPORT`s: every bucket's release in order, and every bucket's writes in order. A link that refuses a write, or
holds more than the backlog the pacer was started with, is closed with every part it still holds unwritten, and a part
for a closed link is unwritten. The pacer never waits for a link: a part whose link hasn't reached it by its bucket's
release is unwritten, and that release is reported unwritten, a fault, so a link lost on its way holds no release; it
is lost for good, every later part for it faults its release too, and its socket, if it comes, is closed. The pacer
ends at its input's end, once every bucket it read is released and written.
It shares each socket's open file, non-blocking as the handing process's transport set it, so it never changes the
file's flags."""

from __future__ import annotations

import asyncio
import gc
import os
import select
import socket
import struct
import sys
import time
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass

#: a bucket handed to the pacer: its index (u64be), its time (f64be, the host's monotonic clock) and its parts (u32be)
HEAD = struct.Struct(">QdI")
#: a part of a bucket: its link's key (u32be), its length (u32be) and whether the link closes after it (u8)
PART = struct.Struct(">II?")
#: a link passed after the start: its key (u32be), with its socket
KEY = struct.Struct(">I")
#: a report: the bucket's index, whether its writes ended (u8; else it was released), when it reached the pacer, its
#: release started and its last byte was written (0 at its release), each in seconds after its time (f64be), and whether
#: every link took its part (u8; at its release, whether every part's link had reached the pacer)
REPORT = struct.Struct(">Q?ddd?")
READY = b"\x01"
#: how long before a bucket's time the pacer stops waiting in `poll` and spins on the clock: a wait ends a little late
SPIN_S = 0.002


@dataclass(frozen=True, slots=True)
class Report:
    """Bucket `t`, when it reached the pacer (`arrived`) and its release started (`start`), each in seconds after the
    bucket's time; then, once its writes ended, when its last byte was written (`end`, None at its release) and whether
    every link took its part (`written`). At its release, `written` is whether every part's link had reached the
    pacer: a release unwritten put some part's bytes on no link."""

    t: int
    arrived: float
    start: float
    end: float | None
    written: bool


class Pacer:
    """The handing side of a pacer process (`start`): `add` gives it a link, `hand` a bucket, `behind` counts the
    buckets whose time has come and whose writes haven't ended, `reports` yields each bucket's two `Report`s as they
    come, and `close` ends its input and waits for it to exit."""

    def __init__(self, process: asyncio.subprocess.Process, channel: socket.socket) -> None:
        self.process, self.channel = process, channel
        self.unreleased: deque[int] = deque()
        self.waiting: deque[tuple[int, float]] = deque()

    @classmethod
    async def start(cls, links: Mapping[int, int] | None = None, *, backlog: int = 0) -> Pacer:
        """A pacer on `links`, each a socket's fd by its key; with `backlog`, a link holding more unwritten bytes than
        that is closed."""
        links = dict(links or {})
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", __name__, str(theirs.fileno()), str(backlog),
                *(f"{key}={fd}" for key, fd in links.items()), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, pass_fds=(theirs.fileno(), *links.values()))
        except BaseException:
            ours.close()
            raise
        finally:
            theirs.close()
        if await process.stdout.read(len(READY)) != READY:
            ours.close()
            raise RuntimeError(f"the egress pacer didn't start (exit {await process.wait()})")
        return cls(process, ours)

    def add(self, key: int, fd: int) -> None:
        """Link `key`, the socket `fd`, for the buckets handed after."""
        socket.send_fds(self.channel, [KEY.pack(key)], [fd])

    def hand(self, t: int, due: float, parts: Sequence[tuple[int, bytes, bool]]) -> None:
        """Bucket `t`, to be released at `due` on the host's monotonic clock: each part its link's key, its bytes and
        whether the link closes once they are written."""
        out = [HEAD.pack(t, due, len(parts))]
        for key, data, last in parts:
            out += [PART.pack(key, len(data), last), data]
        self.process.stdin.writelines(out)
        self.unreleased.append(t)
        self.waiting.append((t, due))

    def behind(self, now: float) -> int:
        return sum(due <= now for _, due in self.waiting)

    async def reports(self) -> AsyncIterator[Report]:
        while True:
            try:
                t, ended, arrived, start, end, written = REPORT.unpack(
                    await self.process.stdout.readexactly(REPORT.size))
            except asyncio.IncompleteReadError as e:
                if e.partial:
                    raise RuntimeError(f"the egress pacer's output ended within a report ({len(e.partial)} bytes)") from e
                return
            expected = self.waiting.popleft()[0] if ended else self.unreleased.popleft()
            if t != expected:
                raise RuntimeError(f"the egress pacer reported bucket {t}, not {expected}")
            yield Report(t, arrived, start, end if ended else None, written)

    async def close(self) -> int:
        self.process.stdin.close()
        code = await self.process.wait()
        self.channel.close()
        return code

    async def kill(self) -> None:
        if self.process.returncode is None:
            self.process.kill()
            await self.process.wait()
        self.channel.close()


class _Bucket:
    __slots__ = ("t", "due", "arrived", "parts", "start", "end", "left", "written", "lost")

    def __init__(self, t: int, due: float, arrived: float, parts: list[tuple[int, bytes, bool]]) -> None:
        self.t, self.due, self.arrived, self.parts = t, due, arrived, parts
        self.start = self.end = 0.0
        self.left, self.written, self.lost = 0, True, False


def _parse(data: bytearray) -> tuple[list[tuple[int, float, list[tuple[int, bytes, bool]]]], int]:
    """The whole buckets at `data`'s head, and how many of its bytes they take."""
    out, at = [], 0
    while len(data) - at >= HEAD.size:
        t, due, n = HEAD.unpack_from(data, at)
        i, parts = at + HEAD.size, []
        for _ in range(n):
            if len(data) - i < PART.size:
                return out, at
            key, size, last = PART.unpack_from(data, i)
            i += PART.size
            if len(data) - i < size:
                return out, at
            parts.append((key, bytes(data[i:i + size]), last))
            i += size
        out.append((t, due, parts))
        at = i
    return out, at


class _Links:
    """The pacer's links by key, each with its pending bytes: [the part's bytes not yet written, its bucket, whether
    the link closes after it]. `lost` holds the keys whose link hadn't reached the pacer by a release of a part for
    it."""

    def __init__(self, fds: dict[int, int], backlog: int) -> None:
        self.fds, self.backlog = fds, backlog
        self.pending: dict[int, deque[list]] = {key: deque() for key in fds}
        self.held: dict[int, int] = dict.fromkeys(fds, 0)
        self.closed: set[int] = set()
        self.lost: set[int] = set()

    def add(self, key: int, fd: int) -> None:
        if key in self.lost:
            os.close(fd)
            return
        self.fds[key], self.pending[key], self.held[key] = fd, deque(), 0
        self.closed.discard(key)

    def missing(self, key: int) -> bool:
        return key not in self.fds and key not in self.closed

    def put(self, key: int, data: bytes, bucket: _Bucket, last: bool) -> bool:
        if key not in self.fds:
            if key not in self.closed:
                self.lost.add(key)
                bucket.lost = True
            bucket.written = False
            return False
        if self.backlog and self.held[key] > self.backlog:
            self.close(key)
            bucket.written = False
            return False
        self.pending[key].append([memoryview(data), bucket, last])
        self.held[key] += len(data)
        bucket.left += 1
        return True

    def flush(self, key: int) -> None:
        fd, pending = self.fds[key], self.pending[key]
        while pending and _ready(fd):
            part = pending[0]
            try:
                n = os.write(fd, part[0])
            except OSError:
                self.close(key)
                return
            self.held[key] -= n
            part[0] = part[0][n:]
            if not part[0]:
                pending.popleft()
                _done(part[1])
                if part[2]:
                    self.close(key)
                    return

    def close(self, key: int) -> None:
        for _, bucket, _ in self.pending.pop(key):
            bucket.written = False
            _done(bucket)
        del self.held[key]
        os.close(self.fds.pop(key))
        self.closed.add(key)

    def waiting(self) -> dict[int, int]:
        """The fd of each link with bytes pending, to its key."""
        return {self.fds[key]: key for key, pending in self.pending.items() if pending}


def _ready(fd: int) -> bool:
    """Whether a write to `fd` won't wait: the socket has room, or an error or a hangup the write raises. Only the pacer
    writes to its links, so a socket with room keeps it until the write, which then takes at least a byte."""
    poller = select.poll()
    poller.register(fd, select.POLLOUT)
    return bool(poller.poll(0))


def _done(bucket: _Bucket) -> None:
    bucket.left -= 1
    if not bucket.left:
        bucket.end = time.monotonic()


def _report(bucket: _Bucket, ended: bool) -> None:
    os.write(1, REPORT.pack(bucket.t, ended, bucket.arrived - bucket.due, bucket.start - bucket.due,
                            bucket.end - bucket.due if ended else 0.0, bucket.written if ended else not bucket.lost))


def _receive(channel: socket.socket, links: _Links) -> bool:
    message, fds, _, _ = socket.recv_fds(channel, KEY.size, 1)
    if not message:
        return False
    if len(message) != KEY.size or len(fds) != 1:
        raise RuntimeError("a link reached the pacer without its key or its socket")
    links.add(KEY.unpack(message)[0], fds[0])
    return True


def _drain(channel: socket.socket, links: _Links) -> socket.socket | None:
    """Take every link the channel holds now, without waiting: the channel, or None once it has ended. `recv_fds`
    drops its flags, so it can't be told not to wait: each take follows a `select` that doesn't."""
    while select.select([channel], [], [], 0)[0]:
        if not _receive(channel, links):
            channel.close()
            return None
    return channel


def main(argv: list[str]) -> int:
    channel: socket.socket | None = socket.socket(fileno=int(argv[0]))
    links = _Links({int(key): int(fd) for key, fd in (a.split("=") for a in argv[2:])}, int(argv[1]))
    gc.disable()
    data, queue, flight, reading = bytearray(), deque(), deque(), True
    os.write(1, READY)
    while reading or queue or flight:
        now = time.monotonic()
        if queue and queue[0].due - now <= SPIN_S:
            bucket = queue.popleft()
            if channel is not None and any(links.missing(key) for key, _, _ in bucket.parts):
                channel = _drain(channel, links)
            while time.monotonic() < bucket.due:
                pass
            bucket.start = time.monotonic()
            keys = [key for key, part, last in bucket.parts if links.put(key, part, bucket, last)]
            for key in dict.fromkeys(keys):
                if key in links.fds:
                    links.flush(key)
            if not keys:
                bucket.end = time.monotonic()
            flight.append(bucket)
            _report(bucket, False)
        else:
            poller, waiting = select.poll(), links.waiting()
            if reading:
                poller.register(0, select.POLLIN)
            if channel is not None:
                poller.register(channel, select.POLLIN)
            for fd in waiting:
                poller.register(fd, select.POLLOUT)
            timeout = None if not queue else max(0.0, (queue[0].due - SPIN_S - now) * 1000)
            for fd, _ in poller.poll(timeout):
                if fd == 0:
                    chunk = os.read(0, 1 << 20)
                    if not chunk:
                        reading = False
                        if data:
                            raise EOFError(f"the pacer's input ended within a bucket ({len(data)} bytes)")
                        continue
                    data += chunk
                    buckets, used = _parse(data)
                    del data[:used]
                    arrived = time.monotonic()
                    queue.extend(_Bucket(t, due, arrived, parts) for t, due, parts in buckets)
                elif channel is not None and fd == channel.fileno():
                    channel = _drain(channel, links)
                elif fd in waiting and waiting[fd] in links.fds:
                    links.flush(waiting[fd])
        while flight and not flight[0].left:
            _report(flight.popleft(), True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
