"""The egress pacer: a process of its own that writes each egress bucket to the link at a time fixed before the bucket
is made, so that nothing the proxy's event loop does within a bucket moves when the bucket leaves.

The loop that runs a grid's ticks also reads the link and every server connection, so when the server writes, which
the prover chooses, moves when the tick runs, and a bucket written on the loop leaves when its tick does. A thread of the
proxy's process would wait on the same interpreter lock. The pacer waits on neither: the proxy hands it each bucket's
bytes with its time once the tick has made them (`Pacer.hand`), and it writes them at that time on the host's
monotonic clock, sleeping until `SPIN_S` before it and spinning on the clock for the rest. It writes to the link's
socket, which the proxy passes it, and reports each bucket as its write ends (`Report`): how long after the bucket's
time the bucket reached it, and its write started and ended, and whether the link took it. `active.Proxy(paced=…)` reads
the reports and fails closed on them. The pacer still shares its host's scheduler: at an ordinary priority, a task that
wakes on its CPU near a bucket's time, the proxy's own reads of the server among them, can hold the write back by a
scheduler slice, which a real-time class or a core of its own on the auditor's host takes away.

The pacer reads each bucket as its index, its time and its length (`HEAD`), then its bytes, and writes a ready byte
(`READY`), then one `REPORT` a bucket, in order. It ends at its input's end, once every bucket it read is written. It
shares the socket's open file, non-blocking as the proxy's transport set it, so it waits for room with `poll` and
never changes the file's flags. Once the link refuses a write, it writes nothing more and reports every later bucket
unwritten."""

from __future__ import annotations

import asyncio
import gc
import os
import select
import struct
import sys
import time
from collections import deque
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

#: a bucket handed to the pacer: its index (u64be), its time (f64be, the host's monotonic clock) and its length (u32be)
HEAD = struct.Struct(">QdI")
#: a bucket's report: its index, then when it reached the pacer, its write started and ended, each in seconds after its
#: time (f64be), and whether the link took it (u8)
REPORT = struct.Struct(">Qddd?")
READY = b"\x01"
#: how long before a bucket's time the pacer stops sleeping and spins on the clock: a sleep ends a little late
SPIN_S = 0.002


@dataclass(frozen=True, slots=True)
class Report:
    """Bucket `t`'s write: when the bucket reached the pacer (`arrived`), and its write started and ended, each in seconds
    after the bucket's time, and whether the link took every byte (`written`)."""

    t: int
    arrived: float
    start: float
    end: float
    written: bool


class Pacer:
    """The proxy's side of a pacer process on the link's socket `link` (`start`): `hand` gives it a bucket, `behind`
    counts the buckets whose time has come and whose write hasn't ended, `reports` yields each bucket's `Report` as its
    write ends, and `close` ends its input and waits for it to exit."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self.waiting: deque[tuple[int, float]] = deque()

    @classmethod
    async def start(cls, link: int) -> Pacer:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", __name__, str(link), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            pass_fds=(link,))
        if await process.stdout.read(len(READY)) != READY:
            raise RuntimeError(f"the egress pacer didn't start (exit {await process.wait()})")
        return cls(process)

    def hand(self, t: int, due: float, cells: Iterable[bytes | memoryview]) -> None:
        """Bucket `t`'s bytes, to be written at `due` on the host's monotonic clock."""
        data = b"".join(cells)
        self.process.stdin.writelines([HEAD.pack(t, due, len(data)), data])
        self.waiting.append((t, due))

    def behind(self, now: float) -> int:
        return sum(due <= now for _, due in self.waiting)

    async def reports(self) -> AsyncIterator[Report]:
        while True:
            try:
                report = Report(*REPORT.unpack(await self.process.stdout.readexactly(REPORT.size)))
            except asyncio.IncompleteReadError:
                return
            t, _ = self.waiting.popleft()
            if report.t != t:
                raise RuntimeError(f"the egress pacer reported bucket {report.t}, not {t}")
            yield report

    async def close(self) -> int:
        self.process.stdin.close()
        return await self.process.wait()

    async def kill(self) -> None:
        if self.process.returncode is None:
            self.process.kill()
            await self.process.wait()


def _read(stream, n: int) -> bytes:
    data = stream.read(n)
    if len(data) not in (0, n):
        raise EOFError(f"the pacer's input ended within a bucket ({len(data)} of {n} bytes)")
    return data


def _wait(due: float) -> None:
    while (left := due - time.monotonic()) > SPIN_S:
        time.sleep(left - SPIN_S)
    while time.monotonic() < due:
        pass


def _write(link: int, data: bytes, poller: select.poll) -> bool:
    view = memoryview(data)
    while view:
        try:
            view = view[os.write(link, view):]
        except BlockingIOError:
            poller.poll()
        except OSError:
            return False
    return True


def main(argv: list[str]) -> int:
    link = int(argv[0])
    gc.disable()
    poller = select.poll()
    poller.register(link, select.POLLOUT)
    source, written = sys.stdin.buffer, True
    os.write(1, READY)
    while head := _read(source, HEAD.size):
        t, due, n = HEAD.unpack(head)
        data = _read(source, n)
        arrived = time.monotonic()
        _wait(due)
        start = time.monotonic()
        written = written and _write(link, data, poller)
        end = time.monotonic()
        os.write(1, REPORT.pack(t, arrived - due, start - due, end - due, written))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
