"""The clock of a run its caller drives event by event."""

from __future__ import annotations

import asyncio
import heapq
import itertools


class VirtualClock:
    """A clock its caller moves, for runs driven event by event (tests, and replays without the host's timer):
    `clock()` reads it, `sleep` waits until `advance` passes the deadline."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self._waiters: list[tuple[float, int, asyncio.Future]] = []
        self._seq = itertools.count()

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        if delay <= 0:
            await asyncio.sleep(0)
            return
        future = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (self.now + delay, next(self._seq), future))
        await future

    def advance(self, to: float) -> None:
        self.now = max(self.now, to)
        while self._waiters and self._waiters[0][0] <= self.now:
            future = heapq.heappop(self._waiters)[2]
            if not future.done():
                future.set_result(None)
