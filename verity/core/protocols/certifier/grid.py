"""The certifier's grid on one link: constant-rate, left-packed, FIFO release (Lean `NetTiming.Protocol.Grid`).

A grid is one link's record for one window: `buckets` rows of `allocation` slots. A slot holds either a real frame
(`bytes` of the fixed frame size) or padding (`None`). Every bucket emits exactly `allocation` frames on the wire: the
real frames first, in queue order (left-packed), then certifier padding. The **occupancy** is the number of real frames in
each bucket. Once the Program fixes the frames' content and their order, occupancy is the only thing the sender still
controls: it is the timing channel.
"""

from __future__ import annotations

import enum
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

Row = tuple[bytes | None, ...]
Grid = tuple[Row, ...]


class Reason(enum.Enum):
    """Why a sampling block's coverage was not established (W5)."""

    MONITOR_UNAVAILABLE = "monitor-unavailable"
    OVERFLOW = "overflow"
    MALFORMED = "malformed"
    DROPPED = "dropped"
    DELIVERY_REFUSED = "delivery-refused"
    COVERAGE_NEVER_ESTABLISHED = "coverage-never-established"


@dataclass(frozen=True, slots=True)
class Status:
    """One untimed coverage status per sampling block: complete when `reason` is None, otherwise not established. It carries
    no bucket index, time or count, so it never shows when within the window something happened."""

    reason: Reason | None = None

    @property
    def complete(self) -> bool:
        return self.reason is None


COMPLETE = Status()


@dataclass(frozen=True, slots=True)
class GridParams:
    """W1: fixed by the protocol before the window, derived from a security parameter. The certifier only executes them."""

    buckets: int  # T, buckets per window
    allocation: int  # r, frames per bucket: the constant rate and the cap
    frame_bytes: int  # B, the frame size in bytes
    bucket_ms: int = 100  # τ_b
    queue_capacity: int = 1 << 16  # Q; exceeding it is an overflow fault

    def __post_init__(self) -> None:
        if min(self.buckets, self.allocation, self.frame_bytes, self.bucket_ms, self.queue_capacity) < 1:
            raise ValueError("grid parameters must be positive")


def row_count(row: Row) -> int:
    """The number of real frames in a bucket (Lean `rowCount`)."""
    return sum(slot is not None for slot in row)


def occupancy(grid: Grid) -> tuple[int, ...]:
    """Real frames per bucket, n_0 … n_{T−1}."""
    return tuple(row_count(row) for row in grid)


def row_left_packed(row: Row) -> bool:
    """Lean `RowLeftPacked`: the real slots form a prefix."""
    return all(slot is not None for slot in row[: row_count(row)])


def left_packed(grid: Grid) -> bool:
    """Lean `LeftPacked`: every row is left-packed (W4)."""
    return all(row_left_packed(row) for row in grid)


def fill(frames: Sequence[bytes], counts: Sequence[int], allocation: int) -> Grid:
    """Lean `fill`: the k-th frame of the queue-order sequence goes to the k-th real slot in index order t·r + s. Bucket t
    holds `counts[t]` frames in slots 0 … counts[t] − 1, with padding after."""
    rows: list[Row] = []
    k = 0
    for n in counts:
        if not 0 <= n <= allocation:
            raise ValueError(f"a bucket holds 0 … {allocation} real frames, not {n}")
        if k + n > len(frames):
            raise ValueError("fewer frames than the counts release")
        rows.append(tuple(frames[k : k + n]) + (None,) * (allocation - n))
        k += n
    return tuple(rows)


def released(allocation: int, enqueued: Sequence[int], queue: int = 0) -> list[int]:
    """Lean `released`: bucket t releases min(queue + enqueued[t], r) frames, and the rest wait, in order."""
    out = []
    for e in enqueued:
        if e < 0:
            raise ValueError("enqueue counts are non-negative")
        n = min(queue + e, allocation)
        out.append(n)
        queue += e - n
    return out


@dataclass(frozen=True, slots=True)
class Record:
    """W5: the network certificate of one link-window is exactly the grid and one status."""

    link: str
    window: int
    grid: Grid
    status: Status


class Certifier:
    """One link's active certifier (W1–W6): constant-rate, left-packed, FIFO and fail-closed.

    - **Enqueue.** The sender hands frames with `enqueue`. A frame handed during bucket t, before its cutoff, is eligible
      for bucket t. With `first`, it goes to the head of the queue, ahead of every frame queued: the ingress
      certifier's own salt frame (`active.Bucketer`). Release counts don't read the order, so `released` is unchanged.
    - **Tick.** `tick` releases bucket t: min(queue, r) frames in queue order, then padding, so the wire always carries r.
    - **Faults.** A fault is an overflow beyond Q, a frame of the wrong size, or one reported with `fault`. It makes the
      rest of the window padding and drops the queue, since nothing is forwarded uncertified, and it records the first
      reason.
    - **Close.** `close_window` publishes the record, with one untimed status, and starts the next window. Queued frames
      carry over.
    """

    def __init__(self, link: str, params: GridParams) -> None:
        self.link = link
        self.params = params
        self.window = 0
        self._queue: deque[bytes] = deque()
        self._rows: list[Row] = []
        self._fault: Reason | None = None

    @property
    def queued(self) -> int:
        return len(self._queue)

    @property
    def faulted(self) -> Reason | None:
        """The window's first fault so far, None while its coverage holds."""
        return self._fault

    def enqueue(self, frame: bytes, *, first: bool = False) -> None:
        if self._fault is not None:
            return
        if len(frame) != self.params.frame_bytes:
            self.fault(Reason.MALFORMED)
        elif len(self._queue) >= self.params.queue_capacity:
            self.fault(Reason.OVERFLOW)
        elif first:
            self._queue.appendleft(bytes(frame))
        else:
            self._queue.append(bytes(frame))

    def fault(self, reason: Reason) -> None:
        if self._fault is None:
            self._fault = reason
            self._queue.clear()

    def tick(self) -> Row:
        if len(self._rows) >= self.params.buckets:
            raise RuntimeError("the window is full; close it first")
        r = self.params.allocation
        if self._fault is not None:
            row: Row = (None,) * r
        else:
            n = min(len(self._queue), r)
            row = tuple(self._queue.popleft() for _ in range(n)) + (None,) * (r - n)
        self._rows.append(row)
        return row

    def close_window(self) -> Record:
        if len(self._rows) != self.params.buckets:
            raise RuntimeError(f"a window has {self.params.buckets} buckets, not {len(self._rows)}")
        record = Record(self.link, self.window, tuple(self._rows), Status(self._fault))
        self._rows, self._fault = [], None
        self.window += 1
        return record
