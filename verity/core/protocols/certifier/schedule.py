"""The Program's side: the logical schedule and the clock-sync advice.

The Program gives every real frame an **anchor** and a **logical index**:
- the anchor is the recorded bucket, on the global bucket clock, of the request it answers;
- the logical index λ is its position within its session, in the Program's canonical gate order.

A window declares one clock-sync value, `ClockSync(delta, rho)`, from the protocol's set `SyncSet`. The rate ρ is a
positive fraction of frames per bucket, and may be below 1 for sessions slower than one frame per bucket. A frame is
eligible at `anchor + delta + ⌊λ / rho⌋`, using the sync of the window that holds its anchor. The sender's shaper hands frames to
the certifier in canonical order, each at `h(k) = max(eligible(k), h(k−1))`. The certifier's own rule (FIFO at the allocation
r, left-packed) then fixes every grid. So the Program's grids are a function of its frames, the recorded anchors and the
sync advice, and the timing channel of accepted runs is at most `log2 #Σ_sync` bits per window, summed from the run's
first window: a later window alone may carry more.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction

from verity.core.protocols.certifier.grid import Grid, GridParams, BlockGrid, Certifier, fill, released


@dataclass(frozen=True, slots=True, order=True)
class ClockSync:
    """One window's clock-sync advice: the offset δ (buckets after the anchor), and the rate ρ (frames per bucket
    within a session, a positive fraction)."""

    delta: int
    rho: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "rho", Fraction(self.rho))
        if self.delta < 0 or self.rho <= 0:
            raise ValueError("clock sync needs delta ≥ 0 and rho > 0")


@dataclass(frozen=True, slots=True)
class SyncSet:
    """Σ_sync: the values a window may declare. The protocol fixes it, from calibration."""

    deltas: tuple[int, ...]
    rhos: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if not self.deltas or not self.rhos:
            raise ValueError("a sync set is non-empty")
        object.__setattr__(self, "rhos", tuple(Fraction(r) for r in self.rhos))
        if min(self.deltas) < 0 or min(self.rhos) <= 0:
            raise ValueError("clock sync needs delta ≥ 0 and rho > 0")
        if len(set(self.deltas)) < len(self.deltas) or len(set(self.rhos)) < len(self.rhos):
            raise ValueError("a sync set lists each delta and each rho once")

    def __iter__(self) -> Iterator[ClockSync]:
        return (ClockSync(d, r) for d in self.deltas for r in self.rhos)

    def __contains__(self, sync: object) -> bool:
        return isinstance(sync, ClockSync) and sync.delta in self.deltas and sync.rho in self.rhos

    @property
    def size(self) -> int:
        return len(self.deltas) * len(self.rhos)

    @property
    def bits(self) -> float:
        """The timing advice per window: log2 #Σ_sync."""
        return math.log2(self.size)


@dataclass(frozen=True, slots=True)
class LogicalFrame:
    """A real frame as the Program fixes it: its content, its anchor and its logical index.

    The index is non-negative, as Lean's ℕ is: a negative λ can make a frame eligible before its anchor, in an earlier
    window, whose grid would then read the later window's sync."""

    payload: bytes
    anchor: int
    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("a logical index is non-negative")


def eligible(frame: LogicalFrame, sync: ClockSync) -> int:
    """The frame's logical release bucket: `anchor + δ + ⌊λ / ρ⌋`."""
    return frame.anchor + sync.delta + (frame.index * sync.rho.denominator) // sync.rho.numerator


def handing_buckets(frames: Sequence[LogicalFrame], syncs: Sequence[ClockSync], window_buckets: int) -> list[int]:
    """When the shaper hands each frame to the certifier: `h(k) = max(eligible(k), h(k−1))`, in canonical order.

    A frame anchored past the last window with advice is handed after the horizon, and so is every frame behind it
    (FIFO)."""
    horizon = len(syncs) * window_buckets
    out, last = [], 0
    for frame in frames:
        w = frame.anchor // window_buckets
        if w < 0:
            raise ValueError("anchors are non-negative")
        last = max(eligible(frame, syncs[w]), last) if w < len(syncs) else max(last, horizon, frame.anchor)
        out.append(last)
    return out


def expected_grids(frames: Sequence[LogicalFrame], syncs: Sequence[ClockSync], params: GridParams) -> list[Grid]:
    """The Program's grids for windows 0 … len(syncs) − 1: the certifier's rule applied to the logical handing times.

    Frames still queued, or handed after the horizon, appear in no grid. Window w's grid reads only syncs[0 … w]: a
    frame is eligible no earlier than its anchor, so a frame handed by the end of window w, and every frame before it
    in canonical order, is anchored in window w or earlier."""
    horizon = len(syncs) * params.buckets
    enqueued = [0] * horizon
    for h in handing_buckets(frames, syncs, params.buckets):
        if h < horizon:
            enqueued[h] += 1
    counts = released(params.allocation, enqueued)
    whole = fill([f.payload for f in frames], counts, params.allocation)
    return [whole[w * params.buckets : (w + 1) * params.buckets] for w in range(len(syncs))]


def run_shaper(
    frames: Sequence[LogicalFrame],
    ready: Sequence[int],
    syncs: Sequence[ClockSync],
    certifier: Certifier,
) -> list[BlockGrid]:
    """An honest sender's shaper driving a certifier over len(syncs) windows.

    Frame k is handed at its logical bucket h(k), or later if it wasn't ready by then. A late frame delays those behind
    it, so the grid deviates from the Program's and the window is a violation. Returns the network certificates."""
    params = certifier.params
    handed = handing_buckets(frames, syncs, params.buckets)
    actual, last = [], 0
    for h, r in zip(handed, ready, strict=True):
        last = max(h, r, last)
        actual.append(last)
    records, k = [], 0
    for t in range(len(syncs) * params.buckets):
        while k < len(frames) and actual[k] == t:
            certifier.enqueue(frames[k].payload)
            k += 1
        certifier.tick()
        if (t + 1) % params.buckets == 0:
            records.append(certifier.close_window())
    return records
