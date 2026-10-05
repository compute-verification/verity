"""The honest prover's online clock-sync choice: each window's sync declared at the window's first bucket, from what the
prover has seen by then.

`calibration.choose_syncs` chooses each window's sync in hindsight, from all of the window's frames. A live prover
declares window w's sync at its first bucket, w·T. By then it has read exactly the frames ready by w·T (read before the
boundary), each anchored in an earlier window, and it knows every earlier window's declaration. `Prover.declare` is
that prover: a rule (`Rule`) gets a summary of each recent window (`Past`) computed from those frames and declarations
only, and the frames it sees are the contiguous canonical prefix ready by w·T, so a frame ready later can't change it.

The rules are one family, `cover-K-M` (`rule`): the tightest value of Σ_sync that would have met each of the last K
windows' seen frames, with M buckets of δ to spare, at the least estimated hand time. `-fallback` declares the envelope
instead while one of the last K windows was late under its own declaration. `last` is `cover-1-0`: the previous
window's hindsight choice on what it showed. A window with no history declares the envelope.

The bound doesn't depend on the rule: the audit charges log2 #Σ_sync bits per window whatever chose the value
(`SecurityProofs.EncardAccSeqsLeSync` counts every advice sequence). The rule sets liveness, since a window its
declared sync doesn't meet is a violation (fail-closed), and the shaping latency. Since the prover reads the frames'
ready buckets, a sender that times its writes steers the declarations: those charged bits become ones it can use.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from verity_network_warden.schedule import ClockSync, LogicalFrame, SyncSet


@dataclass(frozen=True, slots=True)
class Past:
    """An earlier window as the prover sees it at a declaration.

    `frontier[i]`, for the i-th ρ of Σ_sync in ascending order, is the least δ of Σ_sync whose value would have met the
    window's seen frames on every link, with every other window at its declared value, and their total hand time then;
    or None if no δ does. A larger ρ needs at least the same δ, so a None is followed only by Nones."""

    window: int
    sync: ClockSync
    frames: int
    late: bool
    frontier: tuple[tuple[int, int] | None, ...]


Choose = Callable[[Sequence[Past], SyncSet, ClockSync], ClockSync]


@dataclass(frozen=True, slots=True)
class Rule:
    """A causal choice: `choose(past, sync_set, envelope)`, where `past` holds the windows among the last `depth` that
    have a seen frame, oldest first."""

    name: str
    depth: int
    choose: Choose


def cover(k: int, margin: int = 0, fallback: bool = False) -> Rule:
    """`cover-K-M[-fallback]`. Per ρ, δ is the largest frontier δ over the last k windows plus `margin`, rounded up to
    Σ_sync. The value taken is the least estimated total hand time over those windows, ties to the smaller δ, then the
    larger ρ: each window's frontier hand time plus its frames times the extra δ, an upper bound, since raising a
    window's δ by d raises each of its frames' hand times by at most d."""
    if k < 1 or margin < 0:
        raise ValueError("cover needs k ≥ 1 and margin ≥ 0")

    def choose(past: Sequence[Past], sync_set: SyncSet, envelope: ClockSync) -> ClockSync:
        recent = list(past)[-k:]
        if not recent or (fallback and any(p.late for p in recent)):
            return envelope
        deltas, rhos = sorted(sync_set.deltas), sorted(sync_set.rhos)
        best: tuple[tuple[int, int, int], ClockSync] | None = None
        for i, rho in enumerate(rhos):
            xs = [p.frontier[i] for p in recent]
            if None in xs:
                break
            j = bisect.bisect_left(deltas, max(d for d, _ in xs) + margin)
            if j == len(deltas):
                break
            d = deltas[j]
            key = (sum(c + p.frames * (d - dm) for p, (dm, c) in zip(recent, xs, strict=True)), d, -i)
            if best is None or key < best[0]:
                best = (key, ClockSync(d, rho))
        return envelope if best is None else best[1]

    return Rule(f"cover-{k}-{margin}" + ("-fallback" if fallback else ""), k, choose)


def rule(name: str) -> Rule:
    """A rule by name: `last`, or `cover-K-M` with an optional `-fallback`."""
    if name == "last":
        return Rule("last", 1, cover(1).choose)
    parts = name.split("-")
    if parts[0] == "cover" and len(parts) in (3, 4) and parts[1].isdigit() and parts[2].isdigit() and \
            (len(parts) == 3 or parts[3] == "fallback"):
        return cover(int(parts[1]), int(parts[2]), len(parts) == 4)
    raise ValueError(f"no sync rule {name!r}: last, or cover-K-M[-fallback]")


class _Link:
    """One egress link's frames as the prover has read them, by canonical position."""

    def __init__(self, window_buckets: int) -> None:
        self.T = window_buckets
        self.anchor: list[int] = []
        self.index: list[int] = []
        self.ready: list[int] = []
        self.early: dict[int, tuple[LogicalFrame, int]] = {}
        self.seen = 0
        self.positions: dict[int, list[int]] = {}
        self.h_at: dict[int, int] = {}
        self.h_pos = self.h_max = 0

    def arrive(self, position: int, frame: LogicalFrame, ready: int) -> None:
        if position < len(self.anchor) or position in self.early:
            return
        self.early[position] = (frame, ready)
        while len(self.anchor) in self.early:
            frame, ready = self.early.pop(len(self.anchor))
            self.anchor.append(frame.anchor)
            self.index.append(frame.index)
            self.ready.append(ready)

    def see(self, bucket: int) -> None:
        """Extend the seen prefix to the frames ready by `bucket`."""
        while self.seen < len(self.ready) and self.ready[self.seen] <= bucket:
            if self.anchor[self.seen] >= self.ready[self.seen]:
                raise ValueError("a frame is ready only after its anchor's bucket")
            self.positions.setdefault(self.anchor[self.seen] // self.T, []).append(self.seen)
            self.seen += 1

    def _eligible(self, k: int, s: ClockSync) -> int:
        return self.anchor[k] + s.delta + self.index[k] * s.rho.denominator // s.rho.numerator

    def h_before(self, u: int, syncs: Sequence[ClockSync]) -> int:
        """The logical hand time before window u's first frame: the most eligibility under `syncs` before it."""
        if u not in self.h_at:
            first = self.positions[u][0]
            if first < self.h_pos:
                self.h_at[u] = max((self._eligible(k, syncs[self.anchor[k] // self.T]) for k in range(first)), default=0)
            else:
                for k in range(self.h_pos, first):
                    self.h_max = max(self.h_max, self._eligible(k, syncs[self.anchor[k] // self.T]))
                self.h_pos = first
                self.h_at[u] = self.h_max
        return self.h_at[u]

    def cost(self, u: int, sync: ClockSync, syncs: Sequence[ClockSync]) -> int | None:
        """Window u's seen frames' total hand time with u at `sync` and every other window as declared, or None if one
        is late."""
        pos = self.positions[u]
        h, total, T = self.h_before(u, syncs), 0, self.T
        by = {w: (s.delta, s.rho.numerator, s.rho.denominator) for w, s in enumerate(syncs)}
        by[u] = (sync.delta, sync.rho.numerator, sync.rho.denominator)
        anchor, index, ready = self.anchor, self.index, self.ready
        for k in range(pos[0], pos[-1] + 1):
            a = anchor[k]
            if (x := by.get(a // T)) is None:  # past the run's last window: never handed in it
                continue
            d, num, den = x
            h = max(a + d + index[k] * den // num, h)
            if a // T == u:
                if ready[k] > h:
                    return None
                total += h
        return total


class Prover:
    """The online prover of one spatial unit: `arrive(link, position, frame, ready)` as egress frames are read,
    `declare()` at each window's first bucket. `syncs` holds the declarations so far.

    `prior` is what an earlier run of the unit showed (`history()` at its end): a window whose last `depth` windows
    hold fewer than `depth` with a seen frame takes the latest of `prior` to make up the rest."""

    def __init__(self, sync_set: SyncSet, window_buckets: int, rule: Rule, envelope: ClockSync, links: int = 1,
                 prior: Sequence[Past] = ()) -> None:
        self.sync_set, self.T, self.rule, self.envelope = sync_set, window_buckets, rule, envelope
        self.links = [_Link(window_buckets) for _ in range(links)]
        self.syncs: list[ClockSync] = []
        self.prior = list(prior)[-rule.depth:]
        self._past: dict[int, tuple[tuple[int, ...], Past]] = {}

    def arrive(self, link: int, position: int, frame: LogicalFrame, ready: int) -> None:
        self.links[link].arrive(position, frame, ready)

    def declare(self) -> ClockSync:
        """The next window's sync, from the frames ready by its first bucket and the earlier declarations."""
        w = len(self.syncs)
        for link in self.links:
            link.see(w * self.T)
        past = []
        for u in range(max(0, w - self.rule.depth), w):
            key = tuple(len(link.positions.get(u, ())) for link in self.links)
            if not any(key):
                continue
            if self._past.get(u, ((),))[0] != key:
                self._past[u] = (key, self._summarize(u, sum(key)))
            past.append(self._past[u][1])
        for u in [u for u in self._past if u < w - self.rule.depth]:
            del self._past[u]
        past = (self.prior + past)[-self.rule.depth:]
        sync = self.rule.choose(past, self.sync_set, self.envelope)
        self.syncs.append(sync)
        return sync

    def history(self) -> list[Past]:
        """At the end of the run, every frame read: the summaries of its last `depth` windows with a frame, with
        `prior` making up the rest, for the unit's next run."""
        for link in self.links:
            link.see(max(link.ready, default=0))
        w = len(self.syncs)
        own = [self._summarize(u, n) for u in range(max(0, w - self.rule.depth), w)
               if (n := sum(len(link.positions.get(u, ())) for link in self.links))]
        return (self.prior + own)[-self.rule.depth:]

    def _cost(self, u: int, sync: ClockSync) -> int | None:
        total = 0
        for link in self.links:
            if u in link.positions:
                if (c := link.cost(u, sync, self.syncs)) is None:
                    return None
                total += c
        return total

    def _summarize(self, u: int, frames: int) -> Past:
        deltas, rhos = sorted(self.sync_set.deltas), sorted(self.sync_set.rhos)
        frontier: list[tuple[int, int] | None] = []
        lo: int | None = 0
        for rho in rhos:
            if lo is None or (c := self._cost(u, ClockSync(deltas[-1], rho))) is None:
                lo = None
                frontier.append(None)
                continue
            hi, c_hi = len(deltas) - 1, c
            while lo < hi:
                mid = (lo + hi) // 2
                if (c := self._cost(u, ClockSync(deltas[mid], rho))) is None:
                    lo = mid + 1
                else:
                    hi, c_hi = mid, c
            frontier.append((deltas[lo], c_hi))
        return Past(u, self.syncs[u], frames, self._cost(u, self.syncs[u]) is None, tuple(frontier))


def declare(links: Sequence[tuple[Sequence[LogicalFrame], Sequence[int]]], sync_set: SyncSet, window_buckets: int,
            rule: Rule, windows: int, envelope: ClockSync, prior: Sequence[Past] = ()) -> tuple[list[ClockSync], list[Past]]:
    """Replay the online prover over one configuration: every frame arrives at once, with each link's frames in
    canonical order and their ready buckets, and window w's declaration sees only those ready by w·T. Returns the
    declarations and the run's `history()`."""
    prover = Prover(sync_set, window_buckets, rule, envelope, len(links), prior)
    for i, (frames, ready) in enumerate(links):
        for k, (f, r) in enumerate(zip(frames, ready, strict=True)):
            prover.arrive(i, k, f, r)
    return [prover.declare() for _ in range(windows)], prover.history()
