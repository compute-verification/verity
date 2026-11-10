"""The honest prover's online clock-sync choice: each window's sync declared at the window's first bucket, from what the
prover had seen `lag` buckets before it.

`calibration.choose_syncs` chooses each window's sync in hindsight, from all of the window's frames. A live prover
declares window w's sync at its first bucket, w·T, and chooses it from the frames ready by w·T − lag (read before that
bucket's boundary), each anchored in an earlier window, and every earlier window's declaration. `Prover.declare` is
that prover: a rule (`Rule`) gets a summary of each recent window (`Past`) computed from those frames and declarations
only, and the frames it sees are the contiguous canonical prefix ready by w·T − lag, so a frame ready later can't
change it. The lag is what a live prover needs to choose off its tick: from bucket w·T − lag on nothing can change the
choice, so it can be computed elsewhere (`snapshot`) and declared at w·T. One not ready by then is declared as the
envelope (`active.LaggedChoices`). Lag 0 is the choice at the tick itself, the replay's reference (`declare`).
A frame ready no later than its anchor's bucket, which only a tick read early makes, is `Refused`.

The rules (`rule`) build on `cover-K-M`: the tightest value of Σ_sync that would have met each of the last K
windows' seen frames, with M buckets of δ to spare, at the least estimated hand time. `-fallback` declares the envelope
instead while one of the last K windows was late under its own declaration, and `-rhoR` declares the chosen δ at a
ρ R steps slower. `last` is `cover-1-0`: the previous window's hindsight choice on what it showed.
`sticky-K-M-N[-rhoR]` (`sticky`) declares `cover-K-M[-rhoR]` only every N windows and otherwise keeps or escalates
its previous value. A window with no history declares the envelope.

The bound doesn't depend on the rule, nor on the lag, nor on whether a choice was ready: the audit charges
log2 #Σ_sync bits per window whatever chose the value, and the envelope is in Σ_sync
(`SecurityProofs.EncardAccSeqsLeSync` counts every advice sequence). The rule sets liveness, since a window its
declared sync doesn't meet is a violation (fail-closed), and the shaping latency. Since the prover reads the frames'
ready buckets, a sender that times its writes steers the declarations: those charged bits become ones it can use.
`Rule.choices` counts the values it could steer a window's declaration among.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet


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


Choose = Callable[[Sequence[Past], SyncSet, ClockSync, Sequence[ClockSync]], ClockSync]
Choices = Callable[[Sequence[Past], SyncSet, ClockSync, Sequence[ClockSync]], int]


class Refused(ValueError):
    """A declaration refused: a frame read by the window's first bucket is ready no later than its anchor's bucket,
    which a device whose ticks are on time never reads (an ingress tick read before its due time can release a
    request whose response is read before the egress bucket it is anchored in)."""


@dataclass(frozen=True, slots=True)
class Rule:
    """A causal choice: `choose(past, sync_set, envelope, declared)`, where `past` holds the windows among the last
    `depth` that have a seen frame, oldest first, and `declared` every earlier declaration of the run.

    `choices(...)`, on the same arguments, bounds the values `choose` could declare in this window over every set of
    frames the sender could have shown it: a sender that times its writes picks among them, so log2 of it bounds the
    advice bits it can use in the window. The bound charges log2 #Σ_sync whatever the rule."""

    name: str
    depth: int
    choose: Choose
    choices: Choices


def _up(deltas: Sequence[int], d: int) -> int | None:
    j = bisect.bisect_left(deltas, d)
    return deltas[j] if j < len(deltas) else None


def cover(k: int, margin: int = 0, fallback: bool = False, rho_margin: int = 0) -> Rule:
    """`cover-K-M[-rhoR][-fallback]`. Per ρ, δ is the largest frontier δ over the last k windows plus `margin`, rounded
    up to Σ_sync. The value taken is the least estimated total hand time over those windows, ties to the smaller δ,
    then the larger ρ: each window's frontier hand time plus its frames times the extra δ, an upper bound, since
    raising a window's δ by d raises each of its frames' hand times by at most d. With `rho_margin` R, the declared ρ
    is R steps of Σ_sync slower than the chosen one (or the slowest), which only raises hand times."""
    if k < 1 or margin < 0 or rho_margin < 0:
        raise ValueError("cover needs k ≥ 1 and margins ≥ 0")
    c = _Cover(k, margin, fallback, rho_margin)
    name = f"cover-{k}-{margin}" + (f"-rho{rho_margin}" if rho_margin else "") + ("-fallback" if fallback else "")
    return Rule(name, k, c, c.choices)


@dataclass(frozen=True, slots=True)
class _Cover:
    """`cover`'s choice, a value rather than a closure so that a rule pickles into a worker process."""

    k: int
    margin: int
    fallback: bool
    rho_margin: int = 0

    def __call__(self, past: Sequence[Past], sync_set: SyncSet, envelope: ClockSync,
                 declared: Sequence[ClockSync] = ()) -> ClockSync:
        recent = list(past)[-self.k:]
        if not recent or (self.fallback and any(p.late for p in recent)):
            return envelope
        deltas, rhos = sorted(sync_set.deltas), sorted(sync_set.rhos)
        best: tuple[tuple[int, int, int], int, int] | None = None
        for i in range(len(rhos)):
            xs = [p.frontier[i] for p in recent]
            if None in xs:
                break
            d = _up(deltas, max(d for d, _ in xs) + self.margin)
            if d is None:
                break
            key = (sum(c + p.frames * (d - dm) for p, (dm, c) in zip(recent, xs, strict=True)), d, -i)
            if best is None or key < best[0]:
                best = (key, d, i)
        return envelope if best is None else ClockSync(best[1], rhos[max(0, best[2] - self.rho_margin)])

    def choices(self, past: Sequence[Past], sync_set: SyncSet, envelope: ClockSync,
                declared: Sequence[ClockSync] = ()) -> int:
        if not list(past)[-self.k:]:
            return 1
        deltas, rhos = sorted(sync_set.deltas), sorted(sync_set.rhos)
        ds = {d for f in deltas if (d := _up(deltas, f + self.margin)) is not None}
        values = {ClockSync(d, r) for d in ds for r in rhos[:max(1, len(rhos) - self.rho_margin)]}
        return len(values | {envelope})


def sticky(k: int, margin: int, every: int, rho_margin: int = 0) -> Rule:
    """`sticky-K-M-N[-rhoR]`: `cover-K-M[-rhoR]` declares only in the run's first two windows and every N windows
    after (w ≡ 1 mod N), each a free window. In the others the previous declaration p stays while it has M buckets
    of δ to spare over each of the last K windows' frontier at its ρ; otherwise the rule escalates to a value fixed
    by p alone: δ at least p's plus max(M, 1), or past the largest δ, the largest δ at the next slower ρ. So outside
    a free window the sender chooses only between p and its escalation, one bit."""
    if every < 1:
        raise ValueError("sticky needs N ≥ 1")
    if k < 1 or margin < 0 or rho_margin < 0:
        raise ValueError("cover needs k ≥ 1 and margins ≥ 0")
    s = _Sticky(_Cover(k, margin, False, rho_margin), every)
    return Rule(f"sticky-{k}-{margin}-{every}" + (f"-rho{rho_margin}" if rho_margin else ""), k, s, s.choices)


@dataclass(frozen=True, slots=True)
class _Sticky:
    """`sticky`'s choice, a value for the same reason as `_Cover`."""

    base: _Cover
    every: int

    def _escalate(self, p: ClockSync, sync_set: SyncSet) -> ClockSync:
        deltas, rhos = sorted(sync_set.deltas), sorted(sync_set.rhos)
        if (d := _up(deltas, p.delta + max(self.base.margin, 1))) is not None:
            return ClockSync(d, p.rho)
        i = bisect.bisect_left(rhos, p.rho)
        return ClockSync(deltas[-1], rhos[i - 1]) if i > 0 else p

    def _free(self, declared: Sequence[ClockSync]) -> bool:
        return not declared or (len(declared) - 1) % self.every == 0

    def _keeps(self, p: ClockSync, past: Sequence[Past], sync_set: SyncSet) -> bool:
        rhos = sorted(sync_set.rhos)
        if p.rho not in rhos:
            return False
        i = rhos.index(p.rho)
        return all((x := q.frontier[i]) is not None and x[0] + self.base.margin <= p.delta
                   for q in list(past)[-self.base.k:])

    def __call__(self, past: Sequence[Past], sync_set: SyncSet, envelope: ClockSync,
                 declared: Sequence[ClockSync] = ()) -> ClockSync:
        if self._free(declared):
            return self.base(past, sync_set, envelope, declared)
        p = declared[-1]
        return p if self._keeps(p, past, sync_set) else self._escalate(p, sync_set)

    def choices(self, past: Sequence[Past], sync_set: SyncSet, envelope: ClockSync,
                declared: Sequence[ClockSync] = ()) -> int:
        if self._free(declared):
            return self.base.choices(past, sync_set, envelope, declared)
        return len({declared[-1], self._escalate(declared[-1], sync_set)})


def rule(name: str) -> Rule:
    """A rule by name: `last`, `cover-K-M` with an optional `-rhoR` and `-fallback`, or `sticky-K-M-N[-rhoR]`."""
    if name == "last":
        c = cover(1)
        return Rule("last", 1, c.choose, c.choices)
    parts = name.split("-")
    nums = parts[1:3]
    if len(parts) >= 3 and all(x.isdigit() for x in nums):
        k, m = int(parts[1]), int(parts[2])
        rest = parts[3:]
        n = None
        if parts[0] == "sticky" and rest and rest[0].isdigit():
            n, rest = int(rest[0]), rest[1:]
        rho = 0
        if rest and rest[0].startswith("rho") and rest[0][3:].isdigit():
            rho, rest = int(rest[0][3:]), rest[1:]
        if n is not None and not rest:
            return sticky(k, m, n, rho)
        if parts[0] == "cover" and rest in ([], ["fallback"]):
            return cover(k, m, bool(rest), rho)
    raise ValueError(f"no sync rule {name!r}: last, cover-K-M[-rhoR][-fallback] or sticky-K-M-N[-rhoR]")


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

    def see(self, bucket: int) -> bool:
        """Extend the seen prefix to the frames ready by `bucket`; whether one of them is ready no later than its
        anchor's bucket."""
        early = False
        while self.seen < len(self.ready) and self.ready[self.seen] <= bucket:
            early |= self.anchor[self.seen] >= self.ready[self.seen]
            self.positions.setdefault(self.anchor[self.seen] // self.T, []).append(self.seen)
            self.seen += 1
        return early

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
    `declare()` at each window's first bucket. `syncs` holds the declarations so far, and `choices` each one's
    `Rule.choices`, or None for a choice made elsewhere, which isn't counted.

    Window w's choice (`choose`) sees the frames ready by w·T − `lag`, 0 ≤ lag < T, and the declarations of the windows
    before it. `snapshot()` is what it reads, as a prover of its own, so the choice can be made elsewhere once bucket
    w·T − lag has passed, and `declare(sync)` declares a choice made so.

    `prior` is what an earlier run of the unit showed (`history()` at its end): a window whose last `depth` windows
    hold fewer than `depth` with a seen frame takes the latest of `prior` to make up the rest."""

    def __init__(self, sync_set: SyncSet, window_buckets: int, rule: Rule, envelope: ClockSync, links: int = 1,
                 prior: Sequence[Past] = (), lag: int = 0) -> None:
        if not 0 <= lag < window_buckets:
            raise ValueError("the prover's lag is 0 … T − 1 buckets")
        self.sync_set, self.T, self.rule, self.envelope, self.lag = sync_set, window_buckets, rule, envelope, lag
        self.links = [_Link(window_buckets) for _ in range(links)]
        self.syncs: list[ClockSync] = []
        self.choices: list[int | None] = []
        self.prior = list(prior)[-rule.depth:]
        self._past: dict[int, tuple[tuple[int, ...], Past]] = {}

    def arrive(self, link: int, position: int, frame: LogicalFrame, ready: int) -> None:
        self.links[link].arrive(position, frame, ready)

    def _see(self, bucket: int) -> None:
        if any([link.see(bucket) for link in self.links]):
            raise Refused("a frame is ready only after its anchor's bucket")

    def declare(self, sync: ClockSync | None = None) -> ClockSync:
        """Declare the next window's sync: `sync`, a choice made elsewhere (`snapshot().choose()`, or the envelope
        when that wasn't ready), or else `choose()`. Raises `Refused` as `choose` does."""
        if sync is None:
            past, declared = self._view(), tuple(self.syncs)
            sync = self.rule.choose(past, self.sync_set, self.envelope, declared)
            self.choices.append(self.rule.choices(past, self.sync_set, self.envelope, declared))
        else:
            self.choices.append(None)
        self.syncs.append(sync)
        return sync

    def snapshot(self) -> Prover:
        """What the next window's choice reads, as a prover of its own: each link's frames ready by w·T − lag (the
        contiguous canonical prefix), the declarations so far and `prior`. Its `choose()` is this one's; once bucket
        w·T − lag has passed, every frame it reads has arrived and no later one changes it. Raises `Refused` as
        `choose` does, and a second call then returns the snapshot."""
        w = len(self.syncs)
        self._see(w * self.T - self.lag)
        out = Prover(self.sync_set, self.T, self.rule, self.envelope, len(self.links), self.prior, self.lag)
        for mine, link in zip(out.links, self.links, strict=True):
            n = link.seen
            mine.anchor, mine.index, mine.ready = link.anchor[:n], link.index[:n], link.ready[:n]
            mine.see(w * self.T - self.lag)
        out.syncs = list(self.syncs)
        return out

    def choose(self) -> ClockSync:
        """The next window's sync under the rule, from the frames ready by its first bucket less `lag` and the earlier
        declarations. It declares nothing. Raises `Refused` if a frame it reads first then is ready no later than its
        anchor's bucket; it has read them all the same, so the next call is the window's."""
        return self.rule.choose(self._view(), self.sync_set, self.envelope, tuple(self.syncs))

    def _view(self) -> list[Past]:
        """What the next window's choice reads: the summaries of its last `depth` windows with a seen frame."""
        w = len(self.syncs)
        self._see(w * self.T - self.lag)
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
        return (self.prior + past)[-self.rule.depth:]

    def history(self) -> list[Past]:
        """At the end of the run, every frame read: the summaries of its last `depth` windows with a frame, with
        `prior` making up the rest, for the unit's next run; `Refused` as `declare` is."""
        self._see(max((max(link.ready, default=0) for link in self.links), default=0))
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
            rule: Rule, windows: int, envelope: ClockSync, prior: Sequence[Past] = (),
            lag: int = 0) -> tuple[list[ClockSync], list[Past], list[int | None]]:
    """Replay the online prover over one configuration: every frame arrives at once, with each link's frames in
    canonical order and their ready buckets, and window w's declaration sees only those ready by w·T − lag. Every
    choice is ready, so these are a live prover's declarations under the same lag whenever its were. Returns the
    declarations, the run's `history()` and each declaration's `Rule.choices`."""
    prover = Prover(sync_set, window_buckets, rule, envelope, len(links), prior, lag)
    for i, (frames, ready) in enumerate(links):
        for k, (f, r) in enumerate(zip(frames, ready, strict=True)):
            prover.arrive(i, k, f, r)
    syncs = [prover.declare() for _ in range(windows)]
    return syncs, prover.history(), prover.choices
