"""The verifier's plan for an epoch: public, because the verifier's coins and draws are.

A `Plan` says which epoch is audited, which devices the challenger wipes and when, which stores it challenges, which
read windows it opens, how its stores change (`changes`: blocks appended, released and re-keyed under the verifier's
coins, with their `lifetime` in read windows and whether residency is `deferred`), which blocks are out for maintenance,
and how many occupancy profiles the developer registered, which it reads from the registration that commits them (so
the choice among them is advice worth log2 of that, once).
"""
from __future__ import annotations

from dataclasses import dataclass

from .logical import Registration
from .memory import BLOCK_BITS, Append, Challenge, ReadWindow, Rekey, Release, StoreEntry, Wipe
from .time import Interval


@dataclass(frozen=True)
class Maintenance:
    """Zone `block` out of service during `during`: someone the auditor doesn't control has it in hand. Its certifiers'
    records aren't required then, everything its devices hold may leave with whoever services it, and anything may come
    back in with them, so only a wipe after it bounds what the block holds when it rejoins."""

    block: str
    during: Interval


@dataclass(frozen=True)
class Plan:
    epoch: Interval
    wipes: tuple[Wipe, ...] = ()
    challenges: tuple[Challenge, ...] = ()
    reads: tuple[ReadWindow, ...] = ()
    maintenance: tuple[Maintenance, ...] = ()
    #: the number of occupancy profiles the developer may choose from: the registration's (`profiles`), else one
    profiles: int = 1
    changes: tuple[Append | Release | Rekey, ...] = ()
    lifetime: int | None = None
    deferred: bool = False
    #: the tick the prover's registration is due by; the verifier publishes the blocks' salts only after it
    registration: int | None = None

    def __init__(self, epoch: Interval, wipes=(), challenges=(), reads=(), maintenance=(),
                 profiles: Registration | None = None, changes=(), lifetime: int | None = None, deferred: bool = False,
                 registration: int | None = None):
        if not (profiles is None or isinstance(profiles, Registration)):
            raise TypeError("a Plan reads its number of occupancy profiles from the registration that commits them: "
                            "register them, register(program, values, profiles=[...]), and pass that registration")
        for name, value in (("epoch", epoch), ("wipes", tuple(wipes)), ("challenges", tuple(challenges)),
                            ("reads", tuple(reads)), ("maintenance", tuple(maintenance)),
                            ("profiles", 1 if profiles is None else max(len(profiles.profiles), 1)),
                            ("changes", tuple(changes)), ("lifetime", lifetime), ("deferred", deferred),
                            ("registration", epoch.end if registration is None else registration)):
            object.__setattr__(self, name, value)
        if deferred and lifetime is not None and lifetime < 2:
            raise ValueError("deferred residency needs a lifetime of at least 2 read windows")
        if lifetime is not None:
            for store, i, span in self.resident():
                windows = sum(span.start <= w.start < span.end for w in self.reads if w.store == store)
                if windows > lifetime:
                    raise ValueError(f"block {i} of store {store!r} stays {windows} read windows, past its lifetime "
                                     f"of {lifetime}: release or re-key it")

    def out(self, block: str, during: Interval) -> bool:
        """Whether `block` is out for maintenance at any tick of `during`."""
        return any(m.block == block and m.during.start < during.end and during.start < m.during.end
                   for m in self.maintenance)

    def resident(self) -> list[tuple[str, int, Interval]]:
        """Each appended block's residency: (store, index, the ticks it counts). A block released before it lands is
        withdrawn and has none."""
        start: dict[tuple[str, int], int] = {}
        end: dict[tuple[str, int], int] = {}
        for e in self.changes:
            if isinstance(e, Append):
                start.update({(e.store, e.first + k): e.resident_at for k in range(e.count)})
            elif isinstance(e, Release):
                end.update({(e.store, e.first + k): e.at for k in range(e.count)})
            else:
                end.update({(e.store, e.old_first + k): e.at for k in range(e.count)})
                start.update({(e.store, e.new_first + k): e.resident_at for k in range(e.count)})
        return [(s, i, Interval(t, end.get((s, i), self.epoch.end))) for (s, i), t in sorted(start.items())
                if t < end.get((s, i), self.epoch.end)]

    def bits(self, store: StoreEntry, tick: int) -> float:
        """The store's bits at `tick`: its registered bits and every appended block resident then."""
        return store.bits + BLOCK_BITS * sum(tick in span for s, _, span in self.resident() if s == store.name)

    def held(self, store: StoreEntry, during: Interval) -> float:
        """The store's bits on average over `during` (they change mid-window as blocks land and leave)."""
        if not during.length:
            return self.bits(store, during.start)
        overlap = sum(max(0, min(span.end, during.end) - max(span.start, during.start))
                      for s, _, span in self.resident() if s == store.name)
        return store.bits + BLOCK_BITS * overlap / during.length
