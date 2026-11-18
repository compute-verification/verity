"""PoUB's refresh schedule (`audit.Params.refresh`): the session's append slots, each slot's window and kind, and so each
re-key's origin, from public parameters alone. The one Python definition, `Params.slotAt` in `lean/Poub/Session.lean`
line for line; the verifier (`audit.AuditRun`) and the prover read it here."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

REKEY_MAX = 512
"""The most blocks a fresh append holds under refresh, so its tree, a re-key's origin path, is at most 9 levels and an
opening of its re-key stays one proof block (`Poub.REKEY_MAX`)."""
FRESH_SLOTS = 4
"""F, the fresh slots a window has, for a session with refresh that names none (`audit.Params.fresh_slots`)."""

Origin = tuple
"""A re-key's origin as `audit.AuditRun.append` takes it: `("registered", lo)`, the registered class run from block
lo, or `("append", k)`, the fresh append in slot k, whole (`Poub.Origin`)."""

Slot = tuple
"""What a slot holds (`Session.Slot`): `("fresh",)`, `("registered", lo, count)` (the re-key of registered run
[lo, lo + count)) or `("rekey", k)` (the re-key of the fresh append in slot k)."""


@dataclass(frozen=True, slots=True)
class Schedule:
    """The slots of a session with refresh over `blocks` registered blocks, `windows` windows, the period `lifetime`
    = L windows and `fresh_slots` = F fresh slots a window, laid out window by window and never shifted. Window w's
    slots, in order: the registered re-key slot of class w mod L (w ≥ 1, the class nonempty), the re-key slot of each
    fresh slot of windows w − nL for n = ⌊w/L⌋ … 1 (oldest first, each window's in order), then its F fresh slots."""

    blocks: int
    lifetime: int
    windows: int
    fresh_slots: int
    _starts: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("blocks", "lifetime", "windows"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"a schedule's {name} is a positive integer, not {getattr(self, name)!r}")
        if type(self.fresh_slots) is not int or self.fresh_slots < 0:
            raise ValueError(f"a schedule's fresh slots are a nonnegative integer, not {self.fresh_slots!r}")
        starts = [0]
        for w in range(self.windows):
            starts.append(starts[-1] + self.window_size(w))
        object.__setattr__(self, "_starts", tuple(starts))

    def class_run(self, c: int) -> tuple[int, int]:
        """Class c's registered run [lo, hi): the blocks j with j·L // N = c (`Params.classRun`)."""
        n, life = self.blocks, self.lifetime
        return (c * n + life - 1) // life, ((c + 1) * n + life - 1) // life

    def has_registered(self, w: int) -> bool:
        lo, hi = self.class_run(w % self.lifetime)
        return w >= 1 and lo < hi

    def window_size(self, w: int) -> int:
        return int(self.has_registered(w)) + self.fresh_slots * (w // self.lifetime) + self.fresh_slots

    def window_start(self, w: int) -> int:
        """Window w's first slot, for 0 ≤ w ≤ W."""
        return self._starts[w]

    def slots(self, w: int) -> range:
        """Window w's slot indices."""
        return range(self._starts[w], self._starts[w + 1])

    def fresh_slot(self, b: int, i: int) -> int:
        """The index of window b's i-th fresh slot."""
        return self._starts[b + 1] - self.fresh_slots + i

    def slot_in(self, w: int, m: int) -> Slot:
        """The kind of window w's m-th slot (`Params.slotIn`)."""
        r = int(self.has_registered(w))
        if m < r:
            lo, hi = self.class_run(w % self.lifetime)
            return ("registered", lo, hi - lo)
        if m < r + self.fresh_slots * (w // self.lifetime):
            q = m - r
            n = w // self.lifetime - q // self.fresh_slots
            return ("rekey", self.fresh_slot(w - n * self.lifetime, q % self.fresh_slots))
        return ("fresh",)

    def slot_at(self, k: int) -> tuple[int, Slot] | None:
        """Slot k's window and kind, None past the last window's slots (`Params.slotAt`)."""
        if type(k) is not int or not 0 <= k < self.total:
            return None
        w = bisect.bisect_right(self._starts, k) - 1
        return w, self.slot_in(w, k - self._starts[w])

    def window_of(self, k: int) -> int | None:
        """The window slot k joins, used or empty, None past the schedule (`State.slotWindow`)."""
        at = self.slot_at(k)
        return None if at is None else at[0]

    @property
    def total(self) -> int:
        """T, the session's slots, used or empty."""
        return self._starts[-1]


__all__ = ["FRESH_SLOTS", "REKEY_MAX", "Origin", "Schedule", "Slot"]
