"""Opening drawn slots by replaying their requests, for the ``slots`` commit backend (`commit`), under which a run keeps no
step's tables.

After the run, a FIXTURE coin draws slots of the batch (`draw_slots`; PRF-170's plan draws from live coins). Each drawn slot
is replayed alone (`replay_slot`): one row through positions 0 .. the drawn step, on the run's resident weights, each step
under that step's salt key with the slot's own nonce, so its subtree is rebuilt and compared with the slot root the step
committed. A mismatch means the committed outputs are not the ones the slot's request computes. The replayed tables then give
the drawn units (`units.Replayer`), and `Openings.open` opens each unit's outputs against the step's root: the commit units
holding them (`slots.open_unit`'s construction from the replayed slot's rows) are checked by `slots.verify_unit` with their
rows and salts, and the reference's slot root (`slots.commit_slot`, in Python) must equal the replay's.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .commit import Committer
from .driver import Driver, Params, StepResult
from .shape import Shape
from .units import Unit


def draw_slots(batch: int, k: int, seed: bytes, step: int) -> list[int]:
    """``k`` distinct slots of ``batch`` for step ``step`` by a FIXTURE coin."""
    h = hashlib.sha512(b"e2e-h200/slot-draw-fixture\0" + seed + step.to_bytes(4, "big")).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(h[:16], "big")))
    return sorted(int(s) for s in rng.choice(batch, size=min(k, batch), replace=False))


@dataclass
class Opened:
    slot: int
    driver: Driver
    roots_equal: list[bool] = field(default_factory=list)

    @property
    def tokens(self) -> list[np.ndarray]:
        return [r.tokens_in for r in self.driver.results]

    def doc(self) -> dict:
        return {"slot": self.slot, "steps": len(self.roots_equal), "slot_roots_equal": all(self.roots_equal),
                "first_unequal_step": next((i for i, ok in enumerate(self.roots_equal) if not ok), None)}


def _row(r: StepResult, slot: int) -> int:
    rows = np.flatnonzero(r.slots == slot)
    if not len(rows):
        raise ValueError(f"step {r.step} served no row of slot {slot}")
    return int(rows[0])


def replay_slot(shape: Shape, params: Params, committer: Committer, results: list[StepResult], prompts: np.ndarray,
                slot: int, last: int, device: Any, attention: str | None = None) -> Opened:
    """Slot ``slot``'s request replayed alone through step ``last``, each step's slot root compared with the committed one."""
    d = Driver(shape, params, 1, committer, device, keep=True, attention=attention)
    op = Opened(slot, d)
    tok = int(prompts[slot, 0])
    for t in range(last + 1):
        r = results[t]
        b = _row(r, slot)
        res = d.step_slots(np.zeros(1, np.int64), r.positions[b:b + 1], np.array([tok]), salt_key=r.salt_key, nonce0=b)
        op.roots_equal.append(res.slot_roots[0] == r.slot_roots[b])
        tok = int(prompts[slot, t + 1]) if t + 1 < prompts.shape[1] else int(res.tokens_out[0])
    return op


class Openings:
    """Drawn units' openings against their steps' roots, from replayed slots."""

    def __init__(self, committer: Committer, results: list[StepResult]) -> None:
        self.c, self.results = committer, results
        self.S = committer.S
        self._slots: dict[tuple[int, int], tuple[Any, list[bytes]]] = {}
        self._trees: dict[int, Any] = {}

    def slot(self, opened: Opened, step: int) -> tuple[Any, list[bytes]]:
        key = (opened.slot, step)
        if key not in self._slots:
            r = self.results[step]
            rows = self.c.slot_rows(opened.driver.steps[step], 0)
            self._slots[key] = (self.S.commit_slot(self.c.context, self.c.layout, r.salt_key, _row(r, opened.slot), rows), rows)
        return self._slots[key]

    def reference_equal(self, opened: Opened, step: int) -> bool:
        """Whether the reference's slot root of the replayed rows is the replay's (the engine's)."""
        return self.slot(opened, step)[0].root == opened.driver.results[step].slot_roots[0]

    def open(self, opened: Opened, step: int, unit: Unit) -> dict | None:
        """``unit``'s outputs opened against step ``step``'s root, or None for a unit outside the step's tables."""
        if unit.family not in self.c.spans:
            return None
        S, c, r = self.S, self.c, self.results[step]
        b = _row(r, opened.slot)
        sc, rows = self.slot(opened, step)
        if step not in self._trees:
            self._trees[step] = S.step_tree(c.context, c.layout, step, r.slot_roots)
        table = next(t for g in c.groups for t in g if t.op == unit.family)
        lo, hi, scale = unit.span or (0, table.nbytes, table.quant)
        cu = c.units_of(unit.family, lo, hi, scale)
        ok = True
        for u in cu:
            base, n = c.layout.row_base[u], len(c.layout.units[u])
            opening = S.UnitOpening(b, u, sc.strings[base:base + n], sc.tree.open([u]), self._trees[step].open([b]))
            salts = S.row_salts(r.salt_key, b, base, n)
            rws = {i: (rows[base + i], salts[S.SALT_BYTES * i:S.SALT_BYTES * (i + 1)]) for i in range(n)}
            ok = ok and S.verify_unit(r.root, c.context, c.layout, step, len(r.slots), opening, rws)
        return {"commit_units": cu, "verified": bool(ok)}
