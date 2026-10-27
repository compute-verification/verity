"""Opening drawn slots by replaying their requests, for the ``slots`` commit backend (`commit`), under which a run keeps no
step's tables.

After the run, a FIXTURE coin draws slots of the batch (`draw_slots`; the sampling plan draws from live coins). Each drawn slot
is replayed alone (`replay_slot`): one row through positions 0 .. the drawn step, on the run's resident weights, each step
under that step's salt key with the slot's own nonce, so its subtree is rebuilt and compared with the slot root the step
committed. A mismatch means the committed outputs are not the ones the slot's request computes. The replayed tables then give
the drawn units (`units.Replayer`), and `Openings.open` opens each unit's outputs against the step's root: the one commit unit
holding them (`commit.Committer.unit_of`, `slots.open_unit`'s construction from the replayed slot's rows) is checked by
`slots.verify_unit` with its rows and salts, its rows must be the unit's committed outputs, and the reference's slot root
(`slots.commit_slot`, in Python) must equal the replay's.
"""
from __future__ import annotations

import hashlib
import itertools
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


_FORKED: tuple | None = None


def _commit_range(span: tuple[int, int]) -> tuple[list[bytes], list[bytes]]:
    S, context, layout, salts, rows = _FORKED
    lo, hi = span
    first, last = layout.row_base[lo], layout.row_base[hi - 1] + len(layout.units[hi - 1])
    strings = S._strings(rows[first:last], salts[S.SALT_BYTES * first:S.SALT_BYTES * last])
    roots = [S._unit_tree(S.unit_domain(context, layout, u),
                          strings[layout.row_base[u] - first:layout.row_base[u] - first + len(layout.units[u])]).commitment.root
             for u in range(lo, hi)]
    return strings, roots


def commit_slot_parallel(S: Any, context: bytes, layout: Any, salt_key: bytes, slot: int, rows: list[bytes],
                         workers: int) -> Any:
    """`slots.commit_slot`'s strings and slot tree, its unit ranges hashed in forked workers that touch no device. The unit
    trees aren't kept: an opening needs only the strings and the slot tree."""
    global _FORKED
    import multiprocessing as mp
    flat = [r for u in layout.units for r in u]
    if len(rows) != len(flat) or any(len(r) != row.nbytes for r, row in zip(rows, flat, strict=True)):
        raise ValueError("a slot's rows are the layout's, in order")
    n = layout.unit_count
    cuts = [n * i // (4 * workers) for i in range(4 * workers + 1)]
    spans = [(a, b) for a, b in itertools.pairwise(cuts) if b > a]
    _FORKED = (S, context, layout, S.row_salts(salt_key, slot, 0, len(rows)), rows)
    try:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_commit_range, spans, chunksize=1)
    finally:
        _FORKED = None
    strings = tuple(s for p, _ in parts for s in p)
    roots = [r for _, p in parts for r in p]
    return S.SlotCommit(strings, (), S.RootTree(S.slot_domain(context, layout), roots))


class Openings:
    """Drawn units' openings against their steps' roots, from replayed slots."""

    def __init__(self, committer: Committer, results: list[StepResult], workers: int = 1) -> None:
        self.c, self.results, self.workers = committer, results, workers
        self.S = committer.S
        self._slots: dict[tuple[int, int], tuple[Any, list[bytes]]] = {}
        self._trees: dict[int, Any] = {}

    def slot(self, opened: Opened, step: int) -> tuple[Any, list[bytes]]:
        key = (opened.slot, step)
        if key not in self._slots:
            r = self.results[step]
            rows = self.c.slot_rows(opened.driver.steps[step], 0)
            args = (self.c.context, self.c.layout, r.salt_key, _row(r, opened.slot), rows)
            sc = (commit_slot_parallel(self.S, *args, self.workers) if self.workers > 1 else self.S.commit_slot(*args))
            self._slots[key] = (sc, rows)
        return self._slots[key]

    def reference_equal(self, opened: Opened, step: int) -> bool:
        """Whether the reference's slot root of the replayed rows is the replay's (the engine's)."""
        return self.slot(opened, step)[0].root == opened.driver.results[step].slot_roots[0]

    def open(self, opened: Opened, step: int, unit: Unit) -> dict | None:
        """``unit``'s outputs opened against step ``step``'s root: its one commit unit verifies, and its rows are the unit's
        committed outputs. None for a unit outside the step's layout (a subcircuit's)."""
        if unit.family not in self.c.bases:
            return None
        S, c, r = self.S, self.c, self.results[step]
        b = _row(r, opened.slot)
        sc, rows = self.slot(opened, step)
        if step not in self._trees:
            self._trees[step] = S.step_tree(c.context, c.layout, step, r.slot_roots)
        u = c.unit_of(unit.family, unit.index)
        base, n = c.layout.row_base[u], len(c.layout.units[u])
        opening = S.UnitOpening(b, u, sc.strings[base:base + n], sc.tree.open([u]), self._trees[step].open([b]))
        salts = S.row_salts(r.salt_key, b, base, n)
        rws = {i: (rows[base + i], salts[S.SALT_BYTES * i:S.SALT_BYTES * (i + 1)]) for i in range(n)}
        ok = S.verify_unit(r.root, c.context, c.layout, step, len(r.slots), opening, rws)
        mine = b"".join(rows[base:base + n]) == np.ascontiguousarray(unit.got).tobytes()
        return {"commit_unit": u, "rows": n, "verified": bool(ok and mine)}
