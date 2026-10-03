"""The worker's `TensorMeta` lists of `native_collect`'s plans, kept only for the plans a step can still reuse.

The collector keys a plan by the step's geometry (`native_collect.plan_key_of`), so every decode step of a growing KV length
learns a plan of its own. A cache that kept every plan's metas for the life of the process grows by one list per committed
step, and every full collection of the serving process's heap walks all of it. A step reuses a plan learned in its own run
or in the run before it (the warm-up learns the plans the record run follows), so the cache keeps two generations: the
plans of the current run and those of the previous one. `end_run` retires the older generation, and a plan used again moves
to the current one. The C++ collector keeps every plan, so a plan that falls out of this cache is exported again by the
serving thread on its next use and rebuilt with the same layout.

The worker reads and fills the cache while the serving thread asks whether a plan is in it (`in`), so a plan is added to
the current generation before it leaves the previous one, and `in` looks at the previous generation first: a plan never
appears missing while it moves. `end_run` runs between runs, when the worker has drained.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any


class PlanMetas(list):
    """One plan's metas, with their byte total and (computed once, on first use) layout digest."""

    __slots__ = ("nbytes", "digest")

    def __init__(self, metas: Iterable[Any]) -> None:
        super().__init__(metas)
        self.nbytes = sum(m.nbytes for m in self)
        self.digest: bytes | None = None


class PlanCache:
    def __init__(self) -> None:
        self.now: dict[Hashable, PlanMetas] = {}
        self.prev: dict[Hashable, PlanMetas] = {}

    def get(self, key: Hashable) -> PlanMetas | None:
        metas = self.now.get(key)
        if metas is None:
            metas = self.prev.get(key)
            if metas is not None:
                self.now[key] = metas
                self.prev.pop(key, None)
        return metas

    def __setitem__(self, key: Hashable, metas: PlanMetas) -> None:
        self.now[key] = metas

    def __contains__(self, key: Hashable) -> bool:
        return key in self.prev or key in self.now

    def __len__(self) -> int:
        return len(self.now) + len(self.prev)

    def end_run(self) -> None:
        """Retire the previous run's plans. A run that used none (two resets in a row) retires nothing."""
        if self.now:
            self.prev, self.now = self.now, {}
