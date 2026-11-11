"""The worker's records of `native_collect`'s plans, kept only for the plans a step can still reuse.

The collector keys a plan by the step's geometry (`native_collect.plan_key_of`), so every decode step of a growing KV length
learns a plan of its own. A cache that kept every plan's metas for the life of the process grows by one list per committed
step, and every full collection of the serving process's heap walks all of it. A step reuses a plan learned in its own run
or in the run before it (the warm-up learns the plans the record run follows), so the cache keeps two generations: the
plans of the current run and those of the previous one. `end_run` retires the older generation, and a plan used again moves
to the current one. The C++ collector drops its plans by the same rule (`native_plans.h`), and a plan it learns after a drop
takes a version above every dropped one, so a key here never names that plan. A plan the collector holds and this cache does
not (a step the worker has not absorbed yet, or failed) is exported again by the serving thread and rebuilt with the same
layout.

The worker reads and fills the cache while the serving thread asks whether a plan is in it (`in`), so a plan is added to
the current generation before it leaves the previous one, and `in` looks at the previous generation first: a plan never
appears missing while it moves. `end_run` runs between runs: in the committer's reset when the worker has drained, or on the
worker at a detached window's end (`NativeHostCommitter.detach`). There the serving thread asks of a step it ends whether a
plan survives the window ends the worker reaches before that step (`native_collect.NativeCollectCommitter._plan_held`).
"""
from __future__ import annotations

import sys
from collections.abc import Hashable, Iterable, Iterator, Sequence
from itertools import chain
from typing import Any


class PlanMetas(Sequence):
    """One plan's metas as columns, one flat tuple of names, dtypes, byte counts, offsets, ranks and dimensions each. A tuple
    of strs and ints leaves the cyclic collector's sight after its first pass, so a cached plan is one tracked object whatever
    its tensor count. Every step that follows the plan has the same metas, and no step changes them: a tensor's offset is
    both its `dev_off` and its `stream_off`, and `host` and `dev_block` are `None`. `nbytes` is their byte total, `digest`
    their layout digest.

    Iterating or indexing builds the plan's `TensorMeta`s once and keeps them, so every reader sees the same objects
    (`opened.read_meta` checks a meta by identity). The worker's steps read `nbytes`, the digest and `len`, and `finalize`
    reads the columns, `summary()` and `scan()`, so neither builds them: a plan kept with its `TensorMeta`s costs every full
    collection one more object to walk per tensor."""

    __slots__ = ("names", "dtypes", "sizes", "offsets", "ranks", "dims", "chunk", "nbytes", "digest", "_metas")

    def __init__(self, names: Iterable[str], dtypes: Iterable[str], sizes: Iterable[int], offsets: Iterable[int],
                 shapes: Iterable[Iterable[int]], chunk: int) -> None:
        shapes = tuple(map(tuple, shapes))
        self.names, self.dtypes, self.sizes, self.offsets = tuple(names), tuple(dtypes), tuple(map(int, sizes)), tuple(map(int, offsets))
        self.ranks, self.dims = tuple(map(len, shapes)), tuple(map(int, chain.from_iterable(shapes)))
        self.chunk = chunk
        self.nbytes = sum(self.sizes)
        self.digest: bytes | None = None
        self._metas: list | None = None

    def columns(self) -> tuple:
        """The plan as plain tuples, what `of_columns` rebuilds it from in another process: never its built `TensorMeta`s."""
        return self.names, self.dtypes, self.sizes, self.offsets, self.ranks, self.dims, self.chunk, self.digest

    @classmethod
    def of_columns(cls, cols: tuple) -> PlanMetas:
        names, dtypes, sizes, offsets, ranks, dims, chunk, digest = cols
        p = cls.__new__(cls)
        p.names, p.dtypes, p.sizes, p.offsets, p.ranks, p.dims, p.chunk = names, dtypes, sizes, offsets, ranks, dims, chunk
        p.nbytes, p.digest, p._metas = sum(sizes), digest, None
        return p

    @classmethod
    def of_entries(cls, entries: Iterable[tuple], names: Sequence[str], chunk: int) -> PlanMetas:
        """The record of the collector's exported plan: `(name id, torch dtype, sizes, nbytes, offset, gap)` per tensor."""
        nids, dtypes, shapes, sizes, offsets, _gaps = tuple(zip(*entries)) or ((),) * 6
        dtype_names = {d: sys.intern(str(d).replace("torch.", "")) for d in set(dtypes)}
        return cls(map(names.__getitem__, nids), map(dtype_names.__getitem__, dtypes), sizes, offsets, shapes, chunk)

    def shapes(self) -> Iterator[tuple[int, ...]]:
        at = 0
        for r in self.ranks:
            yield self.dims[at:at + r]
            at += r

    def _build(self) -> list:
        from verity_vllm.commit.committer.native_host import TensorMeta

        out, first_leaf, chunk = [], 0, self.chunk
        for i, (name, dtype, shape, nbytes, off) in enumerate(zip(self.names, self.dtypes, self.shapes(), self.sizes, self.offsets)):
            n_leaves = (nbytes + chunk - 1) // chunk
            out.append(TensorMeta(i, name, dtype, shape, nbytes, chunk, first_leaf, n_leaves, dev_off=off, stream_off=off))
            first_leaf += n_leaves
        return out

    def scan(self) -> list:
        """The `TensorMeta`s for one pass: the kept ones if a reader built them, else new ones that die with the pass."""
        return self._metas if self._metas is not None else self._build()

    def summary(self) -> list[list]:
        """`[ordinal, name, dtype, shape, nbytes]` per tensor, as `finalize` sizes a layout."""
        return [[i, n, dt, list(sh), nb] for i, (n, dt, sh, nb) in enumerate(zip(self.names, self.dtypes, self.shapes(), self.sizes))]

    def _kept(self) -> list:
        if self._metas is None:
            self._metas = self._build()
        return self._metas

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._kept())

    def __getitem__(self, i):
        return self._kept()[i]


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
