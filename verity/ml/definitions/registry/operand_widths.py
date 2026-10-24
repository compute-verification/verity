"""The leaf widths of a primitive's operands, as the strict lift (`lifted._strict_wrapper`) scans them: a list, or past the layout's
wide fan-in one (count, width) run per parameter of one width, iterated as the list is (`TableRead27x2_v1`'s table is 2^28 leaves,
whose list would be 2 GB)."""
from __future__ import annotations

from itertools import chain, repeat
from typing import Iterable, Iterator, Sequence

from verity.ml.circuits.layout import WIDE_FAN_IN
from verity.ml.circuits.types import Type


class WidthRuns:
    """Leaf widths as (count, width) runs."""

    __slots__ = ("runs",)

    def __init__(self, params: Sequence[tuple[str, Type]]) -> None:
        runs: list[tuple[int, int]] = []
        for _, t in params:
            w = t.homogeneous_width()
            runs += [(t.leaves, w)] if w is not None else [(1, x) for x in t.leaf_widths()]
        self.runs = tuple(runs)

    def __iter__(self) -> Iterator[int]:
        return chain.from_iterable(repeat(w, n) for n, w in self.runs)


def leaf_widths(params: Sequence[tuple[str, Type]]) -> Iterable[int]:
    """Every operand leaf's width, in order: a list up to `WIDE_FAN_IN` leaves, `WidthRuns` past it."""
    if sum(t.leaves for _, t in params) <= WIDE_FAN_IN:
        return [w for _, t in params for w in t.leaf_widths()]
    return WidthRuns(params)


__all__ = ["WidthRuns", "leaf_widths"]
