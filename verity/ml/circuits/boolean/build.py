"""Building a Boolean Definition's body by hand: the constant bits it reads, a collection's references as a list and a list
of references as a collection, and its AND gates as the census counts them.

A body's builder (`verity.ml.circuits.defs`) takes collections (`refs.Coll`); a hand-built body wires bits one by one, so
it reads a collection's references as a list (`refs`) and makes a list back into one (`coll`, contiguous runs as `Affine`
parts, as a traced body's are, so a body built either way has one descriptor).
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.census import BooleanWork, NoBooleanWork
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Type

__all__ = ["Consts", "ands", "coll", "refs"]


class Consts:
    """The constant bits a body reads, one node each."""

    def __init__(self, B) -> None:
        self.B, self.ref = B, {}

    def bits(self, value: int, w: int) -> list:
        """`value`'s low `w` bits, least significant first."""
        for b in sorted({value >> j & 1 for j in range(w)} - self.ref.keys()):
            self.ref[b] = self.B.call(BL.constant(b)).refs[0]
        return [self.ref[value >> j & 1] for j in range(w)]


def refs(c: Coll) -> list:
    """`c`'s references, one a leaf."""
    return [c.refs[k] for k in range(c.refs.count)]


def coll(t: Type, leaves: list) -> Coll:
    """The collection of type `t` whose leaves are `leaves`, references."""
    return Coll(t, TR._refs(list(leaves)))


def ands(fn: Any, work: BooleanWork | None = None) -> int:
    """A Boolean Definition's AND gates: the ANDs of its Boolean work as its census counts it (`census.BooleanWork`),
    every facet's. `work` shares one count across calls."""
    got = (BooleanWork() if work is None else work).of(fn)
    if isinstance(got, str):
        raise NoBooleanWork({fn.id: f"it reaches {got}, which is not a Boolean gate, a constant bit or an input"})
    return sum(a for _g, a in got.values())
