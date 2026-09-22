"""The four predicate kinds the checker is written in, and two interpreters for them.

Every gadget states its constraints through a context of five predicates:

    rng(name, x, lo, hi)          lo <= x < hi            (a range check; ``x`` may be a hint or an expression)
    eq(name, a, b)                a == b                  (a polynomial identity; ``a`` and ``b`` are ring expressions)
    boolean(name, b)              b in {0, 1}
    lookup(name, table, k)        the row of a fixed table at key ``k``; returns its columns
    lt_pow(name, x, ipow, bits)   0 <= x < 2^bits / ipow  (a range check against a *variable* power of two;
                                  ``ipow = 2^(bits - k)`` comes from a table row, so this states x < 2^k)

The reference realises ``lt_pow`` as the single range check ``x * ipow in [0, 2^bits)``, which is exact over the
integers.  It is kept a separate predicate because a field embedding cannot copy that realisation unless
``2^(2 bits) < p``: the product wraps otherwise (see the package docstring, requirement F2).

Arithmetic between predicates is ordinary ``+ - *`` on the values, so the same gadget code runs on Python ints
(:class:`IntCtx`, the reference: the first violated predicate raises :class:`Reject`) and on symbolic values
(:class:`Z3Ctx`, which collects the predicates into a solver for the adversarial search).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .params import Reject

__all__ = ["IntCtx", "Table", "Z3Ctx"]


@dataclass(frozen=True)
class Table:
    """A fixed function on an integer key range, ``lo <= key < hi``; ``fn(key)`` returns the row's columns."""

    name: str
    lo: int
    hi: int
    outs: tuple[str, ...]
    fn: Callable[[int], tuple[int, ...]]

    @property
    def rows(self) -> int:
        return self.hi - self.lo

    def row(self, key: int) -> tuple[int, ...]:
        if type(key) is not int or not self.lo <= key < self.hi:
            raise KeyError(key)
        return self.fn(key)


class IntCtx:
    """The reference interpreter: values are Python ints; the first violated predicate raises ``Reject``."""

    def rng(self, name: str, x: int, lo: int, hi: int) -> None:
        if type(x) is not int or not lo <= x < hi:
            raise Reject(f"{name}: {x!r} not in [{lo}, {hi})")

    def eq(self, name: str, a: int, b: int) -> None:
        if a != b:
            raise Reject(f"{name}: {a!r} != {b!r}")

    def boolean(self, name: str, b: int) -> None:
        if b not in (0, 1) or type(b) is not int:
            raise Reject(f"{name}: {b!r} is not a bit")

    def lookup(self, name: str, table: Table, key: int) -> tuple[int, ...]:
        try:
            return table.row(key)
        except KeyError:
            raise Reject(f"{name}: key {key!r} is off table {table.name} [{table.lo}, {table.hi})") from None

    def lt_pow(self, name: str, x: int, ipow: int, bits: int) -> None:
        self.rng(name, x * ipow, 0, 2**bits)


class Z3Ctx:
    """Collects the predicates as z3 integer constraints.  A lookup becomes fresh columns tied to the key by a
    disjunction over the rows.  Python ints passed where an expression is expected are checked eagerly."""

    def __init__(self) -> None:
        import z3

        self.z3 = z3
        self.solver = z3.Solver()
        self.count = 0

    def fresh(self, name: str) -> object:
        self.count += 1
        return self.z3.Int(f"{name}#{self.count}")

    def _add(self, cond: object) -> None:
        if isinstance(cond, bool):
            cond = self.z3.BoolVal(cond)
        self.solver.add(cond)

    def rng(self, name: str, x: object, lo: int, hi: int) -> None:
        if isinstance(x, int):
            self._add(lo <= x < hi)
            return
        self._add(self.z3.And(x >= lo, x < hi))

    def eq(self, name: str, a: object, b: object) -> None:
        self._add(a == b)

    def boolean(self, name: str, b: object) -> None:
        if isinstance(b, int):
            self._add(b in (0, 1))
            return
        self._add(self.z3.Or(b == 0, b == 1))

    def lookup(self, name: str, table: Table, key: object) -> tuple[object, ...]:
        z3 = self.z3
        if isinstance(key, int):
            if not table.lo <= key < table.hi:
                self._add(False)
                return tuple(0 for _ in table.outs)
            return table.row(key)
        outs = tuple(self.fresh(f"{name}.{o}") for o in table.outs)
        rows = [
            z3.And(key == k, *[o == v for o, v in zip(outs, table.fn(k))]) for k in range(table.lo, table.hi)
        ]
        self._add(z3.Or(rows))
        return outs

    def lt_pow(self, name: str, x: object, ipow: object, bits: int) -> None:
        self.rng(name, x * ipow, 0, 2**bits)
