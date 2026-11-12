"""A CNF builder over named variables, with Tseitin gates and k-bit words, solved by pysat.

    cnf = Cnf()
    x, y = cnf.word("x", 4), cnf.word("y", 4)
    cnf.parity([cnf.and_(a, b) for a, b in zip(x, y)], odd=True)     # <x, y> = 1 over F_2
    cnf.models(over=x + y)                               # every projection onto those variables, as {variable: bool}

A literal is a nonzero int (pysat's DIMACS form; -v is not v). Words are lists of literals, least significant bit first.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from pysat.solvers import Solver

SOLVER = "cadical153"


class Cnf:
    def __init__(self):
        self.clauses: list[list[int]] = []
        self.names: dict[int, str] = {}
        self.ids: dict[str, int] = {}
        self.top = 0
        self._true = None

    def var(self, name: str | None = None) -> int:
        """A fresh variable (named ones are looked up: one name, one variable)."""
        if name is not None and name in self.ids:
            return self.ids[name]
        self.top += 1
        if name is not None:
            self.ids[name] = self.top
            self.names[self.top] = name
        return self.top

    def word(self, name: str, k: int) -> list[int]:
        return [self.var(f"{name}[{i}]") for i in range(k)]

    def add(self, clause: Iterable[int]) -> None:
        clause = [int(l) for l in clause]
        if any(l == 0 or abs(l) > self.top for l in clause):
            raise ValueError(f"literal out of range in {clause}")
        self.clauses.append(clause)

    @property
    def true(self) -> int:
        if self._true is None:
            self._true = self.var()
            self.add([self._true])
        return self._true

    def and_(self, a: int, b: int) -> int:
        z = self.var()
        self.add([-z, a])
        self.add([-z, b])
        self.add([z, -a, -b])
        return z

    def or_(self, a: int, b: int) -> int:
        return -self.and_(-a, -b)

    def xor(self, a: int, b: int) -> int:
        z = self.var()
        self.add([-z, a, b])
        self.add([-z, -a, -b])
        self.add([z, -a, b])
        self.add([z, a, -b])
        return z

    def xor_all(self, lits: Sequence[int]) -> int:
        """A literal equal to the parity of lits (false for none)."""
        if not lits:
            return -self.true
        z = lits[0]
        for l in lits[1:]:
            z = self.xor(z, l)
        return z

    def parity(self, lits: Sequence[int], odd: bool) -> None:
        z = self.xor_all(lits)
        self.add([z if odd else -z])

    def solve(self, assumptions: Sequence[int] = ()) -> dict[int, bool] | None:
        """A model ({variable: value}) or None when unsatisfiable."""
        with Solver(name=SOLVER, bootstrap_with=self.clauses) as s:
            if not s.solve(assumptions=list(assumptions)):
                return None
            return {abs(l): l > 0 for l in s.get_model()}

    def models(self, over: Sequence[int], limit: int | None = None) -> list[dict[int, bool]]:
        """Every distinct assignment of the variables `over` that extends to a model (up to `limit`), each blocked once
        found, in the solver's order."""
        over = [abs(v) for v in over]
        found = []
        with Solver(name=SOLVER, bootstrap_with=self.clauses) as s:
            while limit is None or len(found) < limit:
                if not s.solve():
                    break
                m = {abs(l): l > 0 for l in s.get_model()}
                proj = {v: m.get(v, False) for v in over}
                found.append(proj)
                s.add_clause([-v if proj[v] else v for v in over])
        return found

    def value(self, model: dict[int, bool], word: Sequence[int]) -> int:
        """A word's value in a model."""
        return sum((model[abs(l)] == (l > 0)) << i for i, l in enumerate(word))
