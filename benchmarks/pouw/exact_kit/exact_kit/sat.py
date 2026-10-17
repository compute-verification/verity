"""A CNF builder over named variables, with Tseitin gates, k-bit words and cardinality constraints, solved by pysat.

    cnf = Cnf()
    x, y = cnf.word("x", 4), cnf.word("y", 4)
    cnf.equal(cnf.add_words(x, y), cnf.const(4, 5))     # x + y = 5 mod 16
    cnf.models(over=x + y)                               # every projection onto those variables, as {name: bool}

A literal is a nonzero int (pysat's DIMACS form; -v is not v). Words are lists of literals, least significant bit first.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from pysat.card import CardEnc, EncType
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

    def const(self, k: int, value: int) -> list[int]:
        return [self.true if value >> i & 1 else -self.true for i in range(k)]

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

    def maj(self, a: int, b: int, c: int) -> int:
        return self.or_(self.or_(self.and_(a, b), self.and_(a, c)), self.and_(b, c))

    def add_words(self, x: Sequence[int], y: Sequence[int], carry_in: int | None = None) -> list[int]:
        """x + y mod 2^k (ripple carry)."""
        if len(x) != len(y):
            raise ValueError("words of different widths")
        c = -self.true if carry_in is None else carry_in
        out = []
        for a, b in zip(x, y):
            out.append(self.xor(self.xor(a, b), c))
            c = self.maj(a, b, c)
        return out

    def xor_words(self, x: Sequence[int], y: Sequence[int]) -> list[int]:
        return [self.xor(a, b) for a, b in zip(x, y, strict=True)]

    def mul_const(self, x: Sequence[int], a: int) -> list[int]:
        """a * x mod 2^k for a constant a (shift and add)."""
        k = len(x)
        acc = self.const(k, 0)
        for i in range(k):
            if a >> i & 1:
                acc = self.add_words(acc, self.const(i, 0) + list(x[:k - i]))
        return acc

    def equal(self, x: Sequence[int], y: Sequence[int]) -> None:
        for a, b in zip(x, y, strict=True):
            self.add([-a, b])
            self.add([a, -b])

    def differ(self, x: Sequence[int], y: Sequence[int]) -> None:
        self.add([self.xor(a, b) for a, b in zip(x, y, strict=True)])

    def at_most(self, lits: Sequence[int], k: int) -> None:
        self._card(CardEnc.atmost(lits=list(lits), bound=k, top_id=self.top, encoding=EncType.seqcounter))

    def at_least(self, lits: Sequence[int], k: int) -> None:
        self._card(CardEnc.atleast(lits=list(lits), bound=k, top_id=self.top, encoding=EncType.seqcounter))

    def exactly(self, lits: Sequence[int], k: int) -> None:
        self.at_most(lits, k)
        self.at_least(lits, k)

    def _card(self, enc) -> None:
        self.top = max(self.top, enc.nv)
        self.clauses.extend(enc.clauses)

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

    def count(self, over: Sequence[int]) -> int:
        return len(self.models(over))

    def value(self, model: dict[int, bool], word: Sequence[int]) -> int:
        """A word's value in a model."""
        return sum((model[abs(l)] == (l > 0)) << i for i, l in enumerate(word))
