"""Exact tensor-core cost of a product of two integer matrices with entry ranges V_A, V_B, sliced into interval digits.

A digit of size d is an interval of d consecutive integers, centred, so its largest magnitude is M(d) = d // 2.
x - c = sum_a w_a x_a (mixed radix, w_a = prod_{a'<a} d_a') covers prod d_a consecutive values, so a slicing of an
operand of range V is a vector of digit sizes with prod >= V (counting: the digit tuple determines x, so prod >= V is
also necessary). Each pair (a, b) of digits is one MMA product on a format f that holds both digits exactly and whose
accumulation is exact for them: integer formats when both sizes are within cap; float formats when both are within cap
and k * M(d_a) * M(d_b) <= 2^acc (the k products and the FP32 result are exact). The pair costs f.cost tensor time per
MAC, in int8-MAC units. The product's cost per MAC of the sub-product is the sum over pairs.

lower(VA, VB) is a proved lower bound: the column relaxation (each column of the grid picks its own A-slicing) and the
row relaxation, each solved exactly by a DP over the breakpoints of a monotone step function. upper(VA, VB) is the cost
of an explicit common slicing (one A-vector, one B-vector) read off the relaxations' minimisers. exact() asserts they
agree, which makes the value the exact minimum over all interval slicings.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache


def M(d: int) -> int:
    return d // 2


@dataclass(frozen=True)
class Fmt:
    name: str
    cost: F
    cap: int
    k: int | None = None
    acc: int | None = None

    def fits(self, da: int, db: int) -> bool:
        if da > self.cap or db > self.cap:
            return False
        if self.acc is None:
            return True
        return self.k * M(da) * M(db) <= 2 ** self.acc

    def maxd(self, db: int) -> int:
        """The largest digit size this format pairs with a digit of size db (0: none)."""
        if db > self.cap:
            return 0
        if self.acc is None or M(db) == 0:
            return self.cap
        L = 2 ** self.acc // (self.k * M(db))
        return min(self.cap, 2 * L + 1)


def inst(name: str) -> tuple[Fmt, ...]:
    s8 = Fmt("s8", F(1), 256)
    if name == "h200-sheet":
        return (s8, Fmt("e4m3", F(1), 33, 32, 14), Fmt("e5m2", F(1), 17, 32, 14),
                Fmt("f16", F(2), 4097, 8, 24), Fmt("bf16", F(2), 513, 8, 24), Fmt("tf32", F(4), 4097, 4, 24),
                Fmt("b1", F(1, 8), 2))
    if name == "h200-measured":
        return (s8, Fmt("e4m3", F(1), 33, 32, 14), Fmt("e5m2", F(1), 17, 32, 14),
                Fmt("f16", F(2), 4097, 16, 24), Fmt("bf16", F(2), 513, 16, 24), Fmt("tf32", F(4), 4097, 8, 24),
                Fmt("b1", F(1, 8), 2))
    if name == "rtx6000-sheet":
        return (s8, Fmt("e4m3", F(1), 33, 32, 24), Fmt("e5m2", F(1), 17, 32, 24), Fmt("e2m1", F(1), 9, 32, 24),
                Fmt("nvf4", F(1, 2), 9, 64, 24),
                Fmt("f16", F(2), 4097, 8, 24), Fmt("bf16", F(2), 513, 8, 24), Fmt("tf32", F(4), 4097, 4, 24))
    if name == "rtx6000-measured":
        return (s8, Fmt("e4m3", F(1), 33, 32, 24), Fmt("e5m2", F(1), 17, 32, 24), Fmt("e2m1", F(1), 9, 32, 24),
                Fmt("nvf4", F(1, 2), 9, 64, 24),
                Fmt("f16", F(2), 4097, 16, 24), Fmt("f16k8", F(4), 4097, 8, 24), Fmt("bf16", F(2), 513, 16, 24),
                Fmt("tf32", F(4), 4097, 4, 24))
    if name in ("rtx6000-sheet-b1", "rtx6000-measured-b1"):
        return inst(name.removesuffix("-b1")) + (Fmt("b1", F(1, 8), 2),)
    if name == "int8-only":
        return (s8,)
    raise KeyError(name)


INF = F(10 ** 9)


class Costs:
    def __init__(self, fmts: tuple[Fmt, ...]):
        self.fmts = fmts
        self.dmax = max(f.cap for f in fmts)

    @lru_cache(maxsize=None)
    def c(self, da: int, db: int) -> F:
        best = INF
        for f in self.fmts:
            if f.cost < best and f.fits(da, db):
                best = f.cost
        return best

    @lru_cache(maxsize=None)
    def _breaks(self, db: int) -> tuple[tuple[int, F], ...]:
        """(d, c(d, db)) at the largest d of each cost level of the step function d -> c(d, db), Pareto-pruned."""
        pts = sorted(((d, self.c(d, db)) for d in {f.maxd(db) for f in self.fmts} - {0, 1}),
                     key=lambda t: (-t[0], t[1]))
        pareto = []
        for d, cost in pts:
            if not pareto or cost < pareto[-1][1]:
                pareto.append((d, cost))
        return tuple(pareto)

    @lru_cache(maxsize=None)
    def g(self, V: int, db: int) -> tuple[F, tuple[int, ...]]:
        """min over A-vectors (prod >= V) of sum_a c(d_a, db), with a minimiser: one column, solved exactly."""
        if V <= 1:
            return F(0), ()
        best = (INF, ())
        for d, cost in self._breaks(db):
            v, rest = self.g(-(-V // d), db)
            if cost + v < best[0]:
                best = (cost + v, (d,) + rest)
        return best

    @lru_cache(maxsize=None)
    def _gbreaks(self, VA: int) -> tuple[tuple[int, F], ...]:
        """(d, g(VA, d)) at the largest d of each level of the non-decreasing step function d -> g(VA, d)."""
        pts, prev = [], None
        for d in range(2, self.dmax + 1):
            v = self.g(VA, d)[0]
            if prev is not None and v != prev[1]:
                pts.append(prev)
            prev = (d, v)
        pts.append(prev)
        return tuple((d, v) for d, v in pts if v < INF)

    def lower_col(self, VA: int, VB: int) -> tuple[F, tuple[int, ...]]:
        """The column relaxation's value and its B-vector."""
        br = self._gbreaks(VA)

        @lru_cache(maxsize=None)
        def h(V: int) -> tuple[F, tuple[int, ...]]:
            if V <= 1:
                return F(0), ()
            best = (INF, ())
            for d, v in br:
                w, rest = h(-(-V // d))
                if v + w < best[0]:
                    best = (v + w, (d,) + rest)
            return best

        return h(VB)

    def lower(self, VA: int, VB: int) -> F:
        return max(self.lower_col(VA, VB)[0], self.lower_col(VB, VA)[0])

    def grid(self, da: tuple[int, ...], db: tuple[int, ...]) -> F:
        return sum((self.c(a, b) for a in da for b in db), F(0))

    def upper(self, VA: int, VB: int) -> tuple[F, tuple[int, ...], tuple[int, ...]]:
        """A common slicing read off the relaxations: each relaxation's outer vector with each column's inner vector."""
        best = (INF, (), ())
        for X, Y, flip in ((VA, VB, False), (VB, VA, True)):
            _, yv = self.lower_col(X, Y)
            for db in set(yv):
                xv = self.g(X, db)[1]
                v = self.grid(xv, yv)
                cand = (v, yv, xv) if flip else (v, xv, yv)
                if v < best[0]:
                    best = cand
        return best

    @lru_cache(maxsize=None)
    def type(self, d: int) -> tuple[F, ...]:
        """The function b -> c(d, b): digits of one type are interchangeable in every slicing."""
        return tuple(self.c(d, b) for b in range(2, self.dmax + 1))

    def classmax(self, d: int) -> int:
        """The largest size with d's type: c is non-decreasing in d, so a class is an interval."""
        t, lo, hi = self.type(d), d, self.dmax
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.type(mid) == t:
                lo = mid
            else:
                hi = mid - 1
        return lo

    @property
    def closure(self) -> tuple[int, ...]:
        """S: every format's cap, closed under s -> T_f(s) = min(cap_f, 2 floor(L_f / M(s)) + 1), L_f = 2^acc // k.

        Raising each digit of an optimal slicing to the largest size its row's formats allow, given the other side's
        digits, keeps it feasible and its cost; repeating on both sides ends with every digit a cap or T_f of a partner
        digit, so some optimal slicing has every digit in S."""
        if not hasattr(self, "_S"):
            S = {f.cap for f in self.fmts}
            while True:
                new = {f.maxd(s) for f in self.fmts for s in S if f.acc is not None and s <= f.cap} - {0, 1}
                if new <= S:
                    break
                S |= new
            self._S = tuple(sorted({self.classmax(s) for s in S} - {1}, reverse=True))
        return self._S

    def exact(self, VA: int, VB: int) -> tuple[F, tuple[int, ...], tuple[int, ...]]:
        """The minimum over all interval slicings: every minimal B-vector over S, the A side by an exact DP."""
        S = self.closure
        best = (INF, (), ())
        for db in vectors(VB, S):
            row = {d: sum((self.c(d, b) for b in db), F(0)) for d in S}

            @lru_cache(maxsize=None)
            def h(V: int) -> tuple[F, tuple[int, ...]]:
                if V <= 1:
                    return F(0), ()
                out = (INF, ())
                for d in S:
                    if row[d] >= INF:
                        continue
                    w, rest = h(-(-V // d))
                    if row[d] + w < out[0]:
                        out = (row[d] + w, (d,) + rest)
                return out

            v, da = h(VA)
            if v < best[0]:
                best = (v, da, db)
        return best


def vectors(V: int, S: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Every non-increasing vector over S whose product first reaches V at its last digit: each minimal multiset of
    digits with product >= V, sorted, is one of them (dropping its smallest digit leaves it below V)."""
    out = []
    asc = sorted(S)

    def rec(prefix: tuple[int, ...], prod: int, hi: int):
        if prod >= V:
            out.append(prefix)
            return
        for d in asc:
            if d > hi:
                break
            rec(prefix + (d,), prod * d, d)

    rec((), 1, max(S))
    return out


if __name__ == "__main__":
    import sys
    C = Costs(inst(sys.argv[1] if len(sys.argv) > 1 else "h200-sheet"))
    print("S =", C.closure)
    for i in range(0, 13):
        row = []
        for j in range(0, 13):
            VA, VB = 255 * 2 ** i + 1, 255 * 2 ** j + 1
            lo, (up, da, db) = C.lower(VA, VB), C.exact(VA, VB)
            row.append(str(up) if lo == up else f"{lo}<{up}")
        print(i, " ".join(f"{x:>6}" for x in row))
