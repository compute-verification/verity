"""The verifier's coins: its own randomness and the one-stage draws it makes from it.

The verifier draws every coin from its own randomness (`OwnRandomness`), with no derivation and no beacon, after the
prover's commitments are registered.  `draw_first` draws a one-stage law's units as the Lean verifier's
`Flock.Draw.drawWith` does from the same bytes, and `closure_of` widens a work draw by the verifier's closure map.  A
drawn unit's outcome is final: nothing redraws.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Any

from verity.ml.randomness import uniform_from_words

from .one_stage.draw import digits, strata_problem

_LAW_KEYS = {"subset": {"law", "k"}, "bernoulli": {"law", "p"}, "stratified": {"law", "k", "strata"},
             "work": {"law", "k", "strata"}}


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _ratio(p: Any) -> tuple[int, int] | None:
    num, _, den = str(p).partition("/")
    if digits(num) and digits(den) and 0 < int(den) and int(num) <= int(den):
        return int(num), int(den)
    return None


def _law_problem(law: Mapping[str, Any]) -> str | None:
    kind = law.get("law")
    if kind not in _LAW_KEYS:
        return "a one-stage law is subset, bernoulli, stratified or work"
    if set(law) != _LAW_KEYS[kind]:
        return f"a {kind} law is exactly {sorted(_LAW_KEYS[kind])}"
    if kind == "bernoulli":
        return None if _ratio(law["p"]) else "a bernoulli law's p is NUM/DEN with 0 <= NUM <= DEN and DEN > 0"
    if not _int(law["k"]) or law["k"] < 0:
        return f"a {kind} law's k is a natural number"
    if kind != "subset" and not isinstance(law["strata"], list):
        return f"a {kind} law is expanded: its strata are a list (one_stage.draw.derive)"
    return None


class _RanOut(Exception):
    pass


def _width(n: int) -> int:
    return (n.bit_length() + 64 + 7) // 8


def _uniform(n: int, read: Callable[[int], bytes]) -> int:
    w = _width(n)
    return uniform_from_words(n, (int.from_bytes(read(w), "big") for _ in iter(int, 1)), 8 * w)


def _subset(k: int, n: int, read: Callable[[int], bytes]) -> list[int]:
    swapped: dict[int, int] = {}
    for i in range(k):
        j = i + _uniform(n - i, read)
        swapped[i], swapped[j] = swapped.get(j, j), swapped.get(i, i)
    return sorted(swapped.get(i, i) for i in range(k))


def _bernoulli(num: int, den: int, n: int, read: Callable[[int], bytes]) -> list[int]:
    if num == 0:
        return []
    if num == den:
        return list(range(n))
    return [u for u in range(n) if _uniform(den, read) < num]


def _on(src: bytes) -> Callable[[int], bytes]:
    at = 0

    def read(w: int) -> bytes:
        nonlocal at
        if at + w > len(src):
            raise _RanOut
        at += w
        return src[at - w:at]
    return read


class OwnRandomness:
    """The verifier's own randomness, read directly: exact samplers over raw bytes from `source` (default the operating
    system's generator), consumed in order, with no hash and no derivation.  They are the Lean verifier's `Flock.Draw` and
    M0's `unit_draw` samplers: `uniform(n)` reads w = ceil((bitlen(n) + 64) / 8)-byte big-endian words and returns the
    first below the largest multiple of n, mod n; `bernoulli(a/b)`, a/b in lowest terms, is `uniform(b) < a` (p = 0 and
    p = 1 read nothing); `subset` is a sparse partial Fisher-Yates."""

    def __init__(self, source: Callable[[int], bytes] = os.urandom) -> None:
        self._source = source

    def read(self, n: int) -> bytes:
        b = self._source(n)
        if len(b) != n:
            raise ValueError("the randomness source ran out")
        return b

    def uniform(self, n: int) -> int:
        if n <= 0:
            raise ValueError("uniform(n) needs n > 0")
        return _uniform(n, self.read)

    def bernoulli(self, p: Fraction) -> bool:
        p = Fraction(p)
        if not 0 <= p <= 1:
            raise ValueError("bernoulli(p) needs 0 <= p <= 1")
        if p in (0, 1):
            return p == 1
        return self.uniform(p.denominator) < p.numerator

    def subset(self, k: int, n: int) -> tuple[int, ...]:
        if not 0 <= k <= n:
            raise ValueError("subset needs 0 <= k <= n")
        return tuple(_subset(k, n, self.read))

    def stream(self, need: int, run: Callable[[Callable[[int], bytes]], list[int]]) -> list[int]:
        """Lean's `Flock.Draw.onWith`: `run` on a stream of `need` bytes, extended by `need` more and rerun from its
        start whenever it runs out, so the draw stays exact."""
        src = self.read(need)
        while True:
            try:
                return run(_on(src))
            except _RanOut:
                src += self.read(need)


def closure_of(closure: Mapping[int, Sequence[int]], units: Sequence[int]) -> list[int]:
    """A draw's closure under the verifier's closure map (Lean `Flock.Draw.closureOf`): every unit in a drawn unit's
    closure that is not itself drawn, ascending."""
    drawn = set(units)
    return sorted({c for u in units for c in closure.get(u, ()) if c not in drawn})


def draw_first(randomness: OwnRandomness, law: Mapping[str, Any], population: int,
               closure: Mapping[int, Sequence[int]] | None = None) -> dict[str, Any]:
    """A draw under a one-stage law: the one-stage draw object over the positions [0, population) (`one_stage.draw`;
    `check_draw` holds it to the registration and the law), as the Lean verifier's `Flock.Draw.drawWith` draws it from
    the same bytes: `subset` and `bernoulli` on one stream (`bernoulli` at the law's NUM/DEN as written), `stratified`
    and `work` one stream per stratum, in order, each stratum's `subset k_s n_s` mapped to the stratum's units.  With the
    verifier's closure map (a work law only, as `flock-verify draw --closure`), the draw carries its `closure`
    (`closure_of`) when that is not empty."""
    problem = _law_problem(law)
    if problem:
        raise ValueError(problem)
    kind = law["law"]
    if closure is not None and kind != "work":
        raise ValueError("a closure widens a work draw only")
    if kind == "subset":
        k = law["k"]
        if k > population:
            raise ValueError(f"a subset draw of {k} exceeds the population of {population}")
        units = randomness.stream(2 * k * _width(population) + 64, lambda read: _subset(k, population, read))
    elif kind == "bernoulli":
        num, den = _ratio(law["p"])
        units = randomness.stream(2 * population * _width(den) + 64, lambda read: _bernoulli(num, den, population, read))
    else:
        problem = strata_problem(law, population)
        if problem:
            raise ValueError(problem)
        units = []
        for s in law["strata"]:
            ranges, k_s = s["units"], s["k"]
            n_s = sum(hi - lo for lo, hi in ranges)
            local = randomness.stream(2 * k_s * _width(n_s) + 64, lambda read: _subset(k_s, n_s, read))
            units += [_nth(ranges, j) for j in local]
        units.sort()
    out = {**dict(law), "population": population, "units": units}
    if closure is not None:
        extra = closure_of(closure, units)
        if extra:
            out["closure"] = extra
    return out


def _nth(ranges: Sequence[Sequence[int]], j: int) -> int:
    for lo, hi in ranges:
        if j < hi - lo:
            return lo + j
        j -= hi - lo
    raise IndexError(j)


__all__ = ["OwnRandomness", "closure_of", "draw_first"]
