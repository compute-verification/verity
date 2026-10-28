"""The openings' pooled bound in exact rationals, which `audit.pooled` and `audit.floors` round up to floats and the line
of record (`line.line`) reads: computed here and in its Lean, `lean/PoubCheck/Pooled.lean`, held to this by the
difftest (`lean/tests/test_pooled.py`, refusals compared word for word).

Window t has U_t receipts and drew s_t of them, S = Σ_t s_t.  Except with δ, the windows' wrong shares f_t have
Σ_t s_t·f_t ≤ d̄·S = S·(1 − δ^(1/S)) and f_t ≤ 1 − s_t/U_t (`Accounting.Guarantees.Poub.PooledDraw`).

- The budget B stands for d̄·S from above: B = S·(1 − q), q = m/2^`ROOT_BITS` with q^S ≤ δ certified in integers
  (`below`: the power by squaring, each product rounded up at `POW_BITS`), m the largest the bisection finds (`root`).
  x ↦ x^S is monotone, so q ≤ δ^(1/S) and B ≥ d̄·S, at most S·2^-64 and the power's rounding above it.
- W (`wrong`) is the most Σ_t f_t·U_t those allow: the windows by U_t/s_t falling (`order`), each taking the largest
  share its cap and the budget left allow, min(1 − s_t/U_t, b/s_t) (the fractional knapsack's greedy, its optimum:
  `Properties.PoubCheck.PooledCovers` proves the Lean's W at least `Accounting.Poub.Draw.pooled`).
- `floors` are each window's share at most on the whole budget, min(1 − s_t/U_t, B/s_t), and `floor` the largest.

The JSON form (`run_json`) is the Lean's: `windows` a list of [U, s], each rational a string `n` or `n/d`; δ given as a
float is its exact value."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

ROOT_BITS = 64
"""q's numerator's bits: q = m/2^64."""
POW_BITS = 256
"""The bits each product of the certified power is rounded up to."""
_RATIONAL = re.compile(r"-?[0-9]+(/[0-9]+)?")


class Refused(ValueError):
    """The windows have no pooled bound, and why: the Lean's reason, word for word."""


class Malformed(ValueError):
    """The JSON form isn't the pooled bound's inputs."""


def number(x: object) -> Fraction:
    """`x` exactly: an integer, a `Fraction`, a float's exact value, or a string `n` or `n/d` with d positive."""
    if isinstance(x, Fraction):
        return x
    if type(x) is int:
        return Fraction(x)
    if type(x) is float:
        if x != x or x in (float("inf"), float("-inf")):
            raise Malformed(f"not a finite number: {x!r}")
        return Fraction(x)
    if isinstance(x, str) and _RATIONAL.fullmatch(x):
        num, _, den = x.partition("/")
        if den and int(den) == 0:
            raise Malformed(f"a rational's denominator is positive: {x!r}")
        return Fraction(int(num), int(den or 1))
    raise Malformed(f"a rational is an integer or a string n or n/d: {x!r}")


def _mul_up(a: int, b: int) -> int:
    return (a * b + (1 << POW_BITS) - 1) >> POW_BITS


def _pow_up(x: int, n: int) -> int:
    """At least (x/2^POW_BITS)^n, times 2^POW_BITS: by squaring, each product rounded up."""
    if n == 0:
        return 1 << POW_BITS
    r = _pow_up(x, n // 2)
    sq = _mul_up(r, r)
    return _mul_up(sq, x) if n % 2 == 1 else sq


def below(S: int, delta: Fraction, m: int) -> bool:
    """Whether (m/2^ROOT_BITS)^S ≤ δ, certified: the rounded-up power is at most δ·2^POW_BITS."""
    return _pow_up(m << (POW_BITS - ROOT_BITS), S) * delta.denominator <= delta.numerator << POW_BITS


def root(S: int, delta: Fraction) -> int:
    """q's numerator m: the bisection's over [0, 2^ROOT_BITS), where `below` holds there, else 0."""
    lo, hi = 0, 1 << ROOT_BITS
    for _ in range(ROOT_BITS + 1):
        if hi <= lo + 1:
            break
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if below(S, delta, mid) else (lo, mid)
    return lo if below(S, delta, lo) else 0


def budget(S: int, delta: Fraction) -> Fraction:
    """B = S·(1 − m/2^ROOT_BITS), at least S·(1 − δ^(1/S))."""
    return S * (1 - Fraction(root(S, delta), 1 << ROOT_BITS))


def share(window: tuple[int, int], b: Fraction) -> Fraction:
    """The most of a window's receipts that can be wrong, as a share, with `b` of the budget left: min(1 − s/U, b/s)."""
    u, s = window
    return min(1 - Fraction(s, u), b / s)


def order(windows: Sequence[tuple[int, int]]) -> list[int]:
    """The windows' indices by U_t/s_t falling, ties in window order."""
    return sorted(range(len(windows)), key=lambda t: Fraction(*windows[t]), reverse=True)


@dataclass(frozen=True, slots=True)
class Bound:
    """The pooled bound: the budget B, each window's share at most (`floors`), the wrong receipts W and the largest
    share."""

    budget: Fraction
    floors: tuple[Fraction, ...]
    wrong: Fraction
    floor: Fraction

    def to_json(self) -> dict:
        return {"budget": str(self.budget), "floor": str(self.floor), "floors": [str(f) for f in self.floors],
                "wrong": str(self.wrong)}


def bound(windows: Sequence[tuple[int, int]], delta: object) -> Bound:
    """The pooled bound of `windows`, window t's receipts U_t and draws s_t, at δ; refused (`Refused`) as the Lean
    refuses."""
    ws = [(u, s) for u, s in windows]
    if any(type(x) is not int or x < 0 for w in ws for x in w):
        raise Malformed("a window is its receipts and draws, nonnegative integers")
    d = number(delta)
    if not ws:
        raise Refused("a session has one or more windows")
    if d <= 0 or d >= 1:
        raise Refused("δ is above 0 and below 1")
    for t, (u, s) in enumerate(ws):
        if s == 0:
            raise Refused(f"window {t} draws no receipt")
        if u < s:
            raise Refused(f"window {t} draws {s} receipts of its {u}")
    B = budget(sum(s for _, s in ws), d)
    wrong, left = Fraction(0), B
    for t in order(ws):
        f = share(ws[t], left)
        wrong, left = wrong + f * ws[t][0], max(Fraction(0), left - f * ws[t][1])
    floors = tuple(share(w, B) for w in ws)
    return Bound(B, floors, wrong, max((Fraction(0), *floors)))


def of_json(j: Mapping) -> tuple[list[tuple[int, int]], Fraction]:
    """The JSON form's windows and δ (`Malformed` if they aren't)."""
    if not isinstance(j, Mapping) or "windows" not in j or "delta" not in j:
        raise Malformed("the pooled bound's inputs are its windows and δ")
    ws, d = j["windows"], j["delta"]
    if not isinstance(ws, list) or any(not isinstance(w, list) or len(w) != 2 for w in ws):
        raise Malformed("windows: a list of [receipts, draws]")
    if any(type(x) is not int or x < 0 for w in ws for x in w):
        raise Malformed("windows: a window's receipts and draws are nonnegative integers")
    if type(d) is not int and not isinstance(d, str):
        raise Malformed("delta: not an integer or a string")
    return [(u, s) for u, s in ws], number(d)


def run_json(j: Mapping) -> dict:
    """The pooled bound of the JSON form's inputs in the JSON form, or `{"refused": why}`."""
    ws, d = of_json(j)
    try:
        return bound(ws, d).to_json()
    except Refused as e:
        return {"refused": str(e)}
