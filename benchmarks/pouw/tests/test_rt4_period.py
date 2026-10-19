"""RT4-INT's reading of an accounting period under prefix credit (#2722, `Guarantees.ComposePeriod` on
`cursor/ca-int-period-ea24` at 714477b8ef), in the numbers its statement and PERIOD-PATH's `period_cost.py` quote:
`a = 0.218`, `b = 0.998`, `p = 7/8`, `p0 = 1/2`, `δ_s = 2^-40`, `ε_♭ = 0.571%`, SERVE-H1's window (`M = 18`,
`r = 2.87`) and W2's draw (`K_d = 8`).

`PeriodTailMax` prices the stopping time the cheater steers: `RoundGame.BudgetAt` is `E[exp(d/Λ)·1{pass}] ≤ 1` given the
past, so some accepted prefix carries share `Λ·ln(1/δ_s)` with probability at most `δ_s`, whatever prefix the cheater
cuts at. The bound is on the prefix's total, `Λ·ln(1/δ_s)` of one window's declared words, so the period's ε is read at
its accepted prefix `j` (`rOfPrefix … (pr.prefixLen x)`), and it is vacuous (at least 1) for short prefixes. #2722 states
work, not credit (`CircuitCreditPeriod` is unstated), and ASK-64 = A credits the live tiles up to the first refused
window. What that credit pays a cheater depends on whether it is discounted by ε at `j`:

* undiscounted, a window that skips every tile whole passes its `K_d` draws with probability `p0^K_d = 2^-8` at
  `1 − b = 0.2%` of a window's work, so a stream of such windows earns 1.95× an honest stream's credit per unit of
  work, every credited word wrong (RT4-INT-8);
* any other skip or fold pays less than honest work in expectation, since a refused window forfeits its credit;
* discounted, `(1 − ε(j))₊` of the prefix's live words, a prefix of at most 201 windows (at the gate's `u = 119/1152`)
  earns nothing, and a prefix's deficit is at most what the discount takes.

A quote of the period's ε at its cap `n` rather than at `j` is false for a prefix cut early: skipping half of window 0's
tiles and refusing window 1 passes with probability at least `2^-4` (Jensen on `p0^H`), while ε(n) says under 1%.
"""
from __future__ import annotations

import math
from fractions import Fraction

A, B, P, P0 = 0.218, 0.998, 7 / 8, 0.5
DELTA_S_LOG2, EPS_FLAT = 40, 0.00571
ROUNDS, R, DRAWS = 18, 2.87, 8
#: the period's admitted share of rows (r20261019-012935-2c52): every tile live, live tiles' words, the gate's credit
U = {"all": 1.0, "live": 17 / 36, "gate": 119 / 1152}


def lam(draws: int = DRAWS, r: float = R, rounds: int = ROUNDS) -> float:
    """Λ at equal rounds and ρ = 1/r (`SplitRateExp`), as #2722's text and `period_cost.lam` take it."""
    w, x = 1 / rounds, draws / (r * rounds)
    return max(A * w / -math.log(P + (1 - P) * math.exp(-x)), B * w / -math.log(P0 + (1 - P0) * math.exp(-x)))


def tax() -> float:
    """`Λ·ln(1/δ_s)`: the prefix's whole logical term, in windows of declared words."""
    return lam() * DELTA_S_LOG2 * math.log(2)


def eps(j: int, u: float) -> float:
    """ε at an accepted prefix of `j` windows: `ε_♭ + Λ·ln(1/δ_s)/(u·j)`."""
    return EPS_FLAT + tax() / (u * j)


def pass_all_skipped(draws: int = DRAWS) -> Fraction:
    """A window whose every tile is skipped whole: each of its `draws` drawn tiles passes with the luck `p0`."""
    return Fraction(1, 2) ** draws


def per_work(skip: float, luck: float, saved: float, draws: int = DRAWS) -> float:
    """Expected credit per unit of work of a window skipping share `skip` of its tiles (each saving `saved` of its work,
    each drawn one passing with `luck`), credited whole when it passes: `(1 − (1 − luck)·skip)^draws / (1 − saved·skip)`.
    The binomial form bounds the draw without replacement from above (`luck^H` is convex in `H`)."""
    return (1 - (1 - luck) * skip) ** draws / (1 - saved * skip)


def test_the_line_is_2722s():
    assert round(lam(), 3) == 0.745 and round(tax(), 2) == 20.65
    first = {k: math.ceil(tax() / (u * (0.01 - EPS_FLAT))) for k, u in U.items()}
    assert first == {"all": 4815, "live": 10195, "gate": 46604}


def test_a_short_prefix_is_vacuous():
    """ε at the accepted prefix is at least 1 up to 20 windows with every tile live, 43 on live words and 201 on the
    gate's credit: a prefix that short proves nothing of its credited words."""
    last = {}
    for k, u in U.items():
        last[k] = max(j for j in range(1, 1000) if eps(j, u) >= 1)
    assert last == {"all": 20, "live": 43, "gate": 201}


def test_an_all_skip_window_pays_on_undiscounted_prefix_credit():
    """RT4-INT-8. At the lucks the headline is computed at, a window of whole skips passes with `2^-8` at `0.2%` of the
    work: 1.95× an honest window's credit per unit of work. Each such accepted prefix is one window (two with
    `2^-16`), where ε is vacuous, so a credit of `(1 − ε(j))₊` of the prefix's live words pays it nothing."""
    p = pass_all_skipped()
    assert p == Fraction(1, 256)
    assert math.isclose(per_work(1.0, P0, B), float(p) / (1 - B)) and per_work(1.0, P0, B) > 1.95
    assert all(eps(j, U["all"]) >= 1 for j in range(1, 21))
    assert float(p) ** 21 < 2.0 ** -160


def test_every_other_strategy_loses_in_expectation():
    """Prefix credit forfeits a refused window, so whole skips of less than 99.8% of the tiles, and every fold, earn less
    credit per unit of work than honest serving at `K_d = 8`; at `K_d = 9` even the all-skip window does
    (`2^-9 < 1 − b`)."""
    grid = [i / 10_000 for i in range(1, 10_001)]
    gains = [s for s in grid if per_work(s, P0, B) > 1]
    assert gains and min(gains) >= 0.998
    assert all(per_work(s, P, A) < 1 for s in grid)
    assert per_work(1.0, P0, B, draws=9) < 1


def test_a_quote_at_the_cap_is_false_for_a_prefix_cut_early():
    """A period of cap 46,604 windows quotes ε(n) < 1% of the gate's credit. Skipping half of window 0's tiles whole and
    refusing window 1 leaves a credited prefix of one window, half of its words wrong, which passes with probability at
    least `p0^(K_d/2) = 2^-4`: far above δ_s. At the prefix the statement reads, ε(1) is above 1."""
    n = 46_604
    assert eps(n, U["gate"]) < 0.01
    assert P0 ** (DRAWS * 0.5) == 2.0 ** -4 > 2.0 ** -DELTA_S_LOG2
    assert eps(1, U["gate"]) > 1
