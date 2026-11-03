"""The pooled bound in exact rationals (`pooled`): the budget's certified root is the largest the power allows, so the
budget is at least d̄·S and within 2S·2^-64 of it; W is the fractional knapsack's optimum, at least every share
vector the sample allows; `audit.pooled` rounds both up; every refusal, and the JSON form the Lean reads."""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction

import pytest

from verity.core.protocols.profile import IntegrityProfile, Subset
from verity.core.protocols.remote_bandwidth import audit, pooled

DELTA = Fraction(1, 2 ** 40)
GATE_RUNS = {"5/0": (2118, 130023424, 48), "5/2": (96, 130023424, 232), "6/0": (2118, 113770496, 48),
             "6/2": (96, 113770496, 232), "7/0": (2118, 130023424, 48), "7/2": (96, 130023424, 232)}
"""Each accepted session of gate runs 5–7 as `gate_run/k`: its record's windows, all alike (n of them, U receipts
drawn s times each), at δ = 2^-40."""
RECORDED = {"5/0": 75094234.44840989, "5/2": 15529188.8879725, "6/0": 65707455.14235865, "6/2": 13588040.276975937,
            "7/0": 75094234.44840989, "7/2": 15529188.8879725}
"""Each one's W as its verifier.json recorded it (`ledger.paced.wrong`), the float greedy the exact one replaced."""


def root_is_tight(S: int, delta: Fraction) -> None:
    """(m/2^64)^S ≤ δ < ((m + 2)/2^64)^S exactly: B ≥ d̄·S, and B − d̄·S < 2S/2^64."""
    m, one = pooled.root(S, delta), 1 << pooled.ROOT_BITS
    assert m ** S * delta.denominator <= delta.numerator * one ** S
    assert (m + 2) ** S * delta.denominator > delta.numerator * one ** S


@pytest.mark.parametrize("S,delta", [(1, Fraction(1, 3)), (2, Fraction(1, 2)), (5, DELTA), (7, Fraction(1e-9)),
                                     (232 * 96, DELTA), (48, Fraction(999, 1000)), (10 ** 4, Fraction(1, 10 ** 30))])
def test_the_root_is_certified_and_within_two_of_the_exact(S, delta):
    root_is_tight(S, delta)


def test_the_budget_is_exact_where_the_root_is_a_power_of_two():
    assert pooled.root(10, DELTA) == 1 << 60 and pooled.budget(10, DELTA) == Fraction(75, 8)
    assert pooled.root(40, DELTA) == 1 << 63 and pooled.budget(40, DELTA) == 20


def test_the_root_is_certified_at_gate_runs_sizes():
    for n, _, s in set(GATE_RUNS.values()):
        root_is_tight(n * s, DELTA)
        dbar_s = n * s * -math.expm1(math.log(2.0 ** -40) / (n * s))
        assert pooled.budget(n * s, DELTA) == pytest.approx(dbar_s, rel=1e-14)


def lp_optimum(ws: list[tuple[int, int]], B: Fraction) -> Fraction:
    """max Σ f_t·U_t over 0 ≤ f_t ≤ 1 − s_t/U_t, Σ s_t·f_t ≤ B: some optimum is a vertex, every share at 0 or its cap
    but at most one, which takes what the budget leaves."""
    caps = [1 - Fraction(s, u) for u, s in ws]
    best = Fraction(0)
    for full in itertools.product((False, True), repeat=len(ws)):
        used = sum(caps[t] * ws[t][1] for t in range(len(ws)) if full[t])
        if used > B:
            continue
        got = sum(caps[t] * ws[t][0] for t in range(len(ws)) if full[t])
        part = [min(caps[t], (B - used) / ws[t][1]) * ws[t][0] for t in range(len(ws)) if not full[t]]
        best = max(best, got + max(part, default=0))
    return best


def test_w_is_the_fractional_knapsacks_optimum_and_at_least_every_share_the_draw_allows():
    rng = random.Random(7)
    for _ in range(300):
        ws = [(u, rng.randint(1, u)) for u in (rng.randint(1, 60) for _ in range(rng.randint(1, 5)))]
        delta = rng.choice([DELTA, Fraction(1, 3), Fraction(1e-9), Fraction(rng.randint(1, 999), 1000)])
        b = pooled.bound(ws, delta)
        assert b.wrong == lp_optimum(ws, b.budget)
        caps = [1 - Fraction(s, u) for u, s in ws]
        for _ in range(5):
            f = [c * Fraction(rng.randint(0, 100), 100) for c in caps]
            used = sum(x * s for x, (_, s) in zip(f, ws))
            if used > b.budget:
                f = [x * b.budget / used for x in f]
            assert sum(x * u for x, (u, _) in zip(f, ws)) <= b.wrong
        assert b.floors == tuple(min(c, b.budget / s) for c, (_, s) in zip(caps, ws))
        assert b.floor == max(b.floors)


def test_the_greedy_takes_the_windows_by_receipts_a_draw_and_a_window_opened_whole_has_none():
    ws = [(1000, 10), (12, 10), (40, 40), (5000, 10)]
    b = pooled.bound(ws, Fraction(1, 10 ** 6))
    assert pooled.order(ws) == [3, 0, 1, 2] and b.floors[2] == 0
    # window 3 takes its cap, 4990 receipts at 9.98 of the budget, and window 0 the rest; window 1 has none left
    assert Fraction(998, 100) < b.budget < Fraction(998, 100) + Fraction(99, 10)
    assert b.wrong == 4990 + (b.budget - Fraction(998, 100)) / 10 * 1000
    assert pooled.order([(4, 2), (2, 1), (6, 3)]) == [0, 1, 2]
    assert pooled.bound([(40, 40)] * 3, DELTA).wrong == 0


def synthetic(n: int, u: int, s: int, delta: float) -> audit.Record:
    windows = tuple(audit.WindowRecord(t, (u,), t, t + 1, t + 1, IntegrityProfile(
        ("receipt",), (u,), {"receipt": Subset(s)}, delta, drawn={"receipt": s})) for t in range(n))
    return audit.Record(bytes(64), bytes(64), u, 4096, windows)


def test_audit_rounds_the_exact_bound_up_at_gate_runs_records():
    """`audit.pooled` is the least float at least the exact W and floor; each recorded W was the float greedy's, within
    four ulps of the exact one on either side."""
    for k, (n, u, s) in GATE_RUNS.items():
        r = synthetic(n, u, s, 2.0 ** -40)
        assert audit.pooled_windows(r) == ([(u, s)] * n, DELTA)
        b = audit.pooled_bound(r)
        wrong, floor = audit.pooled(r)
        for got, exact in ((wrong, b.wrong), (floor, b.floor)):
            assert Fraction(got) >= exact > Fraction(math.nextafter(got, -math.inf))
        assert audit.floors(r) == (floor,) * n
        assert abs(Fraction(RECORDED[k]) - b.wrong) < 4 * math.ulp(RECORDED[k]), k


def test_delta_is_the_floats_exact_value():
    assert pooled.number(2.0 ** -40) == DELTA and pooled.number(1e-9) == Fraction(1e-9) != Fraction("1e-9")
    assert pooled.number("3/6") == Fraction(1, 2)
    for bad in (math.nan, math.inf, "1/0", "1.5", None, True):
        with pytest.raises(pooled.Malformed):
            pooled.number(bad)


@pytest.mark.parametrize("windows,delta,why", [
    ([], DELTA, "a session has one or more windows"),
    ([(5, 2)], 0, "δ is above 0 and below 1"),
    ([(5, 2)], 1, "δ is above 0 and below 1"),
    ([(5, 2)], Fraction(-1, 3), "δ is above 0 and below 1"),
    ([(5, 2), (3, 0), (1, 2)], DELTA, "window 1 draws no receipt"),
    ([(5, 2), (2, 3), (3, 0)], DELTA, "window 1 draws 3 receipts of its 2"),
])
def test_each_refusal(windows, delta, why):
    with pytest.raises(pooled.Refused) as e:
        pooled.bound(windows, delta)
    assert str(e.value) == why
    assert pooled.run_json({"windows": [list(w) for w in windows], "delta": str(delta)}) == {"refused": why}


def test_the_json_form_is_the_bounds_and_refuses_what_isnt_its_inputs():
    j = {"windows": [[4000, 7], [900, 7]], "delta": "1/1099511627776"}
    b = pooled.bound([(4000, 7), (900, 7)], DELTA)
    assert pooled.run_json(j) == b.to_json() == {"budget": str(b.budget), "floor": str(b.floor),
                                                 "floors": [str(f) for f in b.floors], "wrong": str(b.wrong)}
    for bad in ({**j, "delta": 0.5}, {**j, "delta": True}, {**j, "windows": [[1]]}, {**j, "windows": [[2, True]]},
                {**j, "windows": [[-1, 1]]}, {**j, "windows": {}}, {"windows": j["windows"]}, []):
        with pytest.raises(pooled.Malformed):
            pooled.run_json(bad)
