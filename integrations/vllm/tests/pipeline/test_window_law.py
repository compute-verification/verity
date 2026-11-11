"""The work table of a window whose reader templates hold a stated share of wrong units (`window_law.reader_floors`,
`reader_table`), held to the count bound the record states of a design's strata (`profile.Stratified.count_bound`) at the
samples the work sampling design draws of them (`profile.work_k`)."""
from __future__ import annotations

from fractions import Fraction

import pytest

from verity.core.protocols.profile import Stratified, Stratum, work_k
from verity.core.protocols.spot.one_stage import draw as D
from verity_vllm.pipeline import window_law as WL

DELTA = Fraction(1, 100)


def _bound(units: dict[str, int], k: dict[str, int], delta: Fraction = DELTA) -> int:
    """The count bound at ``delta`` over the strata ``units``, stratum t sampling ``k[t]``."""
    return Stratified(tuple(Stratum(t, n, k[t]) for t, n in units.items())).count_bound(delta)


def _shares(units: dict[str, int], k: int) -> dict[str, int]:
    """Each stratum's ceil(k n_s / N)."""
    n = sum(units.values())
    return {t: -(-k * n_s // n) for t, n_s in units.items()}


def _class(positions: int) -> dict[str, int]:
    """Six reader templates of a window of ``positions`` positions whose cut linears take three shapes, at 32, 16 and 16
    calls a position: an A-row reader and an output reader a shape, a unit a position and call."""
    return {f"{side}{j}": positions * calls for side in "ay" for j, calls in enumerate((32, 16, 16))}


@pytest.mark.parametrize("units, eps", [
    ({"a": 40, "b": 20, "c": 20}, Fraction(1, 10)),
    ({"a": 40, "b": 20, "c": 20}, Fraction(4, 125)),
    ({"a": 7, "b": 100, "c": 1}, Fraction(1, 20)),
    ({"a": 64}, Fraction(1, 4)),
    ({"a": 3, "b": 5}, Fraction(1)),
])
def test_the_floors_are_the_least_equal_share_whose_count_bound_holds_eps(units, eps):
    """Against every share from one unit up: the floors are the first share's whose count bound over the templates is at
    most eps of their units, each floor at least 1 and at most its template's units."""
    n = sum(units.values())
    floors = WL.reader_floors(units, eps, DELTA)
    first = next(k for k in range(1, n + 1) if _bound(units, _shares(units, k)) <= eps * n)
    assert floors == _shares(units, first)
    assert all(1 <= floors[t] <= units[t] for t in units) and _bound(units, floors) <= eps * n


@pytest.mark.parametrize("positions", [140, 112, 108])
@pytest.mark.parametrize("eps", [Fraction(4, 125), Fraction(3, 100)])
def test_a_classes_floors_hold_eps_and_the_share_below_them_does_not(positions, eps):
    """At the sizes of a class's six reader templates: the floors' count bound at delta is at most eps of the readers'
    units, the share one below theirs holds no such bound, and the floors sample well under a unit in ten."""
    units = _class(positions)
    n = sum(units.values())
    floors = WL.reader_floors(units, eps, DELTA)
    assert _bound(units, floors) <= eps * n
    k = next(k for k in range(1, n + 1) if _shares(units, k) == floors)
    assert k > 1 and _bound(units, _shares(units, k - 1)) > eps * n
    assert sum(floors.values()) * 10 < n


def test_a_reader_table_samples_each_reader_at_its_floor_or_more_under_any_work_design():
    """The table credits every template alike, each reader at its floor and every other template at floor 1, so under
    ``work:K`` a reader samples its floor at K = 1 and no less at a larger K (`profile.work_k`), and the readers' count
    bound still holds eps, whatever the other templates sample."""
    readers, eps = _class(27), Fraction(1, 20)
    counts = {"rest0": 500, **readers, "rest1": 3}
    table = WL.reader_table(counts, readers, eps, DELTA)
    floors = WL.reader_floors(readers, eps, DELTA)
    assert table == {t: {"work": 1, "floor": floors.get(t, 1)} for t in counts} and list(table) == list(counts)
    works, total = D.work_table(table), sum(counts.values())
    for big_k in (1, 40, 400, total):
        k = {t: work_k(big_k, works[t][0], counts[t], total, works[t][1]) for t in counts}
        assert all(k[t] >= floors[t] for t in readers) and (big_k > 1 or all(k[t] == floors[t] for t in readers))
        assert _bound(readers, k) <= eps * sum(readers.values())


@pytest.mark.parametrize("call, why", [
    (lambda: WL.reader_floors({}, Fraction(1, 10), DELTA), "each template has a unit to sample"),
    (lambda: WL.reader_floors({"a": 4, "b": 0}, Fraction(1, 10), DELTA), "each template has a unit to sample"),
    (lambda: WL.reader_floors({"a": 4}, 0.1, DELTA), "exact fractions"),
    (lambda: WL.reader_floors({"a": 4}, Fraction(0), DELTA), "0 < eps <= 1"),
    (lambda: WL.reader_floors({"a": 4}, Fraction(1, 10), Fraction(1)), "0 < delta < 1"),
    (lambda: WL.reader_table({"a": 4}, [], Fraction(1, 10), DELTA), "at least one"),
    (lambda: WL.reader_table({"a": 4}, ["b"], Fraction(1, 10), DELTA), "templates of the window"),
])
def test_floors_of_no_template_or_at_no_exact_share_are_refused(call, why):
    with pytest.raises(ValueError, match=why):
        call()
