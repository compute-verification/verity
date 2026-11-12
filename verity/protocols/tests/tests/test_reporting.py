"""The rule for a row's error (`reporting`): it is the sum over what the row covers, exactly, and at 1 or more the row
states no amount, in words every table shares."""

from __future__ import annotations

from fractions import Fraction

from verity.protocols import reporting


def test_a_rows_error_is_the_exact_sum_over_its_checks():
    assert reporting.summed(Fraction(1, 3), 3) == 1
    assert reporting.summed(Fraction(1, 2 ** 128)) == Fraction(1, 2 ** 128)
    assert reporting.summed(0.25, 5) == Fraction(5, 4), "a float counts at its exact value"
    assert reporting.summed(2.0 ** -40, 2 ** 40) == 1


def test_a_row_states_no_amount_exactly_from_a_sum_of_one():
    assert reporting.NO_BOUND == "no bound at this period"
    below = reporting.summed(Fraction(1, 10 ** 12), 10 ** 12 - 1)
    assert below < 1 and reporting.no_bound(below) is None and reporting.no_bound(0) is None
    assert reporting.no_bound(Fraction(1)) == "no bound at this period: the row's error, 1, is 1 or more"
    assert reporting.no_bound(reporting.summed(0.01, 100)).startswith(reporting.NO_BOUND)
    assert reporting.no_bound(38.7) == "no bound at this period: the row's error, 38.7, is 1 or more"
