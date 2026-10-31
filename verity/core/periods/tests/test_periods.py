"""`verity.core.periods`: spans, cuts and the monitoring period, their JSON, and the monitoring periods results are
presented for."""

from __future__ import annotations

import pytest

from verity.core.periods import MONITORING_PERIODS, Cut, MonitoringPeriod, PeriodError, TickSpan, block_count


def test_a_span_stops_after_its_last_tick():
    p = TickSpan(3, 4)
    assert p.stop == 7
    assert list(p.ticks) == [3, 4, 5, 6]
    assert list(TickSpan(5, 0).ticks) == []


def test_within_is_inclusive_at_both_ends():
    outer = TickSpan(0, 10)
    assert TickSpan(3, 2).within(outer) and outer.within(outer) and TickSpan(10, 0).within(outer)
    assert not TickSpan(9, 2).within(outer) and not outer.within(TickSpan(1, 9))


@pytest.mark.parametrize("start, length", [(-1, 0), (0, -1), (True, 1), (0, 1.0), ("0", 1)])
def test_a_span_is_of_naturals(start, length):
    with pytest.raises(PeriodError):
        TickSpan(start, length)


def test_a_cut_runs_from_start_to_stop_in_order():
    span = TickSpan(0, 10)
    assert Cut(span, (0, 4, 4, 10)).parts == (TickSpan(0, 4), TickSpan(4, 0), TickSpan(4, 6))
    assert Cut(TickSpan(5, 0), (5,)).parts == ()
    for bad in [(), (1, 10), (0, 9), (0, 11, 10), (0, 5, 3, 10)]:
        with pytest.raises(PeriodError):
            Cut(span, bad)


def test_a_monitoring_period_cuts_each_execution_block_into_sampling_blocks():
    m = MonitoringPeriod.of(TickSpan(0, 10), [0, 4, 10], [[0, 4], [4, 5, 10]])
    assert m.execution_blocks == 2
    assert [m.execution_span(j) for j in range(2)] == [TickSpan(0, 4), TickSpan(4, 6)]
    assert m.sampling_span(0, 0) == TickSpan(0, 4)
    assert [m.sampling_span(1, t) for t in range(2)] == [TickSpan(4, 1), TickSpan(5, 5)]
    assert all(m.sampling_span(j, t).within(m.execution_span(j)) for j in range(2) for t in range(len(m.sampling[j].parts)))


@pytest.mark.parametrize("execution, sampling", [
    ([0, 4, 10], [[0, 4]]),                # one sampling cut short
    ([0, 4, 10], [[0, 4], [5, 10]]),       # a sampling cut not of its block's span
    ([0, 4, 9], [[0, 4], [4, 9]]),         # the execution cut stops short of the span
])
def test_a_monitoring_period_refuses_cuts_that_do_not_nest(execution, sampling):
    with pytest.raises(PeriodError):
        MonitoringPeriod.of(TickSpan(0, 10), execution, sampling)


def test_a_monitoring_period_refuses_a_cut_of_another_span():
    with pytest.raises(PeriodError):
        MonitoringPeriod(TickSpan(0, 10), Cut(TickSpan(0, 9), (0, 9)), (Cut(TickSpan(0, 9), (0, 9)),))


def test_the_json_round_trips_and_accepts_only_what_it_writes():
    m = MonitoringPeriod.of(TickSpan(0, 10), [0, 4, 10], [[0, 4], [4, 5, 10]])
    j = m.to_json()
    assert j == {"execution": [0, 4, 10], "sampling": [[0, 4], [4, 5, 10]], "span": {"len": 10, "start": 0}}
    assert MonitoringPeriod.from_json(j) == m
    assert TickSpan.from_json({"len": 4, "start": 3}) == TickSpan(3, 4)
    for bad in [{"len": 4}, {"len": 4, "start": 3, "clock": "a"}, {"len": True, "start": 3}, [3, 4], None]:
        with pytest.raises(PeriodError):
            TickSpan.from_json(bad)
    for bad in [j | {"blocks": 2}, j | {"sampling": [[0, 4]]}, j | {"execution": "0,4,10"}, j | {"sampling": {}}]:
        with pytest.raises(PeriodError):
            MonitoringPeriod.from_json(bad)


def test_the_monitoring_periods_are_a_week_a_month_and_a_year():
    assert MONITORING_PERIODS == (("week", 604800), ("month", 2592000), ("year", 31536000))


@pytest.mark.parametrize("seconds, block_s", [(10, 3), (9, 3), (2, 3), (0, 1), (7, 7)])
def test_the_block_count_is_a_machines_monitoring_periods_execution_blocks(seconds, block_s):
    n = block_count(seconds, block_s)
    bounds = list(range(0, n * block_s + 1, block_s))
    m = MonitoringPeriod.of(TickSpan(0, n * block_s), bounds, [[a, b] for a, b in zip(bounds, bounds[1:])])
    assert m.execution_blocks == n and n * block_s <= seconds < (n + 1) * block_s
    assert block_count(seconds, block_s, 5) == 5 * n


def test_the_block_count_of_the_reference_deployment():
    assert {name: block_count(s, 600, 12500) for name, s in MONITORING_PERIODS} == {
        "week": 12_600_000, "month": 54_000_000, "year": 657_000_000}


@pytest.mark.parametrize("seconds, block_s, machines", [(10, 0, 1), (-1, 3, 1), (10, 3, -1), (10.0, 3, 1), (10, True, 1)])
def test_the_block_count_refuses_what_is_no_count(seconds, block_s, machines):
    with pytest.raises(PeriodError):
        block_count(seconds, block_s, machines)
