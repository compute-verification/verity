"""`cap_usd_per_day` on a budget line (the CI pool: $65 a day on `vy-coord-`): spend over a rolling 24 hours, not a calendar day,
trips the line and clears by itself as the window moves; `pods create` refuses meanwhile.  Fake clocks only."""
from __future__ import annotations

import json

import pytest

from research.pods import budgets as B
from research.pods import guard as G

from test_budgets_guard import T0, FakeRunPod, cfg, pod, step

POOL = """
[guard]
project = ["vy"]

[budgets]
"vy-coord-" = { cap_usd_per_day = 10, expires = "2026-10-30T00:00Z", max_pod_hours = 168, by = "Daniel: CI pool, $65/day" }
"vy-both-" = { cap_usd = 100, cap_usd_per_day = 10, expires = "2026-10-30T00:00Z", max_pod_hours = 168 }
"""
H = 3600


def test_a_line_takes_a_daily_cap_a_total_cap_or_both_and_needs_one():
    b = B.parse(POOL)
    assert (b.lines["vy-coord-"].cap_usd, b.lines["vy-coord-"].cap_usd_per_day) == (None, 10.0)
    assert (b.lines["vy-both-"].cap_usd, b.lines["vy-both-"].cap_usd_per_day) == (100.0, 10.0)
    with pytest.raises(B.BudgetError, match="missing cap_usd or cap_usd_per_day"):
        B.parse('[budgets]\n"vy-x-" = { expires = "2026-10-30T00:00Z", max_pod_hours = 1 }')
    with pytest.raises(B.BudgetError, match="cap_usd_per_day is not a number of dollars"):
        B.parse('[budgets]\n"vy-x-" = { cap_usd_per_day = -1, expires = "2026-10-30T00:00Z", max_pod_hours = 1 }')


def test_the_rolling_window_trips_the_line_and_clears_by_itself_as_it_moves():
    api = FakeRunPod([pod("a", "vy-coord-1", 2, 5), pod("b", "vy-coord-2", 2, 5)])
    state, lines = G.new_state(T0), []
    for h in range(3):                                          # $4/h: $8 by T0 + 2 h, under the $10
        step(api, POOL, state, T0 + h * H, lines)
    assert api.deleted == [] and state["lines"]["vy-coord-"]["window_usd"] == pytest.approx(8.0)
    step(api, POOL, state, T0 + 2.5 * H, lines)
    assert sorted(api.deleted) == ["a", "b"] and state["lines"]["vy-coord-"]["tripped"]["kind"] == "per_day"
    assert "DAILY CAP $10 reached ($10.00 in the last 24 h)" in state["lines"]["vy-coord-"]["tripped"]["why"]
    api.add(pod("n", "vy-coord-3", 0.2, -3 * 60))
    step(api, POOL, state, T0 + 3 * H, lines)
    assert "n" not in api.deleted and any("OVER-CAP n" in ln for ln in lines)
    tally = state["lines"]["vy-coord-"]
    assert "at its daily cap ($10.10 in the last 24 h of $10)" in B.gate("vy-coord-4", 1, {**state, "budgets": B.parse(POOL).view()}, T0 + 3 * H)
    for m in range(10, 24 * 60 + 1, 10):                        # a poll every 10 min: each records the spend since the last one
        step(api, POOL, state, T0 + 3 * H + m * 60, lines)
    # at T0 + 27 h the first hours' spend has left the window; n's $0.20/h is what's in it
    assert tally["tripped"] is None and any("UNTRIPPED 'vy-coord-'" in ln and "under the daily cap $10" in ln for ln in lines)
    assert tally["spent"] > 10, "the total keeps counting; only the window moved"
    assert B.gate("vy-coord-4", 1, {**state, "budgets": B.parse(POOL).view()}, T0 + 27 * H) is None


def test_a_steady_rate_under_the_daily_cap_never_trips_over_days():
    api = FakeRunPod([pod("a", "vy-coord-1", 0.4, 5)])            # $9.60 a day
    state, lines = G.new_state(T0), []
    for m in range(0, 4 * 24 * 60 + 1, 10):
        step(api, POOL, state, T0 + m * 60, lines)
    t = state["lines"]["vy-coord-"]
    assert api.deleted == [] and t["tripped"] is None and t["spent"] == pytest.approx(0.4 * 96)
    assert 9.6 <= t["window_usd"] <= 9.6 + 0.4 * (G.BUCKET_S + 600) / 3600 + 1e-9, \
        "the window over-counts by at most a bucket and a poll interval, never under"
    assert len(t["window"]) <= 24 * 3600 // G.BUCKET_S + 1


def test_a_line_with_both_caps_trips_on_either_and_clears_only_when_both_allow(tmp_path):
    api = FakeRunPod([pod("a", "vy-both-1", 5, 5)])
    state, lines = G.new_state(T0), []
    step(api, POOL, state, T0, lines)
    step(api, POOL, state, T0 + 2 * H, lines)
    assert state["lines"]["vy-both-"]["tripped"]["kind"] == "per_day" and api.deleted == ["a"]
    raised_day = POOL.replace('cap_usd = 100, cap_usd_per_day = 10', 'cap_usd = 5, cap_usd_per_day = 100')
    step(api, raised_day, state, T0 + 2 * H + 60, lines)
    assert state["lines"]["vy-both-"]["tripped"]["kind"] == "cap", "under the new daily cap, over the total: tripped on the total"


def test_the_window_survives_a_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_GUARD_DIR", str(tmp_path / "guard"))
    (tmp_path / "budgets.toml").write_text(POOL)
    api = FakeRunPod([pod("a", "vy-coord-1", 3, 5)])
    t = [T0]
    clock, sleep = (lambda: t[0]), (lambda s: t.__setitem__(0, t[0] + H))
    assert G.run(cfg(tmp_path), api, clock=clock, sleep=sleep, iterations=3) == 0       # $6 by T0 + 2 h
    assert G.run(cfg(tmp_path), api, clock=clock, sleep=sleep, iterations=3) == 0       # continues: $12 by T0 + 4 h
    assert api.deleted == ["a"], "the restart kept the window: $6 + $6 >= $10"
    state = json.loads((tmp_path / "guard" / "guard-budgets.json").read_text())
    assert state["lines"]["vy-coord-"]["window_usd"] >= 10


def test_status_shows_the_window(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RESEARCH_GUARD_DIR", str(tmp_path / "guard"))
    (tmp_path / "budgets.toml").write_text(POOL)
    t = [T0]
    G.run(cfg(tmp_path), FakeRunPod([pod("a", "vy-coord-1", 2, 5)]), clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + H),
          iterations=2)
    assert G.status(clock=lambda: T0 + H) == 0
    assert "'vy-coord-': $2.00, $2.00 in 24 h of $10/day" in capsys.readouterr().out

