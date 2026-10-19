"""period_cost: INT-PERIOD's windows and node-days come back, the served window's units are 32×32, n_min is the least
n under 1%, the period only shortens with load, draws and nodes, and every lever names what it changes."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pouw_period_cost", Path(__file__).parents[1] / "period_cost.py")
PC = importlib.util.module_from_spec(spec)
sys.modules["pouw_period_cost"] = PC
spec.loader.exec_module(PC)

LINE = PC.configurations()[0]


@pytest.mark.parametrize("change, n", [
    (dict(u=1.0), 4815), (dict(u=17 / 36), 10195), (dict(), 46604), (dict(r=1.0), 17544),
    (dict(u=1.0, draws=64), 842), (dict(u=1.0, draws=13), 3039), (dict(u=1.0, r=1.0), 1813),
])
def test_int_periods_windows_come_back(change, n):
    assert PC.windows(replace(PC.TODAY, **change)) == n


@pytest.mark.parametrize("change", [dict(), dict(u=1.0), dict(r=1.0, draws=64), dict(rounds=72, u=PC.U_FULL)])
def test_n_min_is_the_least_n_under_one_percent(change):
    c = replace(PC.TODAY, **change)
    n = PC.windows(c)
    assert PC.eps(n, c) < PC.TARGET <= PC.eps(n - 1, c)


def test_int_periods_line_costs_its_node_days():
    p = PC.period(LINE)
    assert p["windows"] == 46604 and p["draws"] == 372832
    assert p["prove_process_days"] == pytest.approx(284.8, abs=0.05)
    assert p["lean_process_days"] == pytest.approx(1924.6, abs=0.05)
    assert p["serving_days"] == pytest.approx(236.85, abs=0.01)


def test_the_served_window_is_32x32_units():
    per_step = PC.LAYERS * sum(c for c, _ in PC.SERVED_UNITS_PER_LAYER.values())
    assert per_step == 86016 and per_step * PC.ROUNDS == 1548288
    # step 0's 44 rows reach both 32-row tiles, steps 1-15's 5 rows one, the 2 empty rounds none
    assert per_step + 15 * per_step // 2 == 731136
    assert PC.U_TODAY == (44 + 15 * 5) / (PC.ROWS * PC.ROUNDS)
    assert PC.lanes_per_unit() == pytest.approx(39.62, abs=0.01)
    assert PC.lanes_per_unit(PC.LANE_OUTPUTS, 1.0) == 1.0


def test_the_period_shortens_with_load_draws_and_nodes():
    assert PC.U_TODAY < 0.5 < PC.U_FULL < 1.0
    us = [PC.windows(replace(PC.TODAY, u=u)) for u in (0.05, PC.U_TODAY, 0.5, PC.U_FULL, 1.0)]
    assert us == sorted(us, reverse=True)
    ns = [PC.windows(replace(PC.TODAY, draws=k)) for k in range(1, 257)]
    assert ns == sorted(ns, reverse=True)
    ps = [PC.period(replace(PC.TODAY, nodes=k)) for k in (1, 2, 4, 8, 64)]
    assert [p["days_to_period"] for p in ps] == sorted((p["days_to_period"] for p in ps), reverse=True)
    assert len({round(p["node_days"], 9) for p in ps}) == 1


def test_amortizing_a_process_cuts_only_its_fixed_part():
    w = PC.window_cost(PC.TODAY)
    lean = PC.window_cost(replace(PC.TODAY, lean_windows_per_process=64))
    rt = PC.RATES["w2"]
    assert w["lean_s"] - lean["lean_s"] == pytest.approx(w["statements"] * rt.lean_process_s * 63 / 64)
    assert lean["prove_s"] == w["prove_s"]


def test_every_lever_names_config_fields_an_owner_and_a_work_class():
    fields = {f.name for f in dataclasses.fields(PC.Config)}
    for lv in PC.LEVERS:
        assert set(lv.change) <= fields, lv.name
        assert lv.owner and lv.work in (1, 2, 3, 4) and lv.what
    assert [lv.name for lv in PC.LEVERS].count(PC.SPEND) == 1


def test_the_route_leaves_out_spend_and_each_step_shortens_the_period():
    steps = [PC.TODAY, *PC.route()]
    assert len(steps) > 2
    assert not any(PC.SPEND in c.name for c in steps)
    days = [PC.period(c)["days_to_period"] for c in steps]
    assert all(b < 0.99 * a for a, b in zip(days, days[1:]))
    assert all(c.nodes == PC.NODES for c in steps)


def test_a_statement_is_capped_by_its_provers_memory():
    rt = replace(PC.RATES["w2"], name="t", prove_lane_gb=10.0)
    assert PC.statement_lanes(rt) == int((PC.NODE_GB - rt.prove_gb) // 10.0) < PC.STATEMENT_LANES
    with pytest.raises(ValueError, match="past a"):
        PC.statement_lanes(replace(rt, prove_lane_gb=float(PC.NODE_GB)))


def test_no_budget_is_refused():
    with pytest.raises(ValueError, match="no logical budget"):
        PC.windows(replace(PC.TODAY, eps_flat=0.01))


def test_main_writes_the_report(tmp_path, capsys):
    out, md = tmp_path / "period.json", tmp_path / "period.md"
    assert PC.main(["--json", str(out), "--md", str(md)]) == 0
    rep = json.loads(out.read_text())
    assert {"rates", "configurations", "levers", "draws_scan", "rounds_scan", "sensitivity"} <= set(rep)
    assert rep["configurations"][0]["windows"] == 46604
    assert rep["sensitivity"]["delta_s_log2"]["40"] == 4815
    assert "| lever |" in md.read_text() == capsys.readouterr().out
