"""`sampling_inputs`' closed form for the paced line's sampling term is `paced_line`'s arithmetic on uniform sessions,
with the wrong receipts spilling past one group and not, and with one openings coin over several groups; that arithmetic
is a stretch worked by hand; a region's solution meets the constraints it was solved under; and the code's coin over g
groups (`groups_per_coin`) is the model's shared draws, each plan at g agreeing with it."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))

import sampling_inputs as si  # noqa: E402


def test_the_closed_form_is_the_line_arithmetic_on_uniform_sessions():
    rows = si.check_closed_form()
    assert {row["beta"] > 1 for row in rows} == {True, False} and any(row["d"] < 1 for row in rows)


def test_a_stretch_by_hand():
    seen = [{"released_ns": 100 * g, "passes": 1, "arrivals_ns": [a]} for g, a in enumerate((150, 180, 350, 420))]
    got = si.span_line(si.groups_of(seen), 0, 3, wrong=15.0, blocks=10, per_ns=0.01, chip=0.5)
    # span 300 − 150; L = 150, 100, 150, 120; the group after pays L_max = 150 for 10 receipts, group 0 its gap for 5
    assert got["span_ns"] == 150 and got["l_max_ns"] == 150 and got["passes"] == 3
    assert got["rebuild_blocks"] == pytest.approx(4.5)
    assert got["wrong_absent_blocks"] == pytest.approx((10 * 150 + 5 * 100) / 150)
    assert got["held_lost_blocks"] == pytest.approx(0.5 + (2 * 150 * 4.5 + 2000) / 150)
    assert got["snapshot_lost_blocks"] == pytest.approx(15 / 3 + 4.5 / 3 + 0.5)


def test_a_region_solution_meets_its_constraints():
    p = {"blocks": 130_023_424, "gpus": 8, "shard_bytes": 62 * 2 ** 30, "weights_bytes": 524_204_113_920,
         "c_s": 0.057, "xi": 2.638e6, "chip": 743_590.5, "codec_loss": None}
    prices = {"fit": 0.544, "floor": 0.08, "fit_deployed": 1.09, "floor_deployed": 0.16}
    for flight in (2, 1):
        line = si.solve(p, prices, 0.01, theta=600.0, line_rule=True, flight=flight)
        shared = si.solve(p, prices, 0.01, theta=600.0, line_rule=True, flight=flight, shared=True)
        loose = si.solve(p, prices, 0.01, theta=600.0, line_rule=False, flight=flight)
        for s in (line, shared, loose):
            assert s["eps_held"] <= 0.01 * (1 + 1e-12) and s["o"] == pytest.approx(s["d"] * s["r"] / s["k"])
        for s in (line, shared):
            assert 2 * flight * p["xi"] * s["k"] / (s["r"] * p["blocks"]) <= 0.01 * (1 + 1e-12)
        assert line["d"] >= 1
        if flight == 2:
            assert shared["d"] < 1 and shared["o"] < line["o"]
        cost = {id(s): s["passes_gpus"] + s["proving_gpus_fit"] for s in (line, shared, loose)}
        assert cost[id(shared)] <= cost[id(line)]
        assert cost[id(loose)] <= cost[id(line)] and loose["rebuild_held"] > line["rebuild_held"]


def test_the_code_reads_one_coin_over_g_groups_as_the_merged_window_the_model_drew():
    """`audit.pooled` on each group's receipts and draws naming `groups_per_coin` g is the model's reading of shared
    draws, one window of g groups' receipts and the coin's draws, however the coin's draws fall in its groups."""
    from verity.core.protocols.remote_bandwidth import audit
    for g, shares in ((2, (0, 3)), (4, (1, 0, 2, 0)), (3, (0, 0, 1))):
        n, u = 5, 7_000
        split = si.record_of([(u, shares[x % g]) for x in range(n * g)], g)
        merged = si.record_of([(g * u, sum(shares))] * n)
        assert audit.pooled(split) == audit.pooled(merged)
    assert si.record_of([(u, 1)]).groups_per_coin == 1


def test_a_plan_g_groups_a_coin_agrees_with_the_code_and_costs_no_more_as_g_grows():
    p = {"label": "32B", "blocks": 130_023_424, "gpus": 8, "shard_bytes": 62 * 2 ** 30,
         "weights_bytes": 524_204_113_920, "c_s": 0.057, "xi": 2.638e6, "chip": 743_590.5, "codec_loss": None}
    prices = {"fit": 0.544, "floor": 0.08, "fit_deployed": 1.09, "floor_deployed": 0.16}
    plans = [si.solve(p, prices, 0.01, theta=600.0, line_rule=True, per_coin=g) for g in (1, 2, 4, 8)]
    assert plans[0] == dict(si.solve(p, prices, 0.01, theta=600.0, line_rule=True), per_coin=1)
    cost = [s["passes_gpus"] + s["proving_gpus_fit"] for s in plans]
    assert cost == sorted(cost, reverse=True) and cost[-1] < cost[0]
    for s in plans:
        c = si.coin_check(p, s)
        drawn = s["d"] * s["per_coin"]
        assert drawn == round(drawn) >= 1 and c["draws_per_coin"] == round(drawn) and c["groups"] % s["per_coin"] == 0
        assert s["eps_held"] <= 0.01 * (1 + 1e-12) and s["o"] == pytest.approx(s["d"] * s["r"] / s["k"])
