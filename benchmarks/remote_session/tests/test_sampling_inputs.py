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
            assert s["l_max_s"] == pytest.approx(s["k"] / s["r"] if flight == 1 else 2 * s["k"] / s["r"] + p["c_s"])
            assert flight == 2 or s["k"] / s["r"] >= si.least_period(p, s["k"]) * (1 - 1e-12)
        for s in (line, shared):
            assert 2 * (s["l_max_s"] - p["c_s"]) * p["xi"] / p["blocks"] <= 0.01 * (1 + 1e-12)
        assert line["d"] >= 1
        if flight == 2:
            assert shared["d"] < 1 and shared["o"] < line["o"]
        cost = {id(s): s["passes_gpus"] + s["proving_gpus_fit"] for s in (line, shared, loose)}
        assert cost[id(shared)] <= cost[id(line)]
        assert cost[id(loose)] <= cost[id(line)] and loose["rebuild_held"] > line["rebuild_held"]


def test_the_code_reads_one_coin_over_g_groups_as_the_merged_window_the_model_drew():
    """`audit.pooled` on each group's receipts and draws naming `groups_per_coin` g is the model's reading of shared
    draws, one window of g groups' receipts and the coin's samples, however the coin's samples fall in its groups."""
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


def test_at_one_group_in_flight_a_stretch_ends_at_the_next_coin():
    """A root before the next coin, as a record naming one window in flight has it: the line's stretch is the coin's
    period, whatever the root's latency."""
    seen = [{"released_ns": 100 * g, "passes": 1, "arrivals_ns": [100 * g + a]} for g, a in enumerate((40, 95, 60, 70))]
    got = si.span_line(si.groups_of(seen), 0, 3, wrong=15.0, blocks=10, per_ns=0.01, chip=0.5)
    assert si.groups_of(seen).L == [100, 100, 100, 70] and got["l_max_ns"] == 100 and got["span_ns"] == 200


@pytest.mark.parametrize("point", ["32B", "8B"])
def test_one_group_in_flight_holds_the_same_line_for_fewer_passes_and_the_code_reads_it(point):
    """At one group in flight the line rule's L_max is two in flight's (L_max − c ≤ τ), its plan at each g runs on the
    chain (k/r ≥ k·t_pass + c), `coin_check` reads it as the code does, and it costs less than two in flight's."""
    p = {"label": "32B", "blocks": 130_023_424, "gpus": 8, "shard_bytes": 62 * 2 ** 30,
         "weights_bytes": 524_204_113_920, "c_s": 0.057, "xi": 2.638e6, "chip": 743_590.5, "codec_loss": None}
    if point == "8B":
        p |= {"blocks": 8 * 2_359_296, "shard_bytes": 9_663_676_416, "weights_bytes": 8 * (6.98e9 + 2 * 1.05e9),
              "c_s": 0.072 - 2 / 51.7}
    prices = {"fit": 0.544, "floor": 0.08, "fit_deployed": 1.09, "floor_deployed": 0.16}
    room = si.one_flight_room(p, 0.01)
    assert room["longest_period_s"] == pytest.approx(room["tau_s"] + p["c_s"]) and room["k_max"] >= 1
    for g in (1, 2, 4, 8):
        two, one = (si.solve(p, prices, 0.01, theta=600.0, line_rule=True, flight=f, per_coin=g) for f in (2, 1))
        assert one["flight"] == 1 and one["chain"] and one["l_max_s"] == pytest.approx(one["k"] / one["r"])
        assert one["l_max_s"] <= room["longest_period_s"] * (1 + 1e-12)
        assert two["l_max_s"] <= room["longest_period_s"] * (1 + 1e-12)
        si.coin_check(p, one)
        assert one["passes_gpus"] + one["proving_gpus_fit"] < two["passes_gpus"] + two["proving_gpus_fit"]
    slow = dict(p, c_s=1.0)
    s = si.solve(slow, prices, 0.01, theta=600.0, line_rule=True, flight=1)
    assert s["chain"] and s["k"] / s["r"] >= si.least_period(slow, s["k"]) > 1


def test_a_point_s_own_c_at_one_in_flight_moves_only_the_one_flight_plan():
    """`c1_s` (`--one-flight-c`) is the chain's and the line rule's c at one group in flight; two in flight's plan
    still reads the recorded c."""
    p = {"blocks": 8 * 2_359_296, "gpus": 8, "shard_bytes": 9_663_676_416, "weights_bytes": 8 * (6.98e9 + 2 * 1.05e9),
         "c_s": 0.0333, "xi": 2.638e6, "chip": 743_590.5, "codec_loss": None}
    prices = {"fit": 0.544, "floor": 0.08, "fit_deployed": 1.09, "floor_deployed": 0.16}
    own = dict(p, c1_s=0.0012)
    assert si.c_at(p, 1) == si.c_at(p, 2) == si.c_at(own, 2) == 0.0333 and si.c_at(own, 1) == 0.0012
    assert si.solve(own, prices, 0.01, theta=600.0, line_rule=True) == si.solve(p, prices, 0.01, theta=600.0,
                                                                                 line_rule=True)
    for g in (1, 4):
        a, b = (si.solve(x, prices, 0.01, theta=600.0, line_rule=True, flight=1, per_coin=g) for x in (p, own))
        assert b["k"] / b["r"] <= si.one_flight_room(own, 0.01)["longest_period_s"] * (1 + 1e-12)
        assert b["least_period_s"] == pytest.approx(si.pass_s(own) + 0.0012) and b["chain"]
        assert 2 * (b["l_max_s"] - 0.0012) * own["xi"] / own["blocks"] <= 0.01 * (1 + 1e-12)
        assert b["l_max_s"] < a["l_max_s"] and b["r"] > a["r"]
        si.coin_check(own, b)


def test_one_group_in_flight_has_no_plan_when_a_pass_outlasts_the_line_rule():
    p = {"blocks": 1_000_000, "gpus": 1, "shard_bytes": 1.539e12 * 0.1, "weights_bytes": 4e9, "c_s": 0.01,
         "xi": 2.0e6, "chip": 0.0, "codec_loss": None}
    prices = {"fit": 0.5, "floor": 0.08, "fit_deployed": 1.0, "floor_deployed": 0.16}
    room = si.one_flight_room(p, 0.01)
    assert room["tau_s"] < si.pass_s(p) and room["k_max"] == 0
    assert si.solve(p, prices, 0.01, theta=600.0, line_rule=True, flight=1) is None
    assert si.infeasible({"one_flight": room}).startswith("infeasible: k/r ≤ τ + c")
