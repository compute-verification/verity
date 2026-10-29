"""counter_gtail.py, the searches of `G_tail`'s figures: the per-pair constants `coinTail_le` rests on, `count_tail`,
the least slack, `TailAt`'s arithmetic and `GapFixedAt`'s closed form."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))
import counter_gtail as gt
import counter_pad as cp


@pytest.fixture(scope="module", autouse=True)
def head():
    cp.use_head("d5263f864b")


def test_lanes_are_injective():
    """A column's even and odd words pin its three coin lanes: `2^−21`, under the claimed `2^−7`."""
    assert gt.lane_column()["fiber"] == 1


def test_g_fiber_meets_its_bound():
    """At T 512 × 512 some pair's tag differences are `±(2, −2, 0, 0)` in every row: `7^−8` a column block exactly."""
    f = gt.pair_fibers(512)
    assert f["holds"] and f["worst"]["fiber"] == 343
    assert f["p_block"] == pytest.approx(7.0 ** -8)
    assert gt.pair_fibers(16)["pairs"] == 0


def test_a1_first_digit():
    assert gt.a1_rows(1024)["p_row_max"] <= 0.5


def test_count_tail_small_spaces():
    assert gt.count_tail_search(40, 0)["holds"]


@pytest.mark.parametrize("mu, budget", [(0.5, -42.0), (3.0, -42.0), (1e4, -44.0), (2.0 ** 23, -43.6)])
def test_least_r(mu, budget):
    r = gt.least_r(mu, budget)
    assert gt._log2_tail(mu, r) <= budget
    assert r == 1 or gt._log2_tail(mu, r - 1) > budget


def test_tail_at():
    t = gt.tail_at()
    assert t["holds"] and t["log2_tableTail"] == pytest.approx(-41.0)
    assert t["mem_max_TiB"]["coinTail_2^-42"] == pytest.approx(12.0)


def test_closed_form_bounds_the_calls():
    for name, (t, m, lim32, _) in gt.FIXED_AT.items():
        closed = gt.closed_fixed(cp.pad_rows(m), 32)
        for n, k in ((4096, 4096), (4097, 4110), (4104, 4200), (65536, 2 ** 20)):
            assert gt.gap_fixed(gt.layout(t, m, n, k))["share"] <= closed <= lim32


def test_slack_needed_is_within_coin_slack():
    lay = gt.layout(512, 64, 4096, 4096)
    s = gt.slack_needed(lay, gt.P_WITHIN, gt.P_AB, gt.P_ACROSS)
    assert s["holds"] and 0 < s["slack_share"] < float(gt.COIN_SLACK)
    assert math.isclose(s["coinSlack_share"], 8e-4)
