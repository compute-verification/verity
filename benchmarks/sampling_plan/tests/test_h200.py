"""The H200 plan (`h200.py`): the block's units are circuits' partition, every execution phase is a uniform draw its tables
divide within the statement cap, its K0 is IntegrityPeriodLog's, commitments are priced beside the proving and not in
it, both alternative cuts are priced, the frontier is monotone and an execution phase of `window_blocks` blocks proves within its
GPU time."""

from __future__ import annotations

from dataclasses import replace
from math import ceil, log

import pytest

import h200 as H
import transformer as X

M = H.Shape()


@pytest.fixture(scope="module")
def plan() -> dict:
    return H.report(M, H.SHARE)


def test_the_units_are_circuits_partition():
    """Per request a layer and a position, 132,498 units and 256 (p + 1) for attention's keys; per generated position
    the head's 256 coordinates, its two gathers, norm, quantizer and argmax; per position the embedding."""
    ts = H.templates(M)
    R, L, T, Gn = M.batch, M.layers, M.positions, M.generate
    per_layer = 132_498 * T + 256 * T * (T + 1) // 2
    assert sum(t.units for t in ts) == R * (L * per_layer + T + Gn * (M.vocab + 2 + 1 + 1 + 1))
    assert sum(t.units for t in H.templates(M, pad=True)) == sum(t.units for t in ts)


def test_windows_draw_k_split_evenly_under_the_cap(plan):
    k = ceil(log(len(plan["windows"]) / H.DELTAS[0]) / H.SHARE)
    for w in plan["windows"]:
        cap = X._cap(w["proved_ands"])
        assert w["k"] >= min(k, w["units"]) and w["k"] % w["tables"] == 0 and w["k"] // w["tables"] <= cap
        assert w["tables"] == ceil(min(k, w["units"]) / cap)


def test_k0_is_integrity_period_logs(plan):
    S = len(plan["windows"])
    for w in plan["windows"]:
        for d in H.DELTAS:
            assert w["K0"][str(d)] == min(ceil(log(S / d) * w["units"] / w["k"]), w["units"] - w["k"] + 1)
            assert w["K0"][str(d)] <= H.SHARE * w["units"] * log(S / d) / log(S / H.DELTAS[0]) + 1


def test_commitments_are_beside_the_proving(plan):
    c = plan["cost"]
    assert c["proving"] == pytest.approx(c["statements"] + c["drawn"] + c["replay"])
    assert c["commit"] > 0 and c["commit"] == pytest.approx(H.commit_cost(H.templates(M), H.Rates.fitted()))


def test_both_alternative_cuts_are_priced(plan):
    alt = plan["alternatives"]
    assert alt["tiles"]["windows"] == len(plan["windows"]) and alt["tiles"]["proving"] > plan["cost"]["proving"]
    assert alt["padded"]["windows"] == 14 and alt["padded"]["proving"] < plan["cost"]["proving"]


def test_the_frontier_is_monotone_and_the_window_fits(plan):
    costs = [f["proving"] for f in plan["frontier"]]
    assert costs == sorted(costs)
    B = plan["window_blocks"]
    assert plan["cost"]["proving"] / B <= M.block_s < plan["cost"]["proving"] / (B - 1)


def test_a_longer_step_lengthens_the_block_not_the_units():
    slow = replace(M, step_s=2 * M.step_s)
    assert slow.block_s == 2 * M.block_s
    assert [t.units for t in H.templates(slow)] == [t.units for t in H.templates(M)]
