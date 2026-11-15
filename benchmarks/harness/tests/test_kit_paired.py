"""The harness kit's paired ratio and its interval against values worked by hand, and overhead's name for it."""

from __future__ import annotations

import pytest
from verity_bench import kit

# ---- the paired ratio --------------------------------------------------------------------------------------------------------


def test_the_paired_ratio_is_the_median_of_the_reps_ratios_rep_by_rep():
    g = kit.paired([3.0, 1.0, 4.0], [2.0, 1.0, 2.0])
    assert g == {"gamma": 0.5, "ratio": 1.5, "n": 3, "ratios": [1.0, 1.5, 2.0], "ci95": None, "half_width": None}
    assert kit.paired([2.0, 1.0], [1.0, 2.0])["ratios"] == [0.5, 2.0]
    assert kit.paired([2.0, 1.0], [1.0, 2.0])["ratio"] == 1.25
    assert kit.paired([0.0], [1.0])["gamma"] == -1.0


def test_the_interval_is_the_medians_order_statistics_at_the_confidence():
    arm, stock = [1.5 + 0.001 * i for i in range(20)], [1.0] * 20
    g = kit.paired(arm[::-1], stock)
    # 20 reps at 95%: P(Binomial(20, 1/2) < 6) = 0.0207 and P(< 7) = 0.0577, so the 6th and the 15th ratios
    assert g["ratio"] == pytest.approx(1.5095) and g["gamma"] == pytest.approx(0.5095) and g["n"] == 20
    assert g["ci95"] == pytest.approx([0.505, 0.514]) and g["half_width"] == pytest.approx(0.0045 / 1.5095)
    # 6 reps: P(< 1) = 1/64, so the 1st and the 6th; 5 reps: P(< 1) = 1/32, above 0.025, so no interval
    six = kit.paired([1.1, 1.6, 1.2, 1.5, 1.3, 1.4], [1.0] * 6)
    assert six["ci95"] == pytest.approx([0.1, 0.6]) and six["half_width"] == pytest.approx(0.25 / 1.35)
    for n in (1, 2, 5):
        few = kit.paired([1.5] * n, [1.0] * n)
        assert few["ci95"] is None and few["half_width"] is None and few["n"] == n
    # at 99%: P(< 4) = 0.0013 and P(< 5) = 0.0059, so the 4th and the 17th; at 80%: P(< 7) = 0.0577 and P(< 8) = 0.1316, so
    # the 7th and the 14th
    wide, narrow = kit.paired(arm, stock, confidence=0.99), kit.paired(arm, stock, 0.8)
    assert wide["ci99"] == pytest.approx([0.503, 0.516]) and "ci95" not in wide
    assert narrow["ci80"] == pytest.approx([0.506, 0.513]) and narrow["ratio"] == g["ratio"]
    # 5 reps bound the median at 50%: P(< 2) = 0.1875 and P(< 3) = 0.5, so the 2nd and the 4th
    assert kit.paired([1.5, 1.1, 1.4, 1.2, 1.3], [1.0] * 5, 0.5)["ci50"] == pytest.approx([0.2, 0.4])


def test_an_arm_whose_reps_cost_nothing_has_its_interval_and_no_half_width():
    """A median ratio of 0 has an interval like any other, and no half-width as a share of it."""

    free = kit.paired([0.0] * 6, [1.0] * 6)
    assert free == {"gamma": -1.0, "ratio": 0.0, "n": 6, "ratios": [0.0] * 6, "ci95": [-1.0, -1.0], "half_width": None}
    # most reps free and one not: the median ratio is still 0, and the interval reaches the one that cost
    mostly = kit.paired([0.0, 0.0, 3.0, 0.0, 0.0, 0.0], [1.0, 1.0, 2.0, 1.0, 1.0, 1.0])
    assert (mostly["ratio"], mostly["ci95"], mostly["half_width"]) == (0.0, [-1.0, 0.5], None)
    # one free rep among six leaves the median above 0, and its half-width a share of it
    one = kit.paired([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0] * 6)
    assert (one["ratio"], one["ci95"], one["half_width"]) == (1.0, [-1.0, 0.0], 1.0)


def test_paired_refuses_reps_that_dont_pair_a_stock_rep_of_no_cost_and_a_confidence_out_of_range():
    for arm, stock in (([1.0], [1.0, 2.0]), ([], [])):
        with pytest.raises(ValueError, match="one for one"):
            kit.paired(arm, stock)
    for zero in (0.0, -1.0):
        with pytest.raises(ValueError, match="every stock rep above zero"):
            kit.paired([1.0, 1.0], [1.0, zero])
    for confidence in (0, 1, 95, -0.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            kit.paired([1.0], [1.0], confidence)


def test_overhead_paired_is_the_kits_by_overheads_name_until_serving_cost_names_the_kit():
    from verity_bench import overhead

    arm, stock = [2.0, 2.1, 1.9] * 7, [1.0] * 21
    assert overhead.paired(arm, stock) == kit.paired(arm, stock)
    assert overhead.paired.__module__ == "verity_bench.overhead"
