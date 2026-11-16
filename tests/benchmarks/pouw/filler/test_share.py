"""share.py's lookups and the fillers it reads from a run's lines: the model's counts (Llama-3.1-8B's shape), a
measured curve read log-log, a filler's share and decode slowdown at a batch (`Filler`) and at a (batch, context)
point (`PointFiller`), the filler of a fill.sh run, the fillers of a dfill.sh run with its adaptive split, and the
share of tensor peak the guarantee leaves to undeclared MMAs (`hidden`)."""

from __future__ import annotations

import pytest
import share as S

SPLIT = "dfree-split-w8-nc4-t8"


def _fill(shape, sms, b, ctx, share, slow):
    return {"shape": shape, "fill_sms": sms, "B": b, "L": ctx, "window": {"fill_share": share}, "slowdown": slow}


def _dfill(sms, b, ctx, share, slow, tuned=None, shape=SPLIT, place="split"):
    r = {**_fill(shape, sms, b, ctx, share, slow), "place": place}
    return {**r, "tuned": {"fill_share": tuned[0], "slowdown": tuned[1]}} if tuned else r


def test_the_counts_are_llama_3_1_8bs():
    # hidden 4,096; 32 layers of 32 heads of 128, 8 of them KV heads; an MLP of 14,336; 128,256 tokens
    assert S.LIN == {"qkv": (4096, 6144), "o": (4096, 4096), "gate_up": (4096, 28672), "down": (14336, 4096)}
    assert S.LIN_TOK == 32 * 218_103_808
    assert S.LM_TOK == 525_336_576
    assert S.ATT_TOK_L == 262_144
    assert S.KV_TOK == 65_536  # 64 KiB of int8 KV a token
    assert S.POOL_TOK == 1_220_703  # 80 GB of them
    # a causal prompt's attention: its i-th token reads i positions
    assert S.causal(1) == S.ATT_TOK_L and S.causal(4) == sum(S.ATT_TOK_L * i for i in range(1, 5))


def test_a_is_the_frozen_vectors_and_the_ladder_rises_from_it_to_one():
    ladder = [a for _, a in S.A_LADDER]
    assert ladder[0] == S.A_V99 and ladder[-1] == 1.0 and ladder == sorted(ladder)
    assert ladder[0] == pytest.approx(0.2068, abs=5e-5)  # 0.2424 u a MAC over 1.1722 u a MAC at tensor peak
    assert dict(S.A_LADDER)["product routes only, A = 256/R*"] == pytest.approx(0.8531, abs=5e-5)


def test_a_curve_is_read_log_log_between_its_points_and_in_proportion_past_the_last():
    xs, ys = [2, 8, 32], [3.0, 6.0, 96.0]
    assert S.interp(xs, ys, 1) == 3.0 and S.interp(xs, ys, 2) == 3.0  # at or under the first point: its value
    assert S.interp(xs, ys, 4) == pytest.approx(3.0 * 2 ** 0.5)  # halfway in log x from (2, 3) to (8, 6)
    assert S.interp(xs, ys, 8) == pytest.approx(6.0)
    assert S.interp(xs, ys, 16) == pytest.approx(24.0)  # halfway from (8, 6) to (32, 96)
    assert S.interp(xs, ys, 32) == pytest.approx(96.0)
    assert S.interp(xs, ys, 64) == 192.0  # past the last point: the last value times x over its x


def test_a_filler_is_read_in_log_batch_between_its_points_and_flat_outside_them():
    f = S.Filler([(64, 0.5, 1.08), (4, 0.2, 1.02), (16, 0.3, 1.04)])  # in any order
    assert f.at(1) == (0.2, 1.02) and f.at(4) == (0.2, 1.02)
    assert f.at(8) == pytest.approx((0.25, 1.03))  # halfway in log batch from 4 to 16
    assert f.at(16) == pytest.approx((0.3, 1.04))
    assert f.at(32) == pytest.approx((0.4, 1.06))  # and from 16 to 64
    assert f.at(64) == pytest.approx((0.5, 1.08)) and f.at(256) == (0.5, 1.08)
    assert f.dsms is None
    assert S.Filler([(1, 0.1, 1.0), (4, 0.3, 1.04)]).at(2) == pytest.approx((0.2, 1.02))
    one = S.Filler([(8, 0.222, 1.047)], dsms=128)  # one point: that point at every batch
    assert one.at(1) == one.at(8) == one.at(32) == (0.222, 1.047) and one.dsms == 128


def test_a_point_filler_reads_the_smallest_batch_not_under_the_steps_and_the_longest_context_not_over_its():
    f = S.PointFiller({(32, 2048): (0.5, 1.01), (32, 8192): (0.4, 1.02), (32, 32768): (0.3, 1.03),
                       (128, 2048): (0.2, 1.04)})
    assert f.at(32, 8192) == (0.4, 1.02)  # a measured point is its own
    assert f.at(16, 9000) == (0.4, 1.02)  # under the smallest batch, between two contexts
    assert f.at(32, 40000) == (0.3, 1.03)  # past the longest context
    assert f.at(32, 1024) == (0.5, 1.01)  # under the shortest context: the shortest
    assert f.at(33, 8192) == f.at(128, 2048) == f.at(512, 99) == (0.2, 1.04)  # the next batch up; past the largest
    assert f.at(32) == (0.5, 1.01)  # no context given: 2,048


def test_undeclared_mmas_get_what_the_declared_floor_leaves_of_the_window():
    assert S.hidden(0.5, 0.8) == pytest.approx(0.63125)  # 0.99 x 0.5 + 0.8 u = 1
    assert S.hidden(0.1, 0.2) == 1.0  # never more than the whole peak


def test_the_measured_filler_is_the_big_chain_at_l_2048_by_batch_over_its_commitment_factor():
    rows = [_fill("big", 60, 32, 2048, 0.218, 1.03), _fill("big", 60, 1, 2048, 0.109, 1.01),
            _fill("big", 60, 32, 8192, 0.9, 1.5), _fill("small", 60, 1, 2048, 0.8, 1.4),
            _fill("big", 92, 1, 2048, 0.327, 1.06), _fill("small", 124, 1, 2048, 0.7, 1.3)]
    f = S.measured_fillers(rows)
    assert list(f) == ["60 SMs", "92 SMs"]
    assert f["60 SMs"].at(1) == pytest.approx((0.1, 1.01)) and f["60 SMs"].at(32) == pytest.approx((0.2, 1.03))
    assert f["92 SMs"].at(8) == pytest.approx((0.3, 1.06))
    assert list(S.measured_fillers(rows, "small")) == ["60 SMs", "124 SMs"]


def test_a_dfill_point_is_the_mean_of_its_runs_and_the_tuned_arm_reads_only_the_runs_that_have_one():
    rows = [_dfill(60, 32, 2048, 0.20, 1.02, tuned=(0.30, 1.00)), _dfill(60, 32, 2048, 0.24, 1.04),
            _dfill(60, 32, 8192, 0.10, 1.01), _dfill(92, 32, 2048, 0.40, 1.10, tuned=(0.38, 1.04))]
    s60, s92 = f"{SPLIT} 60 SMs", f"{SPLIT} 92 SMs"
    plain = S.dfill_points(rows, False)
    assert sorted(plain) == [(s60, 32, 2048), (s60, 32, 8192), (s92, 32, 2048)]
    assert plain[(s60, 32, 2048)] == pytest.approx((0.22, 1.03))
    assert plain[(s60, 32, 8192)] == (0.10, 1.01) and plain[(s92, 32, 2048)] == (0.40, 1.10)
    assert S.dfill_points(rows, True) == {(s60, 32, 2048): (0.30, 1.00), (s92, 32, 2048): (0.38, 1.04)}
    f = S.dfill_fillers(rows)
    adaptive = "dfree adaptive ≤ 5%"
    assert list(f) == [s60, s92, adaptive, s60 + " tuned", s92 + " tuned", adaptive + " tuned"]
    assert f[s60].at(32, 2048) == pytest.approx((0.22, 1.03))
    assert f[s60].at(32, 8192) == (0.10, 1.01) and f[s92].at(32, 2048) == (0.40, 1.10)
    assert f[s60 + " tuned"].at(32, 2048) == (0.30, 1.00) and f[s92 + " tuned"].at(32, 2048) == (0.38, 1.04)
    # as tuned for the whole GPU the 92-SM split is over the limit, so the adaptive split is the 60-SM one
    assert f[adaptive].at(32, 2048) == pytest.approx((0.22, 1.03)) and f[adaptive].at(32, 8192) == (0.10, 1.01)
    assert f[adaptive + " tuned"].at(32, 2048) == (0.38, 1.04)
    assert S.dfill_fillers([]) == {}


def test_the_adaptive_split_is_the_largest_share_within_the_limit_among_the_whole_fillers_splits():
    rows = [_dfill(60, 32, 2048, 0.20, 1.02), _dfill(92, 32, 2048, 0.30, S.LIMIT),
            _dfill(124, 32, 2048, 0.40, 1.06),  # over the limit
            _dfill(188, 32, 2048, 0.50, 1.00, shape="dfree-co-w8-nc4-t8", place="co"),  # on every SM: no split
            _dfill(92, 32, 2048, 0.60, 1.00, shape=SPLIT + "-lin"),  # the filler beside one part of the step
            _dfill(92, 32, 2048, 0.70, 1.00, shape=SPLIT + "-att"),
            _dfill(60, 128, 8192, 0.25, 1.07)]  # no split within the limit here: decode keeps the whole GPU
    assert S.adaptive_pick(rows, False) == {(32, 2048): (92, 0.30, S.LIMIT), (128, 8192): (0, 0.0, 1.0)}
    tuned = [_dfill(60, 32, 2048, 0.20, 1.02, tuned=(0.30, 1.00)), _dfill(92, 32, 2048, 0.40, 1.10, tuned=(0.38, 1.04)),
             _dfill(60, 32, 8192, 0.10, 1.01)]
    assert S.adaptive_pick(tuned, True) == {(32, 2048): (92, 0.38, 1.04)}
