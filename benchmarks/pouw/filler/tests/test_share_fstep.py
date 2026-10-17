"""share.py's fused-filler credit: the filler warps count only while the measured core step runs, the
pump on a split's other SMs also through the non-GEMM time, and the pick keeps only configurations within LIMIT."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("share", Path(__file__).resolve().parents[1] / "share.py")
S = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = S
_spec.loader.exec_module(S)


class _M:
    cap = 1e15


def _row(b, l, fw, afw, fsm, fused, pump, slow):
    return {"B": b, "L": l, "FW": fw, "NC": 4, "thr": 0, "AFW": afw, "fill_sms": fsm, "lt_mask": 0, "pdl": 1, "fill_share": fused,
            "fill_share_pump": pump, "slowdown": slow}


def test_filler_warps_are_not_credited_in_non_gemm_time():
    f = S.StepFiller({(32, 2048): (0.5, 0.0, 0.9)})
    other = S.DEC_OTHER_MS
    t, macs = f.apply(_M, 32, 2048, 10.0)
    assert t == pytest.approx((10.0 - other) * 0.9 + other)
    assert macs == pytest.approx(0.5 * (10.0 - other) * 0.9 * 1e-3 * _M.cap)


def test_pump_runs_through_non_gemm_time_which_takes_the_split_slowdown():
    f = S.StepFiller({(32, 2048): (0.2, 0.5, 1.04)})
    other = S.DEC_OTHER_MS
    t, macs = f.apply(_M, 32, 2048, 10.0)
    tc, to = (10.0 - other) * 1.04, other * 1.04
    assert t == pytest.approx(tc + to)
    assert macs == pytest.approx(((0.2 + 0.5) * tc + 0.5 * to) * 1e-3 * _M.cap)
    t_fast, _ = S.StepFiller({(32, 2048): (0.2, 0.5, 0.95)}).apply(_M, 32, 2048, 10.0)
    assert t_fast == pytest.approx((10.0 - other) * 0.95 + other)


def test_pick_drops_configs_over_the_limit_and_splits_unless_asked():
    rows = [_row(32, 2048, 8, 4, 0, 0.55, 0.0, 0.91), _row(32, 2048, 8, 4, 108, 0.21, 0.52, 1.005),
            _row(32, 2048, 8, 4, 124, 0.17, 0.59, S.LIMIT + 0.01), _row(32, 2048, 8, 4, 108, 0.23, 0.52, 1.0)]
    whole = S.fstep_pick(rows, split=False)[(32, 2048)]
    assert whole[0][5] == 0 and whole[1] == pytest.approx(0.55)
    split = S.fstep_pick(rows, split=True)[(32, 2048)]
    assert split[0][5] == 108
    assert split[1] == pytest.approx(0.22) and split[2] == pytest.approx(0.52) and split[3] == pytest.approx(1.0025)


def test_attention_chains_are_part_of_the_config():
    four, two = _row(132, 5120, 4, 4, 0, 0.50, 0.0, 1.01), {**_row(132, 5120, 4, 4, 0, 0.60, 0.0, 1.02), "ANC": 2}
    pts = S.fstep_points([four, two])
    assert len(pts) == 2
    k, fused, _, _ = S.fstep_pick([four, two], split=False)[(132, 5120)]
    assert k[10] == 2 and fused == pytest.approx(0.60)


def test_persistent_attention_blocks_are_part_of_the_config():
    blocks, persistent = _row(256, 2048, 0, 6, 28, 0.33, 0.13, 0.99), {**_row(256, 2048, 0, 6, 28, 0.40, 0.13, 1.0), "APS": 1}
    assert len(S.fstep_points([blocks, persistent])) == 2
    k, fused, _, _ = S.fstep_pick([blocks, persistent], split=True)[(256, 2048)]
    assert k[11] == 1 and fused == pytest.approx(0.40)
    assert S.afw_name(6, 2, 1) == "6 NC 2 persistent"


def test_lazy_filler_warps_are_part_of_the_config():
    eager, lazy = {**_row(132, 5120, 0, 8, 68, 0.37, 0.32, 1.03), "APS": 1}, {
        **_row(132, 5120, 0, 8, 68, 0.43, 0.32, 1.03), "APS": 1, "ALZ": 2}
    assert len(S.fstep_points([eager, lazy])) == 2
    k, fused, _, _ = S.fstep_pick([eager, lazy], split=True)[(132, 5120)]
    assert k[12] == 2 and fused == pytest.approx(0.43)
    assert S.afw_name(8, 2, 1, 2) == "8 NC 2 persistent lazy 2"
