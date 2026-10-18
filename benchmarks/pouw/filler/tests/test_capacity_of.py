"""capacity_of.py: the certified share σ of a measured window, from a scored run's entries and from dcap's lfill lines."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import capacity_of as CO  # noqa: E402

PI = CO.E.machine("VF3r-16")  # the vector of record the readings below are taken on
GHZ = 2.092e9


def rec(pf_lane=0, pump_share=0.584, lane_rate=0.0, step_ms=7.0):
    """an lfill line at B 1 on the leader's split: the pump at `pump_share` of its 140 SMs' tensor peak, the lane filler
    at `lane_rate` warp instructions a SMSP-cycle on the step's 48"""
    pump = pump_share * 1024 * 140 * GHZ
    arm = {"step_ms": step_ms, "mhz": 2092.0, "slowdown": 1.0}
    return {"mode": "lfill", "B": 1, "L": 2048, "step_sms": 48, "fill_sms": 140, "pf_lane": pf_lane, "T": 128, "CH": 8,
            "blk": 2, "step_macs": 8.04e9, "rec_bad_pump": 0, "rec_bad_lane": 0,
            "pump": dict(arm, mac_per_s=pump),
            "lane": dict(arm, pump_mac_per_s=pump, lane_instr_per_s=lane_rate * 4 * 48 * GHZ, watt=500.0)}


def rows(r):
    return {x["arm"]: x for x in CO.lfill_rows(PI, [r])}


def test_pump_alone_matches_the_leaders_reading():
    # the leader's decode B 1 entry (filler share 0.435 of the GPU at 2,092 MHz) reads σ 0.371 at f_max
    got = rows(rec())
    assert got["pump"]["sigma"] == pytest.approx(got["lane"]["sigma"])
    assert 0.35 < got["pump"]["sigma"] < 0.39
    assert got["pump"]["sigma_run_clock"] > got["pump"]["sigma"]


def test_lane_work_raises_sigma():
    base = rows(rec())["pump"]["sigma"]
    lanes = rows(rec(pf_lane=8))["pump"]["sigma"]
    both = rows(rec(pf_lane=8, lane_rate=0.5))["lane"]["sigma"]
    assert base < lanes < both


def dfill(place="co", pump_share=0.58, fill_ms=7.0):
    """a dfill line at B 1: the pump at `pump_share` of the whole GPU's tensor peak beside the step (on every SM, co; or
    on 140 of 188, split)"""
    sms = 188 if place == "co" else 140
    arm = {"ms": fill_ms, "mhz": 2092.0}
    return {"mode": "dfill", "place": place, "B": 1, "L": 2048, "NC": 2, "S": 256, "step_sms": 188 if place == "co" else 48,
            "fill_sms": sms, "step_macs": 8.04e9, "rec_bad": 0, "slowdown": 1.0,
            "window": {"fill_mac_per_s": pump_share * 1024 * 188 * GHZ},
            "arms": {"co_fill" if place == "co" else "split_fill": arm}}


def test_dfill_reads_split_and_co_alike():
    co, split = CO.dfill_rows(PI, [dfill("co"), dfill("split")])
    assert co["sigma"] == pytest.approx(split["sigma"])
    assert 0.47 < co["sigma"] < co["sigma_run_clock"] < 0.60
    slow = CO.dfill_rows(PI, [dfill("co", fill_ms=70.0)])[0]
    assert slow["sigma"] < co["sigma"]


def fstep(pump_share=0.58, fused_ms=7.0, fused0_ms=7.0):
    """an fstep co line at B 1: the pump at `pump_share` of the whole GPU's tensor peak beside the flin step"""
    cap = 1024 * 188 * GHZ
    return {"mode": "fstep", "place": "co", "B": 1, "L": 2048, "M": 32, "step_sms": 188, "fill_sms": 188, "mhz": 2092.0,
            "pump": {"share": pump_share, "alone_share": 0.57, "rec_bad": 0}, "fill_share_lin": 0.0, "fill_share_att": 0.0,
            "step_share_fused": 8.04e9 / (fused_ms * 1e-3) / cap, "slowdown": fused_ms / 6.0, "fused_ms": fused_ms,
            "fused0_ms": fused0_ms, "fused0_vs_base": fused0_ms / 6.0, "lin": {"rec_bad": 0}, "att": {"rec_bad": 0},
            "out_mismatches": 0, "lm_mismatches": 0}


def test_fstep_reads_as_dfill_co():
    (f,) = CO.fstep_rows(PI, [fstep()])
    (d,) = CO.dfill_rows(PI, [dfill("co")])
    assert f["sigma"] == pytest.approx(d["sigma"])
    assert f["sigma_run_clock"] == pytest.approx(d["sigma_run_clock"])
    assert f["co_cost"] == 1.0 and f["rec_bad"] == 0 and f["out_bad"] == 0


def test_cli_reads_lfill_lines(tmp_path, capsys):
    p = tmp_path / "lfill.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in (rec(), rec(pf_lane=8, lane_rate=0.5))) + "\n")
    assert CO.main(["--lfill", str(p), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [(r["pf_lane"], r["arm"]) for r in out["rows"]] == [(0, "pump"), (0, "lane"), (8, "pump"), (8, "lane")]
    assert out["rows"][3]["sigma"] > out["rows"][0]["sigma"]
