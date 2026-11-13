"""summarize.py --arms fp32-beside: the time price of an FP32 add in tensor units, p = R_t,alone / R_f,mixed, judged in
arms C and D against 32 x (1 - 0.005), with R_t,alone from arm A at the point's clock and class, R_f,mixed over the
FP32 warps' adding share of the host interval, and the host stamps' resolution leaving a near point undecided."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import summarize as S

SECS = 1.2
R_T = 132 * 4096 * 1830e6


def bpoint(arm: str, *, x: float = 0, p: float | None = None, share: float = 0.0, clock: str = "1980",
           cls: str = "uniform", host_s: float = SECS) -> dict:
    """A duty.cu fp32-beside line: arm A moves R_T MACs a second over ``host_s``; another arm adds at R_T / ``p`` a
    second over ``share`` of it."""
    adds = R_T / p * host_s * share if p else 0.0
    return {"clock": clock, "arm": arm, "class": cls, "x": x, "host_t0": 1000.0, "host_t1": 1000.0 + host_s,
            "macs": R_T * host_s if arm == "A" else 0.0, "adds": adds, "fp32_share": share, "clock64_mhz": 1900.0,
            "fp32_busy_mhz": 1980.0 if share else 0.0, "macs_per_smclk": 0.0, "adds_per_smclk": 0.0,
            "tensor_mhz": 1830.0 if arm in "ACD" else 0.0, "fp32_mhz": 1980.0 if arm in "BD" else 0.0,
            "both_mhz": 1900.0 if arm == "C" else 0.0}


def run(tmp_path: Path, *pts: dict, smi: str = "", name: str = "out") -> tuple[int, dict]:
    out = tmp_path / name
    out.mkdir()
    (out / "beside.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pts))
    if smi:
        (out / "smi.csv").write_text(smi)
    rc = S.main([str(out), "--arms", "fp32-beside", "--json", str(tmp_path / f"{name}.json")])
    return rc, json.loads((tmp_path / f"{name}.json").read_text())


def test_the_threshold_is_w1s_add_price_less_tt_outs_margin() -> None:
    assert S.P_MIN == pytest.approx(31.84)


def test_p_is_the_tensor_rate_alone_over_the_adds_rate_while_adding(tmp_path: Path) -> None:
    """A C point adding at R_T / 33.5 a second for a quarter of its interval has p = 33.5, though its adds over the
    whole interval are a quarter of that rate; the smi samples inside a point's interval are its smi readings."""
    smi = ("1970/01/01 00:16:40.500, 1830, 1980, 690.5, 700.00, 61, 0x0\n"
           "1970/01/01 00:16:40.700, 1850, 1980, 680.5, 700.00, 61, 0x0\n"
           "1970/01/01 00:16:50.000, 1980, 1980, 100.0, 700.00, 61, 0x0\n")
    rc, v = run(tmp_path, bpoint("A"), bpoint("C", x=128, p=33.5, share=0.25), bpoint("D", x=10, p=32.5, share=0.5),
                smi=smi)
    assert (rc, v["verdict"], v["judged"], v["failing"], v["undecided"]) == (0, "PASS", 2, 0, 0)
    c = next(r for r in v["rows"] if r["arm"] == "C")
    assert c["p"] == pytest.approx(33.5) and c["p_lo"] < c["p"] < c["p_hi"]
    assert c["r_f_mixed"] == pytest.approx(R_T / 33.5) and c["adds_per_s"] == pytest.approx(R_T / 33.5 * 0.25)
    assert c["r_t_alone"] == pytest.approx(R_T)
    assert (c["smi_mhz_mean"], c["smi_mhz_max"]) == (1840.0, 1850.0)
    assert v["lowest"]["arm"] == "D" and v["lowest"]["p"] == pytest.approx(32.5)


def test_fp32_at_a_higher_clock_than_the_tensor_fails(tmp_path: Path) -> None:
    """D's FP32 phases at 1980 MHz beside tensor phases at 1830 price an add at 32 x 1830 / 1980 = 29.58."""
    rc, v = run(tmp_path, bpoint("A"), bpoint("C", x=8, p=33.5, share=1.0),
                bpoint("D", x=100, p=32 * 1830 / 1980, share=0.5))
    assert (rc, v["verdict"], v["failing"]) == (1, "FAIL", 1)
    assert (v["lowest"]["arm"], v["lowest"]["x"]) == ("D", 100) and v["lowest"]["p"] == pytest.approx(29.5758, abs=1e-3)


def test_a_point_within_the_stamps_of_the_threshold_is_undecided(tmp_path: Path) -> None:
    """Two stamps of a millisecond over 1.2 s move p by about 0.17%: 0.05% either side of the threshold is undecided."""
    for f in (1.0005, 0.9995):
        rc, v = run(tmp_path, bpoint("A"), bpoint("C", x=32, p=S.P_MIN * f, share=1.0), name=str(f))
        assert (rc, v["verdict"], v["failing"], v["undecided"]) == (1, "UNDECIDED", 0, 1)


def test_the_tensor_rate_is_arm_as_at_the_points_clock_and_class(tmp_path: Path) -> None:
    """The faster arm A point at another clock prices nothing at this one; a C point with no arm A at its class is
    undecided; arm B's p is given and not judged."""
    fast = bpoint("A", clock="default")
    fast["macs"] *= 2
    rc, v = run(tmp_path, bpoint("A"), fast, bpoint("B", p=20.0, share=0.99), bpoint("C", x=128, p=33.5, share=0.25),
                bpoint("C", x=128, p=33.5, share=0.25, cls="zero"))
    c = next(r for r in v["rows"] if r["arm"] == "C" and r["class"] == "uniform")
    b = next(r for r in v["rows"] if r["arm"] == "B")
    assert c["p"] == pytest.approx(33.5) and b["p"] == pytest.approx(20.0) and not b["judged"]
    assert (rc, v["verdict"], v["failing"], v["undecided"]) == (1, "UNDECIDED", 0, 1)


def test_no_beside_points_is_no_verdict(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    assert S.main([str(tmp_path / "out"), "--arms", "fp32-beside"]) == 2
