"""summarize.py --arms fp32-beside: the time price of an FP32 add in tensor units, p = R_t,alone / R_f,mixed, tests each
reading of the price (per clock 32, the sheet's clocks 32 x 1830 / 1980, a cautious base clock over 1980) at
(1 - 0.005) of it in arms C and D; R_t,alone is arm A's at the point's setting and class, R_f,mixed is over the FP32
warps' adding share of the host interval, an undersaturated arm B or the host stamps' resolution leaves a reading
undecided, and a point's clock is the one it held."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import summarize as S

SECS = 1.2
R_T = 132 * 4096 * 1830e6
IDEAL = 128 * 64 / 67
SHEET = 32 * 1830 / 1980


def bpoint(arm: str, *, x: float = 0, p: float | None = None, share: float = 0.0, clock: str = "1980",
           cls: str = "uniform", host_s: float = SECS, adds_per_smclk: float | None = None) -> dict:
    """A duty.cu fp32-beside line: arm A moves R_T MACs a second over ``host_s``; another arm adds at R_T / ``p`` a
    second over ``share`` of it; arm B adds at the loop's ideal a SM-clock unless ``adds_per_smclk`` says otherwise."""
    adds = R_T / p * host_s * share if p else 0.0
    return {"clock": clock, "arm": arm, "class": cls, "x": x, "host_t0": 1000.0, "host_t1": 1000.0 + host_s,
            "macs": R_T * host_s if arm == "A" else 0.0, "adds": adds, "fp32_share": share, "clock64_mhz": 1830.0,
            "fp32_busy_mhz": 1830.0 if share else 0.0, "macs_per_smclk": 0.0,
            "adds_per_smclk": adds_per_smclk if adds_per_smclk is not None else IDEAL if arm == "B" else 0.0,
            "tensor_mhz": 1830.0 if arm in "ACD" else 0.0, "fp32_mhz": 1830.0 if arm in "BD" else 0.0,
            "both_mhz": 1830.0 if arm == "C" else 0.0}


def run(tmp_path: Path, *pts: dict, smi: str = "", name: str = "out") -> tuple[int, dict]:
    out = tmp_path / name
    out.mkdir()
    (out / "beside.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pts))
    if smi:
        (out / "smi.csv").write_text(smi)
    rc = S.main([str(out), "--arms", "fp32-beside", "--json", str(tmp_path / f"{name}.json")])
    return rc, json.loads((tmp_path / f"{name}.json").read_text())


def verdicts(v: dict) -> dict[str, str]:
    return {r["reading"]: r["verdict"] for r in v["verdicts"]}


def test_the_readings_and_the_loops_ideal_come_from_the_card() -> None:
    rd = {r["reading"]: r for r in S.price_readings()}
    assert rd["per-clock"]["threshold"] == pytest.approx(31.84)
    assert rd["sheet-clocks"]["price"] == pytest.approx(29.5758, abs=1e-4)
    assert rd["cautious"]["price"] == pytest.approx(32 * 1500 / 1980) and "1500 MHz" in rd["cautious"]["source"]
    assert S.price_readings(base_mhz=1600, base_source="x")[2]["price"] == pytest.approx(32 * 1600 / 1980)
    assert S.ideal_adds() == pytest.approx(IDEAL)


def test_p_is_the_tensor_rate_alone_over_the_adds_rate_while_adding(tmp_path: Path) -> None:
    """A C point adding at R_T / 33.5 a second for a quarter of its interval has p = 33.5, though its adds over the
    whole interval are a quarter of that rate; it held the clock nvidia-smi read inside its interval, not 1980."""
    smi = ("1970/01/01 00:16:40.500, 1830, 1980, 690.5, 700.00, 61, 0x0\n"
           "1970/01/01 00:16:40.700, 1850, 1980, 680.5, 700.00, 61, 0x0\n"
           "1970/01/01 00:16:50.000, 1980, 1980, 100.0, 700.00, 61, 0x0\n")
    rc, v = run(tmp_path, bpoint("A"), bpoint("B", p=33.4, share=0.99), bpoint("C", x=128, p=33.5, share=0.25),
                bpoint("D", x=10, p=32.5, share=0.5), smi=smi)
    assert (rc, v["decided"], v["undersaturated"]) == (0, True, False)
    assert verdicts(v) == {"per-clock": "survives", "sheet-clocks": "survives", "cautious": "survives"}
    c = next(r for r in v["rows"] if r["arm"] == "C")
    assert c["p"] == pytest.approx(33.5) and c["p_lo"] < c["p"] < c["p_hi"]
    assert c["r_f_mixed"] == pytest.approx(R_T / 33.5) and c["adds_per_s"] == pytest.approx(R_T / 33.5 * 0.25)
    assert (c["set"], c["held_mhz"], c["smi_mhz_max"]) == ("lgc 1980", 1840.0, 1850.0)
    assert "clock" not in c
    assert v["lowest"]["arm"] == "D" and v["lowest"]["p"] == pytest.approx(32.5)


def test_each_reading_is_tested_on_its_own(tmp_path: Path) -> None:
    """A D point at p = 30.5 refutes the per-clock price and leaves the sheet's and the cautious one standing."""
    rc, v = run(tmp_path, bpoint("A"), bpoint("A", clock="default"), bpoint("B", p=33.4, share=0.99),
                bpoint("C", x=8, p=33.5, share=1.0), bpoint("D", x=100, p=30.5, share=0.5, clock="default"))
    assert verdicts(v) == {"per-clock": "refuted", "sheet-clocks": "survives", "cautious": "survives"}
    assert rc == 0 and v["verdicts"][0]["refuting"] == 1


def test_an_undersaturated_arm_b_survives_nothing(tmp_path: Path) -> None:
    """Arm B under 95% of the loop's ideal under-measures R_f, so p reads high: no reading survives, and a refuted one
    stays refuted."""
    rc, v = run(tmp_path, bpoint("A"), bpoint("B", p=36.5, share=0.99, adds_per_smclk=104.67),
                bpoint("C", x=8, p=SHEET * 0.99, share=1.0))
    assert (rc, v["undersaturated"]) == (1, True)
    assert verdicts(v) == {"per-clock": "refuted", "sheet-clocks": "refuted", "cautious": "undecided"}
    b = next(r for r in v["rows"] if r["arm"] == "B")
    assert b["undersaturated"] and b["saturation"] == pytest.approx(104.67 / IDEAL)
    ok = run(tmp_path, bpoint("A"), bpoint("B", p=33.4, share=0.99, adds_per_smclk=0.951 * IDEAL),
             bpoint("C", x=8, p=33.5, share=1.0), name="ok")[1]
    assert not ok["undersaturated"] and verdicts(ok)["per-clock"] == "survives"


def test_a_point_within_the_stamps_of_a_threshold_is_undecided(tmp_path: Path) -> None:
    """Two stamps of a millisecond over 1.2 s move p by about 0.17%: 0.05% either side of 31.84 is undecided."""
    for f in (1.0005, 0.9995):
        rc, v = run(tmp_path, bpoint("A"), bpoint("B", p=33.4, share=0.99),
                    bpoint("C", x=32, p=31.84 * f, share=1.0), name=str(f))
        assert (rc, verdicts(v)["per-clock"], verdicts(v)["sheet-clocks"]) == (1, "undecided", "survives")


def test_the_tensor_rate_is_arm_as_at_the_points_setting_and_class(tmp_path: Path) -> None:
    """The faster arm A point at another setting prices nothing at this one; a C point with no arm A at its class
    leaves every reading undecided; arm B's p is given and not judged."""
    fast = bpoint("A", clock="default")
    fast["macs"] *= 2
    rc, v = run(tmp_path, bpoint("A"), fast, bpoint("B", p=20.0, share=0.99), bpoint("C", x=128, p=33.5, share=0.25),
                bpoint("C", x=128, p=33.5, share=0.25, cls="zero"))
    c = next(r for r in v["rows"] if r["arm"] == "C" and r["class"] == "uniform")
    b = next(r for r in v["rows"] if r["arm"] == "B")
    assert c["p"] == pytest.approx(33.5) and b["p"] == pytest.approx(20.0) and not b["judged"]
    assert rc == 1 and set(verdicts(v).values()) == {"undecided"}


def test_no_beside_points_is_no_verdict(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    assert S.main([str(tmp_path / "out"), "--arms", "fp32-beside"]) == 2
