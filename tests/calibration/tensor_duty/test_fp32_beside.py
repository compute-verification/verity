"""summarize.py --arms fp32-beside: each clock reading of an FP32 add's price (32, both pipes at 1980 MHz, the default;
29.58, the tensor at 1830 MHz) is judged under CMP-438's unit, each operation charged in seconds at its own pipe's
ceiling: a point whose MACs or adds over its host interval exceed the reading's tensor or FP32 ceiling refutes it, one
within the host stamps' resolution of a ceiling leaves it undecided, and a point's clock is the one it held. p =
R_t,alone / R_f,mixed, the refused unit's price, is a diagnostic that decides nothing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import summarize as S

SECS = 1.2
R_1830 = 132 * 4096 * 1830e6
R_1980 = 132 * 4096 * 1980e6
F_1980 = 132 * 128 * 1980e6
R_A = 0.98 * R_1830
IDEAL = 128 * 64 / 67


def bpoint(arm: str, *, x: float = 0, p: float | None = None, share: float = 0.0, clock: str = "1980",
           cls: str = "uniform", host_s: float = SECS, macs_s: float | None = None, adds_s: float | None = None,
           adds_per_smclk: float | None = None) -> dict:
    """A duty.cu fp32-beside line: arm A moves R_A MACs a second over ``host_s`` unless ``macs_s`` says otherwise;
    another arm adds at R_A / ``p`` a second over ``share`` of it, or at ``adds_s`` over all of it; arm B adds at the
    loop's ideal a SM-clock unless ``adds_per_smclk`` says otherwise."""
    adds = adds_s * host_s if adds_s is not None else R_A / p * host_s * share if p else 0.0
    macs = macs_s if macs_s is not None else R_A if arm == "A" else 0.0
    return {"clock": clock, "arm": arm, "class": cls, "x": x, "host_t0": 1000.0, "host_t1": 1000.0 + host_s,
            "macs": macs * host_s, "adds": adds, "fp32_share": share, "clock64_mhz": 1830.0,
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


def test_the_two_readings_and_their_ceilings_come_from_the_card() -> None:
    """The default first, then the tensor-clock reading, each with its ceilings; no third reading."""
    rd = S.price_readings()
    assert [(r["reading"], r["price"], r["default"]) for r in rd] == [("sm", 32, True), ("tensor", 29.58, False)]
    assert [r["tensor_ceiling"] for r in rd] == pytest.approx([R_1980, R_1830])
    assert [r["fp32_ceiling"] for r in rd] == pytest.approx([F_1980, F_1980])
    assert "never exceeds 1830 MHz" in rd[1]["source"]
    assert S.ideal_adds() == pytest.approx(IDEAL)


def test_points_under_every_ceiling_refute_neither_reading_whatever_p_reads(tmp_path: Path) -> None:
    """A run whose pipes stay under their ceilings refutes neither reading, though its p, the refused unit's price, is
    under both; p is still given, at the clock each point held, not the one asked for."""
    smi = ("1970/01/01 00:16:40.500, 1830, 1980, 690.5, 700.00, 61, 0x0\n"
           "1970/01/01 00:16:40.700, 1850, 1980, 680.5, 700.00, 61, 0x0\n"
           "1970/01/01 00:16:50.000, 1980, 1980, 100.0, 700.00, 61, 0x0\n")
    rc, v = run(tmp_path, bpoint("A"), bpoint("B", p=33.4, share=0.99),
                bpoint("C", x=128, p=29.0, share=0.25, macs_s=0.9 * R_1830),
                bpoint("D", x=10, p=28.5, share=0.5, macs_s=0.5 * R_1830), smi=smi)
    assert (rc, v["decided"]) == (0, True)
    assert verdicts(v) == {"sm": "survives", "tensor": "survives"}
    c = next(r for r in v["rows"] if r["arm"] == "C")
    assert c["p"] == pytest.approx(29.0) and c["p_lo"] < c["p"] < c["p_hi"]
    assert c["r_f_mixed"] == pytest.approx(R_A / 29.0) and c["adds_per_s"] == pytest.approx(R_A / 29.0 * 0.25)
    assert (c["set"], c["held_mhz"], c["smi_mhz_max"]) == ("lgc 1980", 1840.0, 1850.0)
    assert v["lowest"]["arm"] == "D" and v["lowest"]["p"] == pytest.approx(28.5)


def test_a_tensor_rate_between_the_two_clocks_refutes_only_the_1830_reading(tmp_path: Path) -> None:
    """MACs at 95% of the 1980 MHz ceiling are above the 1830 MHz one: the tensor-clock reading falls, the default
    stands."""
    rc, v = run(tmp_path, bpoint("A", macs_s=0.95 * R_1980), bpoint("C", x=8, p=33.5, share=1.0))
    assert (rc, verdicts(v)) == (0, {"sm": "survives", "tensor": "refuted"})
    assert v["verdicts"][1]["refuting"] == 1


def test_adds_over_the_fp32_ceiling_refute_both_readings(tmp_path: Path) -> None:
    rc, v = run(tmp_path, bpoint("A"), bpoint("B", adds_s=1.01 * F_1980, share=1.0))
    assert (rc, verdicts(v)) == (0, {"sm": "refuted", "tensor": "refuted"})


def test_a_point_within_the_stamps_of_a_ceiling_is_undecided(tmp_path: Path) -> None:
    """Two stamps of a millisecond over 1.2 s move a rate by about 0.17%: 0.05% either side of the 1830 MHz ceiling
    leaves that reading undecided, and the run exits 1."""
    for f in (1.0005, 0.9995):
        rc, v = run(tmp_path, bpoint("A", macs_s=f * R_1830), name=str(f))
        assert (rc, verdicts(v)) == (1, {"sm": "survives", "tensor": "undecided"})


def test_p_is_a_diagnostic_and_decides_nothing(tmp_path: Path) -> None:
    """p takes arm A's rate at the point's setting and class: the faster arm A point at another setting prices nothing
    here, a point with no arm A at its class has no p, arm B's p is given, and an undersaturated arm B is marked; none of
    it moves a verdict."""
    fast = bpoint("A", clock="default")
    fast["macs"] *= 1.01
    rc, v = run(tmp_path, bpoint("A"), fast, bpoint("B", p=34.0, share=0.99, adds_per_smclk=104.67),
                bpoint("C", x=128, p=33.5, share=0.25), bpoint("C", x=128, p=33.5, share=0.25, cls="zero"))
    c = next(r for r in v["rows"] if r["arm"] == "C" and r["class"] == "uniform")
    zero = next(r for r in v["rows"] if r["arm"] == "C" and r["class"] == "zero")
    b = next(r for r in v["rows"] if r["arm"] == "B")
    assert c["p"] == pytest.approx(33.5) and zero["p"] is None and b["p"] == pytest.approx(34.0)
    assert b["undersaturated"] and b["saturation"] == pytest.approx(104.67 / IDEAL) and v["undersaturated"]
    assert (rc, verdicts(v)) == (0, {"sm": "survives", "tensor": "survives"})


def test_no_beside_points_is_no_verdict(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    assert S.main([str(tmp_path / "out"), "--arms", "fp32-beside"]) == 2
