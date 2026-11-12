"""summarize.py's reading of run.sh's runs, on made-up arms: which arm counts at its ratio, that a class's rate at a
ratio is its fastest counting arm over every kernel, encoding, round and card, and what the command prints and writes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import summarize as S

CARD = {"card": "h200-a", "uuid": "GPU-0", "clock_mhz": 1830, "clock_locked": True}


def alone(cls: str, tmacs: float, **more: object) -> dict:
    """A wgmma line with no hash beside it, as class_rate.cu prints it and run.sh stamps its round."""
    return {"round": 1, "kernel": "wgmma", "class": cls, "enc": "s8s8", "ratio": 0, "mhz": 1830.0,
            "macs_per_smclk": 4096.0, "tmacs_per_s": tmacs, **more}


def beside(cls: str, tmacs: float, ratio: float, **more: object) -> dict:
    """A wgmma line beside SHA-512 paced to ``ratio`` compressions a T MAC, whose hash reached it inside the window."""
    return alone(cls, tmacs, **{"ratio": ratio, "tensor_paced": 1.0, "ratio_achieved": ratio, "covered": True, **more})


def lt(cls: str, tmacs: float, **more: object) -> dict:
    """A cuBLASLt line: alone only, with no clock of its own and its output checked."""
    return {"round": 1, "kernel": "lt", "class": cls, "enc": "s8s8", "ratio": 0, "tmacs_per_s": tmacs, "checked": True,
            **more}


def write(out: Path, arms: list[dict], card: dict = CARD) -> Path:
    """An out/ directory as run.sh leaves it."""
    out.mkdir()
    (out / "card.json").write_text(json.dumps(card))
    (out / "rates.jsonl").write_text("".join(json.dumps(a) + "\n" for a in arms))
    return out


def test_theta_is_a_whole_tensor_at_the_declared_clock() -> None:
    assert pytest.approx(132 * 4096 * 1830e6 / 1e12) == S.THETA
    assert S.per_clock({"macs_per_smclk": 4096.0}) == pytest.approx(S.THETA)
    assert S.per_clock({"macs_per_smclk": 2048.0}) == pytest.approx(S.THETA / 2)


def test_an_arm_alone_counts_unless_its_output_failed_the_check() -> None:
    assert S.counts(alone("s8", 900.0))
    assert S.counts(lt("s8", 900.0))
    assert not S.counts(lt("s8", 900.0, checked=False))


def test_an_arm_beside_the_hash_counts_only_where_the_hash_reached_its_ratio_inside_the_window() -> None:
    assert S.counts(beside("s8", 700.0, 2.0))
    assert S.counts(beside("s8", 700.0, 2.0, ratio_achieved=1.98)), "to 1% of the target"
    assert not S.counts(beside("s8", 700.0, 2.0, ratio_achieved=1.979))
    assert not S.counts(beside("s8", 700.0, 2.0, covered=False))


def test_a_classs_rate_at_a_ratio_is_its_fastest_counting_arm() -> None:
    slow, fast = alone("s8", 900.0), lt("s8", 950.0, round=2)
    unchecked = lt("s8", 990.0, checked=False)
    short = beside("s8", 980.0, 1.70, ratio_achieved=1.2)
    kept = beside("s8", 700.0, 1.70, enc="u8u8")
    only, uncovered = alone("u4", 960.0), beside("u4", 800.0, 2.3, covered=False)
    got = S.reading([slow, fast, unchecked, short, kept, only, uncovered])
    assert list(got) == list(S.CLASSES)
    assert got["s8"] == {0: {"best": fast, "short": [unchecked]}, 1.70: {"best": kept, "short": [short]}}
    assert got["u4"] == {0: {"best": only, "short": []}, 2.3: {"best": None, "short": [uncovered]}}
    assert got["s2"] == got["u2"] == got["zero"] == {}


def test_rows_reads_the_cards_arms_and_passes_over_lines_that_are_no_arm(tmp_path: Path) -> None:
    out = write(tmp_path / "out", [alone("s8", 900.0)])
    with (out / "rates.jsonl").open("a") as f:
        f.write("warming up\n\n" + json.dumps(lt("zero", 970.0)) + "\n")
    card, arms = S.rows(out)
    assert card == CARD
    assert arms == [alone("s8", 900.0, card="h200-a"), lt("zero", 970.0, card="h200-a")]


def test_render_lists_each_class_and_ratio_with_its_arm() -> None:
    arms = [dict(a, card="h200-a") for a in (
        alone("s8", 989.42976), beside("s8", 742.0, 1.70, tensor_paced=0.75),
        beside("s8", 800.0, 1.70, ratio_achieved=1.5, round=2), lt("zero", 970.0), beside("u4", 800.0, 2.3, covered=False))]
    clock = "1830 MHz, 4096 MACs a SM-clock = 989.4 T MAC/s at 1830 MHz"
    assert S.render([(CARD, arms)]).splitlines() == [
        "theta = 989.4 T MAC/s (4096 MACs a SM-clock, 132 SMs, 1830 MHz)",
        "card h200-a GPU-0: clock 1830 MHz, locked True",
        "",
        "h200-a:",
        f"    s8 at 0 M/T MAC: 989.4 T MAC/s = 1.0000 theta (wgmma s8s8 r1 on h200-a, {clock})",
        (f"    s8 at 1.7 M/T MAC: 742.0 T MAC/s = 0.7499 theta (wgmma s8s8 r1 on h200-a, {clock}, hash 1.700 M/T MAC, "
         "tensor paced to 0.750 of alone)"),
        f"        not counted: 800.0 T MAC/s, wgmma s8s8 r2 on h200-a, {clock}, hash 1.500 M/T MAC",
        "    u4 at 2.3 M/T MAC: no arm counts",
        f"        not counted: 800.0 T MAC/s, wgmma s8s8 r1 on h200-a, {clock}, hash 2.300 M/T MAC",
        "  zero at 0 M/T MAC: 970.0 T MAC/s = 0.9804 theta (lt s8s8 r1 on h200-a)",
    ]


def test_several_cards_get_a_reading_each_and_one_over_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    """The command over two runs' out/ directories: each card's own maximum, then the maximum over both, and the JSON
    holds the reading over all cards with its ratios as strings."""
    a = write(tmp_path / "a", [alone("s8", 900.0), beside("s8", 700.0, 1.70)])
    b = write(tmp_path / "b", [alone("s8", 950.0)], {**CARD, "card": "h200-b", "uuid": "GPU-1"})
    monkeypatch.setattr(sys, "argv", ["summarize.py", str(a), str(b), "--json", str(tmp_path / "reading.json")])
    S.main()
    scopes = capsys.readouterr().out.split("\n\n")
    assert [s.splitlines()[0] for s in scopes[1:]] == ["h200-a:", "h200-b:", "all cards:"]
    assert "s8 at 0 M/T MAC: 900.0 T MAC/s" in scopes[1] and "s8 at 0 M/T MAC: 950.0 T MAC/s" in scopes[2]
    assert "s8 at 0 M/T MAC: 950.0 T MAC/s = 0.9601 theta (wgmma s8s8 r1 on h200-b" in scopes[3]
    assert "s8 at 1.7 M/T MAC: 700.0 T MAC/s" in scopes[3]
    wrote = json.loads((tmp_path / "reading.json").read_text())
    assert wrote["theta"] == S.THETA and [c["card"] for c in wrote["cards"]] == ["h200-a", "h200-b"]
    assert set(wrote["reading"]) == set(S.CLASSES) and set(wrote["reading"]["s8"]) == {"0", "1.7"}
    assert wrote["reading"]["s8"]["0"]["best"]["card"] == "h200-b"
    assert wrote["reading"]["s8"]["1.7"] == {"best": beside("s8", 700.0, 1.70, card="h200-a"), "short": []}
