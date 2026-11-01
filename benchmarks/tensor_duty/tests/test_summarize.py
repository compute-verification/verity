"""summarize.py's verdict, the falsification test of the H200 card's `clock.tensor`: its bound is the card's tensor
rate, a point refutes the card only by its MACs over the host's wall clock, and a point the host stamps' resolution
can't place on either side of the bound leaves the verdict undecided."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import summarize as S
from verity import hardware

BOUND = 132 * 4096 * 1830e6
SECS = 1.2


def point(of_bound: float, *, host_s: float = SECS, event_s: float = SECS, gt_s: float = SECS, **at) -> dict:
    """A duty.cu line whose MACs over ``host_s`` of the host's wall clock are ``of_bound`` of the card's rate."""
    return {"clock": "1980", "arm": "wg_s8", "class": "zero", "fill": "idle", "burst": 65536, "duty": 100,
            "host_t0": 1000.0, "host_t1": 1000.0 + host_s, "event_s": event_s, "gt_s": gt_s,
            "macs": of_bound * BOUND * host_s, "clock64_mhz": 1830.0, "tensor_mhz": 1830.0, "fill_mhz": 0.0,
            "duty_achieved": 1.0, **at}


def run(tmp_path: Path, *pts: dict, name: str = "out") -> tuple[int, dict]:
    """summarize.py's exit code and JSON verdict over one out/ directory holding ``pts``."""
    out = tmp_path / name
    out.mkdir()
    (out / "points.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pts))
    rc = S.main([str(out), "--json", str(tmp_path / f"{name}.json")])
    return rc, json.loads((tmp_path / f"{name}.json").read_text())


def test_the_bound_is_the_cards_tensor_rate_and_follows_its_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    assert S.readings() == (BOUND, 132 * 4096 * 1980e6, 132 * 4096 * 1e6)
    card = copy.deepcopy(hardware.load(S.SKU))
    card["clock"]["tensor"]["value"] = 1900
    monkeypatch.setattr(S, "load", lambda sku: card)
    assert S.readings()[0] == 132 * 4096 * 1900e6


def test_points_under_the_card_pass(tmp_path: Path) -> None:
    rc, v = run(tmp_path, point(0.5), point(0.999, duty=99, fill="alu"))
    assert (rc, v["verdict"], v["refuting"], v["undecided"]) == (0, "PASS", 0, 0)
    assert v["top"]["of_bound"] == pytest.approx(0.999)


def test_a_point_over_the_card_refutes_it(tmp_path: Path) -> None:
    rc, v = run(tmp_path, point(0.5), point(1.01))
    assert (rc, v["verdict"], v["refuting"]) == (1, "REFUTED", 1)


def test_a_point_within_a_stamp_of_the_bound_is_undecided(tmp_path: Path) -> None:
    """At 1.2 s a millisecond stamp moves the rate by about 0.08%: 1.0005 of the bound may be under it, and 0.9995
    over it."""
    for of_bound in (1.0005, 0.9995):
        rc, v = run(tmp_path, point(of_bound), name=str(of_bound))
        assert (rc, v["verdict"], v["refuting"], v["undecided"]) == (1, "UNDECIDED", 0, 1)


def test_the_gpus_own_timers_do_not_judge(tmp_path: Path) -> None:
    """A point the GPU's timers put over the bound passes on the host's wall clock, and one they put under it
    doesn't."""
    fast = point(0.99, event_s=SECS * 0.98, gt_s=SECS * 0.98)
    assert fast["macs"] / fast["event_s"] > BOUND
    assert run(tmp_path, fast, name="fast")[1]["verdict"] == "PASS"
    slow = point(1.01, event_s=SECS * 1.02, gt_s=SECS * 1.02)
    assert slow["macs"] / slow["event_s"] < BOUND
    assert run(tmp_path, slow, name="slow")[1]["verdict"] == "REFUTED"


def test_nvidia_smi_lines_are_read_and_the_rest_counted(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "smi.csv").write_text("1970/01/01 00:16:40.500, 1830, 1980, 690.5, 700.00, 61, 0x0000000000000000\n"
                                 "Unable to determine the device handle for GPU0: Unknown Error\n")
    (out / "points.jsonl").write_text(json.dumps(point(0.5)) + "\n")
    assert S.main([str(out), "--json", str(tmp_path / "v.json")]) == 0
    v = json.loads((tmp_path / "v.json").read_text())
    assert (v["verdict"], v["smi_unread"]) == ("PASS", 1)
    smi, unread = S.samples(out)
    assert [s[1:] for s in smi] == [(1830.0, 690.5, 0)] and len(unread) == 1


def test_no_points_is_no_verdict(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    assert S.main([str(tmp_path / "out")]) == 2
