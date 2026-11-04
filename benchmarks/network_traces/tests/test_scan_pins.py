"""`scan_service.SCAN_PINS`, the scan Programs the period check's rows form takes, against `scan_zk`'s classes: one pin a
class, at its schedule's digest (`WardenParams.digest`) in `SETUPS`' order, each the SHA-512 of the Program its class's
egress tie builds at G = 1 (`EgressTie.partition`'s `program`), so a change to a class or to the scan's generator fails
here until it re-pins. Its Lean copy, `NetworkCheck.Period.scanPins`, is held to it by the network suite's
`test_lean_scan_pins.py`. `scan_period.rows_case` names the pin at its run's schedule as the parameters' Program, never
the scan's registration's. Run as a script, it prints each class's name, schedule digest and Program:

    uv run python benchmarks/network_traces/tests/test_scan_pins.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE)]

import scan_period as SP  # noqa: E402
import scan_zk as SZ  # noqa: E402
import warden_serve as WS  # noqa: E402
from verity.core.protocols.certifier.scan_service import SCAN_PINS, EgressTie  # noqa: E402

SCRIPT = "benchmarks/network_traces/tests/test_scan_pins.py"
REPIN = (f"set scan_service.SCAN_PINS and its Lean copy, NetworkCheck.Period.scanPins, to the pins `uv run python "
         f"{SCRIPT}` prints, in its order")


def pin(s: SZ.Setup) -> tuple[str, str]:
    """Class ``s``'s pin: its schedule's digest and the SHA-512 of the scan Program its egress tie builds at G = 1."""
    return s.params.digest.hex(), EgressTie("scan-pins", s.params, bytes(32), s.cls).partition["program"]


def test_the_pins_are_keyed_by_the_classes_schedules():
    """One pin a class, at its schedule's digest, in `SETUPS`' order: a class added or removed, or one whose schedule
    changed, fails here without building a Program."""
    assert list(SCAN_PINS) == [s.params.digest.hex() for s in SZ.SETUPS.values()], (
        f"SCAN_PINS is not keyed by scan_zk.SETUPS' schedules, one a class: {REPIN}")


@pytest.mark.slow
@pytest.mark.parametrize("name", list(SZ.SETUPS))
def test_each_pin_is_its_class_scan_program(name):
    """Each pin is its class's scan Program, rebuilt. It catches what the keys can't: a change to the scan's generator
    (its templates, `scan_root`, the Program's codec) that moves a class's Program while its schedule stays, after which
    the deployed check would refuse every honest rows-form period of the class. It builds the class's whole Program,
    about a minute and several GiB at w63's size."""
    key, program = pin(SZ.SETUPS[name])
    assert SCAN_PINS.get(key) == program, f"class {name}'s scan Program is {program[:16]}…, not its pin: {REPIN}"


def _run(d: Path, params) -> Path:
    """A network certifier run of link-window 0 at the schedule ``params``, as far as `rows_case` reads it, and its scan
    of that window, whose registration names a Program of its own."""
    run, scan = d / "run", d / "scan"
    (scan / "joint").mkdir(parents=True)
    run.mkdir()
    (run / "deployment.json").write_text(json.dumps({"params": dataclasses.asdict(params), "session": "aa" * 32,
                                                     "key": {"params": [64, 4, 1], "public": "bb" * 80}}))
    (run / "certified.jsonl").write_text("".join(json.dumps({"certificate": {
        "digest": "cc" * 64, "direction": way.value, "key_id": "k", "link": link, "signature": "dd", "window": 0}})
        + "\n" for link, way in WS.CERTIFIED.values()))
    (scan / "answers.json").write_text(json.dumps({"registration": {"program": "ee" * 64}, "windows": [
        {"window": 0, "answer": {"verdict": "accept"}, "registrations": []}]}))
    (scan / "joint" / "program.json").write_text("{}")
    return d


def test_the_rows_case_names_the_scan_program_pinned_at_its_schedule(tmp_path):
    """At a class's schedule the parameters' Program is that schedule's pin, whatever Program the scan registered; a
    schedule no class pins (the test class's at another window) is refused by name, naming `SCAN_PINS`."""
    d = _run(tmp_path / "pinned", SZ.TEST.params)
    case = SP.rows_case(d / "run", d / "scan")
    assert case["params"]["program"] == SCAN_PINS[SZ.TEST.params.digest.hex()] != "ee" * 64, case["params"]["program"]
    other = dataclasses.replace(SZ.TEST.params, window=8)
    d = _run(tmp_path / "other", other)
    with pytest.raises(SystemExit, match=rf"no scan Program is pinned at its schedule's digest "
                                         rf"{other.digest.hex()[:16]}… \(scan_service.SCAN_PINS\)"):
        SP.rows_case(d / "run", d / "scan")


if __name__ == "__main__":
    for name, s in SZ.SETUPS.items():
        print(name, *pin(s))
