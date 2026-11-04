"""`scan_service.SCAN_PINS`, the scan Programs the period check's rows form takes, against `scan_zk`'s classes: one pin a
class, at its schedule's digest (`WardenParams.digest`) in `SETUPS`' order, each the SHA-512 of the Program its class's
egress tie builds at G = 1 (`EgressTie.partition`'s `program`), so a change to a class or to the scan's generator fails
here until it re-pins. Its Lean copy, `NetworkCheck.Period.scanPins`, is held to it by the network suite's
`test_lean_scan_pins.py`. Run as a script, it prints each class's name, schedule digest and Program:

    uv run python benchmarks/network_traces/tests/test_scan_pins.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE)]

import scan_zk as SZ  # noqa: E402
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


if __name__ == "__main__":
    for name, s in SZ.SETUPS.items():
        print(name, *pin(s))
