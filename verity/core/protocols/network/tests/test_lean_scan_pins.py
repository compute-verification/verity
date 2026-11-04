"""`NetworkCheck.Period.scanPins`, the scan Programs the period check's rows form takes, is the Python's copy,
`scan_service.SCAN_PINS`, pair for pair and in order. `benchmarks/network_traces/tests/test_scan_pins.py` rebuilds each
pin from its scan class, and prints them as a script."""
from __future__ import annotations

import re
from pathlib import Path

from verity.core.protocols.certifier.scan_service import SCAN_PINS

PERIOD = Path(__file__).resolve().parents[1] / "lean" / "NetworkCheck" / "Period.lean"


def test_leans_scan_pins_are_scan_services():
    text = re.sub(r"\\\n\s*", "", PERIOD.read_text())
    m = re.search(r"^def scanPins : List \(String × String\) :=\s*\[(.*?)\]$", text, re.M | re.S)
    assert m, f"{PERIOD.name} has no `def scanPins : List (String × String) := [...]`"
    assert re.findall(r'\("([0-9a-f]+)",\s*"([0-9a-f]+)"\)', m.group(1)) == list(SCAN_PINS.items()), (
        f"{PERIOD.name}'s scanPins is not scan_service.SCAN_PINS: set both to the pins `uv run python "
        "benchmarks/network_traces/tests/test_scan_pins.py` prints, in its order")
