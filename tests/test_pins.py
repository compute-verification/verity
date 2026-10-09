"""No pin is stale, and every file that records a Lean source's digest is a pin with its command (`tools/verity/check/pins.py`)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repo_pins_under_test", ROOT / "tools/verity/check/pins.py")
PINS = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PINS
spec.loader.exec_module(PINS)


def test_no_pin_is_stale():
    rows = PINS.stale(ROOT)
    assert not rows, "regenerate each stale pin with its command, never by hand:\n" + PINS.report(rows, ROOT)


def test_every_file_that_records_a_lean_sources_digest_is_a_pin():
    found = PINS.unlisted(ROOT)
    assert not found, ("these record Lean sources' digests: list each in tools/verity/check/pins.py `PINS` with the command "
                       f"that regenerates it: {found}")
