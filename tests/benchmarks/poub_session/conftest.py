from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lean_stands_in


@pytest.fixture(autouse=True)
def _lean_stands_in(monkeypatch):
    """The Lean answers the memory verdicts don't have yet, stood in agreeing with the Python (`lean_stands_in`), so
    the suite's tests of what follows an accept run; a test of the undecided answer calls `lean_stands_in.unwritten`."""
    lean_stands_in.agree(monkeypatch)
