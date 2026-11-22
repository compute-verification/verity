"""The suite's test primitives' Boolean bodies (`tests/verity/circuits/stand_in.py`), registered with the Program's lookup as
it loads, before any test module builds a Program of them."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(next(p for p in Path(__file__).resolve().parents if p.name == "tests") / "verity" / "circuits"))

import stand_in  # noqa: E402,F401
