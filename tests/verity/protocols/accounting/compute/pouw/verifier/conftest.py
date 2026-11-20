"""The suite's test primitives get Boolean bodies before collection."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _stand_in() -> None:
    """The suite's test primitives' Boolean bodies (`tests/verity/circuits/stand_in.py`), registered with the Program's
    lookup as it loads, before any test module builds a Program of them."""
    if "stand_in" in sys.modules:
        return
    tests = next(p for p in Path(__file__).resolve().parents if p.name == "tests")
    spec = importlib.util.spec_from_file_location("stand_in", tests / "verity" / "circuits" / "stand_in.py")
    sys.modules["stand_in"] = mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


_stand_in()
