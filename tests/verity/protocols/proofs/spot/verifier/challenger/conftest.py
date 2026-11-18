import importlib.util
from pathlib import Path

_STAND_IN = importlib.util.spec_from_file_location("stand_in", Path(__file__).resolve().parents[7] / "tests/verity/circuits/stand_in.py")
_STAND_IN.loader.exec_module(importlib.util.module_from_spec(_STAND_IN))
