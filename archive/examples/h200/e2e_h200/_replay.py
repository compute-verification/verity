"""This directory's `replay.py` (beside the package), whichever `replay` the path finds first: other benchmarks have a
`replay.py` of their own, and network_traces' comes first when its `demo_endpoint.py` serves `e2e_h200.engine:frozen`. Where
the path's `replay` is this one (the tests, `run.py`), it is that module; otherwise this file loads under its own name."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

PATH = Path(__file__).resolve().parent.parent / "replay.py"
NAME = "e2e_h200_replay"


def _load() -> ModuleType:
    spec = importlib.util.find_spec("replay")
    if spec is not None and spec.origin and Path(spec.origin).resolve() == PATH:
        return importlib.import_module("replay")
    if NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(NAME, PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[NAME] = mod
        spec.loader.exec_module(mod)
    return sys.modules[NAME]


RP = _load()
