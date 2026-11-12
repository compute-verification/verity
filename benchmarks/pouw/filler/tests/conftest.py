"""The suite's set-up. pytest's ``pythonpath`` (``pyproject.toml``) holds this directory and the harness, and a test
imports their modules by name. A script whose name the harness holds too comes from `load`, under a name of its own:
``comm/report.py``, since the harness's scripts import their own ``report`` by name."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

FILLER = Path(__file__).resolve().parents[1]


def load(file: str) -> ModuleType:
    """``benchmarks/pouw/filler/FILE`` as a module named for its directory and its file (``comm/report.py`` is
    ``comm_report``), run once however many tests ask."""
    name = "_".join(Path(file).with_suffix("").parts)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, FILLER / file)
        sys.modules[name] = module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]
