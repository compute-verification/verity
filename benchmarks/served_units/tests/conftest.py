from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

HERE = Path(__file__).resolve().parent.parent


def load(mp: pytest.MonkeyPatch, file: str) -> ModuleType:
    """``benchmarks/served_units/FILE`` as a module of its own name, registered under ``mp`` (a dataclass resolves its
    module through sys.modules) with sys.path restored after (ks_form.py puts the tree's root on it)."""
    name = f"served_units_{Path(file).stem}"
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    mp.setitem(sys.modules, name, m)
    mp.setattr(sys, "path", [*sys.path])
    spec.loader.exec_module(m)
    return m
