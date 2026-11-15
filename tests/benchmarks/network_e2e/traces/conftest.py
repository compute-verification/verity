from __future__ import annotations

import importlib
from pathlib import Path

import pytest

HERE = (Path(__file__).resolve().parents[4] / "benchmarks" / "network_e2e" / "traces")


@pytest.fixture
def mod(monkeypatch):
    """`mod("traffic")`: a module of the tracer's directory, imported with that directory on sys.path."""
    monkeypatch.syspath_prepend(str(HERE))
    return importlib.import_module
