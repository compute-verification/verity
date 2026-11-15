"""`lean_changed.py --records` takes the Lean slot pool `$LEAN_RECORDS_POOL` names, else `build`."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pool() -> str:
    s = importlib.util.spec_from_file_location("lean_changed_pool", HERE.parent / "lean_changed.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.FAST_POOL


def test_records_runs_take_the_pool_lean_records_pool_names(monkeypatch):
    """A lane with a pool of its own on a node keeps its records trees warm for itself (`LEAN_RECORDS_POOL` names it);
    unset, `--records` takes `build`."""
    monkeypatch.delenv("LEAN_RECORDS_POOL", raising=False)
    assert pool() == "build"
    monkeypatch.setenv("LEAN_RECORDS_POOL", "proofs")
    assert pool() == "proofs"
