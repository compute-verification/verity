"""settings.py, window.py's registry of the environment it reads: a declared variable's default, its value when set, a
required one stopping the job with what to set, an undeclared name refused, and every name window.py reads declared there
and read (a name window.py reads without a declaration stops a run when it gets there; a stale one misleads)."""
from __future__ import annotations

import ast

import pytest

from conftest import HERE, load


@pytest.fixture
def settings(monkeypatch):
    s = load(monkeypatch, "settings.py")
    for k in s.DECLARED:
        monkeypatch.delenv(k, raising=False)
    return s


def test_get(settings, monkeypatch):
    assert settings.get("WIN_UNMEASURED_S") == "900" and settings.get("WIN_MEM_GB") == ""
    monkeypatch.setenv("WIN_UNMEASURED_S", "60")
    assert settings.get("WIN_UNMEASURED_S") == "60"
    with pytest.raises(SystemExit, match="set WIN_CPUS, the most cores"):
        settings.get("WIN_CPUS")
    monkeypatch.setenv("WIN_CPUS", "96")
    assert settings.get("WIN_CPUS") == "96"
    with pytest.raises(KeyError, match="WIN_NOPE is not declared"):
        settings.get("WIN_NOPE")


def test_window_reads_what_settings_declares(settings):
    tree = ast.parse((HERE / "window.py").read_text())
    read = {c.args[0].value for c in ast.walk(tree) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            and c.func.id == "env" and c.args and isinstance(c.args[0], ast.Constant)}
    dynamic = [ast.unparse(c) for c in ast.walk(tree) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
               and c.func.id == "env" and not (c.args and isinstance(c.args[0], ast.Constant))]
    assert not dynamic, f"window.py reads a variable whose name this test can't see: {dynamic}"
    assert read - set(settings.DECLARED) == set(), "window.py reads a variable settings.py doesn't declare"
    assert set(settings.DECLARED) - read == set(), "settings.py declares a variable window.py never reads"
