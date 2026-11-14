"""The guard's settings registry (`tools/agent-guard/settings.py`): each reader behaves as the ``os.environ`` read it replaced,
unset, empty and set alike, and every read through it in the guard names a declared variable."""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1]
NAME = "AGENT_GUARD_MIN_AVAIL"


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)     # no __pycache__ in tools/agent-guard, which boot.sh copies file by file
    spec = importlib.util.spec_from_file_location("guard_settings_under_test", GUARD / "settings.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_get_returns_the_default_only_when_the_variable_is_unset(settings, monkeypatch):
    monkeypatch.delenv(NAME, raising=False)
    assert settings.get(NAME) is None and settings.get(NAME, "0.25") == "0.25"
    monkeypatch.setenv(NAME, "")
    assert settings.get(NAME, "0.25") == ""
    monkeypatch.setenv(NAME, "0.5")
    assert settings.get(NAME, "0.25") == "0.5"


def test_need_raises_key_error_when_unset_and_has_counts_an_empty_variable_as_set(settings, monkeypatch):
    monkeypatch.delenv(NAME, raising=False)
    with pytest.raises(KeyError):
        settings.need(NAME)
    assert not settings.has(NAME)
    monkeypatch.setenv(NAME, "")
    assert settings.need(NAME) == "" and settings.has(NAME)


def test_an_undeclared_name_is_refused_past_an_except_key_error(settings, monkeypatch):
    monkeypatch.setenv("AGENT_GUARD_NOT_DECLARED", "x")
    for read in (settings.get, settings.need, settings.has):
        try:
            read("AGENT_GUARD_NOT_DECLARED")
        except KeyError:
            raise AssertionError(f"{read.__name__}: an undeclared name raised KeyError") from None
        except settings.Undeclared as e:
            assert "AGENT_GUARD_NOT_DECLARED is not declared" in str(e)
        else:
            raise AssertionError(f"{read.__name__}: an undeclared name was read")


def test_every_read_through_the_registry_in_the_guard_is_declared(settings):
    read = re.compile(r"\bsettings\.(?:get|need|has)\(\s*[\"']([A-Za-z0-9_]+)[\"']")
    found = [f"{p.name}: {name}" for p in sorted(GUARD.glob("*.py")) for name in read.findall(p.read_text())
             if name not in settings.DECLARED]
    assert found == [], "declare these in tools/agent-guard/settings.py (DECLARED):\n" + "\n".join(found)
