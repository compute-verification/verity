"""research's settings registry (`research.settings`): each reader behaves as the ``os.environ`` read it replaced, unset,
empty and set alike, and every read through it in research names a declared variable."""
from pathlib import Path

import pytest

from research import settings

SRC = Path(__file__).resolve().parents[1] / "src" / "research"
NAME = "RESEARCH_DISK_HOLD_PCT"


def test_get_returns_the_default_only_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv(NAME, raising=False)
    assert settings.get(NAME) is None and settings.get(NAME, "80") == "80"
    monkeypatch.setenv(NAME, "")
    assert settings.get(NAME, "80") == "" and settings.get(NAME) == ""
    monkeypatch.setenv(NAME, "70")
    assert settings.get(NAME, "80") == "70"


def test_need_raises_key_error_when_unset_and_has_counts_an_empty_variable_as_set(monkeypatch):
    monkeypatch.delenv(NAME, raising=False)
    with pytest.raises(KeyError):
        settings.need(NAME)
    assert not settings.has(NAME)
    monkeypatch.setenv(NAME, "")
    assert settings.need(NAME) == "" and settings.has(NAME)


def test_an_undeclared_name_is_refused_past_an_except_key_error_and_named_reads_one(monkeypatch):
    monkeypatch.setenv("RESEARCH_NOT_DECLARED_ANYWHERE", "x")
    for read in (settings.get, settings.need, settings.has):
        try:
            read("RESEARCH_NOT_DECLARED_ANYWHERE")
        except KeyError:
            raise AssertionError(f"{read.__name__}: an undeclared name raised KeyError") from None
        except settings.Undeclared as e:
            assert "RESEARCH_NOT_DECLARED_ANYWHERE is not declared" in str(e)
        else:
            raise AssertionError(f"{read.__name__}: an undeclared name was read")
    assert settings.named("RESEARCH_NOT_DECLARED_ANYWHERE") == "x" and settings.named("RESEARCH_UNSET_TOO", "d") == "d"


def test_undeclared_finds_a_literal_or_a_constant_name_through_any_import_of_the_registry():
    src = ("from . import settings\nfrom research import settings as rs\nX_ENV = 'NOPE_TWO'\n"
           "settings.get('PATH')\nsettings.get('NOPE_ONE', '')\nrs.need(X_ENV)\nrs.has('USER')\nother.get('NOPE_THREE')\n")
    assert settings.undeclared(src) == [(5, "NOPE_ONE"), (6, "NOPE_TWO")]


def test_every_read_through_the_registry_in_research_is_declared():
    found = [f"{p.relative_to(SRC)}:{line}: {name}" for p in sorted(SRC.rglob("*.py"))
             for line, name in settings.undeclared(p.read_text())]
    assert found == [], "declare these in research/settings.py (DECLARED):\n" + "\n".join(found)
