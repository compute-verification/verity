"""`verity.settings`, the distribution's settings registry: every variable declared once (a type, a role, a doc), read at the
read, an undeclared ``VERITY_ML_*`` variable refused by name, a ``debug`` setting refused on a run of record, and the run's
record of the value settings.  And the circuits area, on the soundness path, reads the environment through it alone: its
direct reads are held at today's count, none."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

from verity import settings as S

REGISTRY = Path(S.__file__)
AREA = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"\bVERITY_ML_[A-Z0-9_]*[A-Z0-9]\b")
#: the attributes of ``os`` that touch the environment
OS_ENV = frozenset({"environ", "environb", "getenv", "getenvb"})


def test_every_setting_has_a_type_a_role_and_a_doc() -> None:
    for name, s in S.SETTINGS.items():
        assert s.name == name and callable(s.type) and s.role in S.ROLES and s.doc.strip(), s


def test_the_registry_imports_the_standard_library_alone() -> None:
    """Every unit may import it, those on the soundness path included, so it imports nothing they may not."""
    nodes = list(ast.walk(ast.parse(REGISTRY.read_text(encoding="utf-8"))))
    mods = {a.name for n in nodes if isinstance(n, ast.Import) for a in n.names}
    mods |= {"." * n.level + (n.module or "") for n in nodes if isinstance(n, ast.ImportFrom)}
    assert all(m.split(".")[0] in sys.stdlib_module_names for m in mods), sorted(mods)


def test_declaring_a_setting_twice_or_with_an_unknown_role_fails() -> None:
    s = S.Setting("VERITY_ML_TEST_X", S.flag, False, S.WHERE, "a test's")
    with pytest.raises(ValueError, match="declared twice"):
        S._declare(s, s)
    with pytest.raises(ValueError, match="role 'telemetry'"):
        S._declare(S.Setting("VERITY_ML_TEST_X", S.flag, False, "telemetry", "a test's"))


def test_flag_reads_both_spellings_and_refuses_the_rest() -> None:
    assert [S.flag(v) for v in ("1", "TRUE", " yes", "on", "0", "False", "no", "OFF")] == [True] * 4 + [False] * 4
    with pytest.raises(ValueError, match="not a flag"):
        S.flag("2")


def test_a_read_parses_the_environment_at_the_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERITY_ML_BOOLEAN_EXHAUSTIVE", raising=False)
    assert S.VERITY_ML_BOOLEAN_EXHAUSTIVE is False
    monkeypatch.setenv("VERITY_ML_BOOLEAN_EXHAUSTIVE", "1")
    assert S.VERITY_ML_BOOLEAN_EXHAUSTIVE is True and S.get("VERITY_ML_BOOLEAN_EXHAUSTIVE", {}) is False
    for off in ("", "0", "off"):
        monkeypatch.setenv("VERITY_ML_BOOLEAN_EXHAUSTIVE", off)
        assert S.VERITY_ML_BOOLEAN_EXHAUSTIVE is False
    monkeypatch.setenv("VERITY_ML_BOOLEAN_EXHAUSTIVE", "2")
    with pytest.raises(ValueError, match="VERITY_ML_BOOLEAN_EXHAUSTIVE='2': not a flag"):
        _ = S.VERITY_ML_BOOLEAN_EXHAUSTIVE
    with pytest.raises(AttributeError):
        _ = S.VERITY_ML_NOT_A_SETTING
    with pytest.raises(AttributeError, match="not a declared setting"):
        S.get("VERITY_ML_NOT_A_SETTING")
    assert "VERITY_ML_BOOLEAN_EXHAUSTIVE" in dir(S)


def test_an_unknown_prefixed_variable_is_refused_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misspelt setting fails the run that carries it, at its first read, instead of running with the default."""
    typo = {"VERITY_ML_BOOLEAN_EXHAUSTVE": "1", "VERITY_ML_BOOLEAN_EXHAUSTIVE": "0", "VERITY_SLOW": "1", "PATH": "/bin"}
    assert S.unknown(typo) == ["VERITY_ML_BOOLEAN_EXHAUSTVE"]
    with pytest.raises(S.UnknownSetting, match="unknown setting VERITY_ML_BOOLEAN_EXHAUSTVE: verity/settings.py declares no"):
        S.validate_environ(typo)
    with pytest.raises(S.UnknownSetting):
        S.get("VERITY_ML_BOOLEAN_EXHAUSTIVE", typo)
    with pytest.raises(S.UnknownSetting):
        S.record(typo)
    S.validate_environ({"VERITY_SLOW": "1", "VERITY_ML_BOOLEAN_EXHAUSTIVE": "1"})
    monkeypatch.setenv("VERITY_ML_BOOLEAN_EXHAUSTVE", "1")
    with pytest.raises(S.UnknownSetting, match="VERITY_ML_BOOLEAN_EXHAUSTVE"):
        _ = S.VERITY_ML_BOOLEAN_EXHAUSTIVE


def _synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registry with one setting of each role, and two credentials: today's declares a ``where`` setting alone."""
    monkeypatch.setattr(S, "SETTINGS", S._declare(
        S.Setting("VERITY_ML_TEST_WIDTH", int, 32, S.VALUE, "a test's value"),
        S.Setting("VERITY_ML_TEST_SEED", str, "", S.VALUE, "a test's credential, by declaration", secret=True),
        S.Setting("VERITY_ML_TEST_API_KEY", str, "", S.VALUE, "a test's credential, by name"),
        S.Setting("VERITY_ML_TEST_SLOW", S.flag, False, S.WHERE, "a test's telemetry"),
        S.Setting("VERITY_ML_TEST_DUMP", str, "", S.DEBUG, "a test's diagnostic")))


def test_a_debug_setting_is_refused_on_a_run_of_record(monkeypatch: pytest.MonkeyPatch) -> None:
    _synthetic(monkeypatch)
    env = {"VERITY_ML_TEST_DUMP": "/tmp/dump", "VERITY_ML_TEST_WIDTH": "8"}
    S.validate_environ(env)
    with pytest.raises(S.DebugOnRecord, match="debug setting VERITY_ML_TEST_DUMP on a run of record"):
        S.validate_environ(env, of_record=True)
    S.validate_environ({**env, "VERITY_ML_TEST_DUMP": ""}, of_record=True)


def test_the_record_carries_the_value_settings_in_force_and_names_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    assert S.record({"VERITY_ML_BOOLEAN_EXHAUSTIVE": "1"})["values"] == {}
    _synthetic(monkeypatch)
    secret = "dummy-not-a-real-key"
    rec = S.record({"VERITY_ML_TEST_WIDTH": "8", "VERITY_ML_TEST_SEED": secret, "VERITY_ML_TEST_API_KEY": secret,
                    "VERITY_ML_TEST_SLOW": "1", "VERITY_ML_TEST_DUMP": "/tmp/dump", "PATH": "/bin"})
    assert rec["schema"] == "verity/settings/v1"
    assert rec["values"] == {"VERITY_ML_TEST_API_KEY": S.REDACTED, "VERITY_ML_TEST_SEED": S.REDACTED, "VERITY_ML_TEST_WIDTH": "8"}
    assert secret not in repr(rec)
    assert rec["digest"] == ("dad12bc28e94a6fcac3667bb83cf251f080616014aad3b66f10120d373c81c0b"
                             "d3821538af55f059ab09b5e49fc510a2ec6d827dd93f61841aa41b46d34dcd23")
    assert S.record({"VERITY_ML_TEST_WIDTH": "16"})["digest"] != rec["digest"] and S.record({})["values"] == {}


def _area() -> list[tuple[str, str]]:
    """Each Python file of the area, this one aside (it spells undeclared names to see them refused), with its text."""
    return [(p.relative_to(AREA.parents[2]).as_posix(), p.read_text(encoding="utf-8")) for p in sorted(AREA.rglob("*.py"))
            if p != Path(__file__).resolve()]


def test_the_area_reads_the_environment_through_the_registry_alone() -> None:
    """The shrink-only ratchet on the area's direct reads, at today's count: none.  ``tests/lint/test_repo_env_reads.py`` holds
    the tree's reads to its list, which has no entry here, and this holds the area to none in its own suite."""
    found = []
    for rel, text in _area():
        if "environ" not in text and "getenv" not in text:
            continue
        for n in ast.walk(ast.parse(text, filename=rel)):
            if (isinstance(n, ast.Attribute) and n.attr in OS_ENV
                    or isinstance(n, ast.ImportFrom) and n.module == "os" and any(a.name in OS_ENV for a in n.names)):
                found.append(f"  {rel}:{n.lineno}")
    assert not found, ("the circuits area reads the environment through verity.settings alone: declare the variable there "
                       "(a type, a default, a role and a doc) and read it as settings.NAME\n" + "\n".join(found))


def test_every_name_the_area_reads_is_declared() -> None:
    """Each ``VERITY_ML_*`` name the area spells (a read, a skip reason, a docstring) and each ``settings.NAME`` it reads."""
    spelt, read = set(), set()
    for rel, text in _area():
        spelt |= set(TOKEN.findall(text))
        if "settings" in text:
            read |= {n.attr for n in ast.walk(ast.parse(text, filename=rel)) if isinstance(n, ast.Attribute)
                     and isinstance(n.value, ast.Name) and n.value.id in ("settings", "S") and n.attr.isupper()
                     and n.attr.startswith(S.PREFIXES)}
    assert read and not (spelt | read) - set(S.SETTINGS), sorted((spelt | read) - set(S.SETTINGS))
