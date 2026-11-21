"""`verity_config.verity_settings`, the settings machinery (`EnvSettings`) and the distribution's own registry: every variable declared once (a
type, a role, a doc), an undeclared ``VERITY_ML_*`` variable refused by name at a read, a program's startup check
(`EnvSettings.check`) and `direct_uses`, which a suite's ratchet counts with; `tests/tools/config/test_settings.py` holds the
machinery's reads, records and run-of-record refusal.  And the circuits area, on the soundness path, reads the environment through it alone:
its direct reads are held at today's count, none."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

from verity_config import verity_settings as S

REGISTRY = Path(S.__file__)
AREA = (Path(__file__).resolve().parents[3] / "verity" / "circuits")
ROOT = Path(__file__).resolve().parents[3]
TOKEN = re.compile(r"\bVERITY_ML_[A-Z0-9_]*[A-Z0-9]\b")
#: the attributes of ``os`` that touch the environment
OS_ENV = frozenset({"environ", "environb", "getenv", "getenvb"})


def test_every_setting_has_a_type_a_role_and_a_doc() -> None:
    for name, s in S.REGISTRY.settings.items():
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
        S.EnvSettings(S.__file__, ("VERITY_ML_",), s, s, schema=S.RECORD_SCHEMA)
    with pytest.raises(ValueError, match="role 'telemetry'"):
        S.EnvSettings(S.__file__, ("VERITY_ML_",), S.Setting("VERITY_ML_TEST_X", S.flag, False, "telemetry", "a test's"),
                   schema=S.RECORD_SCHEMA)


def test_flag_reads_both_spellings_and_refuses_the_rest() -> None:
    assert [S.flag(v) for v in ("1", "TRUE", " yes", "on", "0", "False", "no", "OFF")] == [True] * 4 + [False] * 4
    with pytest.raises(ValueError, match="not a flag"):
        S.flag("2")


def test_an_unknown_prefixed_variable_is_refused_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misspelt setting fails the run that carries it, at its first read, instead of running with the default."""
    typo = {"VERITY_ML_BOOLEAN_EXHAUSTVE": "1", "VERITY_ML_BOOLEAN_EXHAUSTIVE": "0", "VERITY_SLOW": "1", "PATH": "/bin"}
    assert S.REGISTRY.unknown(typo) == ["VERITY_ML_BOOLEAN_EXHAUSTVE"]
    with pytest.raises(S.UnknownSetting, match=r"unknown setting VERITY_ML_BOOLEAN_EXHAUSTVE: nothing in the tree reads it.*"
                                               r"tools/config/verity_config/verity_settings.py"):
        S.REGISTRY.validate_environ(typo)
    with pytest.raises(S.UnknownSetting):
        S.get("VERITY_ML_BOOLEAN_EXHAUSTIVE", typo)
    with pytest.raises(S.UnknownSetting):
        S.record(typo)
    S.REGISTRY.validate_environ({"VERITY_SLOW": "1", "VERITY_ML_BOOLEAN_EXHAUSTIVE": "1"})
    monkeypatch.setenv("VERITY_ML_BOOLEAN_EXHAUSTVE", "1")
    with pytest.raises(S.UnknownSetting, match="VERITY_ML_BOOLEAN_EXHAUSTVE"):
        _ = S.VERITY_ML_BOOLEAN_EXHAUSTIVE


def test_check_refuses_an_unknown_name_a_value_that_doesnt_parse_and_a_debug_setting_on_a_run_of_record() -> None:
    r = S.EnvSettings("x.py", ("X_",), S.Setting("X_TRACE", str, None, S.DEBUG, "a trace file"),
                      S.Setting("X_N", int, 3, S.VALUE, "a number"), schema=S.RECORD_SCHEMA)
    r.check({"X_TRACE": "/t"}, of_record=False)
    r.check({"X_TRACE": ""}, of_record=True)
    with pytest.raises(S.DebugOnRecord, match="X_TRACE: debug setting, refused on a run of record"):
        r.check({"X_TRACE": "/t"}, of_record=True)
    with pytest.raises(ValueError, match="X_N='x'"):
        r.check({"X_N": "x"}, of_record=False)
    with pytest.raises(S.UnknownSetting, match="unknown setting X_NN: nothing in the tree reads it"):
        r.check({"X_NN": "1"}, of_record=False)


def test_direct_uses_sees_each_form() -> None:
    src = ("import os\nimport os as o\nfrom os import environ, getenv as g\n"
           "a = os.environ.get('A')\nb = o.environ['B']\nc = environ.get('C')\nd = g('D')\nos.environ['E'] = '1'\n"
           "e = {**os.environ}\nf = os.getpid()\nx.os.environ\n")
    assert S.direct_uses(src) == [(4, "os.environ"), (5, "os.environ"), (6, "os.environ"), (7, "os.getenv"),
                                  (8, "os.environ"), (9, "os.environ")]


def _area() -> list[tuple[str, str]]:
    """Each Python file of the area and of its tests (at `tests/verity/circuits`), this one aside (it spells undeclared names
    to see them refused), with its text."""
    files = [*AREA.rglob("*.py"), *(ROOT / "tests" / "verity" / "circuits").rglob("*.py")]
    return [(p.relative_to(ROOT).as_posix(), p.read_text(encoding="utf-8")) for p in sorted(files)
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
    assert not found, ("the circuits area reads the environment through verity_config.verity_settings alone: declare the variable there "
                       "(a type, a default, a role and a doc) and read it as settings.NAME\n" + "\n".join(found))


def test_every_name_the_area_reads_is_declared() -> None:
    """Each ``VERITY_ML_*`` name the area spells (a read, a skip reason, a docstring) and each ``settings.NAME`` it reads."""
    spelt, read = set(), set()
    declared = S.REGISTRY.settings
    for rel, text in _area():
        spelt |= set(TOKEN.findall(text))
        if "settings" in text:
            read |= {n.attr for n in ast.walk(ast.parse(text, filename=rel)) if isinstance(n, ast.Attribute)
                     and isinstance(n.value, ast.Name) and n.value.id in ("settings", "S") and n.attr.isupper()
                     and n.attr.startswith(S.REGISTRY.prefixes)}
    assert read and not (spelt | read) - set(declared), sorted((spelt | read) - set(declared))
