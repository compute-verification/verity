"""`settings.py`: every environment variable check reads, declared once (a type, a role, a doc, a `where` setting's reason), read
at the read, the run's record of the value settings, the refusal of an unknown check knob and of a debug setting in a run of
record, and the reads still outside it (`DIRECT`), which only shrink."""
from __future__ import annotations

import ast
import importlib.util
import io
import os
import re
import subprocess
import sys
import tokenize
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


S = _module("check_settings_under_test", HERE / "settings.py")
#: the repository's detector of a direct read (`tests/lint/test_repo_env_reads.py`), so both count alike
LINT = _module("repo_env_reads_for_check", ROOT / "tests" / "lint" / "test_repo_env_reads.py")

#: The direct environment reads left in the package, by file, which only shrink: move a read into `settings.py` and lower its
#: count here.  `slot.py` and `lean_slot.py` are sent alone to a machine as source (``check.py --slots``, ``research.auto``,
#: ``research.remote``), `common.py` is in every Lean package's audit key (`lean_audit.py`), and the read guard (``guard/``)
#: loads in every test process; what each reads is declared in `settings.py` all the same (`test_every_name_read_is_declared`).
DIRECT = {"common.py": 1, "guard/child/sitecustomize.py": 5, "guard/child/verity_suite_reads.py": 2,
          "guard/verity_suite_guard.py": 9, "lean_slot.py": 8, "slot.py": 12}


def _library(suffix: str) -> list[Path]:
    return sorted(p for p in HERE.rglob(f"*{suffix}") if "tests" not in p.relative_to(HERE).parts and "__pycache__" not in p.parts)


def _rel(p: Path) -> str:
    return p.relative_to(HERE).as_posix()


def test_every_setting_has_a_type_a_role_a_doc_and_a_where_setting_its_reason() -> None:
    for s in S.SETTINGS.values():
        assert callable(s.type) and s.role in S.ROLES and s.doc, s
        assert bool(s.reason) == (s.role == S.WHERE), s
    for bad in (S.Setting("CHECK_X", str, None, S.WHERE, "a doc"), S.Setting("CHECK_X", str, None, S.VALUE, "a doc", "why"),
                S.Setting("CHECK_X", str, None, "sometimes", "a doc")):
        with pytest.raises(ValueError, match="CHECK_X"):
            S._declare(bad)
    with pytest.raises(ValueError, match="declared twice"):
        S._declare(*[S.Setting("CHECK_X", str, None, S.VALUE, "a doc")] * 2)


def test_settings_imports_only_the_standard_library() -> None:
    """`slot.py`'s interpreter and every module of the package load it alone, from beside them."""
    tree = ast.parse((HERE / "settings.py").read_text())
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0}
    assert names <= set(sys.stdlib_module_names) | {"__future__"}, names
    assert not any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(tree))


def test_a_read_parses_the_environment_at_the_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHECK_SLOT_RANK", raising=False)
    assert S.CHECK_SLOT_RANK == "" and S.get("CHECK_CACHE", {}) is None and not S.is_set("CHECK_SLOT_RANK")
    monkeypatch.setenv("CHECK_SLOT_RANK", "7")
    assert S.CHECK_SLOT_RANK == "7" and S.raw("CHECK_SLOT_RANK") == "7"
    monkeypatch.setenv("CHECK_SLOT_RANK", "")
    assert S.CHECK_SLOT_RANK == "" and S.is_set("CHECK_SLOT_RANK") and S.raw("CHECK_SLOT_RANK") == ""
    assert S.get("CHECK_CACHE", {"CHECK_CACHE": "/c"}) == "/c" and S.get("CHECK_CACHE", {"CHECK_CACHE": ""}) is None
    with pytest.raises(AttributeError, match="CHECK_NOPE is not a declared setting"):
        S.get("CHECK_NOPE")
    with pytest.raises(AttributeError):
        S.CHECK_NOPE  # noqa: B018
    with pytest.raises(KeyError, match="RESEARCH_RUN_ID"):
        S.require("RESEARCH_RUN_ID", {"RESEARCH_RUN_ID": ""})
    assert S.require("RESEARCH_RUN_ID", {"RESEARCH_RUN_ID": "r1"}) == "r1"


def test_the_record_hashes_the_value_settings_and_carries_the_where_ones_beside() -> None:
    env = {"VERITY_REGRESSION": "1", "VERITY_X": "2", "FLOCK_Y": "3", "VERITY_TEST_CACHE": "/t", "VERITY_SUITE_ROOT": "/s",
           "CHECK_SLOT": "a", "VERITY_API_TOKEN": "dummy", "AWS_ACCESS_KEY_ID": "dummy", "HOME": "/h"}
    r = S.record(env)
    assert r["values"] == {"FLOCK_Y": "3", "VERITY_API_TOKEN": S.REDACTED, "VERITY_REGRESSION": "1", "VERITY_X": "2"}
    assert r["where"] == {"AWS_ACCESS_KEY_ID": S.REDACTED, "CHECK_SLOT": "a", "VERITY_TEST_CACHE": "/t"}
    assert r["schema"] == S.RECORD_SCHEMA and len(r["digest"]) == 128
    assert S.record({**env, "CHECK_SLOT": "b", "HOME": "/g"})["digest"] == r["digest"]
    assert S.record({**env, "VERITY_X": "4"})["digest"] != r["digest"]


def test_roles_take_the_longest_family_and_the_declaration_first() -> None:
    assert [S.role_of(k) for k in ("VERITY_SUITE_ROOT", "VERITY_ANY", "FLOCK_ANY", "VERITY_TEST_CACHE", "VERITY_SKIP_STORE",
                                    "HOME")] == [S.WHERE, S.VALUE, S.VALUE, S.WHERE, S.DEBUG, None]


def test_an_unknown_check_knob_is_refused_by_name_and_a_debug_one_in_a_run_of_record() -> None:
    assert S.unknown({"CHECK_SLOTZ": "", "LEAN_SLOTZ": "", "CHECK_ONLY": "1", "CHECK_SLOTS": "/s", "VERITY_ANY": ""}) == [
        "CHECK_SLOTZ", "LEAN_SLOTZ"]
    [why] = S.refusals({"CHECK_SLOTZ": "/s"})
    assert why.startswith("unknown setting CHECK_SLOTZ: nothing in the tree reads it") and "settings.py" in why
    assert S.refusals({"VERITY_SKIP_STORE": "1"}) == []
    [why] = S.refusals({"VERITY_SKIP_STORE": "1", "RESEARCH_RUN_ID": "r1"})
    assert why.startswith("VERITY_SKIP_STORE=1 in a run of record")
    assert S.refusals({"VERITY_SKIP_STORE": "1"}, of_record=True) == [why]
    assert S.refusals({"VERITY_SKIP_STORE": "", "RESEARCH_RUN_ID": "r1"}) == []


def _cli(env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(HERE / "settings.py"), *args], env=env, capture_output=True, text=True)


def test_the_command_refuses_what_a_train_may_not_take() -> None:
    """`train.sh` runs it (`--of-record`) before it launches anything."""
    base = {k: v for k, v in os.environ.items() if not k.startswith(S.OWNED) and S.role_of(k) != S.DEBUG}
    assert _cli(base, "--of-record").returncode == 0
    p = _cli({**base, "VERITY_SKIP_STORE": "1"}, "--of-record")
    assert p.returncode == 2 and "VERITY_SKIP_STORE=1 in a run of record" in p.stderr
    assert _cli({**base, "VERITY_SKIP_STORE": "1"}).returncode == 0
    p = _cli({**base, "CHECK_SLOTZ": "1"})
    assert p.returncode == 2 and "unknown setting CHECK_SLOTZ" in p.stderr


def test_train_sh_refuses_a_debug_setting_before_anything(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith(S.OWNED)}
    p = subprocess.run(["bash", str(HERE / "train.sh"), "HEAD", "auto"], capture_output=True, text=True, cwd=tmp_path,
                       env={**env, "VERITY_SKIP_STORE": "1"})
    assert p.returncode == 1 and "VERITY_SKIP_STORE=1 in a run of record" in p.stderr, p.stderr
    assert "train: a recorded train takes no debug or unknown check setting" in p.stderr


def test_check_and_the_runner_refuse_an_unknown_knob_at_startup(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    check = _module("check_for_settings", HERE / "check.py")
    monkeypatch.setenv("CHECK_SLOTZ", "1")
    assert check.main([]) == 2
    assert "check: unknown setting CHECK_SLOTZ" in capsys.readouterr().err
    assert check.SUITES.main(["--list"]) == 2
    assert "suites: unknown setting CHECK_SLOTZ" in capsys.readouterr().err


def _settings_named(src: str) -> set[str]:
    """The names `SETTINGS.NAME`, `SETTINGS.get("NAME")`, `.require(...)`, `.raw(...)` and `.is_set(...)` read."""
    out = set()
    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "SETTINGS"
                and n.attr.isupper() and n.attr not in vars(S)):
            out.add(n.attr)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("get", "require", "raw", "is_set")
                and isinstance(n.func.value, (ast.Name, ast.Attribute))
                and (getattr(n.func.value, "id", None) == "SETTINGS" or getattr(n.func.value, "attr", None) == "SETTINGS")
                and n.args and isinstance(n.args[0], ast.Constant)):
            out.add(n.args[0].value)
    return out


OWNED_NAME = re.compile(r"\b(?:CHECK_[A-Z][A-Z0-9_]*|LEAN_SLOT(?:S|_[A-Z][A-Z0-9_]*)?)\b")
#: a shell script's read of a variable with a fallback: ``${NAME:-...}``, ``${NAME:+...}``, ``${NAME-...}``, ``${NAME:?...}``
SHELL_READ = re.compile(r"\$\{([A-Z][A-Z0-9_]*):?[-+?=]")
#: a string's tokens: a literal, and an f-string's text (Python 3.12 tokenizes it apart)
STRING_TOKENS = {tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)}


def _known(name: str) -> bool:
    return name in S.SETTINGS or name in S.ELSEWHERE or S.role_of(name) is not None


def test_every_name_read_is_declared() -> None:
    """Every name the package reads through the registry, every variable its direct reads name (`DIRECT`'s files among them), every
    `CHECK_*` and `LEAN_SLOT*` its Python spells in a string, and every variable its shell scripts read with a fallback."""
    missing: dict[str, set[str]] = {}
    for p in _library(".py"):
        src = p.read_text()
        names = _settings_named(src) | {w.split(" ")[0] for _, w in LINT.reads(src)} - {"<computed>"}
        strings = [t.string for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type in STRING_TOKENS]
        names |= {m for s in strings for m in OWNED_NAME.findall(s)}
        if bad := {n for n in names if not _known(n)}:
            missing[_rel(p)] = bad
    for p in _library(".sh"):
        src = p.read_text()
        if bad := {n for n in {*SHELL_READ.findall(src), *OWNED_NAME.findall(src)} if not _known(n)}:
            missing[_rel(p)] = bad
    assert not missing, f"declare these in tools/verity/check/settings.py (SETTINGS, or ELSEWHERE): {missing}"


def test_direct_reads_only_shrink() -> None:
    """The package reads the environment in `settings.py`; the reads left outside it are `DIRECT`, by file, exactly."""
    found = Counter({_rel(p): n for p in _library(".py") if p.name != "settings.py" and (n := len(LINT.reads(p.read_text())))})
    grown = {f: (n, DIRECT.get(f, 0)) for f, n in found.items() if n > DIRECT.get(f, 0)}
    assert not grown, (f"direct environment reads (found, allowed) {grown}: read the variable through tools/verity/check/settings.py "
                       f"(`SETTINGS.NAME`), declaring it there")
    shrunk = {f: (found.get(f, 0), n) for f, n in DIRECT.items() if found.get(f, 0) < n}
    assert not shrunk, f"direct environment reads (found, allowed) {shrunk}: lower DIRECT in {Path(__file__).name} to what is found"
