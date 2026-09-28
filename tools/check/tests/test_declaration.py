"""`tools/check/ci.toml` declares exactly the jobs check.py runs (names, groups and order, what halts the run, what a merge
requires), and `declaration.py` refuses a malformed declaration rather than guessing."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                      # dataclasses look their module up while the class is built
    spec.loader.exec_module(mod)
    return mod


D = _module("repo_declaration", ROOT / "tools" / "check" / "declaration.py")
CHECK = _module("repo_check_for_declaration", ROOT / "tools" / "check" / "check.py")
TOOL = _module("repo_check_tool", ROOT / "tools" / "check" / "tool.py")
HANDLES = {"ci", "lean", "proofs"}


def test_the_declaration_names_check_pys_jobs_in_their_groups_and_order(tmp_path):
    declared: dict[int, list[str]] = {}
    for j in D.load():
        declared.setdefault(j.group, []).append(j.name)
    runs = [[name for name, _argv, _cwd in g] for g in CHECK.groups(tmp_path, 2)]
    assert [declared[k] for k in sorted(declared)] == runs


def test_what_halts_the_run_and_what_a_merge_requires_match_check_py_and_tool_py():
    jobs = D.load()
    assert {j.name for j in jobs if j.halts} == set(CHECK.HALTS)
    assert {j.name: j.halts_on for j in jobs if j.halts_on} == dict(CHECK.HALTS_ON)
    assert {j.name: list(j.merge_requires) for j in jobs if j.merge_requires} == TOOL.CHECK.merge_requires


def _parse(body: str):
    return D.parse('schema = "ci/v0"\n' + body, HANDLES)


def test_a_well_formed_job_parses_with_its_defaults():
    (j,) = _parse('[[job]]\nname = "a"\nowner = "ci"\ngroup = 1\n')
    assert j == D.Job("a", "ci", 1)


@pytest.mark.parametrize("body, why", [
    ('[[job]]\nname = "a"\nowner = "ci"\n', "missing group"),
    ('[[job]]\nname = "a"\nowner = "ci"\ngroup = 1\nshape = "big"\n', "unknown keys ['shape']"),
    ('[[job]]\nname = "a"\nowner = "nobody"\ngroup = 1\n', "owner 'nobody' isn't a handle"),
    ('[[job]]\nname = "a"\nowner = "ci"\ngroup = true\n', "group must be int"),
    ('[[job]]\nname = "a"\nowner = "ci"\ngroup = 0\n', "group must be 1 or more"),
    ('[[job]]\nname = "a"\nowner = "ci"\ngroup = 1\n[[job]]\nname = "a"\nowner = "lean"\ngroup = 2\n', "duplicate name"),
    ('[[job]]\nname = "a"\nowner = "ci"\ngroup = 1\nmerge_requires = [""]\n', "non-empty path strings"),
    ('', "declares no job"),
    ('extra = 1\n[[job]]\nname = "a"\nowner = "ci"\ngroup = 1\n', "unknown keys ['extra']"),
])
def test_a_malformed_declaration_is_refused_with_the_reason(body, why):
    with pytest.raises(D.DeclarationError, match=why.replace("[", r"\[").replace("]", r"\]").replace("(", r"\(")):
        _parse(body)


def test_a_wrong_schema_is_refused():
    with pytest.raises(D.DeclarationError, match="schema"):
        D.parse('schema = "ci/v9"\n[[job]]\nname = "a"\nowner = "ci"\ngroup = 1\n', HANDLES)
