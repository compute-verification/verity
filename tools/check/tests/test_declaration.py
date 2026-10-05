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


def test_check_runs_the_groups_order_and_halts_as_they_were_before_ci_toml_held_them(tmp_path):
    runs = [[name for name, _argv, _cwd in g] for g in CHECK.groups(tmp_path, 2)]
    assert runs == [["pytest"], ["flock-circuit-build", "circuit-check", "flock-rust-tests"],
                    ["lean-build", "lean-unit-cut", "lean-agreement"], ["lean-audit"], ["lean-changed"], ["lean-suites"]]
    assert CHECK.HALTS == {"lean-changed"} and CHECK.HALTS_ON == {"flock-circuit-build": "error: could not compile"}
    assert CHECK.AFTER == {"lean-suites": "lean-build"}


def test_what_a_merge_requires_matches_tool_py():
    assert {j.name: list(j.merge_requires) for j in D.load() if j.merge_requires} == TOOL.CHECK.merge_requires


def test_moving_a_job_in_the_declaration_moves_it_in_check(tmp_path, monkeypatch):
    moved = [D.Job(j.name, j.owner, 9 if j.name == "circuit-check" else j.group, j.halts, j.halts_on, j.merge_requires, j.after)
             for j in CHECK.JOBS]
    monkeypatch.setattr(CHECK, "JOBS", moved)
    gs = CHECK.groups(tmp_path, 2)
    assert [n for n, _, _ in gs[1]] == ["flock-circuit-build", "flock-rust-tests"] and [n for n, _, _ in gs[-1]] == ["circuit-check"]


def test_a_job_check_builds_but_the_declaration_omits_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(CHECK, "JOBS", [j for j in CHECK.JOBS if j.name != "flock-rust-tests"])
    with pytest.raises(CHECK.DECLARATION.DeclarationError, match="flock-rust-tests"):
        CHECK.groups(tmp_path, 2)


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
    ('[[job]]\nname = "a"\nowner = "ci"\ngroup = 1\nafter = "b"\n', "after 'b', which isn't declared"),
    ('[[job]]\nname = "a"\nowner = "ci"\ngroup = 1\n[[job]]\nname = "b"\nowner = "ci"\ngroup = 1\nafter = "a"\n', "in its own group"),
    ('[[job]]\nname = "a"\nowner = "ci"\ngroup = 1\nafter = "c"\n[[job]]\nname = "b"\nowner = "ci"\ngroup = 2\nafter = "a"\n'
     '[[job]]\nname = "c"\nowner = "ci"\ngroup = 3\n', "which itself waits on 'c'"),
])
def test_a_malformed_declaration_is_refused_with_the_reason(body, why):
    with pytest.raises(D.DeclarationError, match=why.replace("[", r"\[").replace("]", r"\]").replace("(", r"\(")):
        _parse(body)


def test_a_wrong_schema_is_refused():
    with pytest.raises(D.DeclarationError, match="schema"):
        D.parse('schema = "ci/v9"\n[[job]]\nname = "a"\nowner = "ci"\ngroup = 1\n', HANDLES)


def _run_two_groups(tmp_path, monkeypatch, first_status: str):
    """check.run on two one-step groups, `b` declared after `a`: the order the steps started in, and b's record."""
    started, b_waiting = [], CHECK.threading.Event()

    class Event(CHECK.threading.Event):
        def wait(self, timeout=None):
            b_waiting.set()
            return super().wait(timeout)

    def step(name, argv, cwd, out, cache, steps):
        started.append(name)
        if name == "a":
            b_waiting.wait()                      # b reached its wait before a ends, or started without one
        else:
            b_waiting.set()
        return steps.end({"step": name, "status": first_status if name == "a" else "passed", "seconds": 0.0})

    monkeypatch.setattr(CHECK, "AFTER", {"b": "a"})
    monkeypatch.setattr(CHECK.threading, "Event", Event)
    monkeypatch.setattr(CHECK, "preflight", lambda: [])
    monkeypatch.setattr(CHECK, "take_lean_deps_urls", lambda out: None)
    monkeypatch.setattr(CHECK.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(CHECK, "groups", lambda out, n, cache, urls: [[("a", ["true"], tmp_path)], [("b", ["true"], tmp_path)]])
    monkeypatch.setattr(CHECK, "_step", step)
    ok, results = CHECK.run(tmp_path, 2)
    return started, {r["step"]: r for r in results}


def test_a_step_declared_after_another_groups_step_starts_only_once_it_has_passed(tmp_path, monkeypatch):
    started, by = _run_two_groups(tmp_path, monkeypatch, "passed")
    assert started == ["a", "b"] and by["b"]["status"] == "passed"


def test_a_step_declared_after_a_step_that_failed_is_stopped_without_starting(tmp_path, monkeypatch):
    started, by = _run_two_groups(tmp_path, monkeypatch, "failed")
    assert started == ["a"] and by["b"]["status"] == "stopped" and "not started: a failed" in by["b"]["detail"]
