"""No assert in a test holds the whole environment (the testing skill's "Environment by name"). A failing assert prints
the values of its sub-expressions (the sides of a comparison, and each operand, argument and attribute's object on the
way to them) into the suite's log, and `suites.py` prints a failed suite's log into its run's, which is published. The
runner hands a suite the environment of whoever started it (`suites.py`'s `run`), credentials included, so an assert that
holds `os.environ` prints them when it fails. An AST scan over every suite's test tree refuses one: in an assert's
expression or its message, `os.environ` is read only by name, as `os.environ[NAME]`, `os.environ.get(NAME)` or `NAME in
os.environ`. Compare an environment by the names that differ, taken before the assert (``FIX``).

What may stay is listed in ``ALLOWED``, one line of reason each: a whole file (``path``) or one top-level function, class
or assigned name in it (``path::name``). An entry that no longer matches anything fails too.

The scan reads the assert alone and follows no name: an environment held in a variable (``env = dict(os.environ)``, then
``assert got == env``) passes it and prints all the same. Of the three forms it lets through, only ``os.environ[NAME]``
prints one variable alone: for the other two pytest prints the environment's repr as well, cut to its two ends unless
``-vv`` asks for all of it."""

from __future__ import annotations

import ast

import pytest
from test_no_wall_clock import _dotted, _test_sources
from test_repository import ROOT

#: the environment, as `os` names it
ENVIRON = {"environ", "environb"}
FIX = """Compare an environment by name, before the assert:

    want = {**os.environ, NAME: value}
    differ = sorted(k for k in got.keys() | want.keys() if got.get(k) != want.get(k))
    assert differ == []

and read one variable as os.environ[NAME]"""

ALLOWED: dict[str, str] = {
    "adapters/vllm/tests/engine/test_env.py::test_an_inherited_dynamic_tiling_is_refused_by_name_before_anything_is_pinned":
        "takes the set of the environment's names inside the assert; it goes when the set is taken before it",
    "adapters/vllm/tests/test_settings.py::test_setdefault_keeps_what_the_environment_holds_and_environ_is_a_copy":
        "holds the settings' copy to the whole environment with one variable set; it goes when the comparison is by name",
    "adapters/vllm/tests/test_settings_records.py::test_the_commit_stages_as_before":
        "digests the whole environment inside the assert; it goes when the digest is taken before it",
    "benchmarks/one_stage/tests/test_verifier_env.py::test_a0s_lean_verify_runs_under_the_allocator_setting_unless_the_caller_sets_its_own":
        "holds the child's environment to the caller's whole one with the allocator's setting; it goes when the comparison is by name",
    "benchmarks/one_stage/tests/test_verifier_env.py::test_server_rates_lean_runs_under_the_allocator_setting_unless_the_caller_sets_its_own":
        "holds the child's environment to the caller's whole one with the allocator's setting; it goes when the comparison is by name",
}


def _by_name(node: ast.AST, parent: ast.AST, grandparent: ast.AST | None) -> bool:
    """Whether `parent` reads the environment `node` by name: `node[...]`, `node.get(...)`, `... in node`."""
    if isinstance(parent, ast.Subscript):
        return parent.value is node
    if isinstance(parent, ast.Attribute):
        return parent.attr == "get" and isinstance(grandparent, ast.Call) and grandparent.func is parent
    if isinstance(parent, ast.Compare):
        return any(right is node and isinstance(op, (ast.In, ast.NotIn)) for op, right in zip(parent.ops, parent.comparators))
    return False


def held(source: str) -> list[tuple[str, int]]:
    """(top-level name or "<module>", line) for each place an assert of `source` holds the environment whole."""
    tree = ast.parse(source)
    modules, names = {"os"}, set()             # local names of `os`, by `import os as o`; of the environment, by `from os import environ`
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.asname or a.name for a in node.names if a.name == "os"}
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            names |= {a.asname or a.name for a in node.names if a.name in ENVIRON}

    def environ(node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute):
            return node.attr in ENVIRON and _dotted(node.value).split(".")[-1] in modules      # `os.environ`, `mod.os.environ`
        return isinstance(node, ast.Name) and node.id in names

    out = set()
    for top in tree.body:
        where = (top.name if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                 else top.targets[0].id if isinstance(top, ast.Assign) and isinstance(top.targets[0], ast.Name) else "<module>")
        for check in ast.walk(top):
            if not isinstance(check, ast.Assert):
                continue
            parents = {child: node for node in ast.walk(check) for child in ast.iter_child_nodes(node)}
            out |= {(where, node.lineno) for node in parents
                    if environ(node) and not _by_name(node, parents[node], parents.get(parents[node]))}
    return sorted(out, key=lambda o: o[1])


def test_no_assert_holds_the_whole_environment() -> None:
    offenders, used = [], set()
    for path in _test_sources():
        source = (ROOT / path).read_text(encoding="utf-8")
        for where, line in held(source) if "environ" in source else ():
            key = next((k for k in (f"{path}::{where}", path) if k in ALLOWED), None)
            if key:
                used.add(key)
            else:
                offenders.append(f"{path}:{line} ({where})")
    assert not offenders, ("asserts that hold the whole environment: a failing assert prints its sub-expressions' values, and a "
                           f"run's log is published, the credentials of whoever started it among them. {FIX}; or add an "
                           "ALLOWED entry with its reason:\n" + "\n".join(offenders))
    assert not set(ALLOWED) - used, f"ALLOWED entries that match nothing any more: {sorted(set(ALLOWED) - used)}"


@pytest.mark.parametrize("check", [
    'got == os.environ',
    'got == {**os.environ, "A": "1"}',
    'got == [{**os.environ, "A": "1"}]',
    'got == {**os.environ, "A": "1"} and got["A"] == "1", "said"',
    'got == dict(os.environ, A="1")',
    'got == os.environ.copy()',
    'got == os.environ | {"A": "1"}',
    'got == sorted(os.environ.items())',
    'differing(got, os.environ) == []',
    'not differing(got, {**os.environ, "A": "1"})',
    '"A" in set(os.environ)',
    '"A" in os.environ.keys()',
    'os.environ.pop("A") == "1"',
    'got == [os.environ.get(name) for name in os.environ]',
    'not os.environ',
    'got, f"got {got} of {os.environ}"',
])
def test_the_scan_refuses_the_environment_held_whole(check: str) -> None:
    assert held(f"import os\n\n\ndef test_it(got):\n    assert {check}\n") == [("test_it", 5)]


@pytest.mark.parametrize("check", [
    'os.environ["A"] == "1"',
    'os.environ.get("A") == "1"',
    'os.environ.get("A", "") == "" and "B" not in os.environ',
    '"A" in os.environ',
    'got == {"A": os.environ["A"], "B": os.environ.get("B")}',
    'got, f"got {got}, not {os.environ["A"]}"',
    'os.getenv("A") is None',
    'differ == []',
])
def test_the_scan_passes_the_environment_read_by_name(check: str) -> None:
    assert held(f"import os\n\n\ndef test_it(got, differ):\n    assert {check}\n") == []


def test_the_scan_follows_the_names_of_os_and_says_where() -> None:
    src = '''
import os as o
from os import environ as e, environb
import mod
assert e == {}

def test_a(got):
    def child(cmd, env):
        assert env == {**o.environ}
    assert got == dict(mod.os.environ) and e["A"] == "1"
    assert got == environb, "said"
    assert got == mod.environ and got == environ

class TestB:
    def test_c(self, got):
        assert got == [dict(e)]

WANT = [1 for _ in range(1) if e]
'''
    assert held(src) == [("<module>", 5), ("test_a", 9), ("test_a", 10), ("test_a", 11), ("TestB", 16)]
