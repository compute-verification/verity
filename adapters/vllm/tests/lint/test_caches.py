"""`conftest.py`: the lints' caches last as long as a test process stays in this directory, and no longer."""
from __future__ import annotations

import ast
import subprocess
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from . import _ratchet as R
from . import conftest as C

HERE = Path(__file__).parent

#: a directory of lints in small: one cache, two rules that read through it, and a test outside the directory
TOY = {
    "pytest.ini": "[pytest]\n",
    "toy/__init__.py": "",
    "toy/lint/__init__.py": "",
    "toy/lint/_parsed.py": "from functools import cache\n\n\n@cache\ndef parsed(name):\n    return [name]\n",
    "toy/lint/test_rules.py": ("from . import _parsed\n\n\n"
                               "def test_the_first_rule_parses():\n"
                               "    _parsed.parsed('a')\n"
                               "    assert _parsed.parsed.cache_info().currsize == 1\n\n\n"
                               "def test_the_second_rule_finds_it_parsed():\n"
                               "    assert _parsed.parsed.cache_info().currsize == 1\n"),
    "toy/test_after.py": ("from toy.lint import _parsed\n\n\n"
                          "def test_nothing_parsed_is_kept():\n"
                          "    assert _parsed.parsed.cache_info().currsize == 0\n"),
}
RULES = ["toy/lint/test_rules.py::test_the_first_rule_parses", "toy/lint/test_rules.py::test_the_second_rule_finds_it_parsed"]
AFTER = "toy/test_after.py::test_nothing_parsed_is_kept"


def _cached(module: str) -> Any:
    """A cache as the module named ``module`` would define it."""
    def read(name: str) -> list[str]:
        return [name]

    read.__module__ = module
    return cache(read)


def _module(name: str, **names: Any) -> ModuleType:
    module = ModuleType(name)
    vars(module).update(names)
    return module


def test_a_modules_own_caches_are_found_and_one_it_imports_is_its_definers() -> None:
    own, borrowed = _cached("importer"), _cached("definer")
    definer = _module("definer", read=borrowed)
    importer = _module("importer", read=own, borrowed=borrowed, plain=len)
    assert C.caches([importer]) == [own]
    assert C.caches([definer, importer]) == [borrowed, own]


def test_the_lints_modules_are_this_directorys_and_hold_the_shared_trees() -> None:
    modules = C.lint_modules()
    assert all(m.__name__ == __package__ or m.__name__.rpartition(".")[0] == __package__ for m in modules)
    assert {R, C, sys.modules[__name__]} <= set(modules)
    assert any(held is R.parse_src for held in C.caches(modules))


def _out_of_reach(source: str) -> list[str]:
    """The `functools` caches of ``source`` on a method or on a function inside another, where `caches` does not look."""
    tree = ast.parse(source)
    top = {id(node) for node in tree.body}
    return [f"line {node.lineno}: @{name} on `{node.name}`" for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and id(node) not in top
            for name in (ast.unparse(d.func if isinstance(d, ast.Call) else d).rpartition(".")[2] for d in node.decorator_list)
            if name in ("cache", "lru_cache")]


def test_every_cache_here_is_one_the_fixture_reaches() -> None:
    """`caches` finds a cache where a module names it at its top level. One on a method or on a function inside another
    would outlive the directory unseen."""
    below = "@cache\ndef top(): ...\n\n\nclass A:\n    @functools.lru_cache(maxsize=8)\n    def method(self): ...\n"
    assert _out_of_reach(below) == ["line 7: @lru_cache on `method`"]
    found = {p.name: lines for p in sorted(HERE.glob("*.py")) if (lines := _out_of_reach(p.read_text(encoding="utf-8")))}
    assert not found, ("a cache below a module's top level, which `conftest.py` does not empty when a test process leaves this "
                       f"directory; cache a top-level function instead: {found}")


@pytest.mark.parametrize(("emptied", "outcomes"), [
    (True, [f"PASSED {RULES[0]}", f"PASSED {RULES[1]}", f"PASSED {AFTER}"]),
    (False, [f"FAILED {AFTER}", f"PASSED {RULES[0]}", f"PASSED {RULES[1]}"]),
], ids=["with this conftest", "without it"])
def test_a_cache_lasts_the_directory_and_is_empty_at_the_first_test_outside_it(tmp_path, emptied: bool, outcomes: list[str]) -> None:
    """Under this directory's `conftest.py` the second rule finds what the first parsed, and the test after the directory
    finds nothing kept. Without it that test finds the entry still held, so the toy tells the two apart."""
    for name, body in TOY.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(body)
    if emptied:
        (tmp_path / "toy" / "lint" / "conftest.py").write_text((HERE / "conftest.py").read_text(encoding="utf-8"))
    r = subprocess.run([sys.executable, "-m", "pytest", "-rA", "-p", "no:cacheprovider", "-p", "no:xdist",
                        "toy/lint/test_rules.py", "toy/test_after.py"], cwd=tmp_path, capture_output=True, text=True)
    lines = sorted(line.split(" - ")[0] for line in r.stdout.splitlines() if line.startswith(("PASSED ", "FAILED ", "ERROR ")))
    assert lines == sorted(outcomes), r.stdout + r.stderr
