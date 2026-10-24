"""P14: the environment is read in one place, the settings module ``verity_vllm/envs.py``.

As vLLM reads every variable through ``vllm/envs.py``, ``envs.py`` names each variable the package reads once: its type
and default in the ``TYPE_CHECKING`` block, what it does in the comment above its entry in ``environment_variables``.
Code reads it as ``envs.NAME`` after ``from verity_vllm import envs``, which reads the environment at that moment.

Checked:

* ``environ-read`` (against ``allowlists/p14_settings.json``): a read of ``os.environ``/``os.getenv``/... outside
  ``envs.py``, keyed by the variable when it is a literal (as P7 keys it). Writes (a child's environment, vLLM's pins
  in ``engine/env.py``) are not reads; P7 checks them.
* no ``from verity_vllm.envs import NAME``: it would read the variable once, at import.
* no ``envs.NAME`` for a ``NAME`` ``envs.py`` doesn't declare.
* ``envs.py`` declares each variable once, with a type, a default and a comment.

A read through ``envs`` is still an undeclared input for P7 (``environ``, keyed by the variable), so moving a read into
``envs.py`` deletes its entry here and leaves P7's allowlist as it is.
"""
from __future__ import annotations

import ast
from functools import lru_cache

import pytest

from . import _ratchet as R
from .test_p07_declared_inputs import ENV_NAMES, _env_detail

NAME = "p14_settings"
RULE = "P14 (the environment is read in verity_vllm/envs.py)"
SETTINGS_FILE = "verity_vllm/envs.py"
SETTINGS = "verity_vllm.envs"
WRITERS = frozenset({"os.putenv", "os.unsetenv"})
WRITE_METHODS = frozenset({"setdefault", "update", "pop", "popitem", "clear", "__setitem__", "__delitem__"})


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    return {id(c): n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}


def _is_write(origin: str, node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    parent = parents.get(id(node))
    if origin in WRITERS:
        return True
    if isinstance(parent, ast.Subscript) and parent.value is node:
        return isinstance(parent.ctx, (ast.Store, ast.Del))
    return isinstance(parent, ast.Attribute) and parent.attr in WRITE_METHODS


def file_violations(src: str, rel: str) -> list[R.Violation]:
    """``environ-read`` for each read of the environment outside the settings module."""
    if rel == SETTINGS_FILE or not any(s in src for s in ("environ", "getenv")):
        return []
    tree = R.parse_src(src)
    aliases = R.aliases(tree)
    parents = _parents(tree)
    out = []
    for node, symbol in R.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute)):
            continue
        origin = aliases.resolve(node)
        parent = parents.get(id(node))
        if origin not in ENV_NAMES or (isinstance(parent, ast.Attribute) and aliases.resolve(parent) in ENV_NAMES):
            continue
        if not _is_write(origin, node, parents):
            out.append(R.Violation(rel, "environ-read", symbol, _env_detail(node, parents), node.lineno))
    return out


def settings_uses(src: str, rel: str) -> list[tuple[str, int, str]]:
    """(form, line, name) for each ``from verity_vllm.envs import NAME`` ("from-import") and each ``envs.NAME`` ("read")."""
    if "envs" not in src:
        return []
    tree = R.parse_src(src)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == SETTINGS:
            out += [("from-import", node.lineno, a.name) for a in node.names]
    aliases = R.aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            origin = aliases.resolve(node) or ""
            if origin.rpartition(".")[0] == SETTINGS:
                out.append(("read", node.lineno, node.attr))
    return out


def declared(src: str) -> tuple[dict[str, ast.AnnAssign], dict[str, int], list[str]]:
    """The settings module's ``TYPE_CHECKING`` declarations, its ``environment_variables`` keys (name -> line), and its
    source lines."""
    tree = ast.parse(src)
    types: dict[str, ast.AnnAssign] = {}
    keys: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            types.update({s.target.id: s for s in node.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)})
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "environment_variables":
            assert isinstance(node.value, ast.Dict), "environment_variables is a dict literal"
            keys.update({k.value: k.lineno for k in node.value.keys if R.is_str(k)})
    return types, keys, src.splitlines()


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out: list[R.Violation] = []
    for p in R.library_files():
        out += file_violations(R.source(p), R.rel(p))
    return tuple(out)


def test_no_new_reads_outside_the_settings_module() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    f"Read the variable as envs.NAME (`from verity_vllm import envs`), declaring it in {SETTINGS_FILE} with "
                    "its type, default and a comment, or take it as an argument.")


@pytest.mark.slow
def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_settings_are_read_as_declared_variables() -> None:
    types, keys, _ = declared(R.source(R.PKG / "envs.py"))
    bad = []
    for p in R.library_files():
        for form, line, name in settings_uses(R.source(p), R.rel(p)):
            if form == "from-import":
                bad.append(f"  {R.rel(p)}:{line}: `from {SETTINGS} import {name}` reads it once, at import: "
                           f"`from verity_vllm import envs` and read envs.{name} where it is used")
            elif name not in keys and name not in ("environment_variables",):
                bad.append(f"  {R.rel(p)}:{line}: envs.{name} is not declared in {SETTINGS_FILE}: declare it there")
    assert not bad, f"{RULE}:\n" + "\n".join(bad)


def test_each_variable_is_declared_once_typed_defaulted_and_documented() -> None:
    types, keys, lines = declared(R.source(R.PKG / "envs.py"))
    bad = [f"  {n}: in environment_variables, not in the TYPE_CHECKING block (add `{n}: <type> = <default>`)"
           for n in sorted(set(keys) - set(types))]
    bad += [f"  {n}: in the TYPE_CHECKING block, not in environment_variables" for n in sorted(set(types) - set(keys))]
    bad += [f"  {n}: no default in the TYPE_CHECKING block" for n, a in sorted(types.items()) if a.value is None]
    bad += [f"  {n}: no comment on the line above its entry in environment_variables" for n, line in sorted(keys.items())
            if not lines[line - 2].strip().startswith("#")]
    assert not bad, f"{RULE}: {SETTINGS_FILE}:\n" + "\n".join(bad)


def test_detectors_on_examples() -> None:
    src = ("import os\nfrom os import environ\nfrom verity_vllm import envs\nfrom verity_vllm.envs import CXX\n"
           "def run():\n    a = os.environ.get('A')\n    os.environ['B'] = '1'\n    del environ['C']\n"
           "    d = os.getenv('D')\n    os.environ.setdefault('E', '1')\n    f = dict(os.environ)\n    g = envs.G\n")
    got = [(v.kind, v.symbol, v.detail) for v in file_violations(src, "verity_vllm/x.py")]
    assert sorted(got) == [("environ-read", "run", "A"), ("environ-read", "run", "D"), ("environ-read", "run", "os.environ")]
    assert file_violations(src, SETTINGS_FILE) == []
    assert sorted(settings_uses(src, "verity_vllm/x.py")) == [("from-import", 4, "CXX"), ("read", 12, "G")]
    types, keys, lines = declared('if TYPE_CHECKING:\n    A: int = 1\nenvironment_variables: dict = {\n    # A.\n'
                                  '    "A": lambda: 1,\n}\n')
    assert set(types) == {"A"} and keys == {"A": 5} and lines[keys["A"] - 2].strip() == "# A."
