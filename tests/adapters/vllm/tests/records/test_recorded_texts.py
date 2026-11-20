"""The constant text a passing run writes into every record of its kind, pinned by name.

A word change alters no recorded text: a rule, a why, a claim, a convention or a scope a record carries keeps its bytes.
Each site is read from the producing module's source (its string parts in order, a formatted field as `{}`), so a branch
no small run reaches is held too.  A fault or refusal message is no site: it reaches a record only when a check fails.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
from pathlib import Path

import pytest

#: name -> (module, function or "" for the module's top level, what): what is `=NAME` (an assignment to NAME, or to a
#: subscript whose key is NAME), `:KEY` (every dict entry under KEY) or `~REGEX` (every assignment to a name matching REGEX)
SITES = {
    "commit_rules.claims": ("verity_vllm.check.commit_rules", "", "=claim"),
    "static_schedule.RULE": ("verity_vllm.engine.static_schedule", "", "=RULE"),
    "run_facts.profile_id_method": ("verity_vllm.engine.run_facts", "profile_manifest", "=profile_id_method"),
}

#: sha256 of each site's text as main's records carry it
PINS = {
    "commit_rules.claims": "07bb035a8edc959bb14bbf48367e4f9afe926d89b7b1208d3fb5a0800815065e",
    "run_facts.profile_id_method": "6d251dc2d14943b52df29ea91d6ef0b0fadfe91708586fa95e5c847db5e15c56",
    "static_schedule.RULE": "618ec92b9680744475481c46b75a86ad7b465e7a02924d0126756b1a8a73d109",
}


def _parts(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return ["".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}"
                        for v in node.values)]
    return [p for c in ast.iter_child_nodes(node) for p in _parts(c)]


def _target_names(t: ast.AST) -> list[str]:
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
        return [t.slice.value]
    if isinstance(t, ast.Tuple):
        return [n for e in t.elts for n in _target_names(e)]
    return []


def text(module: str, function: str, what: str) -> str:
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin, f"{module}: not importable"
    tree = ast.parse(Path(spec.origin).read_text())
    scopes = [tree] if not function else [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                          and n.name == function]
    assert scopes, f"{module}: no function {function}"
    kind, key = what[0], what[1:]
    parts: list[str] = []
    for scope in scopes:
        for n in ast.walk(scope):
            if kind == ":" and isinstance(n, ast.Dict):
                parts += [p for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and k.value == key for p in _parts(v)]
            elif kind in "=~" and isinstance(n, (ast.Assign, ast.AnnAssign)):
                names = [m for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) for m in _target_names(t)]
                if n.value is not None and any(m == key if kind == "=" else re.match(key, m) for m in names):
                    parts += _parts(n.value)
    assert parts, f"{module}{'.' + function if function else ''}: no {what} site; a moved site moves its entry here"
    return "\x00".join(parts)


def digest(name: str) -> str:
    return hashlib.sha256(text(*SITES[name]).encode()).hexdigest()


def test_every_site_is_pinned():
    assert sorted(SITES) == sorted(PINS), "a site and its pin are added together, the pin from main's text"


@pytest.mark.parametrize("name", sorted(SITES))
def test_a_recorded_text_keeps_its_bytes(name):
    assert digest(name) == PINS[name], (
        f"{name}: the text {SITES[name][0]} writes into its records changed. A word change alters no recorded text: put "
        f"main's bytes back (`git show origin/main:<file>`). Only a new record format re-pins, with its schema id bumped.")
