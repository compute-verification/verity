"""The constant text and keys a passing run writes into the auditor's line and PoUB's replay rows, pinned by name.

A word change alters no recorded text: a premise, a why, a credit's reason or a record's key keeps its bytes. Each site is
read from its file's source (its string parts in order, a formatted field as `{}`), so a branch no small run reaches is
held too. A fault or refusal message is no site: it reaches a record only when a check fails.
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

#: name -> (file, function or "" for the module's top level, what): what is `=NAME` (an assignment to NAME, or to a
#: subscript whose key is NAME), `:KEY` (every dict entry under KEY), `~REGEX` (every assignment to a name matching REGEX),
#: `^` (every value the function returns; `^True`, every `return True, why`'s why), `(NAME` (the first argument of every
#: call to NAME) or `{KEY` (the keys of every dict with a KEY entry)
SITES = {
    "auditor_line.CONDITIONAL_ON": ("benchmarks/remote_session/auditor_line.py", "", "=CONDITIONAL_ON"),
    "auditor_line.GROUNDED": ("benchmarks/remote_session/auditor_line.py", "", "=GROUNDED"),
    "auditor_line.PREMISES": ("benchmarks/remote_session/auditor_line.py", "", "=PREMISES"),
    "auditor_line.UNBOUND": ("benchmarks/remote_session/auditor_line.py", "", "=UNBOUND"),
    "auditor_line.analyse.no_credit": ("benchmarks/remote_session/auditor_line.py", "analyse", "(no_credit"),
    "auditor_line.analyse.skipped": ("benchmarks/remote_session/auditor_line.py", "analyse", "=skipped"),
    "replay.recorded": ("verity/protocols/accounting/poub/verifier/lean/replay.py", "recorded", "^"),
    "replay.held.passing": ("verity/protocols/accounting/poub/verifier/lean/replay.py", "held", "^True"),
    "squaring.plans.keys": ("benchmarks/property_redteam/squaring.py", "main", "{seam_limb"),
}

#: sha256 of each site's text as main's records carry it, but where a move renamed a cited id on purpose
PINS = {
    # `tools/move/device_tokens_map.toml` (`5fd12da779d`) gave the assumption the id trusted-device wherever a string
    # holds it: these two cite the new id, and differ from main's by that word alone
    "auditor_line.CONDITIONAL_ON": "39cda10a9dfcebf0072503b7b55abeeb82ffac02373744da65d0ecc87248eceb",
    "auditor_line.GROUNDED": "e323302f4027b0752544dd4bdfa9d81dd63ab17cfa17cbbb8c5b28ec7bf287af",
    "auditor_line.PREMISES": "9e01ee2d90f0c1488b589082d6cfedd974f8ed030b7fd17828721527de87dbdf",
    "auditor_line.UNBOUND": "996c0a8ea6e56c8e9c310174e0b13f65879bfe26530b228c3a9c97907227f705",
    "auditor_line.analyse.no_credit": "cc2761511a943b54459836ef8adda004b2990619447713a0c184e638aea1f1c9",
    "auditor_line.analyse.skipped": "3823f9504a34b69025269b825800533b0bdc93e62eeef730891fc95571296568",
    "replay.held.passing": "13095915a03e8e568f63bcdca235d05a500d383c62dd0837661f4c22852c266a",
    "replay.recorded": "6cab98c5fe6182db58c1e9f5bf7847033b0820171d707d9c6dff3fb65e72c52a",
    "squaring.plans.keys": "37453880b4d1d52786e34dd0cde12a3e2205f2e566bcfd87789cfe25b6c2b267",
}


def _parts(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return ["".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)]
    return [p for c in ast.iter_child_nodes(node) for p in _parts(c)]


def _target_names(t: ast.AST) -> list[str]:
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
        return [t.slice.value]
    if isinstance(t, ast.Tuple):
        return [n for e in t.elts for n in _target_names(e)]
    return []


def _returned(n: ast.Return, key: str) -> list[str]:
    if not key:
        return _parts(n.value)
    v = n.value
    if (isinstance(v, ast.Tuple) and v.elts and isinstance(v.elts[0], ast.Constant)
            and repr(v.elts[0].value) == key):
        return [p for e in v.elts[1:] for p in _parts(e)]
    return []


def text(path: str, function: str, what: str, source: str | None = None) -> str:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8") if source is None else source)
    scopes = [tree] if not function else [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                          and n.name == function]
    assert scopes, f"{path}: no function {function}"
    kind, key = what[0], what[1:]
    parts: list[str] = []
    for scope in scopes:
        for n in ast.walk(scope):
            if kind == ":" and isinstance(n, ast.Dict):
                parts += [p for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and k.value == key for p in _parts(v)]
            elif kind == "{" and isinstance(n, ast.Dict):
                keys = [k.value for k in n.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                parts += keys if key in keys else []
            elif kind == "(" and isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == key and n.args:
                parts += _parts(n.args[0])
            elif kind == "^" and isinstance(n, ast.Return) and n.value is not None:
                parts += _returned(n, key)
            elif kind in "=~" and isinstance(n, (ast.Assign, ast.AnnAssign)):
                names = [m for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) for m in _target_names(t)]
                if n.value is not None and any(m == key if kind == "=" else re.match(key, m) for m in names):
                    parts += _parts(n.value)
    assert parts, f"{path}{'.' + function if function else ''}: no {what} site; a moved site moves its entry here"
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
