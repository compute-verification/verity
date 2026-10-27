"""Ruff's findings on the integration, held to ``allowlists/ruff.json``: ``ruff check`` under ``integrations/vllm/ruff.toml``,
run from ``integrations/vllm`` with the ruff the root ``dev`` group pins.

A finding is keyed ``(file, kind, symbol, detail)`` like the rule lints' violations (``_ratchet.py``): the file relative to
``integrations/vllm``, the rule's code, the enclosing ``Class.function`` (``<module>`` at top level) and ruff's message, with
``from line N`` dropped, so edits elsewhere in a file don't churn the list.  ``ruff check --fix`` fixes some (F401, F541,
RUF100's stale ``# noqa``); delete the entry each fix makes stale.

The allowlist is a ratchet the ratchet gate holds to no growth, and the gate judges a change to this module, to ``ruff.toml``
or to the ruff pin like one to a rule lint's code: it reruns this lint as the change's base has it, the base's config and pin
included, on the change's tree.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from . import _ratchet as R

NAME = "ruff"
RULE = "ruff (integrations/vllm/ruff.toml)"
CONFIG = R.ROOT / "ruff.toml"
LINE_REF = re.compile(r" from line \d+")


@lru_cache(maxsize=None)
def scopes(tree: ast.Module) -> tuple[tuple[int, int, str], ...]:
    """(first line, last line, qualified name) of every def and class, in source order.  A def is a statement, so only
    statement bodies are walked."""
    out: list[tuple[int, int, str]] = []

    def visit(stmts: list[ast.AST], prefix: str) -> None:
        for s in stmts:
            name = prefix
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}.{s.name}" if prefix else s.name
                out.append((s.lineno, s.end_lineno, name))
            for field in ("body", "orelse", "finalbody", "handlers", "cases"):
                visit(getattr(s, field, []), name)

    visit(tree.body, "")
    return tuple(out)


def enclosing(tree: ast.Module, line: int) -> str:
    """The innermost def or class whose lines hold ``line`` (``<module>`` outside any)."""
    return ([name for first, last, name in scopes(tree) if first <= line <= last] or ["<module>"])[-1]


def violation(finding: dict) -> R.Violation:
    path, line = Path(finding["filename"]), finding["location"]["row"]
    code = finding["code"] or "syntax-error"
    symbol = "<module>" if code == "syntax-error" else enclosing(R.parse(path), line)
    return R.Violation(R.rel(path), code, symbol, LINE_REF.sub("", finding["message"]), line)


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    cmd = [sys.executable, "-m", "ruff", "check", "--config", str(CONFIG), "--no-cache", "--exit-zero",
           "--output-format", "json", ".", str(R.DEFINITIONS)]
    r = subprocess.run(cmd, cwd=R.ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (f"{' '.join(cmd[1:])} exited {r.returncode} (no ruff? `uv sync --all-packages`: the root `dev` "
                               f"group pins it):\n{r.stderr[-2000:]}")
    return tuple(violation(f) for f in json.loads(r.stdout))


def test_no_new_findings() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Fix the code (`ruff check --fix integrations/vllm` fixes some), or, where the rule is wrong for that "
                    "line, say why in a `# noqa: CODE` on it.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_findings_are_keyed_by_their_enclosing_definition() -> None:
    tree = ast.parse("import os\nclass A:\n    def f(self):\n        x = 1\n\n    y = 2\ndef g():\n    pass\n")
    assert [enclosing(tree, n) for n in range(1, 9)] == ["<module>", "A", "A.f", "A.f", "A", "A", "g", "g"]
    assert LINE_REF.sub("", "Redefinition of unused `x` from line 32: `x` redefined here") == \
        "Redefinition of unused `x`: `x` redefined here"
