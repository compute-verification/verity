"""P13: the public modules type-check under ``mypy --strict``.

``PUBLIC`` lists the files of the integration's public interfaces (note:20261024T0615Z-draft-verity-vllm-vision,
sections 3 and 5): the package root and ``LLM``, the engine spec, and the patch owner. When an interface becomes
public (``PUBLIC`` in ``tests/test_vllm_public_api.py`` at the repository root), add its file here.

mypy checks those files with ``--strict``. Imports inside ``verity_vllm`` are followed silently, for their types; every
other import (``verity``, torch, vLLM) is ``Any``, so the check is of the interfaces themselves and costs a few seconds.
An error is keyed ``(file, error code, enclosing def, message)`` against ``allowlists/p13_public_types.json``: a new
error fails (fix it: annotate, narrow, or cast at the boundary), and an entry whose error is gone fails until it is
deleted, so the allowlist only shrinks. mypy is the workspace's dev group's (``uv sync``), below 2, so its messages move
only in a PR that bumps it.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from . import _ratchet as R

NAME = "p13_public_types"
RULE = "P13 (the public modules type-check)"
PUBLIC = ("verity_vllm/__init__.py", "verity_vllm/llm.py", "verity_vllm/engine/spec.py", "verity_vllm/engine/hooks.py")
CONFIG = """\
[mypy]
python_version = 3.12
follow_imports = skip
ignore_missing_imports = True

[mypy-verity_vllm.*]
follow_imports = silent
"""


def enclosing(tree: ast.Module, line: int) -> str:
    """The qualified name of the innermost def or class whose span holds ``line`` (``<module>`` at top level)."""
    best = "<module>"
    for node, symbol in R.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.lineno <= line <= node.end_lineno:
            best = node.name if symbol == "<module>" else f"{symbol}.{node.name}"
    return best


def violations(report: str, trees: dict[str, ast.Module]) -> list[R.Violation]:
    """One violation per error line of mypy's ``-O json`` report; notes are dropped and ``line N`` in a message is
    normalized, so that edits elsewhere in a file don't churn the key."""
    out = []
    for raw in report.splitlines():
        if not raw.startswith("{"):
            continue
        e = json.loads(raw)
        if e.get("severity") != "error":
            continue
        file = Path(e["file"]).as_posix()
        message = re.sub(r"\bline \d+\b", "line N", e["message"])
        symbol = enclosing(trees[file], e["line"]) if file in trees else "<module>"
        out.append(R.Violation(file, e.get("code") or "error", symbol, message, e["line"]))
    return out


@pytest.fixture(scope="module")
def found(tmp_path_factory: pytest.TempPathFactory) -> tuple[R.Violation, ...]:
    tmp = tmp_path_factory.mktemp("mypy")
    (tmp / "mypy.ini").write_text(CONFIG, encoding="utf-8")
    res = subprocess.run([sys.executable, "-m", "mypy", "--strict", "--config-file", str(tmp / "mypy.ini"), "--cache-dir",
                          str(tmp / "cache"), "-O", "json", "--no-error-summary", *PUBLIC],
                         cwd=R.ROOT, capture_output=True, text=True)
    assert res.returncode in (0, 1), (f"{RULE}: mypy did not run (exit {res.returncode}); `uv sync` installs it with the "
                                      f"workspace's dev group.\n{res.stdout[-2000:]}{res.stderr[-2000:]}")
    return tuple(violations(res.stdout, {f: R.parse(R.ROOT / f) for f in PUBLIC}))


def test_the_public_files_exist() -> None:
    missing = [f for f in PUBLIC if not (R.ROOT / f).is_file()]
    assert not missing, f"{RULE}: PUBLIC names files that don't exist (a move renames them here too): {missing}"


def test_no_new_errors(found: tuple[R.Violation, ...]) -> None:
    R.assert_no_new(NAME, found, RULE, "Annotate, narrow or cast at the boundary until mypy --strict passes.")


def test_allowlist_has_no_stale_entries(found: tuple[R.Violation, ...]) -> None:
    R.assert_no_stale(NAME, found, RULE)


def test_violations_from_a_report() -> None:
    src = "def f(x):\n    return x\n\nclass K:\n    def g(self) -> int:\n        return 'a'\n"
    report = "\n".join(json.dumps(e) for e in [
        {"file": "verity_vllm/x.py", "line": 1, "column": 0, "message": "Function is missing a type annotation",
         "hint": None, "code": "no-untyped-def", "severity": "error"},
        {"file": "verity_vllm/x.py", "line": 6, "column": 15, "message": "Incompatible return value (see line 5)",
         "hint": None, "code": "return-value", "severity": "error"},
        {"file": "verity_vllm/x.py", "line": 6, "column": 15, "message": "a note", "hint": None, "code": None,
         "severity": "note"},
    ])
    got = violations("noise\n" + report, {"verity_vllm/x.py": ast.parse(src)})
    assert [v.key for v in got] == [
        ("verity_vllm/x.py", "no-untyped-def", "f", "Function is missing a type annotation"),
        ("verity_vllm/x.py", "return-value", "K.g", "Incompatible return value (see line N)"),
    ]
