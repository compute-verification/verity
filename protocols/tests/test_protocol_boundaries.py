"""Protocols depend inward only: on the core, the standard library and numpy; never on each other, a backend, an
integration or `research`.  The application (`protocols/example.py`) is the only place that imports several."""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PROTOCOLS = sorted(p.name for p in ROOT.iterdir() if (p / "pyproject.toml").exists())
ALLOWED = set(sys.stdlib_module_names) | {"numpy", "verity"}


@pytest.mark.parametrize("name", PROTOCOLS)
def test_protocol_imports_only_the_core(name):
    package = f"verity_{name}"
    offenders = []
    for path in (ROOT / name / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 else [])
            offenders += [f"{path.name}: {n}" for n in names if n.split(".")[0] not in ALLOWED | {package}]
    assert not offenders, offenders
