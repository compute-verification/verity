"""Protocols depend inward only: each imports the standard library, the core `verity` and itself; never another
protocol, a backend, an integration, `research` or a third-party package.  The application composes protocols."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = sorted(p.name for p in ROOT.iterdir() if (p / "pyproject.toml").exists())


def _imports(path: Path) -> list[tuple[str, int]]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            out += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module, node.lineno))
    return out


def test_every_protocol_is_checked():
    assert "sampled_proofs" in PROTOCOLS


@pytest.mark.parametrize("name", PROTOCOLS)
def test_protocol_imports_only_the_core(name):
    package = f"verity_{name}"
    allowed = set(sys.stdlib_module_names) | {"verity", package}
    offenders = [f"{path.relative_to(ROOT)}:{line}: {mod}"
                 for path in sorted((ROOT / name).rglob("*.py")) if "tests" not in path.relative_to(ROOT / name).parts
                 for mod, line in _imports(path) if mod.split(".")[0] not in allowed]
    assert not offenders, "\n".join(offenders)
