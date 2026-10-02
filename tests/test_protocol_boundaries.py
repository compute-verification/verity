"""Protocols depend inward only: each imports the standard library, the core `verity` and itself; never another
protocol, a backend, an integration, `research` or a third-party package.  The application composes protocols."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "protocols"
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


def _sources(name: str) -> list[Path]:
    """A protocol's own Python files: not its tests, not a Lean package's tooling (the scripts beside a lakefile write
    the package's data, as tests do), and nothing under a dot-directory (a Lean package's `.lake`, a `.venv`), where only
    build outputs and fetched dependencies live."""
    base = ROOT / name
    return [p for p in sorted(base.rglob("*.py"))
            if "tests" not in (parts := p.relative_to(base).parts) and not any(x.startswith(".") for x in parts)
            and not any((base.joinpath(*parts[:i]) / "lakefile.toml").exists() for i in range(1, len(parts)))]


def test_a_lean_packages_tooling_is_not_the_protocols_code():
    assert ROOT / "pouw" / "verity_pouw" / "protocol.py" in _sources("pouw")
    assert not [p for p in _sources("pouw") if (ROOT / "pouw" / "lean") in p.parents]


@pytest.mark.parametrize("name", PROTOCOLS)
def test_protocol_imports_only_the_core(name):
    package = f"verity_{name}"
    allowed = set(sys.stdlib_module_names) | {"verity", package}
    offenders = [f"{path.relative_to(ROOT)}:{line}: {mod}"
                 for path in _sources(name)
                 for mod, line in _imports(path) if mod.split(".")[0] not in allowed]
    assert not offenders, "\n".join(offenders)
