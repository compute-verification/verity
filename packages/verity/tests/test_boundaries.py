"""The core dependency rules (README "Development") as import-boundary checks over the source tree.

    ir <- ml,  ir <- commitments,  ir + ml + commitments <- verification;  nothing depends back on a later layer,
    and no core module imports a backend, frontend, integration, or external runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "verity"

#: For each core module, the sibling core modules it may import.  ``errors`` is shared by all.
ALLOWED = {
    "ir": set(),
    "ml": {"ir"},
    "commitments": {"ir"},
    "verification": {"ir", "ml", "commitments"},
    "resource_accounting": {"ir", "verification", "commitments"},
}
FORBIDDEN_EXTERNAL = ("verity_sp1", "verity_vllm", "verity_torch", "sp1", "vllm", "torch", "jax", "onnx", "numpy",
                      "backends", "frontends", "integrations", "research", "veritor")


def _imports(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((a.name, node.lineno) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module, node.lineno))
    return out


def _core_modules() -> list[str]:
    return sorted(p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists())


@pytest.mark.parametrize("module", _core_modules())
def test_core_module_imports_only_inward(module: str) -> None:
    assert module in ALLOWED, f"new core module {module!r}: add it to ALLOWED with its permitted dependencies"
    offenders = []
    for path in (SRC / module).rglob("*.py"):
        for name, line in _imports(path):
            top = name.split(".")[0]
            if top in FORBIDDEN_EXTERNAL:
                offenders.append(f"{path.relative_to(SRC)}:{line}: {name}")
            if top == "verity":
                parts = name.split(".")
                target = parts[1] if len(parts) > 1 else None
                if target in (None, module, "errors"):
                    continue
                if target not in ALLOWED[module]:
                    offenders.append(f"{path.relative_to(SRC)}:{line}: {name}")
    assert not offenders, "\n".join(offenders)
