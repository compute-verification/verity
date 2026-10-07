"""The silicon models are the plain-integer specification the Rust guest, the Lean vectors and the probes are checked against,
so ``verity_catalog.silicon`` stays stdlib-only: no numpy, and of ``verity`` only its errors and the linear-form builder
(``verity.primitives.circuits.boolean.forms``) the formats' Boolean gadgets are written in."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SILICON = Path(__file__).resolve().parents[1] / "verity_catalog" / "silicon"
FROM_VERITY = ("verity.errors", "verity.primitives.circuits.boolean.forms")


def _imports(path: Path) -> list[tuple[str, int]]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            out += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            out.append((node.module, node.lineno))
    return out


def test_silicon_is_stdlib_only() -> None:
    offenders = [f"{path.relative_to(SILICON)}:{line}: {name}"
                 for path in sorted(SILICON.rglob("*.py"))
                 for name, line in _imports(path)
                 if name.split(".")[0] not in sys.stdlib_module_names
                 and not (name.split(".")[0] == "verity_catalog" and name.startswith("verity_catalog.silicon"))
                 and name not in FROM_VERITY]
    assert not offenders, "\n".join(offenders)
