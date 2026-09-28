"""The silicon models are the plain-integer specification the Rust guest, the Lean vectors and the probes are checked against,
so ``verity.ml.circuits.silicon`` stays stdlib-only: no numpy, and of ``verity`` only its errors and the linear-form builder
(``verity.ml.circuits.boolean.forms``) the formats' Boolean gadgets are written in.  And the catalog keeps the rule it
had as core: no module of it but a circuit-check sidecar imports the checker."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[2]
SILICON = CATALOG / "silicon"
FROM_VERITY = ("verity.errors", "verity.ml.circuits.boolean.forms")
#: the checker, which the catalog's modules import no more than core's do (`tests/test_boundaries.py`); its
#: ``*.circuit-check.py`` sidecars, circuit-check's bindings of its Definitions, may (`circuit_check.sidecars`)
CHECKER = "circuit_check"


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
                 and not name.startswith("verity.ml.circuits.silicon")
                 and name not in FROM_VERITY]
    assert not offenders, "\n".join(offenders)


def test_only_a_sidecar_imports_the_checker() -> None:
    offenders = [f"{path.relative_to(CATALOG)}:{line}: {name}"
                 for path in sorted(CATALOG.rglob("*.py"))
                 if not path.name.endswith(".circuit-check.py") and "__pycache__" not in path.parts
                 for name, line in _imports(path)
                 if name.split(".")[0] == CHECKER]
    assert not offenders, "\n".join(offenders)
