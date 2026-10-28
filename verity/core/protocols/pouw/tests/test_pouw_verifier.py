"""PoUW's verifier of record (`verifier`): a module that imports only the standard library, so what decides a window, or
draws its coins, never reaches `verity.ml`, and that names no side of record: the Lean's is the only one."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from verity.core.protocols.pouw import verifier as V


def test_the_verifier_imports_only_the_standard_library_and_has_no_switch() -> None:
    tree = ast.parse(Path(V.__file__).read_text(encoding="utf-8"))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported and imported - {"__future__"} <= set(sys.stdlib_module_names), imported
    assert not hasattr(V, "RECORD_VERDICT")
