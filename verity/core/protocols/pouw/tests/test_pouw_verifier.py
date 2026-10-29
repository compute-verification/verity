"""PoUW's verifier of record (`verifier`): one assignment that names the side of record, in a module that imports only the
standard library, so undoing the switch is one line and what decides a window, or flips its coins, never reaches
`verity.ml`."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from verity.core.protocols.pouw import verifier as V


def test_the_switch_is_one_line_in_a_module_that_imports_only_the_standard_library() -> None:
    tree = ast.parse(Path(V.__file__).read_text(encoding="utf-8"))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported and imported - {"__future__"} <= set(sys.stdlib_module_names), imported
    assigned = [t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)]
    assert assigned.count("RECORD_VERDICT") == 1 and V.RECORD_VERDICT in ("lean", "python")
