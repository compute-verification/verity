"""PoUW's verifier of record (`verifier`): one assignment that names the side of record, in a module that imports nothing,
so undoing the switch is one line and what decides a window never reaches `verity.ml`."""
from __future__ import annotations

import ast
from pathlib import Path

from verity.core.protocols.pouw import verifier as V


def test_the_switch_is_one_line_that_imports_nothing() -> None:
    tree = ast.parse(Path(V.__file__).read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assigned = [t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets]
    assert assigned == ["RECORD_VERDICT"] and V.RECORD_VERDICT in ("lean", "python")
