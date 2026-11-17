"""Which modules of ``verity`` are core (``_imports.SIDES``): each row names a package or a module of the tree.

A row that names nothing rules nothing: what moved from under it is on the side of the row above it.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path

from . import _imports as I
from . import _ratchet as R


def naming_nothing(rows: Iterable[str], repo: Path) -> list[str]:
    """The rows that name neither a package nor a module under ``repo``."""
    out = []
    for row in rows:
        base = repo.joinpath(*row.split("."))
        if not base.is_dir() and not base.with_suffix(".py").is_file():
            out.append(row)
    return out


def test_a_row_that_names_nothing_is_found(tmp_path: Path) -> None:
    (tmp_path / "verity" / "a").mkdir(parents=True)
    (tmp_path / "verity" / "a" / "b.py").touch()
    rows = ["verity", "verity.a", "verity.a.b", "verity.a.c", "verity.b"]
    assert naming_nothing(rows, tmp_path) == ["verity.a.c", "verity.b"]


def test_each_row_names_a_module_of_the_tree_or_warns() -> None:
    """It warns and fails nothing: a move in ``verity/`` is another change's, and the row is renamed after it."""
    stale = naming_nothing(I.SIDES, R.REPO)
    if stale:
        warnings.warn("rows of SIDES (tests/lint/_imports.py) that name no package or module of the tree: rename each to "
                      f"where its modules are now, or delete it: {stale}", stacklevel=1)
