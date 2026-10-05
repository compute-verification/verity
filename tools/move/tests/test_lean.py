"""`lean.py`, the Lean layout move: its renaming rules, the guard against paths a case-insensitive checkout can't hold, and
that the move has nothing left to do on this tree (a rerun renames no module)."""
from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("lean_move_under_test", HERE.parent / "lean.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

TOY = {"security": {"dir": "sec", "proofs": "sec/Proofs"},
       "packages": {"a": {"dir": "a/lean", "into": "security"}, "b": {"dir": "b/lean", "into": "security"}},
       "modules": {"A": "", "A.Protocol": "Definitions.A", "A.SecurityProofs": "Proofs.A", "b:Check": "Proofs.B.Check"}}


def test_longest_prefix_and_package_rules():
    m = M.Map(TOY, ROOT)
    assert m.rename("A.Protocol.X.Y", "a") == "Definitions.A.X.Y"
    assert m.rename("A.SecurityProofs", "a") == "Proofs.A"
    assert m.rename("A", "a") == ""
    assert m.rename("A.Other", "a") is None
    assert m.rename("Check", "b") == "Proofs.B.Check" and m.rename("Check", "a") is None
    assert m.new_path("Definitions.A.X", "a") == "sec/Definitions/A/X.lean"


def test_case_clashes():
    assert M.case_clashes(["d/Assumptions/X.lean", "d/assumptions/n.md", "d/Assumptions.lean"]) == [
        ["d/Assumptions", "d/assumptions"]]
    assert M.case_clashes(["d/Assumptions/X.lean", "d/ASSUMPTIONS.md", "d/assumption-notes/n.md"]) == []


def test_the_tree_is_moved():
    m = M.Map(tomllib.loads((HERE.parent / "lean_map.toml").read_text()), ROOT)
    files = M.tracked(ROOT)
    _, dropped, by_path = M.rename_all(m, M.modules(m, files))
    assert not by_path and not dropped, f"lean.py would still move: {sorted(by_path.items())[:5]}"
    assert not M.case_clashes(files)
