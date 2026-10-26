"""`tools/verity/lean/map.py`: a toy tree of three Lake packages mapped from its sources: modules, declarations, areas,
the cycle between two areas and the edge that breaks it, and the same tree as a move map would leave it."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_map", HERE.parent / "map.py")
M = sys.modules["lean_map"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

FILES = {
    "Security/lakefile.toml": 'name = "security"\n[[lean_lib]]\nname = "Definitions"\n',
    "Security/Proofs/lakefile.toml": ('name = "security_proofs"\nsrcDir = ".."\n[[require]]\nname = "security"\npath = ".."\n'
                                      '[[require]]\nname = "code"\npath = "../../code"\n'
                                      '[[lean_lib]]\nname = "Proofs"\nroots = ["Proofs.A"]\n'),
    "code/lakefile.toml": 'name = "code"\n[[lean_lib]]\nname = "Code"\n',
    "Security/Definitions/A.lean": "import Mathlib\n/-! A docstring: theorem in prose\n  /- nested -/ def too -/\ndef a := 1\n"
                                   "-- theorem in a comment\ntheorem t : a = 1 := rfl\n",
    "Security/Definitions/B/X.lean": "import Definitions.C.Y\ndef x := 2\n",
    "Security/Definitions/B/W.lean": "import Definitions.A\ndef w := 3\n",
    "Security/Definitions/C/Y.lean": "def y := 4\n",
    "Security/Definitions/C/Z.lean": "import Definitions.B.W\n@[simp] private theorem z : True := trivial\n",
    "Security/Definitions.lean": "import Definitions.B.X\nimport Definitions.C.Z\n",
    "Security/Proofs/A.lean": "import Definitions.A\nimport Code.Base\ntheorem p : True := trivial\n",
    "code/Code/Base.lean": "def base := 0\n",
}


def tree(tmp_path: Path) -> Path:
    for rel, text in FILES.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    return tmp_path


def test_modules_belong_to_the_deepest_package_and_count_declarations_outside_comments(tmp_path):
    pkgs, mods, info, loose, dup = M.load(tree(tmp_path))
    assert {p.name for p in pkgs.values()} == {"security", "security_proofs", "code"}
    assert mods["Proofs.A"].package == "Security/Proofs" and mods["Definitions.A"].package == "Security"
    assert mods["Definitions.A"].decls == {"def": 1, "thm": 1}
    assert mods["Definitions.C.Z"].decls == {"thm": 1}
    assert mods["Definitions"].umbrella and not mods["Definitions.A"].umbrella
    assert info["Proofs.A"][:2] == ("Proofs", "A") and info["Code.Base"] == ("Code", "code", "code")
    assert not loose and not dup


def test_the_area_cycle_and_the_edge_that_breaks_it(tmp_path):
    _, mods, info, *_ = M.load(tree(tmp_path))
    weight, _ = M.graph(mods, info, 1)
    assert M.sccs(["A", "B", "C"], {a: {b for x, b in weight if x == a} for a in "ABC"}) == [["B", "C"]]
    assert ("(root)", "B") not in weight, "the umbrella only gathers modules, so its imports are no edges"
    assert len(M.breakers(["A", "B", "C"], weight)) == 1


def test_a_move_map_maps_the_tree_it_would_leave(tmp_path):
    (tmp_path / "moves.toml").write_text('[words]\n"Definitions.C" = "Definitions.B.C"\n"not a module" = "x"\n')
    moves = M.read_moves(tmp_path / "moves.toml")
    assert moves == {"Definitions.C": "Definitions.B.C"}
    _, mods, info, *_ = M.load(tree(tmp_path), moves)
    assert "Definitions.B.C.Z" in mods and "Definitions.B.C.Z" in mods["Definitions"].imports
    weight, _ = M.graph(mods, info, 1)
    assert not M.sccs(["A", "B"], {a: {b for x, b in weight if x == a} for a in "AB"})


def test_every_section_prints(tmp_path, capsys):
    assert M.main(["--root", str(tree(tmp_path)), "--json", str(tmp_path / "map.json")]) == 0
    out = capsys.readouterr().out
    for head in ("(a) Lake packages", "(b) areas x trees", "(c) the import graph", "(d) module granularity",
                 "(e) code and Security/"):
        assert head in out
    assert "security_proofs (Security/Proofs): requires 1 code packages, 1 of them directly" in out
