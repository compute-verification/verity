"""`leansplit.py`, the declaration split and merge, on a toy tree: a split takes a declaration with its docstring and the
scopes it stood in, repoints the importers that read only it, and imports what they reached through the source; a merge
closes the merged module's scopes before its destination's text; and a name the tool can't place stops it before anything
is written."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "leansplit.py"
spec = importlib.util.spec_from_file_location("leansplit_under_test", TOOL)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

BASE = """module

public import Mathlib.Tactic.Ring

namespace Toy

def base (n : Nat) : Nat := n + 1

end Toy
"""

SRC = """module

public import Proofs.Toy.Base

/-! The toy source. -/

@[expose] public section

namespace Toy
open Nat

/-- The one `User` reads. -/
theorem a_le (n : Nat) : n ≤ base n := Nat.le_succ n

theorem b_le (n : Nat) : n ≤ base n + 1 := by
  have := a_le n
  omega

end Toy
"""

USER = """module

public import Proofs.Toy.Src

namespace Toy

theorem user (n : Nat) : n ≤ base n := a_le n

end Toy
"""

SPLIT = {"kind": "split", "source": "Proofs.Toy.Src", "new": "Proofs.Toy.SrcA", "decls": ["Toy.a_le"],
         "imports": ["Proofs.Toy.Base"], "external_imports": [], "source_imports_new": True,
         "importers": ["Proofs.Toy.User"], "drop_source": ["Proofs.Toy.User"], "uses": {"Proofs.Toy.User": ["Proofs.Toy.Base"]}}


@pytest.fixture
def tree(tmp_path: Path):
    d = tmp_path / "Security" / "Proofs" / "Toy"
    d.mkdir(parents=True)
    for name, text in (("Base", BASE), ("Src", SRC), ("User", USER)):
        (d / f"{name}.lean").write_text(text)
    return M.Tree(tmp_path)


def test_a_split_takes_the_declaration_its_docstring_and_its_scopes(tree):
    log: list[str] = []
    M.split(tree, SPLIT, log)
    new = tree.read("Proofs.Toy.SrcA")
    assert new.startswith("module\n\npublic import Proofs.Toy.Base\n")
    assert "@[expose] public section\n" in new and "namespace Toy\nopen Nat\n" in new
    assert "/-- The one `User` reads. -/\ntheorem a_le" in new and new.rstrip().endswith("end Toy")
    src = tree.read("Proofs.Toy.Src")
    assert "a_le (n" not in src and "theorem b_le" in src
    assert "public import Proofs.Toy.SrcA\n" in M.split_header(src)[0]


def test_an_importer_that_reads_only_the_split_part_stops_importing_the_source(tree):
    M.split(tree, SPLIT, [])
    header = M.split_header(tree.read("Proofs.Toy.User"))[0]
    assert M.imports_of(header) == ["Proofs.Toy.SrcA"]


def test_an_importer_imports_what_it_reached_only_through_the_source(tree):
    op = dict(SPLIT, imports=[], uses={"Proofs.Toy.User": ["Proofs.Toy.Base"]})
    M.split(tree, op, [])
    assert M.imports_of(M.split_header(tree.read("Proofs.Toy.User"))[0]) == ["Proofs.Toy.SrcA", "Proofs.Toy.Base"]


def test_a_merge_closes_its_scopes_before_the_destination_text(tree):
    M.merge(tree, {"kind": "merge", "module": "Proofs.Toy.Src", "into": "Proofs.Toy.User"}, [])
    user = tree.read("Proofs.Toy.User")
    assert M.imports_of(M.split_header(user)[0]) == ["Proofs.Toy.Base"]
    body = M.split_header(user)[1]
    closed = body.index("\nend\n", body.index("end Toy"))
    assert body.index("theorem b_le") < closed < body.index("theorem user")
    assert tree.files[tree.mods["Proofs.Toy.Src"]] is None


def test_a_name_it_cannot_find_stops_it_before_anything_is_written(tmp_path, tree):
    root = tree.root
    m = root / "map.toml"
    m.write_text('[[op]]\nkind = "split"\nsource = "Proofs.Toy.Src"\nnew = "Proofs.Toy.SrcC"\ndecls = ["Toy.c_le"]\n')
    r = subprocess.run([sys.executable, str(TOOL), str(root), "--map", str(m)], capture_output=True, text=True)
    assert r.returncode != 0 and "not found: ['Toy.c_le']" in r.stderr
    assert not (root / "Security" / "Proofs" / "Toy" / "SrcC.lean").exists()


MAP = """[[op]]
kind = "split"
source = "Proofs.Toy.Src"
new = "Proofs.Toy.SrcA"
decls = ["Toy.a_le"]
imports = ["Proofs.Toy.Base"]
source_imports_new = true
importers = ["Proofs.Toy.User"]
drop_source = ["Proofs.Toy.User"]

[[op]]
kind = "merge"
module = "Proofs.Toy.Base"
into = "Proofs.Toy.SrcA"
"""


def test_a_second_run_changes_nothing(tree):
    root = tree.root
    (root / "map.toml").write_text(MAP)
    run = lambda: subprocess.run([sys.executable, str(TOOL), str(root), "--map", str(root / "map.toml")],
                                 capture_output=True, text=True, check=True)
    run()
    after = {p: p.read_text() for p in root.rglob("*.lean")}
    toy = root / "Security/Proofs/Toy"
    assert sorted(p.name for p in after) == ["Src.lean", "SrcA.lean", "User.lean"]
    assert "def base" in after[toy / "SrcA.lean"] and "theorem a_le" in after[toy / "SrcA.lean"]
    assert M.imports_of(M.split_header(after[toy / "Src.lean"])[0]) == ["Proofs.Toy.SrcA"]
    run()
    assert {p: p.read_text() for p in root.rglob("*.lean")} == after


def test_every_file_reassembles_from_its_parts(tree):
    for m in tree.mods:
        header, body = M.split_header(tree.read(m))
        units, _ = M.annotate(M.units_of(body))
        assert header + "".join(u.full() for u in units) == tree.read(m)
