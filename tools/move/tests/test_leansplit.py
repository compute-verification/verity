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


def test_a_header_after_a_plain_comment_holds_the_imports():
    text = "/-\nWhat the module proves.\n-/\nmodule\n\npublic import Proofs.Toy.Base\n\nnamespace Toy\nend Toy\n"
    header, body = M.split_header(text)
    assert M.imports_of(header) == ["Proofs.Toy.Base"] and body.startswith("\nnamespace Toy")
    added = M.add_import(text, "Proofs.Toy.Src", None)
    assert "public import Proofs.Toy.Base\npublic import Proofs.Toy.Src\n" in added and added.startswith("/-\n")


OTHER = """module

public import Proofs.Toy.Base

namespace Toy.Inner

def c : Nat := 0

end Toy.Inner
"""

SRC2 = """module

public import Proofs.Toy.Base
public import Proofs.Toy.Other

namespace Toy
open Nat Inner
local notation "𝔹" => Toy.base
local notation "ℂ" => Toy.Inner.c

theorem a_le (n : Nat) : n ≤ 𝔹 n := Nat.le_succ n

theorem c_eq : ℂ = 0 := rfl

end Toy
"""


def test_a_new_module_leaves_out_the_namespaces_and_notations_it_cannot_see(tree):
    d = tree.root / "Security" / "Proofs" / "Toy"
    (d / "Other.lean").write_text(OTHER)
    (d / "Src.lean").write_text(SRC2)
    tree = M.Tree(tree.root)
    op = dict(SPLIT, importers=[], drop_source=[], uses={})
    log: list[str] = []
    M.split(tree, op, log)
    new = tree.read("Proofs.Toy.SrcA")
    assert "namespace Toy\nopen Nat\nlocal notation \"𝔹\" => Toy.base\n" in new and "ℂ" not in new
    assert "open Nat Inner\n" in tree.read("Proofs.Toy.Src")
    assert any("left out open Inner" in ln for ln in log)


SRC3 = """module

public import Proofs.Toy.Base
public import Proofs.Toy.Other

namespace Toy
variable (F : Type) (Hs : Inner.Hashes F) [Inhabited F] (x : F) {k : Nat}
variable (y : F)
include y

theorem a_le (n : Nat) : x = x ∧ n ≤ base n := ⟨rfl, Nat.le_succ n⟩

theorem b_le (n : Nat) (h : Hs = Hs) : n ≤ base n + k := by omega

end Toy
"""


def test_a_new_module_keeps_only_the_variables_its_declarations_name(tree):
    (tree.root / "Security/Proofs/Toy/Other.lean").write_text(OTHER)
    (tree.root / "Security/Proofs/Toy/Src.lean").write_text(SRC3)
    tree = M.Tree(tree.root)
    M.split(tree, dict(SPLIT, importers=[], drop_source=[], uses={}), [])
    new = tree.read("Proofs.Toy.SrcA")
    assert "variable (F : Type) [Inhabited F] (x : F)\nvariable (y : F)\ninclude y\n" in new
    assert "Hs" not in new and "{k" not in new
    assert "(Hs : Inner.Hashes F)" in tree.read("Proofs.Toy.Src")


TOP = """module

public import Proofs.Toy.User

namespace Toy

theorem top (n : Nat) : n ≤ base n + 1 := b_le n

end Toy
"""


def test_a_module_whose_imports_no_longer_reach_what_it_reads_imports_it(tree):
    d = tree.root / "Security/Proofs/Toy"
    (d / "Top.lean").write_text(TOP)
    (d / "Out.lean").write_text(TOP.replace("theorem top", "theorem out"))
    tree = M.Tree(tree.root)
    before = {m: tree.imports(m) for m in tree.mods}
    M.split(tree, SPLIT, [])
    log: list[str] = []
    M.repair(tree, before, [SPLIT], {"Proofs.Toy.Top": ["Proofs.Toy.Src"]}, "Proofs.Toy.Top", log)
    imports = lambda m: M.imports_of(M.split_header(tree.read(m))[0])
    assert imports("Proofs.Toy.Top") == ["Proofs.Toy.User", "Proofs.Toy.Src"]
    assert imports("Proofs.Toy.Out") == ["Proofs.Toy.User", "Proofs.Toy.Src"]
    assert imports("Proofs.Toy.User") == ["Proofs.Toy.SrcA"]
    assert log[-1].startswith("repair: 2 imports")


def test_a_local_notation_is_no_atom_an_importer_reads():
    tree = M.Tree(Path("/nonexistent"))
    tree.mods["Proofs.Toy.Syn"] = Path("/nonexistent/Syn.lean")
    tree.write("Proofs.Toy.Syn", 'local notation "𝔽" => Nat\nmacro "toy_tac" : tactic => `(tactic| rfl)\n')
    assert tree.atoms("Proofs.Toy.Syn") == ["toy_tac"]


def test_every_file_reassembles_from_its_parts(tree):
    for m in tree.mods:
        header, body = M.split_header(tree.read(m))
        units, _ = M.annotate(M.units_of(body))
        assert header + "".join(u.full() for u in units) == tree.read(m)
