"""`tools/verity/lean/survey.py` on small trees written by hand (no Lean needed): `map` on three Lake packages (modules,
declarations, areas, the cycle between two areas and the edge that breaks it, and the tree a move map would leave),
`conventions` on a Security-shaped package (one case of each measure), and `size` on a package of a few files with facts
written by hand."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_survey", HERE.parent / "survey.py")
S = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


# ---- map

MAP = {
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


def test_modules_belong_to_the_deepest_package_and_count_declarations_outside_comments(tmp_path):
    pkgs, mods, info, loose, dup = S.load(write(tmp_path, MAP))
    assert {p.name for p in pkgs.values()} == {"security", "security_proofs", "code"}
    assert mods["Proofs.A"].pkg.dir == "Security/Proofs" and mods["Definitions.A"].pkg.dir == "Security"
    assert mods["Definitions.A"].kinds == {"def": 1, "thm": 1}
    assert mods["Definitions.C.Z"].kinds == {"thm": 1}
    assert mods["Definitions"].umbrella and not mods["Definitions.A"].umbrella
    assert info["Proofs.A"][:2] == ("Proofs", "A") and info["Code.Base"] == ("Code", "code", "code")
    assert not loose and not dup


def test_the_area_cycle_and_the_edge_that_breaks_it(tmp_path):
    _, mods, info, *_ = S.load(write(tmp_path, MAP))
    weight, _ = S.graph(mods, info, 1)
    assert S.sccs(["A", "B", "C"], {a: {b for x, b in weight if x == a} for a in "ABC"}) == [["B", "C"]]
    assert ("(root)", "B") not in weight, "the umbrella only gathers modules, so its imports are no edges"
    assert len(S.breakers(["A", "B", "C"], weight)) == 1


def test_a_move_map_maps_the_tree_it_would_leave(tmp_path):
    (tmp_path / "moves.toml").write_text('[words]\n"Definitions.C" = "Definitions.B.C"\n"not a module" = "x"\n')
    moves = S.read_moves(tmp_path / "moves.toml")
    assert moves == {"Definitions.C": "Definitions.B.C"}
    _, mods, info, *_ = S.load(write(tmp_path, MAP), moves)
    assert "Definitions.B.C.Z" in mods and "Definitions.B.C.Z" in mods["Definitions"].imports
    weight, _ = S.graph(mods, info, 1)
    assert not S.sccs(["A", "B"], {a: {b for x, b in weight if x == a} for a in "AB"})
    _, mods, *_ = S.load(write(tmp_path, MAP), {"Definitions.A": "Definitions.Core.A", "A": "Core.A"})
    assert "Proofs.Core.A" in mods and "Definitions.Core.A" in mods["Proofs.Core.A"].imports, "a word renames inside a name"


def test_every_map_section_prints(tmp_path, capsys):
    assert S.main(["map", "--root", str(write(tmp_path, MAP)), "--json", str(tmp_path / "map.json")]) == 0
    out = capsys.readouterr().out
    for head in ("(a) Lake packages", "(b) areas x trees", "(c) the import graph", "(d) module granularity",
                 "(e) code and Security/"):
        assert head in out
    assert "security_proofs (Security/Proofs): requires 1 code packages, 1 of them directly" in out


# ---- conventions

SOURCES = {
    "Definitions/Pous/Guarantees/Sound.lean": """/-! The guarantee. -/
namespace Pous.Guarantees

/-- What the grader's acceptance means. -/
def Sound : Prop := True

end Pous.Guarantees
""",
    "Definitions/Pous/Assumptions.lean": """/-! The assumption. -/
namespace Pous.Assumptions

/-- pous/honest-store: the store answers. -/
def Honest : Prop := True

end Pous.Assumptions
""",
    "Definitions/Pous/Vocab.lean": """/-! Vocabulary. -/
namespace Pous

def isOk : Prop := True

/-- Rows a warden saw. -/
def wardenRows : Nat := 0

theorem one_le_two : 1 ≤ 2 := by decide

end Pous
""",
    "Proofs/Pous/Sound.lean": """namespace Pous.SecurityProofs

set_option maxHeartbeats 400000 in
theorem Sound : Pous.Guarantees.Sound := trivial

set_option maxHeartbeats 800000 in -- the `simp` closes 40 goals
theorem other : True := trivial

end Pous.SecurityProofs

namespace FlockX

theorem aux : True := trivial

end FlockX
""",
}
LOCK = {"guarantees": {"Pous.SecurityProofs.Sound": {}, "Pous.SecurityProofs.other": {}},
        "assumptions": {"Pous.Assumptions.Honest": {"owner": "@pous"}},
        "shape": {"frozen": {"Pous.SecurityProofs.other": "lemma: a step, not a property"}}}


def security(root: Path) -> Path:
    pkg = write(root / "Security", SOURCES)
    (pkg / "lean-audit.json").write_text(json.dumps(LOCK))
    return pkg


def test_conventions_counts_each_measure(tmp_path):
    security(tmp_path)
    assert S.locked(S.read(tmp_path)[0]) == ["Security"]
    r = S.survey(tmp_path.resolve(), ["Security"], 5)

    assert (r["packages"], r["files"]) == (1, 4)
    assert r["namespace"]["by_tree"]["proofs"] == {"area": 2, "foreign": 1}
    assert r["namespace"]["foreign_roots"] == [("FlockX", 1)]
    assert r["casing"]["counts"]["def (prop): lowerCamel"] == 1
    assert r["casing"]["examples"]["def (prop): lowerCamel"][0].startswith("Pous.isOk ")
    assert r["casing"]["counts"]["theorem: UpperCamel, SecurityProofs.G"] == 1
    assert r["theorem_forms"]["counts"]["uninformative (aux, key, step3)"] == 1
    assert r["retired"]["identifiers"] == {"warden": 1}
    assert r["retired"]["prose"]["warden"] == 1

    hb = r["heartbeats"]
    assert (hb["overrides"], hb["with_reason"], hb["max_value"]) == (2, 1, 800000)
    assert hb["worst_files_no_reason"] == [("Security/Proofs/Pous/Sound.lean", 1)]

    assert r["areas"]["by_tree"] == {"definitions": ["Pous"], "proofs": ["Pous"]}
    dt = r["definitions_tree"]
    assert (dt["theorems"], dt["spec_definitions"]) == (1, 4)
    assert r["docstrings"]["definitions_tree"] == {"decls": 4, "no_docstring": 1}
    assert r["docstrings"]["no_module_doc"] == {"definitions": 0, "proofs": 1}

    lock = r["locks"]["Security"]
    assert lock["forms"] == {"four names": 1, "X.SecurityProofs.G, no X.Guarantees.G": 1}
    assert lock["frozen_by_category"] == {"lemma": 1}
    assert lock["assumption_modules"] == {"Assumptions module": 1}
    assert lock["assumption_docstrings"] == {"docstring": 1}


def test_conventions_prints_and_writes_json(tmp_path, capsys):
    pkg = security(tmp_path)
    out = tmp_path / "survey.json"
    assert S.main(["conventions", "--root", str(tmp_path), "--json", str(out), str(pkg)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("1 packages, 4 files")
    assert "2 overrides in 1 files, 1 with a reason (50.0%)" in printed
    assert json.loads(out.read_text())["locks"]["Security"]["guarantees"] == 2


# ---- size

FILES = {
    "Definitions/A/Basic.lean": "/-! A docstring\nover two lines. -/\n\ndef read (n : Nat) : Nat := n -- a comment\n\n"
                                "structure S where\n  x : Nat\n\ntheorem t : read 1 = 1 := rfl\n",
    "Definitions/A/Old.lean": "-- nothing reads this\ndef old : Nat := 0\n",
    "Definitions/B.lean": "/- a /- nested -/ comment -/\nabbrev b : Nat := 1\n",
    "Proofs/A.lean": "theorem p : True := trivial\n",
}
FACTS = {"pins": {}, "spec_definitions": [
    {"module": "Definitions.A.Basic", "name": "A.read", "kind": "def"},
    {"module": "Definitions.A.Basic", "name": "A.S", "kind": "inductive"},
    {"module": "Definitions.A.Basic", "name": "A.read.eq_1", "kind": "def"},
    {"module": "Definitions.A.Old", "name": "A.old", "kind": "def"},
    {"module": "Definitions.B", "name": "B.b", "kind": "def"},
]}
SIZE_LOCK = {"roots": ["Definitions"], "guarantees": {"A.g": {"owner": "@lane"}}, "reads": {
    "Definitions.A.Basic": {"definitions": {"A.read": "h1", "A.S.x": "h2", "A.S": "h3"}, "guarantees": ["A.g"]},
    "Definitions.B": {"definitions": {"B.b": "h4"}, "guarantees": ["A.g"]},
}}


def package(tmp_path: Path) -> Path:
    write(tmp_path, FILES)
    (tmp_path / "lakefile.toml").write_text('name = "p"\n')
    (tmp_path / "lean-audit.json").write_text(json.dumps(SIZE_LOCK))
    return tmp_path


def test_code_lines_skip_comments_and_docstrings():
    assert S.code_lines(S.C.strip_comments(FILES["Definitions/A/Basic.lean"])) == 4
    assert S.code_lines(S.C.strip_comments(FILES["Definitions/B.lean"])) == 1


def test_areas_and_unread_from_facts(tmp_path):
    m = S.measure(package(tmp_path), "Definitions", FACTS)
    a, b = m["areas"]["A"], m["areas"]["B"]
    assert m["exact"] and set(m["areas"]) == {"A", "B"}
    assert (a["files"], a["definitions"], a["read"], a["unread"], a["theorems"], a["guarantees"]) == (2, 3, 2, 1, 1, 1)
    assert (b["definitions"], b["read"], b["unread"]) == (1, 1, 0)
    assert m["unread"] == {"Definitions.A.Old": ["A.old"]}
    assert "| total | 3 |" in S.size_table(m)


def test_without_facts_counts_the_sources(tmp_path, capsys):
    m = S.measure(package(tmp_path), "Definitions")
    a = m["areas"]["A"]
    assert not m["exact"] and (a["definitions"], a["read"], a["unread"]) == (3, 3, None)
    assert "~3" in S.size_table(m)
    assert S.main(["size", str(tmp_path)]) == 0 and "| A | 2 |" in capsys.readouterr().out
