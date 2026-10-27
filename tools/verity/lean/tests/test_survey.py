"""`tools/verity/lean/survey.py` on small trees written by hand (no Lean needed): `map` on three Lake packages (modules,
declarations, areas, the cycle between two areas and the edge that breaks it, and the tree a move map would leave),
`conventions` on a Security-shaped package (one case of each measure), `size` on a package of a few files with facts
written by hand, and `unused`'s keep rules and deletion on a stand-in package and usage document; and, when `lake` is
installed, `Facts.lean`'s usage mode on the controls."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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
        "assumptions": {"Pous.Assumptions.Honest": {}},
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
FACTS = {"pins": {"A.g": {"kind": "theorem", "reads": [{"name": "A.read", "module": "Definitions.A.Basic"},
                                                      {"name": "B.b", "module": "Definitions.B"}]}}, "spec_definitions": [
    {"module": "Definitions.A.Basic", "name": "A.read", "kind": "def"},
    {"module": "Definitions.A.Basic", "name": "A.S", "kind": "inductive"},
    {"module": "Definitions.A.Basic", "name": "A.read.eq_1", "kind": "def"},
    {"module": "Definitions.A.Old", "name": "A.old", "kind": "def"},
    {"module": "Definitions.B", "name": "B.b", "kind": "def"},
]}
SIZE_LOCK = {"roots": ["Definitions"], "guarantees": {"A.g": {}}, "reads": {
    "Definitions.A.Basic": {"definitions": {"A.read": "h1", "A.S.x": "h2", "A.S": "h3"}},
    "Definitions.B": {"definitions": {"B.b": "h4"}},
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
    by = S.measure(package(tmp_path), "Definitions", FACTS, read_by={"Definitions.B": ["A.g", "B.h"]})
    assert (by["areas"]["A"]["guarantees"], by["areas"]["B"]["guarantees"]) == (0, 2), "a run's report.json `readers`"


def test_without_facts_counts_the_sources(tmp_path, capsys):
    m = S.measure(package(tmp_path), "Definitions")
    a = m["areas"]["A"]
    assert not m["exact"] and (a["definitions"], a["read"], a["unread"], a["guarantees"]) == (3, 3, None, None)
    assert "~3" in S.size_table(m) and S.size_table(m).splitlines()[-1].endswith("| - | - |"), "readers unknown without facts"
    assert S.main(["size", str(tmp_path)]) == 0 and "| A | 2 |" in capsys.readouterr().out


# ---- unused

PKG = "Security/Proofs"
#: the stand-in package's modules: {module: its source}
UNUSED_SOURCES = {
    "Proofs.Root": "import Proofs.A.Main\nimport Proofs.B.Group\n",
    "Proofs.A.Main": "import Proofs.A.Base\nimport Proofs.B.Notation\nimport Proofs.B.Attr\nimport Proofs.B.Refd\n\n"
                     "theorem main : ⟪base⟫ = 1 := rfl\ntheorem other : refd = 2 := rfl\n",
    "Proofs.A.Base": "def base : Nat := 1\n",
    "Proofs.A.Side": "import Proofs.A.Base\n\n-- names `Proofs.B.Named`, in a comment\ntheorem side : base = 1 := rfl\n",
    "Proofs.B.Group": "/-! An aggregator of what nothing uses. -/\nimport Proofs.B.Old\n",
    "Proofs.B.Old": "import Proofs.A.Side\n\ntheorem old : True := trivial\n",
    "Proofs.B.Notation": "notation \"⟪\" x \"⟫\" => x\nlocal notation \"unseen\" => 0\n",
    "Proofs.B.Attr": "theorem attr_lemma : 0 + 0 = 0 := rfl\nattribute [simp] attr_lemma\n",
    "Proofs.B.Named": "import Proofs.B.Dep\n\ntheorem named : True := trivial\n",
    "Proofs.B.Dep": "theorem dep : True := trivial\n",
    "Proofs.B.Refd": "def refd : Nat := 2\n",
}
#: {module: (constants, used, refs)}, as `Facts.lean`'s usage mode would report them
UNUSED_FACTS = {"Proofs.Root": (0, 0, []), "Proofs.A.Main": (2, 1, ["Proofs.A.Base", "Proofs.B.Refd"]),
                "Proofs.A.Base": (1, 1, []), "Proofs.A.Side": (1, 0, ["Proofs.A.Base"]), "Proofs.B.Group": (0, 0, []),
                "Proofs.B.Old": (1, 0, []), "Proofs.B.Notation": (2, 0, []), "Proofs.B.Attr": (1, 0, []),
                "Proofs.B.Named": (1, 0, []), "Proofs.B.Dep": (1, 0, []), "Proofs.B.Refd": (1, 0, [])}
TEST = "tests/test_named.py"
SPEC = "spec/Spec.lean"


def source(module: str) -> str:
    return f"Security/{module.replace('.', '/')}.lean"


@pytest.fixture
def stand_in(tmp_path):
    """(the root holding the stand-in package at `Security/Proofs` and a test naming `Proofs.B.Named`, its lock, the usage
    document)."""
    policy = {"roots": ["Proofs.Root"], "guarantees": {"Proofs.A.main": {}}}
    write(tmp_path, {**{source(m): text for m, text in UNUSED_SOURCES.items()},
                     f"{PKG}/lakefile.toml": 'name = "security_proofs"\nsrcDir = ".."\n',
                     f"{PKG}/{S.C.POLICY}": json.dumps(policy),
                     TEST: 'MODULE = "Proofs.B.Named"\nSOURCE = "import Proofs.B.Named\\n#check named"\n'})
    usage = {"graph": {}, "usage": {"pins": {"Proofs.A.main": "Proofs.A.Main"}, "modules": {
        m: {"constants": c, "used": u, "by": "Proofs.A.main" if u else None, "refs": r} for m, (c, u, r) in UNUSED_FACTS.items()}}}
    return tmp_path, policy, usage


def pruned(root: Path):
    pkgs, ms, _ = S.read(root)
    return pkgs[PKG], {m.name: m for m in ms if m.pkg.dir == PKG}


def decide(root: Path, policy: dict, usage: dict, files: list[str], frozen=(), queued=None, doubts=True) -> dict:
    return S.decide(root, *pruned(root), policy, usage, files, list(frozen), queued, doubts)


def test_decide_keeps_by_each_rule(stand_in):
    root, policy, usage = stand_in
    got = decide(root, policy, usage, [TEST], ["Proofs.B.Old"])
    why = {m: r.split(":", 1)[0] for m, r in got["kept"].items()}
    assert why == {"Proofs.Root": "root", "Proofs.A.Main": "used", "Proofs.A.Base": "used", "Proofs.B.Refd": "refs",
                   "Proofs.B.Notation": "syntax", "Proofs.B.Attr": "attribute", "Proofs.B.Named": "named",
                   "Proofs.B.Dep": "named"}
    assert got["unused"] == ["Proofs.A.Side", "Proofs.B.Group", "Proofs.B.Old"]
    assert got["frozen"] == ["Proofs.B.Old"]


def test_what_a_queued_pr_touches_or_imports_is_kept(stand_in):
    root, policy, usage = stand_in
    got = decide(root, policy, usage, [TEST], queued={"1": {"files": [source("Proofs.A.Side"), "README.md"],
                                                           "imports": ["Mathlib.Logic.Basic"]}})
    assert got["kept"]["Proofs.A.Side"] == "queued: #1 touch or import it"
    assert got["unused"] == ["Proofs.B.Group", "Proofs.B.Old"]
    got = decide(root, policy, usage, [TEST], queued={"2": {"files": [], "imports": ["Proofs.B.Group"]}})
    assert got["unused"] == [] and got["kept"]["Proofs.A.Side"].startswith("queued: imported by Proofs.B.Group")
    got = decide(root, policy, usage, [TEST], queued={"3": {"files": [], "imports": [], "names": ["Proofs.B.Old", "README.md"]}})
    assert got["kept"]["Proofs.B.Old"] == "queued: #3 touch or import it"
    assert got["kept"]["Proofs.B.Group"].startswith("aggregator: ") and got["unused"] == []
    mentioned = {"4": {"files": [], "imports": [], "names": ["Proofs.B.Old"], "reads": []}}
    got = decide(root, policy, usage, [TEST], queued=mentioned)
    assert got["kept"]["Proofs.B.Old"] == "queued: #4 touch or import it" and got["unused"] == ["Proofs.A.Side"]
    read = {"5": {"files": [], "imports": [], "names": ["Proofs.B.Old"], "reads": ["Proofs.B.Old"]}}
    assert decide(root, policy, usage, [TEST], queued=read)["unused"] == []
    assert S.queued_modules({**mentioned, **read}, {m: x.file for m, x in pruned(root)[1].items()}) == {
        "Proofs.B.Old": (["#4", "#5"], "reads")}
    got = decide(root, policy, usage, [TEST], queued={"6": {"files": [source("Proofs.B.Group")], "imports": []}})
    assert got["kept"]["Proofs.B.Group"].startswith("queued: ") and got["unused"] == ["Proofs.A.Side", "Proofs.B.Old"]
    doc = S.listing(root, pruned(root)[1], got, {"Proofs.B.Group": got["kept"]["Proofs.B.Group"]})
    assert [r["module"] for r in doc["held"]] == ["Proofs.B.Group"] and doc["total"]["held"] == 1
    assert "\nheld (1 modules" in S.report(doc)


def test_a_queued_diff_reads_only_what_a_reader_or_an_import_names():
    got = S.pr_entry([
        ("Security/Proofs/Proofs/A.lean", "", "+import Proofs.B\n-import Proofs.C\n"),
        ("tools/x.py", "tools/old.py", "+# as `Proofs.D`'s clash showed\n+SOURCE = \"import Proofs.E\\n\"\n"),
        ("verity/core/difftest/run.py", "", "+MODULE = \"Proofs.F\"\n Proofs.G\n"),
        ("scripts/build.sh", "", "+lake build Proofs.H\n"),
    ])
    assert got == {"files": ["Security/Proofs/Proofs/A.lean", "scripts/build.sh", "tools/old.py", "tools/x.py",
                             "verity/core/difftest/run.py"],
                   "imports": ["Proofs.B"], "names": ["Proofs.D", "Proofs.E", "Proofs.F", "Proofs.H"],
                   "reads": ["Proofs.E", "Proofs.F", "Proofs.H"]}


def test_a_conformance_check_is_kept_without_its_imports(stand_in):
    root, policy, usage = stand_in
    old = root / source("Proofs.B.Old")
    old.write_text("import Proofs.A.Side\n\nexample : 1 + 1 = 2 := rfl\ntheorem old : True := trivial\n")
    got = decide(root, policy, usage, [TEST])
    assert got["kept"]["Proofs.B.Old"].startswith("checks: ") and got["kept"]["Proofs.B.Group"].startswith("aggregator: ")
    assert got["unused"] == ["Proofs.A.Side"]
    assert decide(root, policy, usage, [TEST], doubts=False)["unused"] == ["Proofs.A.Side", "Proofs.B.Group", "Proofs.B.Old"]
    names = ["Proofs.X.ChainVectors", "Proofs.X.PeelChecks.Gram0", "Proofs.X.CheckSum", "Proofs.Vectorsx"]
    assert [bool(S.CHECK_NAME.search(m)) for m in names] == [True, True, False, False]
    old.write_text("import Proofs.A.Side\n\n#guard 1 + 1 == 2\ntheorem old : True := trivial\n")
    assert decide(root, policy, usage, [TEST])["kept"]["Proofs.B.Old"].startswith("checks: a conformance check")
    for title, body, held in (("The joint guarantee's hypotheses hold", "", True),
                              ("The influence theorems are not vacuous", "", True),
                              ("The chain's satisfiability witness", "", True),
                              ("PoUW's per-run guarantee needs the cost model to truncate", "", True),
                              ("A program that beats the honest count", "", True),
                              ("Cost models that truncate", "`fixed` is one, so the hypothesis holds of one.", True),
                              ("`htau` from the checks", "`htau` is vacuous, and the rest needs `x`.", False)):
        old.write_text(f"import Proofs.A.Side\n\n/-!\n# {title}\n\n{body}\n-/\ntheorem old : True := trivial\n")
        why = decide(root, policy, usage, [TEST])["kept"].get("Proofs.B.Old", "")
        assert why.startswith("checks: a non-vacuity witness or a necessity result") == held, title


def test_a_module_a_lean_comment_outside_cites_is_kept(stand_in):
    root, policy, usage = stand_in
    write(root, {SPEC: "/-! The bound holds (`Proofs.A.Side.side`). -/\ndef bound : Nat := 1\n"})
    got = decide(root, policy, usage, [TEST, SPEC])
    assert got["kept"]["Proofs.A.Side"] == f"cited: a Lean comment of {SPEC} cites it"
    assert got["unused"] == ["Proofs.B.Group", "Proofs.B.Old"] and got["docs"] == {}
    assert decide(root, policy, usage, [TEST, SPEC], doubts=False)["unused"] == [
        "Proofs.A.Side", "Proofs.B.Group", "Proofs.B.Old"]
    write(root, {SPEC: "/-! `Proofs.B.Group`: what the bound rests on. -/\ndef bound : Nat := 1\n"})
    got = decide(root, policy, usage, [TEST, SPEC])
    assert got["kept"]["Proofs.B.Old"] == f"cited: imported by Proofs.B.Group, which {SPEC} cites as what it gathers"
    assert got["unused"] == []
    write(root, {SPEC: "/-! `Proofs.B.Group.bound` and `Proofs.Root` are cited, not what they gather. -/\ndef bound : Nat := 1\n"})
    got = decide(root, policy, usage, [TEST, SPEC])
    assert got["kept"]["Proofs.B.Group"].startswith("cited: ") and got["unused"] == ["Proofs.A.Side", "Proofs.B.Old"]


def test_a_constant_named_outside_keeps_its_module(stand_in):
    root, policy, usage = stand_in
    write(root, {source("Proofs.A.Side"): "import Proofs.A.Base\n\nnamespace Pool.Lemmas\n\n"
                                          "theorem side : base = 1 := rfl\n\nend Pool.Lemmas\n"})
    for text, cited in (("/-! The bound rests on `Pool.Lemmas.side`. -/\ndef bound : Nat := 1\n", True),
                        ("namespace Pool\n\n/-- By `Lemmas.side`. -/\ndef bound : Nat := 1\n\nend Pool\n", True),
                        ("open Pool.Lemmas in\n/-- By `side`. -/\ndef bound : Nat := 1\n", True),
                        ("namespace Pool.Lemmas\n\n/-- Each side of it, as words. -/\ndef bound : Nat := 1\n\n"
                         "end Pool.Lemmas\n", False),
                        ("/-! `Lemmas.side` and `side`, outside its namespace. -/\ndef bound : Nat := 1\n", False)):
        write(root, {SPEC: text})
        got = decide(root, policy, usage, [TEST, SPEC])
        assert (got["kept"].get("Proofs.A.Side") == f"cited: a Lean comment of {SPEC} cites it") == cited, text
    write(root, {"tests/test_doc.py": '"""The bound `Pool.Lemmas.side` gives."""\n'})
    assert decide(root, policy, usage, [TEST, "tests/test_doc.py"])["kept"]["Proofs.A.Side"] == "named: tests/test_doc.py"


def test_a_lemma_written_bare_outside_its_namespace_cites_its_module(stand_in):
    root, policy, usage = stand_in
    write(root, {source("Proofs.A.Side"): "import Proofs.A.Base\n\nnamespace Pool.Lemmas\n\n"
                                          "theorem side_eq : base = 1 := rfl\n\nend Pool.Lemmas\n"})
    for text, cited in (("/-! The bound is `side_eq`'s. -/\ndef bound : Nat := 1\n", True),
                        ("namespace Other\n\n/-- Through `Loop.side_eq`. -/\ndef bound : Nat := 1\n\nend Other\n", True),
                        ("/-! The stream `t/side_eq` is a path. -/\ndef bound : Nat := 1\n", False),
                        ("/-! Its own `side_eq`. -/\ntheorem side_eq : True := trivial\n", False),
                        ("/-! side_eq, not written as code. -/\ndef bound : Nat := 1\n", False)):
        write(root, {SPEC: text})
        got = decide(root, policy, usage, [TEST, SPEC])
        assert (got["kept"].get("Proofs.A.Side") == f"cited: a Lean comment of {SPEC} cites it") == cited, text


def test_a_module_whose_path_s_end_a_file_outside_writes_is_held(stand_in):
    """A test opens `Pins.lean` through a directory it joins from parts, and names it only in its docstring, by the end
    of its path: held for its owner, not deleted."""
    root, policy, usage = stand_in
    pins, reader = "Proofs.Pouw.PearlC.Cap.Pins", "verity/core/protocols/pouw/tests/test_cap_lean.py"
    usage["usage"]["modules"][pins] = {"constants": 1, "used": 0, "by": None, "refs": []}
    write(root, {source(pins): "theorem capPin_sm120 : 1 = 1 := by decide\n", reader: (
        '"""Lean\'s integer form of the cap circuit\'s tests (`Security/Proofs/Pouw/PearlC/Cap/IntForm.lean`) against the\n'
        'scheme: each kernel-checked pin in `Cap/Pins.lean` is what Python computes at that shape."""\n'
        "from pathlib import Path\n\n"
        'CAP = Path(__file__).resolve().parents[5] / "Security" / "Proofs" / "Pouw" / "PearlC" / "Cap"\n'
        'PINS = (CAP / "Pins.lean").read_text()\n')})
    got = decide(root, policy, usage, [TEST, reader])
    assert got["kept"][pins] == f"suffix: {reader} writes its path's end, `Cap/Pins.lean`"
    assert pins in decide(root, policy, usage, [TEST, reader], doubts=False)["unused"]

    def why(path: str, text: str, *more: str) -> str:
        write(root, {path: text})
        return decide(root, policy, usage, [TEST, path, *more])["kept"].get(pins, "")

    assert why("tools/x.py", "# from `PearlC/Cap/Pins.lean`\n").endswith("`PearlC/Cap/Pins.lean`")
    assert why("tools/x.py", "# from `Pins.lean`\n") == ""
    assert why("verity/README.md", "The pins are `Cap/Pins.lean`'s.\n") == ""
    assert why(f"{PKG}/Pouw/README.md", "The pins are `Cap/Pins.lean`'s.\n") == ""
    write(root, {"tools/Cap/Pins.lean": "theorem own : True := trivial\n"})
    assert why("tools/x.py", "# from `Cap/Pins.lean`\n", "tools/Cap/Pins.lean") == ""


def test_declared_and_opened_read_names_off_namespaces(tmp_path):
    src = ("namespace A.B\nsection S\ntheorem x : True := trivial\nend S\n@[simp] private lemma hidden : True := trivial\n"
           "protected noncomputable def y := 1\ninstance : Inhabited Nat := ⟨0⟩\n"
           "instance (priority := low) named : Inhabited Nat := ⟨0⟩\nend A.B\nnamespace C\nnamespace D\n"
           "theorem _root_.Top.z : True := trivial\nstructure S where\n  f : Nat\nend C.D\nmutual\n"
           "def m1 : Nat → Nat | 0 => 0 | n + 1 => m2 n\ndef m2 : Nat → Nat | 0 => 0 | n + 1 => m1 n\nend\n"
           "theorem w : True := trivial\n")
    write(tmp_path, {"x.lean": src, "y.lean": "namespace A.B\nopen C D.E in\ndef x := 1\nend A.B\n"})
    assert S.declared(S.Module("", "x.lean", None, tmp_path / "x.lean")) == {
        "A.B.x", "A.B.y", "A.B.named", "Top.z", "C.D.S", "m1", "m2", "w"}
    assert S.opened(S.Module("", "y.lean", None, tmp_path / "y.lean")) == {
        "A", "A.B", "C", "D.E", "in", "A.C", "A.D.E", "A.in", "A.B.C", "A.B.D.E", "A.B.in"}


def test_syntax_nobody_writes_and_a_local_one_keep_nothing(stand_in):
    root, policy, usage = stand_in
    write(root, {source("Proofs.A.Main"): UNUSED_SOURCES["Proofs.A.Main"].replace("⟪base⟫", "base")})
    assert "Proofs.B.Notation" in decide(root, policy, usage, [TEST])["unused"]
    write(root, {source("Proofs.A.Side"): UNUSED_SOURCES["Proofs.A.Side"] + "macro_rules | `(tactic| rfl) => `(tactic| rfl)\n"})
    assert decide(root, policy, usage, [TEST])["kept"]["Proofs.A.Side"].startswith("syntax: ")


def test_an_unbuilt_module_is_kept(stand_in):
    root, policy, usage = stand_in
    del usage["usage"]["modules"]["Proofs.B.Old"]
    assert decide(root, policy, usage, [TEST])["kept"]["Proofs.B.Old"].startswith("unbuilt: ")


def test_listing_carries_bytes_area_owner_and_totals(stand_in):
    """A module's owner is the lane the branches merged into its directory's history name."""
    root, policy, usage = stand_in
    git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *a], cwd=root, check=True,
                                    capture_output=True)
    git("init", "-q")
    git("add", f"{PKG}/A")
    git("commit", "-q", "-m", "Merge branch 'cursor/proofs-sweep-1a2b'")
    doc = S.listing(root, pruned(root)[1], decide(root, policy, usage, [TEST], ["Proofs.B.Old"]))
    rows = {r["module"]: r for r in doc["unused"]}
    assert rows["Proofs.A.Side"]["owner"] == "@proofs" and rows["Proofs.B.Group"]["owner"] is None
    assert rows["Proofs.A.Side"]["bytes"] == len(UNUSED_SOURCES["Proofs.A.Side"].encode())
    assert doc["areas"]["A"]["modules"] == 1 and doc["areas"]["A"]["owners"] == {"@proofs": 1} and doc["areas"]["B"] == {
        "modules": 1, "bytes": len(UNUSED_SOURCES["Proofs.B.Group"].encode()), "frozen": 1,
        "frozen_bytes": len(UNUSED_SOURCES["Proofs.B.Old"].encode()), "owners": {"-": 2}}
    assert doc["total"]["unused"] == 3 and doc["total"]["frozen"] == 1 and doc["total"]["kept_by"]["named"] == 2
    assert "Proofs.A.Side" in S.report(doc)


def test_a_guarantee_is_undeclared_only_where_no_package_listing_it_declares_it(tmp_path):
    write(tmp_path, {"code/lakefile.toml": 'name = "code"\n', "code/Code/Released.lean":
                     "namespace Code.Contract\n\ntheorem holds_computed : True := trivial\n\nend Code.Contract\n"})
    _, ms, _ = S.read(tmp_path)
    policies = {"code": {"guarantees": {"Code.Contract.holds_computed": {}, "Code.gone": {}}}, PKG: {"guarantees": {}}}
    pins = {"Code.Contract.holds_computed": None, "Code.gone": None, "Proofs.A.main": "Proofs.A.Main"}
    assert S.undeclared(ms, policies, pins) == ["Code.gone"]


def test_delete_rewrites_the_imports_through_what_it_deleted(stand_in):
    root, policy, usage = stand_in
    doc = S.listing(root, pruned(root)[1], decide(root, policy, usage, [TEST], ["Proofs.B.Old"]))
    files = {m: x.file for m, x in pruned(root)[1].items()}
    write(root, {"README.md": f"`{PKG}/B/Group.lean` groups them\n",
                 f"x/{S.C.POLICY}": '{"reads": {"Proofs.B.Group.Sub": {}, "Proofs.B.Group": {}}}\n',
                 f"y/{S.C.POLICY}": '{"reads": {"Proofs.B.Group.Sub": {}}}\n'})
    assert S.delete(root, pruned(root)[1], "B", doc, usage, set()) == ["Proofs.B.Group"]
    assert S.stale_mentions(root, ["Proofs.B.Group"], ["README.md", f"x/{S.C.POLICY}", f"y/{S.C.POLICY}"], files) == [
        f"README.md: {PKG}/B/Group.lean", f"x/{S.C.POLICY}: Proofs.B.Group"]
    assert not (root / source("Proofs.B.Group")).exists()
    assert (root / source("Proofs.Root")).read_text() == "import Proofs.A.Main\nimport Proofs.B.Old\n"
    assert S.delete(root, pruned(root)[1], "A", doc, usage, set()) == ["Proofs.A.Side"]
    assert (root / source("Proofs.B.Old")).read_text() == "import Proofs.A.Base\n\ntheorem old : True := trivial\n"
    assert S.delete(root, pruned(root)[1], "A", doc, usage, {"Proofs.A.Side"}) == []


def test_rewrite_keeps_each_line_s_form_and_drops_what_is_reached():
    reach = {"A": {"B"}, "B": set(), "C": set()}.get
    text = "module\n\npublic import A\npublic import D -- gone\nimport E\n\n/-- doc -/\ndef x := 1\n"
    got = S.rewrite_imports(text, {"D", "E"}, {"D": ["B", "C"], "E": ["C"]}, lambda m: reach(m) or set())
    assert got == "module\n\npublic import A\npublic import C\n\n/-- doc -/\ndef x := 1\n"


def test_references_name_paths_modules_and_constants(tmp_path):
    pkg = S.Package("security_proofs", PKG, "Security", [], {}, 0, None)
    mods = {"Proofs.A.Main": f"{PKG}/A/Main.lean", "Proofs.A": f"{PKG}/A.lean", "Proofs.B": f"{PKG}/B.lean"}
    files = {"tests/test_a.py": f'"{PKG}/A/Main.lean"\n', "b.sh": "lake env lean -e Proofs.B.thm\n",
             "c.toml": f'paths = ["{PKG}/A"]\n', "d.md": f"see `{PKG}/B.lean`\n",
             "e.py": "Proofs.Bx Proofs.A.X Proofs.A.Main.Y\n",
             "f.lean": "import Proofs.B\n\n/-! `Proofs.A.Main` proves it, by `Proofs.A.lemma`. -/\n",
             "tests/test_g.py": f'"{PKG}/A/" "{PKG}/B"\n', "tests/test_h.py": 'SOURCE = "import Proofs.A\\n#check 1"\n',
             "tests/test_i.py": 'FILES = ["Proofs/B.lean"]\nP = LEAN / "Proofs" / "A" / "Main.lean"\nQ = "Proofs/B"\n'}
    (tmp_path / PKG / "B").mkdir(parents=True)
    got = S.references(write(tmp_path, files), list(files), pkg, mods)
    named = {k: sorted(Path(f).name for f, _ in v) for k, v in got.items()}
    assert named == {"Proofs.A.Main": ["test_a.py", "test_i.py"], "Proofs.B": ["b.sh", "f.lean", "test_i.py"],
                     "Proofs.A": ["c.toml", "test_h.py"],
                     ("docs", "Proofs.B"): ["d.md"], ("cited", "Proofs.A.Main"): ["f.lean"],
                     ("cited", "Proofs.A"): ["f.lean"]}
    reads = {(k, Path(f).name): r for k, v in got.items() for f, r in v}
    assert not reads["Proofs.A.Main", "test_a.py"] and reads["Proofs.B", "b.sh"] and reads["Proofs.B", "f.lean"]
    assert not reads["Proofs.A", "c.toml"] and reads["Proofs.A", "test_h.py"]
    assert reads[("cited", "Proofs.A.Main"), "f.lean"] and not reads[("cited", "Proofs.A"), "f.lean"]


#: the controls with an unused module (`Extra`), an aggregator of it (`Group`), a notation the proofs write (`Syn`) and a
#: definition only an unlisted theorem of a used module reads (`Base`); `Scratch` (exempt, so its imports are kept)
#: imports only `Defs`
CONTROL_EDITS = {
    "Control.lean": "import Control.Assumptions\nimport Control.Defs\nimport Control.Thms\nimport Control.Group\n",
    "Control/Base.lean": "namespace Control\n\ndef base : Nat := 1\n\nend Control\n",
    "Control/Syn.lean": "import Control.Defs\n\nnamespace Control\n\nnotation \"dbl \" n:max => double n\n\nend Control\n",
    "Control/Extra.lean": "import Control.Defs\n\nnamespace Control\n\ntheorem extra_unused : double 2 = 4 := rfl\n\nend Control\n",
    "Control/Group.lean": "import Control.Extra\n",
    "Scratch.lean": "import Control.Defs\n\n#check @Control.double\n",
}


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_facts_usage_mode_on_the_controls(tmp_path):
    """`Facts.lean`'s usage mode compiled and run on a real build: the stand-in document above is what it writes (each
    guarantee's module, the closure's counts, `refs`), which only a build shows."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    write(pkg, CONTROL_EDITS)
    thms = pkg / "Control" / "Thms.lean"
    thms.write_text(thms.read_text().replace("import Control.Defs\n", "import Control.Defs\nimport Control.Base\n"
                                             "import Control.Syn\n")
                    .replace("\nend Control", "\ntheorem helper2 : base + 1 = dbl 1 := rfl\n\nend Control"))
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=S.C.lake_env())
    policy = S.C.load(pkg)
    pins, extra = S.names_of(policy)
    usage = S.usage_of(pkg, policy["roots"], pins, sorted(set(extra) - set(pins)), tmp_path / "facts.json")
    assert usage["usage"]["pins"] == {"Control.main_thm": "Control.Thms", "Control.Small_fails": "Control.Thms",
                                      "Control.Small_meets": "Control.Thms"}
    facts = usage["usage"]["modules"]
    assert set(facts) == {"Control", "Control.Assumptions", "Control.Defs", "Control.Thms", "Control.Base", "Control.Syn",
                          "Control.Extra", "Control.Group"}
    assert all(facts[m]["used"] > 0 for m in ("Control.Assumptions", "Control.Defs", "Control.Thms"))
    assert facts["Control.Thms"]["used"] < facts["Control.Thms"]["constants"]
    assert all(facts[m]["used"] == 0 for m in ("Control", "Control.Base", "Control.Syn", "Control.Extra", "Control.Group"))
    assert facts["Control"]["constants"] == facts["Control.Group"]["constants"] == 0
    assert sorted(facts["Control.Thms"]["refs"]) == ["Control.Assumptions", "Control.Base", "Control.Defs"]
    assert facts["Control.Extra"]["refs"] == ["Control.Defs"]
    graph = usage["graph"]
    assert "Init.Prelude" in graph and set(graph["Control.Group"]["imports"]) - {"Init"} == {"Control.Extra"}
    pkgs, ms, _ = S.read(tmp_path)
    got = S.decide(tmp_path, pkgs["control"], {m.name: m for m in ms if m.pkg.dir == "control"}, policy, usage, [], [])
    why = {m: r.split(":", 1)[0] for m, r in got["kept"].items()}
    assert why == {"Control": "root", "Control.Assumptions": "used", "Control.Defs": "used", "Control.Thms": "used",
                   "Control.Base": "refs", "Control.Syn": "syntax", "Scratch": "exempt", "RunOk": "runs"}
    assert got["unused"] == ["Control.Extra", "Control.Group"]
