"""`lean.py`, the Lean layout move: its renaming rules, the guard against paths a case-insensitive checkout can't hold, and
that the move has nothing left to do on this tree (a rerun renames no module)."""
from __future__ import annotations

import importlib.util
import json
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


def test_theorems_are_named_through_namespaces():
    text = """namespace Flock
theorem top_ok : True := trivial
namespace HmNets
@[simp] theorem check_ok : True := trivial
end HmNets
section
private theorem Layout.check_ok : True := trivial
end
theorem _root_.Other.x : True := trivial
end Flock
lemma loose : True := trivial
"""
    assert M.theorems(text) == {"Flock.top_ok", "Flock.HmNets.check_ok", "Flock.Layout.check_ok", "Other.x", "loose"}


def test_a_dropped_root_keeps_its_imports_reachable(tmp_path):
    m = M.Map(TOY, tmp_path)
    proofs = tmp_path / "sec" / "Proofs"
    (proofs / "A").mkdir(parents=True)
    (proofs / "A.lean").write_text("import Proofs.A.Used\n\n/-! A's proofs -/\n")
    for n in ("Used", "Only", "Under"):
        (proofs / "A" / f"{n}.lean").write_text("import Proofs.A.Under\n" if n == "Only" else "")
    ren = {"A.SecurityProofs": "Proofs.A", "A.SecurityProofs.Used": "Proofs.A.Used",
           "A.SecurityProofs.Only": "Proofs.A.Only"}
    dropped = {"A": ["A.SecurityProofs", "A.SecurityProofs.Only", "A.SecurityProofs.Used"]}
    assert M.keep_reachable(m, tmp_path, ren, dropped, False) == ["Proofs.A imports Proofs.A.Only"]
    assert (proofs / "A.lean").read_text().startswith("import Proofs.A.Used\nimport Proofs.A.Only\n\n")
    assert M.keep_reachable(m, tmp_path, ren, dropped, False) == []


def test_the_merged_locks_are_laid_out_as_the_audit_writes_them(tmp_path):
    import check as A
    toy = {**TOY, "packages": {**TOY["packages"], "verifier": {"dir": "v/lean", "code": ["Flock"]},
                               "pous": {"dir": "p/lean", "code": ["Grader"]}}}
    m = M.Map(toy, tmp_path)
    rec = {"owner": "@lean", "signature": ["A.SecurityProofs.G : True"], "assumptions": [], "type_hash": "00"}
    locks = {"a": {"roots": ["A"], "guarantees": {"A.SecurityProofs.G": rec}, "reads": {}, "dependencies": {"toolchain": "t"}}}
    M.merge_locks(m, tmp_path, locks, {}, False)
    for rel in ("sec", "sec/Proofs", "v/lean", "p/lean"):
        text = (tmp_path / rel / M.POLICY).read_text()
        assert text == A.text(json.loads(text)), rel
    assert '"roots": ["Definitions", "Specs"]' in (tmp_path / "sec" / M.POLICY).read_text()


def test_the_tree_is_moved():
    m = M.Map(tomllib.loads((HERE.parent / "lean_map.toml").read_text()), ROOT)
    files = M.tracked(ROOT)
    _, dropped, by_path = M.rename_all(m, M.modules(m, files))
    assert not by_path and not dropped, f"lean.py would still move: {sorted(by_path.items())[:5]}"
    assert not M.case_clashes(files)
