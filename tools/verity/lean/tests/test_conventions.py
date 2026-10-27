"""`tools/verity/lean/conventions.py`: the survey's counts on a small Security-shaped package, one case of each measure."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("lean_conventions", HERE.parent / "conventions.py")
C = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

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


def package(root: Path) -> Path:
    pkg = root / "Security"
    for rel, text in SOURCES.items():
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text(text)
    (pkg / "lean-audit.json").write_text(json.dumps(LOCK))
    return pkg


def test_survey_counts_each_measure(tmp_path):
    pkg = package(tmp_path)
    assert C.packages(tmp_path) == [pkg]
    r = C.survey(tmp_path.resolve(), [pkg.resolve()], 5)

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


def test_main_prints_and_writes_json(tmp_path, capsys):
    pkg = package(tmp_path)
    out = tmp_path / "survey.json"
    assert C.main(["--root", str(tmp_path), "--json", str(out), str(pkg)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("1 packages, 4 files")
    assert "2 overrides in 1 files, 1 with a reason (50.0%)" in printed
    assert json.loads(out.read_text())["locks"]["Security"]["guarantees"] == 2
