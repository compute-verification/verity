"""`tools/lean/model_split.py`: a package's mixed modules split into a spec half (`<root>.Protocol.<rest>`) and a proof
half, as a pure move: on a small package built with `lake`, every declaration keeps its name, the spec imports no proof,
the proofs still build, and the lock's `reads` change only in the module names they are keyed by."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_model_split", HERE.parent / "model_split.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
A = S.A

DEFS = """import P.Base

/-! # Defs -/

namespace P

/-- `f`, the identity. -/
def f (n : Nat) : Nat := n

theorem f_id (n : Nat) : f n = n := rfl

@[simp] theorem f_five : f 5 = 5 := rfl

/-- `simp` uses the rfl lemma `f_five` from its default set: neither the term nor the text names it. -/
theorem five_eq : f 5 + 0 = 5 := by simp

def five : {n : Nat // n = 5} := ⟨f 5 + 0, five_eq⟩

private theorem two_pos : 0 < 2 := by decide

/-- Its value names a private lemma, which moves with it. -/
def two : {n : Nat // 0 < n} := ⟨2, two_pos⟩

-- `g` is `f 1`.
open Nat in
def g : Nat := f 1

@[simp] theorem g_eq : g = 1 := rfl

theorem lt3 : 0 < 3 := by decide

/-- Its value's proof names `lt3`, which only `three`'s auxiliary proof uses. -/
def three : {n : Nat // 0 < n} := ⟨3, by exact Nat.lt_of_lt_of_le lt3 (Nat.le_refl 3)⟩

theorem f_four : f 4 = 4 := rfl

/-- `simp` leaves the rfl lemma `f_four` out of the term. -/
theorem four_eq : f 4 = 4 := by simp only [f_four]

def four : {n : Nat // n = 4} := ⟨f 4, four_eq⟩

/-- One. -/
local notation "𝟙" => f 1

def one : Nat := 𝟙

theorem one_eq : 𝟙 = 1 := rfl

/-- An example. -/
example : g = 1 := rfl

section

variable (k : Nat) (b : Fin (base + 1))

/-- It names `k`, a binder of a `variable` whose other binder's type names a proof module's definition. -/
def addk (n : Nat) : Nat := n + k

theorem b_lt : b.val < base + 1 := b.isLt

end

end P
"""
FILES = {
    "lakefile.toml": 'name = "p"\ndefaultTargets = ["P"]\n\n[[lean_lib]]\nname = "P"\n',
    "P.lean": "import P.Thms\nimport P.Assumptions\n",
    "P/Base.lean": "theorem P.base_thm : True := trivial\n\ndef P.base : Nat := 0\n",
    "P/Assumptions.lean": "def P.Hyp : Prop := True\n",
    "P/Defs.lean": DEFS,
    "P/Thms.lean": "import P.Defs\n\ntheorem P.t : P.g = 1 := P.g_eq\n\ntheorem P.u : P.base = 0 := rfl\n",
}
LOCK = {"roots": ["P"], "assumptions": ["P.Assumptions"],
        "pins": {"P.t": {"signature": ["P.t : Eq P.g 1"], "assumptions": [], "type_hash": "aa"}},
        "reads": {"P.Defs": {"digest": "d", "definitions": {"P.f": "1", "P.g": "2", "P.two": "3", "P.three": "4", "P.four": "5",
                                                            "P.five": "6", "P.one": "7", "P.addk": "8"}}}}


def package(tmp: Path) -> Path:
    pkg = tmp / "p"
    for rel, text in FILES.items():
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text(text)
    (pkg / "lean-toolchain").write_text((HERE / "fixtures" / "control" / "lean-toolchain").read_text())
    A.dump(pkg, LOCK)
    subprocess.run(["git", "init", "-q"], cwd=pkg, check=True)
    return pkg


def decls(pkg: Path) -> dict:
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    cfg, out = pkg / ".lake" / "decls-cfg.json", pkg / ".lake" / "decls.json"
    cfg.write_text(json.dumps({"roots": ["P"], "under": ["P"], "out": str(out)}))
    subprocess.run(["lake", "env", "lean", "--run", str(HERE.parent / "Decls.lean"), str(cfg)], cwd=pkg, check=True,
                   capture_output=True, env=A.lake_env())
    return S.load_decls(out)


def names(d: dict) -> set[str]:
    """Every declaration's name, but a local notation's macro and unexpander, which are named after their module."""
    return {n for m in d["modules"].values() for x in m["decls"]
            if not (n := S.user_name(x["n"])).split(".")[-1].lstrip("«").startswith("_aux_")}


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_mixed_module_splits_into_a_spec_half_and_its_proofs_and_still_builds(tmp_path):
    pkg = package(tmp_path)
    before = decls(pkg)
    uses = {x["n"]: x["u"] for x in before["modules"]["P.Defs"]["decls"]}
    assert "P.lt3" not in uses["P.three"] and "P.f_four" not in uses["P.four_eq"] and "P.f_five" not in uses["P.five_eq"]
    p = S.plan(pkg, before, [])
    assert p["side"] == {"P": "proofs", "P.Assumptions": "spec", "P.Base": "proofs", "P.Defs": "mixed", "P.Thms": "proofs"}
    assert p["protocol"] == {"P.Defs": "P.Protocol.Defs"} and p["new_imports"]["P.Protocol.Defs"] == []
    assert S.render(p) == S.render(S.plan(pkg, before, []))
    S.apply(p)
    spec_half, proofs = (pkg / "P/Protocol/Defs.lean").read_text(), (pkg / "P/Defs.lean").read_text()
    assert "import" not in spec_half and "/-- `f`, the identity. -/\ndef f" in spec_half
    assert "-- `g` is `f 1`.\nopen Nat in\ndef g" in spec_half and "two_pos" in spec_half
    assert all(f"theorem {x} " in spec_half for x in ("lt3", "f_four", "four_eq", "f_five", "five_eq"))
    assert not any(x in spec_half for x in ("f_id", "g_eq", "one_eq", "example"))
    assert proofs.startswith("import P.Base\nimport P.Protocol.Defs\n")
    assert all(x in proofs for x in ("theorem f_id", "@[simp] theorem g_eq", "theorem one_eq", "/-- An example. -/\nexample"))
    assert "def " not in proofs and all(x.count('/-- One. -/\nlocal notation "𝟙"') == 1 for x in (spec_half, proofs))
    assert "\nvariable (k : Nat)\n" in spec_half and "base" not in spec_half
    assert "\nvariable (k : Nat) (b : Fin (base + 1))\n" in proofs and "theorem b_lt" in proofs
    assert p["warnings"] == ["P.Defs:54: the spec half leaves out the `variable` binders b (naming P.base), a proof's or one "
                             "left out"]
    after = decls(pkg)
    assert names(after) == names(before)
    lock = A.load(pkg)
    assert lock["reads"] == {"P.Protocol.Defs": LOCK["reads"]["P.Defs"]} and lock["pins"] == LOCK["pins"]
    assert lock["layers"]["P.Protocol"]["imports"] == ["P.Assumptions", "P.Protocol"]
    assert set(after["modules"]["P.Protocol.Defs"]["imports"]) == {"Init"}


def test_a_module_under_a_spec_name_that_holds_a_proof_is_refused(tmp_path):
    pkg = tmp_path / "p"
    (pkg / "P").mkdir(parents=True)
    (pkg / "P/Protocol.lean").write_text("def P.f : Nat := 0\n\ntheorem P.f_eq : P.f = 0 := rfl\n")
    A.dump(pkg, {"roots": ["P"], "reads": {}})
    facts = {"modules": {"P.Protocol": {"imports": ["Init"], "decls": [
        {"n": "P.f", "k": "def", "r": [1, 0, 1, 18], "u": []},
        {"n": "P.f_eq", "k": "theorem", "r": [3, 0, 3, 31], "u": ["P.f"]}]}}, "imports": {"P.Protocol": ["Init"]}}
    with pytest.raises(S.Refused, match="P.Protocol is the spec's by its name but holds 1 proof"):
        S.plan(pkg, facts, [])
