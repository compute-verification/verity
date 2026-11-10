"""The axiom probe `verity lean try` runs (`axiom_probe.py`): its scan of a file's theorems, the file's own status, its line,
and, when `lake` is installed, a sorry-backed theorem beside a clean one on the controls' fixture, in the module system's
form and the plain one."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("axiom_probe_for_tests", HERE.parent / "axiom_probe.py")
AP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AP)

SOURCE = '''module

public import Control.Defs

/-! A docstring's `theorem inDoc : False := sorry` is no theorem,
/- nor a nested comment's theorem nested : False -/ -/

@[expose] public section

namespace A.B

-- theorem inLine : False := sorry
theorem clean (n : Nat) : n + 0 = n := rfl

@[simp] private theorem hidden : True := trivial

def s : String := "\\"
theorem inString : False"

protected lemma prot' : True := trivial

section Inner
theorem _root_.Top.rooted : True := trivial
end Inner

mutual
theorem m : True := trivial
end

end A.B

namespace C
noncomputable section
theorem D.e : True := trivial
end
end C

theorem top : True := trivial
'''


def test_the_scan_names_each_top_level_theorem_under_its_open_namespaces_and_nothing_in_comments_or_strings():
    assert AP.theorems(SOURCE) == ["A.B.clean", "A.B.hidden", "A.B.prot'", "Top.rooted", "A.B.m", "C.D.e", "top"]


def test_the_probe_appends_one_print_axioms_per_theorem_after_the_files_last_line():
    assert AP.with_probe("theorem t : True := trivial", ["t", "A.u"]).splitlines() == [
        "theorem t : True := trivial", "", "#print axioms t", "#print axioms A.u"]


def test_an_error_only_on_an_appended_line_is_the_probes_and_any_other_is_the_files():
    copy = ".lake/axiom-probe/X.lean"
    assert AP.own_rc(1, f"{copy}:12:14: error(lean.unknownIdentifier): Unknown constant `t`\n", 10, copy) == 0
    assert AP.own_rc(1, f"{copy}:3:0: error: unknown identifier 'x'\n{copy}:12:14: error: x\n", 10, copy) == 1
    assert AP.own_rc(1, "Other.lean:12:0: error: x\n", 10, copy) == 1, "an error elsewhere is the file's"
    assert AP.own_rc(1, "", 10, copy) == 1, "a failure with no position is the file's"
    assert AP.own_rc(0, "", 10, copy) == 0


def test_the_line_names_each_theorem_beyond_the_standard_axioms_and_each_it_got_no_answer_for():
    out = ("'A.holey' depends on axioms: [propext,\n sorryAx]\n'_private.«.lake».«axiom-probe».X.0.A.hidden' does not "
           "depend on any axioms\n'A.cls' depends on axioms: [propext, Classical.choice, Quot.sound]\n'A.x'' depends on "
           "axioms: [A.ax]\n")
    assert AP.summary(["A.holey", "A.hidden", "A.cls", "A.x'"], out) == (
        "axiom probe: 2 of its 4 theorems use axioms beyond propext, Classical.choice and Quot.sound: "
        "A.holey (sorryAx); A.x' (A.ax)")
    assert AP.summary(["A.hidden", "A.cls"], out) == (
        "axiom probe: none of its 2 theorems uses an axiom beyond propext, Classical.choice and Quot.sound")
    assert AP.summary(["A.cls", "A.gone"], out).endswith(
        "none of the 1 it read uses an axiom beyond propext, Classical.choice and Quot.sound; no answer for 1 (an error "
        "on its `#print axioms` line above): A.gone")
    assert AP.summary([], "") == "axiom probe: no theorems in it"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_on_the_controls_fixture_the_probe_names_the_sorry_backed_theorem_only(tmp_path):
    """Lean's exit 0 on both files; the probe tells the theorem resting on `sorry` from the clean one, in a `module` (whose
    theorems are private, so named `_private.…` in the answers) as in a plain file importing the package's modules."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    subprocess.run(["lake", "build", "Control.Defs"], cwd=pkg, check=True, capture_output=True)
    files = {"Control/Plain.lean": "import Control.Defs\n\nnamespace Control\n\ntheorem holey : double 1 = 3 := sorry\n\n"
                                   "theorem clean : double 0 = 0 := rfl\n\nend Control\n",
             "Control/Mod.lean": "module\n\nnamespace Probe\n\ntheorem holey : 1 = 2 := by sorry\n\n"
                                 "theorem clean (n : Nat) : n + 0 = n := rfl\n\nend Probe\n"}

    def lean(copy):
        p = subprocess.run(["lake", "env", "lean", copy], cwd=pkg, capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr

    for rel, text in files.items():
        (pkg / rel).write_text(text)
        rc, said, _ = AP.probe(pkg, rel, text, lean)
        ns = "Control" if "Plain" in rel else "Probe"
        assert (rc, said) == (0, f"axiom probe: 1 of its 2 theorems use axioms beyond propext, Classical.choice and "
                                 f"Quot.sound: {ns}.holey (sorryAx)"), rel
        assert not (pkg / AP.COPIES / rel).exists(), "the copy goes"
