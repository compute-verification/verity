"""The axiom probe `verity lean try` runs (`axiom_probe.py`): its scan of a file's theorems and of each one's own lines, the
file's own status, the theorems that failed, its line and what it shows of lean's output, and, when `lake` is installed,
on the controls' fixture, in the module system's form and the plain one: a sorry-backed theorem beside a clean one, and
theorems Lean refuses beside ones it accepts."""
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


OWN_LINES = '''namespace N

def f : Nat → Nat
  | 0 => 0
  | n + 1 => f n

/-- Its docstring,
over two lines. -/
theorem documented : True := trivial

@[simp]
private
theorem attributed : True := trivial
-- a comment under it

open Nat in
set_option maxRecDepth 300 in
/-- Under two `… in`. -/
theorem opened : True := by
  trivial

theorem cases : (n : Nat) → True
| 0 => trivial
| _ + 1 => trivial
termination_by n => n

def g : Nat := 1

  theorem indented : True :=
by
    trivial
  @[simp] theorem inline : f 0 = 0 := by simp [f]

theorem last : True := trivial
end N
'''


def test_a_theorems_own_lines_run_from_its_docstring_attributes_and_ins_to_its_last_line_of_code():
    """Lean puts an error on a whole declaration at its first line, a docstring's or an attribute's when it has one."""
    assert AP.spans(OWN_LINES) == [
        ("N.documented", 6, 9), ("N.attributed", 10, 13), ("N.opened", 14, 20), ("N.cases", 21, 25),
        ("N.indented", 28, 31), ("N.inline", 32, 32), ("N.last", 33, 34)]
    assert AP.theorems(OWN_LINES) == [name for name, _, _ in AP.spans(OWN_LINES)]


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


REFUSED = "namespace P\n\ntheorem clean : True := trivial\n\n/-- Refused. -/\ntheorem refused : way 100000 = 100000 := rfl\n\n" \
          "private theorem wrong : (1 : Nat) = 2 := rfl\n\ndef broken : Nat := missing\n\ntheorem after : True := trivial\n\n" \
          "end P\n"
COPY = ".lake/axiom-probe/X.lean"
#: lean's lines for `REFUSED` with the probe appended, as Lean 4.34.0 prints them for such a file: "does not depend on any
#: axioms" for the theorem whose `rfl` ran out of recursion depth, which the kernel never got
REFUSED_SAID = [
    f"{COPY}:5:0: error: maximum recursion depth has been reached\n",
    "use `set_option maxRecDepth <num>` to increase limit\n",
    f"{COPY}:8:39: error: Type mismatch\n",
    "  rfl\n",
    f"{COPY}:10:20: error(lean.unknownIdentifier): Unknown identifier `missing`\n",
    "'P.clean' does not depend on any axioms\n",
    "'P.refused' does not depend on any axioms\n",
    "'_private.«.lake».«axiom-probe».X.0.P.wrong' depends on axioms: [propext,\n",
    " sorryAx]\n",
    "'P.after' does not depend on any axioms\n"]


def test_a_theorem_failed_when_lean_has_an_error_on_its_own_lines_and_its_answer_is_not_shown():
    said = "".join(REFUSED_SAID)
    bad = AP.failed(AP.spans(REFUSED), said, COPY)
    assert bad == ["P.refused", "P.wrong"], "the error at 10 is a def's, on no theorem's lines"
    assert AP.failed(AP.spans(REFUSED), said.replace(COPY, "Other.lean"), COPY) == [], "an error elsewhere is no theorem's"
    assert AP.shown(said, bad) == "".join(REFUSED_SAID[:6] + REFUSED_SAID[9:])
    assert AP.shown(said, []) == said


def test_the_probe_prints_leans_lines_as_they_come_and_the_answers_last_without_a_failed_theorems(tmp_path, capsys):
    """`#print axioms` answers "does not depend on any axioms" for a theorem Lean refused: the probe's line names it as
    failed, and nothing printed can be read as clean for it."""
    seen = []

    def lean(copy, show):
        for line in REFUSED_SAID:
            show(line)
            seen.append(capsys.readouterr().out)
        return 1, "".join(REFUSED_SAID)

    rc, said, output = AP.probe(tmp_path, "X.lean", REFUSED, lean)
    assert (rc, said) == (1, "axiom probe: 2 of its 4 theorems failed, with Lean's errors above on their own lines, and "
                             "are not read for axioms: P.refused, P.wrong")
    assert seen == REFUSED_SAID[:5] + [""] * 5, "each of lean's lines as it comes, up to the first answer"
    assert capsys.readouterr().out == REFUSED_SAID[5] + REFUSED_SAID[9]
    assert output == "".join(REFUSED_SAID), "the caller gets all lean said"


def test_no_line_when_the_file_fails_on_no_theorems_lines_and_every_answer_is_shown(tmp_path, capsys):
    said = [f"{COPY}:3:20: error(lean.unknownIdentifier): Unknown identifier `missing`\n",
            "'t' does not depend on any axioms\n"]

    def lean(copy, show):
        for line in said:
            show(line)
        return 1, "".join(said)

    assert AP.probe(tmp_path, "X.lean", "theorem t : True := trivial\n\ndef broken : Nat := missing\n", lean) == (
        1, "", "".join(said))
    assert capsys.readouterr().out == "".join(said)


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
    assert AP.summary(["A.holey", "A.hidden", "A.cls"], out, ["A.cls"]) == (
        "axiom probe: 1 of its 3 theorems failed, with Lean's error above on its own lines, and is not read for axioms: "
        "A.cls"), "of a file lean refuses, the theorems that failed and no axioms"


@pytest.fixture(scope="module")
def control(tmp_path_factory):
    """A copy of the controls' fixture with its one module built, and `lake env lean` on a file of it, each line of its
    output handed on."""
    pkg = tmp_path_factory.mktemp("probe") / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    subprocess.run(["lake", "build", "Control.Defs"], cwd=pkg, check=True, capture_output=True)

    def lean(copy, show):
        p = subprocess.run(["lake", "env", "lean", copy], cwd=pkg, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout.splitlines(keepends=True):
            show(line)
        return p.returncode, p.stdout

    return pkg, lean


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_on_the_controls_fixture_the_probe_names_the_sorry_backed_theorem_only(control):
    """Lean's exit 0 on both files; the probe tells the theorem resting on `sorry` from the clean one, in a `module` (whose
    theorems are private, so named `_private.…` in the answers) as in a plain file importing the package's modules."""
    pkg, lean = control
    files = {"Control/Plain.lean": "import Control.Defs\n\nnamespace Control\n\ntheorem holey : double 1 = 3 := sorry\n\n"
                                   "theorem clean : double 0 = 0 := rfl\n\nend Control\n",
             "Control/Mod.lean": "module\n\nnamespace Probe\n\ntheorem holey : 1 = 2 := by sorry\n\n"
                                 "theorem clean (n : Nat) : n + 0 = n := rfl\n\nend Probe\n"}
    for rel, text in files.items():
        (pkg / rel).write_text(text)
        rc, said, _ = AP.probe(pkg, rel, text, lean)
        ns = "Control" if "Plain" in rel else "Probe"
        assert (rc, said) == (0, f"axiom probe: 1 of its 2 theorems use axioms beyond propext, Classical.choice and "
                                 f"Quot.sound: {ns}.holey (sorryAx)"), rel
        assert not (pkg / AP.COPIES / rel).exists(), "the copy goes"


LEAN_REFUSES = '''namespace Probe

def way : Nat → Nat
  | 0 => 0
  | n + 1 => way n + 1

theorem clean (n : Nat) : n + 0 = n := rfl

theorem refused : way 100000 = 100000 := rfl

/-- A docstring
over two lines. -/
theorem documented : way 100000 = 100000 := rfl

@[simp]
theorem attributed : way 100000 = 100000 := rfl

open Nat in
/-- Under an `open … in`. -/
private theorem opened : way 100000 = 100000 := rfl

theorem cases : (n : Nat) → way 100000 = 100000
  | 0 => rfl
  | _ + 1 => rfl

theorem wrong : (1 : Nat) = 2 := rfl

theorem uses : way 100000 = 100000 := refused

def broken : Nat := missing

theorem after : True := trivial

end Probe
'''


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
@pytest.mark.parametrize("head", ["import Control.Defs\n\n", "module\n\n"])
def test_on_the_controls_fixture_a_theorem_lean_refuses_is_named_failed_and_no_answer_for_it_is_shown(control, capsys, head):
    """A term-mode `rfl` that runs out of recursion depth, bare and under a docstring, an attribute, an `open … in` and a
    match, a type mismatch, and a theorem using a refused one: lean exits 1 with its errors on their lines, and the
    theorems it accepts beside them keep their answers.  Nothing printed says a failed one's axioms."""
    pkg, lean = control
    rel, text = "Control/Refused.lean", head + LEAN_REFUSES
    (pkg / rel).write_text(text)
    rc, said, output = AP.probe(pkg, rel, text, lean)
    bad = ["refused", "documented", "attributed", "opened", "cases", "wrong", "uses"]
    assert (rc, said) == (1, "axiom probe: 7 of its 9 theorems failed, with Lean's errors above on their own lines, and "
                             "are not read for axioms: " + ", ".join(f"Probe.{n}" for n in bad))
    out = capsys.readouterr().out
    assert AP.axioms(out) == {"Probe.clean": [], "Probe.after": []}, out
    assert out.count("maximum recursion depth has been reached") >= 5 and "(kernel) unknown constant '" in out
    assert AP.axioms(output)["Probe.refused"] == [], "lean's own answer for the theorem it refused, which reads as clean"
