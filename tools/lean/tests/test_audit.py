"""`tools/lean/audit.py`: its judgement on the facts `Facts.lean` reports (no Lean needed), and the controls when `lake` is
installed (the fixture package passes clean and fails each check once a violation is added)."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_audit", HERE.parent / "audit.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

STD = ["Classical.choice", "Quot.sound", "propext"]


def pin(signature="P.t (h : P.Hyp) (n : Nat) : P.f n = n", hyps=None, reads=None, axioms=STD, kind="theorem", h="00000000000000aa"):
    return {"kind": kind, "module": "P.Thms", "signature": signature, "axioms": axioms, "type_hash": h,
            "hypotheses": hyps if hyps is not None else [{"binder": "h", "type": "P.Hyp", "closed": True, "named": "P.Hyp",
                                                          "module": "P.Assumptions"}],
            "reads": reads if reads is not None else [{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "1" * 32},
                                                      {"name": "P.f.match_1", "module": "P.Defs", "kind": "def", "hash": "2" * 32},
                                                      {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "3" * 32}]}


def facts(**over):
    f = {"lean": "4.34.0", "modules": ["P", "P.Assumptions", "P.Defs", "P.Thms"],
         "imports": {"P": ["Init", "P.Thms"], "P.Assumptions": ["Init"], "P.Defs": ["Init"], "P.Thms": ["P.Assumptions", "P.Defs"]},
         "declarations": {"theorem": 1, "def": 3}, "axioms": STD, "offenders": [], "declared_axioms": [], "escapes": [],
         "pins": {"P.t": pin()}}
    return f | over


FILES = {"P": "P.lean", "P.Assumptions": "P/Assumptions.lean", "P.Defs": "P/Defs.lean", "P.Thms": "P/Thms.lean"}
POLICY = {"roots": ["P"], "assumptions": ["P.Assumptions"], "pins": {"P.t": {}}}


def recorded(f=None):
    fails, pol = A.judge(POLICY, f or facts(), FILES, update=True)
    assert fails == []
    return pol


def test_a_recorded_package_passes_and_its_record_is_the_statement():
    pol = recorded()
    assert pol["pins"]["P.t"] == {"signature": ["P.t (h : P.Hyp) (n : Nat) : P.f n = n"], "assumptions": ["P.Hyp"],
                                  "type_hash": "00000000000000aa"}
    assert list(pol["reads"]["P.Defs"]["definitions"]) == ["P.f"] and pol["reads"]["P.Defs"]["pins"] == ["P.t"]
    assert A.judge(pol, facts(), FILES) == ([], pol)
    assert A.judge(POLICY, facts(), FILES)[0] == ["pins: P.t has no record; run `audit.py --update` and commit the record"]


def test_sorry_native_decide_and_other_axioms_are_named():
    off = [{"name": "P.bad", "module": "P.Thms", "axioms": ["sorryAx"]},
           {"name": "P.big", "module": "P.Thms", "axioms": ["P.big._native.native_decide.ax_1_1"]},
           {"name": "P.big._native.native_decide.ax_1_1", "module": "P.Thms", "axioms": ["P.big._native.native_decide.ax_1_1"]},
           {"name": "P.c", "module": "P.Thms", "axioms": ["Lean.ofReduceBool"]}, {"name": "P.d", "module": "P.Defs", "axioms": ["Foo.ax"]}]
    decl = [{"name": "P.big._native.native_decide.ax_1_1", "module": "P.Thms"}, {"name": "P.magic", "module": "P.Defs"}]
    fails = A.judge(recorded(), facts(offenders=off, declared_axioms=decl), FILES)[0]
    assert "sorry: sorryAx is used by P.bad" in fails
    assert "native_decide: P.big._native.native_decide.ax_1_1 is used by P.big" in fails
    assert "native_decide: Lean.ofReduceBool is used by P.c" in fails and "axiom: Foo.ax is used by P.d" in fails
    assert [f for f in fails if f.startswith("declared:")] == [
        "declared: `axiom P.magic` in P.Defs; state an assumption as a named Prop, taken as a hypothesis"]


def test_escapes_must_be_listed_and_listed_ones_must_exist():
    esc = [{"name": "P.loop", "module": "P.Defs", "why": ["opaque"]}]
    fails = A.judge(recorded(), facts(escapes=esc), FILES)[0]
    assert fails and fails[0].startswith("escape: P.loop (opaque, in P.Defs)")
    pol = recorded() | {"escapes": {"P.loop": "why", "P.gone": "why"}}
    assert A.judge(pol, facts(escapes=esc), FILES)[0] == ["escape: P.gone is listed under `escapes` but is not one; remove the entry"]


def test_every_file_is_audited_or_exempt():
    files = FILES | {"P.Unused": "P/Unused.lean", "Scratch": "Scratch.lean"}
    fails = A.judge(recorded() | {"exempt": {"Scratch": "why", "Old": "why"}}, facts(), files)[0]
    assert [f.split(" ")[1] for f in fails] == ["P/Unused.lean", "`exempt`"]


def test_layers_hold_on_direct_imports():
    pol = recorded() | {"layers": {"P.Defs": {"imports": [], "why": ""}, "P.Thms": {"imports": ["P.Defs"], "except": []}}}
    assert A.judge(pol, facts(), FILES)[0] == ["layers: P.Thms imports P.Assumptions, outside P.Thms's layer (P.Defs)"]
    pol["layers"]["P.Thms"]["except"] = ["P.Thms"]
    assert A.judge(pol, facts(), FILES)[0] == []
    pol["layers"]["P.Model"] = {"imports": []}
    assert A.judge(pol, facts(), FILES)[0] == ["layers: P.Model matches no audited module; remove the entry"]


def test_a_changed_statement_or_meaning_fails_until_updated():
    pol = recorded()
    fails = A.judge(pol, facts(pins={"P.t": pin(signature="P.t (h : P.Hyp) (n : Nat) : P.f n ≤ n", h="00000000000000bb")}), FILES)[0]
    assert len(fails) == 1 and fails[0].startswith("pins: P.t's signature changed:") and "+P.t (h : P.Hyp) (n : Nat) : P.f n ≤ n" in fails[0]
    moved = pin(reads=[{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "9" * 32},
                       {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "3" * 32}])
    assert A.judge(pol, facts(pins={"P.t": moved}), FILES)[0] == ["pins: definitions in P.Defs that P.t read changed (changed: P.f)"]
    grown = pin(reads=pin()["reads"] + [{"name": "P.g", "module": "P.More", "kind": "def", "hash": "4" * 32}])
    assert A.judge(pol, facts(pins={"P.t": grown}), FILES)[0] == ["pins: the pinned statements now read definitions in P.More (P.t)"]


def test_generated_constants_belong_to_their_declaration():
    assert [A.parent(n) for n in ("P.f.match_1", "P.walk._f", "P.Enc.mk", "P.instDecidableFits", "mk")] == [
        "P.f", "P.walk", "P.Enc", "P.instDecidableFits", "mk"]


def test_update_prints_what_a_statement_reviewer_reads(tmp_path):
    (tmp_path / "Defs.lean").write_text("namespace P\n\ndef f (n : Nat) : Nat := n\n\nend P\n")
    old = recorded()
    reads = [{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "9" * 32, "lines": [3, 3]},
             {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "3" * 32, "lines": None}]
    now = facts(pins={"P.t": pin(signature="P.t (h : P.Hyp) (n : Nat) : P.f n ≤ n", h="00000000000000bb", reads=reads)})
    fails, new = A.judge(old, now, FILES, update=True)
    assert fails == []
    text = A.review(old, new, now, {"P.Defs": tmp_path / "Defs.lean"})
    assert text[:5] == ["pin P.t: changed", "  before:", "    P.t (h : P.Hyp) (n : Nat) : P.f n = n", "  after:",
                        "    P.t (h : P.Hyp) (n : Nat) : P.f n ≤ n"]
    assert text[5] == "definition P.f (P.Defs), read by P.t: changed" and text[-1] == "    def f (n : Nat) : Nat := n"
    assert A.statement_changes(text) == text
    assert A.review(new, new, now, {}) == []
    reprinted = dict(new["pins"]["P.t"], signature=["P.t (h : P.Hyp) (n : Nat) : LE.le (P.f n) n"])
    text = A.review(new, dict(new, pins={"P.t": reprinted}), now, {})
    assert text == [f"pin P.t: {A.REPRINTED}"] and A.statement_changes(text) == []


def test_a_closed_hypothesis_must_be_a_named_assumption():
    inline = [{"binder": "h", "type": "∀ (m : Nat), m + 0 = m", "closed": True, "named": None, "module": None},
              {"binder": "hn", "type": "n ≠ 0", "closed": False, "named": None, "module": None}]
    fails, _ = A.judge(POLICY, facts(pins={"P.t": pin(hyps=inline)}), FILES, update=True)
    assert fails == ["pins: P.t assumes (h : ∀ (m : Nat), m + 0 = m), a closed Prop that is not a named assumption (a Prop "
                     "constant of P.Assumptions)"]
    elsewhere = [{"binder": "h", "type": "Q.Hyp", "closed": True, "named": "Q.Hyp", "module": "Q.Stuff"}]
    assert A.judge(POLICY, facts(pins={"P.t": pin(hyps=elsewhere)}), FILES, update=True)[0][0].startswith("pins: P.t assumes (h : Q.Hyp)")


def test_a_pin_must_be_a_theorem_with_allowed_axioms():
    assert A.judge(POLICY, facts(pins={"P.t": {"kind": "missing"}}), FILES, update=True)[0] == ["pins: P.t is missing"]
    assert A.judge(POLICY, facts(pins={"P.t": pin(kind="def")}), FILES, update=True)[0] == ["pins: P.t is a def, not a theorem"]
    fails = A.judge(recorded(), facts(pins={"P.t": pin(axioms=STD + ["sorryAx"])}), FILES)[0]
    assert fails == ["pins: P.t uses sorryAx"]


def test_the_source_scan_refuses_kernel_bypasses_but_not_prose(tmp_path):
    (tmp_path / "P").mkdir()
    (tmp_path / "P" / "A.lean").write_text("theorem a : True := trivial\nset_option debug.skipKernelTC true in\n")
    (tmp_path / "P" / "B.lean").write_text("/-- A docstring may name debug.skipKernelTC. /- nested -/ -/\n"
                                           "-- and so may a comment: set_option debug.skipKernelTC true in\n"
                                           "def s := \"-- not a comment\" -- debug.byAsSorry\n")
    (tmp_path / "lakefile.toml").write_text("# debug.skipKernelTC in a comment\n[leanOptions]\ndebug.byAsSorry = true\n")
    hits = A.scans(tmp_path, A.lean_modules(tmp_path), {})
    assert [h.split(" names ")[0] for h in hits] == ["unchecked: P/A.lean:2", "unchecked: lakefile.toml:3"]


def test_code_run_while_compiling_is_refused_unless_listed(tmp_path):
    (tmp_path / "P").mkdir()
    (tmp_path / "P" / "A.lean").write_text("import Lean\n@[simp] theorem t : True := trivial\nopen Lean in\nrun_cmd pure ()\n"
                                           "/-- #eval in prose -/\nlocal macro \"m\" : term => `(1)\n@[elab_as_elim] def e := 1\n")
    (tmp_path / "P" / "B.lean").write_text("initialize pure ()\n")
    hits = A.scans(tmp_path, A.lean_modules(tmp_path), {})
    assert [h.split(" runs code")[0] for h in hits] == ["compile-time: P/A.lean:4", "compile-time: P/A.lean:6",
                                                        "compile-time: P/B.lean:1"]
    assert A.scans(tmp_path, A.lean_modules(tmp_path), {"compile_time": {"P.A": "why", "P.B": "why"}}) == []
    assert len(A.scans(tmp_path, A.lean_modules(tmp_path), {"compile_time": {"P": "a prefix lists no module"}})) == 3
    (tmp_path / "lakefile.lean").write_text("import Lake\n")
    assert A.scans(tmp_path, {}, {})[0].startswith("compile-time: lakefile.lean runs code")


def test_a_compile_time_listing_is_tied_to_its_file_and_kept_out_of_what_pins_read(tmp_path):
    (tmp_path / "P").mkdir()
    (tmp_path / "P" / "Tac.lean").write_text('elab "t" : tactic => pure ()\n')
    (tmp_path / "P" / "Defs.lean").write_text('macro "m" : term => `(1)\ndef f := 1\n')
    files = A.lean_modules(tmp_path)
    pins = {"pins": {"P.t": {"module": "P.Thms", "reads": [{"name": "P.f", "module": "P.Defs"}]}}}
    listed = {"P.Tac": {"why": "why", "digest": A.sha256(tmp_path / "P" / "Tac.lean")}}
    fails, now, notes = A.listings(tmp_path, files, {"compile_time": {"P.Tac": "why"}}, pins)
    assert [f.split(":")[1] for f in fails] == [" the listing of P.Tac records no digest of P/Tac.lean"]
    assert now == listed and notes[0].startswith("compile_time P.Tac (P/Tac.lean): listed; a reviewer reads")
    assert A.listings(tmp_path, files, {"compile_time": {"P.Tac": "why"}}, pins, update=True)[0] == []
    assert A.listings(tmp_path, files, {"compile_time": listed}, pins) == ([], listed, [])
    (tmp_path / "P" / "Tac.lean").write_text('elab "t" : tactic => do IO.println "now it does IO"\n')
    fails, _, notes = A.listings(tmp_path, files, {"compile_time": listed}, pins)
    assert fails[0].startswith("compile-time: the listing of P.Tac was reviewed against another version of P/Tac.lean")
    assert notes[0].startswith("compile_time P.Tac (P/Tac.lean): the file changed since its listing was reviewed")
    read = {"P.Defs": {"why": "why", "digest": A.sha256(tmp_path / "P" / "Defs.lean")}}
    assert A.listings(tmp_path, files, {"compile_time": read}, pins)[0][0].startswith(
        "compile-time: P.Defs is listed under `compile_time`, and pinned statements are declared or read there (P.t)")
    declared = {"pins": {"P.t": {"module": "P.Tac", "reads": []}}}
    tac = {"P.Tac": {"why": "why", "digest": A.sha256(tmp_path / "P" / "Tac.lean")}}
    assert "(P.t)" in A.listings(tmp_path, files, {"compile_time": tac}, declared)[0][0]
    assert A.listings(tmp_path, files, {"compile_time": {"P": "a prefix"}}, pins)[0] == [
        "compile-time: `compile_time` lists P, which is not a module of the package; list modules one by one"]


def test_a_package_without_a_policy_gets_a_template(tmp_path):
    pkg = tmp_path / "p"
    pkg.mkdir()
    (pkg / "lakefile.toml").write_text('name = "p"\n[[lean_lib]]\nname = "Alpha"\n[[lean_lib]]\nname = "Beta"\n')
    assert A.packages(tmp_path) == [pkg]
    rep = A.audit(pkg, tmp_path / "out")
    assert rep["failures"][0].startswith(f"policy: {pkg} has no lean-audit.json")
    assert rep["failures"][0].endswith('{"roots": ["Alpha", "Beta"], "pins": {}}')


def test_dependency_digests_cover_every_part_of_the_imported_modules_only(tmp_path):
    lib = tmp_path / "p" / ".lake" / "packages" / "dep" / ".lake" / "build" / "lib" / "lean"
    (lib / "D").mkdir(parents=True)
    for rel, text in (("D.olean", "a"), ("D/E.olean", "b"), ("D/E.olean.private", "c"), ("D/E.ilean", "d")):
        (lib / rel).write_text(text)
    loaded = [str(lib / "D.olean"), str(lib / "D" / "E.olean"), str(tmp_path / "p" / ".lake" / "build" / "lib" / "P.olean")]
    digests = lambda: A.dependency_digests(tmp_path / "p", "none/none:none", loaded)
    before = digests()
    assert list(before) == ["dep"]
    (lib / "D" / "E.ilean").write_text("changed")
    (lib / "D" / "F.olean").write_text("a module the package does not import, built or not")
    assert digests() == before
    (lib / "D" / "E.olean.private").write_text("changed")
    assert digests()["dep"] != before["dep"]


def test_package_files_stop_at_nested_packages_and_lake(tmp_path):
    for rel in ("A.lean", "A/B.lean", ".lake/packages/x/X.lean", "sub/lakefile.toml", "sub/S.lean", "lakefile.lean"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("")
    assert A.lean_modules(tmp_path) == {"A": "A.lean", "A.B": "A/B.lean"}


def test_the_policy_layout_is_json():
    pol = recorded()
    assert json.loads(A.layout(pol)) == pol


def test_discovery_skips_the_controls(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "lean-audit.json").write_text("{}")
    assert A.packages(tmp_path) == [tmp_path / "a"]
    assert A.FIXTURES not in A.packages(A.HERE.parent)


def test_a_file_changed_during_the_run_is_reported(tmp_path):
    tool, late = tmp_path / "Facts.lean", tmp_path / "lean"
    tool.write_text("a")
    seals = {str(tool): A.sha256(tool)}
    assert A.tampered(seals, [late]) == []
    late.write_text("x")
    assert A.tampered(seals, [late]) == [] and str(late) in seals
    tool.write_text("b")
    late.write_text("y")
    assert A.tampered(seals, [late]) == [str(tool), str(late)]
    assert A.killed(-9) and A.killed(137) and not A.killed(1)


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_controls(tmp_path):
    toolchain = (A.FIXTURES / "control" / "lean-toolchain").read_text().strip()
    assert A.controls(toolchain, tmp_path, log=lambda *_: None) == []
