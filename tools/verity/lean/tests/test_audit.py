"""`tools/verity/lean/check.py`: its judgement on the facts `Facts.lean` reports (no Lean needed), and the controls when `lake` is
installed (the fixture package passes clean and fails each check once a violation is added)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_audit", HERE.parent / "check.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

STD = ["Classical.choice", "Quot.sound", "propext"]


def sha(s: str) -> str:
    """A stand-in SHA-256 content hash, tied to the stand-in 32-bit one."""
    return hashlib.sha256(s.encode()).hexdigest()


#: the functions whose failures make up a report, and where in what each returns its failures are (None: all of it)
EMITTERS = {"judge": 0, "unreported": None, "dropped": None, "verifier_failures": None, "claim_failures": None,
            "witness_failures": None, "inline_failures": 0, "assumption_records": 1, "shape": None, "spec_reads": None, "tree_failures": None, "scans": None, "listings": 0, "dependency_scans": 0,
            "assign": 1, "proved": 0, "replay_step": 0, "tests": 0, "run_paths": None}


def coded(failures) -> list:
    """``failures``, each asserted a `Failure` whose code names its check as its text's leading word does."""
    for f in failures:
        assert isinstance(f, A.Failure) and f.code, f"a failure without a code: {f!r}"
        assert f.code.split(".")[0] == f.split(":", 1)[0].replace(" ", "-"), (f.code, str(f))
    return failures


@pytest.fixture(autouse=True)
def every_failure_has_a_code(monkeypatch):
    """Every failure any test here provokes, from each function a report's failures come from and from `audit` itself, has a
    code (`Failure`), and a report lists the codes beside its failures (`failure_codes`)."""
    def checked(name, f, at):
        def g(*args, **kw):
            got = f(*args, **kw)
            coded(got if at is None else got[at])
            return got
        return g

    for name, at in EMITTERS.items():
        monkeypatch.setattr(A, name, checked(name, getattr(A, name), at))
    audit = A.audit

    def audited(*args, **kw):
        rep = audit(*args, **kw)
        assert rep["failure_codes"] == [f.code for f in coded(rep["failures"])]
        return rep

    monkeypatch.setattr(A, "audit", audited)


def unshaped(rep: dict) -> list[str]:
    """A report's failures other than the shape rule's, for a test of something else that adds guarantees to the control
    without entering them under `shape.properties`."""
    return [f for f in rep["failures"] if not f.startswith("shape:")]


def fact(signature="P.t (h : P.Hyp) (n : Nat) : P.f n = n", hyps=None, reads=None, axioms=STD, kind="theorem", h="00000000000000aa"):
    reads = reads if reads is not None else [{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "1" * 32, "own": ["P.f.match_1"]},
                                             {"name": "P.f.match_1", "module": "P.Defs", "kind": "def", "hash": "2" * 32},
                                             {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "3" * 32}]
    return {"kind": kind, "module": "P.Thms", "signature": signature, "axioms": axioms, "type_hash": h, "type_sha256": sha(h),
            "hypotheses": hyps if hyps is not None else [{"binder": "h", "type": "P.Hyp", "closed": True, "named": "P.Hyp",
                                                          "module": "P.Assumptions"}],
            "reads": [r | {"sha256": sha(r["hash"])} for r in reads]}


def facts(**over):
    f = {"lean": "4.34.0", "modules": ["P", "P.Assumptions", "P.Defs", "P.Thms"],
         "imports": {"P": ["Init", "P.Thms"], "P.Assumptions": ["Init"], "P.Defs": ["Init"], "P.Thms": ["P.Assumptions", "P.Defs"]},
         "declarations": {"theorem": 1, "def": 3}, "axioms": STD, "offenders": [], "declared_axioms": [], "escapes": [],
         "pins": {"P.t": fact()}}
    return f | over


FILES = {"P": "P.lean", "P.Assumptions": "P/Assumptions.lean", "P.Defs": "P/Defs.lean", "P.Thms": "P/Thms.lean"}
POLICY = {"roots": ["P"], "assumptions": {"P.Hyp": {"owner": "@lean"}}, "guarantees": {"P.t": {"owner": "@lean"}}}


def recorded(f=None):
    fails, pol = A.judge(POLICY, f or facts(), FILES, update=True)
    assert fails == []
    return pol


def test_four_names_holds_the_guarantees_under_the_roots_it_lists_to_the_four_name_form():
    def lifted(name, g, module="P.Guarantees", binders="", args=""):
        return fact(signature=f"{name}{binders} : {g}{args}", hyps=[], reads=[]) | {"lift": {"head": g, "module": module}}
    pins = {"P.t": fact(),
            "P.SecurityProofs.G": lifted("P.SecurityProofs.G", "P.Guarantees.G"),
            "P.SecurityProofs.R": lifted("P.SecurityProofs.R.{u}", "P.Guarantees.R", args=".{u}"),
            "P.SecurityProofs.W": lifted("P.SecurityProofs.W", "P.Guarantees.W", module="P.Spec"),
            "P.SecurityProofs.V": lifted("P.SecurityProofs.V", "P.Guarantees.V", binders=" (n : Nat)", args=" n"),
            "P.SecurityProofs.X": lifted("P.SecurityProofs.X", "P.Guarantees.Y"),
            "Q.u": fact(signature="Q.u : True", hyps=[], reads=[])}
    policy = POLICY | {"four_names": ["P"], "guarantees": {n: {"owner": "@lean"} for n in pins}}
    fails, _ = A.judge(policy, facts(pins=pins), FILES, update=True)
    assert [f.split(" is a guarantee")[0] for f in fails] == [
        f"four names: P.{n}" for n in ("SecurityProofs.V", "SecurityProofs.W", "SecurityProofs.X", "t")]
    assert A.judge(POLICY | {"guarantees": policy["guarantees"]}, facts(pins=pins), FILES, update=True)[0] == []


def test_a_recorded_package_passes_and_its_record_is_the_statement():
    pol = recorded()
    assert pol["guarantees"]["P.t"] == {"owner": "@lean", "signature": ["P.t (h : P.Hyp) (n : Nat) : P.f n = n"],
                                        "assumptions": ["P.Hyp"], "type_hash": sha("00000000000000aa")}
    assert len(pol["reads"]["P.Defs"]["digest"]) == 64 and len(pol["reads"]["P.Defs"]["definitions"]["P.f"]) == 64
    assert list(pol["reads"]["P.Defs"]["definitions"]) == ["P.f"] and pol["reads"]["P.Defs"]["guarantees"] == ["P.t"]
    assert all(g["digest"] == A.module_digest(g["definitions"]) for g in pol["reads"].values())
    assert A.judge(pol, facts(), FILES) == ([], pol)
    assert A.judge(POLICY, facts(), FILES)[0] == ["guarantees: P.t has no record; run `check.py --update` and commit the record",
                                                  "assumptions: P.Hyp has no record; run `check.py --update` and commit the record"]


def test_every_field_of_a_reads_record_is_checked():
    pol = recorded()
    group = pol["reads"]["P.Defs"]
    stale = pol | {"reads": pol["reads"] | {"P.Defs": group | {"digest": "0" * 64}}}
    assert A.judge(stale, facts(), FILES)[0] == ["guarantees: the digest recorded for P.Defs is not the SHA-256 of its definitions; "
                                                 "`python tools/verity/lean/merge.py --normalize` recomputes it"]
    readers = pol | {"reads": pol["reads"] | {"P.Defs": group | {"guarantees": ["P.t", "P.u"]}}}
    assert A.judge(readers, facts(), FILES)[0] == ["guarantees: those that read P.Defs are P.t; the record lists "
                                                   "P.t, P.u; run `check.py --update`"]


def test_every_guarantee_has_an_owner_and_an_update_keeps_it():
    pol = recorded()
    bare = {**POLICY, "guarantees": {"P.t": {}}}
    no_owner = ('guarantees: P.t has no owner: write "owner": "@lane" in its entry, the handle of the lane that answers for '
                'it, or run `check.py --update --owner @lane`')
    fails, written = A.judge(bare, facts(), FILES, update=True)
    assert fails == [no_owner] and written["guarantees"] == {"P.t": {}}, "no record is written for a guarantee without an owner"
    assert A.judge(bare, facts(), FILES, update=True, owner="@proofs")[1]["guarantees"]["P.t"]["owner"] == "@proofs"
    assert A.judge(pol, facts(), FILES, update=True, owner="@proofs")[1] == pol, "an entry's owner is kept"
    assert A.judge({**pol, "guarantees": {"P.t": A.unowned(pol["guarantees"]["P.t"])}}, facts(), FILES)[0] == [no_owner]
    odd = {**pol, "guarantees": {"P.t": pol["guarantees"]["P.t"] | {"owner": "proofs"}}}
    assert A.judge(odd, facts(), FILES)[0] == ["guarantees: P.t's owner 'proofs' is not a handle such as @proofs"]
    handed = {**pol, "guarantees": {"P.t": pol["guarantees"]["P.t"] | {"owner": "@compute-accounting"}}}
    text = A.review(pol, handed, facts(), {})
    assert text == [f"guarantee P.t: owner @lean -> @compute-accounting: {A.OWNED}"] and A.statement_changes(text) == []


def spec_facts(reads=None, outside=(), spec=()):
    """``facts()`` whose guarantee reads ``reads`` under `meaning` and stops at ``outside`` ([name, module] pairs) outside it,
    in a package whose spec modules declare ``spec`` ({name, module, kind})."""
    f = facts(pins={"P.t": fact(reads=reads) | {"outside": [list(o) for o in outside]},
                    "P.lemma": fact(kind="def")}, spec_definitions=list(spec))
    return f


def test_a_guarantee_reads_only_spec_definitions_and_pinned_dependencies():
    """The reads check: `P.Defs` and `P.Assumptions` are the spec, `P.Proofs` is not, `Q.Spec` is the spec of a package `P`
    requires by path and `Q.Math` is not, and `Nat` is a pinned dependency's."""
    pol = recorded() | {"layers": {"P.Defs": {"imports": []}, "P.Assumptions": {"imports": []}},
                        "meaning": ["P.Defs", "P.Assumptions", "Q.Spec"]}
    home = dict.fromkeys(["P", "P.Defs", "P.Assumptions", "P.Proofs", "P.Thms"], A.spec_of(pol)) | \
        dict.fromkeys(["Q.Spec", "Q.Math"], ["Q.Spec"])
    assert A.spec_reads(pol, spec_facts(outside=[("Nat", "Init.Prelude"), ("Q.g", "Q.Spec")]), home) == []
    extra = [{"name": "P.aux", "module": "P.Proofs", "kind": "def", "hash": "5" * 32, "own": ["P.aux.match_1"]},
             {"name": "P.aux.match_1", "module": "P.Proofs", "kind": "def", "hash": "6" * 32}]
    assert A.spec_reads(pol, spec_facts(reads=fact()["reads"] + extra), home) == [
        "reads: P.t read P.aux in P.Proofs, outside the spec (P.Assumptions, P.Defs): a guarantee's statement reads only the "
        "spec of its package and of the packages it requires by path, and pinned dependencies; move the definitions into the "
        "spec, or list P.Proofs under `reads_exempt` with the reason"]
    assert A.spec_reads(pol, spec_facts(outside=[("Q.h", "Q.Math")]), home)[0].startswith(
        "reads: P.t read Q.h in Q.Math, outside the spec (Q.Spec)")
    unlocked = pol | {"meaning": ["P.Assumptions"]}
    assert A.spec_reads(unlocked, spec_facts(outside=[("P.f", "P.Defs")]), home) == [
        "reads: P.t read P.f in P.Defs, a spec module outside `meaning`, so the record holds nothing of them: add P.Defs to "
        "`meaning` and run `check.py --update`"]
    exempt = pol | {"reads_exempt": {"P.Proofs": "until its spec is extracted", "P.Old": "stale"}}
    assert A.spec_reads(exempt, spec_facts(reads=fact()["reads"] + extra), home) == [
        "reads: `reads_exempt` lists P.Old, under which no guarantee reads outside the spec; remove the entry"]
    assert A.spec_reads(recorded(), spec_facts(), dict.fromkeys(FILES, []))[0].startswith(
        "reads: P.t read P.Hyp in P.Assumptions, outside the spec (none: no `layers`)"), "a package without layers has no spec"


def test_a_reads_exemption_names_a_prefix_and_a_reason_and_covers_only_reads_outside_the_spec():
    """Found by red-team-1150: the list form passed with no reason, `null` raised, and an exemption over a spec module
    hid the `meaning` check while counting as used."""
    pol = recorded() | {"layers": {"P.Defs": {"imports": []}, "P.Assumptions": {"imports": []}},
                        "meaning": ["P.Assumptions"]}
    home = dict.fromkeys(["P", "P.Defs", "P.Assumptions", "P.Proofs", "P.Thms"], A.spec_of(pol))
    for bad, said in ((["P.Proofs"], 'is ["P.Proofs"]'), (None, "is null"), ("P.Proofs", 'is "P.Proofs"'),
                      ({"": "why"}, 'lists "" without'), ({"P.Proofs": " "}, 'lists "P.Proofs" without'),
                      ({"P.Proofs": None}, 'lists "P.Proofs" without')):
        assert A.spec_reads(pol | {"reads_exempt": bad}, spec_facts(), home)[0].startswith(f"reads: `reads_exempt` {said}")
    spec_only = pol | {"reads_exempt": {"P.Defs": "hides a spec module outside meaning"}}
    assert A.spec_reads(spec_only, spec_facts(outside=[("P.f", "P.Defs")]), home) == [
        "reads: P.t read P.f in P.Defs, a spec module outside `meaning`, so the record holds nothing of them: add P.Defs to "
        "`meaning` and run `check.py --update`",
        "reads: `reads_exempt` lists P.Defs, under which no guarantee reads outside the spec; remove the entry"]
    whole = pol | {"meaning": ["P"], "reads_exempt": {"P": "covers spec modules only"}}
    assert A.spec_reads(whole, spec_facts(outside=[("P.f", "P.Defs")]), home) == [
        "reads: `reads_exempt` lists P, under which no guarantee reads outside the spec; remove the entry"]


def test_spec_definitions_no_guarantee_reads_are_reported():
    spec = [{"name": n, "module": m, "kind": "def"} for n, m in (
        ("P.f", "P.Defs"), ("P.f.match_1", "P.Defs"), ("P.unused", "P.Defs"), ("P.Hyp", "P.Assumptions"), ("P.Old", "P.Assumptions"),
        ("P.grid", "P.Lifting"), ("P.elsewhere", "P.Defs"))]
    f = spec_facts(outside=[("P.grid", "P.Lifting")], spec=spec)
    assert A.unread_spec(f, {"P.elsewhere"}) == {"P.Assumptions": ["P.Old"], "P.Defs": ["P.unused"]}


def test_homes_give_each_module_its_package_s_spec(tmp_path):
    core, pkg = tmp_path / "core", tmp_path / "pkg"
    for d, files in ((core, ("Q/Spec.lean", "Q/Math.lean")), (pkg, ("P/Defs.lean", "P/Thms.lean"))):
        for rel in files:
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text("")
    (core / A.POLICY).write_text(json.dumps({"roots": ["Q"], "layers": {"Q.Spec": {"imports": []}}}))
    (pkg / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "path", "name": "core", "dir": "../core"}]}))
    pol = {"roots": ["P"], "layers": {"P.Defs": {"imports": []}}}
    assert A.homes(pkg, pol) == {"P.Defs": ["P.Defs"], "P.Thms": ["P.Defs"], "Q.Spec": ["Q.Spec"], "Q.Math": ["Q.Spec"]}
    (pkg / A.POLICY).write_text(json.dumps(pol | {"guarantees": {}, "reads": {"Q.Spec": {"definitions": {"Q.g": "1" * 64}}}}))
    assert A.read_by_dependents(core, tmp_path) == {"Q.g"} and A.read_by_dependents(pkg, tmp_path) == set()
    assert A.read_by_dependents(core, tmp_path / "pkg") == set() == A.read_by_dependents(A.FIXTURES / "pass", A.ROOT)


def test_a_guarantee_reading_its_prover_s_path_dependency_outside_its_spec_fails(tmp_path):
    """`s` names `p` under `proved_in`; `p` requires `s` and `v` (no `layers`, so no spec) by path. A guarantee of `s` that
    reads `V.Bytes` fails the reads check unless `reads_exempt` covers it, and `p`'s own modules are no spec of `s`: the
    check does not take `v`'s modules for a pinned dependency's (red-team B3, the verifier's `Flock.*` under Security)."""
    s, p, v = tmp_path / "s", tmp_path / "p", tmp_path / "v"
    for d, rel in ((s, "S/Defs.lean"), (p, "SP/Main.lean"), (v, "V/Bytes.lean")):
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text("")
    (v / A.POLICY).write_text(json.dumps({"roots": ["V"]}))
    (p / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "path", "name": n, "dir": f"../{n}"}
                                                                  for n in ("s", "v")]}))
    pol = {"roots": ["S"], "proved_in": ["../p"], "layers": {"S.Defs": {"imports": []}}, "meaning": ["S", "V"]}
    home = A.reads_home(s, pol)
    assert home == {"S.Defs": ["S.Defs"], "SP.Main": [], "V.Bytes": []}
    reads = lambda m: {"pins": {"S.t": {"kind": "theorem", "outside": [], "reads": [{"name": f"{m}.x", "module": m}]}}}
    assert A.spec_reads(pol, reads("V.Bytes"), home)[0].startswith(
        "reads: S.t read V.Bytes.x in V.Bytes, outside the spec (none: no `layers`)")
    assert A.spec_reads(pol | {"reads_exempt": {"V": "until its spec is extracted"}}, reads("V.Bytes"), home) == []
    assert A.spec_reads(pol, reads("SP.Main"), home)[0].startswith("reads: S.t read SP.Main.x in SP.Main, outside the spec")
    assert A.spec_reads(pol, reads("V.Bytes"), A.homes(s, pol)) == [], "without the prover's path dependencies it passes"


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


def test_a_sorry_for_an_unreported_error_is_named_apart_from_a_written_one():
    """A build that passes on a `sorryAx _ true` (the `unreported` control's local-notation binder) never ran its proofs; a
    spec package's fast path audits no proving package, so `proved` names those of each one it reads."""
    off = [{"name": "P.stub", "module": "P.Thms", "axioms": ["sorryAx"]}, {"name": "P.unrun", "module": "P.Thms", "axioms": ["sorryAx"]}]
    fails = A.judge(recorded(), facts(offenders=off, unreported_errors=[{"name": "P.unrun", "module": "P.Thms"}]), FILES)[0]
    assert "sorry: sorryAx is used by P.stub, P.unrun" in fails
    (u,) = [f for f in fails if f.startswith("unreported:")]
    assert u.startswith("unreported: P.unrun hold a `sorry` Lean made for an elaboration error it did not report")
    assert A.unreported(facts(), "../Proofs's ") == []
    assert A.unreported(facts(unreported_errors=[{"name": "Q.x", "module": "Q"}]), "../Proofs's ")[0].startswith(
        "unreported: ../Proofs's Q.x hold")


def test_a_declaration_the_olean_does_not_hold_is_dropped(tmp_path):
    """Lean can drop a declaration with no error (PartRead at a1b479d26: the theorems after a `variable` using a local
    notation are in no `.olean`), so no axiom shows it: the source's declaration lines are matched to the module's spans.
    Keywords in comments, docstrings and strings, indented ones and `example` start none."""
    (tmp_path / "P").mkdir()
    (tmp_path / "P" / "Thms.lean").write_text(
        "/-- A doc.\ntheorem in_doc : True := trivial -/\n@[simp] theorem kept : True :=\n  trivial\n"
        "-- def in_comment\nexample : True := trivial\nprivate noncomputable def gone := 1\n"
        "structure S where\n  def_like : Nat\ninstance : Inhabited S := ⟨⟨0⟩⟩\n")
    files = {"P.Thms": "P/Thms.lean"}
    held = {"P.Thms": [[1, 4], [8, 9]]}
    (d,) = A.dropped(tmp_path, files, {"spans": held}, "../Proofs's ")
    assert d.startswith("dropped: ../Proofs's P/Thms.lean:7 (def), P/Thms.lean:10 (instance) start a declaration")
    assert A.dropped(tmp_path, files, {"spans": held | {"P.Thms": [[1, 4], [7, 7], [8, 10]]}}) == []
    assert A.dropped(tmp_path, files, {}) == []


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


def test_a_sealed_layer_answers_only_to_its_own_rule_and_is_reached_only_by_name():
    """A sealed layer under a vocabulary prefix (an area's `Guarantees` under the area): the vocabulary's rule doesn't hold
    its modules, and the vocabulary's own prefix doesn't admit them, so the vocabulary imports them only by listing them."""
    thms = {"imports": ["P.Assumptions", "P.Defs"]}
    pol = recorded() | {"layers": {"P": {"imports": ["P"]}, "P.Thms": thms}}
    assert A.judge(pol, facts(), FILES)[0] == []
    pol["layers"]["P.Thms"] = thms | {"sealed": True}
    assert A.judge(pol, facts(), FILES)[0] == ["layers: P imports P.Thms, outside P's layer (P)"]
    pol["layers"]["P"] = {"imports": ["P", "P.Thms"]}
    assert A.judge(pol, facts(), FILES)[0] == []
    pol["layers"]["P.Thms"] = {"imports": ["P.Defs"], "sealed": True}
    assert A.judge(pol, facts(), FILES)[0] == ["layers: P.Thms imports P.Assumptions, outside P.Thms's layer (P.Defs)"]


TREES = {"P.Defs": "definitions", "P.Assumptions": "definitions", "P.Props": "properties", "P.Thms": "proofs",
         "P.Props.Proofs": "proofs"}
AUX = [{"name": "P.aux", "module": "P.Thms", "kind": "def", "hash": "5" * 32, "own": ["P.aux.match_1"]},
       {"name": "P.aux.match_1", "module": "P.Thms", "kind": "def", "hash": "6" * 32}]


def test_a_module_s_tree_is_its_longest_prefix_s_and_reaches_the_packages_requiring_it(tmp_path):
    assert [A.tree_of(TREES, m) for m in ("P.Props.Proofs.X", "P.Props.X", "P.Thms", "P.Thmsx", "Q")] == [
        "proofs", "properties", "proofs", None, None]
    spec, props = tmp_path / "spec", tmp_path / "props"
    for d in (spec, props):
        d.mkdir()
    (spec / A.POLICY).write_text(json.dumps({"roots": ["P"], "trees": {"definitions": ["P.Defs"], "proofs": ["P.Thms"],
                                                                       "except": {"P.Thms.Old": "the dependency's own"}}}))
    (props / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "path", "name": "spec", "dir": "../spec"}]}))
    assert A.tree_roles(props, {"trees": {"properties": ["P.Props"], "proofs": ["P.Props.Proofs"]}}) == {
        "P.Defs": "definitions", "P.Thms": "proofs", "P.Props": "properties", "P.Props.Proofs": "proofs"}
    assert A.tree_roles(props, {}) == {"P.Defs": "definitions", "P.Thms": "proofs"}
    assert A.tree_failures({}, facts(), A.tree_roles(props, {})) == [], "a dependency's `except` is its own"


def test_a_statement_reads_nothing_of_the_proofs_tree_and_the_trusted_trees_import_none_of_it():
    """Daniel's three trees (24 Oct): a guarantee's or property's statement reads only the definitions tree and the code, and
    no module of the definitions or properties tree imports the proofs tree; each refusal says what to move where."""
    assert A.tree_failures({}, spec_facts(), TREES) == []
    read = A.tree_failures({}, spec_facts(reads=fact()["reads"] + AUX), TREES)
    assert read == [
        "trees: P.t read P.aux in P.Thms, a module of the proofs tree (P.Props.Proofs, P.Thms): a guarantee's or property's "
        "statement reads only the definitions tree (P.Assumptions, P.Defs), the code it is about and pinned dependencies. "
        "Move the definitions into the definitions tree (Definitions/<Area>/), or take the statement off the lock's "
        "guarantees: one over proof-only definitions is a lemma of the proofs tree"]
    assert read[0].code == "trees.reads" and read[0].entry == "P.Thms"
    assert A.tree_failures({}, spec_facts(outside=[("P.lemmaDef", "P.Props.Proofs.X")]), TREES)[0].startswith(
        "trees: P.t read P.lemmaDef in P.Props.Proofs.X, a module of the proofs tree"), "where the walk stops too"
    imports = facts()["imports"] | {"P.Defs": ["Init", "P.Thms"], "P.Props": ["P.Defs", "P.Props.Proofs.Y"],
                                    "P.Props.Proofs.Y": ["P.Thms"]}
    modules = facts()["modules"] + ["P.Props", "P.Props.Proofs.Y"]
    assert A.tree_failures({}, facts(imports=imports, modules=modules), TREES) == [
        "trees: P.Defs, of the definitions tree, imports P.Thms, of the proofs tree (P.Props.Proofs, P.Thms): what the "
        "definitions and the properties say never rests on a proof. Move what P.Defs uses from P.Thms into the definitions "
        "tree (Definitions/<Area>/), or move P.Defs into the proofs tree",
        "trees: P.Props, of the properties tree, imports P.Props.Proofs.Y, of the proofs tree (P.Props.Proofs, P.Thms): what "
        "the definitions and the properties say never rests on a proof. Move what P.Props uses from P.Props.Proofs.Y into "
        "the definitions tree (Definitions/<Area>/), or move P.Props into the proofs tree"]
    assert A.tree_failures({}, spec_facts(reads=fact()["reads"] + AUX), {}) == [], "a package without trees"


def test_an_exception_to_the_trees_names_a_prefix_and_a_reason_and_excepts_something():
    both = facts(pins={"P.t": fact(reads=fact()["reads"] + AUX)}, imports=facts()["imports"] | {"P.Defs": ["P.Thms"]})
    why = "C-Flock is excepted until proofs extracts its spec"
    assert A.tree_failures({"trees": {"except": {"P.Thms": why, "P.Defs": why}}}, both, TREES) == []
    assert A.tree_failures({"trees": {"except": {"P": why}}}, both, TREES) == [], "a prefix covers its modules"
    assert A.tree_failures({"trees": {"except": {"P.Thms": why, "P.Defs": why, "P.Old": why}}}, both, TREES) == [
        "trees: `except` lists P.Old, under which no statement reads the proofs tree and no module of the definitions or "
        "properties tree imports it; remove the entry"]
    assert [f.code for f in A.tree_failures({"trees": {"except": {"P.Defs": why}}}, both, TREES)] == ["trees.reads"]
    for bad, code in (({"proofs": "P.Thms"}, "trees.form"), ({"proof": ["P.Thms"]}, "trees.form"), ([], "trees.form"),
                      ({"except": {"P.Thms": " "}}, "trees.except-form"), ({"except": ["P.Thms"]}, "trees.except-form"),
                      ({"except": {"": why}}, "trees.except-form")):
        assert [f.code for f in A.tree_failures({"trees": bad}, both, TREES)] == [code], bad


def test_the_reads_check_leaves_the_proofs_tree_to_the_trees_check():
    """A read of the proofs tree is the trees check's: `reads_exempt` covers none, so an exception has one place."""
    pol = recorded() | {"layers": {"P.Defs": {"imports": []}, "P.Assumptions": {"imports": []}},
                        "meaning": ["P.Defs", "P.Assumptions", "P.Thms"]}
    home = dict.fromkeys(["P", "P.Defs", "P.Assumptions", "P.Thms"], A.spec_of(pol))
    f = spec_facts(reads=fact()["reads"] + AUX)
    assert A.spec_reads(pol, f, home)[0].startswith("reads: P.t read P.aux in P.Thms, outside the spec")
    assert A.spec_reads(pol, f, home, TREES) == []
    assert A.spec_reads(pol | {"reads_exempt": {"P.Thms": "why"}}, f, home, TREES) == [
        "reads: `reads_exempt` lists P.Thms, in the proofs tree, whose reads the trees check judges: remove the entry (an "
        "exception to the trees rule is listed under `trees`' `except`)"]


def test_a_changed_statement_or_meaning_fails_until_updated():
    pol = recorded()
    fails = A.judge(pol, facts(pins={"P.t": fact(signature="P.t (h : P.Hyp) (n : Nat) : P.f n ≤ n", h="00000000000000bb")}), FILES)[0]
    assert len(fails) == 1 and fails[0].startswith("guarantees: P.t's signature changed:") and "+P.t (h : P.Hyp) (n : Nat) : P.f n ≤ n" in fails[0]
    moved = fact(reads=[{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "9" * 32},
                       {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "3" * 32}])
    assert A.judge(pol, facts(pins={"P.t": moved}), FILES)[0] == ["guarantees: definitions in P.Defs that P.t read changed (changed: P.f)"]
    grown = fact(reads=fact()["reads"] + [{"name": "P.g", "module": "P.More", "kind": "def", "hash": "4" * 32}])
    assert A.judge(pol, facts(pins={"P.t": grown}), FILES)[0] == ["guarantees: they now read definitions in P.More (P.t)"]


def test_only_record_differences_are_regenerated_and_the_regenerated_record_passes():
    pol = recorded()
    group = pol["reads"]["P.Defs"]
    hyp = [{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "1" * 32},
           {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "8" * 32}]
    applied = [{"binder": "h", "type": "P.Premises.Hyp n", "closed": False, "named": None, "module": None}]
    premised = fact(signature="P.t (n : Nat) (h : P.Premises.Hyp n) : P.f n = n", hyps=applied)
    premised["premises"] = [{"binder": "h", "head": "P.Hyp", "module": "P.Assumptions"}]
    for policy, now in [
            (POLICY, facts()),                                                      # owned, no record yet
            ({**A.judge(POLICY, facts(pins={"P.t": premised}), FILES, update=True)[1], "assumptions": {}},
             facts(pins={"P.t": premised})),                                        # a premise the lock doesn't list
            (pol, facts(pins={"P.t": fact(signature="P.t (h : P.Hyp) (n : Nat) : P.f n ≤ n", h="00000000000000bb")})),
            (pol, facts(pins={"P.t": fact(reads=fact()["reads"] + [{"name": "P.g", "module": "P.More", "kind": "def",
                                                                    "hash": "4" * 32}])})),
            (pol, facts(pins={"P.t": fact(reads=hyp)})),
            (pol | {"reads": pol["reads"] | {"P.Defs": group | {"digest": "0" * 64, "guarantees": ["P.t", "P.u"]}}}, facts())]:
        fails = A.judge(policy, now, FILES)[0]
        assert fails and all(A.RECORD_DIFFERENCE.match(f) for f in fails), fails
        regen = A.regenerated(policy, now, FILES, fails)
        assert regen == A.judge(policy, now, FILES, update=True)[1] and A.judge(regen, now, FILES)[0] == []
    assert A.judge(pol, facts(pins={"P.t": fact(reads=hyp)}), FILES)[0][-1].startswith(
        "assumptions: P.Hyp's definition in P.Assumptions changed;")
    signed = facts(pins={"P.t": fact(signature="P.t (h : P.Hyp) (n : Nat) : P.f n ≤ n", h="00000000000000bb")})
    for policy, now, files in [(pol, facts(), FILES),                                      # passes
                               ({**POLICY, "guarantees": {"P.t": {}}}, facts(), FILES),    # no record and no owner
                               (pol, facts(pins={}), FILES),                               # renamed or gone: `--moved`
                               (pol | {"guarantees": {"P.t": pol["guarantees"]["P.t"] | {"owner": "lean"}}}, facts(), FILES),
                               (pol, signed, FILES | {"Scratch": "Scratch.lean"})]:        # a difference beside an orphan
        assert A.regenerated(policy, now, files, A.judge(policy, now, files)[0]) is None


def test_every_record_difference_code_is_one_update_settles():
    """Each code of `RECORD_DIFFERENCE` comes from a record that differs from the build, and `regenerated` writes the record
    that then passes: what a train commits is the code set, so a code in it that no case provokes fails here."""
    pol = recorded()
    group = pol["reads"]["P.Defs"]
    entry = pol["guarantees"]["P.t"]
    premised = fact(signature="P.t (n : Nat) (h : P.Premises.Hyp n) : P.f n = n",
                    hyps=[{"binder": "h", "type": "P.Premises.Hyp n", "closed": False, "named": None, "module": None}])
    premised["premises"] = [{"binder": "h", "head": "P.Hyp", "module": "P.Assumptions"}]
    hyp = [{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "1" * 32, "own": ["P.f.match_1"]},
           {"name": "P.f.match_1", "module": "P.Defs", "kind": "def", "hash": "2" * 32},
           {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "8" * 32}]
    cases = {
        "guarantees.unrecorded": (POLICY, facts()),
        "assumptions.unrecorded": (pol | {"assumptions": {"P.Hyp": {"owner": "@lean"}}}, facts()),
        "assumptions.unlisted-premise": ({**A.judge(POLICY, facts(pins={"P.t": premised}), FILES, update=True)[1],
                                          "assumptions": {}}, facts(pins={"P.t": premised})),
        "guarantees.signature": (pol, facts(pins={"P.t": fact(signature="P.t (h : P.Hyp) (n : Nat) : P.f n ≤ n")})),
        "guarantees.assumptions": (pol | {"guarantees": {"P.t": entry | {"assumptions": []}}}, facts()),
        "guarantees.type-hash": (pol | {"guarantees": {"P.t": entry | {"type_hash": sha("00000000000000bb")}}}, facts()),
        "guarantees.reads-new": (pol | {"reads": {m: g for m, g in pol["reads"].items() if m != "P.Defs"}}, facts()),
        "guarantees.reads-gone": (pol | {"reads": pol["reads"] | {"P.Old": group}}, facts()),
        "guarantees.reads-changed": (pol, facts(pins={"P.t": fact(reads=[{**hyp[0], "hash": "9" * 32}, *hyp[1:2],
                                                                          fact()["reads"][2]])})),
        "guarantees.readers": (pol | {"reads": pol["reads"] | {"P.Defs": group | {"guarantees": ["P.t", "P.u"]}}}, facts()),
        "guarantees.reads-digest": (pol | {"reads": pol["reads"] | {"P.Defs": group | {"digest": "0" * 64}}}, facts()),
        "assumptions.digest": (pol, facts(pins={"P.t": fact(reads=hyp)})),
        "assumptions.unread": (pol | {"assumptions": pol["assumptions"] | {"P.Other": {"owner": "@lean", "digest": "0" * 64}}},
                               facts()),
    }
    assert set(cases) == A.RECORD_DIFFERENCE
    for code, (policy, now) in cases.items():
        fails = A.judge(policy, now, FILES)[0]
        assert code in {f.code for f in fails} and all(map(A.RECORD_DIFFERENCE.match, fails)), (code, fails)
        regen = A.regenerated(policy, now, FILES, fails)
        assert regen is not None and A.judge(regen, now, FILES)[0] == [], code


def test_what_settles_is_keyed_on_codes_not_text():
    """A record difference reworded still settles, and a string that reads like one but has no record-difference code (a
    bare string, or another check's failure) settles nothing: rewording a message changes nothing a train commits."""
    now = facts()
    (unrecorded, *_) = A.judge(POLICY, now, FILES)[0]
    assert unrecorded.code == "guarantees.unrecorded" and unrecorded.entry == "P.t"
    reworded = A.Failure(unrecorded.code, unrecorded.entry, "guarantees: P.t is not recorded yet")
    assert A.regenerated(POLICY, now, FILES, [reworded]) == A.judge(POLICY, now, FILES, update=True)[1]
    assert A.regenerated(POLICY, now, FILES, [str(unrecorded)]) is None, "the same text without its code settles nothing"
    lookalike = A.Failure("guarantees.no-owner", "P.t", str(unrecorded))
    assert A.regenerated(POLICY, now, FILES, [lookalike]) is None
    assert A.regenerated(POLICY, now, FILES, [reworded, lookalike]) is None
    assert A.code_of(unrecorded) == "guarantees.unrecorded" and A.code_of(str(unrecorded)) is None
    import copy
    back = copy.deepcopy(unrecorded)
    assert (back, back.code, back.entry) == (unrecorded, unrecorded.code, unrecorded.entry)
    assert json.loads(json.dumps([unrecorded])) == [str(unrecorded)], "a report holds its failures' text"


def test_reports_list_their_failures_codes_beside_them(tmp_path, monkeypatch):
    """`report.json`'s and `audit.json`'s packages carry `failure_codes`, one per failure in its order, and a train's
    `--regenerate` moves the codes with the failures it settles (`settled_codes`)."""
    a = tmp_path / "a"
    a.mkdir()
    (a / A.POLICY).write_text("{}")
    gone = A.Failure("guarantees.reads-gone", "P.Defs", "guarantees: they no longer read P.Defs")
    orphan = A.Failure("orphan.file", "X.lean", "orphan: X.lean is not imported by the roots (P) nor under `exempt`: nothing "
                       "builds or audits it")

    def fake(pkg, out, *args, **_):
        rec = out / f"audit-{pkg.name}" / A.POLICY
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text("{}")
        return {"package": str(pkg), "seconds": {}, "failures": list(fails)} | (
            {"regenerated": str(rec)} if all(map(A.RECORD_DIFFERENCE.match, fails)) else {})

    monkeypatch.setattr(A, "audit", fake)
    out = tmp_path / "out"
    for fails, flags, codes, settled in (([gone, orphan], [], ["guarantees.reads-gone", "orphan.file"], None),
                                         ([gone], ["--regenerate"], [], ["guarantees.reads-gone"])):
        A.main([str(a), *flags, "--out", str(out), "--repo", str(tmp_path)])
        (rep,) = json.loads((out / "audit.json").read_text())["packages"]
        assert rep["failure_codes"] == codes and rep.get("settled_codes") == settled
        assert len(rep["failures"]) == len(rep["failure_codes"])
    assert A.reported({"failures": [gone, "a bare string"]})["failure_codes"] == ["guarantees.reads-gone", None]


def test_a_32_bit_record_is_checked_as_it_is_and_rehashed_by_update():
    new = recorded()
    old = new | {"guarantees": {"P.t": new["guarantees"]["P.t"] | {"type_hash": "00000000000000aa"}},
                 "reads": A.read_groups(A.guarantee_reads(facts()), old=True)}
    assert len(old["reads"]["P.Defs"]["definitions"]["P.f"]) == 16 and A.old_group(old["reads"]["P.Defs"])
    assert old["reads"]["P.Defs"]["digest"] == A.module_digest(old["reads"]["P.Defs"]["definitions"])
    assert A.judge(old, facts(), FILES)[0] == []
    fails = A.judge(old, facts(pins={"P.t": fact(h="00000000000000bb")}), FILES)[0]
    assert fails == ["guarantees: P.t's type hash changed:\n00000000000000aa -> 00000000000000bb"]
    fails, rehashed = A.judge(old, facts(), FILES, update=True)
    assert fails == [] and rehashed == new
    text = A.review(old, rehashed, facts(), {})
    assert text == [f"guarantee P.t: {A.REHASHED}", f"reads: 2 {A.REGROUPED}"] and A.statement_changes(text) == []
    moved = facts(pins={"P.t": fact(reads=[{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "9" * 32},
                                          {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "3" * 32}])})
    text = A.review(old, A.judge(old, moved, FILES, update=True)[1], moved, {})
    assert text == [f"guarantee P.t: {A.REHASHED}", "definition P.f (P.Defs), read by P.t: changed", f"reads: 1 {A.REGROUPED}"]


def test_facts_names_a_module_without_listing_every_module():
    """`EnvironmentHeader.moduleNames` maps over every imported module on each call. Indexed once per constant a
    guarantee's reads reach, it was 1,471 s of `Security`'s 1,504 s audit (check r20261012-005706-568f)."""
    for tool in ("Facts.lean", "Upstream.lean"):
        assert "moduleNames[" not in (HERE.parent / tool).read_text(), \
            f"{tool}: name a module by `env.header.modules[i].module` (`moduleName env i`), not `env.header.moduleNames[i]`"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_sha256_hash_is_expr_hash_s_equivalence(tmp_path):
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    (pkg / "Control" / "Alpha.lean").write_text(
        "namespace Control\ntheorem a (x : Nat) : x + 0 = x := rfl\ntheorem b (y : Nat) : y + 0 = y := rfl\n"
        "theorem c {x : Nat} : x + 0 = x := rfl\ntheorem d (x : Nat) : 0 + x = x := Nat.zero_add x\nend Control\n")
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.Alpha\n")
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    f = A.facts_of(pkg, {"roots": ["Control"], "guarantees": {f"Control.{n}": {"owner": "@lean"} for n in "abcd"}}, tmp_path)
    got = {n: f["pins"][f"Control.{n}"]["type_sha256"] for n in "abcd"}
    assert got["a"] == got["b"] == got["c"] != got["d"] and len(got["a"]) == 64 and not (tmp_path / "canon.jsonl").exists()


SHAPE_SPEC = """namespace Control
def verify (x : Nat) : Bool := x % 2 == 0
structure Sess where
  x : Nat
  hown : verify x = true
abbrev Outer (_ : Nat) := Sess
structure Stmt where
  c : Nat
  hscope : c < 100
structure World where
  Ω : Type
  nonempty : Nonempty Ω
def Accepts (s : Sess) : Prop := verify s.x = true
def reads (s : Sess) : Nat := s.x + 1
def prob (_ : Nat → Prop) (n : Nat) : Nat := n
structure Commit where
  scheme : Nat
  root : Nat
def Opens (c : Commit) (o : Nat) : Prop := verify (c.root + o) = true
end Control
"""
SHAPE_THMS = """import Control.ShapeSpec
import Control.ShapePrem
namespace Control
theorem guarded (s : Sess) (_hacc : Accepts s) (_hop : Accepts s → reads s > 0) : reads s > 0 := by unfold reads; omega
theorem mentions (s : Sess) (_hacc : Accepts s) (_hm : reads s = s.x + 1) : True := trivial
theorem underForall (s : Sess) (_hacc : Accepts s) (_hA : ∀ j, Uniform j) : True := trivial
theorem family (_Vs : Fin 3 → Stmt) (zs : Fin 3 → Outer 0) (_hacc : ∀ j, Accepts (zs j)) : True := trivial
theorem world (_W : World) (s : Sess) (_hacc : Accepts s) : True := trivial
theorem event (zs : Nat → Sess) (_hA : ∀ j, Uniform j) (_hq : reads (zs 0) > 0) :
    prob (fun o => Accepts (zs o) ∧ reads (zs o) = 0) 1 ≤ 1 ∨ ∃ k : Nat, prob (fun o => Accepts (zs (o + k)) ∧ True) k ≤ k :=
  Or.inl (Nat.le_refl 1)
theorem eventLate (zs : Nat → Sess) : prob (fun o => reads (zs o) = 0 ∧ Accepts (zs o)) 1 ≤ 1 := Nat.le_refl 1
theorem shared (c c' : Commit) (o o' : Nat) (hr : c.root = c'.root) (_h : Opens c o) (_h' : Opens c' o') :
    c.root = c'.root := hr
theorem sharedUsed (c c' : Commit) (o o' : Nat) (hr : c.root = c'.root) (_h : Opens c o) (_h' : Opens c' o') :
    hr = hr := rfl
theorem sharedOther (c c' : Commit) (o o' : Nat) (_hr : c.root = c'.scheme) (_h : Opens c o) (_h' : Opens c' o') :
    True := trivial
theorem sharedUnread (c c' : Commit) (o : Nat) (_hr : c.root = c'.root) (_h : Opens c o) : True := trivial
theorem late (c : Commit) (w : Nat) (_h : Opens c 0) (_hw : c.root = w) : w = w := rfl
theorem early (c : Commit) (w : Nat) (_hw : c.root = w) (_h : Opens c 0) : w = w := rfl
theorem lateUnused (c : Commit) (w : Nat) (_h : Opens c 0) (_hw : c.root = w) : True := trivial
theorem closed (c : Commit) (w : Nat) (_hq : c.root = 2) (_h : Opens c 0) : w = w := rfl
theorem written (c : Commit) (_h : Opens c 0) (_hq : c.root = 2) : True := trivial
end Control
"""


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_shape_reads_what_a_hypothesis_applies_through_binders_and_families(tmp_path):
    """`Facts.lean`'s shape on real statements (the friction note `proofs/20261009T1340Z-friction-shape-check-session-
    mentions-pass`): mentioning a session whose structure names the function (`hown`) is no acceptance, nor is an
    implication whose consequent isn't one (`_hop`); a premise under `∀` is read by its head; a family of structures
    (`Vs : Fin 3 → Stmt`, an `abbrev` of one) holds its fields; a structure's own `Nonempty` is a hypothesis
    too, since a field `Nonempty {x // Bad x}` is an existence claim. A probability's events are read under the
    conclusion's binders (`∨ ∃ k, prob …`), and a family's field that applies the function is refused beside them."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    (pkg / "Control" / "ShapeSpec.lean").write_text(SHAPE_SPEC)
    (pkg / "Control" / "ShapePrem.lean").write_text("namespace Control\ndef Uniform (j : Nat) : Prop := j = j\nend Control\n")
    (pkg / "Control" / "ShapeThms.lean").write_text(SHAPE_THMS)
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.ShapeThms\n")
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    names = ["guarded", "mentions", "underForall", "family", "world", "event", "eventLate", "shared", "sharedUsed",
             "sharedOther", "sharedUnread", "late", "early", "lateUnused", "closed", "written"]
    f = A.facts_of(pkg, {"roots": ["Control"], "guarantees": {f"Control.{n}": {} for n in names}}, tmp_path)
    entry = {"function": "Control.verify", "outcome": "accept"}
    got = {n: [w.split(" : ")[0].split(" whose")[0] for w in A.refusals(entry, f["pins"][f"Control.{n}"]["shape"],
                                                                        ["Control.ShapePrem"], events=["Control.prob"])]
           for n in names}
    assert got == {"guarded": ["takes _hop"], "mentions": ["takes _hm"], "underForall": [], "family": ["takes _Vs.hscope"],
                   "world": ["takes _W.nonempty"], "event": ["takes zs.hown", "takes _hq"],
                   "eventLate": ["bounds an event of Control.prob"], "shared": [], "sharedUsed": ["takes hr"],
                   "sharedOther": ["takes _hr"], "sharedUnread": ["takes _hr"], "late": [], "early": [],
                   "lateUnused": ["takes _hw"], "closed": ["takes _hq"], "written": ["takes _hq"]}, got
    shared = f["pins"]["Control.shared"]["shape"][0]
    assert [h.get("shares") for h in shared["hypotheses"]] == [
        {"inputs": [0, 1], "path": ["Control.Commit.root"]}, None, None], "`c.root = c'.root`, one field of two inputs"
    assert [i["uses"] for i in shared["inputs"]] == [[], [], [], []] and shared["uses"] == [0, 1]


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def unclaimed(policy: dict) -> dict:
    """The control's lock without its `claims` section, for a test that lists guarantees of its own."""
    return {k: v for k, v in policy.items() if k != "claims"}


def test_moved_declarations_are_compared_under_their_old_names(tmp_path):
    """`Facts.lean`'s renaming and unfolding on a real build: a guarantee lifted into a `Guarantees` definition (its old theorem
    kept as the lemma that proves it), one whose statement holds a proof Lean abstracts out of a definition (`_proof_1`), and a
    guarantee renamed, all the same statement, a definition renamed with its content and one renamed with a new body; then a
    statement that reads a moved definition beside one that has its old name, which the audit refuses."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.Moves\n")
    moves = pkg / "Control" / "Moves.lean"
    fin = "(⟨n, Nat.lt_succ_of_le (Nat.le_refl n)⟩ : Fin (n + 1)).val = n"
    moves.write_text("namespace Control\ndef f (n : Nat) : Nat := n + 0\ndef g (n : Nat) : Nat := n\n"
                     "theorem a (n : Nat) : f n = n := rfl\ntheorem b (n : Nat) : g n = n := rfl\n"
                     f"theorem c (n : Nat) : {fin} := rfl\nend Control\n")
    pol = unclaimed(A.load(pkg))
    A.dump(pkg, pol | {"layers": pol["layers"] | {"Control.Moves": {"imports": []}},
                       "guarantees": {n: {"owner": "@lean"} for n in ("Control.main_thm", "Control.a", "Control.b", "Control.c")}})
    build = lambda: subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    audit = lambda **kw: A.audit(pkg, tmp_path / "out", update=True, replay=False, runs=False, log=lambda *_: None, **kw)
    build()
    assert unshaped(audit()) == []
    before = (pkg / A.POLICY).read_text()
    record = tmp_path / "out" / "audit-control" / A.POLICY
    assert record.read_text() == before, "the record beside the reports, where a recorded run publishes it"
    record.unlink()
    assert unshaped(audit()) == [] and (pkg / A.POLICY).read_text() == before
    assert record.read_text() == before, "an update that changes nothing writes it all the same"
    (pkg / A.POLICY).write_text(before.replace('"guarantees"', '"pins"'))
    rep = audit()
    assert unshaped(rep) == [] and "review" not in rep and (pkg / A.POLICY).read_text() == before, "renamed, not reviewed"
    moved = {"Control.a": "Control.SecurityProofs.a", "Control.b": "Control.SecurityProofs.b",
             "Control.c": "Control.SecurityProofs.c", "Control.f": "Control.Protocol.f", "Control.g": "Control.Protocol.g"}
    after = ("def Control.Protocol.f (n : Nat) : Nat := n + 0\ndef Control.Protocol.g (n : Nat) : Nat := 0 + n\n"
             "theorem Control.a (n : Nat) : Control.Protocol.f n = n := rfl\n"
             "theorem Control.SecurityProofs.b (n : Nat) : Control.Protocol.g n = n := Nat.zero_add n\n"
             f"theorem Control.c (n : Nat) : {fin} := rfl\n")
    spec = pkg / "Control" / "Guarantees.lean"

    def stated_as_spec():
        pol = A.load(pkg)
        A.dump(pkg, pol | {"layers": pol["layers"] | {"Control.Guarantees": {"imports": ["Control.Moves"]}}})
    stated = ("import Control.Moves\ndef Control.Guarantees.A : Prop := ∀ n, Control.Protocol.f n = n\n"
              f"def Control.Guarantees.C : Prop := ∀ n : Nat, {fin}\n")
    moves.write_text(after)
    spec.write_text(stated)
    stated_as_spec()
    (pkg / "Control" / "Lifted.lean").write_text(
        "import Control.Guarantees\ntheorem Control.SecurityProofs.a : Control.Guarantees.A := Control.a\n"
        "theorem Control.SecurityProofs.c : Control.Guarantees.C := Control.c\n")
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.Lifted\n")
    build()
    rep = audit(moved=moved)
    assert unshaped(rep) == []
    assert rep["before"] == {k: json.loads(before)[k] for k in ("guarantees", "reads")} and rep["guarantees"] == A.load(pkg)["guarantees"]
    must = A.statement_changes(rep["review"])
    assert set(rep["review"]) - set(must) == {
        f"guarantee Control.SecurityProofs.a: moved from Control.a, lifted into Control.Guarantees.A: {A.MOVED}",
        f"guarantee Control.SecurityProofs.c: moved from Control.c, lifted into Control.Guarantees.C: {A.MOVED}",
        f"guarantee Control.SecurityProofs.b: moved from Control.b: {A.MOVED}", f"reads: 1 {A.MOVED_READS}"}
    assert must[0] == ("definition Control.Protocol.g (Control.Moves), was Control.g (Control.Moves), read by "
                       "Control.SecurityProofs.b: changed") and must[-1] == "    def Control.Protocol.g (n : Nat) : Nat := 0 + n"
    (pkg / A.POLICY).write_text(before)
    stated_as_spec()
    moves.write_text("def Control.f (n : Nat) : Nat := n\n" + after)
    spec.write_text(stated.replace("f n = n\n", "f n = Control.f n\n"))
    build()
    rep = audit(moved=moved)
    assert [f.split(":")[0] for f in unshaped(rep)] == ["moved"] and "Control.Protocol.f would be Control.f" in rep["failures"][0]


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_lift_into_guarantees_is_certified_on_a_real_build(tmp_path):
    """`Facts.lean`'s one-step unfolding on a real build, without `--moved`: lifts with no binders, with binders whose argument
    the definition's own binder must shift (`Q`), universe-polymorphic (`R`), and with a named assumption kept as a binder
    (`V`) are each the same statement; a definition that differs (`S`, `n * 2` for `n + n`), one that takes the named
    assumption in (`T`) and one outside a `Guarantees` module (`Spec.W`) are changes of meaning; the guarantee nobody
    lifted keeps its record byte for byte."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.Lifts\n")
    lifts = pkg / "Control" / "Lifts.lean"
    lifts.write_text(
        "import Control.Assumptions\nimport Control.Defs\nnamespace Control\n"
        "theorem p : ∀ n : Nat, double n = n + n := by intro n; unfold double; omega\n"
        "theorem q (n : Nat) : ∀ m : Nat, double n + m = m + double n := fun _ => Nat.add_comm _ _\n"
        "theorem r.{u} (α : Sort u) (a : α) : a = a := rfl\n"
        "theorem s (n : Nat) : double n = n + n := by unfold double; omega\n"
        "theorem t (h : Hyp) (n : Nat) : double n = n + n := by unfold double; omega\n"
        "theorem v (h : Hyp) (n : Nat) : double n = n + n := by unfold double; omega\n"
        "theorem w (n : Nat) : double n = n + n := by unfold double; omega\n"
        "end Control\n")
    A.dump(pkg, unclaimed(A.load(pkg)) | {"guarantees": {f"Control.{n}": {"owner": "@lean"} for n in ("main_thm", *"pqrstvw")}})
    build = lambda: subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    audit = lambda: A.audit(pkg, tmp_path / "out", update=True, replay=False, runs=False, log=lambda *_: None)
    build()
    assert unshaped(audit()) == []
    before = A.load(pkg)
    (pkg / "Control" / "Guarantees.lean").write_text(
        "import Control.Assumptions\nimport Control.Defs\nnamespace Control.Guarantees\nopen Control\nuniverse u\n"
        "def P : Prop := ∀ n : Nat, double n = n + n\n"
        "def Q (n : Nat) : Prop := ∀ m : Nat, double n + m = m + double n\n"
        "def R (α : Sort u) (a : α) : Prop := a = a\n"
        "def S (n : Nat) : Prop := double n = n * 2\n"
        "def T : Prop := Hyp → ∀ n : Nat, double n = n + n\n"
        "def V (n : Nat) : Prop := double n = n + n\n"
        "end Control.Guarantees\n")
    (pkg / "Control" / "Spec.lean").write_text(
        "import Control.Defs\nnamespace Control.Spec\ndef W (n : Nat) : Prop := Control.double n = n + n\nend Control.Spec\n")
    lifts.write_text(
        "import Control.Guarantees\nimport Control.Spec\nnamespace Control\n"
        "theorem p : Guarantees.P := by intro n; unfold double; omega\n"
        "theorem q (n : Nat) : Guarantees.Q n := fun _ => Nat.add_comm _ _\n"
        "theorem r.{u} (α : Sort u) (a : α) : Guarantees.R α a := rfl\n"
        "theorem s (n : Nat) : Guarantees.S n := by unfold Guarantees.S double; omega\n"
        "theorem t : Guarantees.T := by intro _ n; unfold double; omega\n"
        "theorem v (h : Hyp) (n : Nat) : Guarantees.V n := by unfold Guarantees.V double; omega\n"
        "theorem w (n : Nat) : Spec.W n := by unfold Spec.W double; omega\n"
        "end Control\n")
    pol = A.load(pkg)
    A.dump(pkg, pol | {"layers": pol["layers"] | {"Control.Guarantees": {"imports": ["Control.Assumptions", "Control.Defs"]},
                                                  "Control.Spec": {"imports": ["Control.Defs"]}}})
    build()
    rep = audit()
    assert unshaped(rep) == []
    after = A.load(pkg)
    assert after["guarantees"]["Control.main_thm"] == before["guarantees"]["Control.main_thm"]
    assert after["guarantees"]["Control.q"]["signature"] == ["Control.q (n : Nat) : Control.Guarantees.Q n"]
    must = A.statement_changes(rep["review"])
    assert set(rep["review"]) - set(must) == {f"guarantee Control.{n}: lifted into Control.Guarantees.{n.upper()}: {A.LIFTED}"
                                              for n in "pqrv"}
    assert [line for line in must if line.startswith(("guarantee", "definition"))] == [
        "guarantee Control.s: changed", "guarantee Control.t: changed", "guarantee Control.w: changed",
        "definition Control.Guarantees.S (Control.Guarantees), read by Control.s: new",
        "definition Control.Guarantees.T (Control.Guarantees), read by Control.t: new",
        "definition Control.Spec.W (Control.Spec), read by Control.w: new"]
    assert "  named assumptions: Control.Hyp -> none" in must


def test_an_instance_is_anonymous_where_its_name_is_the_keyword_and_gets_lean_s_name(tmp_path):
    (tmp_path / "P").mkdir()
    (tmp_path / "P" / "I.lean").write_text(
        "namespace P\n/-- instance : here is prose -/\ninstance : Inhabited Nat := ⟨0⟩\n"
        "@[simp] private instance (priority := (low + 1))\n    : ToString Nat := ⟨fun _ => \"\"⟩\n"
        "instance instOk : Inhabited Bool := ⟨true⟩\nend P\n")
    found = [{"name": "P.instInhabitedNat_p", "id": "instInhabitedNat_p", "module": "P.I", "at": [3, 0]},
             {"name": "P.instToStringNat_p", "id": "instToStringNat_p", "module": "P.I", "at": [4, 16]},
             {"name": "P.instOk", "id": "instOk", "module": "P.I", "at": [6, 9]},
             {"name": "P.instReprT", "id": "instReprT", "module": "P.I", "at": [2, 0]},
             {"name": "Q.instX", "id": "instX", "module": "Q", "at": [1, 0]}]
    anon = A.anonymous_instances(tmp_path, {"P.I": "P/I.lean"}, {"instances": found})
    assert [i["name"] for i in anon] == ["P.instInhabitedNat_p", "P.instToStringNat_p"]
    A.name_instances(tmp_path, anon)
    assert (tmp_path / "P" / "I.lean").read_text() == (
        "namespace P\n/-- instance : here is prose -/\ninstance instInhabitedNat_p : Inhabited Nat := ⟨0⟩\n"
        "@[simp] private instance (priority := (low + 1)) instToStringNat_p\n    : ToString Nat := ⟨fun _ => \"\"⟩\n"
        "instance instOk : Inhabited Bool := ⟨true⟩\nend P\n")


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_naming_the_anonymous_instances_keeps_their_names_and_the_record(tmp_path):
    """On a real build: the instances Lean named, a private one with a priority, a local one and one whose type is the
    package's own among them, fail the audit; `name` writes their names, and the rebuilt package passes with the same
    instances, the same record and nothing anonymous left. A derived instance and a named one are left as they are."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.Inst\n")
    (pkg / "Control" / "Inst.lean").write_text(
        "namespace Control\nstructure Box where\n  n : Nat\n  deriving Repr\n/-- documented -/\ninstance : Inhabited Box := ⟨⟨0⟩⟩\n"
        "@[simp] private instance (priority := low)\n    : ToString Box := ⟨fun b => toString b.n⟩\n"
        "local instance : Inhabited Nat := ⟨7⟩\ninstance : Inhabited (Fin 3) := ⟨0⟩\n"
        "instance instNoBox : Inhabited (List Box) := ⟨[]⟩\nend Control\n")
    build = lambda: subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    audit = lambda **kw: A.audit(pkg, tmp_path / "out", replay=False, runs=False, log=lambda *_: None, **kw)
    build()
    assert audit(update=True)["failures"] == [
        f"instance: Control/Inst.lean:{line} declares an instance without a name, which Lean names Control.{n} from its type "
        f"(and, when nothing in the type is the project's own, its project's root): a move or a split renames it, and with it "
        f"every record whose definitions use it. Write `instance {n}` there (`check.py --name-instances` writes each one)"
        for line, n in ((6, "instInhabitedBox"), (7, "instToStringBox"), (9, "instInhabitedNat_control"),
                        (10, "instInhabitedFinOfNatNat_control"))]
    record = (pkg / A.POLICY).read_text()
    facts = lambda: sorted(i["name"] for i in json.loads((tmp_path / "out" / "audit-control" / "facts.json").read_text())["instances"])
    names = facts()
    named = audit(name=True)
    assert len(named["named_instances"]) == 4
    assert named["failures"] == ["instance: named 4 anonymous instances in Control/Inst.lean, so this audit is of sources it has "
                                 "since rewritten: build and audit again to see that no record changed"]
    assert "local instance instInhabitedNat_control : Inhabited Nat" in (pkg / "Control" / "Inst.lean").read_text()
    build()
    assert audit()["failures"] == []
    assert facts() == names and (pkg / A.POLICY).read_text() == record


def test_all_neither_updates_records_nor_names_instances(capsys):
    for flag in ("--update", "--name-instances"):
        with pytest.raises(SystemExit):
            A.main(["--all", flag])
        assert "neither updates records nor names instances" in capsys.readouterr().err


def test_a_package_pinning_arklib_is_refused_below_the_memory_floor(tmp_path):
    """A machine with less memory than ArkLib's build peaks at (swap included, within its cgroup) refuses to audit a package
    that pins it, and the refusal prints the fast path's line on a node; other packages, or a floor set to 0, go ahead."""
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc" / "meminfo").write_text("MemTotal:       16000000 kB\nMemFree:  100 kB\nSwapTotal:      4000000 kB\n")
    cg = tmp_path / "cg"
    cg.mkdir()
    assert round(A.memory_gib(tmp_path / "proc", cg), 1) == 19.1
    (cg / "memory.max").write_text(f"{8 * 2**30}\n")
    (cg / "memory.swap.max").write_text("max\n")
    assert round(A.memory_gib(tmp_path / "proc", cg), 1) == 11.8
    (cg / "memory.swap.max").write_text("0\n")
    assert A.memory_gib(tmp_path / "proc", cg) == 8
    heavy, light = tmp_path / "Proofs", tmp_path / "Spec"
    for p, deps in ((heavy, ["mathlib", "Arklib"]), (light, ["mathlib"])):
        p.mkdir()
        (p / "lake-manifest.json").write_text(json.dumps({"packages": [{"name": d} for d in deps]}))
    why = A.memory_refusal([light, heavy], 16)
    assert why.startswith(f"{heavy} pins ArkLib") and "lean_changed.py --records" in why and str(light) in why
    assert A.memory_refusal([light], 16) is None
    assert A.memory_refusal([light, heavy], 64) is None
    assert A.memory_refusal([heavy], 16, env={"VERITY_LEAN_MEM_FLOOR_GIB": "0"}) is None


def test_a_mathlib_build_runs_no_more_lean_at_once_than_its_scope_s_memory_holds(tmp_path):
    """Two queued Security builds were OOM-killed in 64 and 16 GB scopes on 18 Oct, with a `lean` per CPU: Lake runs as many
    at once as `LEAN_NUM_THREADS`, so a package that pins Mathlib or ArkLib builds with at most one per `BUILD_GIB_PER_JOB`.
    The scope's limit is read where `/proc/self/cgroup` says the process is, and the smallest of it and its parents' holds."""
    proc, cg = tmp_path / "proc", tmp_path / "cg"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text(f"MemTotal: {512 * 2**20} kB\nSwapTotal: 0 kB\n")
    (proc / "self" / "cgroup").write_text("0::/user.slice/cluster-r1.scope\n")
    (cg / "user.slice" / "cluster-r1.scope").mkdir(parents=True)
    (cg / "memory.max").write_text("max\n")
    assert A.memory_gib(proc, cg) == 512
    (cg / "user.slice" / "cluster-r1.scope" / "memory.max").write_text(f"{64 * 2**30}\n")
    assert A.memory_gib(proc, cg) == 64
    (cg / "user.slice" / "memory.max").write_text(f"{32 * 2**30}\n")
    assert A.memory_gib(proc, cg) == 32, "a parent's smaller limit holds"
    heavy, light = tmp_path / "heavy", tmp_path / "light"
    for p, deps in ((heavy, ["mathlib"]), (light, ["batteries"])):
        p.mkdir()
        (p / "lake-manifest.json").write_text(json.dumps({"packages": [{"name": d} for d in deps]}))
    env = {}
    assert A.build_threads(heavy, env, have_gib=64) == min(A.cores(), 64 // A.BUILD_GIB_PER_JOB)
    assert A.build_threads(heavy, env, have_gib=4) == 1
    assert A.build_threads(light, env, have_gib=4) is None
    assert A.build_threads(heavy, {"LEAN_NUM_THREADS": "3"}, have_gib=4) is None, "an explicit setting wins"
    assert A.replay_threads(heavy, env, have_gib=64) == min(A.cores(), 64 // A.REPLAY_GIB_PER_THREAD)
    assert A.replay_threads(light, env, have_gib=4) is None


def test_a_spec_whose_guarantees_an_arklib_package_proves_needs_the_spec_floor(tmp_path):
    """A whole-Security replay holds about 120 GiB whatever its threads (killed at 128 GiB with 16 in r20261019-084601-01a0),
    so auditing a spec with `proved_in` beside an ArkLib package needs `SPEC_FLOOR_GIB`, and is sent to the shape that has
    it; the package alone needs the build's floor."""
    proofs, spec = tmp_path / "Proofs", tmp_path / "Spec"
    for p in (proofs, spec):
        p.mkdir()
    (proofs / "lake-manifest.json").write_text(json.dumps({"packages": [{"name": "Arklib"}]}))
    (spec / A.POLICY).write_text(json.dumps({"roots": ["S"], "proved_in": ["../Proofs"]}))
    assert A.memory_refusal([proofs], 32) is None
    why = A.memory_refusal([spec, proofs], 32)
    assert why and f"under the {A.SPEC_FLOOR_GIB} GiB floor" in why and "facts for every guarantee it proves" in why
    assert "--kind lean-audit" in (A.memory_refusal([spec, proofs], 128) or "")
    assert A.memory_refusal([spec, proofs], A.SPEC_FLOOR_GIB) is None


PLUGGED = '''name = "Dep"
defaultTargets = ["Dep"]

[[lean_lib]]
name = "Plug"
srcDir = "scripts"
roots = ["Plug"]

[[lean_lib]]
name = "Dep"
plugins = ["Plug:shared"]
requiresModuleSystem = true

[[lean_lib]]
name = "DepTest"
plugins = ["Plug:shared"]
'''


def test_the_dependencies_plugins_are_named_once_each():
    """`plugin_targets`: each plugin a fetched dependency's libraries load, once, under the dependency's name."""
    import tempfile
    pkg = Path(tempfile.mkdtemp())
    for name, text in (("Dep", PLUGGED), ("doc", 'name = "«doc-gen4»"\n[[lean_lib]]\nname = "D"\nplugins = ["P:shared"]\n'),
                       ("plain", 'name = "plain"\n[[lean_lib]]\nname = "Plain"\n'), ("broken", "name = \n")):
        (pkg / ".lake" / "packages" / name).mkdir(parents=True)
        (pkg / ".lake" / "packages" / name / "lakefile.toml").write_text(text)
    assert A.plugin_targets(pkg) == ["@Dep/Plug:shared", "@doc-gen4/P:shared"]
    assert A.plugin_targets(pkg / "none") == []


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_plugin_two_libraries_load_is_linked_alone_first_so_the_build_links_it_no_more(tmp_path):
    """INF-62: Lake 4.34 links a plugin twice when a module-system library and a classic one load it, as ArkLib's `ArkLib` and
    `ArkLibTest` load `ArkLibLintPlugin:shared`, and the two links of one file race (three proofs builds failed so on 18 Oct).
    Whether a plain build schedules the second link depends on Lake's scheduling, so the test checks only the fix: linked
    alone first (`plugin_targets`), the build finds it up to date and links nothing."""
    dep, root = tmp_path / "dep", tmp_path / "root"
    (dep / "scripts").mkdir(parents=True)
    root.mkdir()
    toolchain = (A.FIXTURES / "control" / "lean-toolchain").read_text()
    (dep / "lakefile.toml").write_text(PLUGGED)
    (dep / "scripts" / "Plug.lean").write_text("def plug : Nat := 1\n")
    (dep / "Dep.lean").write_text("module\npublic def dep : Nat := 2\n")
    (dep / "DepTest.lean").write_text("def depTest : Nat := 3\n")
    (root / "lakefile.toml").write_text('name = "root"\ndefaultTargets = ["Root"]\n\n[[require]]\nname = "Dep"\n'
                                        'path = "../dep"\n\n[[lean_lib]]\nname = "Root"\n')
    (root / "Root.lean").write_text("import Dep\nimport DepTest\ndef root : Nat := dep + depTest\n")
    for d in (dep, root):
        (d / "lean-toolchain").write_text(toolchain)
    lake = lambda *a: subprocess.run(["lake", *a], cwd=root, capture_output=True, text=True, env=A.lake_env())
    (root / ".lake" / "packages").mkdir(parents=True)
    (root / ".lake" / "packages" / "Dep").symlink_to(dep)
    assert A.plugin_targets(root) == ["@Dep/Plug:shared"]
    assert lake("build", *A.plugin_targets(root)).returncode == 0
    out = lake("build")
    assert out.returncode == 0 and "Plug:shared" not in out.stdout, out.stdout


def test_update_refuses_a_shared_shipped_tree(tmp_path, capsys):
    pkg = tmp_path / "Security"
    pkg.mkdir()
    (tmp_path / "READY.json").write_text("{}\n")
    with pytest.raises(SystemExit):
        A.main([str(pkg), "--update"])
    assert "--cwd clone" in capsys.readouterr().err


def test_a_theorem_s_statement_is_read_from_its_source():
    src = ("namespace P\n/-- twice -/\n@[simp] theorem t.{u} {α : Sort u} (f : α → α) -- a note\n    (h : ∀ x, f x = x) :\n"
           "    ∀ x, f (f x) = x := by\n  intro x; simp [h]\n"
           "theorem s : (fun (n : Nat) => n) 0 = 0\n  | _ => rfl\n")
    assert A.statement(src, (2, 0)) == ("/-- twice -/", ".{u}", "{α : Sort u} (f : α → α) (h : ∀ x, f x = x)",
                                         "∀ x, f (f x) = x")
    assert A.statement(src, (7, 0)) == ("", "", "", "(fun (n : Nat) => n) 0 = 0")


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_lift_writes_the_four_name_form_and_the_update_certifies_it(tmp_path):
    """On a real build: `lift` writes `def G : Prop` (doc comment, universe parameters, binders across lines) into a
    `Guarantees` module, which then imports the module of the definition it reads, and `theorem X.SecurityProofs.G :
    X.Guarantees.G` beside the proof, or in the module `beside` names, which then imports the spec module and the proof's;
    after a build, `--update --moved` with the moves it returns certifies each as the same statement under its old name,
    the record changes no statement, and `shape.inline`'s entry (`twice`'s `m = n`) follows its guarantee."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text()
                                      + "import Control.Lifts\nimport Control.Guarantees\nimport Control.Collect\n")
    (pkg / "Control" / "Guarantees.lean").write_text("")
    (pkg / "Control" / "Collect.lean").write_text("import Control.Defs\n")
    (pkg / "Control" / "Lifts.lean").write_text(
        "import Control.Defs\nopen Nat\nnamespace Control.Sub\n"
        "/-- Doubling is adding. -/\ntheorem twice {m : Nat} (n : Nat)\n    (h : m = n) :\n    double n = n + n := by\n"
        "  subst h; unfold double; omega\n"
        "theorem refl.{u} (α : Sort u) (a : α) : a = a := rfl\n"
        "section\nvariable {k : Nat} (j : Nat) [Inhabited (Fin k)]\n"
        "open Classical in\ntheorem pick (p : Prop) : (if p then j else j) = j + 0 := by split <;> rfl\nend\n"
        "end Control.Sub\n")
    pol = unclaimed(A.load(pkg))
    why = {"Eq m n": "inline when the rule came"}
    A.dump(pkg, pol | {"guarantees": {n: {"owner": "@lean"} for n in ("Control.main_thm", "Control.Sub.twice", "Control.Sub.refl",
                                                                        "Control.Sub.pick")},
                       "layers": pol["layers"] | {"Control.Guarantees": {"imports": ["Control.Defs"]}},
                       "shape": (pol.get("shape") or {}) | {"inline": {"Control.Sub.twice": why}}})
    build = lambda: subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    audit = lambda **kw: A.audit(pkg, tmp_path / "out", replay=False, runs=False, log=lambda *_: None, **kw)
    build()
    assert unshaped(audit(update=True)) == []
    rep = audit(lifts={"Control.Sub.twice": {"as": "Control.Guarantees.Twice", "into": "Control.Guarantees"},
                       "Control.Sub.refl": {"as": "Control.Guarantees.Refl", "into": "Control.Guarantees",
                                            "beside": "Control.Collect"},
                       "Control.Sub.pick": {"as": "Control.Guarantees.Pick", "into": "Control.Guarantees"}})
    assert unshaped(rep) == [] and rep["moved"] == {"Control.Sub.twice": "Control.SecurityProofs.Twice",
                                                      "Control.Sub.refl": "Control.SecurityProofs.Refl",
                                                      "Control.Sub.pick": "Control.SecurityProofs.Pick"}
    assert (pkg / "Control" / "Guarantees.lean").read_text() == (
        "import Control.Defs\n\nnamespace Control.Sub\nopen Nat\n\nopen Classical in\n"
        "def _root_.Control.Guarantees.Pick : Prop :=\n  ∀ (j : Nat) (p : Prop),\n    (if p then j else j) = j + 0\n\n"
        "end Control.Sub\n\n"
        "namespace Control.Sub\nopen Nat\n\ndef _root_.Control.Guarantees.Refl.{u} : Prop :=\n"
        "  ∀ (α : Sort u) (a : α),\n    a = a\n\nend Control.Sub\n\n"
        "namespace Control.Sub\nopen Nat\n\n/-- Doubling is adding. -/\ndef _root_.Control.Guarantees.Twice : Prop :=\n"
        "  ∀ {m : Nat} (n : Nat) (h : m = n),\n    double n = n + n\n\nend Control.Sub\n")
    lifts = (pkg / "Control" / "Lifts.lean").read_text()
    assert lifts.startswith("import Control.Defs\nimport Control.Guarantees\nopen Nat\n") and lifts.endswith(
        "\ntheorem Control.SecurityProofs.Pick : Control.Guarantees.Pick := @Control.Sub.pick\n\n"
        "theorem Control.SecurityProofs.Twice : Control.Guarantees.Twice := @Control.Sub.twice\n")
    assert "Refl" not in lifts
    assert (pkg / "Control" / "Collect.lean").read_text() == (
        "import Control.Defs\nimport Control.Guarantees\nimport Control.Lifts\n\n"
        "theorem Control.SecurityProofs.Refl.{u} : Control.Guarantees.Refl.{u} := @Control.Sub.refl.{u}\n")
    build()
    rep = audit(update=True, moved=rep["moved"])
    assert unshaped(rep) == [] and A.statement_changes(rep["review"]) == []
    assert set(A.load(pkg)["guarantees"]) == {"Control.main_thm", "Control.SecurityProofs.Twice", "Control.SecurityProofs.Refl",
                                              "Control.SecurityProofs.Pick"}
    assert A.load(pkg)["shape"]["inline"] == {"Control.SecurityProofs.Twice": why}


def test_a_lift_whose_statement_reads_a_proof_package_s_definition_is_refused(tmp_path):
    (tmp_path / "spec" / "Spec").mkdir(parents=True)
    (tmp_path / "proofs" / "Proofs").mkdir(parents=True)
    (tmp_path / "spec" / "Spec" / "Guarantees.lean").write_text("")
    (tmp_path / "proofs" / "Proofs" / "D.lean").write_text("def P.d : Nat := 0\n")
    (tmp_path / "proofs" / "Proofs" / "T.lean").write_text("import Proofs.D\ntheorem P.t : P.d = 0 := rfl\n")
    for read in ({"reads": [{"name": "P.d", "module": "Proofs.D"}]}, {"outside": [["P.d", "Proofs.D"]]}):
        facts = {"pins": {"P.t": {"kind": "theorem", "at": [2, 0], "module": "Proofs.T", **read}}}
        with pytest.raises(ValueError, match="reads definitions of Proofs.D"):
            A.lift(tmp_path / "spec", {"proved_in": ["../proofs"]}, facts,
                   {"P.t": {"as": "P.Guarantees.T", "into": "Spec.Guarantees"}})
    assert (tmp_path / "spec" / "Spec" / "Guarantees.lean").read_text() == ""


def test_a_lift_takes_the_variables_its_statement_uses_and_no_namespace_only_a_proof_declares():
    src = ("namespace P\nvariable {α : Type} [Fintype α]\nsection\nvariable (gone : Nat)\nend\n"
           "variable {β : Type} [DecidableEq α] (k : ℕ) {n m : ℕ} [Inhabited (Fin n)]\n"
           "open Classical in\n/-- doc -/\ntheorem t (x : α) : k = k := rfl\n")
    start = src.index("/--")
    variables, open_in = A.in_scope(src, start)
    assert variables == ["{α : Type}", "[Fintype α]", "{β : Type}", "[DecidableEq α]", "(k : ℕ)", "{n m : ℕ}",
                         "[Inhabited (Fin n)]"] and open_in == "open Classical in"
    assert A.used(variables, "(x : α) k = k") == ["{α : Type}", "[Fintype α]", "[DecidableEq α]", "(k : ℕ)"]
    assert A.in_scope("theorem t : 1 = 1 := rfl\n", 0) == ([], "")
    assert [A.unproved_opens(line, {"PThm"}) for line in ("open P PThm", "open PThm", "open scoped PThm", "  open P")] == [
        "open P", "", "open scoped PThm", "  open P"]


def test_generated_constants_belong_to_their_declaration():
    assert [A.parent(n) for n in ("P.f.match_1", "P.walk._f", "P.Enc.mk", "P.instDecidableFits", "mk")] == [
        "P.f", "P.walk", "P.Enc", "P.instDecidableFits", "mk"]


def test_a_definitions_record_is_the_same_whichever_guarantees_read_it():
    """A generated constant is part of its declaration's digest only when the declaration reaches it itself (`own`), so
    removing the guarantee that read `P.Enc.casesOn` through another definition's match leaves `P.Enc`'s record as it was,
    rather than showing it changed with no change to its source; `P.Enc.casesOn` is a record of its own, which goes."""
    enc = {"name": "P.Enc", "module": "P.Defs", "kind": "inductive", "hash": "1" * 32, "own": ["P.Enc.mk"]}
    mk = {"name": "P.Enc.mk", "module": "P.Defs", "kind": "constructor", "hash": "2" * 32, "own": []}
    cases = {"name": "P.Enc.casesOn", "module": "P.Defs", "kind": "def", "hash": "3" * 32, "own": []}
    g = {"name": "P.g", "module": "P.Defs", "kind": "def", "hash": "4" * 32, "own": []}
    f = {"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "5" * 32, "own": ["P.f.match_1"]}
    m1 = {"name": "P.f.match_1", "module": "P.Defs", "kind": "def", "hash": "6" * 32, "own": []}
    both = facts(pins={"P.s": fact(reads=[enc, mk, f, m1]), "P.t": fact(reads=[enc, mk, g, cases])})
    one = facts(pins={"P.s": fact(reads=[enc, mk, f, m1])})
    defs = [A.read_groups(A.guarantee_reads(x))["P.Defs"]["definitions"] for x in (both, one)]
    assert sorted(defs[0]) == ["P.Enc", "P.Enc.casesOn", "P.f", "P.g"] and sorted(defs[1]) == ["P.Enc", "P.f"]
    hashed = [{**r, "sha256": sha(r["hash"])} for r in (enc, mk, f, m1)]
    assert defs[0]["P.Enc"] == defs[1]["P.Enc"] == A.digest(hashed[:2]) and defs[0]["P.f"] == A.digest(hashed[2:])
    alone = facts(pins={"P.t": fact(reads=[g, m1])})
    assert sorted(A.read_groups(A.guarantee_reads(alone))["P.Defs"]["definitions"]) == ["P.f.match_1", "P.g"]
    walk = {"name": "P.walk", "module": "P.Defs", "kind": "def", "hash": "7" * 32, "own": ["P.walk._f"]}
    walk_f = {"name": "P.walk._f", "module": "P.Defs", "kind": "def", "hash": "8" * 32, "own": ["P.g", "P.walk.match_1"]}
    walk_m = {"name": "P.walk.match_1", "module": "P.Defs", "kind": "def", "hash": "9" * 32, "own": []}
    rec = facts(pins={"P.t": fact(reads=[walk, walk_f, walk_m, g])})
    assert sorted(A.read_groups(A.guarantee_reads(rec))["P.Defs"]["definitions"]) == ["P.g", "P.walk"]   # through its `_f`


def test_regrouped_records_of_unchanged_declarations_are_one_line_the_dm_leaves_out():
    """A record whose declarations are unchanged but whose generated constants are grouped differently (`group_keys`) is
    counted on one line that `statement_changes` leaves out, so Daniel's DM lists only real changes; a changed declaration
    among them is still listed."""
    enc = {"name": "P.Enc", "module": "P.Defs", "kind": "inductive", "hash": "1" * 32, "own": ["P.Enc.mk"]}
    mk = {"name": "P.Enc.mk", "module": "P.Defs", "kind": "constructor", "hash": "2" * 32, "own": []}
    cases = {"name": "P.Enc.casesOn", "module": "P.Defs", "kind": "def", "hash": "3" * 32, "own": []}
    g = {"name": "P.g", "module": "P.Defs", "kind": "def", "hash": "4" * 32, "own": []}
    hyp = {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "6" * 32, "own": []}
    pol = {**POLICY, "guarantees": {"P.t": {"owner": "@lean"}}}
    now = facts(pins={"P.t": fact(reads=[enc, mk, cases, g, hyp])})
    sha_ = lambda *rs: A.digest([{**r, "sha256": sha(r["hash"])} for r in rs])
    by_parent = {"P.Enc": sha_(enc, mk, cases), "P.g": sha_(g)}
    first = A.judge(pol, now, FILES, update=True)[1]
    old = first | {"reads": first["reads"] | {"P.Defs": {"digest": A.module_digest(by_parent), "definitions": by_parent,
                                                          "guarantees": ["P.t"]}}}
    new = A.judge(old, now, FILES, update=True)[1]
    notes = A.review(old, new, now, {})
    assert notes == [f"reads: 2 {A.REGROUPED_OWN}"] and A.statement_changes(notes) == []
    now2 = facts(pins={"P.t": fact(reads=[enc, mk, cases, {**g, "hash": "5" * 32}, hyp])})
    notes = A.review(old, A.judge(old, now2, FILES, update=True)[1], now2, {})
    assert [n.split(":")[0] for n in A.statement_changes(notes)] == ["definition P.g (P.Defs), read by P.t"]


def test_update_prints_each_changed_guarantee_and_definition(tmp_path):
    (tmp_path / "Defs.lean").write_text("namespace P\n\ndef f (n : Nat) : Nat := n\n\nend P\n")
    old = recorded()
    reads = [{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "9" * 32, "lines": [3, 3]},
             {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "3" * 32, "lines": None}]
    now = facts(pins={"P.t": fact(signature="P.t (h : P.Hyp) (n : Nat) : P.f n ≤ n", h="00000000000000bb", reads=reads)})
    fails, new = A.judge(old, now, FILES, update=True)
    assert fails == []
    text = A.review(old, new, now, {"P.Defs": tmp_path / "Defs.lean"})
    assert text[:5] == ["guarantee P.t: changed", "  before:", "    P.t (h : P.Hyp) (n : Nat) : P.f n = n", "  after:",
                        "    P.t (h : P.Hyp) (n : Nat) : P.f n ≤ n"]
    assert text[5] == "definition P.f (P.Defs), read by P.t: changed" and text[-1] == "    def f (n : Nat) : Nat := n"
    assert A.statement_changes(text) == text
    assert A.review(new, new, now, {}) == []
    reprinted = dict(new["guarantees"]["P.t"], signature=["P.t (h : P.Hyp) (n : Nat) : LE.le (P.f n) n"])
    text = A.review(new, dict(new, guarantees={"P.t": reprinted}), now, {})
    assert text == [f"guarantee P.t: {A.REPRINTED}"] and A.statement_changes(text) == []


def moved_fact(name="Q.t", was="P.t", h="00000000000000cc", as_was="00000000000000aa", head=None, f="1" * 32, f_was="1" * 32,
              extra=()):
    """A guarantee ``was`` renamed to ``name``, reading `P.f` renamed to `Q.f` (in `Q.Defs`), as `Facts.lean` reports it after
    `--moved`: its statement's text under the old names hashes like ``as_was``, `Q.f`'s like ``f_was``; with ``head``,
    its type is that definition applied to `n`, which unfolded under the old names hashes like ``as_was``."""
    reads = [{"name": "Q.f", "module": "Q.Defs", "kind": "def", "hash": f, "own": ["Q.f.match_1"], "moved": {"as": "P.f", "sha256": sha(f_was)}},
             {"name": "Q.f.match_1", "module": "Q.Defs", "kind": "def", "hash": "2" * 32,
              "moved": {"as": "P.f.match_1", "sha256": sha("2" * 32)}},
             {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "3" * 32,
              "moved": {"as": "P.Hyp", "sha256": sha("3" * 32)}}, *extra]
    signature = f"{name} (h : P.Hyp) (n : Nat) : {head + ' n' if head else 'Q.f n = n'}"
    p = fact(signature=signature, h=h, reads=[{k: v for k, v in r.items() if k != "moved"} for r in reads])
    for r, m in zip(p["reads"], reads):
        r["moved"] = m["moved"]
    p["moved"] = {"as": was, "sha256": sha(head or as_was)} | ({"head": head, "unfolded_sha256": sha(as_was)} if head else {})
    return p | ({"lift": {"head": head, "module": "Q.Guarantees", "unfolded_sha256": sha(head)}} if head else {})


def test_a_definition_read_in_another_module_is_the_same_definition():
    old = recorded()
    rehomed = facts(pins={"P.t": fact(reads=[r | {"module": "P.Model"} if r["name"].startswith("P.f") else r
                                            for r in fact()["reads"]])})
    new = A.judge(old, rehomed, FILES, update=True)[1]
    assert set(new["reads"]) == {"P.Assumptions", "P.Model"}
    text = A.review(old, new, rehomed, {})
    assert text == [f"reads: 1 {A.MOVED_READS}"] and A.statement_changes(text) == []


def test_a_moved_guarantee_and_its_definitions_are_compared_with_their_old_records(tmp_path):
    old = recorded()
    current = {**POLICY, "guarantees": {"Q.t": old["guarantees"]["P.t"]}}
    now = facts(pins={"Q.t": moved_fact()})
    new = A.judge(current, now, FILES, update=True)[1]
    text = A.review(old, new, now, {})
    assert text == [f"guarantee Q.t: moved from P.t: {A.MOVED}", f"reads: 1 {A.MOVED_READS}"] and A.statement_changes(text) == []
    (tmp_path / "Defs.lean").write_text("namespace Q\n\ndef f (n : Nat) : Nat := n * 1\n\nend Q\n")
    now = facts(pins={"Q.t": moved_fact(f="9" * 32, f_was="9" * 32)})
    now["pins"]["Q.t"]["reads"][0]["lines"] = [3, 3]
    text = A.review(old, A.judge(current, now, FILES, update=True)[1], now, {"Q.Defs": tmp_path / "Defs.lean"})
    assert text == [f"guarantee Q.t: moved from P.t: {A.MOVED}", "definition Q.f (Q.Defs), was P.f (P.Defs), read by Q.t: changed",
                    f"  now ({os.path.relpath(tmp_path / 'Defs.lean')}:3-3):",
                    "    def f (n : Nat) : Nat := n * 1"]
    assert A.statement_changes(text) == text[1:]
    now = facts(pins={"Q.t": moved_fact(as_was="00000000000000bb")})
    text = A.review(old, A.judge(current, now, FILES, update=True)[1], now, {})
    assert text[:2] == ["guarantee Q.t: changed (moved from P.t)", "  before:"] and A.statement_changes(text) == text[:-1]


def test_a_definition_the_base_records_under_its_new_name_is_compared_under_it():
    """A base lock re-recorded after the move: `--moved` changes nothing in it, and a changed definition is shown once, by the
    name the base records."""
    now = facts(pins={"Q.t": moved_fact()})
    base = A.judge({**POLICY, "guarantees": {"Q.t": {}}}, now, FILES, update=True)[1]
    assert "Q.f" in base["reads"]["Q.Defs"]["definitions"]
    assert A.review(base, A.judge(base, now, FILES, update=True)[1], now, {}) == []
    now = facts(pins={"Q.t": moved_fact(f="9" * 32, f_was="9" * 32)})
    text = A.review(base, A.judge(base, now, FILES, update=True)[1], now, {})
    assert text == ["definition Q.f (Q.Defs), read by Q.t: changed"]


def test_a_moved_statement_lifted_into_a_guarantees_definition_is_the_same_statement():
    old = recorded()
    current = {**POLICY, "guarantees": {"Q.t": old["guarantees"]["P.t"]}}
    head = {"name": "Q.T", "module": "Q.Guarantees", "kind": "def", "hash": "5" * 32, "moved": {"as": "Q.T", "sha256": sha("5" * 32)}}
    now = facts(pins={"Q.t": moved_fact(head="Q.T", extra=[head])})
    new = A.judge(current, now, FILES, update=True)[1]
    assert new["guarantees"]["Q.t"]["signature"] == ["Q.t (h : P.Hyp) (n : Nat) : Q.T n"] and "Q.Guarantees" in new["reads"]
    text = A.review(old, new, now, {})
    assert text == [f"guarantee Q.t: moved from P.t, lifted into Q.T: {A.MOVED}", f"reads: 1 {A.MOVED_READS}"]
    now = facts(pins={"Q.t": moved_fact(head="Q.T", extra=[head], as_was="00000000000000bb")})
    text = A.review(old, A.judge(current, now, FILES, update=True)[1], now, {})
    assert text[0] == "guarantee Q.t: changed (moved from P.t)" and "definition Q.T (Q.Guarantees), read by Q.t: new" in text


def lifted_fact(unfolded="00000000000000aa", module="P.Guarantees", hyps=None, name="P.t"):
    """`P.t` lifted into `P.T` (in ``module``): `theorem P.t (h : P.Hyp) (n : Nat) : P.T n`, whose type with `P.T` unfolded
    once hashes like ``unfolded``, as `Facts.lean` reports it without `--moved`."""
    head = {"name": "P.T", "module": module, "kind": "def", "hash": "5" * 32}
    p = fact(signature=f"{name} (h : P.Hyp) (n : Nat) : P.T n", h="00000000000000bb", reads=fact()["reads"] + [head], hyps=hyps)
    p["reads"][0] |= {"lines": None}
    return p | {"lift": {"head": "P.T", "module": module, "unfolded_sha256": sha(unfolded)}}


def test_a_guarantee_lifted_into_a_guarantees_definition_is_the_same_statement():
    old = recorded()
    now = facts(pins={"P.t": lifted_fact()})
    new = A.judge(old, now, FILES, update=True)[1]
    assert new["guarantees"]["P.t"]["type_hash"] == sha("00000000000000bb") and "P.Guarantees" in new["reads"]
    text = A.review(old, new, now, {})
    assert text == [f"guarantee P.t: lifted into P.T: {A.LIFTED}"] and A.statement_changes(text) == []
    assert A.lifted_into(old["guarantees"]["P.t"], new["guarantees"]["P.t"], now["pins"]["P.t"], {}) == "P.T"


def test_a_lift_the_audit_cannot_certify_is_a_change_of_meaning():
    """The unfolded definition says something else, it is not in a `Guarantees` module, or the lift took a named assumption
    into the definition; and a lift's definition that another, changed guarantee also reads is shown in full."""
    old = recorded()
    for now in (facts(pins={"P.t": lifted_fact(unfolded="00000000000000cc")}),
                facts(pins={"P.t": lifted_fact(module="P.Defs")}), facts(pins={"P.t": lifted_fact(hyps=[])})):
        text = A.review(old, A.judge(old, now, FILES, update=True)[1], now, {})
        assert text[:2] == ["guarantee P.t: changed", "  before:"] and A.statement_changes(text) == text
        module = now["pins"]["P.t"]["lift"]["module"]
        assert f"definition P.T ({module}), read by P.t: new" in text
    assert "  named assumptions: P.Hyp -> none" in text
    two = facts(pins={"P.t": fact(), "P.u": fact(signature="P.u : True", h="00000000000000dd")})
    old = A.judge(POLICY | {"guarantees": {"P.t": {"owner": "@lean"}, "P.u": {"owner": "@lean"}}}, two, FILES, update=True)[1]
    now = facts(pins={"P.t": lifted_fact(), "P.u": lifted_fact(name="P.u", unfolded="00000000000000ee")})
    text = A.review(old, A.judge(old, now, FILES, update=True)[1], now, {})
    assert text[0] == f"guarantee P.t: lifted into P.T: {A.LIFTED}" and text[1] == "guarantee P.u: changed"
    assert "definition P.T (P.Guarantees), read by P.t, P.u: new" in text


def test_a_map_that_cannot_tell_declarations_apart_compares_nothing_under_the_old_names():
    old = recorded()
    current = {**POLICY, "guarantees": {"Q.t": old["guarantees"]["P.t"]}}
    clash = [{"name": "Q.f", "as": "P.f", "with": "P.f"}]
    now = facts(pins={"Q.t": moved_fact()}, moved_conflicts=clash)
    fails, new = A.judge(current, now, FILES, update=True)
    assert fails == ["moved: under the old names, Q.f would be P.f, and so would P.f: the map cannot tell them apart, so the "
                     "review compares nothing under the old names; map only the declarations that moved"]
    assert A.review(old, new, now, {})[:2] == ["guarantee P.t: removed", "  before:"]


def test_a_move_map_renames_by_its_longest_prefix_and_must_map_back():
    moved = {"A": "X", "A.b": "Y"}
    assert [A.renamed(n, moved) for n in ("A", "A.b.c", "A.d", "Ab.c", "Z")] == ["X", "Y.c", "X.d", "Ab.c", "Z"]
    assert A.move_errors(moved) == [] and A.move_errors({"A": "X", "A.y": "X.y"}) == []
    assert A.move_errors({"A": "A", "B": "X", "C": "X"}) == ["A moves to itself", "B, C all move to X"]
    assert A.move_errors({"A": "X", "B": "X.y"}) == [
        "B moves to X.y, under X, where A moves; the names under X.y would map back under A, not B"]


def test_a_closed_hypothesis_must_be_a_named_assumption():
    inline = [{"binder": "h", "type": "∀ (m : Nat), m + 0 = m", "closed": True, "named": None, "module": None},
              {"binder": "hn", "type": "n ≠ 0", "closed": False, "named": None, "module": None}]
    fails, _ = A.judge(POLICY, facts(pins={"P.t": fact(hyps=inline)}), FILES, update=True)
    assert fails == ["guarantees: P.t assumes (h : ∀ (m : Nat), m + 0 = m), a closed Prop that is not a named assumption: a "
                     "Prop constant listed under `assumptions`"]
    elsewhere = [{"binder": "h", "type": "Q.Hyp", "closed": True, "named": "Q.Hyp", "module": "Q.Stuff"}]
    assert A.judge(POLICY, facts(pins={"P.t": fact(hyps=elsewhere)}), FILES, update=True)[0][0].startswith("guarantees: P.t assumes (h : Q.Hyp)")


def test_a_premise_the_lock_does_not_list_fails():
    """A statement that takes an assumption applied to its own variables (`h : Premises.Hyp n`, an open hypothesis the
    hypotheses rule never names; `Facts.lean`'s `premises` sees through the abbrev and the `Guarantees` definitions to
    `P.Hyp`) fails while the lock doesn't list it, and an update adds it with the guarantee's owner; on 14 Oct a merge
    dropped `Flock.Assumptions.RegistrantCovers`' entry this way and the check passed. Without an import graph to say
    which modules the repository builds, a premise outside an assumptions module is not counted (the next test has one)."""
    applied = [{"binder": "h", "type": "P.Premises.Hyp n", "closed": False, "named": None, "module": None}]
    pin = fact(signature="P.t (n : Nat) (h : P.Premises.Hyp n) : P.f n = n", hyps=applied)
    pin["premises"] = [{"binder": "h", "head": "P.Hyp", "module": "P.Assumptions"},
                       {"binder": "hn", "head": "P.f", "module": "P.Defs"}]
    f = facts(pins={"P.t": pin})
    fails, pol = A.judge({**POLICY, "assumptions": {}}, f, FILES, update=True)
    assert fails == [] and pol["assumptions"] == {
        "P.Hyp": {"owner": "@lean", "digest": pol["reads"]["P.Assumptions"]["definitions"]["P.Hyp"]}}
    assert A.judge(pol, f, FILES)[0] == []
    digest = pol["assumptions"]["P.Hyp"]["digest"]
    assert A.judge({**pol, "assumptions": {}}, f, FILES)[0] == [
        "assumptions: P.Hyp is a premise of P.t, an assumption the lock doesn't list; run `check.py --update` to add it "
        "with its guarantees' owner and commit the record (Daniel's DM shows it when it lands), or write \"P.Hyp\": "
        f"{{\"owner\": \"@lean\", \"digest\": \"{digest}\"}} under `assumptions`."]


def test_a_premise_defined_anywhere_the_repository_builds_is_listed_and_a_dependencys_is_not():
    """LEAN-37 (RT2-LOGIC-2): `ProofLayer.SoundAt`, a premise defined in `Definitions/Accounting/TS`, passed unlisted, since
    only a head in an assumptions module counted. A head in any module whose `.olean` a package of the repository builds
    is listed now, in a lifted guarantee too (its premises are read through the `Guarantees` definition); one from the
    toolchain or a fetched dependency (`Nat.le`, Mathlib's) is not."""
    pin = fact(signature="P.t (n : Nat) : P.Guarantees.t n", hyps=[])
    pin["premises"] = [{"binder": "h", "head": "P.Sound", "module": "P.Defs"},
                       {"binder": "hn", "head": "Nat.le", "module": "Init.Prelude"},
                       {"binder": "hd", "head": "Disjoint", "module": "Mathlib.Order.Disjoint"}]
    graph = {"P.Defs": {"imports": ["Init"], "olean": "/r/Security/.lake/build/lib/lean/P/Defs.olean"},
             "Init.Prelude": {"imports": [], "olean": "/elan/toolchains/v4/lib/lean/Init/Prelude.olean"},
             "Mathlib.Order.Disjoint": {"imports": [], "olean": "/r/Security/.lake/packages/mathlib/.lake/build/lib/lean/"
                                                                 "Mathlib/Order/Disjoint.olean"}}
    f = facts(pins={"P.t": pin}, graph=graph)
    assert A.own_modules(f) == {"P.Defs"}
    fails, pol = A.judge({**POLICY, "assumptions": {}}, f, FILES, update=True)
    assert fails == [] and list(pol["assumptions"]) == ["P.Sound"] and pol["assumptions"]["P.Sound"]["owner"] == "@lean"
    assert A.judge(pol, f, FILES)[0] == [] and A.judge(pol, f, FILES, update=True)[1]["assumptions"] == pol["assumptions"]
    assert A.judge({**pol, "assumptions": {}}, f, FILES)[0] == [
        "assumptions: P.Sound is a premise of P.t, an assumption the lock doesn't list; run `check.py --update` to add it "
        "with its guarantees' owner and commit the record (Daniel's DM shows it when it lands), or write \"P.Sound\": "
        "{\"owner\": \"@lean\"} under `assumptions`."]


def asserted(head, module="P.Assumptions", props=False):
    return {"head": head, "module": module, "props": props}


def aliased(unfolds: dict, closed: bool = False) -> dict:
    """Facts whose guarantee `P.t` takes `P.Both`, a definition of `P.Defs` (a module the repository builds) that unfolds as
    ``unfolds`` says (`Facts.lean`'s `unfolds`), as an open hypothesis or, ``closed``, as a closed one."""
    hyps = [{"binder": "h", "type": "P.Both", "closed": True, "named": "P.Both", "module": "P.Defs"}]
    pin = fact(signature="P.t (h : P.Both) (n : Nat) : P.f n = n", hyps=hyps) if closed else fact()
    pin["premises"] = [{"binder": "h", "head": "P.Both", "module": "P.Defs", "unfolds": unfolds}]
    graph = {"P.Defs": {"imports": ["Init"], "olean": "/r/Security/.lake/build/lib/lean/P/Defs.olean"}}
    return facts(pins={"P.t": pin}, graph=graph)


def test_an_alias_of_listed_assumptions_is_no_new_named_assumption():
    """TOP-90 row 17 (#4440, #4358): a premise that only packs listed assumptions (`def Both n := Hyp ∧ Hyp2 n`, or through
    another such definition) failed as a new named assumption, and listing or renaming it took another records run. It is
    not one: the check holds its parts to the lock in its place (so a listed part reached only through it is read), an
    update doesn't add it, and taken closed, its parts are the guarantee's named assumptions."""
    listed = {**POLICY, "assumptions": {"P.Hyp": {"owner": "@lean"}, "P.Hyp2": {"owner": "@lean"}}}
    for unfolds in ({"P.Both": [asserted("P.Hyp"), asserted("P.Hyp2")]},
                    {"P.Both": [asserted("P.Inner", "P.Defs"), asserted("P.Hyp")], "P.Inner": [asserted("P.Hyp2")]}):
        fails, pol = A.judge(listed, aliased(unfolds), FILES, update=True)
        assert fails == [] and set(pol["assumptions"]) == {"P.Hyp", "P.Hyp2"}, (fails, pol["assumptions"])
        assert A.judge(pol, aliased(unfolds), FILES)[0] == []
    closed = aliased({"P.Both": [asserted("P.Hyp"), asserted("P.Hyp2")]}, closed=True)
    fails, pol = A.judge(listed, closed, FILES, update=True)
    assert fails == [] and pol["guarantees"]["P.t"]["assumptions"] == ["P.Hyp", "P.Hyp2"]
    assert A.judge(pol, closed, FILES)[0] == []


def test_an_alias_hiding_anything_but_listed_assumptions_is_a_new_one_and_its_failure_names_the_entry():
    """Fail-closed: a premise that unfolds to anything but listed assumptions (one the lock doesn't list, `False`, a bound
    predicate, a constant applied to a proposition or a predicate, a part that doesn't unfold) stays a new named assumption,
    taken open or closed; its failure says what it unfolds to and the entry an update would write, so one edit settles it."""
    listed = {**POLICY, "assumptions": {"P.Hyp": {"owner": "@lean"}}}
    pol = A.judge(listed, aliased({"P.Both": [asserted("P.Hyp")]}), FILES, update=True)[1]
    cases = {"P.Hidden": [asserted("P.Hyp"), asserted("P.Hidden")],
             "False": [asserted("P.Hyp"), asserted("False", "Init.Prelude")],
             "a bound predicate": [asserted("P.Hyp"), asserted(None, None)],
             "P.Hyp applied to a proposition or predicate": [asserted("P.Hyp", props=True)],
             "P.Inner": [asserted("P.Hyp"), asserted("P.Inner", "P.Defs")]}
    for why, parts in cases.items():
        got = aliased({"P.Both": parts})
        fails = [f for f in A.judge(pol, got, FILES)[0] if f.code == "assumptions.unlisted-premise"]
        beside = " beside the listed P.Hyp" if why != "P.Hyp applied to a proposition or predicate" else ""
        assert fails == [
            "assumptions: P.Both is a premise of P.t, an assumption the lock doesn't list; run `check.py --update` to add it "
            "with its guarantees' owner and commit the record (Daniel's DM shows it when it lands), or write \"P.Both\": "
            f"{{\"owner\": \"@lean\"}} under `assumptions`. It unfolds to {why}{beside}, so it is no alias of listed "
            "assumptions, which the lock would not count."], (why, fails)
        assert "P.Both" in A.judge(listed, got, FILES, update=True)[1]["assumptions"]
        assert A.judge({**pol, "assumptions": pol["assumptions"] | {"P.Both": {"owner": "@lean"}}}, got, FILES)[0] == []
        closed = aliased({"P.Both": parts}, closed=True)
        assert "guarantees: P.t assumes (h : P.Both), a closed Prop" in "".join(A.judge(listed, closed, FILES, update=True)[0])
    hidden = aliased({"P.Both": cases["P.Hidden"]})
    hidden["pins"]["P.u"] = fact() | {"premises": [{"binder": "h", "head": "P.Hidden", "module": "P.Assumptions"}]}
    fails, pol = A.judge({**listed, "guarantees": {"P.t": {"owner": "@lean"}, "P.u": {"owner": "@lean"}}}, hidden, FILES,
                         update=True)
    assert fails == [] and set(pol["assumptions"]) == {"P.Hyp", "P.Hidden"} and A.judge(pol, hidden, FILES)[0] == []


ALIASES = """import Control.Assumptions

namespace Control

def Both (n : Nat) : Prop := A n ∧ B n
abbrev BothR (n : Nat) : Prop := A n ∧ B n
abbrev AR (n : Nat) : Prop := A n
def Nested (n : Nat) : Prop := ∀ m, m ≤ n → Both m ∧ AR m
def Closed : Prop := ∀ n, Both n
def Sneaky (n : Nat) : Prop := A n ∧ Hidden n
abbrev SneakyR (n : Nat) : Prop := A n ∧ Hidden n
def Absurd (n : Nat) : Prop := A n ∧ False
def AllR (p : Nat → Prop) : List Nat → Prop
  | [] => True
  | x :: xs => p x ∧ AllR p xs
def Rec (l : List Nat) : Prop := AllR A l

theorem t_both (n : Nat) (_h : Both n) : True := trivial
theorem t_bothR (n : Nat) (_h : BothR n) : True := trivial
theorem t_nested (n : Nat) (_h : Nested n) : True := trivial
theorem t_closed (_h : Closed) : True := trivial
theorem t_sneaky (n : Nat) (_h : Sneaky n) : True := trivial
theorem t_sneakyR (n : Nat) (_h : SneakyR n) : True := trivial
theorem t_absurd (n : Nat) (_h : Absurd n) : True := trivial
theorem t_rec (l : List Nat) (_h : Rec l) : True := trivial

end Control
"""


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_aliases_of_listed_assumptions_pass_and_aliases_hiding_others_fail_on_a_real_build(tmp_path):
    """`Facts.lean`'s `unfolds` on a real build, with `A` and `B` listed: a `def` or `abbrev` alias of them, one nested
    through another and an abbrev, and one taken closed are no new named assumptions; one that hides the unlisted `Hidden`,
    `False` or a recursive definition applied to a predicate is, and so is `Hidden` itself behind an abbrev, which `Facts.lean`
    read as `And` and dropped. An update lists `Hidden`, which makes `Sneaky` an alias, and the record of `t_closed` names
    `A` and `B`."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    src = pkg / "Control" / "Assumptions.lean"
    src.write_text(src.read_text().replace("end Control", "def A (n : Nat) : Prop := n + 0 = n\ndef B (n : Nat) : Prop := "
                                           "0 + n = n\ndef Hidden (n : Nat) : Prop := n * 1 = n\n\nend Control"))
    (pkg / "Control" / "Aliases.lean").write_text(ALIASES)
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.Aliases\n")
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    names = ["t_both", "t_bothR", "t_nested", "t_closed", "t_sneaky", "t_sneakyR", "t_absurd", "t_rec"]
    policy = unclaimed(A.load(pkg))
    policy["guarantees"] = policy["guarantees"] | {f"Control.{n}": {"owner": "@lean"} for n in names}
    policy["assumptions"] = policy["assumptions"] | {"Control.A": {"owner": "@lean"}, "Control.B": {"owner": "@lean"}}
    f = A.facts_of(pkg, policy, tmp_path)
    files = A.lean_modules(pkg)
    new = lambda fails: sorted(x.split(" ")[1] for x in fails if x.code in ("assumptions.unlisted-premise",
                                                                           "guarantees.unnamed-hypothesis"))
    assert new(A.judge(policy, f, files)[0]) == ["Control.Absurd", "Control.Hidden", "Control.Rec", "Control.Sneaky"]
    fails, pol = A.judge(policy, f, files, update=True)
    assert new(fails) == [] and sorted(set(pol["assumptions"]) - set(policy["assumptions"])) == [
        "Control.Absurd", "Control.Hidden", "Control.Rec"]
    assert pol["guarantees"]["Control.t_closed"]["assumptions"] == ["Control.A", "Control.B"]
    assert new(A.judge(pol, A.facts_of(pkg, pol, tmp_path), files)[0]) == []


def test_a_premise_no_reads_record_holds_is_digested_with_what_it_reads():
    """LEAN-60: a premise whose module is outside `meaning` (a Proofs module) has no `reads` record, and its entry was an
    owner alone, so #4641's strengthened `RecDeploy.PlainLeavesAdmitted` changed no record. Its digest is now its walk's
    (`Facts.lean`'s `assumption_reads`: it and what it reads): a change to either fails under its name, and an update
    writes it, which the review prints."""
    pin = fact(signature="P.t (n : Nat) (h : P.Leaves n) : P.f n = n", hyps=[])
    pin["premises"] = [{"binder": "h", "head": "P.Leaves", "module": "P.Thms"}]
    graph = {"P.Thms": {"imports": ["Init"], "olean": "/r/Security/Proofs/.lake/build/lib/lean/P/Thms.olean"}}
    walk = lambda bound: {"P.Leaves": [{"name": "P.Leaves", "module": "P.Thms", "kind": "def", "sha256": sha("leaves")},
                                       {"name": "P.bound", "module": "P.Thms", "kind": "def", "sha256": sha(bound)}]}
    f = facts(pins={"P.t": pin}, graph=graph, assumption_reads=walk("n + 1"))
    fails, pol = A.judge({**POLICY, "assumptions": {}}, f, FILES, update=True)
    assert fails == [] and pol["assumptions"] == {"P.Leaves": {"owner": "@lean", "digest": A.digest(walk("n + 1")["P.Leaves"])}}
    assert A.judge(pol, f, FILES)[0] == [] and A.judge(pol, f, FILES, update=True)[1]["assumptions"] == pol["assumptions"]
    moved = facts(pins={"P.t": pin}, graph=graph, assumption_reads=walk("n + 2"))
    assert A.judge(pol, moved, FILES)[0] == [
        "assumptions: P.Leaves's definition in P.Thms, or a definition it reads, changed; run `check.py --update` and commit "
        "the record (Daniel's DM shows the change when it lands)"]
    new = A.judge(pol, moved, FILES, update=True)[1]
    assert A.assumption_changes(pol, new) == ["assumption P.Leaves: changed (its statement or a definition it reads)"]
    bare = {**pol, "assumptions": {"P.Leaves": {"owner": "@lean"}}}
    assert A.judge(bare, f, FILES)[0][0].startswith("assumptions: P.Leaves has no record")
    assert A.assumption_changes(bare, pol) == ["assumption P.Leaves: recorded: its statement and the definitions it reads, "
                                               "digested"]
    assert A.judge(bare, facts(pins={"P.t": pin}, graph=graph), FILES)[0] == [], "older facts, without the walks"


LEAVES = "import Control.LeafBound\nnamespace Control\ndef Leaves (n : Nat) : Prop := n < leafBound n\nend Control\n"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_premise_outside_meaning_changes_its_record_when_it_or_what_it_reads_changes(tmp_path):
    """LEAN-60 on a real build: a premise defined outside `meaning` (`Control.Leaves`, as `RecDeploy.PlainLeavesAdmitted`
    is in a Proofs module), reading a definition of another such module (`leafBound`), gets a digest; a rebuild that
    changes neither keeps the lock byte for byte, and a change to the definition it reads, or to its own statement, changes
    its record and nothing else, which the audit refuses until `--update` records it."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    bound = pkg / "Control" / "LeafBound.lean"
    leaves = pkg / "Control" / "Leaves.lean"
    bound.write_text("namespace Control\ndef leafBound (n : Nat) : Nat := n + 1\nend Control\n")
    leaves.write_text(LEAVES)
    (pkg / "Control" / "LeafThm.lean").write_text(
        "import Control.Leaves\nnamespace Control\ntheorem leaf (n : Nat) (_h : Leaves n) : n < n + 1 := Nat.lt_succ_self n\n"
        "end Control\n")
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.LeafThm\n")
    pol = A.load(pkg)
    A.dump(pkg, pol | {"meaning": ["Control.Assumptions", "Control.Defs", "Control.Thms"],
                       "reads_exempt": {"Control.Leaves": "the premise outside `meaning` this test digests"},
                       "guarantees": pol["guarantees"] | {"Control.leaf": {"owner": "@lean"}}})
    build = lambda: subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    audit = lambda update: A.audit(pkg, tmp_path / "out", update=update, replay=False, runs=False, log=lambda *_: None)
    build()
    assert unshaped(audit(True)) == []
    first = (pkg / A.POLICY).read_text()
    rec = json.loads(first)["assumptions"]
    assert set(rec) == {"Control.Hyp", "Control.Leaves"} and rec["Control.Leaves"]["owner"] == "@lean"
    assert len(rec["Control.Leaves"]["digest"]) == 64 and "Control.Leaves" not in json.dumps(json.loads(first)["reads"])
    build()
    assert unshaped(audit(False)) == [] and unshaped(audit(True)) == [] and (pkg / A.POLICY).read_text() == first
    seen = {rec["Control.Leaves"]["digest"]}
    for path, text in ((bound, "namespace Control\ndef leafBound (n : Nat) : Nat := n + 2\nend Control\n"),
                       (leaves, LEAVES.replace("n < leafBound n", "n + 1 < leafBound n"))):
        before = json.loads((pkg / A.POLICY).read_text())
        path.write_text(text)
        build()
        assert [f.code for f in audit(False)["failures"] if not f.code.startswith("shape")] == ["assumptions.digest"]
        rep = audit(True)
        after = json.loads((pkg / A.POLICY).read_text())
        assert unshaped(rep) == [] and A.statement_changes(rep["review"]) == [
            "assumption Control.Leaves: changed (its statement or a definition it reads)"]
        assert after["assumptions"]["Control.Hyp"] == before["assumptions"]["Control.Hyp"]
        assert after["assumptions"]["Control.Leaves"]["digest"] not in seen
        seen.add(after["assumptions"]["Control.Leaves"]["digest"])
        after["assumptions"]["Control.Leaves"] = before["assumptions"]["Control.Leaves"]
        assert after == before, "the premise's record changed, and nothing else"


def test_a_statement_whose_premises_could_not_be_read_fails():
    """`Facts.lean` writes `premises: null` when `premisesOf` throws on a statement; read as no premises, every premise of it
    would go unlisted, so it fails instead. Facts without the field (older documents) are not held to it."""
    pin = fact() | {"premises": None}
    fails, _ = A.judge(POLICY, facts(pins={"P.t": pin}), FILES, update=True)
    assert fails == ["guarantees: P.t's premises could not be read (`Facts.lean`'s `premisesOf` failed on its statement), so "
                     "the lock can't be held to them: make `premisesOf` read this statement's form"]
    assert A.judge(POLICY, facts(), FILES, update=True)[0] == []


def test_the_verifiers_path_has_no_unlisted_compiled_code_and_hashes_through_its_parameter():
    """Daniel's checks (a) and (b) on the verifier's path (`verifier` in the policy): an `@[extern]` or `@[implemented_by]`
    constant its entries reach fails unless `allow` names it (a name prefix, or `module:` for a module), and so does each
    entry into the `hash` namespace that bypasses `stop`, naming its path; a missing entry is named; no rule, no checks."""
    path = {"missing": ["P.gone"], "reached": 40,
            "implemented_by": [{"name": "P.fast", "module": "P.Defs", "by": "P.fastImpl"}],
            "extern": [{"name": "Nat.add", "module": "Init.Prelude"}, {"name": "ByteArray.push", "module": "Init.Data.ByteArray"}],
            "hash": [{"name": "P.Sha512.hash", "module": "P.Hash", "path": ["P.verify", "P.tag", "P.Sha512.hash"]}]}
    rule = {"entries": ["P.verify", "P.gone"], "stop": ["P.key"], "hash": "P.Sha512", "allow": ["Nat.add", "module:Init.Data"]}
    fails = A.verifier_failures(rule, path)
    assert fails == [
        "verifier: entry P.gone is not a constant of the build; fix `verifier.entries`",
        "verifier: P.fast (@[implemented_by], in P.Defs) is on the path from P.verify, P.gone: compiled code no proof sees; "
        "if it is a core primitive, run `check.py --update`, which lists it under `verifier.allow` (the landing DM shows "
        "it), else take it off the path",
        "verifier: P.verify, P.gone reach P.Sha512.hash outside the hash parameter, by P.verify -> P.tag -> P.Sha512.hash; "
        "route it through the parameter, or add the public constant it serves to `verifier.stop`"]
    assert A.verifier_failures(None, path) == []
    assert A.verifier_failures(rule, None)[0].startswith("verifier: `Facts.lean` reported no verifier path")
    f = facts(); f["verifier"] = {**path, "missing": [], "implemented_by": [], "hash": []}
    assert A.judge({**POLICY, "verifier": rule}, f, FILES, update=True)[0] == []


def test_verifier_allow_is_a_record_of_the_compiled_code_on_the_path_by_name():
    """LEAN-33: `verifier.allow` lists by name each `@[extern]` and `@[implemented_by]` constant the verifier reaches, as
    `--update` writes it, so the landing DM shows one added: an Init extern the list doesn't name fails (no `module:Init`
    admitting all of Init), a name that admits nothing on the path fails, and a `module:` entry still admits its modules."""
    path = {"missing": [], "reached": 40, "hash": [], "csimp": [],
            "implemented_by": [{"name": "Array.foldlM", "module": "Init.Data.Array.Basic", "by": "Array.foldlMUnsafe"}],
            "extern": [{"name": "Nat.add", "module": "Init.Prelude"}, {"name": "ByteArray.push", "module": "Init.Data.ByteArray.Basic"}]}
    rule = {"entries": ["P.verify"], "stop": [], "hash": "P.Sha512", "allow": ["Array.foldlM", "Nat.add"]}
    assert A.verifier_failures(rule, path) == [
        "verifier: ByteArray.push (@[extern], in Init.Data.ByteArray.Basic) is on the path from P.verify: compiled code no "
        "proof sees; if it is a core primitive, run `check.py --update`, which lists it under `verifier.allow` (the landing "
        "DM shows it), else take it off the path"]
    assert A.verifier_failures({**rule, "allow": [*rule["allow"], "ByteArray.push", "UInt64.add"]}, path) == [
        "verifier: `verifier.allow` lists UInt64.add, which admits nothing on the path from P.verify; run `check.py --update`"]
    assert A.verifier_failures({**rule, "allow": ["module:Init"]}, path) == []
    assert A.verifier_failures({**rule, "allow": ["module:Init.Data"]}, path)[0].startswith("verifier: Nat.add (@[extern]")
    f = facts(); f["verifier"] = path
    fails, pol = A.judge({**POLICY, "verifier": rule}, f, FILES, update=True)
    assert fails == [] and pol["verifier"] == {**rule, "allow": ["Array.foldlM", "ByteArray.push", "Nat.add"]}
    assert A.verifier_failures(pol["verifier"], path) == []
    kept = A.judge({**POLICY, "verifier": {**rule, "allow": ["module:Init.Data"]}}, f, FILES, update=True)[1]["verifier"]
    assert kept["allow"] == ["module:Init.Data", "Nat.add"]
    notes = A.review({"verifier": rule}, {"verifier": pol["verifier"]}, {"pins": {}, "verifier": path}, {})
    assert notes == ["verifier: allowed ByteArray.push (@[extern], in Init.Data.ByteArray.Basic): compiled code no proof "
                     "sees, now on the verifier's path"]
    assert A.statement_changes(notes) == notes


def test_every_claim_has_a_witness_that_it_fails_and_witnesses_that_its_antecedents_are_met():
    """LEAN-26: each predicate the spec declares under `claims.modules` names a `fails` theorem stating `¬ C …` with that
    claim at the head, and `meets` theorems that are not `C` itself, each a guarantee; one without witnesses fails unless
    `claims.not` gives the reason, and a named assumption is no claim."""
    def spec(name, predicate=True):
        return {"name": name, "module": "P.Claims", "kind": "def", "statement": False, "predicate": predicate}

    def thm(head, negated):
        conclusion = [{"head": head, "defs": [head], "refl": False, "assumed": False}]
        return fact() | {"kind": "theorem", "body": {"head": head, "negated": negated, "conclusion": conclusion, "absurd": False}}
    defs = [spec("P.KS"), spec("P.Complete"), spec("P.Helper"), spec("P.Hyp"), spec("P.size", predicate=False)]
    pins = {"P.t": fact(), "P.KS_fails": thm("P.KS", True), "P.KS_meets": thm("P.ListWithin", False),
            "P.Complete_fails": thm("P.KS", True), "P.Complete_meets": thm("P.Complete", False)}
    rule = {"modules": ["P.Claims"], "not": {"P.Helper": "a collision, which the claims read"},
            "witnesses": {"P.KS": {"fails": "P.KS_fails", "meets": ["P.KS_meets"]},
                          "P.Complete": {"fails": "P.Complete_fails", "meets": ["P.Complete_meets", "P.Complete_zk"]}}}
    policy = {**POLICY, "claims": rule, "guarantees": {n: {"owner": "@lean"} for n in pins}}
    f = facts(pins=pins, spec_definitions=defs)
    assert A.claim_failures(policy, f) == [
        "claims: P.Complete_fails does not state that P.Complete fails: past its ∀ and ∃ binders its statement must be "
        "`¬ P.Complete …`, and it is ¬ P.KS …",
        "claims: P.Complete_meets states P.Complete itself: a `meets` witness states the claim's checkable antecedents at a "
        "deployed record (a decoder meeting its bound, a non-empty scope), not the claim",
        "claims: P.Complete_zk, P.Complete's `meets` witness, is not under `guarantees`: list it there, so its record shows "
        "in the landing DM"]
    assert A.claim_failures({**policy, "claims": {**rule, "not": {}}}, f)[0].startswith(
        "claims: P.Helper (P.Claims) is a claim definition with no witnesses: name its `P.Helper_fails`")
    good = {**rule, "witnesses": {"P.KS": rule["witnesses"]["P.KS"]}, "not": {**rule["not"], "P.Complete": "not yet stated"}}
    assert A.claim_failures({**policy, "claims": good}, f) == []
    assert A.claim_failures({**policy, "claims": {**good, "not": {**good["not"], "P.Gone": "moved"}}}, f) == [
        "claims: P.Gone is listed under `claims` but is no predicate of the spec under P.Claims; remove the entry"]
    assert A.claim_failures(POLICY, f) == []
    assert A.claim_witnesses(policy) == {"P.KS_fails", "P.KS_meets", "P.Complete_fails", "P.Complete_meets", "P.Complete_zk"}


def test_a_claim_a_guarantee_takes_as_a_premise_keeps_its_claims_entry():
    """A predicate under `claims.modules` that is also a named assumption (a claim a guarantee takes as a premise, as
    `SampledLeakAccepted` takes `Claims.Sampling.KS`) keeps its `claims.witnesses` or `claims.not` entry, whose witnesses are
    still checked, and needs none; an entry naming no predicate is stale, and a claim that is no assumption needs one."""
    defs = [{"name": n, "module": "P.Claims", "kind": "def", "statement": False, "predicate": True}
            for n in ["P.KS", "P.Within", "P.Other"]]
    within = {"head": "P.Within", "defs": ["P.Within"], "refl": False, "assumed": False}
    pins = {"P.t": fact(), "P.KS_fails": fact() | {"body": {"head": "P.KS", "negated": True}},
            "P.KS_meets": fact() | {"body": {"head": "P.Within", "negated": False, "conclusion": [within], "absurd": False}}}
    rule = {"modules": ["P.Claims"], "witnesses": {"P.KS": {"fails": "P.KS_fails", "meets": ["P.KS_meets"]}},
            "not": {"P.Within": "a decoder's bound, which the claims read", "P.Other": "a helper"}}
    policy = {**POLICY, "assumptions": {**POLICY["assumptions"], "P.KS": {"owner": "@proofs"}, "P.Within": {"owner": "@proofs"}},
              "claims": rule, "guarantees": {n: {"owner": "@lean"} for n in pins}}
    f = facts(pins=pins, spec_definitions=defs)
    assert A.claim_failures(policy, f) == []
    assert A.claim_failures({**policy, "claims": {**rule, "not": {**rule["not"], "P.Gone": "moved"}}}, f) == [
        "claims: P.Gone is listed under `claims` but is no predicate of the spec under P.Claims; remove the entry"]
    assert A.claim_failures({**policy, "claims": {**rule, "witnesses": {}, "not": {"P.Other": "a helper"}}}, f) == []
    assert A.claim_failures({**policy, "claims": {**rule, "not": {"P.Within": rule["not"]["P.Within"]}}}, f) == [
        "claims: P.Other (P.Claims) is a claim definition with no witnesses: name its `P.Other_fails` (a theorem that it "
        "is false at a toy instance) and `P.Other_meets` (its checkable antecedents hold at each deployed record) under "
        "`claims.witnesses`, or list it under `claims.not` with the reason it is no claim"]
    swapped = {**rule, "witnesses": {"P.KS": {"fails": "P.KS_meets", "meets": ["P.KS_meets"]}}}
    assert A.claim_failures({**policy, "claims": swapped}, f) == [
        "claims: P.KS_meets does not state that P.KS fails: past its ∀ and ∃ binders its statement must be `¬ P.KS …`, "
        "and it is P.Within …"]


def test_a_meets_witness_says_something_about_the_deployment():
    """LEAN-32: a `meets` witness whose statement is `True`, a conjunction of `True`s, a reflexivity, one of its own
    hypotheses, a bound variable, a fact of Init alone, or anything under a `False` hypothesis fails; one whose every
    conjunct applies a definition of the spec (`∃ s w, InScope deploy s ∧ RelS s w`, `deploy.zk = true`) passes."""
    def k(head, defs=("P.deploy",), refl=False, assumed=False):
        return {"head": head, "defs": list(defs), "refl": refl, "assumed": assumed}

    def judged(*conclusion, absurd=False):
        pins = {"P.KS_fails": fact() | {"body": {"head": "P.KS", "negated": True}},
                "P.KS_meets": fact() | {"body": {"head": conclusion[0]["head"] if len(conclusion) == 1 else "And",
                                                 "negated": False, "conclusion": list(conclusion), "absurd": absurd}}}
        rule = {"modules": ["P.Claims"], "witnesses": {"P.KS": {"fails": "P.KS_fails", "meets": ["P.KS_meets"]}}}
        f = facts(pins=pins, spec_definitions=[{"name": "P.KS", "module": "P.Claims", "kind": "def", "predicate": True}])
        fails = A.claim_failures({**POLICY, "claims": rule, "guarantees": {n: {"owner": "@lean"} for n in pins}}, f)
        assert all(x.endswith("applies a definition of the spec (the deployed record, a predicate or a field of it)")
                   for x in fails)
        return [x.removeprefix("claims: P.KS_meets, P.KS's `meets` witness, ").partition(":")[0] for x in fails]

    assert judged(k("P.InScope", ["P.InScope", "P.deploy"]), k("P.RelS", ["P.RelS"])) == []
    assert judged(k("Eq", ["P.Deploy.zk", "P.deploy"])) == []
    assert judged(k("True", [])) == ["has a conjunct (True …) that applies no definition under `meaning` (`True`, or a "
                                     "fact of Init or Mathlib alone)"]
    assert judged(k("True", []), k("True", [])) == judged(k("True", []))
    assert judged(k("P.InScope"), k("True", [])) == judged(k("True", []))
    assert judged(k("LT.lt", [])) == ["has a conjunct (LT.lt …) that applies no definition under `meaning` (`True`, or a "
                                      "fact of Init or Mathlib alone)"]
    assert judged(k("Eq", refl=True)) == ["has a conjunct that is a reflexivity (Eq of two equal sides)"]
    assert judged(k("P.Ok", assumed=True)) == ["has a conjunct (P.Ok …) that is one of its own hypotheses"]
    assert judged(k(None)) == ["has a conjunct that applies a bound variable, not a constant"]
    assert judged(k("P.Ok"), absurd=True) == ["takes `False` as a hypothesis, so it proves anything"]
    assert judged(k("P.InScope"), k("P.KS")) == ["has a conjunct that states P.KS itself"]
    assert A.vacuous_meets({"head": "P.Ok", "negated": False}, "P.KS").startswith("has no conclusion")


def wk(head, defs=None, assumed=False):
    """A conjunct of a statement's conclusion, as `Facts.lean`'s `conclusionOf` reports it."""
    return {"head": head, "defs": list(defs if defs is not None else [head, "P.deploy"]), "refl": False, "assumed": assumed}


def wbody(head, negated=False, binders=(), conclusion=None, absurd=False, lifted=None):
    """A statement's `body`, as `Facts.lean` reports it."""
    return {"head": head, "negated": negated, "binders": list(binders),
            "conclusion": [wk(head)] if conclusion is None else conclusion, "absurd": absurd, "lifted": lifted}


HOLDS = {"name": "P.Holds", "module": "P.Claims", "kind": "def", "hash": "4" * 32}
COMPILED = {"binder": "hc", "head": "P.Compiled", "module": "P.Defs"}
WOWN = {"P", "P.Assumptions", "P.Claims", "P.Defs", "P.Thms"}


def witnessed(guarantee=None, honest=None, nontrivial=None, **policy):
    """(policy, facts) of `P.sound : ∀ d, P.Compiled d → P.Hyp → P.Holds d`, which reads the claim `P.Holds`, with its
    `honest` witness `P.sound_honest : P.Compiled P.deploy` and its `nontrivial` witness `P.sound_nontrivial : ¬ P.Holds
    P.acceptAll`, each part overridden by the arguments."""
    pins = {"P.sound": fact(reads=[HOLDS]) | {"premises": [COMPILED], "body": wbody("P.Holds", binders=["∀", "→", "→"])}
            | (guarantee or {}),
            "P.sound_honest": fact(hyps=[]) | {"body": wbody("P.Compiled")} | (honest or {}),
            "P.sound_nontrivial": fact(hyps=[]) | {"body": wbody("P.Holds", negated=True)} | (nontrivial or {})}
    pol = {**POLICY, "guarantees": {n: {"owner": "@lean"} for n in pins},
           "witnesses": {"P.sound": {"honest": ["P.sound_honest"], "nontrivial": "P.sound_nontrivial"}}} | policy
    return pol, facts(pins=pins, own_modules=sorted(WOWN))


def witness_codes(pol, f):
    return [x.code for x in A.witness_failures(pol, f, WOWN)]


def test_a_guarantee_of_what_an_acceptance_means_has_an_honest_and_a_nontrivial_witness():
    """A guarantee whose statement reads a claim names, under `witnesses`, `honest` theorems concluding its premises
    at the deployed instance and a `nontrivial` theorem that its conclusion fails at the verifier that accepts everything,
    each a guarantee; without them it fails, naming both and their statements' shapes. The witnesses are no guarantees the
    rule holds to witnesses of their own, and a guarantee that reads no claim needs none."""
    pol, f = witnessed()
    assert A.witness_failures(pol, f, WOWN) == []
    assert A.claim_witnesses(pol) == {"P.sound_honest", "P.sound_nontrivial"}
    assert A.witness_failures({**pol, "witnesses": {}}, f, WOWN) == [
        "witnesses: P.sound lists no witnesses that it says something, and the rule holds it to them (it reads the claim "
        "P.Holds): prove `P.sound_honest : P.Compiled d …` (its hypotheses hold at the honest deployed instance d, at the "
        "deployed verifier, taking only named assumptions of an `Assumptions` or `shape.premises` module) and "
        "`P.sound_nontrivial : ¬ P.Holds at acceptAll …` (its conclusion is false for a verifier that accepts "
        "everything). Each takes no binder over its instance and no hypothesis beyond named assumptions the lock lists: "
        "it builds the instance at the model level (a small hand-built descriptor, not a parsed file), with any one field "
        "that can't be built a named assumption. List both under `guarantees`, and write \"P.sound\": {\"honest\": "
        "[\"P.sound_honest\"], \"nontrivial\": \"P.sound_nontrivial\"} under `witnesses` (LEAN-66)"]
    out_of_scope = witnessed(guarantee={"reads": fact()["reads"]})
    assert witness_codes({**out_of_scope[0], "witnesses": {}}, out_of_scope[1]) == []
    assert witness_codes(*out_of_scope) == []
    assert witness_codes({**pol, "witnesses": {"P.sound": {"honest": "P.sound_honest", "nontrivial": "P.sound_nontrivial"}}},
                         f) == ["witnesses.witness-form"]
    assert witness_codes({**pol, "witnesses": []}, f) == ["witnesses.form"]
    assert witness_codes({**pol, "witnesses": {**pol["witnesses"], "P.gone": pol["witnesses"]["P.sound"]}},
                         f) == ["witnesses.stale"]
    unlisted = {**pol, "guarantees": {n: r for n, r in pol["guarantees"].items() if n != "P.sound_nontrivial"}}
    assert witness_codes(unlisted, f) == ["witnesses.unlisted-witness"]
    lemma = witnessed(nontrivial={"kind": "def"})
    assert witness_codes(*lemma) == ["witnesses.witness-not-a-theorem"]
    assert witness_codes(*witnessed(nontrivial={"body": None})) == ["witnesses.no-body"]
    soundness = witnessed(guarantee={"reads": [{"name": "Properties.Verity.Soundness.Sound", "module": "Properties.Verity.Soundness",
                                                "kind": "def", "hash": "5" * 32}]})
    assert A.witness_failures({**soundness[0], "witnesses": {}}, soundness[1], WOWN)[0].startswith(
        "witnesses: P.sound lists no witnesses that it says something, and the rule holds it to them (it reads "
        "Properties.Verity.Soundness.Sound, the one theorem's vocabulary)")


def test_the_three_vacuous_guarantees_of_26_october_fail_the_witnesses_rule():
    """The three cases: a conclusion `∃ X, C X` whose X the prover picks (`∃ X, ¬ C acceptAll X` holds of any such
    conclusion, so the `nontrivial` lemma is `∀ X, ¬ C acceptAll X`); a premise that can never hold (no `honest` lemma
    concludes it, except by assuming it, assuming `False`, or concluding something else); and a bound whose break always
    holds (a `nontrivial` lemma negating `C` alone says nothing of `C ∨ Break`)."""
    some_x = {"body": wbody("P.Holds", binders=["∀", "→", "→", "∃"])}
    assert A.witness_failures(*witnessed(some_x, nontrivial={"body": wbody("P.Holds", negated=True, binders=["∃"])}),
                              WOWN) == [
        "witnesses: P.sound_nontrivial negates P.Holds under 0 `∀` after its last `∃`, and P.sound's conclusion has 1 `∃`: "
        "each is a `∀` before the `¬` (`∀ X, ¬ C … X`), since `∃ X, ¬ C … X` holds whenever the prover can pick a bad X"]
    assert witness_codes(*witnessed(some_x, nontrivial={"body": wbody("P.Holds", negated=True, binders=["∀"])})) == []
    assert "∀ X…, ¬ P.Holds at acceptAll" in A.witness_shapes("P.sound", witnessed(some_x)[1]["pins"]["P.sound"], {}, WOWN)

    never = {"premises": [COMPILED], "body": wbody("P.Compiled", conclusion=[wk("P.Compiled", assumed=True)])}
    assert witness_codes(*witnessed(honest={"body": wbody("P.Other")})) == ["witnesses.honest-uncovered"]
    assert A.witness_failures(*witnessed(honest={"body": wbody("P.Other")}), WOWN)[0] == (
        "witnesses: P.sound's `honest` witnesses (P.sound_honest) conclude nothing of P.Compiled at the deployed instance: "
        "each premise outside a named assumption's module, and a property's acceptance, is a conjunct of one (a premise "
        "that can never hold makes P.sound say nothing)")
    assert witness_codes(*witnessed(honest=never)) == ["witnesses.honest-form"] * 2
    assert witness_codes(*witnessed(honest={"body": wbody("P.Compiled", absurd=True)})) == ["witnesses.honest-form"]
    assert witness_codes(*witnessed(honest={"hypotheses": [{"binder": "hd", "type": "P.Compiled d", "closed": False}]})) == \
        ["witnesses.honest-form"]
    assert witness_codes(*witnessed(honest={"body": wbody("True", conclusion=[wk("True", [])])})) == \
        ["witnesses.honest-form", "witnesses.honest-uncovered"]
    assumed = {"premises": [COMPILED | {"module": "P.Assumptions"}]}
    assert witness_codes(*witnessed(assumed, honest={"body": wbody("P.Other")})) == []
    device = witnessed(honest={"body": wbody("P.Other")}, shape={"premises": ["P.Defs"]})
    assert witness_codes(*device) == []

    broken = {"body": wbody("Or", binders=["∀", "→", "→"], conclusion=[wk("Or", ["P.Holds", "P.Break"])])}
    assert A.witness_failures(*witnessed(broken), WOWN) == [
        "witnesses: P.sound_nontrivial does not state that P.sound's conclusion fails at the verifier that accepts "
        "everything: past its ∀ and ∃ binders its statement must be `¬ C …` with C one of Or or a definition unfolding to one, and "
        "it is ¬ P.Holds … (for a bound with a break, `¬ (C … ∨ Break …)`: the break fails too)"]
    assert witness_codes(*witnessed(broken, nontrivial={"body": wbody("Or", negated=True)})) == []
    assert "¬ (… ∨ …) at acceptAll …` (its conclusion is false for a verifier that accepts everything, the break included" in \
        A.witness_shapes("P.sound", witnessed(broken)[1]["pins"]["P.sound"], {}, WOWN)
    assert witness_codes(*witnessed(nontrivial={"body": wbody("P.Holds")})) == ["witnesses.nontrivial-form"]
    assert witness_codes(*witnessed(nontrivial={"body": wbody("P.Holds", negated=True, absurd=True)})) == \
        ["witnesses.nontrivial-assumes"]


def test_a_property_of_an_acceptance_has_an_honest_witness_that_the_deployment_is_accepted():
    """A guarantee `shape.properties` lists as a property of a function's acceptance is in scope whatever it reads, and
    its honest witnesses conclude that the function accepts the deployed instance as well as its premises; a property of a
    rejection is not in scope by that alone."""
    prop = {"properties": {"P.sound": {"function": "P.verify", "outcome": "accept"}}}
    pol, f = witnessed({"reads": fact()["reads"]}, shape=prop)
    assert A.witness_failures(pol, f, WOWN) == [
        "witnesses: P.sound's `honest` witnesses (P.sound_honest) conclude nothing of P.verify accepting at the deployed "
        "instance: each premise outside a named assumption's module, and a property's acceptance, is a conjunct of one (a "
        "premise that can never hold makes P.sound say nothing)"]
    accepts = {"body": wbody("And", conclusion=[wk("P.Compiled"), wk("Eq", ["P.verify", "P.deploy"])])}
    assert witness_codes(*witnessed({"reads": fact()["reads"]}, accepts, shape=prop)) == []
    assert A.witness_failures({**pol, "witnesses": {}}, f, WOWN)[0].startswith(
        "witnesses: P.sound lists no witnesses that it says something, and the rule holds it to them (a property of "
        "P.verify accepting): prove `P.sound_honest : P.Compiled d … ∧ P.verify accepts d …`")
    both = witnessed({"body": wbody("And", conclusion=[wk("P.Holds"), wk("P.Bound")])})[1]["pins"]["P.sound"]
    assert "`P.sound_nontrivial : ¬ (one of P.Holds, P.Bound) at acceptAll …`" in A.witness_shapes("P.sound", both, {}, WOWN)
    rejects = {"properties": {"P.sound": {"function": "P.verify", "outcome": "reject"}}}
    assert witness_codes({**pol, "witnesses": {}, "shape": rejects}, f) == []


def test_an_unwitnessed_guarantee_is_listed_with_its_statement_and_the_list_only_shrinks():
    """The guarantees in scope when the rule came are listed under `unwitnessed` with their statements' hashes: one whose
    statement changed carries its witnesses (the hash no longer matches); an entry of one out of scope fails, and one beside
    the witnesses it would exempt, or of a witness, does nothing, the witnesses checked as ever. An entry may exempt one
    half (`missing`) of a guarantee listing the other. `--update` keeps an entry while it matches (re-keyed when the
    guarantee moved), drops every other, and adds none, so a new guarantee in scope fails until it has witnesses."""
    pol, f = witnessed()
    h = f["pins"]["P.sound"]["type_sha256"]
    bare = {**POLICY, "guarantees": {"P.sound": {"owner": "@lean", "type_hash": h}}, "unwitnessed": {"P.sound": h}}
    assert A.witness_failures(bare, f, WOWN) == []
    assert A.unwitnessed_kept(bare, f, {}) == {"P.sound": h}
    changed = {**bare, "unwitnessed": {"P.sound": sha("other")}}
    assert witness_codes(changed, f) == ["witnesses.unwitnessed-changed"]
    assert A.witness_failures(changed, f, WOWN)[0].startswith(
        "witnesses: P.sound changed since `unwitnessed` listed it, and the rule holds it to witnesses (it reads the claim "
        "P.Holds), so it carries them now: prove `P.sound_honest : P.Compiled d …`")
    assert A.unwitnessed_kept(changed, f, {}) == {}
    assert witness_codes({**pol, "unwitnessed": {"P.sound": h}}, f) == []
    assert witness_codes(*witnessed(honest={"body": wbody("P.Other")}, unwitnessed={"P.sound": h})) == \
        ["witnesses.honest-uncovered"]
    assert A.unwitnessed_kept({**pol, "unwitnessed": {"P.sound": h}}, f, {}) == {}
    half = {"type_hash": h, "missing": "nontrivial", "why": "PRF-212 / #4867: not yet closed"}
    honest_only = {**pol, "witnesses": {"P.sound": {"honest": ["P.sound_honest"]}}, "unwitnessed": {"P.sound": half}}
    assert witness_codes(honest_only, f) == []
    assert A.unwitnessed_kept(honest_only, f, {}) == {"P.sound": half}
    assert witness_codes({**honest_only, "unwitnessed": {"P.sound": half | {"missing": "honest"}}}, f) == \
        ["witnesses.witness-form"]
    assert witness_codes(*witnessed(honest={"body": wbody("P.Other")}, unwitnessed={"P.sound": half},
                                    witnesses={"P.sound": {"honest": ["P.sound_honest"]}})) == ["witnesses.honest-uncovered"]
    assert witness_codes({**pol, "unwitnessed": {"P.sound": half}}, f) == []
    assert A.unwitnessed_kept({**pol, "unwitnessed": {"P.sound": half}}, f, {}) == {}
    assert witness_codes({**bare, "unwitnessed": {"P.sound": half}}, f) == ["witnesses.unwitnessed-half"]
    assert witness_codes({**honest_only, "unwitnessed": {"P.sound": half | {"type_hash": sha("other")}}}, f) == \
        ["witnesses.unwitnessed-changed"]
    assert witness_codes({**bare, "unwitnessed": {"P.sound": {"type_hash": h}}}, f) == ["witnesses.form"]
    became = {**pol, "unwitnessed": {"P.sound_honest": f["pins"]["P.sound_honest"]["type_sha256"]}}
    assert witness_codes(became, f) == []
    assert A.unwitnessed_kept(became, f, {}) == {}
    flight = {**bare, "unwitnessed": {"P.sound": h, "P.ahead": sha("a guarantee a PR ahead on next brings")}}
    assert witness_codes(flight, f) == []
    assert A.unwitnessed_kept(flight, f, {}) == flight["unwitnessed"]
    out = witnessed(guarantee={"reads": fact()["reads"]})[1]
    assert witness_codes(bare, out) == ["witnesses.unwitnessed-stale"]
    assert A.unwitnessed_kept(bare, out, {}) == {}
    assert witness_codes({**bare, "unwitnessed": []}, f) == ["witnesses.form"]

    two = facts(pins={**f["pins"], "P.sound2": f["pins"]["P.sound"]}, own_modules=sorted(WOWN))
    grown = {**bare, "guarantees": {**bare["guarantees"], "P.sound2": {"owner": "@lean"}}}
    assert witness_codes(grown, two) == ["witnesses.missing"]
    assert A.unwitnessed_kept(grown, two, {}) == {"P.sound": h}

    old = sha("before the move")
    moved = facts(pins={"P.sound": f["pins"]["P.sound"] | {"moved": {"sha256": old}}}, own_modules=sorted(WOWN))
    was = {**bare, "guarantees": {"P.sound": {"owner": "@lean", "type_hash": old}}, "unwitnessed": {"P.sound": old}}
    assert witness_codes(was, moved) == ["witnesses.unwitnessed-changed"]
    assert A.unwitnessed_kept(was, moved, {}) == {"P.sound": h}
    assert A.unwitnessed_kept({**was, "guarantees": {"P.sound": {"owner": "@lean", "type_hash": sha("x")}}}, moved, {}) == {}

    assert A.renamed_witnesses({"witnesses": {"P.a": {"honest": ["P.a_h"], "nontrivial": "P.a_n"}},
                                "unwitnessed": {"P.b": h}}, {"P": "Q"}) == {
        "witnesses": {"Q.a": {"honest": ["Q.a_h"], "nontrivial": "Q.a_n"}}, "unwitnessed": {"Q.b": h}}
    assert A.renamed_witnesses(POLICY, {"P": "Q"}) == {}


def premise(text, head=None, unfolded=None, range=False, closed=False, outer=True, binder="h"):
    """One of `Facts.lean`'s `inline` entries: a `Prop` a guarantee takes."""
    return {"binder": binder, "type": text, "head": head, "unfolded": unfolded or head, "range": range, "closed": closed,
            "outer": outer}


def inlined(*premises, **policy):
    """(policy, facts) with `P.t` taking ``premises`` (`P.Fits` and `P.Hyp` listed, `P.SecurityProofs.u` a listed
    guarantee lifted into `P.Guarantees.U`) and recorded; `P.Fits`, which no fact reads, is listed again after the
    update, which lists only the assumptions read."""
    u = fact(signature="P.SecurityProofs.u : P.Guarantees.U", hyps=[], h="00000000000000bb") | {
        "lift": {"head": "P.Guarantees.U", "module": "P.Guarantees"}, "inline": []}
    f = facts(pins={"P.t": fact() | {"inline": list(premises)}, "P.SecurityProofs.u": u})
    pol = POLICY | {"assumptions": {"P.Hyp": {"owner": "@lean"}, "P.Fits": {"owner": "@lean"}},
                    "guarantees": {"P.t": {"owner": "@lean"}, "P.SecurityProofs.u": {"owner": "@lean"}}}
    pol = A.judge(pol, f, FILES, update=True)[1]
    return pol | {"assumptions": {"P.Fits": {"owner": "@lean"}} | pol["assumptions"]} | policy, f


def inline_codes(pol, f) -> list[str]:
    return [x.code for x in A.judge(pol, f, FILES)[0] if "inline" in x.code]


def test_a_guarantee_takes_named_assumptions_other_guarantees_and_plain_ranges_inline():
    """Each `Prop` a guarantee takes that the rule allows: a listed assumption (as written or past reducible unfolding), a
    premise of this repository the premise rule lists, another listed guarantee's statement, a plain range, and a closed
    hypothesis of the statement as written (the hypotheses rule's)."""
    ok = [premise("P.Fits n", "P.Fits"), premise("P.Within n", "P.Within", unfolded="P.Fits"),
          premise("P.Guarantees.U", "P.Guarantees.U"), premise("0 < x", "LT.lt", range=True),
          premise("∀ (i : Fin n), 0 ≤ w i", "LE.le", range=True), premise("P.Hyp", "P.Hyp", closed=True),
          premise("1 = 1", "Eq", closed=True)]
    pol, f = inlined(*ok)
    assert inline_codes(pol, f) == []
    owned = facts(pins=f["pins"] | {"P.t": f["pins"]["P.t"] | {
        "inline": [premise("P.Leaves n", "P.Leaves")], "premises": [{"binder": "h", "head": "P.Leaves", "module": "P.Thms"}]}},
        graph={"P.Thms": {"imports": ["Init"], "olean": "/r/P/.lake/build/lib/lean/P/Thms.olean"}})
    assert inline_codes(pol, owned) == []
    _, inline, offenders = A.inline_failures(pol, f)
    assert inline == {} and offenders == {}


def test_an_unnamed_inline_hypothesis_fails_with_the_fix():
    """A `Prop` a guarantee takes that is no named assumption, no other listed guarantee's statement and no plain range
    fails with the rule's code and fix: an equation, a bound against a term, the guarantee's own statement, the statement
    of a guarantee the lock doesn't list, and a closed one inside a conjunct (not the statement as written)."""
    bad = [premise("P.f n = 3", "Eq"), premise("x ≤ 2 * y", "LE.le"), premise("P.Guarantees.T", "P.Guarantees.T"),
           premise("P.Unlisted n", "P.Unlisted"), premise("1 = 1", "Eq", closed=True, outer=False)]
    pol, f = inlined(*bad)
    f["pins"]["P.t"]["lift"] = {"head": "P.Guarantees.T", "module": "P.Guarantees"}
    fails = [x for x in A.judge(pol, f, FILES)[0] if "inline" in x.code]
    assert [(x.code, x.entry) for x in fails] == [("guarantees.inline-hypothesis", "P.t")]
    assert fails[0] == ("guarantees: P.t takes (P.f n = 3); (x ≤ 2 * y); (P.Guarantees.T); (P.Unlisted n); (1 = 1): an "
                        "inline hypothesis, neither a named assumption the lock lists, another listed guarantee's statement "
                        "nor a plain range (`0 < x`, `0 ≤ x`, `x ≤ 1`, `x ≤ y`, `∀ i, 0 ≤ f i`, over bound variables and "
                        "numerals); name it as a Prop in an Assumptions module and list it in the lock, or state it as a "
                        "lock-listed guarantee's conclusion")
    assert A.INLINE_FIX in fails[0]
    assert A.inline_failures(pol, f)[2] == {"P.t": ["P.f n = 3", "x ≤ 2 * y", "P.Guarantees.T", "P.Unlisted n", "1 = 1"]}
    unread = facts(pins=f["pins"] | {"P.t": f["pins"]["P.t"] | {"inline": None}})
    assert inline_codes(pol, unread) == ["guarantees.inline-unread"]
    assert inline_codes(pol, facts(pins={k: {x: y for x, y in v.items() if x != "inline"} for k, v in f["pins"].items()})) \
        == [], "facts of an older audit tool, without `inline`"


def test_a_grandfathered_inline_hypothesis_passes_and_the_list_only_shrinks():
    """`shape.inline` lists the offenders when the rule came, keyed by their text with its whitespace as one space: they
    pass, a new one fails, and `--update` drops each entry whose premise no longer offends (all of a guarantee's, its key
    with them, and the map once empty), keeps those of a guarantee the lock doesn't list yet, and adds none; `--moved`
    carries an entry to its guarantee's new name. A property `shape.properties` holds is the shape rule's; a frozen one
    is judged here. A witness `witnesses` or `claims.witnesses` names is that rule's, and `--update` drops its entries.
    An entry with no reason fails."""
    why = "inline when the rule came"
    pol, f = inlined(premise("P.f n\n    = 3", "Eq"), premise("P.g n = 4", "Eq"))
    allow = {"P.t": {"P.f n = 3": why, "P.g n = 4": why, "P.h n = 5": why}, "P.ahead": {"x = y": why}}
    held = pol | {"shape": {"premises": ["P.Assumptions"], "inline": allow}}
    assert inline_codes(held, f) == []
    fails, out = A.judge(held, f, FILES, update=True)
    assert [x for x in fails if "inline" in x.code] == []
    assert out["shape"] == {"premises": ["P.Assumptions"], "inline": {"P.ahead": {"x = y": why},
                                                                      "P.t": {"P.f n = 3": why, "P.g n = 4": why}}}
    grown = inlined(premise("P.f n = 3", "Eq"), premise("P.k n = 6", "Eq"))[1]
    fails = [x for x in A.judge(held, grown, FILES)[0] if "inline" in x.code]
    assert [x.code for x in fails] == ["guarantees.inline-hypothesis"] and fails[0].startswith(
        "guarantees: P.t takes (P.k n = 6): an inline hypothesis")
    assert A.judge(held, grown, FILES, update=True)[1]["shape"]["inline"]["P.t"] == {"P.f n = 3": why}
    fixed = inlined(premise("P.Fits n", "P.Fits"))[1]
    assert A.judge(held, fixed, FILES, update=True)[1]["shape"] == {"premises": ["P.Assumptions"],
                                                                    "inline": {"P.ahead": {"x = y": why}}}
    alone = pol | {"shape": {"inline": {"P.t": {"P.f n = 3": why}}}}
    assert A.judge(alone, fixed, FILES, update=True)[1]["shape"] == {}
    assert A.inline_failures(held, grown)[1:] == ({"P.ahead": {"x = y": why}, "P.t": {"P.f n = 3": why}},
                                                  {"P.t": ["P.f n = 3", "P.k n = 6"]})
    assert A.renamed_witnesses({"shape": held["shape"]}, {"P.t": "P.SecurityProofs.T"}) == {"shape": {
        "premises": ["P.Assumptions"], "inline": {"P.SecurityProofs.T": allow["P.t"], "P.ahead": allow["P.ahead"]}}}
    for named in ({"witnesses": {"P.SecurityProofs.u": {"honest": ["P.t"], "nontrivial": "P.n"}}},
                  {"claims": {"modules": ["P.Claims"], "witnesses": {"P.C": {"fails": "P.n", "meets": ["P.t"]}}}}):
        assert A.inline_failures(held | named, grown) == ([], {"P.ahead": {"x = y": why}}, {})
    prop = {"function": "P.f", "outcome": "none"}
    assert inline_codes(pol | {"shape": {"properties": {"P.t": prop}}}, grown) == []
    assert inline_codes(pol | {"shape": {"properties": {"P.t": prop}, "frozen": {"P.t": "frozen: open"}}}, grown) == \
        ["guarantees.inline-hypothesis"]
    assert inline_codes(pol | {"shape": {"inline": {"P.t": {"P.f n = 3": " "}}}}, grown) == ["shape.inline-form"]
    assert inline_codes(pol | {"shape": {"inline": ["P.t"]}}, grown) == ["shape.inline-form"]


WHY = "inline when the rule came"


def renamed_inline(was, now, consts, constants, moves=None):
    """(policy, facts): `P.t`'s `shape.inline` lists ``was``, and its build takes ``now`` instead, naming ``consts``; the
    build holds ``constants`` ({name: its kind, or (its kind, the text its SHA-256 is of)}) and was read with ``moves``."""
    pol, f = inlined(premise(now, "Eq") | {"consts": consts})
    table = {n: {"kind": k} if isinstance(k, str) else {"kind": k[0], "sha256": sha(k[1])} for n, k in constants.items()}
    return pol | {"shape": {"inline": {"P.t": {was: WHY}}}}, f | {"constants": table, "moves": moves or {}}


def test_a_grandfathered_inline_premise_pairs_with_no_other_premise_than_itself_renamed():
    """An entry takes a premise's place only through renamed declarations, so each of these fails and `--update` drops
    the entry: a predicate for another that is still declared, a constructor for its sibling (kept or gone), a bound for
    another with both declared, or the old one gone while the lock read the new one, a type for another with both
    declared, swapped projections (alone or beside a rename), a rename beside another change or with another field
    after it, a field of a bound variable for a declaration, two declarations for one, a new name `constants` doesn't
    hold, and a gone name in facts that hold no constants (an older audit tool's)."""
    cases = [("P.ok g", "P.bad g", ["P.bad"], {"P.ok": ("def", "ok"), "P.bad": ("def", "bad")}),
             ("g = P.Gate.and", "g = P.Gate.or", ["P.Gate.or"], {"P.Gate.and": "constructor", "P.Gate.or": "constructor"}),
             ("g = P.Gate.and", "g = P.Gate.or", ["P.Gate.or"], {"P.Gate.or": "constructor"}),
             ("x ≤ P.lo", "x ≤ P.hi", ["P.hi"], {"P.lo": ("def", "0"), "P.hi": ("def", "10")}),
             ("Op", "IntOp", ["IntOp"], {"Op": "inductive", "IntOp": "inductive"}),
             ("p.1 < p.2", "p.2 < p.1", [], {}),
             ("P.Old.lt p.1 p.2", "P.New.lt p.2 p.1", ["P.New.lt"], {"P.New.lt": ("def", "lt")}),
             ("P.Old.lo < x", "P.New.lo ≤ x", ["P.New.lo"], {"P.New.lo": ("def", "lo")}),
             ("P.Old.δ.floor ≤ x", "P.New.δ.ceil ≤ x", ["P.New.δ"], {"P.New.δ": ("def", "δ")}),
             ("P.ok x.f", "P.ok P.c", ["P.ok", "P.c"], {"P.ok": ("def", "ok"), "P.c": ("def", "c")}),
             ("P.a ≤ P.b", "P.c ≤ P.c", ["P.c"], {"P.c": ("def", "c")}),
             ("P.a ≤ P.c", "P.c ≤ P.c", ["P.c"], {"P.c": ("def", "c")}),
             ("x ≤ P.Old.lo", "x ≤ P.New.lo", ["P.New.lo"], {})]
    for was, now, consts, constants in cases:
        pol, f = renamed_inline(was, now, consts, constants)
        assert inline_codes(pol, f) == ["guarantees.inline-hypothesis"], (was, now)
        assert A.inline_failures(pol, f)[1] == {}, (was, now)
    pol, f = renamed_inline("x ≤ P.lo", "x ≤ P.hi", ["P.hi"], {"P.hi": ("def", "10")})
    assert inline_codes(pol, f) == []
    read = pol | {"reads": {"P.Defs": {"definitions": {"P.hi": sha("10")}, "guarantees": ["P.t"]}}}
    assert inline_codes(read, f) == ["guarantees.inline-hypothesis"]
    older = {k: v for k, v in f.items() if k != "constants"}
    assert inline_codes(pol, older) == ["guarantees.inline-hypothesis"]


def test_a_grandfathered_inline_premise_follows_its_declarations_renames_and_update_rekeys_it():
    """A premise that is a grandfathered entry with names renamed takes the entry's place: a declaration `--moved` moved (a
    namespace's constructor too, and one whose old name is still declared otherwise), one whose old name is gone from the
    build within its root namespace (with the fields printed after it unchanged), and a definition with the same SHA-256
    as the old name's. `--update` keys the entry by the premise as printed, with its reason."""
    cases = [("P.Old.ok g", "P.New.ok g", ["P.New.ok"], {"P.Old.ok": ("def", "ok"), "P.New.ok": ("def", "new")},
              {"P.Old": "P.New"}),
             ("List.instMembership.mem t.items (Flock.Firewall.Item.coins k)",
              "List.instMembership.mem t.items (Flock.Gate.Item.coins k)", ["List.instMembership", "Flock.Gate.Item.coins"],
              {"Flock.Gate.Item.coins": "constructor", "List.instMembership": ("def", "mem")}, {"Flock.Firewall": "Flock.Gate"}),
             ("Real.instLE.le (P.Old.size s).cast P.Old.δ.cast", "Real.instLE.le (P.New.size s).cast P.New.δ.cast",
              ["Real", "Real.instLE", "P.New.size", "P.New.δ"], {"P.New.size": ("def", "size"), "P.New.δ": ("def", "δ")},
              None),
             ("P.ok g", "P.ok' g", ["P.ok'"], {"P.ok": ("def", "ok"), "P.ok'": ("def", "ok")}, None)]
    for was, now, consts, constants, moves in cases:
        pol, f = renamed_inline(was, now, consts, constants, moves)
        assert inline_codes(pol, f) == [], (was, now)
        assert A.inline_failures(pol, f)[1:] == ({"P.t": {now: WHY}}, {"P.t": [now]})
        assert A.judge(pol, f, FILES, update=True)[1]["shape"] == {"inline": {"P.t": {now: WHY}}}
    pol, f = renamed_inline("P.Old.ok g", "P.New.ok g", ["P.New.ok"], {"P.Old.ok": ("def", "ok"), "P.New.ok": ("def", "new")})
    assert inline_codes(pol, f) == ["guarantees.inline-hypothesis"], "without the move, both declared and different"


INLINE_G = "import Control.Defs\n\ndef Control.Guarantees.G : Prop := ∀ n : Nat, 0 < n → Control.double n = 5 → True\n"
INLINE = """import Control.Assumptions
import Control.Guarantees

namespace Control

theorem ranges (n : Nat) (x y : Int) (w : Fin 3 → Nat) (hn : 0 < n) (hx : x ≤ 1) (hxy : (n : Int) ≤ y)
    (hw : ∀ i, 0 ≤ w i) (hf : Fits n) (hall : ∀ ω : Nat, Fits (ω * n)) : True := trivial

theorem offends (n : Nat) (x y : Int) (h2 : x ≤ 2 * y) (he : double n = 4 ∧ 0 < n) (hω : ∀ ω : Nat, double ω = n) :
    True := trivial

theorem SecurityProofs.G : Guarantees.G := fun _ _ _ => trivial

theorem uses_g : True ∧ (Guarantees.G → True) := ⟨trivial, fun _ => trivial⟩

end Control
"""


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_inline_hypotheses_are_read_past_lifts_foralls_and_conjunctions_on_a_real_build(tmp_path):
    """`Facts.lean`'s `inline` on a real build: plain ranges over bound variables and numerals, under a cast and of a
    family (`∀ i, 0 ≤ w i`), and a named assumption, bare and under a leading `∀ ω`, pass; a bound against a term, an
    equation split out of a conjunction and one under `∀ ω` fail, and so does the equation a guarantee lifted into a
    `Guarantees` definition takes, which the hypotheses rule never sees (its statement as written binds nothing); that
    definition taken inside a conjunct is another listed guarantee's statement. Each premise names the constants its text
    shows, and the build reports those and the names of the lock's `shape.inline` texts it declares, with a definition's
    SHA-256 blind to its name: an entry under a name the build lacks, or under a definition the same as the printed one,
    takes its premise's place; one under another declared definition doesn't."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    src = pkg / "Control" / "Assumptions.lean"
    src.write_text(src.read_text().replace("end Control", "def Fits (n : Nat) : Prop := n < 100\n\n"
                                           "def twice (n : Nat) : Nat := 2 * n\n\nend Control"))
    (pkg / "Control" / "Guarantees.lean").write_text(INLINE_G)
    (pkg / "Control" / "Inline.lean").write_text(INLINE)
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.Inline\n")
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    names = ["Control.ranges", "Control.offends", "Control.SecurityProofs.G", "Control.uses_g"]
    policy = unclaimed(A.load(pkg))
    policy["guarantees"] = policy["guarantees"] | {n: {"owner": "@lean"} for n in names}
    policy["assumptions"] = policy["assumptions"] | {"Control.Fits": {"owner": "@lean"}}
    f = A.facts_of(pkg, policy, tmp_path)
    got = {n: [(p["binder"], p["range"]) for p in f["pins"][n]["inline"]] for n in names}
    assert got["Control.ranges"] == [("hn", True), ("hx", True), ("hxy", True), ("hw", True), ("hf", False),
                                     ("hall", False)]
    assert got["Control.offends"] == [("h2", False), ("he", False), ("he", True), ("hω", False)]
    assert [p["head"] for p in f["pins"]["Control.uses_g"]["inline"]] == ["Control.Guarantees.G"]
    _, _, offenders = A.inline_failures(policy, f)
    assert sorted(offenders) == ["Control.SecurityProofs.G", "Control.offends"]
    assert len(offenders["Control.offends"]) == 3 and offenders["Control.offends"][2].startswith("∀")
    assert len(offenders["Control.SecurityProofs.G"]) == 1 and "5" in offenders["Control.SecurityProofs.G"][0]
    fails = [x for x in A.judge(policy, f, A.lean_modules(pkg))[0] if x.code == "guarantees.inline-hypothesis"]
    assert sorted(x.entry for x in fails) == ["Control.SecurityProofs.G", "Control.offends"]

    held = policy | {"shape": {"inline": {"Control.offends": {"Eq (Control.twice n) 4": WHY,
                                                              "∀ (ω : Nat), Eq (Control.Small ω) n": WHY},
                                          "Control.SecurityProofs.G": {"Eq (Control.thrice n) 5": WHY}}}}
    f = A.facts_of(pkg, held, tmp_path)
    assert [p["consts"] for p in f["pins"]["Control.offends"]["inline"]][1:] == \
        [["Control.double", "Eq"], ["Nat", "instLTNat"], ["Control.double", "Eq", "Nat"]]
    c = f["constants"]
    assert c["Control.twice"]["sha256"] == c["Control.double"]["sha256"] != c["Control.Small"]["sha256"]
    assert c["Control.double"]["kind"] == "def" and "Control.thrice" not in c and c["Eq"] == {"kind": "inductive"}
    fails = [x for x in A.judge(held, f, A.lean_modules(pkg))[0] if x.code == "guarantees.inline-hypothesis"]
    assert [x.entry for x in fails] == ["Control.offends"] and fails[0].startswith(
        "guarantees: Control.offends takes (Int.instLEInt.le x (instHMul.hMul 2 y)); (∀ (ω : Nat), Eq (Control.double ω) "
        "n): an inline hypothesis")
    assert A.judge(held, f, A.lean_modules(pkg), update=True)[1]["shape"]["inline"] == {
        "Control.SecurityProofs.G": {"Eq (Control.double n) 5": WHY}, "Control.offends": {"Eq (Control.double n) 4": WHY}}


def test_a_lifted_guarantee_is_witnessed_as_the_statement_it_lifts():
    """A four-name guarantee (`P.SecurityProofs.sound : P.Guarantees.sound`) is read with its `Guarantees` definition
    unfolded once (`lifted`): its nontrivial witness negates the head of the lifted statement, and its witnesses are not
    held to `four_names`."""
    lifted = wbody("P.Holds", binders=["∀", "→", "→", "∃"])
    g = {"body": wbody("P.Guarantees.sound", conclusion=[wk("P.Guarantees.sound")], lifted=lifted)}
    assert witness_codes(*witnessed(g, nontrivial={"body": wbody("P.Holds", negated=True, binders=["∀"])})) == []
    assert witness_codes(*witnessed(g, nontrivial={"body": wbody("P.Holds", negated=True)})) == \
        ["witnesses.nontrivial-exists"]
    pol, f = witnessed(g, nontrivial={"body": wbody("P.Holds", negated=True, binders=["∀"])}, four_names=["P"])
    fails = A.judge(pol, f, FILES)[0]
    assert [x.entry for x in fails if x.code == "four-names"] == ["P.sound"]
    assert not [x for x in fails if x.code.startswith("witnesses")]


def test_a_witness_constructs_its_instance_and_takes_only_listed_assumptions():
    """A witness that ranges over its instance (`∀ n Reg I (ht : …) D (hD : parse … = .ok D) …,
    ¬ …`, as `sampledKSLivePub_nontrivial` once did) or takes a hypothesis that is no named assumption the lock
    lists fails; one whose only hypothesis is a listed named assumption (`FalseInScope → ¬ …`) passes."""
    one19 = {"body": wbody("P.Holds", negated=True, binders=["∀", "∀", "∀", "→", "∀", "→"]),
             "hypotheses": [{"binder": "ht", "type": "I.tags.typed = false", "closed": False},
                            {"binder": "hD", "type": "Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D",
                             "closed": False}]}
    fails = A.witness_failures(*witnessed(nontrivial=one19), WOWN)
    assert [x.code for x in fails] == ["witnesses.instance-binders"] + ["witnesses.nontrivial-assumes"] * 2
    assert fails[0] == (
        "witnesses: P.sound_nontrivial, P.sound's `nontrivial` witness, takes 4 `∀` binder(s) before its conclusion: it "
        "ranges over its instance (`∀ inst, …`) instead of constructing it. Build the instance at the model level (a small "
        "hand-built descriptor rather than a parsed file), and isolate any one field that can't be built as a named "
        "assumption the lock lists (LEAN-66)")
    assert fails[1] == ("witnesses: P.sound_nontrivial, P.sound's `nontrivial` witness, takes (ht : I.tags.typed = false), "
                        "a hypothesis over its own inputs: it assumes nothing beyond named assumptions the lock lists")
    assert witness_codes(*witnessed(honest={"body": wbody("P.Compiled", binders=["∀"])})) == ["witnesses.instance-binders"]
    unlisted = [{"binder": "h", "type": "P.FalseInScope", "closed": True, "named": "P.FalseInScope", "module": "P.Assumptions"}]
    in_scope = {"body": wbody("P.Holds", negated=True, binders=["→"]), "hypotheses": unlisted}
    assert witness_codes(*witnessed(nontrivial=in_scope)) == ["witnesses.nontrivial-assumes"]
    assert witness_codes(*witnessed(honest={"hypotheses": unlisted})) == ["witnesses.honest-form"]
    pol, f = witnessed(nontrivial=in_scope)
    assert witness_codes({**pol, "assumptions": {**pol["assumptions"], "P.FalseInScope": {"owner": "@proofs"}}}, f) == []


def test_an_honest_witness_at_a_verifier_that_accepts_everything_fails():
    """`HonestH200` builds its 'honest' period with `accepts := fun _ _ => True`: that is the accept-all verifier,
    the nontrivial lemma's object, so an honest witness whose statement applies one (`Facts.lean`'s `accept_all`) fails."""
    h200 = {"body": wbody("P.Compiled"), "accept_all": ["P.HonestH200.jointRun.accepts"]}
    assert A.witness_failures(*witnessed(honest=h200), WOWN) == [
        "witnesses: P.sound_honest, P.sound's `honest` witness, is at a verifier that accepts everything "
        "(P.HonestH200.jointRun.accepts): an honest lemma is at the deployed verifier, and the accept-all one is the "
        "`nontrivial` lemma's object (LEAN-68's HonestH200)"]
    assert witness_codes(*witnessed(nontrivial={"accept_all": ["P.RecAt.acceptAll"]})) == []


def test_an_exclusion_theorem_has_an_honest_witness_and_an_explicit_circuit_beating_its_bound():
    """A guarantee concluding a bound over a partition or a covering and reading no acceptance (NCI's
    exclusion, training's serving-block bound) is of the `exclusion` kind: its honest witnesses conclude its premises at
    the deployed partition and parameters, and its nontrivial one states, at an explicit circuit, (each other premise
    holds) ∧ ¬ (a named premise) ∧ ¬ (the bound), its conjuncts in any order; a bare
    `¬ bound` reads no premise and fails. A statement that only reads a partition, or reads a verifier's acceptance as
    well, is no exclusion."""
    cover = {"name": "P.Covering.load", "module": "P.Covering", "kind": "def", "hash": "6" * 32}
    covers = {"binder": "hc", "type": "P.Covers", "closed": True, "named": "P.Covers", "module": "P.Assumptions"}
    g = {"reads": [cover], "premises": [COMPILED, {"binder": "hc", "head": "P.Covers", "module": "P.Assumptions"}],
         "hypotheses": [covers],
         "body": wbody("LE.le", binders=["∀", "→", "→"], conclusion=[wk("LE.le", ["P.Covering.load"])])}

    def neg(inner):
        return {"head": "Not", "defs": [], "refl": False, "assumed": False, "negated": True, "inner": inner}

    def nt(*conjuncts, **more):
        return {"body": wbody("And", conclusion=list(conjuncts))} | more
    beats = nt(wk("P.Compiled"), neg("P.Covers"), neg("LE.le"))
    pol, f = witnessed(g, nontrivial=beats)
    pol["assumptions"] = {**pol["assumptions"], "P.Covers": {"owner": "@network-accounting"}}
    assert A.witness_scope("P.sound", pol, f["pins"]["P.sound"]) == (
        "exclusion", "it concludes a bound and reads P.Covering.load of P.Covering, a partition or covering, and no acceptance")
    assert A.witness_failures(pol, f, WOWN) == []
    as_kind = {"P.sound": {**pol["witnesses"]["P.sound"], "kind": "exclusion"}}
    assert witness_codes({**pol, "witnesses": as_kind}, f) == []
    assert witness_codes({**pol, "witnesses": {"P.sound": {**as_kind["P.sound"], "kind": "acceptance"}}}, f) == \
        ["witnesses.witness-form"]
    assert witness_codes(*witnessed(g, nontrivial=nt(neg("LE.le"), wk("P.Compiled"), neg("P.Covers")))) == []
    bare = witnessed(g, nontrivial={"body": wbody("LE.le", negated=True, conclusion=[neg("LE.le")])})[1]
    assert A.witness_failures(pol, bare, WOWN) == [
        "witnesses: P.sound_nontrivial drops no premise of P.sound: an exclusion's nontrivial lemma states, at an "
        "explicit circuit, (each other premise holds) ∧ ¬ (the dropped premise) ∧ ¬ (the bound), one conjunct `¬ P …` "
        "with P one of P.Compiled, P.Covers; a bare `¬ bound` reads none"]
    no_bound = witnessed(g, nontrivial=nt(wk("P.Compiled"), neg("P.Covers")))[1]
    assert witness_codes(pol, no_bound) == ["witnesses.nontrivial-form"]
    other = witnessed(g, nontrivial=nt(neg("P.Covers"), neg("LE.le")))[1]
    assert A.witness_failures(pol, other, WOWN) == [
        "witnesses: P.sound_nontrivial concludes nothing of P.Compiled at its circuit: P.sound's premises other than the "
        "one it drops (P.Covers) hold there, each a conjunct of its conclusion"]
    keeps = witnessed(g, nontrivial=beats | {"hypotheses": [covers]})[1]
    assert A.witness_failures(pol, keeps, WOWN) == [
        "witnesses: P.sound_nontrivial, P.sound's `nontrivial` witness, takes P.Covers, the premise it drops: an "
        "exclusion's nontrivial lemma drops the named premise, and an explicit circuit then beats the bound"]
    regions = {"binder": "hr", "head": "Eq", "module": "Init.Core"}
    eq = witnessed(g | {"premises": [*g["premises"], regions]}, nontrivial=beats)
    assert witness_codes(pol, eq[1]) == ["witnesses.nontrivial-premises"]
    assert witness_codes(pol, witnessed(g | {"premises": [*g["premises"], regions]},
                                        nontrivial=nt(wk("P.Compiled"), wk("Eq", []), neg("P.Covers"), neg("LE.le")))[1]) == []
    missing = A.witness_failures({**pol, "witnesses": {}}, f, WOWN)[0]
    assert "`P.sound_nontrivial : (each other premise at c₀) ∧ ¬ (P.Compiled or P.Covers) c₀ … ∧ ¬ LE.le c₀ …` (an " \
           "explicit circuit c₀ beats the bound once one named premise is dropped)" in missing and \
           '"kind": "exclusion"' in missing
    both = witnessed(g | {"reads": [cover, HOLDS]})
    assert A.witness_scope("P.sound", both[0], both[1]["pins"]["P.sound"])[0] == "acceptance"
    gate = witnessed(g | {"body": wbody("P.Covering.load", binders=["∀"])})
    assert A.witness_scope("P.sound", gate[0], gate[1]["pins"]["P.sound"]) is None
    run = {"name": "P.Run.accepts", "module": "P.Defs", "kind": "def", "hash": "7" * 32}
    verified = witnessed(g | {"reads": [cover, run]})
    assert A.witness_scope("P.sound", verified[0], verified[1]["pins"]["P.sound"]) == (
        "acceptance", "it reads P.Run.accepts, a verifier's acceptance")
    device = witnessed(g | {"reads": [cover, run | {"name": "P.Device.Accepts"}]})
    assert A.witness_scope("P.sound", device[0], device[1]["pins"]["P.sound"])[0] == "acceptance"


def test_a_nontrivial_witness_may_negate_what_the_guarantee_unfolds_to():
    """`RecursiveKS` fixes its verifier inside `RecursiveKSAt`, so its nontrivial
    witness negates `RecursiveKSAll`, which unfolds to the same `KSLin2On` at accept-all: `Facts.lean`'s `chain` reads both
    through the definitions their heads unfold through."""
    g = {"body": wbody("P.Guarantees.KS", conclusion=[wk("P.Guarantees.KS")]) | {"chain": ["P.Guarantees.KS", "P.KSAt",
                                                                                         "P.KSLin"]}}
    ks_all = {"body": wbody("P.KSAll", negated=True) | {"chain": ["P.KSAll", "P.KSLin"]}}
    assert witness_codes(*witnessed(g, nontrivial=ks_all)) == []
    other = {"body": wbody("P.Other", negated=True) | {"chain": ["P.Other", "P.Unrelated"]}}
    assert witness_codes(*witnessed(g, nontrivial=other)) == ["witnesses.nontrivial-form"]


def test_an_assumption_is_listed_by_name_with_its_record():
    """Each assumption has an owner and the digest of its definition as `reads` holds it, so a changed assumption is its own
    record; an entry no guarantee reads fails; an update converts a list of assumption modules into the definitions read
    there, owned by the owner of the guarantees that read them."""
    pol = recorded()
    assert pol["assumptions"] == {"P.Hyp": {"owner": "@lean", "digest": pol["reads"]["P.Assumptions"]["definitions"]["P.Hyp"]}}
    changed = facts(pins={"P.t": fact(reads=[{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "1" * 32},
                                             {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "4" * 32}])})
    assert "assumptions: P.Hyp's definition in P.Assumptions changed; run `check.py --update` and commit the record " \
           "(Daniel's DM shows the change when it lands)" in A.judge(pol, changed, FILES)[0]
    unread = {**pol, "assumptions": pol["assumptions"] | {"P.Other": {"owner": "@lean", "digest": "0" * 64}}}
    assert A.judge(unread, facts(), FILES)[0] == ["assumptions: P.Other is listed, but no guarantee reads it; remove the entry"]
    assert A.judge(unread, facts(), FILES, update=True)[1]["assumptions"] == pol["assumptions"]
    unowned = {**pol, "assumptions": {"P.Hyp": {"digest": pol["assumptions"]["P.Hyp"]["digest"]}}}
    assert A.judge(unowned, facts(), FILES)[0] == ['assumptions: P.Hyp has no owner: write "owner": "@lane" in its entry, or '
                                                   'run `check.py --update --owner @lane`']
    unlisted = {**pol, "assumptions": {}}
    assert A.judge(unlisted, facts(), FILES)[0][0] == ("guarantees: P.t assumes (h : P.Hyp), a closed Prop that is not a "
                                                       "named assumption: a Prop constant listed under `assumptions`")
    modules = {**POLICY, "assumptions": ["P.Assumptions"]}
    assert A.judge(modules, facts(), FILES)[0][0].startswith("assumptions: the policy lists assumption modules (P.Assumptions)")
    fails, converted = A.judge(modules, facts(), FILES, update=True)
    assert fails == [] and converted["assumptions"] == pol["assumptions"]


def test_a_guarantee_must_be_a_theorem_with_allowed_axioms():
    assert A.judge(POLICY, facts(pins={"P.t": {"kind": "missing"}}), FILES, update=True)[0] == ["guarantees: P.t is missing"]
    assert A.judge(POLICY, facts(pins={"P.t": fact(kind="def")}), FILES, update=True)[0] == ["guarantees: P.t is a def, not a theorem"]
    fails = A.judge(recorded(), facts(pins={"P.t": fact(axioms=STD + ["sorryAx"])}), FILES)[0]
    assert fails == ["guarantees: P.t uses sorryAx"]


def test_the_source_scan_refuses_kernel_bypasses_but_not_prose(tmp_path):
    (tmp_path / "P").mkdir()
    (tmp_path / "P" / "A.lean").write_text("theorem a : True := trivial\nset_option debug.skipKernelTC true in\n")
    (tmp_path / "P" / "B.lean").write_text("/-- A docstring may name debug.skipKernelTC. /- nested -/ -/\n"
                                           "-- and so may a comment: set_option debug.skipKernelTC true in\n"
                                           "def s := \"-- not a comment\" -- debug.byAsSorry\n")
    (tmp_path / "lakefile.toml").write_text("# debug.skipKernelTC in a comment\n[leanOptions]\ndebug.byAsSorry = true\n")
    hits = A.scans(tmp_path, A.lean_modules(tmp_path), {})
    assert [h.split(" names ")[0] for h in hits] == ["unchecked: P/A.lean:2", "unchecked: lakefile.toml:3"]


def test_each_heartbeat_override_is_listed_with_its_value_reason_and_module_build_time_and_none_is_refused(tmp_path):
    """`heartbeats` (Daniel, 2026-10-09: not banned, listed): an override with its reason ending its line, one with the
    comments just above as its reason, one with none, `synthInstance`'s, and one in a comment, which is not an override;
    each with its module's time from the audit's build log (`build_seconds`, which skips a non-module like `:c.o`), or
    None for a module the build didn't build."""
    (tmp_path / "P").mkdir()
    (tmp_path / "P" / "A.lean").write_text(
        "set_option maxHeartbeats 400000 in -- the `omega` over 40 cases\ntheorem a : True := trivial\n\n"
        "-- unfolding `step` twice;\n-- measured 9 s\nset_option maxHeartbeats 0 in\ntheorem b : True := trivial\n")
    (tmp_path / "P" / "B.lean").write_text("theorem c : True := trivial\nset_option synthInstance.maxHeartbeats 100000\n"
                                           "/- set_option maxHeartbeats 1 in -/\n")
    log = tmp_path / "build.log"
    log.write_text("✔ [1/3] Built P.A (1m 2.5s)\n✔ [2/3] Built P.A:c.o (900ms)\n✔ [3/3] Built P.C (250ms)\n")
    assert A.build_seconds([log, tmp_path / "missing.log"]) == {"P.A": 62.5, "P.C": 0.2}
    got = A.heartbeats(tmp_path, A.lean_modules(tmp_path), A.build_seconds([log]))
    assert [(h["file"], h["line"], h["option"], h["value"], h["reason"], h["module_seconds"]) for h in got] == [
        ("P/A.lean", 1, "maxHeartbeats", 400000, "the `omega` over 40 cases", 62.5),
        ("P/A.lean", 6, "maxHeartbeats", 0, "unfolding `step` twice; measured 9 s", 62.5),
        ("P/B.lean", 2, "synthInstance.maxHeartbeats", 100000, None, None)]
    assert A.scans(tmp_path, A.lean_modules(tmp_path), {}) == []


def test_the_dependency_scan_reads_what_the_build_takes_from_fetched_packages(tmp_path):
    dep, other = tmp_path / ".lake" / "packages" / "dep", tmp_path / ".lake" / "packages" / "other"
    lib = dep / ".lake" / "build" / "lib" / "lean"
    for path, text in ((dep / "D.lean", "/-- Prose may name debug.skipKernelTC. -/\ndef d := 1\n"),
                       (dep / "D" / "E.lean", "def e := 1\nset_option debug.skipKernelTC true in\ntheorem t : True := trivial\n"),
                       (dep / "D" / "F.lean", "set_option debug.byAsSorry true\n"),
                       (dep / "lakefile.toml", "name = \"dep\"\n[leanOptions]\ndebug.byAsSorry = true\n"),
                       (other / "lakefile.toml", "# nothing of it is imported\n[leanOptions]\ndebug.byAsSorry = true\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    oleans = [str(lib / "D.olean"), str(lib / "D" / "E.olean"), str(tmp_path / ".lake" / "build" / "lib" / "lean" / "P.olean"),
              str(tmp_path / "toolchain" / "lib" / "lean" / "Init.olean")]
    fails, read = A.dependency_scans(tmp_path, oleans, {})
    assert read == 3 and [f.split(" (")[0] for f in fails] == ["unchecked: .lake/packages/dep/D/E.lean:2",
                                                               "unchecked: .lake/packages/dep/lakefile.toml:3"]
    listed = {"D.E": "why", "dep/lakefile.toml": "why", "D.F": "not imported"}
    assert A.dependency_scans(tmp_path, oleans, {"dependency_unchecked": listed})[0] == [
        "unchecked: `dependency_unchecked` lists D.F, which the package does not import or which names no such option; "
        "remove the entry"]
    (dep / "D.lean").unlink()
    assert A.dependency_scans(tmp_path, oleans[:1], {"dependency_unchecked": {"dep/lakefile.toml": "why"}})[0] == [
        "unchecked: the package imports D from dep, whose source .lake/packages/dep/D.lean is missing, so nothing scanned it"]


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


def test_a_compile_time_listing_is_tied_to_its_file_and_kept_out_of_what_guarantees_read(tmp_path):
    (tmp_path / "P").mkdir()
    (tmp_path / "P" / "Tac.lean").write_text('elab "t" : tactic => pure ()\n')
    (tmp_path / "P" / "Defs.lean").write_text('macro "m" : term => `(1)\ndef f := 1\n')
    files = A.lean_modules(tmp_path)
    cited = {"pins": {"P.t": {"module": "P.Thms", "reads": [{"name": "P.f", "module": "P.Defs"}]}}}
    listed = {"P.Tac": {"why": "why", "digest": A.sha256(tmp_path / "P" / "Tac.lean")}}
    fails, now, notes = A.listings(tmp_path, files, {"compile_time": {"P.Tac": "why"}}, cited)
    assert [f.split(":")[1] for f in fails] == [" the listing of P.Tac records no digest of P/Tac.lean"]
    assert now == listed and notes[0].startswith("compile_time P.Tac (P/Tac.lean): listed; a reviewer reads")
    assert A.listings(tmp_path, files, {"compile_time": {"P.Tac": "why"}}, cited, update=True)[0] == []
    assert A.listings(tmp_path, files, {"compile_time": listed}, cited) == ([], listed, [])
    (tmp_path / "P" / "Tac.lean").write_text('elab "t" : tactic => do IO.println "now it does IO"\n')
    fails, _, notes = A.listings(tmp_path, files, {"compile_time": listed}, cited)
    assert fails[0].startswith("compile-time: the listing of P.Tac was reviewed against another version of P/Tac.lean")
    assert notes[0].startswith("compile_time P.Tac (P/Tac.lean): the file changed since its listing was reviewed")
    read = {"P.Defs": {"why": "why", "digest": A.sha256(tmp_path / "P" / "Defs.lean")}}
    assert A.listings(tmp_path, files, {"compile_time": read}, cited)[0][0].startswith(
        "compile-time: P.Defs is listed under `compile_time`, and guarantees are declared or read there (P.t)")
    declared = {"pins": {"P.t": {"module": "P.Tac", "reads": []}}}
    tac = {"P.Tac": {"why": "why", "digest": A.sha256(tmp_path / "P" / "Tac.lean")}}
    assert "(P.t)" in A.listings(tmp_path, files, {"compile_time": tac}, declared)[0][0]
    assert A.listings(tmp_path, files, {"compile_time": {"P": "a prefix"}}, cited)[0] == [
        "compile-time: `compile_time` lists P, which is not a module of the package; list modules one by one"]


def test_a_package_without_a_policy_gets_a_template(tmp_path):
    pkg = tmp_path / "p"
    pkg.mkdir()
    (pkg / "lakefile.toml").write_text('name = "p"\n[[lean_lib]]\nname = "Alpha"\n[[lean_lib]]\nname = "Beta"\n')
    assert A.packages(tmp_path) == [pkg]
    rep = A.audit(pkg, tmp_path / "out")
    assert rep["failures"][0].startswith(f"policy: {pkg} has no lean-audit.json")
    assert rep["failures"][0].endswith('{"roots": ["Alpha", "Beta"], "guarantees": {}}') and '{"owner": "@lane"}' in rep["failures"][0]


def test_a_record_from_before_the_rename_is_read_as_guarantees_and_one_holding_both_is_refused(tmp_path):
    pol = recorded()
    old = {("pins" if k == "guarantees" else k): v for k, v in pol.items()}
    old["reads"] = {m: {("pins" if k == "guarantees" else k): v for k, v in g.items()} for m, g in pol["reads"].items()}
    assert A.as_guarantees(json.loads(json.dumps(old))) == pol
    assert A.judge(A.as_guarantees(old), facts(), FILES) == ([], pol)
    with pytest.raises(ValueError, match=r"the record holds both `pins` and `guarantees`"):
        A.as_guarantees(pol | {"pins": {"P.u": {}}})
    with pytest.raises(ValueError, match=r"reads\[P.Defs\] holds both"):
        A.as_guarantees(pol | {"reads": {"P.Defs": pol["reads"]["P.Defs"] | {"pins": ["P.u"]}}})
    pkg = tmp_path / "p"
    pkg.mkdir()
    (pkg / A.POLICY).write_text(A.text(pol | {"pins": {"P.u": {}}}))
    rep = A.audit(pkg, tmp_path / "out")
    assert len(rep["failures"]) == 1 and rep["failures"][0].startswith(f"policy: {pkg}/{A.POLICY}: the record holds both")


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


def test_module_names_follow_src_dir(tmp_path):
    """A nested package with `srcDir = ".."` (`Security/Proofs` under `Security`) names its modules from the directory
    above it, as Lake does; a library's own `srcDir` is under the package's; a file outside every source directory keeps
    its name from the package."""
    proofs = tmp_path / "Security" / "Proofs"
    for rel, text in {"lakefile.toml": 'name = "proofs"\nsrcDir = ".."\n[[lean_lib]]\nname = "Proofs"\n',
                      "Proofs.lean": "", "Pouw/End.lean": ""}.items():
        (proofs / rel).parent.mkdir(parents=True, exist_ok=True)
        (proofs / rel).write_text(text)
    (tmp_path / "Security" / "Spec.lean").write_text("")
    assert A.lean_modules(proofs) == {"Proofs.Proofs": "Proofs.lean", "Proofs.Pouw.End": "Pouw/End.lean"}
    pkg = tmp_path / "p"
    for rel, text in {"lakefile.toml": 'name = "p"\n[[lean_lib]]\nname = "P"\nsrcDir = "src"\n',
                      "src/P.lean": "", "src/P/A.lean": "", "Check.lean": ""}.items():
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text(text)
    assert A.lean_modules(pkg) == {"P": "src/P.lean", "P.A": "src/P/A.lean", "Check": "Check.lean"}


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


def test_lean_runs_on_the_cpus_the_audit_may_use_unless_told_otherwise(monkeypatch):
    monkeypatch.delenv("LEAN_NUM_THREADS", raising=False)
    assert A.lake_env()["LEAN_NUM_THREADS"] == str(A.cores())
    monkeypatch.setenv("LEAN_NUM_THREADS", "3")
    assert A.lake_env()["LEAN_NUM_THREADS"] == "3"


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="no CPU affinity on this platform")
def test_a_job_pinned_to_one_cpu_runs_lean_on_one_thread(monkeypatch):
    """Lean's own default is every CPU of the machine, however few the job's cpuset holds."""
    monkeypatch.delenv("LEAN_NUM_THREADS", raising=False)
    cpu = min(os.sched_getaffinity(0))
    code = (f"import importlib.util, os\nos.sched_setaffinity(0, {{{cpu}}})\n"
            f"spec = importlib.util.spec_from_file_location('lean_audit', {str(HERE.parent / 'check.py')!r})\n"
            "A = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(A)\nprint(A.lake_env()['LEAN_NUM_THREADS'])\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "1"


def test_a_trains_audit_passes_record_differences_and_names_the_records_to_commit(tmp_path, monkeypatch):
    """`--regenerate` (a train's lean-audit step): a package whose every failure is a record difference, so the audit wrote
    the record `--update` would, passes, and `audit.json` names that record for the train to commit; one with any other
    failure fails as before, and the packages' own locks are never written."""
    a, b = tmp_path / "a", tmp_path / "b"
    for p in (a, b):
        p.mkdir()
        (p / A.POLICY).write_text("{}")

    def fake(pkg, out, *args, **_):
        rec = out / f"audit-{pkg.name}" / A.POLICY
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text("{}")
        regen = {"regenerated": str(rec)} if pkg.name == "a" else {}
        return {"package": str(pkg), "seconds": {}, "failures": ["guarantees: they no longer read P.Defs"]} | regen

    monkeypatch.setattr(A, "audit", fake)
    out = tmp_path / "out"
    assert A.main([str(a), "--out", str(out), "--repo", str(tmp_path)]) == 1
    assert A.main([str(a), "--regenerate", "--out", str(out), "--repo", str(tmp_path)]) == 0
    rep = json.loads((out / "audit.json").read_text())
    assert rep["regenerated"] == {"a": str(out / "audit-a" / A.POLICY)} and rep["packages"][0]["settled"]
    assert A.main([str(a), str(b), "--regenerate", "--out", str(out), "--repo", str(tmp_path)]) == 1
    assert json.loads((out / "audit.json").read_text())["failed"] == [str(b)]
    assert (a / A.POLICY).read_text() == "{}"
    with pytest.raises(SystemExit):
        A.main([str(a), "--regenerate", "--update"])


def test_packages_are_audited_side_by_side_and_a_build_waits_for_the_builds_it_requires(tmp_path, monkeypatch):
    """`b` requires `a` by path, and `c` requires nothing. Each fake audit blocks on the event it needs, so the run finishes
    only when `c` runs beside `a`'s build, and `b`'s build beside the rest of `a`'s audit."""
    a, b, c = tmp_path / "a", tmp_path / "a" / "b", tmp_path / "c"
    for p in (a, b, c):
        p.mkdir(parents=True, exist_ok=True)
        (p / A.POLICY).write_text("{}")
    (b / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "path", "name": "a", "dir": ".."}]}))
    started = {p.name: threading.Event() for p in (a, b, c)}
    events = []

    def fake(pkg, out, build, update, replay, fresh, log=print, built=lambda: None, **_):
        events.append(("start", pkg.name))
        started[pkg.name].set()
        if pkg.name == "a":
            started["c"].wait()
            built()
            events.append(("built", "a"))
            started["b"].wait()
        else:
            built()
        return {"package": str(pkg), "seconds": {}, "failures": ["fake"] if pkg.name == "c" else []}

    monkeypatch.setattr(A, "audit", fake)
    out = tmp_path / "out"
    assert A.main([str(a), str(b), str(c), "--out", str(out)]) == 1
    assert events.index(("built", "a")) < events.index(("start", "b"))
    doc = json.loads((out / "audit.json").read_text())
    assert [r["package"] for r in doc["packages"]] == [str(a), str(b), str(c)] and doc["failed"] == [str(c)]


def test_two_builds_that_write_one_path_dependency_s_lake_run_one_at_a_time(tmp_path, monkeypatch):
    """`a` and `b` both require `d` by path, as four of flock's dependents did in check 4cd5, whose builds rebuilt flock's
    stale modules into its one `.lake/build` at once, so that one's `clang` read a `.c` file another was rewriting. The first
    fake build holds on until the other package's build has started or waits for it: without the wait, the two overlap."""
    d, a, b = tmp_path / "d", tmp_path / "a", tmp_path / "b"
    for p in (d, a, b):
        p.mkdir()
    for p in (a, b):
        (p / A.POLICY).write_text(json.dumps({"roots": [p.name.upper()]}))
        (p / "lean-toolchain").write_text("leanprover/lean4:v4.34.0\n")
        (p / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "path", "name": "d", "dir": "../d"}]}))
    state = threading.Condition()
    building, done, overlaps, waits = set(), set(), [], {}

    def fake_run(argv, cwd, log=None, timeout=0):
        me, other = cwd.name, ({"a", "b"} - {cwd.name}).pop()
        with state:
            if building:
                overlaps.append((me, sorted(building)))
            building.add(me)
            state.notify_all()
            state.wait_for(lambda: other in building or other in waits or other in done)
            building.discard(me)
            done.add(me)
            state.notify_all()
        return 1, "fake build"

    def log_for(name):
        def log(line):
            with state:
                waits[name] = line
                state.notify_all()
        return log

    monkeypatch.setattr(A, "sandboxed", lambda: False)
    monkeypatch.setattr(A, "setup", lambda pkg, toolchain, log: (0, ""))
    monkeypatch.setattr(A, "run", fake_run)
    reps = {}
    threads = [threading.Thread(target=lambda p=p: reps.__setitem__(
        p.name, A.audit(p, tmp_path / "out", build=True, log=log_for(p.name), runs=False))) for p in (a, b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlaps == []
    [(second, line)] = waits.items()
    assert line == f"build: {tmp_path / second} waits for a build that writes the .lake of {d.resolve()}"
    assert [reps[p]["failures"] for p in "ab"] == [["build: exit 1\nfake build"]] * 2


def test_a_guarantee_is_proved_by_exactly_one_package():
    spec = facts(modules=["P.Defs"], pins={"P.t": {"kind": "missing"}, "P.u": fact() | {"module": "P.Defs"},
                                            "P.v": {"kind": "missing"}, "P.w": {"kind": "missing"}})
    proofs = facts(modules=["Q.Thms"], pins={"P.t": fact() | {"module": "Q.Thms"}, "P.u": fact() | {"module": "P.Defs"},
                                              "P.v": fact() | {"module": "P.Lemmas"}, "P.w": {"kind": "missing"}})
    other = facts(modules=["R.Thms"], pins={"P.t": fact() | {"module": "R.Thms"}, "P.u": {"kind": "missing"},
                                             "P.v": {"kind": "missing"}, "P.w": {"kind": "missing"}})
    sources = [("this package", spec, {"P.Defs": "P/Defs.lean", "P.Lemmas": "P/Lemmas.lean"}),
               ("../q", proofs, {"Q.Thms": "Q/Thms.lean"}), ("../r", other, {"R.Thms": "R/Thms.lean"})]
    pins, fails, by = A.assign(["P.t", "P.u", "P.v", "P.w"], sources[:2])
    assert fails == ["guarantees: P.v is declared in P.Lemmas, which no package of this package, ../q audits: a guarantee is "
                     "proved in a module under the roots of the spec package or of a package its `proved_in` names"]
    assert pins["P.t"]["module"] == "Q.Thms" and pins["P.u"]["module"] == "P.Defs" and pins["P.w"] == {"kind": "missing"}
    assert by == {"this package": ["P.u"], "../q": ["P.t"]}
    assert A.judge(POLICY | {"guarantees": {"P.w": {"owner": "@lean"}}}, facts(pins={"P.w": pins["P.w"]}), FILES, update=True)[0] == [
        "guarantees: P.w is missing"]
    fails = A.assign(["P.t"], sources)[1]
    assert fails == ["guarantees: P.t is proved in Q.Thms (../q) and in R.Thms (../r): one package proves a guarantee; "
                     "remove the others"]


def test_a_proving_package_requires_the_spec_records_no_guarantee_and_keeps_its_own_module_root(tmp_path):
    spec, q = tmp_path / "spec", tmp_path / "proofs"
    for d, rel in ((spec, "Core/Protocol.lean"), (q, "Core/SecurityProofs.lean")):
        (d / rel).parent.mkdir(parents=True)
        (d / rel).write_text("")
    (q / A.POLICY).write_text(json.dumps({"roots": ["Core.SecurityProofs"], "guarantees": {"Core.t": {}}}))
    policy = {"roots": ["Core.Protocol"], "proved_in": ["../proofs", "../nowhere", "."], "guarantees": {}}
    fails = A.proved(spec, policy, facts(modules=[], pins={}), tmp_path / "work")[0]
    assert fails == ["proved_in: ../proofs does not require this package by path, so its theorems are not about this "
                     "package's statements", f"proved_in: ../nowhere is no package with a {A.POLICY}",
                     "proved_in: . is this package"]
    assert A.proved(spec, policy | {"proved_in": "../proofs"}, facts(modules=[], pins={}), tmp_path / "work")[0] == [
        "proved_in: a list of paths, each relative to this package"]
    (q / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "path", "name": "spec", "dir": "../spec"}]}))
    (q / A.POLICY).write_text("{}")
    assert A.proved(spec, policy | {"proved_in": ["../proofs"]}, facts(modules=[], pins={}), tmp_path / "work")[0] == [
        f"proved_in: ../proofs's {A.POLICY} names no `roots`"]
    (q / A.POLICY).write_text(json.dumps({"roots": ["Core.SecurityProofs"], "guarantees": {"Core.t": {}}}))
    fails = A.proved(spec, policy | {"proved_in": ["../proofs"]}, facts(modules=[], pins={}), tmp_path / "work")[0]
    assert fails == [f"proved_in: ../proofs's {A.POLICY} lists guarantees; the guarantees it proves are recorded here, in the "
                     "spec's lock, and it records none",
                     "proved_in: ../proofs's modules share the root Core with a package it requires: Lean looks a module up "
                     "in the first package on its path holding its root, so `lake env lean` cannot load both; put its "
                     "modules under a root of their own (declarations keep their names)"]


def test_a_package_proving_anothers_guarantees_is_audited_beside_it(tmp_path, monkeypatch):
    """`s` names `p` under `proved_in`, and `p` requires `s`: auditing `s` audits `p` too, `--update` only `s`, and `s`'s
    reading of `p` waits for `p`'s build, which waits for `s`'s."""
    s, p = tmp_path / "s", tmp_path / "p"
    for d in (s, p):
        d.mkdir()
    (s / A.POLICY).write_text(json.dumps({"roots": ["S"], "proved_in": ["../p"]}))
    (p / A.POLICY).write_text(json.dumps({"roots": ["SP"]}))
    (p / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "path", "name": "s", "dir": "../s"}]}))
    events, calls = [], {}

    def fake(pkg, out, build, update, replay, fresh, log=print, built=lambda: None, ready=lambda q: None, **kw):
        calls[pkg.name] = {"update": update, "moved": kw.get("moved")}
        events.append(("start", pkg.name))
        built()
        events.append(("built", pkg.name))
        if pkg.name == "s":
            ready(p.resolve())
            events.append(("read", "s"))
        return {"package": str(pkg), "seconds": {}, "failures": []}

    monkeypatch.setattr(A, "audit", fake)
    (tmp_path / "moves.json").write_text('{"S.a": "S.b"}')
    assert A.main([str(s), "--update", "--moved", str(tmp_path / "moves.json"), "--out", str(tmp_path / "out")]) == 0
    assert calls == {"s": {"update": True, "moved": {"S.a": "S.b"}}, "p": {"update": False, "moved": None}}
    assert events.index(("built", "s")) < events.index(("start", "p")) < events.index(("built", "p")) < events.index(("read", "s"))
    doc = json.loads((tmp_path / "out" / "audit.json").read_text())
    assert [r["package"] for r in doc["packages"]] == [str(s), str(p)]


def test_a_spec_never_reads_a_proving_package_whose_build_did_not_pass(tmp_path, monkeypatch):
    """`p`'s build fails (killed for memory, say): `s`'s `ready` says so, and `proved` refuses to read `p`'s `.lake`, which
    holds whatever an earlier or interrupted build left, rather than report a module it lacks as a missing import."""
    s, p = tmp_path / "s", tmp_path / "p"
    for d, rel in ((s, "S/Defs.lean"), (p, "SP/Thms.lean")):
        (d / rel).parent.mkdir(parents=True)
        (d / rel).write_text("")
    (s / A.POLICY).write_text(json.dumps({"roots": ["S"], "proved_in": ["../p"]}))
    (p / A.POLICY).write_text(json.dumps({"roots": ["SP"]}))
    (p / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "path", "name": "s", "dir": "../s"}]}))
    seen = {}

    def fake(pkg, out, build, update, replay, fresh, log=print, built=lambda: None, ready=lambda q: None, **kw):
        if pkg.name == "s":
            built()
            seen["ready"] = ready(p.resolve())
            return {"package": str(pkg), "seconds": {}, "failures": []}
        return {"package": str(pkg), "seconds": {}, "failures": ["build: exit 137"]}

    monkeypatch.setattr(A, "audit", fake)
    assert A.main([str(s), "--out", str(tmp_path / "out")]) == 1
    assert seen["ready"] is False
    with pytest.raises(RuntimeError, match=r"\.\./p's build did not pass"):
        A.proved(s, json.loads((s / A.POLICY).read_text()), facts(modules=[], pins={}), tmp_path / "work",
                 ready=lambda q: False)


CORE = {"Core/Protocol.lean": "namespace Core\ndef double (n : Nat) : Nat := 2 * n\nend Core\n",
        "Core/Assumptions.lean": "namespace Core\ndef Hyp : Prop := ∀ n : Nat, n + 0 = n\nend Core\n",
        "Core/Guarantees.lean": "import Core.Protocol\nimport Core.Assumptions\nnamespace Core.Guarantees\n"
                                "def Main : Prop := Core.Hyp → ∀ n, Core.double n = n + n\nend Core.Guarantees\n"}
PROOFS = ("import Core.Guarantees\nnamespace Core.SecurityProofs\n"
          "theorem inline (h : Core.Hyp) (n : Nat) : Core.double n = n + n := by unfold Core.double; omega\n"
          "theorem main : Core.Guarantees.Main := fun h n => inline h n\nend Core.SecurityProofs\n")
SPEC = {"assumptions": ["Core.Assumptions"], "meaning": ["Core.Protocol", "Core.Assumptions", "Core.Guarantees"],
        "layers": {"Core.Protocol": {"imports": ["Core.Protocol"], "why": "the definitions import no proof"},
                   "Core.Assumptions": {"imports": ["Core.Assumptions"], "why": "a named assumption imports no proof"},
                   "Core.Guarantees": {"imports": ["Core.Protocol", "Core.Assumptions", "Core.Guarantees"],
                                       "why": "a guarantee is stated over the spec"}},
        "guarantees": {"Core.SecurityProofs.main": {"owner": "@lean"}, "Core.SecurityProofs.inline": {"owner": "@lean"}},
        "shape": {"frozen": {"Core.SecurityProofs.main": "frozen: a test of moving proofs, not of the shape",
                             "Core.SecurityProofs.inline": "frozen: a test of moving proofs, not of the shape"}}}


def _package(d: Path, files: dict[str, str], lakefile: str, policy: dict) -> None:
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    (d / "lakefile.toml").write_text(lakefile)
    (d / "lean-toolchain").write_text((A.FIXTURES / "control" / "lean-toolchain").read_text())
    A.dump(d, policy)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_moving_the_proofs_into_another_package_leaves_the_spec_record_as_it_was(tmp_path):
    """The pure-move property on a real build: a spec (`Protocol`, `Assumptions`, `Guarantees`) whose two guarantees (one
    stated as a `Guarantees` definition, one inline with a named assumption) are proved first in the spec's own package, then
    in a package of their own that requires it, its modules under a root of their own and the declarations under their old
    names. The spec's `--update` writes the same guarantees and reads both times, and its audit passes beside the proofs'.
    Then a proof module the proofs' roots import but do not cover (exempt) fails the spec's audit: its replay skips it."""
    one, spec, proofs, out = tmp_path / "one", tmp_path / "spec", tmp_path / "proofs", tmp_path / "out"
    _package(one, CORE | {"Core/SecurityProofs.lean": PROOFS, "Core.lean": "".join(
        f"import Core.{m}\n" for m in ("Protocol", "Assumptions", "Guarantees", "SecurityProofs"))},
        'name = "one"\ndefaultTargets = ["Core"]\n[[lean_lib]]\nname = "Core"\n', SPEC | {"roots": ["Core"]})
    roots = ["Core.Protocol", "Core.Assumptions", "Core.Guarantees"]
    _package(spec, CORE, f'name = "spec"\ndefaultTargets = ["Core"]\n[[lean_lib]]\nname = "Core"\nroots = {json.dumps(roots)}\n',
             SPEC | {"roots": roots, "proved_in": ["../proofs"]})
    _package(proofs, {"CoreProofs.lean": PROOFS}, 'name = "proofs"\ndefaultTargets = ["CoreProofs"]\n[[require]]\n'
             'name = "spec"\npath = "../spec"\n[[lean_lib]]\nname = "CoreProofs"\n', {"roots": ["CoreProofs"]})
    build = lambda d: subprocess.run(["lake", "build"], cwd=d, check=True, capture_output=True, env=A.lake_env())
    for d in (one, spec, proofs):
        build(d)
    for d in (one, proofs):
        assert unshaped(A.audit(d, out, update=True, replay=False, runs=False, log=lambda *_: None)) == []
    proofs_policy = (proofs / A.POLICY).read_text()
    assert set(json.loads(proofs_policy)) == {"roots", "dependencies"}
    assert A.main([str(spec), "--update", "--no-replay", "--out", str(out)]) == 0
    assert (proofs / A.POLICY).read_text() == proofs_policy, "the proofs' policy is not the one `--update` writes"
    split, whole = A.load(spec), A.load(one)
    assert split["guarantees"] == whole["guarantees"] and split["reads"] == whole["reads"]
    assert set(whole["reads"]) == {"Core.Protocol", "Core.Assumptions", "Core.Guarantees"}
    assert split["guarantees"]["Core.SecurityProofs.inline"]["assumptions"] == ["Core.Hyp"]
    assert A.main([str(spec), "--out", str(out)]) == 0
    reps = {Path(r["package"]).name: r for r in json.loads((out / "audit.json").read_text())["packages"]}
    assert reps["spec"]["proved_in"] == {"this package": [], "../proofs": sorted(SPEC["guarantees"])} and "replay" in reps["proofs"]["seconds"]
    (proofs / "Extra").mkdir()
    (proofs / "Extra" / "Lemma.lean").write_text("theorem Extra.lemma : True := trivial\n")
    (proofs / "CoreProofs.lean").write_text("import Extra.Lemma\n" + PROOFS)
    (proofs / "lakefile.toml").write_text((proofs / "lakefile.toml").read_text() + 'roots = ["CoreProofs", "Extra.Lemma"]\n')
    A.dump(proofs, A.load(proofs) | {"exempt": {"Extra": "a lemma outside the roots"}})
    build(proofs)
    assert unshaped(A.audit(spec, out, replay=False, runs=False, log=lambda *_: None)) == [
        "proved_in: ../proofs's roots (CoreProofs) import its Extra.Lemma, which are not under them, so its replay does not "
        "check them"]
    (proofs / "CoreProofs.lean").write_text(PROOFS.replace(
        "namespace Core.SecurityProofs\n", "namespace Core.SecurityProofs\ndef twice (n : Nat) : Nat := n + n\n").replace(
        "Core.double n = n + n := by unfold Core.double; omega", "Core.double n = twice n := by unfold Core.double twice; omega"))
    (proofs / "lakefile.toml").write_text((proofs / "lakefile.toml").read_text().replace('roots = ["CoreProofs", "Extra.Lemma"]\n', ""))
    A.dump(proofs, {k: v for k, v in A.load(proofs).items() if k != "exempt"})
    build(proofs)
    assert any(f.startswith("reads: Core.SecurityProofs.inline read Core.SecurityProofs.twice in CoreProofs, outside the spec")
               for f in A.audit(spec, out, replay=False, runs=False, log=lambda *_: None)["failures"]), \
        "a statement reading a proof package's definition fails: that package requires the spec, so it is no spec of it"


@pytest.mark.parametrize("spec", [True, False])
def test_the_steps_reading_only_the_packages_own_facts_start_before_its_proved_in_reading(tmp_path, monkeypatch, spec):
    """A spec's `proved_in` reading (`proved`) waits for its provers' builds and reads them, and the kernel replay, the
    dependency scans and digests and the upstream scan read only the package's own facts, which `proved` returns unchanged
    beside its pins: each starts before `proved`, is given the package's own facts, and its result is judged as before (a
    recorded dependency digest that differs still fails). A package without `proved_in` has no wait to fill: its replay
    starts after its facts and the rest run in turn, as their Python would contend with the audits on the critical path."""
    pkg = tmp_path / "spec"
    pkg.mkdir()
    (pkg / "lean-toolchain").write_text("leanprover/lean4:v0\n")
    A.dump(pkg, {"roots": ["P"], "upstream": {"scan": {}}, "dependencies": {"mathlib": "d0"}} |
           ({"proved_in": ["../proofs"]} if spec else {}))
    own = {"lean": "v0", "modules": ["P"], "declarations": 1, "axioms": [], "escapes": [], "oleans": ["own.olean"],
           "pins": {}}
    order, given = [], {}

    class Recording(A.ThreadPoolExecutor):
        def submit(self, fn, *args, **kw):
            if fn is A.timed:
                order.append(getattr(args[0], "__name__", str(args[0])))
            return super().submit(fn, *args, **kw)

    def proved(pkg, policy, facts, work, moved, ready):
        order.append("proved")
        return [], facts | {"pins": {"P.g": {"kind": "theorem"}}, "oleans": ["prover.olean"]}, {"../proofs": ["P.g"]}

    def stub(name, result):
        def f(*args):
            given[name] = args
            return result
        f.__name__ = name
        return f

    monkeypatch.setattr(A, "ThreadPoolExecutor", Recording)
    monkeypatch.setattr(A, "facts_of", lambda *a: own)
    monkeypatch.setattr(A, "proved", proved)
    monkeypatch.setattr(A, "replay_step", stub("replay_step", ([], {"checked": 1, "reused": 0})))
    monkeypatch.setattr(A, "dependency_scans", stub("dependency_scans", ([A.Failure("unchecked.dependency", None, "unchecked: a hit")], 7)))
    monkeypatch.setattr(A, "dependency_digests", stub("dependency_digests", {"mathlib": "d1"}))
    monkeypatch.setattr(A, "sibling", lambda name: type("U", (), {"pinned": staticmethod(stub("pinned", ({"x": 1}, [])))}))
    for name, value in (("judge", ([], {})), ("dropped", []), ("spec_reads", []), ("shape", []), ("unread_spec", {}),
                        ("read_by_dependents", set()), ("reads_home", {}), ("scans", []), ("heartbeats", []),
                        ("anonymous_instances", []), ("run_paths", []), ("listings", ([], {}, [])), ("regenerated", None)):
        monkeypatch.setattr(A, name, lambda *a, value=value, **k: value)
    rep = A.audit(pkg, tmp_path / "out", runs=False, log=lambda *_: None)
    assert order == (["pinned", "replay_step", "dependency_scans", "dependency_digests", "proved"] if spec else ["replay_step"])
    assert given["replay_step"][2] is own and given["dependency_scans"][1] == own["oleans"] == given["dependency_digests"][2]
    assert rep["failures"][0] == "unchecked: a hit" and rep["failures"][1].startswith("dependencies: the .olean files of the")
    assert rep["dependency_sources"] == 7 and rep["upstream"] == {"x": 1} and rep["replay"] == {"checked": 1, "reused": 0}
    assert {"audit", "dependency-sources", "upstream", "dependencies", "replay"} <= set(rep["seconds"])


def _runs(tmp_path, monkeypatch, entry, exit=0):
    """A package whose exempt module `T` is a test under `runs`, the commands it would run (`lake` faked to exit ``exit``; a
    `generate` runs for real), and a function auditing it: (failures, the `runs` section as found)."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "T.lean").write_text("")
    ran = []

    def run(argv, cwd, log=None, timeout=0):
        ran.append(argv)
        if argv[0] == "lake":
            return exit, "lake's output"
        r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    monkeypatch.setattr(A, "run", run)
    monkeypatch.setattr(A, "sandboxed", lambda: False)
    policy = {"roots": ["P"], "exempt": {"T": "a test"}, "runs": {"T": entry}}
    files = {"T": "T.lean"}
    return pkg, ran, lambda update=False, policy=policy: A.tests(pkg, policy, files, tmp_path / "work", tmp_path, update)


def test_a_run_reads_its_data_only_with_the_recorded_sha256(tmp_path, monkeypatch):
    import gzip

    entry = {"args": ["{d/v.txt}", "7"], "data": {"d/v.txt": sha("12\n")}}
    pkg, ran, tests = _runs(tmp_path, monkeypatch, entry)
    (pkg / "d").mkdir()
    (pkg / "d" / "v.txt.gz").write_bytes(gzip.compress(b"12\n"))
    assert tests() == ([], {"T": entry})
    unpacked = pkg / ".lake" / "runs" / "T" / "data" / "d" / "v.txt"
    assert ran == [["lake", "env", "lean", "--run", "T.lean", str(unpacked), "7"]] and unpacked.read_text() == "12\n"
    (pkg / "d" / "v.txt.gz").write_bytes(gzip.compress(b"13\n"))
    fails, _ = tests()
    assert [f.split(" has ")[0] for f in fails] == ["runs: T's data d/v.txt"] and len(ran) == 1      # failed unrun
    assert tests(update=True) == ([], {"T": entry | {"data": {"d/v.txt": sha("13\n")}}}) and len(ran) == 2
    (pkg / "d" / "v.txt.gz").unlink()
    assert tests()[0] == [f"runs: T's data d/v.txt is missing from {pkg}"]


def test_a_run_generates_its_data_and_fails_on_its_exit(tmp_path, monkeypatch):
    gen = "import pathlib, sys; pathlib.Path(sys.argv[2], 'v.txt').write_text(pathlib.Path(sys.argv[1]).name + '\\n')"
    entry = {"args": ["{v.txt}"], "data": {"v.txt": sha(f"{tmp_path.name}\n")}, "generate": ["python3", "-c", gen, "{repo}", "{out}"]}
    pkg, ran, tests = _runs(tmp_path, monkeypatch, entry, exit=1)
    out = pkg / ".lake" / "runs" / "T" / "generated"
    assert tests()[0] == ["runs: T exited 1:\nlake's output"]
    assert ran[0] == [A.sys.executable, "-c", gen, str(tmp_path), str(out)] and ran[1][-1] == str(out / "v.txt")
    bad = {"T": entry | {"generate": ["python3", "-c", "raise SystemExit(4)"]}}
    assert tests(policy={"exempt": {"T": "a test"}, "runs": bad})[0][0].startswith("runs: T's `generate` exited 4")
    assert tests(policy={"exempt": {}, "runs": {"T": entry}})[0] == [
        "runs: T is not under `exempt`: a run is a test, not a proof the audit reads"]
    assert tests(policy={"exempt": {"U": "?"}, "runs": {"U": entry}})[0] == ["runs: U is not a module of the package"]


def test_a_run_names_paths_outside_its_package_only_through_repo(tmp_path):
    pkg = tmp_path / "a" / "lean"
    pkg.mkdir(parents=True)
    runs = {"T": {"generate": ["python3", "../../b/gen.py", "{repo}", "--out", "{out}", "-c", "x = 1/2"],
                  "args": ["{v.txt}", "data/v.txt", "/etc/v.txt"]},
            "U": {"generate": ["python3", "{repo}/b/gen.py", "scripts/../gen.py", ".."]}}
    assert A.run_paths(pkg, {"runs": runs}, tmp_path) == [
        "runs: T's `generate` names ../../b/gen.py, outside the package: check's audit runs on a scratch tree, where it "
        "finds nothing or another file; write {repo}/b/gen.py",
        "runs: T's `args` names /etc/v.txt, outside the package: check's audit runs on a scratch tree, where it finds "
        "nothing or another file; write a path under `{repo}`",
        "runs: U's `generate` names .., outside the package: check's audit runs on a scratch tree, where it finds nothing "
        "or another file; write {repo}/a"]


def counts(rep: dict) -> tuple[int, int]:
    """The own modules a replay checked and reused."""
    return rep["replay"]["checked"], rep["replay"]["reused"]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_replay_reuses_only_modules_whose_olean_files_and_imports_passed_it(tmp_path):
    """On a real build: a second audit reuses every module's replay; a change to `Control.Defs` replays it and what imports
    it; a declaration that skipped the kernel is still rejected beside reused modules, and a failed replay stores the
    modules the kernel accepted (`Control`, which imports the bad one) and never the rejected one, which fails again."""
    pkg, cache = tmp_path / "control", tmp_path / "replay-cache"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    build = lambda: subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    audit = lambda: A.audit(pkg, tmp_path / "out", update=True, runs=False, log=lambda *_: None, replay_cache=cache)
    build()
    rep = audit()
    assert rep["failures"] == [] and counts(rep) == (4, 0)
    assert len(list(cache.glob("*.json"))) == 4
    rep = audit()
    assert rep["failures"] == [] and counts(rep) == (0, 4)
    defs = pkg / "Control" / "Defs.lean"
    defs.write_text(defs.read_text() + "\ndef Control.extra : Nat := 1\n")
    build()
    rep = audit()
    assert rep["failures"] == [] and counts(rep) == (3, 1), "Defs, Thms and Control; not Assumptions"
    kernel = json.loads((HERE / "fixtures" / "violations.json").read_text())["kernel"]
    for rel, text in kernel["write"].items():
        (pkg / rel).write_text(text)
    for rel, (old, new) in kernel["replace"].items():
        (pkg / rel).write_text((pkg / rel).read_text().replace(old, new))
    A.dump(pkg, A.load(pkg) | kernel["policy"])
    build()
    stored = set(cache.glob("*.json"))
    rep = audit()
    assert counts(rep) == (2, 3), "Bad and Control"
    assert any(f.startswith(kernel["fails"]) for f in rep["failures"]), rep["failures"]
    assert len(set(cache.glob("*.json")) - stored) == 1
    rep = audit()
    assert counts(rep) == (1, 4), "Bad alone"
    assert any(f.startswith(kernel["fails"]) for f in rep["failures"]), rep["failures"]


def test_the_closure_plan_joins_every_module_above_one_that_has_not_passed():
    """A dependency module that has not passed is checked, and every module importing it joins the replay unchecked, since
    an import of one would bring it in unchecked; the rest stay imported."""
    graph = {"Init": [], "Std": ["Init"], "A": ["Init"], "B": ["A", "Std"], "C": ["Std"], "Own": ["B", "C"]}
    facts = {"graph": {m: {"imports": i} for m, i in graph.items()}, "modules": ["Own"]}
    assert A.closure_plan(facts, lambda m: m != "A") == (["A"], ["B"])
    assert A.closure_plan(facts, lambda m: True) == ([], [])
    assert A.closure_plan(facts, lambda m: False) == (["A", "B", "C", "Init", "Std"], [])


def test_a_replay_key_covers_every_module_it_imports_transitively(tmp_path):
    """A module's key changes with the `.olean` of a module it imports through another, and with nothing else's."""
    oleans = {m: tmp_path / f"{m}.olean" for m in ("Init", "A", "B", "Other")}
    for m, p in oleans.items():
        p.write_text(m)
    graph = {"Init": [], "A": ["Init"], "B": ["A"], "Other": ["Init"]}
    facts = {"graph": {m: {"olean": str(oleans[m]), "imports": i} for m, i in graph.items()}, "modules": ["B"]}
    before = A.replay_keys(facts, "t")
    oleans["Init"].write_text("changed")
    after = A.replay_keys(facts, "t")
    assert all(before[m] != after[m] for m in graph)
    oleans["Init"].write_text("Init")
    oleans["Other"].write_text("changed")
    after = A.replay_keys(facts, "t")
    assert [m for m in graph if before[m] != after[m]] == ["Other"]
    assert A.replay_keys(facts, "another toolchain")["Init"] != before["Init"]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_closure_replay_checks_each_imported_module_once_and_then_reuses_it(tmp_path):
    """On a real build: `--closure` sends every module the control imports (the toolchain's `Init`) through the kernel, a
    second audit checks none of them, and a change to an own module checks only it and what imports it."""
    pkg, cache = tmp_path / "control", tmp_path / "replay-cache"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    build = lambda: subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    audit = lambda: A.audit(pkg, tmp_path / "out", update=True, runs=False, log=lambda *_: None, replay_cache=cache,
                            closure=True)
    build()
    rep = audit()
    c = rep["replay"]["closure"]
    assert rep["failures"] == [] and counts(rep) == (4, 0) and c["checked"] == c["modules"] > 100 and c["reused"] == 0, rep
    print(f"closure cold: {c}, {rep['seconds']['replay']} s {rep['replay']['phases']}, peak {rep['replay']['peak_gib']} GiB")
    rep = audit()
    assert rep["failures"] == [] and counts(rep) == (0, 4)
    assert rep["replay"]["closure"] == c | {"checked": 0, "reused": c["modules"]}
    print(f"closure warm: {rep['replay']['closure']}, {rep['seconds']['replay']} s")
    defs = pkg / "Control" / "Defs.lean"
    defs.write_text(defs.read_text() + "\ndef Control.extra : Nat := 1\n")
    build()
    rep = audit()
    assert rep["failures"] == [] and counts(rep) == (3, 1) and rep["replay"]["closure"]["checked"] == 0


def test_a_replay_marks_every_entry_it_reuses_used_so_eviction_keeps_the_closure(tmp_path, monkeypatch):
    """The replay cache keeps its `REPLAYS_KEPT` newest entries by mtime, and a reuse never touched one: Mathlib's and
    ArkLib's entries, stored once and reused by every replay, were the first evicted, so Security's and Properties' replays
    checked 1,300 to 3,800 closure modules again in most of 26 Oct's Lean checks (150 to 260 s each). A replay now marks
    each entry it reuses, its own modules' and the closure's, and stores those it checked; nothing else is touched."""
    graph = {"Init": [], "Std": ["Init"], "Old": ["Init"], "A": ["Std"], "B": ["A"]}
    facts = {"graph": {m: {"imports": i} for m, i in graph.items()}, "modules": ["A", "B"], "axioms": []}
    cache = tmp_path / "replay"
    cache.mkdir()
    for m in ("Init", "Old", "A", "Unrelated"):
        (cache / f"{m}.json").write_text("{}")
        os.utime(cache / f"{m}.json", (1000, 1000))
    monkeypatch.setattr(A, "replay_keys", lambda facts, toolchain: {m: m for m in facts["graph"]})
    monkeypatch.setattr(A, "replay_threads", lambda pkg: None)
    argv = []

    def run(cmd, cwd, log, timeout=None):
        argv.extend(cmd)
        (tmp_path / "replay.json").write_text(json.dumps({"axioms": []}))
        return 0, ""

    monkeypatch.setattr(A, "run", run)
    fails, stats = A.replay_step(tmp_path, {"roots": ["A"]}, facts, "t", tmp_path, cache, closure=True)
    assert fails == [] and stats["checked"] == 1 and stats["reused"] == 1 and stats["closure"]["checked"] == 1
    assert argv[argv.index("--closure") + 1:] == ["Std"] and argv[argv.index("--reused") + 1] == "A"
    mtime = {p.stem: p.stat().st_mtime for p in cache.glob("*.json")}
    assert sorted(mtime) == ["A", "B", "Init", "Old", "Std", "Unrelated"]
    assert mtime["Init"] > 1000 and mtime["Old"] > 1000 and mtime["A"] > 1000 and mtime["Unrelated"] == 1000


def test_two_packages_storing_one_replay_key_at_once_both_succeed(tmp_path, monkeypatch):
    """The gate audits its packages in threads of one process, and two whose closures share a module store its replay key
    at once; with the temporary file named by the pid alone, the second store renamed the first's away and the first's
    rename failed (r20261020-021950-2eb0). The store between the first's write and its rename runs on another thread."""
    real, nested = os.replace, []

    def replace(src, dst):
        if not nested:
            nested.append(1)
            t = threading.Thread(target=A.store_replay, args=(tmp_path, "k", "M"))
            t.start()
            t.join()
        real(src, dst)

    monkeypatch.setattr(A.os, "replace", replace)
    A.store_replay(tmp_path, "k", "M")
    assert json.loads((tmp_path / "k.json").read_text())["module"] == "M" and not list(tmp_path.glob(".*.tmp"))


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_an_equation_lemma_two_modules_realize_counts_as_declared_by_the_imported_one(tmp_path):
    """`f.eq_1` is realized in B and in C, which don't import each other; D imports C and uses it. With B replayed and C
    imported (it passed), D's use of the lemma is declared by C, which D imports (#2626's train: ZkQueryRound's
    `match_7.eq_2` and the `else_eq` that SimZk and ZkEq both realize). With B and C both imported (E, replayed, imports
    B), the imported environment names one module for the lemma, B, and D's import of C still declares it (the next
    train: `Exists.choose.congr_simp`, realized in Init.Control.Lawful.Lemmas and in the UpwardEnumerable that
    Init.Data.Range.Polymorphic.Instances imports)."""
    pkg = tmp_path / "realized"
    (pkg / "R").mkdir(parents=True)
    shutil.copy(HERE / "fixtures" / "control" / "lean-toolchain", pkg)
    (pkg / "lakefile.toml").write_text('name = "realized"\ndefaultTargets = ["R"]\n\n[[lean_lib]]\nname = "R"\n')
    (pkg / "R.lean").write_text("import R.D\nimport R.E\n")
    (pkg / "R" / "A.lean").write_text("def f : Nat → Nat\n  | 0 => 1\n  | n + 1 => n\n")
    for m in ("B", "C"):
        (pkg / "R" / f"{m}.lean").write_text(f"import R.A\ntheorem {m.lower()} : f 0 = 1 := by rw [f]\n")
    (pkg / "R" / "D.lean").write_text("import R.C\ntheorem d : f 0 = 1 := by rw [f]\n")
    (pkg / "R" / "E.lean").write_text("import R.B\ntheorem e : True := trivial\n")
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    for closure in (["R.B"], ["R.E"]):
        r = subprocess.run(["lake", "env", "lean", "--run", str(A.TOOL / "Replay.lean"), str(tmp_path / "replay.json"),
                            "R.D", "--closure", *closure], cwd=pkg, capture_output=True, text=True, env=A.lake_env())
        assert r.returncode == 0, (closure, r.stderr + r.stdout)


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_checked_declaration_waits_for_what_an_unchecked_one_it_unfolds_needs(tmp_path):
    """D's `a_t : f = 6 := rfl` makes the kernel unfold `f`, from M, which joins the replay unchecked (it passed, and it
    imports G, which hasn't), and `f`'s body mentions G's `zz_g`, checked in the same replay. Ordered only by what it
    mentions itself, `a_t` went to the kernel before `zz_g` was in it: "unknown constant" (#2626's third train,
    `Flock.sizeOf_lt_of_mem_toArray` and `Std.DTreeMap.Raw.Const.toList`)."""
    pkg = tmp_path / "behind"
    (pkg / "R").mkdir(parents=True)
    shutil.copy(HERE / "fixtures" / "control" / "lean-toolchain", pkg)
    (pkg / "lakefile.toml").write_text('name = "behind"\ndefaultTargets = ["R"]\n\n[[lean_lib]]\nname = "R"\n')
    (pkg / "R.lean").write_text("import R.D\n")
    (pkg / "R" / "G.lean").write_text("def zz_g : Nat := 5\n")
    (pkg / "R" / "M.lean").write_text("import R.G\ndef f : Nat := zz_g + 1\n")
    (pkg / "R" / "D.lean").write_text("import R.M\ntheorem a_t : f = 6 := rfl\n")
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    r = subprocess.run(["lake", "env", "lean", "--run", str(A.TOOL / "Replay.lean"), str(tmp_path / "replay.json"), "R.D",
                        "--closure", "R.G", "--closure-reused", "R.M"], cwd=pkg, capture_output=True, text=True,
                       env=A.lake_env())
    assert r.returncode == 0, r.stderr + r.stdout


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_controls(tmp_path):
    toolchain = (A.FIXTURES / "control" / "lean-toolchain").read_text().strip()
    assert A.controls(toolchain, tmp_path, log=lambda *_: None) == []


def test_a_step_records_the_peak_memory_of_every_process_it_waited_for(tmp_path):
    """`run` reaps with `os.wait4`, so a step's `peak_gib` covers the grandchildren a `lake` runs, not only `lake` itself;
    the output and exit code are what `subprocess.run` gave.  It runs in an interpreter of its own: a child's `ru_maxrss`
    counts the resident set it was forked with, so in a test worker grown past 0.3 GiB even `true` peaked there (8577, 12 Oct)."""
    child = "import sys; b = bytearray(300 << 20); b[::4096] = b'x' * len(b[::4096]); print('out'); sys.exit(3)"
    script = f"""import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("lean_audit", {str(HERE.parent / "check.py")!r})
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
tmp = Path({str(tmp_path)!r})
rep = {{"seconds": {{}}}}
rc, text = A.run(["bash", "-c", sys.executable + " -c " + repr({child!r}) + "; exit $?"], tmp, tmp / "step.log")
assert (rc, text) == (3, "out\\n") and (tmp / "step.log").read_text() == "out\\n", (rc, text)
A.took(rep, "replay", 0.0)
assert rep["peak_gib"]["replay"] >= 0.3, rep
A.run(["true"], tmp)
A.took(rep, "fresh", 0.0)
assert rep["peak_gib"]["fresh"] < 0.3, rep                              # each step's own peak, not the run's so far
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def hyp(name, at, reaches, module=None, source="binder"):
    return {"source": source, "name": name, "at": at, "type": name, "head": None, "module": module, "reaches": reaches}


def part(hyps=(), inputs=(), conclusion=()):
    return {"inputs": [{"name": n, "at": a, "head": None} for n, a in inputs], "hypotheses": list(hyps),
            "conclusion": list(conclusion)}


ACCEPTS = hyp("h", 1, ["P.check", "Except.ok"])


def test_a_property_takes_one_acceptance_its_inputs_and_premises_and_concludes_what_follows():
    """The shape rule (`refusals`): one hypothesis that its function accepted, besides premises (a constant of a module
    under `shape.premises`); what follows the first input bound after the acceptance is the conclusion, so a later binder
    is the conclusion's own; a second acceptance is the first's pair, and a hypothesis that is neither is refused, and so is a field of a
    structure an input ranges over that is neither."""
    entry = {"function": "P.check", "outcome": "accept"}
    fits = part([ACCEPTS, hyp("q", 3, ["P.Q"])], inputs=[("x", 0), ("w", 2)], conclusion=["P.Good"])
    assert A.refusals(entry, [fits], []) == []
    loose = part([ACCEPTS, hyp("q", 2, ["P.Q"], module="P.Other")], inputs=[("x", 0)])
    assert [w.split(",")[0] for w in A.refusals(entry, [loose], [])] == ["takes q : q"]
    assert A.refusals(entry, [loose], ["P.Other"]) == []
    twice = part([ACCEPTS, hyp("h2", 2, ["P.check"])], inputs=[("x", 0)])
    assert A.refusals(entry, [twice], []) == [], "two acceptances are one, of the pair (a binding property)"
    field = part([hyp("s.hscope", 0, ["P.Scope"], source="field"), ACCEPTS], inputs=[("s", 0)])
    assert "a field of an input it ranges over" in A.refusals(entry, [field], [])[0]
    nonempty = part([hyp("W.nonempty", 0, ["Nonempty"], source="field") | {"head": "Nonempty"}, ACCEPTS], inputs=[("W", 0)])
    assert "a field of an input it ranges over" in A.refusals(entry, [nonempty], [])[0], "`Nonempty` is a hypothesis too"
    implied = part([ACCEPTS, hyp("hop", 2, ["P.check", "P.Q"]) | {"guarded": True, "applies": ["P.Q"]}], inputs=[("x", 0)])
    assert "under a hypothesis of its own" in A.refusals(entry, [implied], [])[0]
    named = part([hyp("hm", 1, ["P.check", "P.S"]) | {"applies": ["P.S"]}], inputs=[("x", 0)])
    assert A.refusals(entry, [named], [])[0].startswith("takes no hypothesis"), "mentioning is not applying"
    assert "takes no hypothesis that its function returned accept" in A.refusals(entry, [part(inputs=[("x", 0)])], [])[0]
    assert A.refusals({"function": "P.check", "outcome": "reject"}, [fits], [])[0].startswith("takes no hypothesis")
    assert A.refusals({"function": "P.check", "outcome": "reject"},
                      [part([hyp("h", 1, ["P.check", "P.reject"])], inputs=[("x", 0)])], []) == []
    assert A.refusals(None, [fits], [])[0].startswith("has no entry under shape.properties")


def test_a_statement_in_another_order_or_with_a_shared_field_fits_as_its_restatement_would():
    """LEAN-47: `OnePair` bound `w k` before its two acceptances and `LiveCoinBinding` took `c.root = c'.root`, each
    equivalent to a form that fits. Binders reorder with the acceptances, the premises and what they mention first, so a
    hypothesis on a binder the acceptance doesn't read is the conclusion's, when the conclusion reads that binder too; an
    equation of one field of two inputs the acceptance reads is the field bound once. Nothing else moves: a hypothesis on
    what the acceptance reads, one whose proof a kept binder mentions, a structure's fields, a probability statement's
    hypotheses, and any statement whose facts don't say what each binder mentions are judged as written."""
    entry = {"function": "P.check", "outcome": "accept"}
    acc = lambda name, at, uses: hyp(name, at, ["P.check"]) | {"uses": uses}
    q = lambda name, at, uses, **kw: hyp(name, at, ["P.Q"]) | {"uses": uses} | kw
    inputs = lambda *ns: [{"name": n, "at": a, "head": None, "uses": []} for n, a in ns]
    names = lambda got: [w.split(" : ")[0] for w in A.refusals(entry, [got], [])]
    # ∀ m m' w k, check m → check m' → EpochRoot m w k → EpochRoot m' w k → ∃ …, P m w k
    one_pair = {"inputs": inputs(("m", 0), ("m'", 1), ("w", 2), ("k", 3)), "conclusion": ["P.Good"], "uses": [0, 1, 2, 3],
                "hypotheses": [acc("h", 4, [0]), acc("h'", 5, [1]), q("e", 6, [0, 2, 3]), q("e'", 7, [1, 2, 3])]}
    assert names(one_pair) == []
    assert names(one_pair | {"uses": [0, 1]}) == ["takes e", "takes e'"], "`∀ w k, E w k → C`, C without w or k, is a hypothesis"
    assert names({k: v for k, v in one_pair.items() if k != "uses"}) == ["takes e", "takes e'"], "facts without `uses`"
    reads = one_pair | {"hypotheses": [acc("h", 4, [0, 2]), *one_pair["hypotheses"][1:]]}
    assert names(reads) == [], "the acceptance reads w, and E m w k still picks out the conclusion's k"
    reads = one_pair | {"hypotheses": [acc("h", 4, [0, 2, 3]), *one_pair["hypotheses"][1:]]}
    assert names(reads) == ["takes e", "takes e'"], "the acceptance reads w and k: E m w k is a hypothesis on what it reads"
    proof = one_pair | {"hypotheses": [q("e", 4, [0, 2, 3]), acc("h", 5, [0, 4]), acc("h'", 6, [1])]}
    assert names(proof) == ["takes e"], "the acceptance mentions e's proof"
    fields = one_pair | {"hypotheses": [*one_pair["hypotheses"], hyp("w.ok", 2, ["P.Ok"], source="field")]}
    assert names(fields) == ["takes w.ok", "takes e", "takes e'"], "a structure's field stays a hypothesis"
    event = {"head": "G.prob", "conjuncts": [{"applies": ["P.check"], "guarded": False}, {"applies": ["P.Bad"], "guarded": False}]}
    late = one_pair | {"hypotheses": one_pair["hypotheses"][2:], "events": [event]}
    assert [w.split(" : ")[0] for w in A.refusals(entry, [late], [], events=["G.prob"])] == ["takes e", "takes e'"]
    # ∀ c c' o o', c.root = c'.root → accepts c o → accepts c' o' → …
    root = {"inputs": inputs(("c", 0), ("c'", 1), ("o", 2), ("o'", 3)), "conclusion": ["P.Good"], "uses": [0, 1, 2, 3],
            "hypotheses": [q("r", 4, [0, 1], shares={"inputs": [0, 1], "path": ["P.C.root"]}),
                           acc("h", 5, [0, 2]), acc("h'", 6, [1, 3])]}
    assert names(root) == []
    for bad, why in ((root | {"uses": [0, 1, 4]}, "the conclusion mentions its proof"),
                     (root | {"hypotheses": [root["hypotheses"][0], acc("h", 5, [0, 2])], "uses": [0, 2]},
                      "the acceptance doesn't read c', nor the conclusion"),
                     (root | {"hypotheses": [root["hypotheses"][0] | {"shares": {"inputs": [0, 1], "path": []}},
                                             *root["hypotheses"][1:]]}, "`c = c'` binds no field"),
                     (root | {"hypotheses": [root["hypotheses"][0] | {"shares": {"inputs": [0, 0], "path": ["P.C.root"]}},
                                             *root["hypotheses"][1:]]}, "one input")):
        assert names(bad) == ["takes r"], why


def test_a_function_that_always_returns_takes_no_acceptance_and_each_conjunct_must_be_about_it():
    """Outcome `none` (`step`, a constant): no acceptance, the function anywhere in the statement; a conjunction is a
    property when each conjunct is, so a conjunct of bare arithmetic is refused by its number."""
    entry = {"function": ["P.step", "P.LIMIT"], "outcome": "none"}
    assert A.refusals(entry, [part(inputs=[("s", 0)], conclusion=["P.step"]), part(conclusion=["P.LIMIT"])], []) == []
    out = A.refusals(entry, [part(conclusion=["P.step"]), part(conclusion=["Nat", "LE.le"])], [])
    assert out == ["(conjunct 2) says nothing about P.step or P.LIMIT, the function its entry names"]
    assert A.reaches("P.protocol.check", {"P.protocol", "P.Protocol.check"})
    assert not A.reaches("P.protocol.check", {"P.protocol"})


def test_an_entry_may_name_each_conjuncts_own_function_and_outcome():
    """`conjuncts` in an entry: soundness of a check and simulation of a server, one statement, each conjunct held to its
    own function and outcome, so the second needs no acceptance of the first's function; the list has one entry per
    conjunct, each naming both, and replaces the entry's own function and outcome rather than joining them."""
    sound = part([ACCEPTS], inputs=[("x", 0)], conclusion=["P.Good"])
    sim = part(inputs=[("s", 0)], conclusion=["P.run"])
    entry = {"conjuncts": [{"function": "P.check", "outcome": "accept"}, {"function": "P.run", "outcome": "none"}]}
    assert A.refusals(entry, [sound, sim], []) == []
    assert A.refusals({"function": "P.check", "outcome": "accept"}, [sound, sim], [])[0].startswith(
        "(conjunct 2) takes no hypothesis that its function returned accept")
    swapped = {"conjuncts": entry["conjuncts"][::-1]}
    assert {w.split(")")[0] for w in A.refusals(swapped, [sound, sim], [])} == {"(conjunct 1", "(conjunct 2"}
    for bad in ({"conjuncts": entry["conjuncts"][:1]}, {"conjuncts": [entry["conjuncts"][0], {"function": "P.run"}]},
                {"conjuncts": "P.run"}):
        assert A.refusals(bad, [sound, sim], [])[0].startswith("has 2 conjunct(s)"), bad
    both = entry | {"function": "P.check"}
    assert "both `conjuncts` and a function" in A.refusals(both, [sound, sim], [])[0]


def test_a_probability_statement_has_its_acceptance_in_the_event():
    """`shape.events`: a conjunct bounding `prob (fun o => accepts o ∧ … ∧ fails o)` takes premises only, so its
    hypotheses, an acceptance outside the event among them, are each refused; an event of a constant not listed is not
    read, and one with a conjunct before its last that isn't an acceptance (a hypothesis inside the event), one whose
    acceptance holds under a hypothesis, or one with no conjunct after its acceptance, is refused."""
    entry, prob = {"function": ["P.check", "P.verify"], "outcome": "accept"}, ["G.prob"]
    c = lambda applies, guarded=False: {"applies": applies, "guarded": guarded}
    event = lambda *cs, head="G.prob": {"head": head, "conjuncts": list(cs)}
    held = part([hyp("hA3", 0, ["W.Uniform"], module="W.Premises"), hyp("hk", 2, ["LE.le"]),
                 hyp("zr.hown", 1, ["P.check"], source="field"), ACCEPTS], inputs=[("zr", 1), ("k", 2)])
    fits = held | {"events": [event(c(["P.check", "P.Accepts"]), c(["P.verify"]), c(["Not", "P.holds"])),
                              event(c(["P.check"]), c(["P.check"])), event(c(["P.Q"]), head="Exists")]}
    assert [w.split(" : ")[0] for w in A.refusals(entry, [fits], ["W.Premises"], events=prob)] == \
        ["takes zr.hown", "takes h", "takes hk"]
    assert "its event's first conjunct" in A.refusals(entry, [fits], ["W.Premises"], events=prob)[0]
    assert A.refusals(entry, [part(inputs=[("x", 0)]) | {"events": fits["events"]}], [], events=prob) == []
    assert A.refusals(entry, [fits], ["W.Premises"]) == [], "not listed: the old rule, which takes zr.hown as the acceptance"
    for bad in ([event(c(["P.check"]), c(["P.Q"]), c(["P.holds"]))], [event(c(["P.check"], guarded=True), c(["P.holds"]))],
                [event(c(["P.check"]))], [fits["events"][0], event(c(["P.Q"]), c(["P.holds"]))]):
        got = A.refusals(entry, [part(inputs=[("x", 0)]) | {"events": bad}], [], events=prob)
        assert [w.split(" whose")[0].split(" that")[0] for w in got] == ["bounds an event of G.prob", "takes no hypothesis"], bad
        assert A.refusals(entry, [fits | {"events": bad}], ["W.Premises"], events=prob)[0].startswith("bounds an event")


def test_the_frozen_list_exempts_a_guarantee_until_it_fits_and_then_must_shrink():
    """`shape` with `shape.frozen`: a frozen guarantee that doesn't fit passes, one that fits fails until its entry goes,
    an entry needs a reason and a guarantee or unstated statement of this package, an unfrozen misfit fails, as does a
    guarantee whose shape `Facts.lean` couldn't compute, and a package with no `shape` section is checked all the same."""
    entries = {"P.a": {"function": "P.check", "outcome": "accept"}, "P.b": {"function": "P.check", "outcome": "accept"}}
    pol = lambda frozen: {"guarantees": {"P.a": {}, "P.b": {}}, "shape": {"premises": [], "properties": entries, "frozen": frozen}}
    good, bad = [part([ACCEPTS], inputs=[("x", 0)])], [part(inputs=[("x", 0)])]
    facts = {"pins": {"P.a": {"shape": good}, "P.b": {"shape": bad}}}
    assert A.shape(pol({"P.b": "restatable: split the iff."}), facts) == []
    fails = A.shape(pol({"P.b": "restatable: split it", "P.a": " ", "P.z": "frozen: gone"}), facts)
    assert [f.split()[1] for f in fails] == ["shape.frozen", "shape.frozen[P.a]", "P.a"], fails
    assert "which only shrinks" in fails[2]
    assert [f.split()[1] for f in A.shape(pol({}), facts)] == ["P.b"]
    assert [f.split()[1] for f in A.shape({"guarantees": {"P.b": {}}}, facts)] == ["P.b"], "no shape section: still checked"
    assert A.shape({"guarantees": {"P.b": {}}, "shape": {"frozen": {"P.b": "frozen: not swept yet"}}}, facts) == []
    loose = facts | {"spec_definitions": [{"name": "P.A", "module": "P.Guarantees", "kind": "def", "statement": True},
                                          {"name": "P.Pred", "module": "P.Guarantees", "kind": "def", "statement": False},
                                          {"name": "P.Cond", "module": "P.Defs", "kind": "def", "statement": True}],
                     "pins": {"P.a": {"shape": good, "lift": {"head": "P.A"}}, "P.b": {"shape": good}}}
    assert A.shape(pol({}), loose) == []
    loose["pins"]["P.a"]["lift"] = None
    assert [f.split()[1] for f in A.shape(pol({}), loose)] == ["P.A"], "a statement no guarantee states fails"
    held = pol({}) | {"shape": pol({})["shape"] | {"statements": ["P.Defs"]}}
    assert [f.split()[1] for f in A.shape(held, loose)] == ["P.Cond"], "where shape.statements says statements are"
    assert A.shape(pol({"P.A": "unstated: memory-accounting lists it."}), loose) == [], "until shape.frozen names it"
    loose["pins"]["P.a"]["lift"] = {"head": "P.A"}
    assert "neither a guarantee" in A.shape(pol({"P.A": "unstated: listed now"}), loose)[0]
    gone = {"pins": {"P.a": {"shape": good}, "P.b": {"shape": {"error": "boom"}}}}
    assert A.shape(pol({}), gone) == ["shape: P.b has no shape in the facts (boom)"], "a shape Facts couldn't read fails"


def test_a_frozen_reason_starts_with_its_category_and_an_unstated_statement_is_unstated():
    """Each `shape.frozen` reason starts with one of `CATEGORIES`, so a record counts its guarantees by category: a
    guarantee's isn't `unstated`, and a statement no guarantee states has only that one."""
    pol = {"guarantees": {"P.b": {}}, "shape": {"frozen": {"P.b": "RESTATE: split the iff"}}}
    facts = {"pins": {"P.b": {"shape": [part(inputs=[("x", 0)])]}},
             "spec_definitions": [{"name": "P.A", "module": "P.Guarantees", "kind": "def", "statement": True}]}
    out = A.shape(pol | {"shape": {"frozen": {"P.b": "RESTATE: split the iff", "P.A": "unstated: owed"}}}, facts)
    assert [f.split()[1] for f in out] == ["shape.frozen[P.b]"] and "starts with no category" in out[0]
    out = A.shape(pol | {"shape": {"frozen": {"P.b": "unstated: x", "P.A": "lemma: y"}}}, facts)
    assert sorted(f.split()[1] for f in out) == ["shape.frozen[P.A]", "shape.frozen[P.b]"], out
    assert A.category("falsified (conjecture): rests on P.C") == "falsified" and A.category("lemmas") is None


def test_a_property_on_a_conjecture_lists_it_and_one_on_a_falsified_conjecture_comes_off():
    """`shape.conjectures`: a hypothesis whose head is a named conjecture is a premise, and a property resting on one
    (a hypothesis reaching it, even inside an existential) lists it under `rests_on`, so it is listed as conditional; a
    conjecture marked falsified refuses every property resting on it, which stays frozen only as `falsified`; an open
    conjecture names its falsification test, and every entry is a named assumption with a reason."""
    entry = {"function": "P.check", "outcome": "accept"}
    rests = part([ACCEPTS, hyp("c", 2, ["P.Conj"]) | {"head": "P.Conj"}], inputs=[("x", 0)])
    pol = lambda status, frozen=None, rests_on=("P.Conj",): {
        "assumptions": {"P.Conj": {}}, "guarantees": {"P.a": {}},
        "shape": {"properties": {"P.a": entry | {"rests_on": list(rests_on)}}, "frozen": frozen or {},
                  "conjectures": {"P.Conj": {"status": status, "test": "tests/test_conj.py", "why": "open since Oct 8"}}}}
    facts = {"pins": {"P.a": {"shape": [rests]}}}
    assert A.shape(pol("open"), facts) == []
    assert "but its entry's `rests_on` lists none" in A.shape(pol("open", rests_on=()), facts)[0]
    out = A.shape(pol("falsified"), facts)
    assert len(out) == 1 and "which shape.conjectures marks falsified" in out[0], out
    assert A.shape(pol("falsified", {"P.a": "falsified (conjecture): rests on P.Conj"}), facts) == []
    out = A.shape(pol("falsified", {"P.a": "frozen: rests on P.Conj"}), facts)
    assert out and "its reason's category is `falsified`" in out[0], out
    deep = {"pins": {"P.a": {"shape": [part([ACCEPTS, hyp("e", 2, ["Exists", "P.Conj"])], inputs=[("x", 0)])]}}}
    assert "marks falsified" in " ".join(A.shape(pol("falsified", {}), deep)), "a conjecture inside a hypothesis counts"
    bad = pol("open") | {"assumptions": {}}
    bad["shape"]["conjectures"]["P.Conj"] = {"status": "open"}
    out = A.shape(bad, facts)
    assert ["not a named assumption" in out[0], "gives no reason" in out[1], "names no falsification test" in out[2]] \
        == [True] * 3, out
    bad["shape"]["conjectures"]["P.Conj"] = {"status": "maybe", "why": "x"}
    assert "has no status" in A.shape(bad, facts)[1]


def test_a_dev_cache_run_restores_dependencies_and_is_never_a_verdict(monkeypatch):
    """`--dev-cache` lets setup.sh and Lake restore the verified build cache (no `LEAN_CACHE=off`, no
    `LAKE_ARTIFACT_CACHE=false`), so its reports carry the fast path's kind, which `check` and `research merge` refuse:
    the gate's from-source build is the cache's root of trust and must never read it."""
    monkeypatch.setattr(A, "ARTIFACTS", None)
    monkeypatch.setattr(A, "DEV_CACHE", False)
    env = A.lake_env()
    assert env["LEAN_CACHE"] == "off" and env["LAKE_ARTIFACT_CACHE"] == "false"
    monkeypatch.setenv("LEAN_CACHE", "off")
    monkeypatch.setenv("LAKE_ARTIFACT_CACHE", "false")
    monkeypatch.setattr(A, "DEV_CACHE", True)
    env = A.lake_env()
    assert "LEAN_CACHE" not in env and "LAKE_ARTIFACT_CACHE" not in env
    spec = importlib.util.spec_from_file_location("lean_changed_under_test", HERE.parent / "lean_changed.py")
    LC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LC)
    assert A.DEV_KIND == LC.FAST_KIND


def test_a_dev_cache_setup_restores_the_packages_required_by_path_too(monkeypatch, tmp_path):
    """Security/Proofs at the published commit rebuilt 1,607 of its modules with `--dev-cache` (21 Oct), because only its
    own build was restored: Security and Flock, which it requires by path, were built again, not byte-identically, and
    everything above them compiled. With `--dev-cache`, setup restores each path dependency's build too, lending one with no fetched dependencies of its own the root's for the restore; without it, or
    after a failed setup.sh, it restores nothing more."""
    calls, seen = [], []
    deps = [tmp_path / "Security", tmp_path / "flock"]
    (tmp_path / "Proofs" / ".lake" / "packages" / "mathlib").mkdir(parents=True)
    (tmp_path / "flock" / ".lake" / "packages").mkdir(parents=True)

    def run(argv, cwd, log=None, timeout=0):
        calls.append(argv)
        seen.append({d.name: (d / ".lake" / "packages").is_symlink() for d in deps})
        return 0, ""

    monkeypatch.setattr(A, "run", run)
    monkeypatch.setattr(A, "path_deps", lambda p: deps)
    monkeypatch.setattr(A, "DEV_CACHE", True)
    assert A.dev_restore(tmp_path / "Proofs", 0, "", tmp_path / "setup.log") == (0, "")
    assert calls == [[A.sys.executable, str(A.TOOL / "cache.py"), "get", *map(str, deps)]]
    assert seen == [{"Security": True, "flock": False}] and not (deps[0] / ".lake" / "packages").exists()
    calls.clear()
    assert A.dev_restore(tmp_path / "Proofs", 2, "x", tmp_path / "setup.log") == (2, "x") and calls == []
    monkeypatch.setattr(A, "DEV_CACHE", False)
    assert A.dev_restore(tmp_path / "Proofs", 0, "", tmp_path / "setup.log") == (0, "") and calls == []
