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
    assert A.judge(POLICY, facts(), FILES)[0] == ["guarantees: P.t has no record; run `audit.py --update` and commit the record",
                                                  "assumptions: P.Hyp has no record; run `audit.py --update` and commit the record"]


def test_every_field_of_a_reads_record_is_checked():
    pol = recorded()
    group = pol["reads"]["P.Defs"]
    stale = pol | {"reads": pol["reads"] | {"P.Defs": group | {"digest": "0" * 64}}}
    assert A.judge(stale, facts(), FILES)[0] == ["guarantees: the digest recorded for P.Defs is not the SHA-256 of its definitions; "
                                                 "`python tools/verity/lean/merge.py --normalize` recomputes it"]
    readers = pol | {"reads": pol["reads"] | {"P.Defs": group | {"guarantees": ["P.t", "P.u"]}}}
    assert A.judge(readers, facts(), FILES)[0] == ["guarantees: those that read P.Defs are P.t; the record lists "
                                                   "P.t, P.u; run `audit.py --update`"]


def test_every_guarantee_has_an_owner_and_an_update_keeps_it():
    pol = recorded()
    bare = {**POLICY, "guarantees": {"P.t": {}}}
    no_owner = ('guarantees: P.t has no owner: write "owner": "@lane" in its entry, the handle of the lane that answers for '
                'it, or run `audit.py --update --owner @lane`')
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
        "`meaning` and run `audit.py --update`"]
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
        "`meaning` and run `audit.py --update`",
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
    for policy, now in [
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
                               (POLICY, facts(), FILES),                                   # no record yet
                               (pol, facts(pins={}), FILES),                               # renamed or gone: `--moved`
                               (pol | {"guarantees": {"P.t": pol["guarantees"]["P.t"] | {"owner": "lean"}}}, facts(), FILES),
                               (pol, signed, FILES | {"Scratch": "Scratch.lean"})]:        # a difference beside an orphan
        assert A.regenerated(policy, now, files, A.judge(policy, now, files)[0]) is None


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
    names = ["guarded", "mentions", "underForall", "family", "world", "event", "eventLate"]
    f = A.facts_of(pkg, {"roots": ["Control"], "guarantees": {f"Control.{n}": {} for n in names}}, tmp_path)
    entry = {"function": "Control.verify", "outcome": "accept"}
    got = {n: [w.split(" : ")[0].split(" whose")[0] for w in A.refusals(entry, f["pins"][f"Control.{n}"]["shape"],
                                                                        ["Control.ShapePrem"], events=["Control.prob"])]
           for n in names}
    assert got == {"guarded": ["takes _hop"], "mentions": ["takes _hm"], "underForall": [], "family": ["takes _Vs.hscope"],
                   "world": ["takes _W.nonempty"], "event": ["takes zs.hown", "takes _hq"],
                   "eventLate": ["bounds an event of Control.prob"]}, got


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
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
    pol = A.load(pkg)
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
    A.dump(pkg, A.load(pkg) | {"guarantees": {f"Control.{n}": {"owner": "@lean"} for n in ("main_thm", *"pqrstvw")}})
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
        f"every record whose definitions use it. Write `instance {n}` there (`audit.py --name-instances` writes each one)"
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
    and the record changes no statement."""
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
    pol = A.load(pkg)
    A.dump(pkg, pol | {"guarantees": {n: {"owner": "@lean"} for n in ("Control.main_thm", "Control.Sub.twice", "Control.Sub.refl",
                                                                        "Control.Sub.pick")},
                       "layers": pol["layers"] | {"Control.Guarantees": {"imports": ["Control.Defs"]}}})
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


def test_an_assumption_is_listed_by_name_with_its_record():
    """Each assumption has an owner and the digest of its definition as `reads` holds it, so a changed assumption is its own
    record; an entry no guarantee reads fails; an update converts a list of assumption modules into the definitions read
    there, owned by the owner of the guarantees that read them."""
    pol = recorded()
    assert pol["assumptions"] == {"P.Hyp": {"owner": "@lean", "digest": pol["reads"]["P.Assumptions"]["definitions"]["P.Hyp"]}}
    changed = facts(pins={"P.t": fact(reads=[{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "1" * 32},
                                             {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "4" * 32}])})
    assert "assumptions: P.Hyp's definition in P.Assumptions changed; run `audit.py --update` and commit the record " \
           "(Daniel's DM shows the change when it lands)" in A.judge(pol, changed, FILES)[0]
    unread = {**pol, "assumptions": pol["assumptions"] | {"P.Other": {"owner": "@lean", "digest": "0" * 64}}}
    assert A.judge(unread, facts(), FILES)[0] == ["assumptions: P.Other is listed, but no guarantee reads it; remove the entry"]
    assert A.judge(unread, facts(), FILES, update=True)[1]["assumptions"] == pol["assumptions"]
    unowned = {**pol, "assumptions": {"P.Hyp": {"digest": pol["assumptions"]["P.Hyp"]["digest"]}}}
    assert A.judge(unowned, facts(), FILES)[0] == ['assumptions: P.Hyp has no owner: write "owner": "@lane" in its entry, or '
                                                   'run `audit.py --update --owner @lane`']
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


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_replay_reuses_only_modules_whose_olean_files_and_imports_passed_it(tmp_path):
    """On a real build: a second audit reuses every module's replay; a change to `Control.Defs` replays it and what imports
    it; a declaration that skipped the kernel is still rejected beside reused modules, and a failed replay stores nothing."""
    pkg, cache = tmp_path / "control", tmp_path / "replay-cache"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    build = lambda: subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    audit = lambda: A.audit(pkg, tmp_path / "out", update=True, runs=False, log=lambda *_: None, replay_cache=cache)
    build()
    rep = audit()
    assert rep["failures"] == [] and rep["replay"] == {"checked": 4, "reused": 0}
    assert len(list(cache.glob("*.json"))) == 4
    rep = audit()
    assert rep["failures"] == [] and rep["replay"] == {"checked": 0, "reused": 4}
    defs = pkg / "Control" / "Defs.lean"
    defs.write_text(defs.read_text() + "\ndef Control.extra : Nat := 1\n")
    build()
    rep = audit()
    assert rep["failures"] == [] and rep["replay"] == {"checked": 3, "reused": 1}, "Defs, Thms and Control; not Assumptions"
    kernel = json.loads((HERE / "fixtures" / "violations.json").read_text())["kernel"]
    for rel, text in kernel["write"].items():
        (pkg / rel).write_text(text)
    for rel, (old, new) in kernel["replace"].items():
        (pkg / rel).write_text((pkg / rel).read_text().replace(old, new))
    A.dump(pkg, A.load(pkg) | kernel["policy"])
    build()
    stored = set(cache.glob("*.json"))
    rep = audit()
    assert rep["replay"] == {"checked": 2, "reused": 3}, "Bad and Control"
    assert any(f.startswith(kernel["fails"]) for f in rep["failures"]), rep["failures"]
    assert set(cache.glob("*.json")) == stored


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_controls(tmp_path):
    toolchain = (A.FIXTURES / "control" / "lean-toolchain").read_text().strip()
    assert A.controls(toolchain, tmp_path, log=lambda *_: None) == []


def test_a_step_records_the_peak_memory_of_every_process_it_waited_for(tmp_path):
    """`run` reaps with `os.wait4`, so a step's `peak_gib` covers the grandchildren a `lake` runs, not only `lake` itself;
    the output and exit code are what `subprocess.run` gave. A child's `ru_maxrss` starts at its parent's high-water
    resident set (Linux carries it across `exec`), so even `true` reports the test process's own size: the reset is
    checked against a peak no process here reaches."""
    child = "import sys; b = bytearray(300 << 20); b[::4096] = b'x' * len(b[::4096]); print('out'); sys.exit(3)"
    rep = {"seconds": {}}
    A._peak.bytes = 0
    rc, text = A.run(["bash", "-c", f"{sys.executable} -c \"{child}\"; exit $?"], tmp_path, tmp_path / "step.log")
    assert (rc, text) == (3, "out\n") and (tmp_path / "step.log").read_text() == "out\n"
    A.took(rep, "replay", 0.0)
    assert rep["peak_gib"]["replay"] >= 0.3
    A._peak.bytes = 64 << 30
    A.took(rep, "big", 0.0)
    A.run(["true"], tmp_path)
    A.took(rep, "fresh", 0.0)
    assert rep["peak_gib"]["big"] == 64.0 and rep["peak_gib"]["fresh"] < 64  # each step's own peak, not the run's so far


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
