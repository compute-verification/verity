"""`tools/lean/audit.py`: its judgement on the facts `Facts.lean` reports (no Lean needed), and the controls when `lake` is
installed (the fixture package passes clean and fails each check once a violation is added)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_audit", HERE.parent / "audit.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

STD = ["Classical.choice", "Quot.sound", "propext"]


def sha(s: str) -> str:
    """A stand-in SHA-256 content hash, tied to the stand-in 32-bit one."""
    return hashlib.sha256(s.encode()).hexdigest()


def fact(signature="P.t (h : P.Hyp) (n : Nat) : P.f n = n", hyps=None, reads=None, axioms=STD, kind="theorem", h="00000000000000aa"):
    reads = reads if reads is not None else [{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "1" * 32},
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
POLICY = {"roots": ["P"], "assumptions": ["P.Assumptions"], "guarantees": {"P.t": {}}}


def recorded(f=None):
    fails, pol = A.judge(POLICY, f or facts(), FILES, update=True)
    assert fails == []
    return pol


def test_a_recorded_package_passes_and_its_record_is_the_statement():
    pol = recorded()
    assert pol["guarantees"]["P.t"] == {"signature": ["P.t (h : P.Hyp) (n : Nat) : P.f n = n"], "assumptions": ["P.Hyp"],
                                        "type_hash": sha("00000000000000aa")}
    assert len(pol["reads"]["P.Defs"]["digest"]) == 64 and len(pol["reads"]["P.Defs"]["definitions"]["P.f"]) == 64
    assert list(pol["reads"]["P.Defs"]["definitions"]) == ["P.f"] and pol["reads"]["P.Defs"]["guarantees"] == ["P.t"]
    assert all(g["digest"] == A.module_digest(g["definitions"]) for g in pol["reads"].values())
    assert A.judge(pol, facts(), FILES) == ([], pol)
    assert A.judge(POLICY, facts(), FILES)[0] == ["guarantees: P.t has no record; run `audit.py --update` and commit the record"]


def test_every_field_of_a_reads_record_is_checked():
    pol = recorded()
    group = pol["reads"]["P.Defs"]
    stale = pol | {"reads": pol["reads"] | {"P.Defs": group | {"digest": "0" * 64}}}
    assert A.judge(stale, facts(), FILES)[0] == ["guarantees: the digest recorded for P.Defs is not the SHA-256 of its definitions; "
                                                 "`python tools/lean/merge.py --normalize` recomputes it"]
    readers = pol | {"reads": pol["reads"] | {"P.Defs": group | {"guarantees": ["P.t", "P.u"]}}}
    assert A.judge(readers, facts(), FILES)[0] == ["guarantees: those that read P.Defs are P.t; the record lists "
                                                   "P.t, P.u; run `audit.py --update`"]


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


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_sha256_hash_is_expr_hash_s_equivalence(tmp_path):
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    (pkg / "Control" / "Alpha.lean").write_text(
        "namespace Control\ntheorem a (x : Nat) : x + 0 = x := rfl\ntheorem b (y : Nat) : y + 0 = y := rfl\n"
        "theorem c {x : Nat} : x + 0 = x := rfl\ntheorem d (x : Nat) : 0 + x = x := Nat.zero_add x\nend Control\n")
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.Alpha\n")
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    f = A.facts_of(pkg, {"roots": ["Control"], "guarantees": {f"Control.{n}": {} for n in "abcd"}}, tmp_path)
    got = {n: f["pins"][f"Control.{n}"]["type_sha256"] for n in "abcd"}
    assert got["a"] == got["b"] == got["c"] != got["d"] and len(got["a"]) == 64 and not (tmp_path / "canon.jsonl").exists()


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_moved_declarations_are_compared_under_their_old_names(tmp_path):
    """`Facts.lean`'s renaming and unfolding on a real build: a guarantee restated as a `Prop` definition (its old theorem kept as
    the lemma that proves it), one whose statement holds a proof Lean abstracts out of a definition (`_proof_1`), and a
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
    A.dump(pkg, A.load(pkg) | {"guarantees": {"Control.main_thm": {}, "Control.a": {}, "Control.b": {}, "Control.c": {}}})
    build = lambda: subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    audit = lambda **kw: A.audit(pkg, tmp_path / "out", update=True, replay=False, runs=False, log=lambda *_: None, **kw)
    build()
    assert audit()["failures"] == []
    before = (pkg / A.POLICY).read_text()
    record = tmp_path / "out" / "audit-control" / A.POLICY
    assert record.read_text() == before, "the record beside the reports, where a recorded run publishes it"
    record.unlink()
    assert audit()["failures"] == [] and (pkg / A.POLICY).read_text() == before
    assert record.read_text() == before, "an update that changes nothing writes it all the same"
    (pkg / A.POLICY).write_text(before.replace('"guarantees"', '"pins"'))
    rep = audit()
    assert rep["failures"] == [] and "review" not in rep and (pkg / A.POLICY).read_text() == before, "renamed, not reviewed"
    moved = {"Control.a": "Control.SecurityProofs.a", "Control.b": "Control.SecurityProofs.b",
             "Control.c": "Control.SecurityProofs.c", "Control.f": "Control.Protocol.f", "Control.g": "Control.Protocol.g"}
    after = ("def Control.Protocol.f (n : Nat) : Nat := n + 0\ndef Control.Protocol.g (n : Nat) : Nat := 0 + n\n"
             "def Control.Guarantees.A : Prop := ∀ n, Control.Protocol.f n = n\n"
             "theorem Control.a (n : Nat) : Control.Protocol.f n = n := rfl\n"
             "theorem Control.SecurityProofs.a : Control.Guarantees.A := Control.a\n"
             "theorem Control.SecurityProofs.b (n : Nat) : Control.Protocol.g n = n := Nat.zero_add n\n"
             f"def Control.Guarantees.C : Prop := ∀ n : Nat, {fin}\ntheorem Control.c (n : Nat) : {fin} := rfl\n"
             "theorem Control.SecurityProofs.c : Control.Guarantees.C := Control.c\n")
    moves.write_text(after)
    build()
    rep = audit(moved=moved)
    assert rep["failures"] == []
    assert rep["before"] == {k: json.loads(before)[k] for k in ("guarantees", "reads")} and rep["guarantees"] == A.load(pkg)["guarantees"]
    must = A.statement_changes(rep["review"])
    assert set(rep["review"]) - set(must) == {
        f"guarantee Control.SecurityProofs.a: moved from Control.a, restated as Control.Guarantees.A: {A.MOVED}",
        f"guarantee Control.SecurityProofs.c: moved from Control.c, restated as Control.Guarantees.C: {A.MOVED}",
        f"guarantee Control.SecurityProofs.b: moved from Control.b: {A.MOVED}", f"reads: 1 {A.MOVED_READS}"}
    assert must[0] == ("definition Control.Protocol.g (Control.Moves), was Control.g (Control.Moves), read by "
                       "Control.SecurityProofs.b: changed") and must[-1] == "    def Control.Protocol.g (n : Nat) : Nat := 0 + n"
    (pkg / A.POLICY).write_text(before)
    moves.write_text("def Control.f (n : Nat) : Nat := n\n" + after.replace("f n = n\n", "f n = Control.f n\n"))
    build()
    rep = audit(moved=moved)
    assert [f.split(":")[0] for f in rep["failures"]] == ["moved"] and "Control.Protocol.f would be Control.f" in rep["failures"][0]


def test_generated_constants_belong_to_their_declaration():
    assert [A.parent(n) for n in ("P.f.match_1", "P.walk._f", "P.Enc.mk", "P.instDecidableFits", "mk")] == [
        "P.f", "P.walk", "P.Enc", "P.instDecidableFits", "mk"]


def test_update_prints_what_a_statement_reviewer_reads(tmp_path):
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
    its type is that `Prop` definition, whose value under the old names hashes like ``as_was``."""
    reads = [{"name": "Q.f", "module": "Q.Defs", "kind": "def", "hash": f, "moved": {"as": "P.f", "sha256": sha(f_was)}},
             {"name": "Q.f.match_1", "module": "Q.Defs", "kind": "def", "hash": "2" * 32,
              "moved": {"as": "P.f.match_1", "sha256": sha("2" * 32)}},
             {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "3" * 32,
              "moved": {"as": "P.Hyp", "sha256": sha("3" * 32)}}, *extra]
    signature = f"{name} : {head}" if head else f"{name} (h : P.Hyp) (n : Nat) : Q.f n = n"
    p = fact(signature=signature, h=h, reads=[{k: v for k, v in r.items() if k != "moved"} for r in reads],
            hyps=[] if head else None)
    for r, m in zip(p["reads"], reads):
        r["moved"] = m["moved"]
    p["moved"] = {"as": was, "sha256": sha(head or as_was)} | ({"head": head, "unfolded_sha256": sha(as_was)} if head else {})
    return p


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


def test_a_statement_restated_as_a_prop_definition_is_the_same_statement():
    old = recorded()
    current = {**POLICY, "guarantees": {"Q.t": old["guarantees"]["P.t"]}}
    head = {"name": "Q.T", "module": "Q.Guarantees", "kind": "def", "hash": "5" * 32, "moved": {"as": "Q.T", "sha256": sha("5" * 32)}}
    now = facts(pins={"Q.t": moved_fact(head="Q.T", extra=[head])})
    new = A.judge(current, now, FILES, update=True)[1]
    assert new["guarantees"]["Q.t"]["signature"] == ["Q.t : Q.T"] and "Q.Guarantees" in new["reads"]
    text = A.review(old, new, now, {})
    assert text == [f"guarantee Q.t: moved from P.t, restated as Q.T: {A.MOVED}", f"reads: 1 {A.MOVED_READS}"]
    now = facts(pins={"Q.t": moved_fact(head="Q.T", extra=[head], as_was="00000000000000bb")})
    text = A.review(old, A.judge(current, now, FILES, update=True)[1], now, {})
    assert text[0] == "guarantee Q.t: changed (moved from P.t)" and "definition Q.T (Q.Guarantees), read by Q.t: new" in text


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
    assert fails == ["guarantees: P.t assumes (h : ∀ (m : Nat), m + 0 = m), a closed Prop that is not a named assumption (a Prop "
                     "constant of P.Assumptions)"]
    elsewhere = [{"binder": "h", "type": "Q.Hyp", "closed": True, "named": "Q.Hyp", "module": "Q.Stuff"}]
    assert A.judge(POLICY, facts(pins={"P.t": fact(hyps=elsewhere)}), FILES, update=True)[0][0].startswith("guarantees: P.t assumes (h : Q.Hyp)")


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
    assert rep["failures"][0].endswith('{"roots": ["Alpha", "Beta"], "guarantees": {}}')


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


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_controls(tmp_path):
    toolchain = (A.FIXTURES / "control" / "lean-toolchain").read_text().strip()
    assert A.controls(toolchain, tmp_path, log=lambda *_: None) == []
