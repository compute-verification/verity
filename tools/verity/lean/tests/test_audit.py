"""`tools/verity/lean/check.py`: its judgement on the facts `Facts.lean` reports (no Lean needed), and the controls when `lake` is
installed (the fixture package passes clean and fails each check once a violation is added)."""
from __future__ import annotations

import ast
import gzip
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
            "witness_failures": None, "four_name_failures": None, "inline_failures": 0, "assumption_records": 2, "shape": None, "spec_reads": None, "tree_failures": None, "scans": None, "listings": 0, "dependency_scans": 0,
            "assign": 1, "proved": 0, "replay_step": 0, "tests": 0, "run_paths": None, "sampled_recompile": 0}


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


@pytest.fixture(autouse=True)
def every_statement_change_names_its_entry(monkeypatch):
    """Every line of any review a test here provokes that heads a statement change (`statement_changes` keeps it, and it is
    no detail under another line) is a `Change`, so a report's `changes` lists it with its owners."""
    review = A.review

    def reviewed(*args, **kw):
        notes = review(*args, **kw)
        for n in A.statement_changes(notes):
            assert n[:1] == " " or (isinstance(n, A.Change) and n.kind in A.ORDER and n.entry), \
                f"a statement change without its kind and entry: {n!r}"
        return notes

    monkeypatch.setattr(A, "review", reviewed)


@pytest.fixture(autouse=True)
def seal_off_the_checkout(tmp_path_factory, monkeypatch):
    """`main` seals a run under `SEAL_BASE` (`ROOT/.lake/sealed`); here it seals under the test session's tmp instead, so a
    real-build audit never writes or reads under the checkout, which the suite's read guard would flag as undeclared."""
    monkeypatch.setattr(A, "SEAL_BASE", tmp_path_factory.mktemp("sealed"))


@pytest.fixture
def sandboxed_node(monkeypatch):
    """A node whose builds are sandboxed, the only kind on which `audit` uses its replay cache: a test of what the cache
    reuses builds with `lake` itself and audits without building, so the node it runs on can't change what it sees."""
    monkeypatch.setattr(A, "sandboxed", lambda: True)


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
POLICY = {"roots": ["P"], "assumptions_module": "P.Assumptions", "assumptions": {"P.Hyp": {"owner": "@lean"}},
          "guarantees": {"P.t": {"owner": "@lean"}}}


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
    policy = POLICY | {"four_names": ["P", "Q"], "guarantees": {n: {"owner": "@lean"} for n in pins},
                       "four_names_except": {"Q.u": "waits for its lift"}}
    fails, _ = A.judge(policy, facts(pins=pins), FILES, update=True)
    assert [f.split(" is a guarantee")[0] for f in fails] == [
        f"four names: P.{n}" for n in ("SecurityProofs.V", "SecurityProofs.W", "SecurityProofs.X", "t")]
    assert A.judge(POLICY | {"guarantees": policy["guarantees"]}, facts(pins=pins), FILES, update=True)[0] == []


def four_named_pins(**more) -> dict:
    """Facts for the four-name rule alone: `P.SecurityProofs.G`, in the form, and ``more`` ({name: signature}), not."""
    pins = {"P.SecurityProofs.G": fact(signature="P.SecurityProofs.G : P.Guarantees.G", hyps=[], reads=[])
            | {"lift": {"head": "P.Guarantees.G", "module": "P.Guarantees"}}}
    return {"pins": pins | {n: fact(signature=s, hyps=[], reads=[]) for n, s in more.items()}}


def test_four_names_lists_the_root_of_every_guarantee():
    """A lock that lists `four_names` lists every namespace root its guarantees are under, so a new area's guarantees are
    held to the four-name form from its first one; the failure names the fix."""
    f = four_named_pins(**{"Q.u": "Q.u : True"})
    policy = {"four_names": ["P"], "guarantees": dict.fromkeys(f["pins"], {"owner": "@lean"})}
    fails = A.four_name_failures(policy, f)
    assert [(x.code, x.entry) for x in fails] == [("four-names.root", "Q")]
    assert "add Q to it, and state each guarantee under it as `X.SecurityProofs.G : X.Guarantees.G`" in fails[0]
    assert A.four_name_failures(policy | {"four_names": ["P", "Q"], "shape": {"frozen": {"Q.u": "frozen: open"}}}, f) == []
    assert A.four_name_failures({"guarantees": policy["guarantees"]}, f) == [], "a lock without `four_names` holds none"


def test_a_four_names_exception_goes_once_it_holds_nothing():
    """`four_names_except` holds a guarantee that isn't in the four-name form yet, and only shrinks: an entry for one in the
    form, one frozen or a witness, and one that is no guarantee, each fails until it goes."""
    f = four_named_pins(**{"P.t": "P.t : True", "P.w": "P.w : True"})
    policy = {"four_names": ["P"], "guarantees": dict.fromkeys(f["pins"], {"owner": "@lean"}),
              "four_names_except": {"P.t": "waits for its lift"}, "shape": {"frozen": {"P.w": "frozen: open"}}}
    assert A.four_name_failures(policy, f) == []
    stale = A.four_name_failures(policy | {"four_names_except": {"P.t": "waits", "P.SecurityProofs.G": "waits",
                                                                 "P.w": "waits", "P.gone": "waits"}}, f)
    assert [(x.code, x.entry) for x in stale] == [("four-names.stale", n) for n in ("P.SecurityProofs.G", "P.w", "P.gone")]
    assert [str(x).split(", which is ")[1] for x in stale] == ["in the four-name form: remove the entry",
                                                              "a witness, frozen or a generic claim: remove the entry",
                                                              "no guarantee of this lock: remove the entry"]
    assert [x.entry for x in A.four_name_failures(policy | {"four_names_except": {}}, f)] == ["P.t"]


def test_four_names_leaves_a_generic_claims_guarantee_to_the_claims_rule():
    """A guarantee `claims.generic` lists is stated as its claim (`theorem g : C`, `generic_failures`), which the four-name
    form can't be, so the four-name rule holds it to nothing, and an exception for it is stale."""
    f = four_named_pins(**{"P.g": "P.g : P.C"})
    policy = {"four_names": ["P"], "guarantees": dict.fromkeys(f["pins"], {"owner": "@lean"}),
              "claims": {"generic": {"P.C": "P.g"}}}
    assert A.four_name_failures(policy, f) == []
    assert [x.entry for x in A.four_name_failures(policy | {"claims": {}}, f)] == ["P.g"]
    stale = A.four_name_failures(policy | {"four_names_except": {"P.g": "waits for its lift"}}, f)
    assert [(x.code, x.entry) for x in stale] == [("four-names.stale", "P.g")]


def test_each_four_names_exception_gives_its_reason():
    f = four_named_pins(**{"P.t": "P.t : True"})
    policy = {"four_names": ["P"], "guarantees": dict.fromkeys(f["pins"], {"owner": "@lean"})}
    for bad in ({"P.t": ""}, {"P.t": None}, ["P.t"]):
        assert [x.code for x in A.four_name_failures(policy | {"four_names_except": bad}, f)] == ["four-names.except-form"]


def test_a_recorded_package_passes_and_its_record_is_the_statement():
    pol = recorded()
    assert pol["guarantees"]["P.t"] == {"owner": "@lean", "signature": ["P.t (h : P.Hyp) (n : Nat) : P.f n = n"],
                                        "assumptions": ["P.Hyp"],
                                        "premises": {"assumptions": ["P.Hyp"], "domain": [], "open": [], "reads": []},
                                        "type_hash": sha("00000000000000aa")}
    assert len(pol["reads"]["P.Defs"]["digest"]) == 64 and len(pol["reads"]["P.Defs"]["definitions"]["P.f"]) == 64
    assert list(pol["reads"]["P.Defs"]["definitions"]) == ["P.f"] and set(pol["reads"]["P.Defs"]) == {"digest", "definitions"}
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



def test_a_lock_listing_its_readers_is_a_record_difference_an_update_drops():
    """A lock that still lists each module's readers (`guarantees`, or `pins` from before the rename), which the
    type hashes and the definitions' hashes determine: it fails as `guarantees.readers`, a record difference, and the record
    regenerated (or updated) is today's, with nothing in the review."""
    pol = recorded()
    for key in A.READER_KEYS:
        listed = pol | {"reads": {m: g | {key: ["P.t"]} for m, g in pol["reads"].items()}}
        assert A.with_readers(listed) == sorted(pol["reads"])
        fails = A.judge(listed, facts(), FILES)[0]
        assert fails == [f"guarantees: `reads` lists the guarantees that read {', '.join(sorted(pol['reads']))}, which the "
                         "type hashes and the definitions' hashes determine and the record no longer holds; run `check.py "
                         "--update`, or `python tools/verity/lean/merge.py --normalize` drops them"]
        assert [f.code for f in fails] == ["guarantees.readers"] and A.RECORD_DIFFERENCE.match(fails[0])
        assert A.regenerated(listed, facts(), FILES, fails) == pol == A.judge(listed, facts(), FILES, update=True)[1]
        assert A.review(listed, pol, facts(), {}) == []


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


def laneless(lock: dict) -> dict:
    """``lock`` as a tree with no lanes holds it: no entry's owner, issue or due date, and no section of their defaults."""
    return {k: {n: {f: v for f, v in r.items() if f not in A.LANE_FIELDS} if isinstance(r, dict) else r
                for n, r in sec.items()} if k in A.LANE_ENTRIES and isinstance(sec, dict) else sec
            for k, sec in lock.items() if k not in A.LANE_SECTIONS}


def unowned_and_its_facts() -> tuple[dict, dict]:
    """An unowned lock that passes, listing a guarantee and the open premise it takes, and its package's facts."""
    now = sounded(assumption_reads={"P.Sound": [{"name": "P.Sound", "kind": "def", "module": "P.Defs", "sha256": "5" * 64}]})
    seed = {"P.Defs": {"issue": "LEAN-1", "due": "2026-11-30"}}
    owned = A.judge({**POLICY, "assumptions": {}, "open_seed": seed}, now, FILES, update=True)[1]
    assert A.judge(owned, now, FILES)[0] == [] and set(owned["open"]["P.Sound"]) == {"owner", "digest", "issue", "due"}
    return laneless(owned), now


def test_a_lock_with_no_lane_field_is_unowned_and_owes_no_owner_and_no_ticket():
    """A lock is unowned by its content: it lists a guarantee or a premise, and holds no owner, issue, due date or section
    of their defaults.  No entry of it owes an owner, and an `open` entry is its digest alone; a premise it doesn't list
    still fails.  One lane field makes it an owned lock, held to every rule, so a lock that lost only some of them fails."""
    lock, now = unowned_and_its_facts()
    assert A.unowned_lock(lock) and set(lock["open"]["P.Sound"]) == {"digest"} and lock["guarantees"]["P.t"]
    assert A.judge(lock, now, FILES)[0] == []
    assert [f.code for f in A.judge({**lock, "open": {}}, now, FILES)[0]] == ["open.unlisted"]
    for section, entry, field, value in (("guarantees", "P.t", "owner", "@lean"), ("open", "P.Sound", "owner", "@lean"),
                                         ("open", "P.Sound", "due", "2026-11-30"), ("open", "P.Sound", "issue", "LEAN-1")):
        half = {**lock, section: {entry: lock[section][entry] | {field: value}}}
        assert not A.unowned_lock(half) and {f.code for f in A.judge(half, now, FILES)[0]} & {
            "guarantees.no-owner", "open.no-owner", "open.unlisted", "open.form"}, (section, field)
    for section, value in (("default_owner", "@lean"), ("open_default_issue", "LEAN-1"), ("open_seed", {})):
        assert not A.unowned_lock({**lock, section: value}), section
        assert "guarantees.no-owner" in {f.code for f in A.judge({**lock, section: value}, now, FILES)[0]}, section
    for nothing_to_own in ({}, {"roots": ["P"]}, {"roots": ["P"], "guarantees": {}, "open": {}},
                           {"roots": ["P"], "assumptions": ["P.Assumptions"]}):
        assert not A.unowned_lock(nothing_to_own)
    assert A.unowned_lock({"guarantees": {"P.t": None}}), "a listed guarantee with no record yet is an entry"


def test_an_update_and_a_bookkeeping_repair_write_no_owner_and_no_ticket_into_an_unowned_lock():
    """`--update` writes an unowned lock's records and no owner; and the repair `--bookkeeping warn` makes lists no
    premise on a ticket in it, so the lock that comes out is unowned too."""
    lock, now = unowned_and_its_facts()
    assert A.judge(lock, now, FILES, update=True) == ([], lock)
    bare = {**lock, "guarantees": {"P.t": None}}
    assert A.judge(bare, now, FILES, update=True) == ([], lock)
    unlisted = {**lock, "open": {}}
    fails = A.judge(unlisted, now, FILES)[0]
    fixed = A.repaired(unlisted, now, FILES, fails, TICKET)
    assert A.unowned_lock(fixed) and not fixed.get("open")
    assert [f.code for f in A.judge(fixed, now, FILES)[0]] == ["open.unlisted"]


def test_an_update_given_an_owner_writes_every_entrys_owner_into_an_unowned_lock():
    """`--owner` is an explicit act: a run given one reads an unowned lock as owned, so `--update --owner` owns each
    guarantee and each premise in it, as it does in a new package's lock."""
    owned, now = recorded(), facts()
    lock = laneless(owned)
    assert A.unowned_lock(lock) and set(lock["assumptions"]) == {"P.Hyp"} and A.judge(lock, now, FILES)[0] == []
    fails, written = A.judge(lock, now, FILES, update=True, owner="@proofs")
    assert fails == [] and not A.unowned_lock(written)
    assert written["guarantees"]["P.t"]["owner"] == written["assumptions"]["P.Hyp"]["owner"] == "@proofs"
    assert laneless(written) == lock, "nothing but the owners is written"
    fresh = {**lock, "guarantees": {"P.t": {}}, "assumptions": {"P.Hyp": {}}}
    assert A.judge(fresh, now, FILES, update=True, owner="@proofs") == ([], written), "a new package's first update"


def test_an_update_given_an_owner_holds_an_unowned_lock_to_the_ticket_rule():
    """In the run that owns it every rule applies, the open premises' ticket rule too: an `open` entry that is its digest
    alone fails there as it does on an owned lock, so the update can't pass and write a lock the next check fails."""
    lock, now = unowned_and_its_facts()
    assert A.judge(lock, now, FILES)[0] == []
    owned = {**lock, "guarantees": {"P.t": lock["guarantees"]["P.t"] | {"owner": "@proofs"}}}
    expect = [f.code for f in A.judge(owned, now, FILES, update=True)[0]]
    assert expect == ["open.unlisted"]
    assert [f.code for f in A.judge(lock, now, FILES, update=True, owner="@proofs")[0]] == expect


def test_the_plain_audit_after_an_update_given_an_owner_reads_the_lock_as_owned():
    """The lock `--update --owner` wrote names its owners, so it is owned by its content: the next audit, given no owner,
    passes it and holds it to every owner rule."""
    lock, now = laneless(recorded()), facts()
    written = A.judge(lock, now, FILES, update=True, owner="@proofs")[1]
    assert A.judge(written, now, FILES) == ([], written) and not A.unowned_lock(written)
    for section, entry in (("guarantees", "P.t"), ("assumptions", "P.Hyp")):
        half = {**written, section: {entry: {k: v for k, v in written[section][entry].items() if k != "owner"}}}
        assert [f.code for f in A.judge(half, now, FILES)[0]] == [f"{section}.no-owner"], section


def test_the_lean_check_names_an_unowned_lock_and_repairs_no_bookkeeping_in_it(tmp_path, monkeypatch, capsys):
    """`check.py` itself (`main` and `audit`, the package's Lean stubbed out): an unowned lock's line says `unowned` and
    its report holds `unowned`, an owned lock's neither; and under `--bookkeeping warn` an unowned lock that doesn't list
    a premise fails on it, with no repair."""
    audit = A.audit
    monkeypatch.setattr(A, "audit", lambda *a, seals=None, **k: audit(*a, **k))   # a seal copies a real build
    lock, now = unowned_and_its_facts()

    def run(name, lock, *flags):
        (sub := tmp_path / name).mkdir()
        pkg, _ = bookkeeping_audit(sub, monkeypatch, lock | {"shape": {"frozen": {"P.t": "frozen: open"}}}, now)
        code = A.main([str(pkg), "--regenerate", *flags, "--no-runs", "--out", str(sub / "out"), "--repo", str(tmp_path)])
        (rep,) = json.loads((sub / "out" / "audit.json").read_text())["packages"]
        (head,) = (line for line in capsys.readouterr().out.splitlines() if line.startswith(f"AUDIT {pkg}: "))
        return code, rep, head
    code, rep, head = run("passes", lock)
    assert code == 0 and rep["unowned"] is True and head.startswith(f"AUDIT {tmp_path}/passes/pkg: PASS  ")
    assert head.endswith("; unowned")
    code, rep, head = run("unlisted", {**lock, "open": {}}, "--bookkeeping", "warn")
    assert code == 1 and rep["failure_codes"] == ["open.unlisted"] and rep["unowned"] is True and head.endswith("; unowned")
    assert not {"repairs", "bookkeeping", "regenerated", "bookkeeping_left"} & set(rep)
    owned = recorded()
    assert laneless(owned) != owned
    now = facts()
    code, rep, head = run("owned", owned)
    assert code == 0 and "unowned" not in rep and "unowned" not in head


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
    def reads(m):
        return {"pins": {"S.t": {"kind": "theorem", "outside": [], "reads": [{"name": f"{m}.x", "module": m}]}}}
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


def test_a_layer_admits_a_toolchain_module_its_imports_already_reach():
    """A proof's `import all Init.Data.Array.Basic` brings nothing in: `Init` reaches it. A toolchain module nothing admitted
    reaches is still outside the layer, and so is a module this repository builds, however it is reached."""
    lib, tc = "/r/Q/.lake/build/lib/lean/", "/elan/toolchains/v4/lib/lean/"
    graph = {"P.Defs": {"imports": ["Init", "Q.Draw"], "olean": "/r/P/.lake/build/lib/lean/P/Defs.olean"},
             "Q.Draw": {"imports": ["Init", "Q.Json"], "olean": lib + "Q/Draw.olean"},
             "Q.Json": {"imports": ["Init"], "olean": lib + "Q/Json.olean"},
             "Init": {"imports": ["Init.Data.Array.Basic"], "olean": tc + "Init.olean"},
             "Init.Data.Array.Basic": {"imports": [], "olean": tc + "Init/Data/Array/Basic.olean"},
             "Lean.Data.Json.Basic": {"imports": ["Init"], "olean": tc + "Lean/Data/Json/Basic.olean"}}
    pol = recorded() | {"layers": {"P.Defs": {"imports": ["Q.Draw"]}}}
    def fails(*imports):
        f = facts(graph=graph)
        return A.judge(pol, f | {"imports": f["imports"] | {"P.Defs": ["Init", "Q.Draw", *imports]}}, FILES)[0]
    assert fails("Init.Data.Array.Basic") == []
    assert fails("Lean.Data.Json.Basic") == [
        "layers: P.Defs imports Lean.Data.Json.Basic, outside P.Defs's layer (Q.Draw)"]
    assert fails("Q.Json") == ["layers: P.Defs imports Q.Json, outside P.Defs's layer (Q.Draw)"]


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
    """The three trees: a guarantee's or property's statement reads only the definitions tree and the code, and
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


def test_the_record_follows_a_statement_through_the_modules_the_trees_except(tmp_path, monkeypatch):
    """`trees`' `except` exempts a module from the trees rule, not from the record: `Facts.lean` walks a guarantee's
    statement through `record_scope` (`meaning`, then the excepted prefixes), for a package's own guarantees and for those
    a `proved_in` package proves, as `security`'s are. The witnesses rule reads what the record holds: a claim, or a
    verifier's acceptance, read only through an excepted module holds the guarantee to witnesses as one under `meaning`
    does."""
    why = "C-Flock is excepted until its spec is extracted"
    pol = {"roots": ["S"], "meaning": ["S.Defs"], "trees": {"proofs": ["SP"], "except": {"SP.Loop": why, "S.Defs": why}}}
    assert A.record_scope(pol) == ["S.Defs", "SP.Loop"]
    assert A.record_scope({"roots": ["S"]}) == ["S"]
    assert A.record_scope({"roots": ["S"], "trees": {"except": ["SP.Loop"]}}) == ["S"], "a malformed `except` is the trees check's"

    configs = []

    def run(argv, cwd, log=None, timeout=0):
        cfg = json.loads(Path(argv[-1]).read_text())
        configs.append(cfg)
        Path(cfg["out"]).write_text(json.dumps({"pins": {}}))
        Path(cfg["canon"]).write_text("")
        return 0, ""
    monkeypatch.setattr(A, "facts_program", lambda pkg: "facts")
    monkeypatch.setattr(A, "run", run)
    A.facts_of(tmp_path, pol, tmp_path)
    assert configs[0]["meaning"] == ["S.Defs"] and configs[0]["recorded"] == ["S.Defs", "SP.Loop"]

    s, p = tmp_path / "s", tmp_path / "p"
    for d, rel in ((s, "S/Defs.lean"), (p, "SP/Loop.lean")):
        (d / rel).parent.mkdir(parents=True)
        (d / rel).write_text("")
    (s / A.POLICY).write_text(json.dumps(pol | {"proved_in": ["../p"]}))
    (p / A.POLICY).write_text(json.dumps({"roots": ["SP"]}))
    (p / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "path", "name": "s", "dir": "../s"}]}))
    scopes = []

    class Read(Exception):
        pass

    def facts_of(q, qpol, work, moved=None, keep=True, sealed=None):
        scopes.append(A.record_scope(qpol))
        raise Read
    monkeypatch.setattr(A, "facts_of", facts_of)
    with pytest.raises(Read):
        A.proved(s, pol | {"proved_in": ["../p"]}, facts(modules=[], pins={}), tmp_path / "work")
    assert scopes == [["S.Defs", "SP.Loop"]], "the prover's facts are walked through the spec's excepted modules"

    claim = {"name": "S.Claims.Holds", "module": "S.Defs.Claims", "kind": "def", "hash": "4" * 32}
    loop = {"name": "S.Guarantees.Loop", "module": "SP.Loop", "kind": "def", "hash": "5" * 32}
    verifier = {"name": "SP.Loop.verifyRows", "module": "SP.Loop", "kind": "def", "hash": "6" * 32}
    assert A.witness_scope("S.t", pol, fact(reads=[loop, claim])) == ("acceptance", "it reads the claim S.Claims.Holds"), (
        "a claim read only through the excepted statement")
    assert A.witness_scope("S.t", pol, fact(reads=[loop, verifier])) == (
        "acceptance", "it reads SP.Loop.verifyRows, a verifier's acceptance"), "a verifier defined in the excepted module"
    assert A.witness_scope("S.t", pol, fact(reads=[loop])) is None


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


def cited(policy: dict | None = None) -> tuple[dict, dict]:
    """(facts, the policy `--update` writes) of a package whose `P.t` takes `P.Guarantees.U`, the statement of `P.u`,
    another listed guarantee."""
    u = {"name": "P.Guarantees.U", "module": "P.Guarantees", "kind": "def", "hash": "7" * 32}
    t = fact(signature="P.t (h : P.Hyp) (u : P.Guarantees.U) (n : Nat) : P.f n = n",
             hyps=fact()["hypotheses"] + [{"binder": "u", "type": "P.Guarantees.U", "closed": True,
                                           "named": "P.Guarantees.U", "module": "P.Guarantees"}],
             reads=fact()["reads"] + [u])
    t["premises"] = [{"binder": "u", "head": "P.Guarantees.U", "module": "P.Guarantees"}]
    lifted = fact(signature="P.u : P.Guarantees.U", hyps=[], reads=[u]) | {
        "lift": {"head": "P.Guarantees.U", "module": "P.Guarantees"}}
    now = facts(pins={"P.t": t, "P.u": lifted},
                graph={"P.Guarantees": {"imports": ["Init"], "olean": "/r/P/.lake/build/lib/lean/P/Guarantees.olean"}})
    fails, pol = A.judge(policy or POLICY | {"guarantees": {"P.t": {"owner": "@lean"}, "P.u": {"owner": "@lean"}}},
                         now, FILES, update=True)
    assert fails == [], fails
    return now, pol


def test_every_record_difference_code_is_one_update_settles():
    """Each code of `RECORD_DIFFERENCE` comes from a record that differs from the build, and `regenerated` writes the record
    that then passes: what drive commits is the code set, so a code in it that no case provokes fails here."""
    pol = recorded()
    group = pol["reads"]["P.Defs"]
    entry = pol["guarantees"]["P.t"]
    premised = fact(signature="P.t (n : Nat) (h : P.Premises.Hyp n) : P.f n = n",
                    hyps=[{"binder": "h", "type": "P.Premises.Hyp n", "closed": False, "named": None, "module": None}])
    premised["premises"] = [{"binder": "h", "head": "P.Hyp", "module": "P.Assumptions"}]
    hyp = [{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "1" * 32, "own": ["P.f.match_1"]},
           {"name": "P.f.match_1", "module": "P.Defs", "kind": "def", "hash": "2" * 32},
           {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "8" * 32}]
    sound = {"name": "P.Sound", "module": "P.Defs", "kind": "def", "hash": "5" * 32}

    def opened(digest="5"):
        pin = fact(signature="P.t (h : P.Hyp) (s : P.Sound) (n : Nat) : P.f n = n",
                   hyps=fact()["hypotheses"] + [{"binder": "s", "type": "P.Sound", "closed": True, "named": "P.Sound",
                                                 "module": "P.Defs"}],
                   reads=fact()["reads"] + [sound | {"hash": digest * 32}])
        pin["premises"] = [{"binder": "s", "head": "P.Sound", "module": "P.Defs"}]
        return facts(pins={"P.t": pin}, graph={"P.Defs": {"imports": ["Init"], "olean": "/r/P/.lake/build/lib/lean/P/Defs.olean"}})
    opol = A.judge(POLICY | {"open_seed": {"P.Defs": {"issue": "LEAN-1", "due": "2026-11-30"}}}, opened(), FILES,
                   update=True)[1]
    ticket = opol["open"]["P.Sound"]
    bare = facts(pins={"P.t": fact(reads=fact()["reads"] + [sound])})
    scope = {k: v for k, v in ticket.items() if k in ("owner", "digest")} | {"why": "the circuits the bound is over"}
    dpol = {k: v for k, v in opol.items() if k != "open"} | {"domain": {"P.Sound": scope}}
    cfacts, cpol = cited()
    cites = cpol["domain"]["P.Guarantees.U"]

    def lifted(body):
        return facts(pins={"P.t": fact() | {"lift": {"head": "P.T", "module": "P.Defs", "unfolded_sha256": sha(body)}}})
    cases = {
        "guarantees.unrecorded": (POLICY, facts()),
        "assumptions.unrecorded": (pol | {"assumptions": {"P.Hyp": {"owner": "@lean"}}}, facts()),
        "assumptions.unlisted-premise": ({**A.judge(POLICY, facts(pins={"P.t": premised}), FILES, update=True)[1],
                                          "assumptions": {}}, facts(pins={"P.t": premised})),
        "guarantees.signature": (pol, facts(pins={"P.t": fact(signature="P.t (h : P.Hyp) (n : Nat) : P.f n ≤ n")})),
        "guarantees.assumptions": (pol | {"guarantees": {"P.t": entry | {"assumptions": []}}}, facts()),
        "guarantees.type-hash": (pol | {"guarantees": {"P.t": entry | {"type_hash": sha("00000000000000bb")}}}, facts()),
        "guarantees.premises-changed": (pol | {"guarantees": {"P.t": entry | {"premises": ["P.Hyp"]}}}, facts()),
        "guarantees.reads-new": (pol | {"reads": {m: g for m, g in pol["reads"].items() if m != "P.Defs"}}, facts()),
        "guarantees.reads-gone": (pol | {"reads": pol["reads"] | {"P.Old": group}}, facts()),
        "guarantees.reads-changed": (pol, facts(pins={"P.t": fact(reads=[{**hyp[0], "hash": "9" * 32}, *hyp[1:2],
                                                                          fact()["reads"][2]])})),
        "guarantees.readers": (pol | {"reads": pol["reads"] | {"P.Defs": group | {"guarantees": ["P.t", "P.u"]}}}, facts()),
        "guarantees.reads-digest": (pol | {"reads": pol["reads"] | {"P.Defs": group | {"digest": "0" * 64}}}, facts()),
        "assumptions.digest": (pol, facts(pins={"P.t": fact(reads=hyp)})),
        "assumptions.unread": (pol | {"assumptions": pol["assumptions"] | {"P.Other": {"owner": "@lean", "digest": "0" * 64}}},
                               facts()),
        "assumptions.read-only": (pol | {"assumptions": pol["assumptions"] | {"P.Sound": {"owner": "@lean"}}}, bare),
        "assumptions.moved": (opol | {"assumptions": opol["assumptions"] | {"P.Sound": {"owner": "@lean",
                                                                                         "digest": "0" * 64}}}, opened()),
        "open.digest": (opol, opened("6")),
        "open.unrecorded": (opol | {"open": {"P.Sound": {k: v for k, v in ticket.items() if k != "digest"}}}, opened()),
        "open.unread": (opol | {"open": opol["open"] | {"P.Other": ticket}}, opened()),
        "open.read-only": (opol, bare),
        "open.declared": (pol | {"assumptions": {}, "open": {"P.Hyp": pol["assumptions"]["P.Hyp"] | ticket}}, facts()),
        "open.cited": (cpol | {"domain": {}, "open": {"P.Guarantees.U": {"owner": "@lean", "digest": cites["digest"]}
                                                      | {k: ticket[k] for k in ("issue", "due")}}}, cfacts),
        "domain.digest": (dpol, opened("6")),
        "domain.unrecorded": (dpol | {"domain": {"P.Sound": {k: v for k, v in scope.items() if k != "digest"}}}, opened()),
        "domain.unread": (dpol | {"domain": dpol["domain"] | {"P.Other": scope}}, opened()),
        "domain.read-only": (dpol, bare),
        "domain.declared": (pol | {"assumptions": {}, "domain": {"P.Hyp": pol["assumptions"]["P.Hyp"] | {"why": "x"}}},
                            facts()),
        "domain.uncited": (cpol | {"domain": {"P.Guarantees.U": {k: v for k, v in cites.items() if k != "cites"}}}, cfacts),
        "domain.unlisted": ({k: v for k, v in cpol.items() if k != "domain"}, cfacts),
        "shape.inline-unlisted": (pol | {"shape": {"inline": {"P.gone": {"x = y": "why"}}}}, facts()),
        "guarantees.unfolded-hash": (recorded(lifted("first")), lifted("restated")),
    }
    assert set(cases) == A.RECORD_DIFFERENCE
    for code, (policy, now) in cases.items():
        fails = A.judge(policy, now, FILES)[0]
        assert code in {f.code for f in fails} and all(map(A.RECORD_DIFFERENCE.match, fails)), (code, fails)
        regen = A.regenerated(policy, now, FILES, fails)
        assert regen is not None and A.judge(regen, now, FILES)[0] == [], code


def test_what_settles_is_keyed_on_codes_not_text():
    """A record difference reworded still settles, and a string that reads like one but has no record-difference code (a
    bare string, or another check's failure) settles nothing: rewording a message changes nothing drive commits."""
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


def test_a_regenerated_record_s_statement_changes_are_listed_with_their_owners():
    """`changes`, a report's list of what a regenerated record changes in what a guarantee says, for whoever rings the
    owners: one item for each line of the review that heads a change, with its entry, the lock's section it is in, its
    owners and the line.  A guarantee's owner and a listed premise's is its entry's; a definition's are those of the
    guarantees that read it, each once; a guarantee the record drops is named from the lock before; a definition no
    guarantee reads any more has the lock's `default_owner`, else none.  The items are read off each line's `Change`, never
    its text: a reworded line names the same owners, and a bare string is no item.  No lock makes it raise."""
    g = {"name": "P.g", "module": "P.Defs", "kind": "def", "hash": "4" * 32}

    def pins(rel="=", h="00000000000000aa", f="1", more=()):
        reads = [r | {"hash": f * 32} if r["name"] == "P.f" else r for r in fact()["reads"]]
        return {"P.t": fact(signature=f"P.t (h : P.Hyp) (n : Nat) : P.f n {rel} n", h=h, reads=reads),
                "P.u": fact(signature="P.u (h : P.Hyp) (n : Nat) : P.f n = n", reads=[*reads, *more])}

    def heads(notes):
        return [n for n in A.statement_changes(notes) if n[:1] != " "]
    old = A.judge(POLICY | {"guarantees": {"P.t": {"owner": "@zed"}, "P.u": {"owner": "@amy"}}}, facts(pins=pins(more=[g])),
                  FILES, update=True)[1]
    now = facts(pins=pins("≤", "00000000000000bb", "9"))
    fails = A.judge(old, now, FILES)[0]
    assert {f.code for f in fails} == {"guarantees.signature", "guarantees.reads-changed"}
    regen = A.regenerated(old, now, FILES, fails)
    notes = A.review(old, regen, now, {})
    listed = A.changes(notes, old, regen)
    assert listed == [
        {"entry": "P.t", "kind": "guarantees", "owners": ["@zed"], "line": "guarantee P.t: changed"},
        {"entry": "P.f", "kind": "reads", "owners": ["@amy", "@zed"],
         "line": "definition P.f (P.Defs), read by P.t, P.u: changed"},
        {"entry": "P.g", "kind": "reads", "owners": [], "line": "definition P.g (P.Defs): gone"}]
    assert [c["line"] for c in listed] == heads(notes) and json.loads(json.dumps(listed)) == listed
    assert A.changes(notes, old, regen | {"default_owner": "@lean"})[2]["owners"] == ["@lean"]
    assert A.changes(notes, old, regen | {"default_owner": "@lean"})[:2] == listed[:2], "an owned entry keeps its owners"

    dropped = {**old, "guarantees": {"P.t": old["guarantees"]["P.t"]}}
    notes = A.review(old, dropped, facts(pins={"P.t": pins(more=[g])["P.t"]}), {})
    assert A.changes(notes, old, dropped)[0] == {"entry": "P.u", "kind": "guarantees", "owners": ["@amy"],
                                                 "line": "guarantee P.u: removed"}

    handed = {**old, "guarantees": old["guarantees"] | {"P.t": old["guarantees"]["P.t"] | {"owner": "@lean"}}}
    notes = A.review(old, handed, facts(pins=pins(more=[g])), {})
    assert notes == [f"guarantee P.t: owner @zed -> @lean: {A.OWNED}"] and A.changes(notes, old, handed) == []

    reworded = A.Change("guarantees", "P.t", "P.t no longer says what its record says")
    assert A.changes([reworded], old, regen) == [{"entry": "P.t", "kind": "guarantees", "owners": ["@zed"],
                                                  "line": str(reworded)}]
    assert A.changes(["guarantee P.t: changed"], old, regen) == [], "the same text without its `Change` names no entry"
    import copy
    back = copy.deepcopy(A.Change("reads", "P.f", "definition P.f: changed", ["P.t"]))
    assert (back, back.kind, back.entry, back.readers) == ("definition P.f: changed", "reads", "P.f", ("P.t",))
    odd = {"guarantees": {"P.t": {"owner": ["@zed"]}, "P.u": {"owner": "amy"}}, "assumptions": ["P.Assumptions"], "open": None,
           "default_owner": 7}
    assert [c["owners"] for c in A.changes(A.review(old, regen, now, {}), odd, odd)] == [[], [], []]


def test_a_changed_premise_is_listed_under_its_section_with_its_owner():
    """A listed premise whose record changes (here an open premise whose walk changed, `open.digest`, a record difference)
    is an item of `changes` under the lock's section that lists it, owned by its entry's owner."""
    pin = fact(signature="P.t (n : Nat) (h : P.Leaves n) : P.f n = n", hyps=[])
    pin["premises"] = [{"binder": "h", "head": "P.Leaves", "module": "P.Thms"}]
    graph = {"P.Thms": {"imports": ["Init"], "olean": "/r/Security/Proofs/.lake/build/lib/lean/P/Thms.olean"}}

    def built(bound):
        walk = [{"name": "P.Leaves", "module": "P.Thms", "kind": "def", "sha256": sha("leaves")},
                {"name": "P.bound", "module": "P.Thms", "kind": "def", "sha256": sha(bound)}]
        return facts(pins={"P.t": pin}, graph=graph, assumption_reads={"P.Leaves": walk})
    seed = {"@amy": {"issue": "LEAN-1", "due": "2026-11-30"}}
    lock = A.judge({**POLICY, "assumptions": {}, "guarantees": {"P.t": {"owner": "@amy"}}, "open_seed": seed}, built("n + 1"),
                   FILES, update=True)[1]
    assert lock["open"]["P.Leaves"]["owner"] == "@amy"
    now = built("n + 2")
    fails = A.judge(lock, now, FILES)[0]
    assert [f.code for f in fails] == ["open.digest"]
    regen = A.regenerated(lock, now, FILES, fails)
    assert A.changes(A.review(lock, regen, now, {}), lock, regen) == [
        {"entry": "P.Leaves", "kind": "open", "owners": ["@amy"],
         "line": "open premise P.Leaves: changed (its statement or a definition it reads)"}]


SCOPE = "the compiled circuits the bound is stated over"


def domain_by_hand() -> tuple[dict, dict, dict, dict, dict]:
    """(`witnessed`'s facts; its lock with `P.Compiled` open; that lock without the entry; that one with the entry written
    under `domain` by hand, P.sound's record still taking it as open; and the lock an `--update` of that writes)."""
    pol, f = witnessed()
    opened = A.judge(pol | {"open_seed": {"P.Defs": {"issue": "LEAN-1", "due": "2026-11-30"}}}, f, FILES, update=True)[1]
    unlisted = {k: v for k, v in opened.items() if k != "open"}
    entry = {k: v for k, v in opened["open"]["P.Compiled"].items() if k not in ("issue", "due")} | {"owner": "@amy"}
    hand = unlisted | {"domain": {"P.Compiled": entry | {"why": SCOPE}}}
    dom = A.judge(hand, f, FILES, update=True)[1]
    assert A.judge(dom, f, FILES)[0] == [] and dom["guarantees"]["P.sound"]["premises"]["domain"] == ["P.Compiled"]
    return f, opened, unlisted, hand, dom


def test_a_domain_hypothesis_the_lock_did_not_list_gets_a_line_with_its_why_and_its_takers():
    """A premise under `domain` that the lock before listed under none of `assumptions`, `domain` and `open` gets one line
    of the review, with its `why` and the guarantees whose records take it: a statement change, and an item of `changes`
    under `domain` with its entry's owner.  So does the one an update lists itself, another listed guarantee's statement,
    with the guarantee it cites.  One that was open keeps the line that says so, and one the lock before lists the same
    way gets none: the review compares two locks, and an entry written into the first is in both."""
    scope = SCOPE
    f, opened, unlisted, hand, dom = domain_by_hand()
    line = f"domain hypothesis P.Compiled, taken by P.sound: new: {scope}"
    assert A.assumption_changes(unlisted, dom) == [line]
    notes = A.review(unlisted, dom, f, {})
    assert line in A.statement_changes(notes)
    assert [c for c in A.changes(notes, unlisted, dom) if c["kind"] == "domain"] == [
        {"entry": "P.Compiled", "kind": "domain", "owners": ["@amy"], "line": line}]
    assert A.assumption_changes(opened, dom) == [f"domain hypothesis P.Compiled: a domain hypothesis: {scope} (was open)"]
    assert A.assumption_changes(hand, dom) == [] and A.assumption_changes(dom, dom) == []

    several = {"domain": {"P.Scope": {"owner": "@amy", "why": scope}},
               "guarantees": {"P.v": {"premises": {"domain": ["P.Scope"]}}, "P.t": {"premises": {"domain": ["P.Scope"]}},
                              "P.w": {"premises": {"domain": []}}, "P.x": {"owner": "@lean"}}}
    assert A.assumption_changes({}, several) == [f"domain hypothesis P.Scope, taken by P.t, P.v: new: {scope}"]

    before = POLICY | {"guarantees": {"P.t": {"owner": "@lean"}, "P.u": {"owner": "@lean"}}}
    assert A.assumption_changes(before, cited()[1]) == [
        "domain hypothesis P.Guarantees.U, taken by P.t: new: P.u's statement, another listed guarantee, citing P.u"]


def test_a_premise_a_record_takes_as_another_kind_is_named_on_its_guarantee_s_line_a_statement_change():
    """A record that takes a premise as another kind than it did (`premises`), at the same type hash and named assumptions,
    is no reprint: its guarantee's line names the premise, how it is taken now and how it was, a domain hypothesis with
    its `why`, and is a statement change under `guarantees` with the guarantee's owner.  It reads the records, so the
    update of a lock whose `domain` entry a person wrote by hand, which changes no listed premise, prints it too.
    Records with equal `premises`, or one that holds none, still print as the same statement."""
    f, opened, unlisted, hand, dom = domain_by_hand()
    line = f"guarantee P.sound: takes P.Compiled as a domain hypothesis (was open): {SCOPE}"
    notes = A.review(hand, dom, f, {})
    assert notes == [line] == A.statement_changes(notes) and A.assumption_changes(hand, dom) == []
    assert A.changes(notes, hand, dom) == [{"entry": "P.sound", "kind": "guarantees", "owners": ["@lean"], "line": line}]
    assert A.review(opened, dom, f, {}) == [line, f"domain hypothesis P.Compiled: a domain hypothesis: {SCOPE} (was open)"]
    assert A.review(dom, opened, f, {})[0] == "guarantee P.sound: takes P.Compiled as open (was a domain hypothesis)"

    def taking(lock: dict, **premises: list[str]) -> dict:
        rec = lock["guarantees"]["P.sound"]
        return lock | {"guarantees": lock["guarantees"] | {"P.sound": rec | {"premises": rec["premises"] | premises}}}
    assert A.review(taking(hand, open=[]), dom, f, {}) == [
        f"guarantee P.sound: takes P.Compiled as a domain hypothesis (new): {SCOPE}"]
    assert A.review(dom, taking(dom, domain=[], reads=["P.Compiled"]), f, {}) == [
        "guarantee P.sound: reads P.Compiled without taking it (was a domain hypothesis)"]
    assert A.review(dom, taking(dom, domain=[], assumptions=[]), f, {}) == [
        "guarantee P.sound: no longer takes P.Compiled (was a domain hypothesis); no longer takes P.Hyp (was a named "
        "assumption)"]
    reprinted = dom["guarantees"]["P.sound"] | {"signature": ["P.sound, printed otherwise"]}
    same = [f"guarantee P.sound: {A.REPRINTED}"]
    assert A.review(dom, dom | {"guarantees": dom["guarantees"] | {"P.sound": reprinted}}, f, {}) == same
    bare = {k: v for k, v in reprinted.items() if k != "premises"}
    assert A.review(dom | {"guarantees": dom["guarantees"] | {"P.sound": bare}}, dom, f, {}) == same


TICKET = {"issue": "LEAN-186", "due": "2026-11-14", "owner": "@lean"}


def sounded(owners=("@lean",), **over) -> dict:
    """Facts of a package whose guarantees (`P.t`, then `P.u`, ... one per owner) take `P.Sound`, a premise in `P.Defs`."""
    def pin():
        p = fact(signature="P.t (n : Nat) : P.Guarantees.t n", hyps=[])
        p["premises"] = [{"binder": "s", "head": "P.Sound", "module": "P.Defs"}]
        return p
    names = ["P.t", "P.u", "P.v"][:len(owners)]
    return facts(pins={n: pin() for n in names},
                 graph={"P.Defs": {"imports": ["Init"], "olean": "/r/P/.lake/build/lib/lean/P/Defs.olean"}}, **over)


def bookkeeping_cases() -> dict[str, tuple[dict, dict]]:
    """{code: (lock, facts)} failing with that code of `BOOKKEEPING`, each repairable by `repaired`."""
    pol, now = recorded(), sounded()
    seed = {"P.Defs": {"issue": "LEAN-1", "due": "2026-11-30"}, "Q.Assumptions": {"issue": "LEAN-1", "due": "2026-11-30"}}
    opol = A.judge({**POLICY, "assumptions": {}, "open_seed": seed}, now, FILES, update=True)[1]
    sound = opol["open"]["P.Sound"]
    wpol, wfacts = witnessed(guarantee={"premises": [COMPILED, {"binder": "hx", "head": "Q.X", "module": "Q.Assumptions"}]})
    wrec = A.judge({**wpol, "open_seed": seed}, wfacts, FILES, update=True)[1]
    covered = {**wrec, "open": {"Q.X": wrec["open"]["Q.X"]}, "domain": {"P.Compiled": {"why": "the deployed circuits"}}}
    uncovered = {**wrec, "open": {"P.Compiled": wrec["open"]["P.Compiled"]},
                 "domain": {"Q.X": {"owner": "@lean", "why": "the deployed circuits"}}}
    bare = {k: v for k, v in wrec.items() if k != "witnesses"} | {
        "unwitnessed": {"P.sound": wfacts["pins"]["P.sound"]["type_sha256"]}}
    frozen = {"frozen": {"P.t": "frozen: open"}}
    deployed = {**pol, "trust": {"base": HYP}, "deployed": ["P.t"]}

    def without(rec, key):
        return {k: v for k, v in rec.items() if k != key}
    return {
        "trust.gap-gone": ({**deployed, "trust_gap": {"P.t": ["P.Sound"]}}, facts()),
        "trust.gap-stale": ({**deployed, "trust_gap": {"P.u": ["P.Sound"]}}, facts()),
        "open.unlisted": ({**POLICY, "assumptions": {}}, now),
        "assumptions.outside-module": ({**opol, "open": {}, "assumptions": {"P.Sound": {"owner": "@lean"}}}, now),
        "domain.uncovered": (uncovered, wfacts),
        "open.seed": ({**opol, "open_seed": seed}, now),
        "witnesses.missing": ({**bare, "unwitnessed": {}}, wfacts),
        "witnesses.unwitnessed-changed": ({**bare, "unwitnessed": {"P.sound": sha("its statement before")}}, wfacts),
        "witnesses.stale": ({**wrec, "witnesses": wrec["witnesses"] | {"P.gone": wrec["witnesses"]["P.sound"]}}, wfacts),
        "witnesses.assumed-stale": ({**wrec, "witnesses_assumed": {"P.Gone": "assumed as it stands"}}, wfacts),
        "witnesses.unwitnessed-unlisted": ({**bare, "unwitnessed": bare["unwitnessed"] | {"P.gone": sha("listed once")}}, wfacts),
        "shape.properties-stale": ({**pol, "shape": frozen | {"properties": {"P.gone": {"function": "P.f",
                                                                                       "outcome": "accept"}}}}, facts()),
        "shape.frozen-stale": ({**pol, "shape": {"frozen": frozen["frozen"] | {"P.gone": "frozen: open"}}}, facts()),
        "guarantees.no-owner": ({**pol, "guarantees": {"P.t": without(pol["guarantees"]["P.t"], "owner")}}, facts()),
        "assumptions.no-owner": ({**pol, "assumptions": {"P.Hyp": without(pol["assumptions"]["P.Hyp"], "owner")}}, facts()),
        "open.no-owner": ({**opol, "open": {"P.Sound": without(sound, "owner")}}, now),
        "domain.no-owner": (covered, wfacts),
    }


def test_every_bookkeeping_code_is_repaired_with_a_warning_and_fails_without_the_flag():
    """Each code of `BOOKKEEPING` (drive's lock run after a landing repairs the bookkeeping and warns) comes
    from a lock `repaired` mends, so the policy's checks pass on the lock it writes, and each repair is one warning line naming
    the entry and its owner. Without `--bookkeeping warn` the same lock fails: no bookkeeping code is a record difference, so
    `regenerated` settles none of it. A code in the set that no case provokes fails here."""
    cases = bookkeeping_cases()
    assert set(cases) == A.BOOKKEEPING and not set(A.BOOKKEEPING) & set(A.RECORD_DIFFERENCE)
    for code, (policy, now) in cases.items():
        shaped = (lambda lock: A.shape(lock, now)) if code.startswith("shape.") else (lambda lock: [])
        fails = A.judge(policy, now, FILES)[0] + shaped(policy)
        assert code in {f.code for f in fails}, (code, fails)
        assert A.regenerated(policy, now, FILES, fails) is None, code
        fixed = A.repaired(policy, now, FILES, fails, TICKET)
        assert A.judge(fixed, now, FILES)[0] + shaped(fixed) == [], code
        warned = A.bookkeeping_warnings(fails, fixed, TICKET)
        for f in (f for f in fails if f.code == code):
            assert any(w.startswith(f"bookkeeping: {f.entry or 'the lock'} (@lean): ") and w.endswith(f"({code})")
                       for w in warned), (code, warned)
        assert len(warned) == sum(map(A.BOOKKEEPING.match, fails)), (code, warned)
    def fix(code):
        policy, now = cases[code]
        return A.repaired(policy, now, FILES, A.judge(policy, now, FILES)[0], TICKET), now
    fixed, _ = fix("open.unlisted")
    assert {k: v for k, v in fixed["open"]["P.Sound"].items() if k != "digest"} == {
        "owner": "@lean", "issue": "LEAN-186", "due": "2026-11-14"}
    for code in ("witnesses.missing", "witnesses.unwitnessed-changed"):
        fixed, now = fix(code)
        assert fixed["unwitnessed"]["P.sound"] == now["pins"]["P.sound"]["type_sha256"], "listed at its hashes now"
    fixed, now = fix("witnesses.unwitnessed-unlisted")
    assert fixed["unwitnessed"] == {"P.sound": now["pins"]["P.sound"]["type_sha256"]}, "only the entry of no guarantee goes"
    assert "open_seed" not in fix("open.seed")[0]


def test_each_bookkeeping_repair_is_listed_with_its_entry_its_code_and_its_owners():
    """`bookkeeping_repairs`, a report's `repairs`, for whoever rings the owners: one item for each repair, in the failures'
    order, with the failure's entry (None: the lock) and code, the line `bookkeeping_warnings` gives, and for owners the
    handle that line names, or none where it says `no owner` (a lock whose guarantees name none and that gives no
    `default_owner`).  For every code of `BOOKKEEPING`."""
    for code, (policy, now) in bookkeeping_cases().items():
        shaped = (lambda lock: A.shape(lock, now)) if code.startswith("shape.") else (lambda lock: [])
        fails = A.judge(policy, now, FILES)[0] + shaped(policy)
        fixed = A.repaired(policy, now, FILES, fails, TICKET)
        listed = A.bookkeeping_repairs(fails, fixed, TICKET, policy)
        assert [r["line"] for r in listed] == A.bookkeeping_warnings(fails, fixed, TICKET, policy), code
        assert [(r["entry"], r["code"]) for r in listed] == [(f.entry, f.code) for f in fails if A.BOOKKEEPING.match(f)], code
        assert code in {r["code"] for r in listed} and all(r["owners"] == ["@lean"] for r in listed), (code, listed)
        assert all(set(r) == {"entry", "code", "owners", "line"} for r in listed)
        assert json.loads(json.dumps(listed)) == listed, code
    policy, now = bookkeeping_cases()["open.seed"]
    (seed,) = A.bookkeeping_repairs(A.judge(policy, now, FILES)[0], policy, TICKET)
    assert seed == {"entry": None, "code": "open.seed", "owners": ["@lean"],
                    "line": "bookkeeping: the lock (@lean): `open_seed` consumed (open.seed)"}

    now = sounded(owners=("@zed", "@amy"))
    lock = {**POLICY, "assumptions": {}, "guarantees": {"P.t": {"owner": "@zed"}, "P.u": {"owner": "@amy"}}}
    ticket = A.bookkeeping_ticket(lock, "2026-11-14")
    fails = A.judge(lock, now, FILES)[0]
    (sound,) = (r for r in A.bookkeeping_repairs(fails, A.repaired(lock, now, FILES, fails, ticket), ticket)
                if r["entry"] == "P.Sound")
    assert (sound["code"], sound["owners"]) == ("open.unlisted", ["@amy"]) and " (@amy): " in sound["line"]

    ownerless = {**POLICY, "assumptions": {}, "guarantees": {"P.t": {}}, "open_default_issue": "LEAN-200"}
    fails = A.judge(ownerless, sounded(), FILES)[0]
    listed = A.bookkeeping_repairs(fails, A.repaired(ownerless, sounded(), FILES, fails, ticket), ticket)
    assert {(r["entry"], r["code"]) for r in listed} == {("P.t", "guarantees.no-owner"), ("P.Sound", "open.unlisted")}
    assert all(r["owners"] == [] and " (no owner): " in r["line"] for r in listed), listed


def test_bookkeeping_files_a_premise_on_the_locks_default_issue_and_its_guarantees_first_owner():
    """An unlisted premise is filed on the lock's `open_default_issue`, due on the run's date; its owner is its guarantees'
    (the first, of several), and an entry whose guarantees name no owner gets the lock's `default_owner`. Neither lives in
    the code: a lock that gives no `open_default_issue` gets the premise listed without one, a lock that passes its checks
    (so `main` stays green), and the warning asks the next PR for it."""
    now = sounded(owners=("@zed", "@amy"))
    guarantees = {"P.t": {"owner": "@zed"}, "P.u": {"owner": "@amy"}}
    lock = {**POLICY, "assumptions": {}, "guarantees": guarantees, "open_default_issue": "LEAN-200", "default_owner": "@lean"}
    ticket = A.bookkeeping_ticket(lock, "2026-11-14")
    assert ticket == {"issue": "LEAN-200", "due": "2026-11-14", "owner": "@lean"}
    assert A.bookkeeping_ticket(POLICY, "2026-11-14") == {"due": "2026-11-14"}
    fails = A.judge(lock, now, FILES)[0]
    fixed = A.repaired(lock, now, FILES, fails, ticket)
    assert {k: v for k, v in fixed["open"]["P.Sound"].items() if k != "digest"} == {
        "owner": "@amy", "issue": "LEAN-200", "due": "2026-11-14"}
    assert A.judge(fixed, now, FILES)[0] == []
    assert "bookkeeping: P.Sound (@amy): listed under `open` on LEAN-200, due 2026-11-14 (open.unlisted)" in \
        A.bookkeeping_warnings(fails, fixed, ticket)
    ownerless = {**POLICY, "assumptions": {}, "guarantees": {"P.t": {}}, "open_default_issue": "LEAN-200",
                 "default_owner": "@lean"}
    fails = A.judge(ownerless, sounded(), FILES)[0]
    fixed = A.repaired(ownerless, sounded(), FILES, fails, ticket)
    assert fixed["guarantees"]["P.t"]["owner"] == fixed["open"]["P.Sound"]["owner"] == "@lean"
    assert A.judge(fixed, sounded(), FILES)[0] == []
    for bad in ("", 7, ["LEAN-1"]):
        assert "open.default-issue-form" in {f.code for f in A.judge(lock | {"open_default_issue": bad}, now, FILES)[0]}
    for bad in ("", "lean", 7):
        assert "open.default-owner-form" in {f.code for f in A.judge(lock | {"default_owner": bad}, now, FILES)[0]}

    bare = {**POLICY, "assumptions": {}, "guarantees": guarantees}
    ticket = A.bookkeeping_ticket(bare, "2026-11-14")
    fails = A.judge(bare, now, FILES)[0]
    fixed = A.repaired(bare, now, FILES, fails, ticket)
    assert {k: v for k, v in fixed["open"]["P.Sound"].items() if k != "digest"} == {"owner": "@amy", "due": "2026-11-14"}
    assert A.judge(fixed, now, FILES)[0] == []
    assert ("bookkeeping: P.Sound (@amy): listed under `open` without an issue, due 2026-11-14, since the lock gives no "
            "`open_default_issue`: the next PR gives it one (open.unlisted)") in A.bookkeeping_warnings(fails, fixed, ticket)
    assert A.judge({**fixed, "open": {"P.Sound": fixed["open"]["P.Sound"] | {"issue": "lean-1"}}}, now, FILES)[0][0].code \
        == "open.form", "an issue, when given, is a Linear key"


#: a trust base of `facts()`'s one named assumption
HYP = {"P.Hyp": "the trust assumption it states"}


def opened(now: dict) -> dict:
    """The lock of `sounded`'s package, its guarantees recorded and `P.Sound` listed under `open`."""
    seed = {"P.Defs": {"issue": "LEAN-1", "due": "2026-11-30"}}
    return A.judge({**POLICY, "assumptions": {}, "open_seed": seed}, now, FILES, update=True)[1]


def test_a_deployed_guarantee_s_gap_is_what_it_rests_on_outside_the_trust_base_through_the_guarantees_it_cites():
    """A deployed guarantee rests on the named assumptions and open premises its record lists, and through a domain
    hypothesis that is another listed guarantee's statement (`cites`) on that guarantee's, transitively (a cycle ends).  A
    domain hypothesis that cites none is part of what it says, and a premise it only reads is no premise it takes.  A
    premise of its own trust (`trust.extra`, with its reason) leaves its gap and no other guarantee's."""
    lock = {"trust": {"base": {"T": "the base"}}, "deployed": ["G", "K"],
            "domain": {"D.Scope": {"why": "its scope"}, "D.H": {"why": "H's statement", "cites": "H"},
                       "D.G": {"why": "G's statement", "cites": "G"}},
            "guarantees": {"G": {"premises": {"assumptions": ["T", "A"], "domain": ["D.Scope", "D.H"], "open": ["O"],
                                              "reads": ["R"]}},
                           "H": {"premises": {"assumptions": [], "domain": ["D.G"], "open": ["O2", "T"], "reads": []}},
                           "K": {"premises": {"assumptions": ["T"], "domain": [], "open": [], "reads": ["R"]}}}}
    assert A.trust_gaps(lock) == {"G": ["A", "O", "O2"], "K": []}
    assert A.trust_failures(lock | {"trust_gap": {"G": ["A", "O", "O2"]}}) == []
    own = lock | {"trust": {"base": {"T": "the base"}, "extra": {"G": {"O2": "the layer below's rule"}}},
                  "guarantees": lock["guarantees"] | {"K": {"premises": {"assumptions": ["O2"]}}}}
    assert A.trust_gaps(own) == {"G": ["A", "O"], "K": ["O2"]}
    why = {"P.Sound": "its rule"}
    assert A.renamed_witnesses({"trust": {"base": HYP, "extra": {"P.t": why}}, "deployed": ["P.t"],
                                "trust_gap": {"P.t": ["R.x", "P.Sound"]}}, {"P": "Q"}) == {
        "trust": {"base": {"Q.Hyp": HYP["P.Hyp"]}, "extra": {"Q.t": {"Q.Sound": "its rule"}}}, "deployed": ["Q.t"],
        "trust_gap": {"Q.t": ["Q.Sound", "R.x"]}}


def test_a_deployed_guarantee_resting_on_a_premise_outside_the_trust_base_fails_until_its_trust_gap_entry_lists_it():
    """`deployed` lists `P.t`, which takes `P.Sound`, an open premise `trust` doesn't list: that is its gap.  The lock fails
    until `trust_gap` lists it, and neither an update nor the lock run's bookkeeping lists it, so a premise enters the gap
    only by a person's hand; listed, the lock passes, and so does one whose trust holds the premise, in the base or as
    `P.t`'s own with its reason.  A guarantee's own trust is a deployed guarantee's, and each holds a premise and a
    reason."""
    now = sounded()
    lock = {**opened(now), "trust": {"base": HYP}, "deployed": ["P.t"]}
    assert A.judge({k: v for k, v in lock.items() if k != "deployed"}, now, FILES)[0] == []
    fails = A.judge(lock, now, FILES)[0]
    assert [(f.code, f.entry) for f in fails] == [("trust.gap-new", "P.t")] and "rests on P.Sound, outside" in fails[0]
    assert A.regenerated(lock, now, FILES, fails) is None and not A.BOOKKEEPING.match(fails[0])
    up_fails, up = A.judge(lock, now, FILES, update=True)
    assert [f.code for f in up_fails] == ["trust.gap-new"] and "trust_gap" not in up
    fixed = A.repaired(lock, now, FILES, fails, TICKET)
    assert [f.code for f in A.judge(fixed, now, FILES)[0]] == ["trust.gap-new"]
    assert A.judge(lock | {"trust_gap": {"P.t": ["P.Sound"]}}, now, FILES)[0] == []
    assert A.judge(lock | {"trust": {"base": HYP | {"P.Sound": "a trust assumption"}}}, now, FILES)[0] == []
    own = {"P.Sound": "its rule"}
    assert A.judge(lock | {"trust": {"base": HYP, "extra": {"P.t": own}}}, now, FILES)[0] == []
    elsewhere = {"trust": {"base": HYP, "extra": {"P.u": own}}, "trust_gap": {"P.t": ["P.Sound"]}}
    assert [(f.code, f.entry) for f in A.judge(lock | elsewhere, now, FILES)[0]] == [("trust.extra-unlisted", "P.u")]
    assert A.trust_gaps(lock) == {"P.t": ["P.Sound"]}
    assert [f.code for f in A.judge(lock | {"deployed": ["P.t", "P.gone"], "trust_gap": {"P.t": ["P.Sound"]}}, now,
                                    FILES)[0]] == ["trust.deployed-unlisted"]
    for bad in (["P.Hyp"], {"base": ["P.Hyp"]}, {"base": {"P.Hyp": "two\nlines"}}, {"extra": {}},
                {"base": HYP, "extra": {"P.t": {}}}, {"base": HYP, "extra": {"P.t": {"P.Sound": " "}}},
                {"base": HYP, "also": []}):
        assert [f.code for f in A.judge(lock | {"trust": bad}, now, FILES)[0]] == ["trust.form"], bad


def test_a_discharged_premise_fails_until_its_trust_gap_entry_drops_which_an_update_and_the_lock_run_do(
        tmp_path, monkeypatch):
    """`P.t`'s proof stops taking `P.Sound`: its build takes only `P.Hyp`, which `trust` lists.  Its `trust_gap` entry still
    listing `P.Sound` fails, as bookkeeping (not a record difference, so `--regenerate` alone doesn't settle it), until the
    entry drops: `--update` drops it, and so does the lock run under `--bookkeeping warn`, whose warning names the premise.
    The report holds each deployed guarantee's gap."""
    before = {**opened(sounded()), "trust": {"base": HYP}, "deployed": ["P.t"], "trust_gap": {"P.t": ["P.Sound"]}}
    assert A.judge(before, sounded(), FILES)[0] == []
    assert "trust.gap-gone" in [f.code for f in A.judge(before, facts(), FILES)[0]]
    lock = {**recorded(), "trust": {"base": HYP}, "deployed": ["P.t"], "trust_gap": {"P.t": ["P.Sound"]}}
    fails = A.judge(lock, facts(), FILES)[0]
    assert [(f.code, f.entry) for f in fails] == [("trust.gap-gone", "P.t")] and "lists P.Sound, which it no" in fails[0]
    assert A.BOOKKEEPING.match(fails[0]) and A.regenerated(lock, facts(), FILES, fails) is None
    up_fails, up = A.judge(lock, facts(), FILES, update=True)
    assert up_fails == [] and up["trust_gap"] == {}
    fixed = A.repaired(lock, facts(), FILES, fails, TICKET)
    assert fixed["trust_gap"] == {} and A.judge(fixed, facts(), FILES)[0] == []
    assert A.bookkeeping_warnings(fails, fixed, TICKET, lock) == [
        "bookkeeping: P.t (@lean): dropped from its `trust_gap` entry, outside its trust no longer: P.Sound "
        "(trust.gap-gone)"]
    pkg, audit = bookkeeping_audit(tmp_path, monkeypatch, lock | {"shape": {"frozen": {"P.t": "frozen: open"}}}, facts())
    plain, _ = audit()
    assert A.reported(plain)["failure_codes"] == ["trust.gap-gone"] and plain["trust_gap"] == {"P.t": []}
    rep, work = audit("2026-11-14")
    assert "bookkeeping_left" not in rep and json.loads((work / A.POLICY).read_text())["trust_gap"] == {}
    assert any(w.endswith("P.Sound (trust.gap-gone)") for w in rep["bookkeeping"])


def bookkeeping_audit(tmp_path, monkeypatch, lock: dict, now: dict, replayed=()):
    """A package whose build has ``now`` for facts (its Lean stubbed out), with ``lock`` for its lock, and a function auditing
    it with `--bookkeeping warn`'s due date or without it."""
    pkg = tmp_path / "pkg"
    for rel in FILES.values():
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text("")
    (pkg / "lean-toolchain").write_text("leanprover/lean4:v0\n")
    A.dump(pkg, lock | {"dependencies": {"mathlib": "d1"}})
    monkeypatch.setattr(A, "facts_of", lambda *a, **k: now | {"oleans": []})
    monkeypatch.setattr(A, "replay_step", lambda *a: (list(replayed), {"checked": 1, "reused": 0}))
    monkeypatch.setattr(A, "dependency_scans", lambda *a: ([], 0))
    monkeypatch.setattr(A, "dependency_digests", lambda *a: {"mathlib": "d1"})
    for name, value in (("scans", []), ("heartbeats", []), ("anonymous_instances", []), ("run_paths", []),
                        ("listings", ([], {}, [])), ("read_by_dependents", set()), ("reads_home", {})):
        monkeypatch.setattr(A, name, lambda *a, value=value, **k: value)

    def audit(bookkeeping=None):
        out = tmp_path / f"out-{bookkeeping}"
        return A.audit(pkg, out, runs=False, log=lambda *_: None, bookkeeping=bookkeeping), out / "audit-pkg"
    return pkg, audit


def test_an_audit_under_the_flag_writes_the_repaired_lock_and_what_else_fails_still_fails(tmp_path, monkeypatch):
    """`audit(bookkeeping=DUE)`: a package that fails only on bookkeeping passes with the repaired lock beside its report
    (`regenerated`, which drive commits) and its warnings (`bookkeeping`), and its own lock untouched; the same package fails
    without the flag. What the flag never repairs still fails under it: an axiom beyond the standard three (`sorryAx`), a
    `native_decide` (`Lean.ofReduceBool`), a replay the kernel rejects, and a malformed lock. A lock that gives no
    `open_default_issue` passes too, the premise listed without one and a warning saying so."""
    lock = {**POLICY, "assumptions": {}, "shape": {"frozen": {"P.t": "frozen: open"}}, "open_default_issue": "LEAN-186"}
    pkg, audit = bookkeeping_audit(tmp_path, monkeypatch, lock, sounded())
    before = (pkg / A.POLICY).read_text()
    plain, _ = audit()
    assert "open.unlisted" in A.reported(plain)["failure_codes"] and "regenerated" not in plain
    rep, work = audit("2026-11-14")
    assert rep["failures"] and all(A.code_of(f) in set(A.BOOKKEEPING) | set(A.RECORD_DIFFERENCE) for f in rep["failures"]), (
        "the failures it repaired, which `main` reports as settled", rep["failures"])
    assert "bookkeeping_left" not in rep
    assert rep["regenerated"] == str(work / A.POLICY) and (pkg / A.POLICY).read_text() == before
    fixed = json.loads((work / A.POLICY).read_text())
    assert fixed["open"]["P.Sound"]["issue"] == "LEAN-186" and fixed["open"]["P.Sound"]["due"] == "2026-11-14"
    assert "bookkeeping: P.Sound (@lean): listed under `open` on LEAN-186, due 2026-11-14 (open.unlisted)" in rep["bookkeeping"]
    (sub := tmp_path / "issueless").mkdir()
    _, audit_bare = bookkeeping_audit(sub, monkeypatch, {k: v for k, v in lock.items() if k != "open_default_issue"}, sounded())
    rep, work = audit_bare("2026-11-14")
    assert "bookkeeping_left" not in rep and "issue" not in json.loads((work / A.POLICY).read_text())["open"]["P.Sound"]
    assert any(w.startswith("bookkeeping: P.Sound (@lean): listed under `open` without an issue") for w in rep["bookkeeping"])
    for name, now, replayed, broken in (
            ("sorry", sounded(offenders=[{"name": "P.t", "axioms": ["sorryAx"]}]), (), lock),
            ("native_decide", sounded(offenders=[{"name": "P.t", "axioms": ["Lean.ofReduceBool"]}]), (), lock),
            ("replay", sounded(), (A.Failure("replay.rejected", None, "replay: the kernel rejected P.t"),), lock),
            ("malformed", sounded(), (), lock | {"open_default_issue": 7})):
        sub = tmp_path / name
        sub.mkdir()
        _, audit = bookkeeping_audit(sub, monkeypatch, broken, now, replayed)
        rep, work = audit("2026-11-14")
        assert rep["failures"] and "regenerated" not in rep and "bookkeeping" not in rep, (name, rep)
        assert any(A.code_of(f) not in A.BOOKKEEPING for f in rep["failures"]), (name, rep["failures"])


def test_an_unwitnessed_entry_of_no_guarantee_fails_its_audit_and_an_update_or_the_lock_run_drops_it(tmp_path, monkeypatch):
    """An `unwitnessed` entry of a name the lock lists no guarantee under fails the audit, with the fix, and is no record
    difference, so nothing is regenerated for it. The lock run passes on the lock without it, with one warning, and an
    `--update` writes the lock without it and doesn't fail on it; the entry of the listed guarantee stays in both."""
    lock, now = bookkeeping_cases()["witnesses.unwitnessed-unlisted"]
    kept = {"P.sound": now["pins"]["P.sound"]["type_sha256"]}
    frozen = {"frozen": dict.fromkeys(lock["guarantees"], "frozen: open")}
    pkg, audit = bookkeeping_audit(tmp_path, monkeypatch, lock | {"default_owner": "@lean", "shape": frozen}, now)
    before = (pkg / A.POLICY).read_text()
    plain, _ = audit()
    assert plain["failures"] == [
        "witnesses: `unwitnessed` lists P.gone, which is not a guarantee of this package; run `check.py --update`, which "
        "drops the entry, or remove it"]
    assert A.reported(plain)["failure_codes"] == ["witnesses.unwitnessed-unlisted"] and "regenerated" not in plain
    rep, work = audit("2026-11-14")
    assert "bookkeeping_left" not in rep and (pkg / A.POLICY).read_text() == before
    assert json.loads((work / A.POLICY).read_text())["unwitnessed"] == kept
    assert rep["bookkeeping"] == ["bookkeeping: P.gone (@lean): its `unwitnessed` entry dropped: no guarantee has it "
                                  "(witnesses.unwitnessed-unlisted)"]
    updated = A.audit(pkg, tmp_path / "out-update", update=True, runs=False, log=lambda *_: None)
    assert updated["failures"] == [] and json.loads((pkg / A.POLICY).read_text())["unwitnessed"] == kept
    assert audit()[0]["failures"] == []


def test_the_lock_runs_flag_settles_a_repaired_package_and_lists_its_warnings(tmp_path, monkeypatch):
    """`--regenerate --bookkeeping warn`: a package whose audit repaired its bookkeeping (`regenerated` beside `bookkeeping`)
    passes, and audit.json lists each warning under its package's path for drive to ring; the flag goes only with
    `--regenerate`, and the audit is given the due date seven days from the run's."""
    a = tmp_path / "a"
    a.mkdir()
    (a / A.POLICY).write_text("{}")
    given = []
    unlisted = A.Failure("open.unlisted", "P.Sound", "open: P.Sound is unlisted")

    def fake(pkg, out, *args, bookkeeping=None, **_):
        given.append(bookkeeping)
        rec = out / f"audit-{pkg.name}" / A.POLICY
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text("{}")
        warned = ["bookkeeping: P.Sound (@lean): listed under `open` on LEAN-186, due D (open.unlisted)"]
        return {"package": str(pkg), "seconds": {}, "failures": [unlisted]} | (
            {"regenerated": str(rec), "bookkeeping": warned} if bookkeeping else {})

    monkeypatch.setattr(A, "audit", fake)
    out = tmp_path / "out"
    assert A.main([str(a), "--regenerate", "--out", str(out), "--repo", str(tmp_path)]) == 1 and given == [None]
    assert "bookkeeping" not in json.loads((out / "audit.json").read_text())
    assert A.main([str(a), "--regenerate", "--bookkeeping", "warn", "--out", str(out), "--repo", str(tmp_path)]) == 0
    doc = json.loads((out / "audit.json").read_text())
    assert doc["bookkeeping"] == ["a: bookkeeping: P.Sound (@lean): listed under `open` on LEAN-186, due D (open.unlisted)"]
    assert doc["regenerated"] == {"a": str(out / "audit-a" / A.POLICY)} and doc["packages"][0]["settled_codes"] == [
        "open.unlisted"]
    assert len(given) == 2 and len(given[1]) == 10 and given[1] > "2026-01-01"
    with pytest.raises(SystemExit):
        A.main([str(a), "--bookkeeping", "warn"])


def test_a_lock_run_s_audit_json_lists_each_statement_change_and_each_repair_with_its_owners(tmp_path, monkeypatch):
    """`check.py --regenerate --bookkeeping warn` itself (`main` and `audit`, the package's Lean stubbed out) on a package
    whose lock has no record of its guarantee (a record difference) and doesn't list a premise it takes (bookkeeping):
    audit.json's package, and its report.json, list each statement change of the regenerated record (`changes`) and each
    repair (`repairs`) with the entry, its kind or its failure's code, its owners and the printed line; the lines are the
    review's heads and the warnings, so whoever rings the owners reads no text.  Without `--bookkeeping warn` that package
    fails and lists neither.  A package that fails on a record difference alone lists its changes, and no repairs unless
    the flag ran (then none); one that passes lists neither.  The verdicts are those the run had without the lists."""
    audit = A.audit
    monkeypatch.setattr(A, "audit", lambda *a, seals=None, **k: audit(*a, **k))   # a seal copies a real build

    def run(name, lock, now, *flags):
        (sub := tmp_path / name).mkdir(exist_ok=True)
        pkg, _ = bookkeeping_audit(sub, monkeypatch, lock, now)
        code = A.main([str(pkg), "--regenerate", *flags, "--no-runs", "--out", str(sub / "out"), "--repo", str(tmp_path)])
        doc = json.loads((sub / "out" / "audit.json").read_text())
        (rep,) = doc["packages"]
        assert json.loads((sub / "out" / "audit-pkg" / "report.json").read_text()).get("changes") == rep.get("changes")
        return code, doc, rep
    frozen = {"shape": {"frozen": {"P.t": "frozen: open"}}}
    lock = {**POLICY, "assumptions": {}, "open_default_issue": "LEAN-186"} | frozen
    code, doc, rep = run("both", lock, sounded(), "--bookkeeping", "warn")
    assert code == 0 and rep["settled_codes"] == ["guarantees.unrecorded", "open.unlisted"] and rep["statement_changes"]
    assert rep["changes"] == [
        {"entry": "P.t", "kind": "guarantees", "owners": ["@lean"], "line": "guarantee P.t: new"},
        {"entry": "P.Hyp", "kind": "reads", "owners": ["@lean"], "line": "definition P.Hyp (P.Assumptions), read by P.t: new"},
        {"entry": "P.f", "kind": "reads", "owners": ["@lean"], "line": "definition P.f (P.Defs), read by P.t: new"}]
    assert [c["line"] for c in rep["changes"]] == [n for n in A.statement_changes(rep["review"]) if n[:1] != " "]
    (fix,) = rep["repairs"]
    assert (fix["entry"], fix["code"], fix["owners"]) == ("P.Sound", "open.unlisted", ["@lean"])
    assert [fix["line"]] == rep["bookkeeping"] and doc["bookkeeping"] == [f"both/pkg: {fix['line']}"]
    assert fix["line"].startswith("bookkeeping: P.Sound (@lean): listed under `open` on LEAN-186, due ")
    assert set(doc) == {"passed", "failed", "packages", "regenerated", "bookkeeping"}, "the lists are their package's"

    code, doc, rep = run("both", lock, sounded())
    assert code == 1 and rep["failure_codes"] == ["guarantees.unrecorded", "open.unlisted"]
    assert not {"changes", "repairs", "bookkeeping", "regenerated", "settled"} & set(rep)

    was = recorded() | frozen
    changed = facts(pins={"P.t": fact(signature="P.t (h : P.Hyp) (n : Nat) : P.f n ≤ n", h="00000000000000bb")})
    code, _, rep = run("record", was, changed)
    assert code == 0 and rep["settled_codes"] == ["guarantees.signature"] and "repairs" not in rep and "bookkeeping" not in rep
    assert rep["changes"] == [{"entry": "P.t", "kind": "guarantees", "owners": ["@lean"], "line": "guarantee P.t: changed"}]
    code, _, warned = run("record", was, changed, "--bookkeeping", "warn")
    assert code == 0 and warned["changes"] == rep["changes"] and warned["repairs"] == warned["bookkeeping"] == []

    for flags in ((), ("--bookkeeping", "warn")):
        code, doc, rep = run("passes", was, facts(), *flags)
        assert code == 0 and doc["regenerated"] == {} and rep["failures"] == [] and "settled" not in rep
        assert not {"changes", "repairs", "bookkeeping", "review"} & set(rep)


def test_reports_list_their_failures_codes_beside_them(tmp_path, monkeypatch):
    """`report.json`'s and `audit.json`'s packages carry `failure_codes` and `failure_entries`, one per failure in its order
    (the entry each is about, which with its code names it where its text must not), and the Lean check's `--regenerate`
    moves both with the failures it settles (`settled_codes`, `settled_entries`)."""
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
        assert rep["failure_entries"] == ["P.Defs", "X.lean"][:len(codes)]
        assert rep.get("settled_entries") == (settled and ["P.Defs"])
        assert len(rep["failures"]) == len(rep["failure_codes"])
    several = A.Failure("trees.form", None, "trees: `trees` is not an object")
    got = A.reported({"failures": [gone, "a bare string", several]})
    assert got["failure_codes"] == ["guarantees.reads-gone", None, "trees.form"]
    assert got["failure_entries"] == ["P.Defs", None, None], "null for a bare string and for a failure about no one entry"


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
    guarantee's reads reach, it takes nearly all of `security`'s audit."""
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


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_facts_from_judges_from_a_stored_document_as_a_full_audit_does(tmp_path):
    """Stage B of the lock run (`--facts-from`, `audit_from_facts`) gives the full audit's verdict and record from the
    stored facts document alone, with no build: the control fixture passes both, each lock violation fails both the same,
    and an `--update` writes the same lock byte for byte. The document is schema-versioned and position-free, so no file's
    bytes, a declaration's line or a module's build path leaks into what the graph's early cutoff compares."""
    def noop(*_):
        return None
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    A.audit(pkg, tmp_path / "pop", build=False, replay=False, runs=False, update=True, log=noop)
    recorded = A.load(pkg)

    doc = tmp_path / A.FACTS_DOC
    full = A.audit(pkg, tmp_path / "full", build=False, replay=False, runs=False, facts_out=doc, log=noop)
    assert full["failures"] == [], full["failures"]

    body = json.loads(gzip.decompress(doc.read_bytes()))
    assert body["schema"] == A.FACTS_SCHEMA and "spans" not in body["facts"] and "oleans" not in body["facts"]
    main = body["facts"]["pins"]["Control.main_thm"]
    assert "at" not in main and len(main["type_sha256"]) == 64 and main["hypotheses"], "the augmented facts, position-free"
    assert all("olean" not in g for g in body["facts"]["graph"].values()) and "/.elan/" not in json.dumps(body)
    assert doc.read_bytes()[4:8] == bytes(4), "no timestamp in the gzip, so equal facts are equal bytes"

    assert A.audit_from_facts(pkg, tmp_path / "sb", doc, log=noop)["failures"] == []

    # an --update from the document writes the lock an --update from the build writes, byte for byte, from the same start
    stripped = json.loads(json.dumps(recorded))
    for g in stripped["guarantees"].values():
        for k in ("type_hash", "signature", "premises", "assumptions"):
            g.pop(k, None)
    stripped.pop("dependencies", None)
    for a in stripped.get("assumptions", {}).values():
        a.pop("digest", None)
    A.dump(pkg, stripped)
    before = (pkg / A.POLICY).read_text()
    A.audit(pkg, tmp_path / "uf", build=False, replay=False, runs=False, update=True, log=noop)
    from_build = (pkg / A.POLICY).read_text()
    A.dump(pkg, stripped)
    A.audit_from_facts(pkg, tmp_path / "us", doc, update=True, log=noop)
    assert (pkg / A.POLICY).read_text() == from_build != before, "the record from the document is the record from the build"

    # each lock violation fails stage B exactly as the full audit, with the document (the facts) untouched
    mutations = {
        "a guarantee's recorded type hash": lambda l: l["guarantees"]["Control.main_thm"].__setitem__("type_hash", "0" * 64),
        "a guarantee with no owner": lambda l: l["guarantees"]["Control.main_thm"].pop("owner"),
        "a recorded assumption's digest": lambda l: l["assumptions"]["Control.Hyp"].__setitem__("digest", "0" * 64),
        "a witness naming a guarantee that is gone": lambda l: l["claims"]["witnesses"]["Control.Small"].__setitem__("fails", "Control.Nope"),
        "the dependencies' recorded toolchain digest": lambda l: l["dependencies"].__setitem__("toolchain", "0" * 64),
    }
    for name, mutate in mutations.items():
        bad = json.loads(json.dumps(recorded))
        mutate(bad)
        A.dump(pkg, bad)
        fb = A.audit(pkg, tmp_path / "mb", build=False, replay=False, runs=False, log=noop)["failures"]
        sb = A.audit_from_facts(pkg, tmp_path / "ms", doc, log=noop)["failures"]
        assert fb, f"the full audit itself missed {name}"
        assert list(map(str, sb)) == list(map(str, fb)), f"stage B differs from the full audit on {name}"
        assert [f.code for f in sb] == [f.code for f in fb], f"the codes differ on {name}"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_facts_from_applies_the_reflexive_witness_exemption_as_the_full_audit_does(tmp_path):
    """The reflexive exemption lives in `judge` (`witness_scope` -> `reflexive`), which `--facts-from` runs, so a reflexive
    guarantee (the claims tuples' `claims = claims`) is held to no witnesses in stage B exactly as in a full audit, and a
    non-reflexive `Eq` that stays in scope fails both the same. Stage B reads the reflexivity from the stored facts
    (`body`), never from a build."""
    def noop(*_):
        return None
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    (pkg / "Control" / "Claims.lean").write_text(REFLEXIVE)
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.Claims\n")
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    names = ["Control.SecurityProofs.Stated", "Control.statedAt", "Control.same"]
    policy = unclaimed(A.load(pkg))
    policy["guarantees"] = policy["guarantees"] | {n: {"owner": "@lean"} for n in names}
    A.dump(pkg, policy)
    A.audit(pkg, tmp_path / "pop", build=False, replay=False, runs=False, update=True, log=noop)

    doc = tmp_path / A.FACTS_DOC
    full = A.audit(pkg, tmp_path / "full", build=False, replay=False, runs=False, facts_out=doc, log=noop)
    sb = A.audit_from_facts(pkg, tmp_path / "sb", doc, log=noop)
    assert list(map(str, sb["failures"])) == list(map(str, full["failures"]))
    # the two reflexive guarantees are exempt and the non-reflexive `Holds (f n) = Holds (g n)` is not, in stage B as in full
    body = json.loads(gzip.decompress(doc.read_bytes()))["facts"]["pins"]
    assert [A.reflexive(body[n]) for n in names] == [True, True, False]
    assert [f.entry for f in sb["failures"] if (f.code or "").startswith("witnesses")] == ["Control.same"]


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_facts_only_writes_the_document_facts_from_judges_under_its_build_s_key(tmp_path):
    """Stage A (`--facts-only`, `audit(facts_cache=)`) writes the package's facts document under its content address
    (`facts_key`), so `--facts-from` judges from it with no build and gives the full audit's verdict. The key names the
    build the facts were read from: a second read of the same `.olean` files writes the identical document under the same
    key, and a change to the lock slice `Facts.lean` reads moves it. A lock failure does not fail stage A, which has
    still written the document for `--facts-from` to judge."""
    def noop(*_):
        return None
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    A.audit(pkg, tmp_path / "pop", build=False, replay=False, runs=False, update=True, log=noop)  # record so it passes
    recorded = A.load(pkg)

    cache = tmp_path / "facts-cache"
    a = A.audit(pkg, tmp_path / "a", build=False, replay=False, runs=False, facts_cache=cache, log=noop)
    key = a["facts_key"]
    doc = cache / f"{key}.json.gz"
    assert a["failures"] == [] and a["facts_document"] == str(doc) and doc.is_file(), a["failures"]
    body = json.loads(gzip.decompress(doc.read_bytes()))
    assert body["schema"] == A.FACTS_SCHEMA and "oleans" not in body["facts"] and "proved_keys" not in body["facts"]

    assert A.audit_from_facts(pkg, tmp_path / "sb", doc, log=noop)["failures"] == [], "stage B on stage A's document"
    assert A.main([str(pkg), "--facts-from", str(doc), "--out", str(tmp_path / "cli")]) == 0

    # the key names the build: a second read of the same .olean files is the identical document under the same key
    b = A.audit(pkg, tmp_path / "b", build=False, replay=False, runs=False, facts_cache=cache, log=noop)
    assert b["facts_key"] == key and (cache / f"{key}.json.gz").read_bytes() == doc.read_bytes()
    assert [p.name for p in cache.iterdir()] == [f"{key}.json.gz"], "one document, not one per read"

    # a change to the lock slice Facts.lean reads (a guarantee name) moves the key to a different address
    grown = json.loads(json.dumps(recorded))
    grown["guarantees"]["Control.unused"] = {"owner": "@lean"}
    A.dump(pkg, grown)
    c = A.audit(pkg, tmp_path / "c", build=False, replay=False, runs=False, facts_cache=cache, log=noop)
    assert c["facts_key"] != key
    A.dump(pkg, recorded)

    # stage A writes the document even when the lock fails; the CLI passes, and --facts-from reports the failure the
    # full audit does, from that very document
    bad = json.loads(json.dumps(recorded))
    bad["guarantees"]["Control.main_thm"]["type_hash"] = "0" * 64
    A.dump(pkg, bad)
    fresh = tmp_path / "fresh-cache"
    assert A.main([str(pkg), "--facts-only", str(fresh), "--out", str(tmp_path / "fo")]) == 0, \
        "stage A passes once it has written the document, whatever the lock says"
    (stored,) = list(fresh.glob("*.json.gz"))
    sb = A.audit_from_facts(pkg, tmp_path / "sbx", stored, log=noop)["failures"]
    fb = A.audit(pkg, tmp_path / "fb", build=False, replay=False, runs=False, log=noop)["failures"]
    assert sb and list(map(str, sb)) == list(map(str, fb)), "stage B on the stored document is the full audit's verdict"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_facts_key_folds_source_digests_so_a_scanned_source_edit_with_the_same_olean_misses(tmp_path):
    """`facts_key` folds the digest of every source file the document records and of the lakefile, so an edit the source
    scans read changes the key even when it leaves the `.olean` bytes unchanged. Lean's own example is a `set_option
    maxHeartbeats`, which `heartbeats` reads into a finding the document carries: without the fold the key would match and
    the stored finding would be reused against a source it no longer describes. The edit here is not rebuilt, so the facts
    (the `.olean` files) are identical and only the source digest moves, giving a different key and a cache miss."""
    def noop(*_):
        return None
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    A.audit(pkg, tmp_path / "pop", build=False, replay=False, runs=False, update=True, log=noop)  # record so it passes

    cache = tmp_path / "facts-cache"
    key = A.audit(pkg, tmp_path / "a", build=False, replay=False, runs=False, facts_cache=cache, log=noop)["facts_key"]

    # a top-level line the heartbeats scan reads, appended without a rebuild: the .olean is untouched
    rel = sorted(A.lean_modules(pkg).values())[0]
    (pkg / rel).write_text((pkg / rel).read_text() + "\nset_option maxHeartbeats 400000\n")

    after = A.audit(pkg, tmp_path / "b", build=False, replay=False, runs=False, facts_cache=cache, log=noop)
    assert after["facts_key"] != key, "a scanned source edit with the same .olean must change the key"
    assert {p.name for p in cache.iterdir()} == {f"{key}.json.gz", f"{after['facts_key']}.json.gz"}, "a new address: a cache miss"
    # the finding the edit changes is in the new document, so reusing the old one would have been stale
    body = json.loads(gzip.decompress((cache / f"{after['facts_key']}.json.gz").read_bytes()))
    assert any(h["option"] == "maxHeartbeats" and h["value"] == 400000 for h in body["findings"]["heartbeats"])


def test_build_inputs_holds_what_the_lock_gives_every_step_before_the_judge():
    """`build_inputs`, which a lock run compares before it keeps a pass for the commit of its regenerated locks
    (`lean_audit.keep_regenerated`).  A record's value is not in it (a statement's hash, an assumption's digest, a `reads`
    group, an owner), nor is a section only the judge reads; what `Facts.lean` walks is (a listed guarantee, a listed
    premise, a name an inline text prints, an honest witness, `shape.premises`), and so is every other section, one this
    file doesn't know included."""
    pol = recorded()
    base = A.build_inputs(pol)
    stale = json.loads(json.dumps(pol))
    stale["guarantees"]["P.t"] |= {"type_hash": "0" * 64, "owner": "@other"}
    stale["assumptions"]["P.Hyp"] |= {"digest": "0" * 64, "owner": "@other"}
    stale["reads"] = {m: g | {"digest": "0" * 64} for m, g in stale["reads"].items()} | {"P.More": {"digest": "1" * 64}}
    stale |= {"format": 0, "unwitnessed": {"P.t": "0" * 64}, "trust_gap": {"P.t": "why"}, "witnesses_assumed": {"P.t": []},
              "open_seed": {"P.t": []}, "witnesses": {"P.t": {"meets": ["P.w"]}},
              "shape": {"properties": {"P.t": {"function": "P.f", "outcome": "none"}}, "frozen": {"P.t": "0" * 64}}}
    assert stale != pol and A.build_inputs(stale) == base
    for change in [{"guarantees": pol["guarantees"] | {"P.u": {"owner": "@lean"}}},      # a listed guarantee
                   {"assumptions": {}}, {"open": {"P.Open": {"owner": "@lean"}}}, {"domain": {"P.Dom": {}}},
                   {"shape": {"inline": {"P.t": {"P.Small 3": "why"}}}},                 # a name an inline text prints
                   {"shape": {"premises": ["P.Premises"]}},
                   {"witnesses": {"P.t": {"honest": ["P.w"]}}},
                   {"roots": ["P", "Q"]}, {"exempt": {"Scratch": "why"}}, {"runs": {"T": {}}}, {"proved_in": ["../q"]},
                   {"compile_time": {"P.Gen": "why"}}, {"dependency_unchecked": {"D": "why"}}, {"upstream": {}},
                   {"axioms": []}, {"meaning": ["P.Defs"]}, {"layers": {}}, {"trees": {}}, {"claims": {}},
                   {"a_section_of_a_later_check": 1}]:
        assert A.build_inputs(pol | change) != base, change
    with pytest.raises(KeyError):
        A.build_inputs({})


def test_no_step_before_the_judge_reads_a_section_build_inputs_leaves_out():
    """`build_inputs` leaves `JUDGED` out, so no step of `audit` before the judge may read one of those sections: a lock
    run keeps a pass for the commit of its regenerated locks on the ground that their audit would build, seal, extract,
    replay and test exactly what the run did (`lean_audit.keep_regenerated`).  This walks `check.py`: from `audit`, through
    every function it hands the lock to (by position or keyword, through `timed` and an executor's `submit`), short of
    the ones `audit_from_facts` runs on the live lock too (the judge) and of `facts_config`, whose slice `build_inputs`
    holds.  The exceptions, each read by hand: `proved` takes `guarantees`, `shape` and `witnesses` and `listed_premises`
    the premise sections only to fill the lock slice it gives the prover's `facts_of` (`facts_config` again) and
    `assign` (the names); it loads the prover's lock itself (`qpol`), whose `guarantees` it reads as empty or not.

    A lock loaded by its path is handed down by nobody, so that walk does not meet it.  Every call of `load` and every
    use of `POLICY` in `check.py`, a mention in an f-string and a test that the file is there aside, is therefore in a
    function listed here with the sections it reads of that lock, which the same walk finds from the load.  One that a
    step before the judge runs (`before`) reads only what `build_inputs` compares; a new one fails until it is listed,
    and as a handed-down read does when it reads more.  The others (`apart`) are the judge's, called only where
    `audit_from_facts` calls them too, on the locks of the tree it is given, or no step of an audit.  The lock's file is
    named by `POLICY` alone, and what `check.py` calls of the siblings it loads never names it."""
    judge = {"_finish", "judge", "inline_failures", "renamed_current", "audit.judged", "facts_config"}
    names = {"proved": {"guarantees", "shape", "witnesses"}, "listed_premises": {"assumptions", "domain", "open"}}
    funcs: dict[str, ast.FunctionDef] = {}
    module = ast.parse((HERE.parent / "check.py").read_text())
    for top in module.body:
        for n in ast.walk(top):
            if isinstance(n, ast.FunctionDef):
                funcs[n.name if n is top else f"{top.name}.{n.name}"] = n

    def called(n: ast.Call) -> tuple[ast.AST, list[ast.AST]]:
        """(What ``n`` calls, its positional arguments), through `timed` and an executor's `submit`."""
        f, args = n.func, list(n.args)
        while args and isinstance(args[0], ast.Name) and (
                (isinstance(f, ast.Attribute) and f.attr == "submit") or (isinstance(f, ast.Name) and f.id == "timed")):
            f, args = args[0], args[1:]
        return f, args

    def walk(todo: list[tuple[str, set[str]]], short: set[str] = judge) -> tuple[dict[str, set[str]], set[str], set[str]]:
        """({function: the sections it reads of the lock it holds}, the functions that use that lock in a way this does
        not read, the calls that hand it past the callee's parameters) from ``todo``, each a function and the names it
        holds the lock under, short of the functions in ``short``."""
        reads: dict[str, set[str]] = {}
        odd, loose = set(), set()
        while todo:
            name, held = todo.pop(0)
            if name in reads or name in short:
                continue
            keys, used = reads.setdefault(name, set()), set()
            for n in ast.walk(funcs[name]):
                f = getattr(n, "func", None)
                if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in held \
                        and isinstance(n.slice, ast.Constant):
                    keys.add(n.slice.value)
                    used.add(id(n.value))
                elif isinstance(n, ast.Compare) and len(n.ops) == 1 \
                        and isinstance(n.ops[0], (ast.Is, ast.IsNot, ast.In, ast.NotIn)):
                    for side, other in ((n.left, n.comparators[0]), (n.comparators[0], n.left)):
                        if isinstance(side, ast.Name) and side.id in held and (isinstance(n.ops[0], (ast.Is, ast.IsNot)) or (
                                side is n.comparators[0] and isinstance(other, ast.Constant))):
                            keys.update([other.value] if isinstance(n.ops[0], (ast.In, ast.NotIn)) else [])
                            used.add(id(side))
                elif isinstance(n, ast.Call) and isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                        and f.value.id in held and f.attr == "get" and n.args and isinstance(n.args[0], ast.Constant):
                    keys.add(n.args[0].value)
                    used.add(id(f.value))
                elif isinstance(n, ast.Call):
                    f, args = called(n)
                    given = [(i, a) for i, a in enumerate(args)] + [(k.arg, k.value) for k in n.keywords]
                    for at, a in given:
                        if not (isinstance(a, ast.Name) and a.id in held):
                            continue
                        used.add(id(a))
                        callee = f.id if isinstance(f, ast.Name) else None
                        callee = next((c for c in (f"{name}.{callee}", f"{callee}.__init__", callee) if c in funcs), None)
                        params = [p.arg for p in funcs[callee].args.args][callee.endswith(".__init__"):] if callee else []
                        if isinstance(at, int) and at >= len(params):
                            loose.add(f"{name} -> {ast.unparse(f)}")
                        else:
                            todo.append((callee, {params[at] if isinstance(at, int) else at}))
            if any(isinstance(n, ast.Name) and n.id in held and isinstance(n.ctx, ast.Load) and id(n) not in used
                   for n in ast.walk(funcs[name])):
                odd.add(name)
        return reads, odd, loose

    reads, odd, loose = walk([("audit", {"policy", "current"}), ("proved", {"policy", "qpol"}),
                              ("replay_in_steps", {"policy"})])
    # `final_replay(*args)` is `replay_in_steps(*args)`, walked from here, which only hands the lock to `replay_step`;
    # `Pipeline` keeps the lock, and only to hand it to `replay_step`; `facts_key` reads `FINDINGS_SECTIONS` by a variable
    assert loose == {"audit -> final_replay"} and odd == {"Pipeline.__init__", "facts_key"}, (loose, odd)
    pipeline = ast.unparse(next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "Pipeline"))
    assert pipeline.count("self.policy") == 2 and "replay_step(self.pkg, self.policy," in pipeline
    assert {"tests", "scans", "listings", "dependency_scans", "facts_of", "facts_key", "proved", "lift", "replay_step",
            "Pipeline.__init__", "run_paths"} <= set(reads), sorted(reads)
    assert not set(A.FINDINGS_SECTIONS) & A.JUDGED

    parent = {id(c): p for p in ast.walk(module) for c in ast.iter_child_nodes(p)}
    owner = {id(n): name for name, f in funcs.items() for n in ast.walk(f)}     # the innermost: a nested one is listed later

    def sections(lock: ast.AST) -> set[str] | None:
        """The sections read of the lock the expression ``lock`` gives: the one it is asked for there, or what the walk
        finds from the name it is bound to or from the function it is handed to; None (all of it) when this can't tell."""
        use, key = parent[id(lock)], None
        if isinstance(use, ast.Attribute) and use.attr == "get" and isinstance(got := parent[id(use)], ast.Call) and got.args:
            key = got.args[0]
        elif isinstance(use, ast.Subscript) and use.value is lock:
            key = use.slice
        if isinstance(key, ast.Constant):
            return {key.value}
        if isinstance(use, ast.Assign) and len(use.targets) == 1 and isinstance(use.targets[0], ast.Name):
            start = (owner[id(lock)], {use.targets[0].id})
        elif isinstance(use, ast.Call) and lock in use.args and isinstance(use.func, ast.Name) and use.func.id in funcs:
            start = (use.func.id, {funcs[use.func.id].args.args[use.args.index(lock)].arg})
        else:
            return None
        found, unread, past = walk([start], set())
        return None if unread or past else set().union(*found.values())

    by_path: dict[str, set[str] | None] = {}
    for n in ast.walk(module):
        if not (isinstance(n, ast.Name) and n.id in ("load", "POLICY") and isinstance(n.ctx, ast.Load)):
            continue
        use, lock = parent[id(n)], None
        if n.id == "load":
            lock = use if isinstance(use, ast.Call) and use.func is n else None
        else:
            path = isinstance(use, ast.BinOp) and isinstance(use.op, ast.Div) and use.right is n
            how = parent[id(use)] if path else use
            if isinstance(how, ast.FormattedValue) or (path and isinstance(how, ast.Attribute) and how.attr in ("exists", "is_file")):
                continue
            # `json.loads((pkg / POLICY).read_text())` is the lock, as `load(pkg)` is
            parsed = parent[id(parent[id(how)])] if path and isinstance(how, ast.Attribute) and how.attr == "read_text" else None
            lock = parsed if isinstance(parsed, ast.Call) and ast.unparse(parsed.func) == "json.loads" else None
        got, name = None if lock is None else sections(lock), owner.get(id(n), "")
        by_path[name] = None if got is None or by_path.get(name, set()) is None else by_path.get(name, set()) | got
    whole = None
    before = {"memory_refusal": {"proved_in"}, "main": {"proved_in"},       # which packages a run audits, and on what machine
              "proved": {"roots", "exempt", "guarantees", "compile_time"}}   # the prover's lock, `qpol` above
    apart = {"_open": whole,                                                 # the lock the walk starts from: `policy` in `audit`
             "homes": {"layers"}, "tree_roles": {"trees"}, "read_by_dependents": {"reads"}, "_finish": whole,   # the judge's
             "load": whole, "dump": whole,                                   # the reader and the writer themselves
             "packages": whole,                                              # the file's name, among a directory's
             "controls_of": whole,                                           # a control's copy of a fixture
             "unowned_at": whole}                                            # the report's `unowned`, of the judged lock
    for name, got in by_path.items():
        if name not in apart:
            reads.setdefault(name, set()).update(A.JUDGED if got is None else got)
    for name, keys in sorted(reads.items()):
        assert not (bad := (keys & A.JUDGED) - names.get(name, set())), (
            f"`{name}` reads the lock's {', '.join(sorted(bad))} before the judge, so two locks that differ there no longer "
            f"give an audit the same build, facts, replay or tests: read it in the judge, or take the section out of `JUDGED` "
            f"(`build_inputs` then compares it, and a lock run that rewrites it keeps no pass for its lock commit)")
    listed = before | apart
    changed = {f: by_path.get(f, "no load") for f in sorted({*by_path, *listed}) if by_path.get(f, "") != listed.get(f, "")}
    assert not changed, (
        f"check.py's loads of a lock by its path changed ({changed}): list each function with the sections it reads of that "
        f"lock, under `before` when a step of `audit` before the judge runs it, else under `apart` with why it is apart")

    def callers(name: str) -> set[str]:
        return {owner.get(id(n), "") for n in ast.walk(module) if isinstance(n, ast.Call) and getattr(called(n)[0], "id", None) == name}

    both, judged = {"audit", "audit_from_facts"}, {"audit.judged", "audit_from_facts.judged"}
    assert {f: callers(f) for f in ("_open", "_finish", "read_by_dependents", "tree_roles", "reads_home", "homes")} == {
        "_open": both, "_finish": both, "read_by_dependents": both, "tree_roles": judged, "reads_home": judged,
        "homes": {"reads_home"}}, "a load that is apart as the judge's is called only where `audit_from_facts` calls it too"
    assert all("rep, policy, work = _open(pkg, out)" in ast.unparse(funcs[f]) for f in both), "the name the walk starts from"
    assert [n.value for n in ast.walk(module) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and (n.value == A.POLICY or n.value.endswith("/" + A.POLICY))] == [A.POLICY], "a lock's path that `POLICY` does not name"
    # the siblings check.py loads name the lock's file in `policy_roots` (its `roots`, for `lean_audit.py`'s index) and in
    # `upstream.py`'s own command line, and check.py calls neither
    siblings = {c.args[0].value for c in ast.walk(module) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "sibling"}
    naming = {f"{s}.{getattr(top, 'name', '')}" for s in siblings for top in ast.parse((HERE.parent / f"{s}.py").read_text()).body
              if any((isinstance(v := getattr(n, "value", None), str) and A.POLICY in v) or getattr(n, "id", None) == "POLICY"
                     for n in ast.walk(top))}
    assert naming == {"module_keys.policy_roots", "upstream.", "upstream.main"}, (
        f"of the siblings check.py loads ({', '.join(sorted(siblings))}), these name the lock's file: {sorted(naming)}; one that "
        f"a step before the judge calls reads the lock by its path, where this walk does not follow it")
    assert not {"policy_roots", "main"} & {n.attr for n in ast.walk(module) if isinstance(n, ast.Attribute)}, \
        "check.py calls a sibling's function that reads the lock by its path"


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_two_locks_that_differ_only_in_records_give_the_same_facts_document_and_replay(tmp_path, sandboxed_node):
    """What a lock run rests on when it keeps a pass for the commit of its regenerated locks (`lean_audit.keep_regenerated`),
    on a real build: audited under two locks that differ only in record fields (a statement's hash, an assumption's
    digest), one build has the same `build_inputs`, the same `facts_key` (which folds every own module's `replay_keys`,
    the sources and the lock slice `Facts.lean` reads) and a byte-identical facts document, each written by its own
    extraction, and the replay under the second lock checks no module again.  The stale lock's regenerated lock, judged
    from that document (`audit_from_facts`), leaves what a full audit of it leaves: nothing.  A lock that lists a name
    the other doesn't (an inline entry of no guarantee, which a regeneration drops) differs on `build_inputs`, and
    `Facts.lean` writes another document for it."""
    def noop(*_):
        return None
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    A.audit(pkg, tmp_path / "pop", build=False, replay=False, runs=False, update=True, log=noop)
    recorded = A.load(pkg)

    def audited(name: str, lock: dict) -> tuple[dict, bytes]:
        A.dump(pkg, lock)
        rep = A.audit(pkg, tmp_path / name, runs=False, facts_cache=tmp_path / f"{name}-facts", facts_reuse=True,
                      replay_cache=tmp_path / "replay", store=True, log=noop)
        return rep, (tmp_path / f"{name}-facts" / f"{rep['facts_key']}.json.gz").read_bytes()

    a, doc = audited("a", recorded)
    assert a["failures"] == [] and counts(a)[0] > 0 and counts(a)[1] == 0

    stale = json.loads(json.dumps(recorded))
    stale["guarantees"]["Control.main_thm"]["type_hash"] = "0" * 64
    stale["assumptions"]["Control.Hyp"]["digest"] = "0" * 64
    b, again = audited("b", stale)
    assert b["failures"] and all(A.RECORD_DIFFERENCE.match(f) for f in b["failures"]), b["failures"]
    assert A.build_inputs(stale) == A.build_inputs(recorded)
    assert b["facts_key"] == a["facts_key"] and again == doc, "the same facts document, extracted again"
    assert counts(b) == (0, counts(a)[0]), "and the same replay inputs: every module's key is the one that passed"

    regenerated = A.load(Path(b["regenerated"]).parent)
    assert regenerated == recorded and A.build_inputs(regenerated) == A.build_inputs(stale)
    A.dump(pkg, regenerated)
    stored = tmp_path / "b-facts" / f"{b['facts_key']}.json.gz"
    assert A.audit_from_facts(pkg, tmp_path / "sb", stored, log=noop)["failures"] == []
    assert A.audit(pkg, tmp_path / "full", runs=False, replay_cache=tmp_path / "replay", store=True, log=noop)["failures"] == []

    inline = json.loads(json.dumps(recorded))
    inline.setdefault("shape", {}).setdefault("inline", {})["Control.gone"] = {"Control.Small 3": "why"}
    c, other = audited("c", inline)
    dropped = A.load(Path(c["regenerated"]).parent)
    assert "Control.gone" not in (dropped.get("shape") or {}).get("inline", {})
    assert A.build_inputs(dropped) != A.build_inputs(inline), "what a lock run compares, so it keeps no pass here"
    assert c["facts_key"] != a["facts_key"] and other != doc


def test_the_facts_cache_reads_a_document_only_by_its_key_and_refuses_a_tampered_or_stale_one(tmp_path):
    """The facts cache (`store_cached_facts`, `read_cached_facts`) a reuse reads looks a document up only by the `facts_key`
    the current tree recomputes, never a caller's path, and judges it only when its sidecar vouches for it: a document whose
    bytes don't match the sidecar's SHA-256 is refused, so is a sidecar recording another key, and a missing sidecar (a
    crash between the two writes) is a miss, never a hit. The body carries no key, so it stays the bytes `--facts-only`
    writes and the graph's content cutoff compares."""
    cache = tmp_path / "facts"
    doc = {"schema": A.FACTS_SCHEMA, "toolchain": "t", "files": {"M": "M.lean"}, "facts": {"x": 1}, "findings": {}}
    key, other = "a" * 64, "b" * 64

    A.store_cached_facts(cache, key, doc)
    assert sorted(p.name for p in cache.iterdir()) == [f"{key}.json", f"{key}.json.gz"], "the pair, and no temporary file"
    assert A.read_cached_facts(cache, key) == doc, "a hit by the recomputed key"
    stored = (cache / f"{key}.json.gz").read_bytes()
    assert stored == A.facts_bytes(doc) and "key" not in json.loads(gzip.decompress(stored)), "the --facts-only bytes"
    assert A.read_cached_facts(cache, other) is None, "a key no run stored is a miss"

    A.write_facts_document(cache / f"{key}.json.gz", doc | {"facts": {"x": 2}})
    with pytest.raises(RuntimeError, match="SHA-256: tampered"):
        A.read_cached_facts(cache, key)

    A.store_cached_facts(cache, other, doc)
    A.store_cached_facts(cache, key, doc)
    (cache / f"{key}.json").write_text((cache / f"{other}.json").read_text())
    with pytest.raises(RuntimeError, match="stale or misfiled"):
        A.read_cached_facts(cache, key)

    A.store_cached_facts(cache, key, doc)
    side = json.loads((cache / f"{key}.json").read_text())
    (cache / f"{key}.json").write_text(json.dumps(side | {"schema": "lean-facts/v0"}))
    with pytest.raises(RuntimeError, match="refused"):
        A.read_cached_facts(cache, key)
    for garbage in ("[]", "{not json"):
        (cache / f"{key}.json").write_text(garbage)
        with pytest.raises(RuntimeError, match="refused"):
            A.read_cached_facts(cache, key)

    A.store_cached_facts(cache, key, doc)
    (cache / f"{key}.json").unlink()
    assert A.read_cached_facts(cache, key) is None, "a document without its sidecar is a miss, never a hit"


def test_two_writers_of_one_key_each_write_their_own_temporary_file_and_the_last_rename_wins(tmp_path, monkeypatch):
    """Two writers of one facts or replay entry (`write_atomic`), the second running whole inside the first's rename, with
    the process and thread ids they'd share in different PID namespaces: a temporary name made of those ids is one file,
    which the second's rename takes from the first, and the first's then fails (`FileNotFoundError`). Each writes its own
    temporary file, so both renames succeed, the later one's bytes stay, and no temporary file is left."""
    monkeypatch.setattr(A.os, "getpid", lambda: 1)
    monkeypatch.setattr(A.threading, "get_ident", lambda: 1)
    cache, key = tmp_path / "facts", "a" * 64
    first = {"schema": A.FACTS_SCHEMA, "toolchain": "t", "files": {}, "facts": {"x": 1}, "findings": {}}
    second = first | {"facts": {"x": 2}}
    replace, nested = os.replace, []

    def interleaved(src, dst):
        if not nested:
            nested.append(dst)
            A.store_cached_facts(cache, key, second)
        replace(src, dst)

    monkeypatch.setattr(A.os, "replace", interleaved)
    A.store_cached_facts(cache, key, first)
    assert nested == [cache / f"{key}.json.gz"], "the second writer ran inside the first's first rename"
    assert sorted(p.name for p in cache.iterdir()) == [f"{key}.json", f"{key}.json.gz"], "no temporary file is left"
    assert A.read_cached_facts(cache, key) == first, "the first writer's renames came last, and its pair is whole"

    replay, nested[:] = tmp_path / "replay", []
    replay.mkdir()
    monkeypatch.setattr(A.os, "replace", lambda src, dst: (nested or (nested.append(dst), A.store_replay(replay, key, "N")),
                                                           replace(src, dst)))
    A.store_replay(replay, key, "M")
    assert [p.name for p in replay.iterdir()] == [f"{key}.json"]
    assert json.loads((replay / f"{key}.json").read_text())["module"] == "M"


def test_a_facts_or_replay_cache_that_a_build_could_write_is_refused(tmp_path, capsys):
    """A facts sidecar's SHA-256 catches corruption and staleness, not a writer that rewrites the document and its sidecar
    consistently, and a replay entry is no harder to forge. So each cache must lie outside all a sandboxed build writes:
    every audited package's `.lake` and its path dependencies' (`writable`, `sandbox.sh`'s RW_DIRs). `build_roots_over`
    finds an overlap either way (the cache in a root, or a root in the cache) and through a symlink, and the CLI refuses
    such a cache before anything builds."""
    pkg, dep = tmp_path / "tree" / "P", tmp_path / "tree" / "D"
    (pkg / ".lake").mkdir(parents=True)
    dep.mkdir()
    (pkg / "lake-manifest.json").write_text(json.dumps({"packages": [{"name": "D", "type": "path", "dir": "../D"}]}))
    roots = [pkg.resolve() / ".lake", dep.resolve() / ".lake"]
    assert A.writable(pkg) == roots
    assert A.build_roots_over(tmp_path / "facts", [pkg]) == []
    assert A.build_roots_over(tmp_path / "tree" / "facts", [pkg]) == [], "beside the packages, as lean_changed keeps it"
    assert A.build_roots_over(pkg / ".lake" / "facts", [pkg]) == roots[:1]
    assert A.build_roots_over(dep / ".lake" / "x", [pkg]) == roots[1:], "a path dependency's .lake is written too"
    assert A.build_roots_over(tmp_path / "tree", [pkg]) == sorted(roots), "a cache holding a root"
    (tmp_path / "link").symlink_to(pkg / ".lake", target_is_directory=True)
    assert A.build_roots_over(tmp_path / "link" / "facts", [pkg]) == roots[:1], "resolved, as sandbox.sh binds"

    ctl = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", ctl, ignore=shutil.ignore_patterns(".lake"))
    for flag in ("--facts-cache", "--replay-cache"):
        with pytest.raises(SystemExit):
            A.main([str(ctl), flag, str(ctl / ".lake" / "cache"), "--out", str(tmp_path / "out")])
        assert f"{flag} {ctl / '.lake' / 'cache'} overlaps {(ctl / '.lake').resolve()}" in capsys.readouterr().err
    assert not (ctl / ".lake").exists(), "refused before anything built"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_facts_cache_stores_the_sealed_document_under_its_build_s_key_and_judges_as_usual(tmp_path, monkeypatch):
    """`--facts-cache` (`audit(facts_cache=, facts_reuse=True)`, what lean_changed's records step passes) judges as a plain
    audit does and also stores the sealed document with its sidecar under the build's `facts_key`; a second audit of the
    same tree recomputes that key and reads the document back as a hit. `--facts-only` (the graph's stage A) still writes
    the document alone, so its output stays the bytes the graph's cutoff compares, and the two flags exclude each other. A
    node whose builds aren't sandboxed stores nothing: any build there could have written the cache."""
    def noop(*_):
        return None
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    A.audit(pkg, tmp_path / "pop", build=False, replay=False, runs=False, update=True, log=noop)  # record so it passes
    plain = A.audit(pkg, tmp_path / "plain", build=False, replay=False, runs=False, log=noop)

    monkeypatch.setattr(A, "sandboxed", lambda: False)
    bare = tmp_path / "bare"
    d = A.audit(pkg, tmp_path / "d", build=False, replay=False, runs=False, facts_cache=bare, facts_reuse=True, log=noop)
    assert not bare.exists() and "facts_key" not in d and "sandbox" in d["facts_cache"], "no sandbox: nothing stored"
    assert d["failures"] == [], "and the audit judges as usual"

    monkeypatch.setattr(A, "sandboxed", lambda: True)
    cache = tmp_path / "facts"
    a = A.audit(pkg, tmp_path / "a", build=False, replay=False, runs=False, facts_cache=cache, facts_reuse=True, log=noop)
    key = a["facts_key"]
    assert list(map(str, a["failures"])) == list(map(str, plain["failures"])) == [], "it judges as a plain audit does"
    assert sorted(p.name for p in cache.iterdir()) == [f"{key}.json", f"{key}.json.gz"]
    b = A.audit(pkg, tmp_path / "b", build=False, replay=False, runs=False, facts_cache=cache, facts_reuse=True, log=noop)
    assert b["facts_key"] == key and A.read_cached_facts(cache, key) is not None, "the recomputed key is a hit"

    only = tmp_path / "only"
    c = A.audit(pkg, tmp_path / "c", build=False, replay=False, runs=False, facts_cache=only, log=noop)
    assert c["facts_key"] == key and [p.name for p in only.iterdir()] == [f"{key}.json.gz"], "--facts-only: no sidecar"
    assert (only / f"{key}.json.gz").read_bytes() == (cache / f"{key}.json.gz").read_bytes(), "one document, two caches"

    cli = tmp_path / "cli"
    assert A.main([str(pkg), "--facts-cache", str(cli), "--no-replay", "--no-runs", "--out", str(tmp_path / "cli-out")]) == 0
    (side,) = cli.glob("*.json")
    assert A.read_cached_facts(cli, side.stem) is not None, "the CLI stores a pair a reuse reads back"
    with pytest.raises(SystemExit):
        A.main([str(pkg), "--facts-cache", str(tmp_path / "x"), "--facts-only", str(tmp_path / "y")])


def test_the_facts_are_kept_only_gzipped_as_the_program_wrote_them(tmp_path, monkeypatch):
    """A recorded run publishes the work directory, where `Facts.lean`'s document is hundreds of MB for `security`: it is
    kept there only as `facts.json.gz`, its bytes gzipped with no timestamp (equal facts are equal bytes, which the store
    keeps once), and the facts the audit judges are that document's. A `.gz` an earlier run left is gone before it runs."""
    doc = {"pins": {"C.g": {"kind": "theorem", "reads": [{"name": "C.d"}]}}, "modules": ["C"], "note": "é"}
    written = json.dumps(doc, ensure_ascii=False).encode()

    def program(argv, cwd, log):
        assert argv[:3] == ["lake", "env", "facts"] and not (tmp_path / A.FACTS).exists()
        config = json.loads(Path(argv[3]).read_text())
        Path(config["out"]).write_bytes(written)
        Path(config["canon"]).write_text("".join(json.dumps(r) + "\n" for r in ({"id": "pin:C.g", "canon": "g"},
                                                                             {"id": "read:C.d", "canon": "d"})))
        return 0, ""

    monkeypatch.setattr(A, "facts_program", lambda pkg: "facts")
    monkeypatch.setattr(A, "run", program)
    (tmp_path / A.FACTS).write_bytes(b"stale")
    facts = A.facts_of(tmp_path, {"roots": ["C"]}, tmp_path)
    kept = tmp_path / A.FACTS
    assert sorted(p.name for p in tmp_path.iterdir()) == ["facts-config.json", "facts.json.gz"]
    assert gzip.decompress(kept.read_bytes()) == written and kept.read_bytes()[4:8] == bytes(4), "no timestamp"
    assert A.stored_facts(kept) == doc and facts["note"] == "é"
    assert facts["pins"]["C.g"]["type_sha256"] == hashlib.sha256(b"g").hexdigest()
    assert facts["pins"]["C.g"]["reads"][0]["sha256"] == hashlib.sha256(b"d").hexdigest()
    (tmp_path / "facts.json").write_bytes(written)
    assert A.stored_facts(tmp_path / "facts.json") == doc, "an older run's plain facts.json still reads"


def test_the_program_writes_the_facts_outside_the_audit_s_output_and_they_are_kept_only_when_asked(tmp_path, monkeypatch):
    """`Facts.lean` writes its document and canonical forms (hundreds of MB for `security`) in a directory of their own
    outside the work directory a recorded run publishes, and that directory is gone once they are read, or the program
    failed. Without ``keep`` the work directory keeps no document, and none an earlier run left there, gzipped or plain."""
    work, wrote, failing = tmp_path / "work", [], []

    def program(argv, cwd, log):
        config = json.loads(Path(argv[3]).read_text())
        out, canon = Path(config["out"]), Path(config["canon"])
        wrote.append(out.parent)
        if failing:
            return 1, "a root that was not built"
        out.write_text(json.dumps({"pins": {}, "modules": ["C"]}))
        canon.write_text("")
        return 0, ""

    monkeypatch.setattr(A, "facts_program", lambda pkg: "facts")
    monkeypatch.setattr(A, "run", program)
    work.mkdir()
    for stale in (A.FACTS, "facts.json", "canon.jsonl"):
        (work / stale).write_text("an earlier run's")
    assert A.facts_of(tmp_path, {"roots": ["C"]}, work, keep=False)["modules"] == ["C"]
    assert sorted(p.name for p in work.iterdir()) == ["facts-config.json"]
    failing.append(True)
    with pytest.raises(RuntimeError, match="Facts.lean"):
        A.facts_of(tmp_path, {"roots": ["C"]}, work)
    assert len(wrote) == 2 and not any(d.is_relative_to(work) or d.exists() for d in wrote)
    failing.clear()
    A.facts_of(tmp_path, {"roots": ["C"]}, work)
    assert len(wrote) == 3 and not wrote[-1].is_relative_to(work) and not wrote[-1].exists()
    assert sorted(p.name for p in work.iterdir()) == ["facts-config.json", A.FACTS]


@pytest.mark.parametrize("flags,kept", [([], {"a": "all", "b": "all"}), (["--facts", "named"], {"a": "own", "b": "none"}),
                                        (["--facts", "none"], {"a": "none", "b": "none"})])
def test_facts_says_which_audits_keep_their_facts_document(tmp_path, monkeypatch, flags, kept):
    """A check run keeps every audit's facts and its provers' (`status_export.py --facts` reads them); a records run's PKG
    (`--facts named`) keeps only its own (`trusted_size.py --run`), not those of the prover audited beside it; `none`
    keeps none."""
    a, b = tmp_path / "a", tmp_path / "b"
    for p in (a, b):
        p.mkdir()
    (a / A.POLICY).write_text(json.dumps({"proved_in": ["../b"]}))
    (b / A.POLICY).write_text("{}")
    got = {}

    def fake(pkg, out, *args, built=lambda: None, kept_facts="all", **_):
        built()
        got[pkg.name] = kept_facts
        return {"package": str(pkg), "seconds": {}, "failures": []}

    monkeypatch.setattr(A, "audit", fake)
    assert A.main([str(a), *flags, "--out", str(tmp_path / "out"), "--repo", str(tmp_path)]) == 0
    assert got == kept


@pytest.mark.parametrize("kept_facts,keeps", [("all", [True, True]), ("own", [True, False]), ("none", [False, False])])
def test_an_audit_keeps_the_package_s_and_its_provers_facts_as_kept_facts_says(tmp_path, monkeypatch, kept_facts, keeps):
    pkg = tmp_path / "s"
    pkg.mkdir()
    (pkg / A.POLICY).write_text(json.dumps({"roots": ["S"], "proved_in": ["../p"]}))
    (pkg / "lean-toolchain").write_text("leanprover/lean4:v4.34.0\n")
    seen = []

    def facts_of(q, policy, work, moved=None, keep=True, sealed=None):
        seen.append(keep)
        return {"modules": [], "oleans": [], "pins": {}}

    def proved(q, policy, facts, work, moved=None, ready=None, keep=True, seals=None, keys=False):
        seen.append(keep)
        raise RuntimeError("stop after the provers' facts")

    monkeypatch.setattr(A, "facts_of", facts_of)
    monkeypatch.setattr(A, "proved", proved)
    rep = A.audit(pkg, tmp_path / "out", replay=False, runs=False, log=lambda *_: None, kept_facts=kept_facts)
    assert seen == keeps and rep["failures"] == ["lean: stop after the provers' facts"]


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
    """`Facts.lean`'s shape on real statements: mentioning a session whose structure names the function (`hown`) is no
    acceptance, nor is an implication whose consequent isn't one (`_hop`); a premise under `∀` is read by its head; a
    family of structures (`Vs : Fin 3 → Stmt`, an `abbrev` of one) holds its fields; a structure's own `Nonempty` is a
    hypothesis too, since a field `Nonempty {x // Bad x}` is an existence claim. A probability's events are read under the
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
    def build():
        return subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    def audit(**kw):
        return A.audit(pkg, tmp_path / "out", update=True, replay=False, runs=False, log=lambda *_: None, **kw)
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


SPLIT_KEPT = "def pick : Nat → Nat\n  | 0 => 1\n  | n + 1 => n\ndef three : Fin 5 := ⟨3, by decide⟩\n"
SPLIT_OFF = "def pick' : Nat → Nat\n  | 0 => 1\n  | n + 1 => n\ndef three' : Fin 5 := ⟨3, by decide⟩\n"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_file_split_changes_no_definition_s_hash_though_lean_renames_the_helpers_it_shared(tmp_path):
    """Lean shares a matcher and a proof it abstracts out of a definition between the declarations of one module: `pick'`
    reads `pick.match_1` and `three'` reads `three._proof_1`, until a split moves them into a file of their own, where they
    get `pick'.match_1` and `three'._proof_1`.  A definition's hash holds those in place, not by name, so after the split
    every definition read has the hash it had, and the review says only that two moved."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    kept, off = pkg / "Control" / "Split.lean", pkg / "Control" / "SplitOff.lean"
    kept.write_text(f"namespace Control\n{SPLIT_KEPT}{SPLIT_OFF}end Control\n")
    off.write_text("namespace Control\nend Control\n")
    (pkg / "Control" / "SplitThms.lean").write_text(
        "import Control.Split\nimport Control.SplitOff\nnamespace Control\n"
        "theorem picked : pick' 0 = 1 := rfl\ntheorem third : three'.val = 3 := rfl\nend Control\n")
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.SplitThms\n")
    pol = unclaimed(A.load(pkg))
    names = ("Control.main_thm", "Control.picked", "Control.third")
    A.dump(pkg, pol | {"layers": pol["layers"] | {"Control.Split": {"imports": []}, "Control.SplitOff": {"imports": []}},
                       "guarantees": {n: {"owner": "@lean"} for n in names}})

    def built() -> tuple[dict, dict, dict]:
        subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
        rep = A.audit(pkg, tmp_path / "out", update=True, replay=False, runs=False, log=lambda *_: None)
        assert unshaped(rep) == []
        f = A.facts_of(pkg, {"roots": ["Control"], "guarantees": dict.fromkeys(names, {})}, tmp_path)
        own = {r["name"]: r["own"] for p in ("Control.picked", "Control.third") for r in f["pins"][p]["reads"]}
        return rep, A.load(pkg), own

    def hashes(rec: dict) -> dict:
        return {d: h for g in rec["reads"].values() for d, h in A.definitions(g).items()}

    _, before, own = built()
    assert own["Control.pick'"] == own["Control.three'"] == [], "both use the helpers of the declarations before them"
    kept.write_text(f"namespace Control\n{SPLIT_KEPT}end Control\n")
    off.write_text(f"namespace Control\n{SPLIT_OFF}end Control\n")
    rep, after, own = built()
    assert (own["Control.pick'"], own["Control.three'"]) == (["Control.pick'.match_1"], ["Control.three'._proof_1"])
    assert hashes(after) == hashes(before) and after["guarantees"] == before["guarantees"]
    assert list(after["reads"]["Control.SplitOff"]["definitions"]) == ["Control.pick'", "Control.three'"]
    assert rep["review"] == [f"reads: 2 {A.MOVED_READS}"]


PORT = """namespace Control
inductive K | a | b | c | d
private def p : K → Nat
  | .a => 1
  | _ => 0
def q : K → Nat
  | .a => 1
  | _ => 0
def Spec (k : K) : Prop := match k with
  | .a => q k = 1
  | _ => True
theorem picked : q .a = 1 := rfl
theorem spec (k : K) : Spec k := by cases k <;> trivial
end Control
"""


def ported(text: str) -> str:
    """``text`` written in Lean's module system, as the port writes it: its declarations public, the bodies its theorems
    unfold exposed."""
    for a, b in (("inductive", "public inductive"), ("def q", "@[expose] public def q"),
                 ("def Spec", "@[expose] public def Spec"), ("theorem", "public theorem")):
        text = text.replace(a, b)
    return "module\n\n" + text


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_port_to_the_module_system_leaves_the_lock_byte_identical_though_lean_renumbers_the_helpers(tmp_path):
    """A public declaration of a module never shares the matcher a private one made, so a port to the module system
    gives `q` its own `q.match_1` and `q._sparseCasesOn_1`, which the statement `Spec` (which `spec` is lifted into) then
    shares, where both read the private `p`'s before. Every text holds those helpers in place, a read definition's and a lifted
    statement's alike, so the port's record is byte for byte the one before it, and passes as it is."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    port = pkg / "Control" / "Port.lean"
    port.write_text(PORT)
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.Port\n")
    pol = unclaimed(A.load(pkg))
    names = ("Control.main_thm", "Control.picked", "Control.spec")
    A.dump(pkg, pol | {"layers": pol["layers"] | {"Control.Port": {"imports": []}},
                       "guarantees": {n: {"owner": "@lean"} for n in names}})

    def built() -> tuple[str, dict]:
        subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
        assert unshaped(A.audit(pkg, tmp_path / "out", update=True, replay=False, runs=False, log=lambda *_: None)) == []
        f = A.facts_of(pkg, {"roots": ["Control"], "guarantees": dict.fromkeys(names, {})}, tmp_path)
        own = {r["name"]: r["own"] for r in f["pins"]["Control.spec"]["reads"]}
        return (pkg / A.POLICY).read_text(), own

    before, own = built()
    assert own["Control.q"] == own["Control.Spec"] == [], "both share the private `p`'s matcher"
    port.write_text(ported(PORT))
    after, own = built()
    assert own["Control.q"] == ["Control.q.match_1"] and own["Control.Spec"] == []
    assert after == before
    assert unshaped(A.audit(pkg, tmp_path / "out", replay=False, runs=False, log=lambda *_: None)) == []


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_an_antecedent_in_a_conclusion_headed_by_the_package_s_definition_is_a_named_premise(tmp_path):
    """`Facts.lean`'s `premisesOf` on a real build: in a conclusion (`… ∧ ((∀ n, Opens n) → …)`), a hypothesis whose head is
    a definition of the package's spec (`Opens`, outside an assumptions module, as `TapOpens` is) is a premise the
    guarantee rests on, an open one. An update lists it under `open` only with the issue `open_seed` gives it, with the
    guarantee's owner and the digest its `reads` record holds, and the guarantee's record lists it under its `premises`'
    `open`, as `main_thm`'s lists the assumption it takes under `assumptions`; one headed by the toolchain's `Eq`
    (`m = 0 → …`) is what the conclusion means, and none."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    defs = pkg / "Control" / "Defs.lean"
    defs.write_text(defs.read_text().replace("end Control", "def Opens (n : Nat) : Prop := n + 0 = n\n\nend Control"))
    (pkg / "Control" / "Rests.lean").write_text(
        "import Control.Defs\nnamespace Control\n"
        "theorem rests (m : Nat) : double m = m + m ∧ ((∀ n, Opens n) → m = 0 → double m = 0) :=\n"
        "  ⟨by unfold double; omega, fun _ h => by subst h; rfl⟩\nend Control\n")
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.Rests\n")
    pol = unclaimed(A.load(pkg)) | {"guarantees": {n: {"owner": "@lean"} for n in ("Control.main_thm", "Control.rests")}}
    A.dump(pkg, pol)
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    assert [f.split(",")[0] for f in unshaped(A.audit(pkg, tmp_path / "out", update=True, replay=False, runs=False,
                                                     log=lambda *_: None))] == [
        "open: Control.Opens is a premise of Control.rests"]
    ticket = {"issue": "LEAN-1", "due": "2026-11-30"}
    A.dump(pkg, pol | {"open_seed": {"Control.Defs": ticket}})
    assert unshaped(A.audit(pkg, tmp_path / "out", update=True, replay=False, runs=False, log=lambda *_: None)) == []
    rec = A.load(pkg)
    assert {g: r["premises"] for g, r in rec["guarantees"].items()} == {
        "Control.main_thm": {"assumptions": ["Control.Hyp"], "domain": [], "open": [], "reads": []},
        "Control.rests": {"assumptions": [], "domain": [], "open": ["Control.Opens"], "reads": []}}
    assert "Control.Opens" not in rec["assumptions"] and "open_seed" not in rec
    assert rec["open"]["Control.Opens"] == {
        "owner": "@lean", "digest": rec["reads"]["Control.Defs"]["definitions"]["Control.Opens"]} | ticket
    f = A.facts_of(pkg, {"roots": ["Control"], "guarantees": dict.fromkeys(("Control.main_thm", "Control.rests"), {})},
                   tmp_path)
    assert [(p["head"], p.get("antecedent")) for p in f["pins"]["Control.rests"]["premises"]] == [("Control.Opens", True)]
    assert [(p["head"], p.get("antecedent")) for p in f["pins"]["Control.main_thm"]["premises"]] == [("Control.Hyp", None)]
    assert unshaped(A.audit(pkg, tmp_path / "out", replay=False, runs=False, log=lambda *_: None)) == []


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_an_honest_witness_concludes_a_premise_at_a_defeq_argument_and_not_at_another_instance(tmp_path):
    """`Facts.lean`'s `agree` on a real build: `fixed_sound` takes `Fixed (@Val.v Unit zeroA)`. An honest witness at
    `zeroB`, another instance with the same field, concludes it (`isDefEq`, where the texts differ), and so does one at
    every `n`; one at `one`, an argument that differs from the premise's only in its instance, and not definitionally,
    doesn't, though with the instances erased the two would read alike."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    (pkg / "Control" / "Agree.lean").write_text(
        "namespace Control\nclass Val (α : Type) where v : Nat\n"
        "instance zeroA : Val Unit := ⟨0⟩\ninstance zeroB : Val Unit := ⟨0⟩\ninstance one : Val Unit := ⟨1⟩\n"
        "def Fixed (n : Nat) : Prop := n ≤ n\n"
        "theorem fixed_sound (h : Fixed (@Val.v Unit zeroA)) : True := trivial\n"
        "theorem fixed_same : Fixed (@Val.v Unit zeroB) := Nat.le_refl _\n"
        "theorem fixed_other : Fixed (@Val.v Unit one) := Nat.le_refl _\n"
        "theorem fixed_all : ∀ n, Fixed n := Nat.le_refl\nend Control\n")
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.Agree\n")
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    names = ("Control.fixed_sound", "Control.fixed_same", "Control.fixed_other", "Control.fixed_all")

    def cover(witness: str) -> tuple:
        f = A.facts_of(pkg, {"roots": ["Control"], "guarantees": dict.fromkeys(names, {}),
                             "witnesses": {"Control.fixed_sound": {"honest": [witness]}}}, tmp_path)
        (p,) = f["pins"]["Control.fixed_sound"]["premises"]
        return p["head"], p["agree"], "Control.Fixed" in A.honest_concludes(f["pins"]["Control.fixed_sound"],
                                                                           [f["pins"][witness]])

    assert cover("Control.fixed_same") == ("Control.Fixed", True, True)
    assert cover("Control.fixed_all") == ("Control.Fixed", True, True)
    assert cover("Control.fixed_other") == ("Control.Fixed", False, False)


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
    def build():
        return subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    def audit():
        return A.audit(pkg, tmp_path / "out", update=True, replay=False, runs=False, log=lambda *_: None)
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
    def build():
        return subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    def audit(**kw):
        return A.audit(pkg, tmp_path / "out", replay=False, runs=False, log=lambda *_: None, **kw)
    build()
    assert audit(update=True)["failures"] == [
        f"instance: Control/Inst.lean:{line} declares an instance without a name, which Lean names Control.{n} from its type "
        f"(and, when nothing in the type is the project's own, its project's root): a move or a split renames it, and with it "
        f"every record whose definitions use it. Write `instance {n}` there (`check.py --name-instances` writes each one)"
        for line, n in ((6, "instInhabitedBox"), (7, "instToStringBox"), (9, "instInhabitedNat_control"),
                        (10, "instInhabitedFinOfNatNat_control"))]
    record = (pkg / A.POLICY).read_text()
    def facts():
        work = tmp_path / "out" / "audit-control"
        assert not (work / "facts.json").exists(), "the facts are kept only gzipped"
        return sorted(i["name"] for i in A.stored_facts(work / A.FACTS)["instances"])
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
    """A Security build with a `lean` per CPU is OOM-killed even in a 64 GB scope: Lake runs as many
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


def test_a_heavy_build_pipelines_its_replay_under_lean_audit_s_thread_setting(tmp_path, monkeypatch):
    """`lean_audit.py` runs every check.py with `LEAN_NUM_THREADS` set, under which `replay_threads` is None for every
    package, so a pipeline keyed on it would never run in a check. Heaviness is the manifest's (`pins_heavy`), so a heavy
    package's sandboxed build with a replay cache on a large machine pipelines whatever that variable says."""
    heavy, light = tmp_path / "heavy", tmp_path / "light"
    for p, deps in ((heavy, ["mathlib", "Arklib"]), (light, ["batteries"])):
        p.mkdir()
        (p / "lake-manifest.json").write_text(json.dumps({"packages": [{"name": d} for d in deps]}))
    cache = tmp_path / "replay"
    monkeypatch.setattr(A, "cores", lambda: A.PIPELINE_MIN_CORES)
    monkeypatch.setattr(A, "sandboxed", lambda: True)
    monkeypatch.setenv("LEAN_NUM_THREADS", str(A.PIPELINE_MIN_CORES))
    assert A.replay_threads(heavy) is None, "the setting that hid the pipeline"
    assert A.pins_heavy(heavy) and not A.pins_heavy(light)
    assert A.pipelined(heavy, True, True, cache, None)
    assert not A.pipelined(light, True, True, cache, None), "a light package's replay is short"
    assert not any(A.pipelined(heavy, *args) for args in ((False, True, cache, None), (True, False, cache, None),
                                                          (True, True, None, None), (True, True, cache, {"G": {}})))
    monkeypatch.setattr(A, "sandboxed", lambda: False)
    assert not A.pipelined(heavy, True, True, cache, None), "outside the sandbox the build could write the cache"
    monkeypatch.setattr(A, "sandboxed", lambda: True)
    monkeypatch.setattr(A, "cores", lambda: A.PIPELINE_MIN_CORES - 1)
    assert not A.pipelined(heavy, True, True, cache, None), "a build bound by its cores keeps them"


def test_a_spec_whose_guarantees_an_arklib_package_proves_needs_the_spec_floor(tmp_path):
    """A whole-Security replay holds about 120 GiB whatever its threads (a 128 GiB cap kills it at 16),
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
    """Lake 4.34 links a plugin twice when a module-system library and a classic one load it, as ArkLib's `ArkLib` and
    `ArkLibTest` load `ArkLibLintPlugin:shared`, and the two links of one file race, failing the build.
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
    def lake(*a):
        return subprocess.run(["lake", *a], cwd=root, capture_output=True, text=True, env=A.lake_env())
    (root / ".lake" / "packages").mkdir(parents=True)
    (root / ".lake" / "packages" / "Dep").symlink_to(dep)
    assert A.plugin_targets(root) == ["@Dep/Plug:shared"]
    assert lake("build", *A.plugin_targets(root)).returncode == 0
    out = lake("build")
    assert out.returncode == 0 and "Plug:shared" not in out.stdout, out.stdout


def test_update_refuses_a_shared_shipped_tree(tmp_path, capsys):
    pkg = tmp_path / "security"
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
    def build():
        return subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    def audit(**kw):
        return A.audit(pkg, tmp_path / "out", replay=False, runs=False, log=lambda *_: None, **kw)
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
    counted on one line that `statement_changes` leaves out, so the landing DM lists only real changes; a changed declaration
    among them is still listed."""
    enc = {"name": "P.Enc", "module": "P.Defs", "kind": "inductive", "hash": "1" * 32, "own": ["P.Enc.mk"]}
    mk = {"name": "P.Enc.mk", "module": "P.Defs", "kind": "constructor", "hash": "2" * 32, "own": []}
    cases = {"name": "P.Enc.casesOn", "module": "P.Defs", "kind": "def", "hash": "3" * 32, "own": []}
    g = {"name": "P.g", "module": "P.Defs", "kind": "def", "hash": "4" * 32, "own": []}
    hyp = {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "6" * 32, "own": []}
    pol = {**POLICY, "guarantees": {"P.t": {"owner": "@lean"}}}
    now = facts(pins={"P.t": fact(reads=[enc, mk, cases, g, hyp])})
    def sha_(*rs):
        return A.digest([{**r, "sha256": sha(r["hash"])} for r in rs])
    by_parent = {"P.Enc": sha_(enc, mk, cases), "P.g": sha_(g)}
    first = A.judge(pol, now, FILES, update=True)[1]
    old = first | {"reads": first["reads"] | {"P.Defs": {"digest": A.module_digest(by_parent), "definitions": by_parent}}}
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
    gone = facts(pins={"P.t": fact(reads=fact()["reads"][2:])})
    assert "definition P.f (P.Defs): gone" in A.review(old, A.judge(old, gone, FILES, update=True)[1], gone, {})
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
    assert fails == ["guarantees: P.t assumes (h : ∀ (m : Nat), m + 0 = m), a closed Prop that is not a named premise: a "
                     "Prop constant listed under `assumptions`, `domain` or `open`"]
    elsewhere = [{"binder": "h", "type": "Q.Hyp", "closed": True, "named": "Q.Hyp", "module": "Q.Stuff"}]
    assert A.judge(POLICY, facts(pins={"P.t": fact(hyps=elsewhere)}), FILES, update=True)[0][0].startswith("guarantees: P.t assumes (h : Q.Hyp)")


def test_a_premise_the_lock_does_not_list_fails():
    """A statement that takes an assumption applied to its own variables (`h : Premises.Hyp n`, an open hypothesis the
    hypotheses rule never names; `Facts.lean`'s `premises` sees through the abbrev and the `Guarantees` definitions to
    `P.Hyp`) fails while the lock doesn't list it, and an update adds it with the guarantee's owner, so a merge that drops
    a premise's entry fails. Without an import graph to say
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
        "with its guarantees' owner and commit the record (the landing message shows it), or write \"P.Hyp\": "
        f"{{\"owner\": \"@lean\", \"digest\": \"{digest}\"}} under `assumptions`."]


def test_a_premise_defined_anywhere_the_repository_builds_is_an_open_premise_and_a_dependencys_is_none():
    """A premise is listed wherever the repository defines it, not only in an assumptions module (`ProofLayer.SoundAt`, in
    `Definitions/Accounting/TS`): a head in any module whose `.olean` a package of the repository builds
    is listed, in a lifted guarantee too (its premises are read through the `Guarantees` definition); one from the
    toolchain or a fetched dependency (`Nat.le`, Mathlib's) is not. Declared outside `assumptions_module`, it is an open
    premise: an update lists it under `open` only with the issue `open_seed` gives it (and consumes the seed), and the
    lock that lists it passes; one the lock doesn't list fails, update or not, naming the entry to write."""
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
    unlisted = ("open: P.Sound is a premise of P.t, declared in P.Defs rather than P.Assumptions: an open premise the lock "
                "doesn't list. Discharge it (prove it, or drop it from the statements); or, if it is a real assumption, "
                "move its definition into P.Assumptions; or list it, writing \"P.Sound\": {\"owner\": \"@lean\", \"issue\": "
                "\"TEAM-N\", \"due\": \"YYYY-MM-DD\"} under `open` with the issue that discharges it, or with a one-line "
                "`why` in place of the issue and due date under `domain` if it is part of what the guarantee says")
    bare = {**POLICY, "assumptions": {}}
    assert A.judge(bare, f, FILES, update=True)[0] == [unlisted] and unlisted in A.judge(bare, f, FILES)[0]
    seed = {"P.Defs": {"issue": "LEAN-1", "due": "2026-11-30"}, "@lean": {"issue": "LEAN-2", "due": "2026-12-01"}}
    fails, pol = A.judge({**bare, "open_seed": seed}, f, FILES, update=True)
    assert fails == [] and pol["assumptions"] == {} and "open_seed" not in pol
    assert pol["open"] == {"P.Sound": {"owner": "@lean", "issue": "LEAN-1", "due": "2026-11-30"}}
    assert pol["guarantees"]["P.t"]["premises"] == {"assumptions": [], "domain": [], "open": ["P.Sound"], "reads": ["P.Hyp"]}
    assert A.judge(pol, f, FILES)[0] == [] and A.judge(pol, f, FILES, update=True)[1]["open"] == pol["open"]
    assert A.judge({**bare, "open_seed": {"@lean": seed["@lean"]}}, f, FILES, update=True)[1]["open"]["P.Sound"][
        "issue"] == "LEAN-2", "a seed's module prefix comes before its owner"
    assert [x.code for x in A.judge({**pol, "open_seed": seed}, f, FILES)[0]] == ["open.seed"]
    assert [x.code for x in A.judge({**bare, "open_seed": {"P": {"issue": "lean-1"}}}, f, FILES, update=True)[0]] == \
        ["open.seed-form", "open.unlisted"]
    assert [x.code for x in A.judge({**pol, "open": {"P.Sound": pol["open"]["P.Sound"] | {"due": "soon"}}}, f, FILES)[0]] \
        == ["open.form"]


#: an `open` entry as a person writes it for a `Prop` a statement holds without taking it
ON_READ = {"owner": "@lean", "issue": "LEAN-1", "due": "2026-11-30", "why": "P.t holds it as a conjunct of its event"}
TWO = {**POLICY, "guarantees": {"P.t": {"owner": "@lean"}, "P.u": {"owner": "@lean"}}}


def holding(module: str = "P.Net.Assumptions", h: str = "5") -> dict:
    """Facts of a package whose `P.t` takes `P.Hyp` and reads `P.Held`, a `Prop` of ``module`` that no statement takes
    (a conjunct of `P.t`'s event), beside `P.u`, which reads nothing."""
    held = {"name": "P.Held", "module": module, "kind": "def", "hash": h * 32}
    return facts(pins={"P.t": fact(reads=fact()["reads"] + [held]), "P.u": fact(signature="P.u : True", hyps=[], reads=[])})


def taking(now: dict, *names: str, module: str = "P.Net.Assumptions") -> dict:
    """``now`` (`holding`'s facts) with each guarantee of ``names`` (`P.u` alone, by default) also taking `P.Held`, a `Prop`
    of ``module``, as a premise."""
    pins = {n: p | {"premises": [*(p.get("premises") or []), {"binder": "s", "head": "P.Held", "module": module}]}
            if n in (names or ("P.u",)) else p for n, p in now["pins"].items()}
    return {**now, "pins": pins, "graph": {"P.Defs": {"imports": ["Init"], "olean": "/r/P/.lake/build/lib/lean/P/Defs.olean"}}}


def test_an_open_entry_that_gives_a_why_lists_a_prop_a_statement_reads_and_none_takes():
    """A `Prop` a statement only reads is listed in none, and an update lists none.  An `open` entry a person writes for it
    with a one-line `why` stays: its digest is its `reads` record's, a changed `Prop` fails under its name, a lock run
    leaves the entry as it is, and each guarantee that reads the `Prop` (wherever it is declared), and no other, lists it
    under `premises.open` in place of `premises.reads`."""
    none = {"assumptions": [], "domain": [], "open": [], "reads": []}
    for module in ("P.Net.Assumptions", "P.Defs"):
        now = holding(module)
        fails, plain = A.judge(TWO, now, FILES, update=True)
        assert fails == [] and "open" not in plain, "an update writes no such entry"
        assert plain["guarantees"]["P.t"]["premises"] == none | {
            "assumptions": ["P.Hyp"], "reads": ["P.Held"] if module == "P.Net.Assumptions" else []}
        written = plain | {"open": {"P.Held": ON_READ}}
        fails = A.judge(written, now, FILES)[0]
        assert [(x.code, x.entry) for x in fails] == [("open.unrecorded", "P.Held"), ("guarantees.premises-changed", "P.t")]
        pol = A.regenerated(written, now, FILES, fails)
        digest = pol["reads"][module]["definitions"]["P.Held"]
        assert pol["open"] == {"P.Held": {"owner": "@lean", "digest": digest} | {k: ON_READ[k] for k in ("issue", "due", "why")}}
        assert list(pol["open"]["P.Held"]) == ["owner", "digest", "issue", "due", "why"]
        assert pol["guarantees"]["P.t"]["premises"] == none | {"assumptions": ["P.Hyp"], "open": ["P.Held"]}
        assert pol["guarantees"]["P.u"]["premises"] == none
        assert A.judge(pol, now, FILES) == ([], pol)
        assert A.judge(pol, now, FILES, update=True) == ([], pol) and A.repaired(pol, now, FILES, [], TICKET) == pol, \
            "a lock run leaves it in place"
        fails = A.judge(pol, holding(module, "6"), FILES)[0]
        assert "open.digest" in {x.code for x in fails} and all(map(A.RECORD_DIFFERENCE.match, fails))
        new = A.regenerated(pol, holding(module, "6"), FILES, fails)
        assert new["open"]["P.Held"] == pol["open"]["P.Held"] | {"digest": new["reads"][module]["definitions"]["P.Held"]}
        assert new["open"]["P.Held"]["digest"] != digest


def test_an_open_entry_that_rests_on_a_read_gives_its_issue_its_due_date_its_owner_and_one_line_of_why():
    """Such an entry is a person's: it owes the issue that discharges it and its due date, both, an owner, and a `why` of
    one line, whether or not a statement takes its `Prop`, and each failure takes a person.  On an unowned lock it is its
    digest and its `why`, and the run that owns the lock holds it to the ticket."""
    now = holding()
    pol = A.judge(TWO | {"open": {"P.Held": ON_READ}}, now, FILES, update=True)[1]
    entry = pol["open"]["P.Held"]
    for facts_ in (now, taking(now)):
        for bad in ({k: v for k, v in entry.items() if k != "issue"}, {k: v for k, v in entry.items() if k != "due"},
                    entry | {"issue": "lean-1"}, entry | {"due": "soon"}, entry | {"why": ""}, entry | {"why": "two\nlines"},
                    entry | {"why": True}):
            fails = A.judge(pol | {"open": {"P.Held": bad}}, facts_, FILES)[0]
            assert [(x.code, x.entry) for x in fails if x.code.startswith("open.")] == [("open.form", "P.Held")], bad
            form = next(x for x in fails if x.code == "open.form")
            assert form == ("open: P.Held's entry, which gives a `why`, gives the `issue` that discharges it as a Linear "
                            "key (TEAM-N), its `due` as a date (YYYY-MM-DD) and its `why` as one line naming a statement "
                            "that holds it without taking it")
            assert A.regenerated(pol | {"open": {"P.Held": bad}}, facts_, FILES, fails) is None and not A.BOOKKEEPING.match(form)
    ownerless = pol | {"open": {"P.Held": {k: v for k, v in entry.items() if k != "owner"}}}
    assert [x.code for x in A.judge(ownerless, now, FILES)[0]] == ["open.no-owner"]
    free = laneless(pol)
    assert A.unowned_lock(free) and free["open"]["P.Held"] == {"digest": entry["digest"], "why": entry["why"]}
    assert A.judge(free, now, FILES) == ([], free) and A.judge(free, now, FILES, update=True) == ([], free)
    assert [x.code for x in A.judge(free, now, FILES, update=True, owner="@proofs")[0]] == ["open.form"]


def test_a_why_on_an_entry_whose_prop_no_guarantee_reads_without_taking_it_fails_until_a_person_removes_it():
    """The first wrong case: an `open` entry says the guarantees that read its `Prop` without taking it rest on it, and
    there is none, since each guarantee that reads the `Prop` takes it.  The `why` excepts nothing: it fails, naming the
    fix, and neither an update nor a lock run settles it or touches the `why`."""
    now = sounded()
    lock = opened(now)
    reads_it = {"name": "P.Sound", "module": "P.Defs", "kind": "def", "hash": "5" * 32, "sha256": sha("5" * 32)}
    both = {**now, "pins": {"P.t": now["pins"]["P.t"] | {"reads": now["pins"]["P.t"]["reads"] + [reads_it]}}}
    for facts_, lock in ((now, lock), (both, opened(both))):
        assert A.judge(lock, facts_, FILES)[0] == []
        wrong = {**lock, "open": {"P.Sound": lock["open"]["P.Sound"] | {"why": ON_READ["why"]}}}
        fails = A.judge(wrong, facts_, FILES)[0]
        assert [(x.code, x.entry) for x in fails] == [("open.why-stale", "P.Sound")]
        assert fails[0] == ("open: P.Sound's entry gives a `why`, which says that the guarantees reading P.Sound without "
                            "taking it rest on it, and none does: it is a premise of P.t; remove the `why`")
        assert not A.RECORD_DIFFERENCE.match(fails[0]) and not A.BOOKKEEPING.match(fails[0])
        assert A.regenerated(wrong, facts_, FILES, fails) is None
        again, written = A.judge(wrong, facts_, FILES, update=True)
        assert [x.code for x in again] == ["open.why-stale"] and written["open"] == wrong["open"]
        assert A.repaired(wrong, facts_, FILES, fails, TICKET)["open"] == wrong["open"]


def test_once_a_statement_takes_a_prop_whose_entry_rests_on_a_read_its_readers_still_list_it_as_open():
    """A `Prop` an `open` entry holds on a read, which a statement then takes as a premise.  The `why` stays, and so does
    the record of each guarantee that only reads the `Prop`: it still lists it under `premises.open`, as the one that
    takes it now does, so the one record that changes is the taker's.  Once the last guarantee that only read it takes it
    too, the `why` fails until a person removes it, and the entry stays `open` without it."""
    none = {"assumptions": [], "domain": [], "open": [], "reads": []}
    for module in ("P.Net.Assumptions", "P.Defs"):
        now = holding(module)
        pol = A.judge(TWO | {"open": {"P.Held": ON_READ}}, now, FILES, update=True)[1]
        then = taking(now, module=module)
        fails = A.judge(pol, then, FILES)[0]
        assert [(x.code, x.entry) for x in fails] == [("guarantees.premises-changed", "P.u")]
        new = A.regenerated(pol, then, FILES, fails)
        assert new["open"] == pol["open"] and new["guarantees"]["P.t"] == pol["guarantees"]["P.t"]
        assert new["guarantees"]["P.t"]["premises"] == none | {"assumptions": ["P.Hyp"], "open": ["P.Held"]}
        assert new["guarantees"]["P.u"]["premises"] == none | {"open": ["P.Held"]}
        assert A.judge(new, then, FILES) == ([], new) and A.judge(new, then, FILES, update=True) == ([], new)
        assert A.repaired(new, then, FILES, [], TICKET) == new
        last = taking(then, "P.t", module=module)
        fails = A.judge(new, last, FILES)[0]
        assert fails == ["open: P.Held's entry gives a `why`, which says that the guarantees reading P.Held without taking "
                         "it rest on it, and none does: it is a premise of P.t, P.u; remove the `why`"]
        assert fails[0].code == "open.why-stale" and A.regenerated(new, last, FILES, fails) is None
        fixed = new | {"open": {"P.Held": {k: v for k, v in new["open"]["P.Held"].items() if k != "why"}}}
        assert A.judge(fixed, last, FILES) == ([], fixed), "the `why` removed"


def test_an_entry_without_a_why_for_a_prop_no_statement_takes_fails_and_an_update_drops_it():
    """The second wrong case: a `Prop` a statement only reads, listed with no `why`.  It fails as before, a record
    difference an update settles by dropping the entry, and its message names both fixes, under `domain` and
    `assumptions` too."""
    now = holding()
    pol = A.judge(TWO | {"open": {"P.Held": ON_READ}}, now, FILES, update=True)[1]
    entry = {k: v for k, v in pol["open"]["P.Held"].items() if k != "why"}
    fails = A.judge(pol | {"open": {"P.Held": entry}}, now, FILES)[0]
    assert [(x.code, x.entry) for x in fails] == [("open.read-only", "P.Held"), ("guarantees.premises-changed", "P.t")]
    fix = ("P.Held is read by a statement, but none takes it as a premise: its `reads` record holds it; remove the entry, "
           "or, if a statement holds it without taking it (a conjunct of its event) and nothing proves it yet, ")
    assert fails[0] == f"open: {fix}give its entry a one-line `why` naming that statement"
    new = A.regenerated(pol | {"open": {"P.Held": entry}}, now, FILES, fails)
    assert new["open"] == {} and new["guarantees"]["P.t"]["premises"]["reads"] == ["P.Held"]
    assert A.judge(new, now, FILES)[0] == []
    scope = {k: entry[k] for k in ("owner", "digest")} | {"why": "the runs the bound is over"}
    for section, listed in (("domain", scope), ("assumptions", {k: entry[k] for k in ("owner", "digest")})):
        elsewhere = {k: v for k, v in new.items() if k != "open"} | {section: new.get(section, {}) | {"P.Held": listed}}
        fails = A.judge(elsewhere, now, FILES)[0]
        assert fails == [f"{section}: {fix}list it under `open` with the issue that discharges it, its due date and a "
                         f"one-line `why` naming that statement"]


def test_a_deployed_guarantee_that_reads_an_open_premise_resting_on_a_read_has_it_in_its_gap():
    """What a guarantee's record lists under `premises.open` is in its trust gap, so a deployed guarantee that holds such
    a `Prop` fails until a person lists it in its `trust_gap` entry, as for a premise it takes."""
    now = holding()
    pol = A.judge(TWO | {"open": {"P.Held": ON_READ}}, now, FILES, update=True)[1]
    lock = {**pol, "trust": {"base": HYP}, "deployed": ["P.t"]}
    fails = A.judge(lock, now, FILES)[0]
    assert [(x.code, x.entry) for x in fails] == [("trust.gap-new", "P.t")] and "rests on P.Held, outside" in fails[0]
    assert A.judge({**lock, "trust_gap": {"P.t": ["P.Held"]}}, now, FILES)[0] == []
    then = taking(now)
    taken = A.judge({**lock, "trust_gap": {"P.t": ["P.Held"]}}, then, FILES, update=True)[1]
    assert A.judge(taken, then, FILES)[0] == [] and taken["trust_gap"] == {"P.t": ["P.Held"]}
    assert [(x.code, x.entry) for x in A.judge({**taken, "trust_gap": {}}, then, FILES)[0]] == [("trust.gap-new", "P.t")]


def test_an_open_premise_a_merge_also_kept_under_assumptions_is_a_record_difference_the_update_drops():
    """A lock merged by a base whose `merge.py` keeps an `assumptions` entry one side changed beside the `open` entry
    the other side moved it to lists the premise twice. Declared outside `assumptions_module`, it is no named
    assumption, so the check names one failure, `assumptions.moved`, a record difference, and the update keeps the
    `open` entry alone."""
    pin = fact(signature="P.t (n : Nat) : P.Guarantees.t n", hyps=[])
    pin["premises"] = [{"binder": "h", "head": "P.Sound", "module": "P.Defs"}]
    f = facts(pins={"P.t": pin}, graph={"P.Defs": {"imports": ["Init"],
                                                    "olean": "/r/Security/.lake/build/lib/lean/P/Defs.olean"}})
    ticket = {"P.Defs": {"issue": "LEAN-1", "due": "2026-11-30"}}
    pol = A.judge({**POLICY, "assumptions": {}, "open_seed": ticket}, f, FILES, update=True)[1]
    merged = {**pol, "assumptions": {"P.Sound": {"owner": "@lean", "digest": "0" * 64}}}
    fails = A.judge(merged, f, FILES)[0]
    assert [x.code for x in fails] == ["assumptions.moved"] and A.RECORD_DIFFERENCE.match(fails[0])
    new = A.regenerated(merged, f, FILES, fails)
    assert new["assumptions"] == {} and new["open"] == pol["open"] and A.judge(new, f, FILES)[0] == []


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
    """A premise that only packs listed assumptions (`def Both n := Hyp ∧ Hyp2 n`, or through another such definition) is
    no new named assumption: the check holds its parts to the lock in its place (so a listed part reached only through it is read), an
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


def test_a_guarantee_s_record_lists_the_named_assumptions_it_rests_on():
    """Each guarantee's record lists its `premises` by kind: the named assumptions and the open premises its statement
    rests on (`assumption_records`), an alias of listed ones as those, and the Props it only `reads`, each sorted and
    empty for a statement that takes none. A record whose lists differ from the build's, or that holds none (written
    before records held them), fails as a record difference, which `--update` settles."""
    listed = {**POLICY, "assumptions": {"P.Hyp": {"owner": "@lean"}, "P.Hyp2": {"owner": "@lean"}},
              "guarantees": {"P.t": {"owner": "@lean"}, "P.u": {"owner": "@lean"}}}
    f = aliased({"P.Both": [asserted("P.Inner", "P.Defs"), asserted("P.Hyp")], "P.Inner": [asserted("P.Hyp2")]})
    f["pins"]["P.u"] = fact(signature="P.u : True", hyps=[], reads=[])
    fails, pol = A.judge(listed, f, FILES, update=True)
    assert fails == [] and pol["guarantees"]["P.t"]["premises"] == {"assumptions": ["P.Hyp", "P.Hyp2"], "domain": [], "open": [],
                                                                    "reads": []}
    assert pol["guarantees"]["P.u"]["premises"] == {"assumptions": [], "domain": [], "open": [], "reads": []}
    assert list(pol["guarantees"]["P.t"]) == ["owner", "signature", "assumptions", "premises", "type_hash"]
    assert A.judge(pol, f, FILES)[0] == []
    empty = {"assumptions": [], "domain": [], "open": [], "reads": []}
    for was, said in ((empty | {"assumptions": ["P.Hyp"]}, "P.Hyp"), (empty, "none"), (["P.Hyp"], "none"), (None, "none")):
        rec = {k: v for k, v in pol["guarantees"]["P.t"].items() if k != "premises"} | ({} if was is None else {"premises": was})
        fails = A.judge(pol | {"guarantees": pol["guarantees"] | {"P.t": rec}}, f, FILES)[0]
        assert fails == [f"guarantees: P.t's premises changed (assumptions {said} -> P.Hyp, P.Hyp2); run `check.py "
                         f"--update` and commit the record"]
        assert A.RECORD_DIFFERENCE.match(fails[0])


def test_an_alias_hiding_anything_but_listed_assumptions_is_a_new_one_and_its_failure_names_the_entry():
    """Fail-closed: a premise that unfolds to anything but listed assumptions (one the lock doesn't list, `False`, a bound
    predicate, a constant applied to a proposition or a predicate, a part that doesn't unfold) stays a new named assumption,
    taken open or closed; its failure says what it unfolds to and the entry to write, so one edit settles it. Declared
    outside `assumptions_module`, it is an open premise, which an update lists only under `open_seed`."""
    listed = {**POLICY, "assumptions": {"P.Hyp": {"owner": "@lean"}}}
    pol = A.judge(listed, aliased({"P.Both": [asserted("P.Hyp")]}), FILES, update=True)[1]
    ticket = {"issue": "LEAN-1", "due": "2026-11-30"}
    cases = {"P.Hidden": [asserted("P.Hyp"), asserted("P.Hidden")],
             "False": [asserted("P.Hyp"), asserted("False", "Init.Prelude")],
             "a bound predicate": [asserted("P.Hyp"), asserted(None, None)],
             "P.Hyp applied to a proposition or predicate": [asserted("P.Hyp", props=True)],
             "P.Inner": [asserted("P.Hyp"), asserted("P.Inner", "P.Defs")]}
    for why, parts in cases.items():
        got = aliased({"P.Both": parts})
        fails = [f for f in A.judge(pol, got, FILES)[0] if f.code == "open.unlisted"]
        beside = " beside the listed P.Hyp" if why != "P.Hyp applied to a proposition or predicate" else ""
        assert fails == [
            "open: P.Both is a premise of P.t, declared in P.Defs rather than P.Assumptions: an open premise the lock "
            "doesn't list. Discharge it (prove it, or drop it from the statements); or, if it is a real assumption, move its "
            "definition into P.Assumptions; or list it, writing \"P.Both\": {\"owner\": \"@lean\", \"issue\": \"TEAM-N\", "
            f"\"due\": \"YYYY-MM-DD\"}} under `open` with the issue that discharges it, or with a one-line `why` in place "
            f"of the issue and due date under `domain` if it is part of what the guarantee says. It unfolds to "
            f"{why}{beside}, so it is no alias of listed premises, which the lock would not count."], (why, fails)
        assert "P.Both" in A.judge({**listed, "open_seed": {"P.Defs": ticket}}, got, FILES, update=True)[1]["open"]
        assert A.judge({**pol, "open": {"P.Both": {"owner": "@lean"} | ticket}}, got, FILES)[0] == [
            "guarantees: P.t's premises changed (open none -> P.Both); run `check.py --update` and commit the record"]
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
    read as `And` and dropped. An update lists `Hidden`, which makes `Sneaky` an alias, and lists the open `Absurd` and
    `Rec` only under `open_seed`; the record of `t_closed` names `A` and `B`."""
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
    def new(fails):
        return sorted(x.split(" ")[1] for x in fails if x.code in ("assumptions.unlisted-premise", "open.unlisted",
                                                                  "guarantees.unnamed-hypothesis"))
    assert new(A.judge(policy, f, files)[0]) == ["Control.Absurd", "Control.Hidden", "Control.Rec", "Control.Sneaky"]
    assert new(A.judge(policy, f, files, update=True)[0]) == ["Control.Absurd", "Control.Rec"], "open: only with an issue"
    fails, pol = A.judge(policy | {"open_seed": {"Control.Aliases": {"issue": "LEAN-1", "due": "2026-11-30"}}}, f, files,
                         update=True)
    assert new(fails) == [] and sorted(set(pol["assumptions"]) - set(policy["assumptions"])) == ["Control.Hidden"]
    assert sorted(pol["open"]) == ["Control.Absurd", "Control.Rec"]
    assert pol["guarantees"]["Control.t_closed"]["assumptions"] == ["Control.A", "Control.B"]
    assert new(A.judge(pol, A.facts_of(pkg, pol, tmp_path), files)[0]) == []


def test_a_premise_no_reads_record_holds_is_digested_with_what_it_reads():
    """A premise whose module is outside `meaning` (a Proofs module) has no `reads` record, so an entry of an owner alone
    would let a strengthened premise change no record. Its digest is its walk's
    (`Facts.lean`'s `assumption_reads`: it and what it reads): a change to either fails under its name, and an update
    writes it, which the review prints."""
    pin = fact(signature="P.t (n : Nat) (h : P.Leaves n) : P.f n = n", hyps=[])
    pin["premises"] = [{"binder": "h", "head": "P.Leaves", "module": "P.Thms"}]
    graph = {"P.Thms": {"imports": ["Init"], "olean": "/r/Security/Proofs/.lake/build/lib/lean/P/Thms.olean"}}
    def walk(bound):
        return {"P.Leaves": [{"name": "P.Leaves", "module": "P.Thms", "kind": "def", "sha256": sha("leaves")},
                             {"name": "P.bound", "module": "P.Thms", "kind": "def", "sha256": sha(bound)}]}
    f = facts(pins={"P.t": pin}, graph=graph, assumption_reads=walk("n + 1"))
    ticket = {"issue": "LEAN-1", "due": "2026-11-30"}
    fails, pol = A.judge({**POLICY, "assumptions": {}, "open_seed": {"@lean": ticket}}, f, FILES, update=True)
    assert fails == [] and pol["open"] == {"P.Leaves": {"owner": "@lean", "digest": A.digest(walk("n + 1")["P.Leaves"])} | ticket}
    assert A.judge(pol, f, FILES)[0] == [] and A.judge(pol, f, FILES, update=True)[1]["open"] == pol["open"]
    moved = facts(pins={"P.t": pin}, graph=graph, assumption_reads=walk("n + 2"))
    assert A.judge(pol, moved, FILES)[0] == [
        "open: P.Leaves's definition in P.Thms, or a definition it reads, changed; run `check.py --update` and commit "
        "the record (the landing message shows the change)"]
    new = A.judge(pol, moved, FILES, update=True)[1]
    assert A.assumption_changes(pol, new) == ["open premise P.Leaves: changed (its statement or a definition it reads)"]
    bare = {**pol, "open": {"P.Leaves": {"owner": "@lean"} | ticket}}
    assert A.judge(bare, f, FILES)[0][0].startswith("open: P.Leaves has no record")
    assert A.assumption_changes(bare, pol) == ["open premise P.Leaves: recorded: its statement and the definitions it "
                                               "reads, digested"]
    assert A.judge(bare, facts(pins={"P.t": pin}, graph=graph), FILES)[0] == [], "older facts, without the walks"


LEAVES = "import Control.LeafBound\nnamespace Control\ndef Leaves (n : Nat) : Prop := n < leafBound n\nend Control\n"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_premise_outside_meaning_changes_its_record_when_it_or_what_it_reads_changes(tmp_path):
    """On a real build: a premise defined outside `meaning` (`Control.Leaves`, as `RecDeploy.PlainLeavesAdmitted`
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
                       "open_seed": {"Control.Leaves": {"issue": "LEAN-1", "due": "2026-11-30"}},
                       "reads_exempt": {"Control.Leaves": "the premise outside `meaning` this test digests"},
                       "guarantees": pol["guarantees"] | {"Control.leaf": {"owner": "@lean"}}})
    def build():
        return subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    def audit(update):
        return A.audit(pkg, tmp_path / "out", update=update, replay=False, runs=False, log=lambda *_: None)
    build()
    assert unshaped(audit(True)) == []
    first = (pkg / A.POLICY).read_text()
    assert set(json.loads(first)["assumptions"]) == {"Control.Hyp"}
    rec = json.loads(first)["open"]
    assert set(rec) == {"Control.Leaves"} and rec["Control.Leaves"]["owner"] == "@lean"
    assert len(rec["Control.Leaves"]["digest"]) == 64 and "Control.Leaves" not in json.dumps(json.loads(first)["reads"])
    build()
    assert unshaped(audit(False)) == [] and unshaped(audit(True)) == [] and (pkg / A.POLICY).read_text() == first
    seen = {rec["Control.Leaves"]["digest"]}
    for path, text in ((bound, "namespace Control\ndef leafBound (n : Nat) : Nat := n + 2\nend Control\n"),
                       (leaves, LEAVES.replace("n < leafBound n", "n + 1 < leafBound n"))):
        before = json.loads((pkg / A.POLICY).read_text())
        path.write_text(text)
        build()
        assert [f.code for f in audit(False)["failures"] if not f.code.startswith("shape")] == ["open.digest"]
        rep = audit(True)
        after = json.loads((pkg / A.POLICY).read_text())
        assert unshaped(rep) == [] and A.statement_changes(rep["review"]) == [
            "open premise Control.Leaves: changed (its statement or a definition it reads)"]
        assert after["assumptions"]["Control.Hyp"] == before["assumptions"]["Control.Hyp"]
        assert after["open"]["Control.Leaves"]["digest"] not in seen
        seen.add(after["open"]["Control.Leaves"]["digest"])
        after["open"]["Control.Leaves"] = before["open"]["Control.Leaves"]
        assert after == before, "the premise's record changed, and nothing else"


def excepted_toy(tmp_path, module: str, source: str, proof: str, guarantee: str, defs: str = "", claims_not=None):
    """The control package with the proofs-tree module ``module`` (``source``) under `trees`' `except` and outside
    `meaning`, ``proof`` (a theorem of its statement, in namespace `Control`) in a module of its own, that theorem listed
    as the guarantee ``guarantee``, and ``defs`` added to `Control.Defs`: the package, its build, its audit (``update`` or
    not, under ``moved``) and its lock."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    if defs:
        d = pkg / "Control" / "Defs.lean"
        d.write_text(d.read_text().replace("end Control", f"{defs}\nend Control"))
    leaf = module.split(".")[-1]
    (pkg / "Control" / f"{leaf}.lean").write_text(source)
    (pkg / "Control" / f"{leaf}Thm.lean").write_text(f"import {module}\nnamespace Control\n{proof}end Control\n")
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + f"import {module}Thm\n")
    pol = A.load(pkg)
    A.dump(pkg, pol | {"meaning": ["Control.Assumptions", "Control.Defs", "Control.Thms"],
                       "trees": pol["trees"] | {"proofs": [*pol["trees"]["proofs"], module],
                                                "except": {module: "the statement this test records, in a proof module"}},
                       "claims": pol["claims"] | ({"not": claims_not} if claims_not else {}),
                       "guarantees": pol["guarantees"] | {guarantee: {"owner": "@lean"}}})

    def build():
        return subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())

    def audit(update, moved=None):
        return A.audit(pkg, tmp_path / "out", update=update, replay=False, runs=False, log=lambda *_: None, moved=moved)

    def lock():
        return json.loads((pkg / A.POLICY).read_text())
    return pkg, build, audit, lock


LOOP = ("import Control.Assumptions\nimport Control.Defs\nnamespace Control.Guarantees\n"
        "def Loop : Prop := ∀ n : Nat, Control.Hyp → Control.twice n\nend Control.Guarantees\n")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_premise_of_a_statement_in_an_excepted_module_changes_its_record(tmp_path):
    """On a real build: a guarantee whose statement is a `def G : Prop` of a proofs-tree module under `trees`' `except`
    and outside `meaning` (`Control.Guarantees.Loop`, as Flock's `SessionLoopZKDeployed` is in
    `Proofs.Flock.SessionLoop`) is recorded through `G`'s body. A premise added to it that no named assumption stands for
    (`n ≠ 5`), or dropped from it while another guarantee still takes it (`Hyp`, whose assumption entry and module record
    stay, though the build no longer has `G` read it), changes the record, and the drop its `premises` too, which the
    audit refuses until `--update` records it, and the review prints `G` as it is now. For the drop the guarantee's own
    line comes first and names the premise it no longer takes: its type hash is the same and its `premises` are not, so
    it is no reprint. A walk that stopped at `G` would record its name alone, and
    neither change would move the lock. What `G` reads under `meaning` (`twice`) is recorded too."""
    pkg, build, audit, lock = excepted_toy(
        tmp_path, "Control.Loop", LOOP,
        "theorem doubles : Guarantees.Loop := by\n  unfold Guarantees.Loop\n  intros\n  unfold twice double\n  omega\n",
        "Control.doubles", defs="def twice (n : Nat) : Prop := double n = n + n\n",
        claims_not={"Control.twice": "the toy Prop the statement reads"})
    loop = pkg / "Control" / "Loop.lean"
    build()
    rep = audit(True)
    assert unshaped(rep) == []
    first = lock()
    assert first["guarantees"]["Control.doubles"]["signature"] == ["Control.doubles : Control.Guarantees.Loop"]
    assert list(first["reads"]["Control.Loop"]["definitions"]) == ["Control.Guarantees.Loop"]
    assert rep["readers"]["Control.Loop"] == ["Control.doubles"]
    assert "Control.doubles" in rep["readers"]["Control.Defs"], "and through it, what it reads under `meaning`"
    assert "Control.twice" in first["reads"]["Control.Defs"]["definitions"]
    seen = {first["reads"]["Control.Loop"]["digest"]}
    assert first["guarantees"]["Control.doubles"]["premises"] == {"assumptions": ["Control.Hyp"], "domain": [], "open": [], "reads": []}
    for text, shows, hyp_readers, premises in (
            (LOOP.replace("Control.Hyp →", "Control.Hyp → n ≠ 5 →"), "n ≠ 5", ["Control.doubles", "Control.main_thm"],
             ["Control.Hyp"]),
            (LOOP.replace("Control.Hyp → ", ""), "∀ n : Nat, Control.twice n", ["Control.main_thm"], [])):
        before = lock()
        loop.write_text(text)
        build()
        codes = sorted(f.code for f in audit(False)["failures"] if not f.code.startswith("shape"))
        assert codes == ["guarantees.reads-changed"] if premises else ["guarantees.premises-changed",
                                                                       "guarantees.reads-changed"], codes
        rep = audit(True)
        after = lock()
        assert after["guarantees"]["Control.doubles"].pop("premises") == {"assumptions": premises, "domain": [], "open": [], "reads": []}
        before["guarantees"]["Control.doubles"].pop("premises")
        assert unshaped(rep) == [] and after["guarantees"] == before["guarantees"]
        assert after["assumptions"] == before["assumptions"] and set(after["assumptions"]) == {"Control.Hyp"}
        assert rep["readers"]["Control.Assumptions"] == hyp_readers
        assert after["reads"]["Control.Assumptions"] == before["reads"]["Control.Assumptions"]
        assert after["reads"]["Control.Loop"]["digest"] not in seen
        seen.add(after["reads"]["Control.Loop"]["digest"])
        changes = A.statement_changes(rep["review"])
        dropped = [] if premises else ["guarantee Control.doubles: no longer takes Control.Hyp (was a named assumption)"]
        assert changes[:len(dropped) + 1] == [
            *dropped, "definition Control.Guarantees.Loop (Control.Loop), read by Control.doubles: changed"], changes
        assert any(shows in line for line in changes[len(dropped) + 1:]), changes
        for module in ("Control.Loop", "Control.Assumptions"):
            after["reads"][module] = before["reads"][module]
        assert after == before, "the statement's record changed, and nothing else"


VERIFY = ("import Control.Defs\nnamespace Control\ndef verifyDouble (n : Nat) : Prop := double n = n + n\n"
          "namespace Guarantees\ndef Doubled : Prop := ∀ n : Nat, Control.verifyDouble n\nend Guarantees\nend Control\n")
DOUBLED = "theorem doubled : Guarantees.Doubled := by\n  unfold Guarantees.Doubled\n  intros\n  unfold verifyDouble double\n  omega\n"


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_verifier_read_only_in_an_excepted_module_holds_its_statement_to_witnesses(tmp_path):
    """On a real build: a guarantee whose statement (`Control.Guarantees.Doubled`, a `def … : Prop` of a proofs-tree module
    under `trees`' `except`, as Flock's `SessionLoopZK` is) applies a verifier's acceptance (`verifyDouble`) defined in
    that module and read nowhere under `meaning` states what an acceptance means: the witnesses rule holds it to its
    witnesses as it would one read under `meaning`, and `--update`, which records it, lists it under no `unwitnessed`.
    An entry there at its record's type hash exempts it, and `--update` keeps that entry. A rule that read only what the
    walk under `meaning` reaches saw no read at all, and held the guarantee to nothing."""
    pkg, build, audit, lock = excepted_toy(tmp_path, "Control.Verify", VERIFY, DOUBLED, "Control.doubled")
    build()
    fails = unshaped(audit(True))
    assert [f.code for f in fails] == ["witnesses.missing"], fails
    assert fails[0].startswith("witnesses: Control.doubled lists no witnesses") and \
        "it reads Control.verifyDouble, a verifier's acceptance" in fails[0], fails[0]
    rec = lock()
    assert "Control.verifyDouble" in rec["reads"]["Control.Verify"]["definitions"]
    assert "Control.doubled" not in rec.get("unwitnessed", {})
    entry = {"Control.doubled": rec["guarantees"]["Control.doubled"]["type_hash"]}
    A.dump(pkg, rec | {"unwitnessed": entry})
    assert unshaped(audit(False)) == []
    assert unshaped(audit(True)) == [] and lock()["unwitnessed"] == entry


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_renamed_unwitnessed_guarantee_passes_under_moved_and_its_entry_is_re_keyed(tmp_path):
    """On a real build: a guarantee `unwitnessed` lists, renamed with the statement it is (`Control.doubled :
    Guarantees.Doubled` to `Control.twofold : Guarantees.Twofold`, as a namespace rename renames both), says the same,
    but its type hash reads names. Under `--update --moved` its entry, at the hash of its text under the old
    names, passes, and the lock lists it under the new name at the hash as built, which passes without `--moved`. The
    unit test hands the rule a moved hash; this one checks that `Facts.lean`'s is the one the record and its entry hold.
    An entry of a name the lock lists no guarantee under fails until then, and that update drops it without failing on it."""
    pkg, build, audit, lock = excepted_toy(tmp_path, "Control.Verify", VERIFY, DOUBLED, "Control.doubled")
    build()
    audit(True)
    was = lock()["guarantees"]["Control.doubled"]["type_hash"]
    A.dump(pkg, lock() | {"unwitnessed": {"Control.doubled": was}})
    assert unshaped(audit(False)) == []
    A.dump(pkg, lock() | {"unwitnessed": {"Control.doubled": was, "Control.gone": was}})
    assert [(f.code, f.entry) for f in unshaped(audit(False))] == [("witnesses.unwitnessed-unlisted", "Control.gone")]
    for path in (pkg / "Control" / "Verify.lean", pkg / "Control" / "VerifyThm.lean"):
        path.write_text(path.read_text().replace("Doubled", "Twofold").replace("doubled", "twofold"))
    build()
    rep = audit(True, {"Control.doubled": "Control.twofold", "Control.Guarantees.Doubled": "Control.Guarantees.Twofold"})
    assert unshaped(rep) == [], unshaped(rep)
    rec = lock()
    now = rec["guarantees"]["Control.twofold"]["type_hash"]
    assert now != was and rec["unwitnessed"] == {"Control.twofold": now} and "Control.doubled" not in rec["guarantees"]
    assert unshaped(audit(False)) == []


HIDDEN = ("theorem hidden (n : Nat) (_h : Hiding n) (hs : Small n) : verifySmall n := hs\n"
          "theorem hidden_honest (_h : Hiding 1) : Small 1 ∧ double 1 = 2 := ⟨by unfold Small; omega, rfl⟩\n"
          "theorem hidden_nontrivial : ¬ verifySmall 10 := by unfold verifySmall; omega\n")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_premise_witnesses_assumed_lists_is_taken_and_another_listed_one_is_still_concluded(tmp_path):
    """On a real build: `Control.hidden` reads the verifier `verifySmall` and takes `Hiding n`, a named assumption of a
    definitions module that no theorem concludes (as `Claims.Hm96Hiding` is), and `Small n`; its honest witness takes
    `Hiding 1` and concludes `Small 1` (in a conjunction, which `lifted` leaves as written). Without a `witnesses_assumed`
    entry the witness fails for taking `Hiding` and for not concluding it, though the lock lists it (under `open`, as a
    premise outside the assumptions module). With one, `Hiding` is taken, and `--update` keeps the entry as it is.
    `Small`, listed under `open` too, is
    still concluded: a witness that concludes something else fails. Once no guarantee takes `Hiding` as a premise, the
    entry fails as stale."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    defs, thms = pkg / "Control" / "Defs.lean", pkg / "Control" / "Thms.lean"
    defs.write_text(defs.read_text().replace(
        "end Control", "def verifySmall (n : Nat) : Prop := n < 10\ndef Hiding (n : Nat) : Prop := n ≠ 7\n\nend Control"))
    thms.write_text(thms.read_text().replace("end Control", f"{HIDDEN}\nend Control"))
    pol = A.load(pkg)
    names = ("Control.hidden", "Control.hidden_honest", "Control.hidden_nontrivial")
    A.dump(pkg, pol | {"assumptions": pol["assumptions"] | {"Control.Hiding": {"owner": "@lean"}},
                       "open_seed": {"Control": {"issue": "LEAN-1", "due": "2026-11-30"}},
                       "claims": pol["claims"] | {"not": {"Control.verifySmall": "the toy verifier the guarantee reads"}},
                       "guarantees": pol["guarantees"] | {n: {"owner": "@lean"} for n in names},
                       "witnesses": {"Control.hidden": {"honest": ["Control.hidden_honest"],
                                                        "nontrivial": "Control.hidden_nontrivial"}}})

    def build():
        return subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())

    def witnessing(update):
        return [f for f in A.audit(pkg, tmp_path / "out", update=update, replay=False, runs=False, log=lambda *_: None)[
            "failures"] if f.code.startswith("witnesses")]

    def lock():
        return json.loads((pkg / A.POLICY).read_text())
    build()
    fails = witnessing(True)
    assert sorted(f.code for f in fails) == ["witnesses.honest-form"] * 2 + ["witnesses.honest-uncovered"], fails
    assert any("takes (_h : Control.Hiding 1), which is no named assumption the lock lists" in f for f in fails), fails
    assert any("rests on Control.Hiding, no named assumption of an `Assumptions`" in f for f in fails), fails
    assert any("conclude nothing of Control.Hiding at the deployed instance" in f for f in fails), fails
    assert {"Control.Hiding", "Control.Small"} <= set(lock()["open"]) and "Control.Hiding" not in lock()["assumptions"]
    why = "the toy's hiding assumption, which no theorem concludes"
    A.dump(pkg, lock() | {"witnesses_assumed": {"Control.Hiding": why}})
    assert witnessing(False) == [] and witnessing(True) == [] and lock()["witnesses_assumed"] == {"Control.Hiding": why}

    concluded = "Small 1 ∧ double 1 = 2 := ⟨by unfold Small"
    other = "verifySmall 1 ∧ double 1 = 2 := ⟨by unfold verifySmall"
    thms.write_text(thms.read_text().replace(concluded, other))
    build()
    fails = witnessing(True)
    assert [f.code for f in fails] == ["witnesses.honest-uncovered"] and \
        "conclude nothing of Control.Small at the deployed instance" in fails[0], fails

    thms.write_text(thms.read_text().replace(other, concluded).replace("(_h : Hiding n) ", "").replace("(_h : Hiding 1) ", ""))
    build()
    assert witnessing(True) == ["witnesses: `witnesses_assumed` lists Control.Hiding, which no guarantee the lock lists "
                                "takes as a premise; remove the entry"]


PARTITION = ("namespace Control\ndef Bound (x y : Nat) : Prop := x ≤ y\ndef Fits (n : Nat) : Prop := n ≤ 5\n"
             "namespace Guarantees\ndef Bounded : Prop := ∀ n : Nat, Fits n → Bound n 5\nend Guarantees\nend Control\n")
BOUNDED = "theorem bounded : Guarantees.Bounded := by\n  unfold Guarantees.Bounded Bound Fits\n  intro n h\n  exact h\n"


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_bound_a_definition_hides_holds_its_guarantee_to_exclusion_witnesses(tmp_path):
    """On a real build: `Control.bounded : Guarantees.Bounded`, `∀ n, Fits n → Bound n 5` with `Bound x y := x ≤ y` in
    `Control.Partition`, concludes a bound of a partition module two definitions deep (as `Nci.Products.Guarantees.Bound`
    concludes `PartitionBound`, whose `≤` is under its `∀`): `lifted` unfolds `Guarantees.Bounded` to `Bound`, and the
    `≤` is `Bound`'s. `--update` fails it with `witnesses.missing` of the `exclusion` kind until its honest and
    nontrivial witnesses are listed. A `chain` that recorded only the definitions it unfolds saw no bound head, and held
    it to nothing."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    (pkg / "Control" / "Partition.lean").write_text(PARTITION)
    thms = pkg / "Control" / "Thms.lean"
    thms.write_text("import Control.Partition\n" + thms.read_text().replace("end Control", (
        f"{BOUNDED}theorem bounded_honest : Fits 3 ∧ double 1 = 2 := ⟨by unfold Fits; omega, rfl⟩\n"
        "theorem bounded_nontrivial : ¬ Fits 6 ∧ ¬ Bound 6 5 := ⟨by unfold Fits; omega, by unfold Bound; omega⟩\n"
        "\nend Control")))
    pol = A.load(pkg)
    A.dump(pkg, pol | {"trees": pol["trees"] | {"definitions": [*pol["trees"]["definitions"], "Control.Partition"]},
                       "layers": pol["layers"] | {"Control.Partition": {"imports": [], "why": "the bound imports nothing"}},
                       "guarantees": pol["guarantees"] | {"Control.bounded": {"owner": "@lean"}}})
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())

    def witnessing(update):
        return [f for f in A.audit(pkg, tmp_path / "out", update=update, replay=False, runs=False, log=lambda *_: None)[
            "failures"] if f.code.startswith("witnesses")]
    fails = witnessing(True)
    assert [f.code for f in fails] == ["witnesses.missing"], fails
    assert "it concludes a bound and reads Control.Bound of Control.Partition, a partition or covering" in fails[0] and \
        '"kind": "exclusion"' in fails[0], fails[0]
    rec = json.loads((pkg / A.POLICY).read_text())
    names = ("Control.bounded_honest", "Control.bounded_nontrivial")
    A.dump(pkg, rec | {"guarantees": rec["guarantees"] | {n: {"owner": "@lean"} for n in names},
                       "witnesses": {"Control.bounded": {"honest": [names[0]], "nontrivial": names[1], "kind": "exclusion"}}})
    assert witnessing(False) == [] and witnessing(True) == []


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_bound_hidden_in_an_excepted_module_holds_its_guarantee_to_exclusion_witnesses(tmp_path):
    """On a real build: the same statement in a proofs-tree module under `trees`' `except`, outside `meaning` (as Flock's
    statements are in `Proofs.Flock`): `chain` unfolds through the modules the record follows (`record_scope`), so
    `--update` fails it with `witnesses.missing` of the `exclusion` kind and lists it under no `unwitnessed`. An entry
    there at its record's type hash exempts it, and `--update` keeps that entry. A `chain` that stopped outside `meaning`
    recorded nothing past the statement's own name."""
    pkg, build, audit, lock = excepted_toy(tmp_path, "Control.Partition", PARTITION, BOUNDED, "Control.bounded")
    build()

    def witnessing(update):
        return [f for f in audit(update)["failures"] if f.code.startswith("witnesses")]
    fails = witnessing(True)
    assert [f.code for f in fails] == ["witnesses.missing"], fails
    assert '"kind": "exclusion"' in fails[0] and "reads Control.Bound of Control.Partition" in fails[0], fails[0]
    rec = lock()
    assert "Control.bounded" not in rec.get("unwitnessed", {})
    entry = {"Control.bounded": rec["guarantees"]["Control.bounded"]["type_hash"]}
    A.dump(pkg, rec | {"unwitnessed": entry})
    assert witnessing(False) == [] and witnessing(True) == [] and lock()["unwitnessed"] == entry


CAPPED = ("theorem capped : Guarantees.Capped := by\n  unfold Guarantees.Capped Bound\n  intro n\n  omega\n"
          "theorem capped_honest : Bound (min 2 3) 5 ∧ double 1 = 2 := ⟨by unfold Bound; omega, rfl⟩\n"
          "theorem capped_nontrivial : ¬ Bound (min 4 3) 2 := by unfold Bound; omega\n"
          "theorem bounded_honest : Fits 3 ∧ double 1 = 2 := ⟨by unfold Fits; omega, rfl⟩\n"
          "theorem bounded_nontrivial : ¬ Bound 6 5 := by unfold Bound; omega\n")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_premise_free_exclusion_bound_is_witnessed_by_a_tighter_bound_failing_at_an_explicit_circuit(tmp_path):
    """On a real build: `Control.capped : Guarantees.Capped`, `∀ n, Bound (min n 3) 5`, takes no premise (as training's
    `ServingBlocks.UncertifiedElementary` takes none), so its nontrivial witness is the bare `¬ Bound (min 4 3) 2`: a
    tighter bound fails at an explicit circuit. `Control.bounded`, which takes `Fits n`, still fails with a bare
    `¬ Bound 6 5`, which drops no premise. A rule that asked every exclusion theorem to drop a premise failed `capped`'s
    witness too, whatever it stated."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    (pkg / "Control" / "Partition.lean").write_text(PARTITION.replace(
        "end Guarantees", "def Capped : Prop := ∀ n : Nat, Bound (min n 3) 5\nend Guarantees"))
    thms = pkg / "Control" / "Thms.lean"
    thms.write_text("import Control.Partition\n" + thms.read_text().replace("end Control", f"{BOUNDED}{CAPPED}\nend Control"))
    pol = A.load(pkg)
    names = [f"Control.{g}{s}" for g in ("capped", "bounded") for s in ("", "_honest", "_nontrivial")]
    A.dump(pkg, pol | {"trees": pol["trees"] | {"definitions": [*pol["trees"]["definitions"], "Control.Partition"]},
                       "layers": pol["layers"] | {"Control.Partition": {"imports": [], "why": "the bound imports nothing"}},
                       "guarantees": pol["guarantees"] | {n: {"owner": "@lean"} for n in names},
                       "witnesses": {f"Control.{g}": {"honest": [f"Control.{g}_honest"], "nontrivial": f"Control.{g}_nontrivial",
                                                      "kind": "exclusion"} for g in ("capped", "bounded")}})
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())

    def witnessing(update):
        return [f for f in A.audit(pkg, tmp_path / "out", update=update, replay=False, runs=False, log=lambda *_: None)[
            "failures"] if f.code.startswith("witnesses")]
    fails = witnessing(True)
    assert [(f.code, f.entry) for f in fails] == [("witnesses.nontrivial-drops", "Control.bounded_nontrivial")], fails
    assert "with P one of Control.Fits; a bare `¬ bound` reads none" in fails[0], fails[0]
    rec = json.loads((pkg / A.POLICY).read_text())
    A.dump(pkg, rec | {"witnesses": {}})
    missing = {f.entry: str(f) for f in witnessing(False) if f.code == "witnesses.missing"}
    assert {"Control.capped", "Control.bounded"} <= set(missing), missing
    assert "`Control.capped_nontrivial : ¬ Control.Bound c₀ …` (it takes no premise to drop" in missing["Control.capped"]
    assert "once one named premise is dropped" in missing["Control.bounded"]


def test_a_statement_whose_premises_could_not_be_read_fails():
    """`Facts.lean` writes `premises: null` when `premisesOf` throws on a statement; read as no premises, every premise of it
    would go unlisted, so it fails instead. Facts without the field (older documents) are not held to it."""
    pin = fact() | {"premises": None}
    fails, _ = A.judge(POLICY, facts(pins={"P.t": pin}), FILES, update=True)
    assert fails == ["guarantees: P.t's premises could not be read (`Facts.lean`'s `premisesOf` failed on its statement), so "
                     "the lock can't be held to them: make `premisesOf` read this statement's form"]
    assert A.judge(POLICY, facts(), FILES, update=True)[0] == []


def test_the_verifiers_path_has_no_unlisted_compiled_code_and_hashes_through_its_parameter():
    """The verifier's path (`verifier` in the policy) has no unlisted compiled code and hashes through its parameter: an `@[extern]` or `@[implemented_by]`
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
    """`verifier.allow` lists by name each `@[extern]` and `@[implemented_by]` constant the verifier reaches, as
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


def test_verifier_allow_with_a_name_and_its_prefix_passes_both_modes():
    """`--update` writes `Nat.gcd` and `Nat.gcd._unary` by name when both are on the path, and `Nat.gcd` also admits
    `Nat.gcd._unary` as a name prefix: the check without `--update` counts every entry that admits a constant as used, so
    the record `--update` writes passes it, and neither entry reads as one that admits nothing."""
    path = {"missing": [], "reached": 40, "hash": [], "csimp": [], "implemented_by": [],
            "extern": [{"name": "Nat.gcd", "module": "Init.Data.Nat.Gcd"},
                       {"name": "Nat.gcd._unary", "module": "Init.Data.Nat.Gcd"}]}
    rule = {"entries": ["P.verify"], "stop": [], "hash": "P.Sha512", "allow": []}
    f = facts(); f["verifier"] = path
    fails, pol = A.judge({**POLICY, "verifier": rule}, f, FILES, update=True)
    assert fails == [] and pol["verifier"]["allow"] == ["Nat.gcd", "Nat.gcd._unary"]
    assert A.verifier_failures(pol["verifier"], path) == []
    assert A.verifier_failures({**rule, "allow": ["Nat.gcd", "Nat.gcd._unary", "Nat.lcm"]}, path) == [
        "verifier: `verifier.allow` lists Nat.lcm, which admits nothing on the path from P.verify; run `check.py --update`"]


def test_every_claim_has_a_witness_that_it_fails_and_witnesses_that_its_antecedents_are_met():
    """Each predicate the spec declares under `claims.modules` names a `fails` theorem stating `¬ C …` with that
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
    `SampledLeakAccepted` takes `Claims.SamplingGeneric.Compiled.KS`) keeps its `claims.witnesses` or `claims.not`
    entry, whose witnesses are still checked, and needs none; an entry naming no predicate is stale, and a claim that is
    no assumption needs one."""
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


def generic_fixture():
    """(policy, facts) shaped like GENSAMPLE's compiler statements: `P.Compiler`, a closed `def : Prop` under
    `claims.modules` over every proof system, stated by `P.compiler` (`theorem P.compiler : P.Compiler`), which takes
    `P.Arg.KS H A` and `P.Compiled.Sessions H L` inline, reads the claims and the hash vocabulary, has no shape entry, and
    is listed under `claims.generic` with its witnesses."""
    def spec(name, statement):
        return {"name": name, "module": "P.Claims", "kind": "def", "statement": statement, "predicate": True}

    reads = [{"name": "P.Compiler", "module": "P.Claims", "kind": "def", "hash": "4" * 32},
             {"name": "P.Arg.KS", "module": "P.Claims", "kind": "def", "hash": "5" * 32},
             {"name": "Hash.Msg", "module": "Definitions.Core.Hash", "kind": "def", "hash": "6" * 32},
             {"name": "List.length", "module": "Init.Prelude", "kind": "def", "hash": "7" * 32}]
    g = fact(signature="P.compiler : P.Compiler", hyps=[], reads=reads, h="00000000000000cc") | {
        "body": {"head": "P.Compiler", "negated": False, "binders": [], "conclusion": [], "absurd": False},
        "inline": [premise("P.Arg.KS H A", "P.Arg.KS", binder="a"), premise("P.Compiled.Sessions H L",
                                                                        "P.Compiled.Sessions", binder="a")],
        "premises": []}
    built = {m: {"olean": f"/r/Security/.lake/build/lib/lean/{m}.olean"}
             for m in ("P.Claims", "Definitions.Core.Hash", "Definitions.Pouw", "PouwCheck.Check", "P.Guarantees")}
    graph = built | {"Init.Prelude": {"olean": "/elan/toolchains/v4/lib/lean/Init/Prelude.olean"},
                     "Mathlib.Data": {"olean": "/r/Security/.lake/packages/mathlib/.lake/build/lib/lean/Mathlib/Data.olean"}}
    f = facts(pins={"P.t": fact(), "P.compiler": g}, spec_definitions=[spec("P.Compiler", True), spec("P.KS", False)],
              graph=graph)
    policy = {**POLICY, "guarantees": {"P.t": {"owner": "@lean"}, "P.compiler": {"owner": "@proofs"}},
              "claims": {"modules": ["P.Claims"], "generic": {"P.Compiler": "P.compiler"},
                         "not": {"P.KS": "a helper the claims read"}},
              "witnesses": {"P.compiler": {"honest": ["P.compiler_honest"], "nontrivial": "P.compiler_nontrivial"}}}
    return policy, f


def generic_codes(policy, f) -> list[tuple[str, str]]:
    return [(x.code, x.entry) for x in A.claim_failures(policy, f)]


def test_a_generic_statement_s_guarantee_is_exempt_from_the_inline_and_shape_rules():
    """A closed statement proved of every instance, listed under `claims.generic` with its guarantee and witnesses, passes
    the claims rule, and its guarantee's inline antecedents and missing shape entry fail nothing; the same guarantee
    listed nowhere fails both rules, and its claim the claims rule."""
    policy, f = generic_fixture()
    assert A.claim_failures(policy, f) == []
    assert A.generic_claims(policy) == {"P.Compiler": "P.compiler"}
    ifails, _, offenders = A.inline_failures(policy, f)
    assert [x for x in ifails if x.entry == "P.compiler"] == [] and "P.compiler" not in offenders
    assert [x for x in A.shape(policy, f) if x.entry == "P.compiler"] == []
    bare = {**policy, "claims": {k: v for k, v in policy["claims"].items() if k != "generic"}}
    assert [x.code for x in A.inline_failures(bare, f)[0] if x.entry == "P.compiler"] == ["guarantees.inline-hypothesis"]
    assert [x for x in A.shape(bare, f) if x.entry == "P.compiler"] != []
    assert A.claim_failures(bare, f) == [
        "claims: P.Compiler (P.Claims) is a claim definition with no witnesses: name its `P.Compiler_fails` (a theorem "
        "that it is false at a toy instance) and `P.Compiler_meets` (its checkable antecedents hold at each deployed "
        "record) under `claims.witnesses`, or list it under `claims.not` with the reason it is no claim, or, proved of "
        "every instance, under `claims.generic` with the guarantee that states it"]


def test_a_generic_statement_is_refused_when_it_takes_parameters():
    policy, f = generic_fixture()
    f["spec_definitions"][0]["statement"] = False
    assert generic_codes(policy, f) == [("claims.generic-parameters", "P.Compiler")]
    assert "close it, or give it `claims.witnesses`" in A.claim_failures(policy, f)[0]


def test_a_generic_statement_is_refused_when_its_guarantee_states_something_else():
    """The guarantee states `C` itself: not another head, not `¬ C`, not `C` under binders, and it is a listed theorem."""
    policy, f = generic_fixture()
    body = f["pins"]["P.compiler"]["body"]
    for wrong in ({"head": "P.Other"}, {"negated": True}, {"binders": ["∀"]}):
        f["pins"]["P.compiler"]["body"] = body | wrong
        assert generic_codes(policy, f) == [("claims.generic-statement", "P.Compiler")], wrong
    assert A.claim_failures(policy, f)[0].endswith("state it as `theorem P.compiler : P.Compiler`")
    f["pins"]["P.compiler"]["body"] = {"head": "P.Lifted"}
    f["pins"]["P.compiler"]["lift"] = {"head": "P.Compiler", "module": "P.Claims"}
    assert generic_codes(policy, f) == []
    unlisted = {**policy, "guarantees": {"P.t": {"owner": "@lean"}}}
    assert generic_codes(unlisted, f) == [("claims.generic-unlisted", "P.Compiler")]


def test_a_generic_statement_is_refused_without_both_witnesses_or_under_unwitnessed():
    policy, f = generic_fixture()
    for ws in ({}, {"P.compiler": {"honest": ["P.compiler_honest"]}}, {"P.compiler": {"honest": [], "nontrivial": "n"}}):
        assert generic_codes({**policy, "witnesses": ws}, f) == [("claims.generic-witnesses", "P.Compiler")], ws
    assert "at least two sessions or rounds" in A.claim_failures({**policy, "witnesses": {}}, f)[0]
    exempt = {**policy, "unwitnessed": {"P.compiler": f["pins"]["P.compiler"]["type_sha256"]}}
    assert generic_codes(exempt, f) == [("claims.generic-unwitnessed", "P.Compiler")]


def test_a_generic_statement_reads_only_the_claims_vocabulary_core_and_what_no_package_here_builds():
    """A generic statement may read `claims.modules`, `Definitions.Core` and modules the import graph places outside this
    repository (Init, Mathlib); a read of any other in-tree module (code, a `Guarantees` module), or of one the graph
    doesn't place, fails, naming the module."""
    policy, f = generic_fixture()
    reads = f["pins"]["P.compiler"]["reads"]
    for module in ("PouwCheck.Check", "P.Guarantees", "Unplaced.Module"):
        f["pins"]["P.compiler"]["reads"] = reads + [{"name": "Q.deployed", "module": module, "kind": "def"}]
        assert generic_codes(policy, f) == [("claims.generic-reads", "P.Compiler")], module
    assert A.claim_failures(policy, f) == [
        "claims: claims.generic[P.Compiler]: P.compiler reads Unplaced.Module: a generic statement reads only the claims "
        "vocabulary and Core; a statement that reads Unplaced.Module is about a deployment and belongs under the "
        "guarantee rules"]
    f["pins"]["P.compiler"]["reads"] = reads + [{"name": "Mathlib.thing", "module": "Mathlib.Data", "kind": "def"}]
    assert generic_codes(policy, f) == []


def test_a_generic_statement_reading_another_area_s_deployed_numbers_is_refused():
    """A deployed number in a definitions module outside the claims and Core (`Definitions.Pouw`'s) makes the statement
    one about the deployment, though it is spec text like the claims."""
    policy, f = generic_fixture()
    f["pins"]["P.compiler"]["reads"] += [{"name": "Pouw.tileRows", "module": "Definitions.Pouw", "kind": "def"}]
    assert generic_codes(policy, f) == [("claims.generic-reads", "P.Compiler")]
    assert "P.compiler reads Definitions.Pouw:" in A.claim_failures(policy, f)[0]


def test_a_generic_entry_is_one_claim_s_only_entry_and_names_a_claim():
    policy, f = generic_fixture()
    rule = policy["claims"]
    both = {**policy, "claims": {**rule, "not": {**rule["not"], "P.Compiler": "proved of every program"}}}
    assert generic_codes(both, f) == [("claims.generic-both", "P.Compiler")]
    stale = {**policy, "claims": {**rule, "generic": {**rule["generic"], "P.Gone": "P.compiler"}}}
    assert generic_codes(stale, f) == [("claims.stale", "P.Gone")]
    assert generic_codes({**policy, "claims": {**rule, "generic": ["P.Compiler"]}}, f)[0] == ("claims.generic-form", None)


def test_claims_not_excuses_no_closed_statement():
    """A closed `def C : Prop` under `claims.modules` that `claims.not` lists fails with the fix (move it to
    `claims.generic` with its guarantee and witnesses), and passes once it is there; a predicate with parameters under
    `claims.not`, a helper, still passes."""
    policy, f = generic_fixture()
    rule = policy["claims"]
    excused = {**policy, "claims": {**rule, "generic": {}, "not": {**rule["not"], "P.Compiler": "proved of every program"}}}
    assert generic_codes(excused, f) == [("claims.not-closed", "P.Compiler")]
    assert A.claim_failures(excused, f)[0] == (
        "claims: P.Compiler (P.Claims) is a closed statement (`def P.Compiler : Prop`) that `claims.not` excuses, so "
        "nothing holds it to witnesses: move it to `claims.generic` with the guarantee that states it (`theorem …: "
        "P.Compiler`, under `guarantees`) and its `witnesses`, honest lemmas, its antecedents holding together at one "
        "instance (at least two sessions or rounds where it composes them), and a nontrivial one, its conclusion failing "
        "at an instance")
    assert generic_codes(policy, f) == []


def test_a_meets_witness_says_something_about_the_deployment():
    """A `meets` witness whose statement is `True`, a conjunction of `True`s, a reflexivity, one of its own
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
        "deployed verifier, taking only named assumptions of an `Assumptions` or `shape.premises` module or "
        "`witnesses_assumed`) and "
        "`P.sound_nontrivial : ¬ P.Holds at acceptAll …` (its conclusion is false for a verifier that accepts "
        "everything). Each takes no binder over its instance and no hypothesis beyond named assumptions the lock lists: "
        "it builds the instance at the model level (a small hand-built descriptor, not a parsed file), with any one field "
        "that can't be built a named assumption. List both under `guarantees`, and write \"P.sound\": {\"honest\": "
        "[\"P.sound_honest\"], \"nontrivial\": \"P.sound_nontrivial\"} under `witnesses`: a guarantee carries an honest and "
        "a nontrivial witness. Only if `unwitnessed` listed it and a merge dropped the entry, write the entry back: "
        f"\"P.sound\": \"{f['pins']['P.sound']['type_sha256']}\"; a new guarantee gets its witnesses, not an entry "
        "(`UNWITNESSED` only shrinks)"]
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
        "each premise outside a named assumption's module and `witnesses_assumed`, and a property's acceptance, is a "
        "conjunct of one (a premise that can never hold makes P.sound say nothing)")
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


def test_an_honest_witness_may_state_a_premise_as_an_order_and_a_nontrivial_one_negates_alone():
    """An `honest` lemma may state a plain-range premise at the deployed numbers (`0 ≤ 0`, applying no definition) beside
    a conjunct that applies one, though not on its own. A `nontrivial` lemma is `¬ C …` itself: a conjunction with a `¬ C`
    conjunct is refused, since `Facts.lean` reports no binders inside a conjunct, so `True ∧ (0 = 1 → ¬ C …)` would
    pass."""
    order = {"head": "LE.le", "defs": [], "refl": False, "assumed": False}
    both = wbody("And", conclusion=[wk("P.Compiled"), order])
    assert witness_codes(*witnessed(honest={"body": both})) == []
    assert witness_codes(*witnessed(honest={"body": wbody("LE.le", conclusion=[order])})) == \
        ["witnesses.honest-form", "witnesses.honest-uncovered"]
    fails = {"head": "Not", "defs": ["P.Holds"], "refl": False, "assumed": False, "negated": True, "inner": "P.Holds"}
    assert witness_codes(*witnessed(nontrivial={"body": wbody("And", conclusion=[wk("True", []), fails])})) == \
        ["witnesses.nontrivial-form"]


def test_a_property_of_an_acceptance_has_an_honest_witness_that_the_deployment_is_accepted():
    """A guarantee `shape.properties` lists as a property of a function's acceptance is in scope whatever it reads, and
    its honest witnesses conclude that the function accepts the deployed instance as well as its premises; a property of a
    rejection is not in scope by that alone."""
    prop = {"properties": {"P.sound": {"function": "P.verify", "outcome": "accept"}}}
    pol, f = witnessed({"reads": fact()["reads"]}, shape=prop)
    assert A.witness_failures(pol, f, WOWN) == [
        "witnesses: P.sound's `honest` witnesses (P.sound_honest) conclude nothing of P.verify accepting at the deployed "
        "instance: each premise outside a named assumption's module and `witnesses_assumed`, and a property's acceptance, "
        "is a conjunct of one (a premise that can never hold makes P.sound say nothing)"]
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
    the witnesses it would exempt, or of a witness, does nothing, the witnesses checked as ever. An entry of a name the
    lock lists no guarantee under fails, with the fix, and an update drops it and doesn't fail on it. An entry may exempt
    one half (`missing`) of a guarantee listing the other. Under `--moved`, an entry at its text under the old names
    matches when its record has that hash too, and the lock judged is re-keyed with its guarantees. `--update` keeps an
    entry while it matches (re-keyed when the guarantee moved), drops every other, and adds none, so a new guarantee in
    scope fails until it has witnesses."""
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
    gone = {**bare, "unwitnessed": {"P.sound": h, "P.gone": sha("a guarantee the lock listed once")}}
    assert A.witness_failures(gone, f, WOWN) == [
        "witnesses: `unwitnessed` lists P.gone, which is not a guarantee of this package; run `check.py --update`, which "
        "drops the entry, or remove it"]
    assert witness_codes(gone, f) == ["witnesses.unwitnessed-unlisted"]
    assert A.unwitnessed_kept(gone, f, {}) == {"P.sound": h}
    assert "witnesses.unwitnessed-unlisted" in {x.code for x in A.judge(gone, f, FILES)[0]}
    fails, rec = A.judge(gone, f, FILES, update=True)
    assert "witnesses.unwitnessed-unlisted" not in {x.code for x in fails} and rec["unwitnessed"] == {"P.sound": h}
    unlisted = {**POLICY, "guarantees": {}, "unwitnessed": {"P.sound": h}}
    assert witness_codes(unlisted, f) == ["witnesses.unwitnessed-unlisted"] and A.unwitnessed_kept(unlisted, f, {}) == {}
    assert A.unwitnessed_kept(unlisted, f, {"P.sound": {"owner": "@lean", "type_hash": h}}) == {"P.sound": h}, \
        "a name among the records the update writes keeps its entry"
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
    assert witness_codes(was, moved) == []
    assert A.unwitnessed_kept(was, moved, {}) == {"P.sound": h}
    stale = {**was, "guarantees": {"P.sound": {"owner": "@lean", "type_hash": sha("x")}}}
    assert witness_codes(stale, moved) == ["witnesses.unwitnessed-changed"]
    assert A.unwitnessed_kept(stale, moved, {}) == {}
    assert witness_codes({**was, "unwitnessed": {"P.sound": sha("other")}}, moved) == ["witnesses.unwitnessed-changed"]
    named = facts(pins={"Q.sound": moved["pins"]["P.sound"]}, own_modules=sorted(WOWN))
    cur = A.renamed_current({**was, "unwitnessed": {"P.sound": old, "P.gone": old}}, {"P.sound": "Q.sound", "P.gone": "Q.gone"})
    assert set(cur["guarantees"]) == {"Q.sound"} and cur["unwitnessed"] == {"Q.sound": old, "Q.gone": old}
    assert [(x.code, x.entry) for x in A.witness_failures(cur, named, WOWN)] == [("witnesses.unwitnessed-unlisted", "Q.gone")]
    assert A.unwitnessed_kept(cur, named, {}) == {"Q.sound": h}, "re-keyed with its guarantee, the entry of none dropped"

    assert A.renamed_witnesses({"witnesses": {"P.a": {"honest": ["P.a_h"], "nontrivial": "P.a_n"}},
                                "unwitnessed": {"P.b": h}}, {"P": "Q"}) == {
        "witnesses": {"Q.a": {"honest": ["Q.a_h"], "nontrivial": "Q.a_n"}}, "unwitnessed": {"Q.b": h}}
    assert A.renamed_witnesses(POLICY, {"P": "Q"}) == {}


def test_a_listed_guarantee_left_with_neither_witnesses_nor_an_unwitnessed_entry_is_held_to_its_witnesses():
    """A guarantee in scope that the lock lists with neither witnesses nor an `unwitnessed` entry, as a merge leaves one
    that one side lists and whose entry the other side dropped: the check holds it to its witnesses
    (`witnesses.missing`). The failure says how to carry them and, for that case alone, prints the entry to write back
    at its statement as built, in the form the lock run writes: its type hash, or {type_hash, unfolded_hash} of a
    guarantee stated as a definition; a new guarantee gets its witnesses. Written back as printed, the lock passes.
    `--update` lists it under no `unwitnessed`, which only shrinks; after a landing the lock run lists it there and
    warns."""
    lock, now = bookkeeping_cases()["witnesses.missing"]
    h = now["pins"]["P.sound"]["type_sha256"]
    assert "P.sound" in lock["guarantees"] and "witnesses" not in lock and lock["unwitnessed"] == {}
    (missing,) = fails = A.judge(lock, now, FILES)[0]
    assert (missing.code, missing.entry) == ("witnesses.missing", "P.sound")
    assert missing == (
        "witnesses: P.sound lists no witnesses that it says something, and the rule holds it to them (it reads the claim "
        "P.Holds): prove `P.sound_honest : P.Compiled d …` (its hypotheses hold at the honest deployed instance d, at the "
        "deployed verifier, taking only named assumptions of an `Assumptions` or `shape.premises` module or "
        "`witnesses_assumed`) and "
        "`P.sound_nontrivial : ¬ P.Holds at acceptAll …` (its conclusion is false for a verifier that accepts "
        "everything). Each takes no binder over its instance and no hypothesis beyond named assumptions the lock lists: "
        "it builds the instance at the model level (a small hand-built descriptor, not a parsed file), with any one field "
        "that can't be built a named assumption. List both under `guarantees`, and write \"P.sound\": {\"honest\": "
        "[\"P.sound_honest\"], \"nontrivial\": \"P.sound_nontrivial\"} under `witnesses`: a guarantee carries an honest and "
        "a nontrivial witness. Only if `unwitnessed` listed it and a merge dropped the entry, write the entry back: "
        f"\"P.sound\": \"{h}\"; a new guarantee gets its witnesses, not an entry (`UNWITNESSED` only shrinks)")

    def printed(failure) -> dict:
        return json.loads("{" + failure.split("write the entry back: ")[1].split("; a new guarantee gets")[0] + "}")
    assert printed(missing) == {"P.sound": h} == {"P.sound": lock["guarantees"]["P.sound"]["type_hash"]}
    assert A.judge({**lock, "unwitnessed": printed(missing)}, now, FILES)[0] == []
    kept, rec = A.judge(lock, now, FILES, update=True)
    assert [(x.code, x.entry) for x in kept] == [("witnesses.missing", "P.sound")] and rec["unwitnessed"] == {}
    fixed = A.repaired(lock, now, FILES, fails, TICKET)
    assert fixed["unwitnessed"] == printed(missing) and A.judge(fixed, now, FILES)[0] == []
    assert A.bookkeeping_warnings(fails, fixed, TICKET) == [
        "bookkeeping: P.sound (@lean): listed under `unwitnessed` at its statement's hashes now: it owes its `honest` and "
        "`nontrivial` witnesses (witnesses.missing)"]

    body = wbody("P.Holds", binders=["∀", "→", "→"])
    _, stated = witnessed({"lift": lifted_pin("P.Holds, first")["lift"], "body": body})
    pair = {"type_hash": stated["pins"]["P.sound"]["type_sha256"], "unfolded_hash": sha("P.Holds, first")}
    bare = {**POLICY, "guarantees": {"P.sound": {"owner": "@lean", "type_hash": pair["type_hash"]}}}
    (lifted,) = A.witness_failures(bare, stated, WOWN)
    assert lifted.code == "witnesses.missing" and lifted.endswith(
        "write the entry back: \"P.sound\": " + json.dumps(pair) + "; a new guarantee gets its witnesses, not an entry "
        "(`UNWITNESSED` only shrinks)")
    assert printed(lifted) == {"P.sound": pair}
    assert A.witness_failures({**bare, "unwitnessed": printed(lifted)}, stated, WOWN) == []
    assert A.repaired(bare, stated, FILES, A.judge(bare, stated, FILES)[0], TICKET)["unwitnessed"] == {"P.sound": pair}


def lifted_pin(body: str) -> dict:
    """`P.SecurityProofs.G : P.Guarantees.G` as `Facts.lean` reports it, with `P.Guarantees.G` defined as ``body``: its
    signature and type hash are the constant's, and only `lift`'s hash of it unfolded once reads the body."""
    return fact(signature="P.SecurityProofs.G : P.Guarantees.G", hyps=[], reads=[], h="00000000000000cc") | {
        "lift": {"head": "P.Guarantees.G", "module": "P.Guarantees", "unfolded_sha256": sha(body)}}


def test_a_restated_definition_a_guarantee_is_stated_as_changes_its_record():
    """A guarantee stated as `theorem G : Guarantees.G` has the constant's signature and type hash, which a restated body
    leaves alone: its record also holds the hash of the statement with the definition unfolded once (`unfolded_hash`), so
    the restatement is a changed record, which `--update` settles, and a record without the hash fails until `--update`
    writes it."""
    pol = {**POLICY, "assumptions": {}, "guarantees": {"P.SecurityProofs.G": {"owner": "@lean"}}}
    was, now = (facts(pins={"P.SecurityProofs.G": lifted_pin(b)}) for b in ("∀ n, P.f n = n", "∀ n, P.f n ≤ n"))
    fails, rec = A.judge(pol, was, FILES, update=True)
    assert fails == [] and rec["guarantees"]["P.SecurityProofs.G"]["unfolded_hash"] == sha("∀ n, P.f n = n")
    assert A.judge(rec, was, FILES)[0] == []
    (restated,) = A.judge(rec, now, FILES)[0]
    assert restated.code == "guarantees.unfolded-hash" and restated.entry == "P.SecurityProofs.G"
    assert str(restated).startswith("guarantees: P.SecurityProofs.G's statement, its definition P.Guarantees.G unfolded once, "
                                    "changed (") and "the definition was restated" in str(restated)
    assert A.RECORD_DIFFERENCE.match(restated)
    regen = A.regenerated(rec, now, FILES, [restated])
    assert regen["guarantees"]["P.SecurityProofs.G"]["unfolded_hash"] == sha("∀ n, P.f n ≤ n")
    assert regen["guarantees"]["P.SecurityProofs.G"]["signature"] == rec["guarantees"]["P.SecurityProofs.G"]["signature"]
    old = {**rec, "guarantees": {"P.SecurityProofs.G": {k: v for k, v in rec["guarantees"]["P.SecurityProofs.G"].items()
                                                         if k != "unfolded_hash"}}}
    (unkeyed,) = A.judge(old, was, FILES)[0]
    assert unkeyed.code == "guarantees.unfolded-hash" and "has no hash in its record" in str(unkeyed)
    assert "unfolded_hash" not in A.judge(pol, facts(), FILES, update=True)[1]["guarantees"]["P.SecurityProofs.G"]


def test_an_unwitnessed_entry_of_a_restated_definition_fails():
    """Of a guarantee stated as a definition, the `unwitnessed` entry holds the statement unfolded once (`unfolded_hash`)
    beside its type hash: a restated body fails it as a changed statement does, and `--update` drops it. An entry without
    `unfolded_hash` exempts it only at the hash of its statement unfolded (its text before a lift), which `--update` keys
    (a whole exemption becomes {type_hash, unfolded_hash}, a half keeps its `missing` and `why`). At its type hash, the
    constant's, which a restated body leaves as it is, it fails with or without `--update`, which drops it: a bare entry
    written at the type hash was keyed by `--update` at whatever the body had become."""
    body = wbody("P.Holds", binders=["∀", "→", "→"])
    pol, f = witnessed({"lift": lifted_pin("P.Holds, first")["lift"], "body": body})
    _, g = witnessed({"lift": lifted_pin("P.Holds, restated")["lift"], "body": body})
    h, u = f["pins"]["P.sound"]["type_sha256"], sha("P.Holds, first")
    bare = {**POLICY, "guarantees": {"P.sound": {"owner": "@lean", "type_hash": h}}}
    keyed = {**bare, "unwitnessed": {"P.sound": {"type_hash": h, "unfolded_hash": u}}}
    assert witness_codes(keyed, f) == []
    assert A.unwitnessed_kept(keyed, f, {}) == keyed["unwitnessed"]
    assert witness_codes(keyed, g) == ["witnesses.unwitnessed-changed"]
    assert A.unwitnessed_kept(keyed, g, {}) == {}
    unkeyed = {**bare, "unwitnessed": {"P.sound": h}}
    assert A.witness_failures(unkeyed, f, WOWN)[0].startswith(
        "witnesses: `unwitnessed` lists P.sound without `unfolded_hash`, and its statement is the definition "
        "P.Guarantees.G, whose body its type hash doesn't cover, so the entry can't show the statement is the one it was "
        "listed with, and the rule holds it to witnesses (it reads the claim P.Holds), so it carries them now: prove "
        "`P.sound_honest")
    assert witness_codes(unkeyed, g) == ["witnesses.unwitnessed-changed"]
    assert A.unwitnessed_kept(unkeyed, f, {}) == A.unwitnessed_kept(unkeyed, g, {}) == {}
    fails, out = A.judge({**pol, "witnesses": {}, "unwitnessed": {"P.sound": h}}, g, FILES, update=True)
    assert [x.code for x in fails if x.code.startswith("witnesses")] == ["witnesses.unwitnessed-changed"]
    assert "P.sound" not in out["unwitnessed"]
    before = {**bare, "unwitnessed": {"P.sound": u}}
    assert witness_codes(before, f) == [] and A.unwitnessed_kept(before, f, {}) == keyed["unwitnessed"]
    assert witness_codes(before, g) == ["witnesses.unwitnessed-changed"] and A.unwitnessed_kept(before, g, {}) == {}
    half = {"type_hash": h, "missing": "nontrivial", "why": "PRF-212: not yet closed"}
    honest_only = {**pol, "witnesses": {"P.sound": {"honest": ["P.sound_honest"]}}, "unwitnessed": {"P.sound": half}}
    assert witness_codes(honest_only, f) == ["witnesses.unwitnessed-changed"]
    assert A.unwitnessed_kept(honest_only, f, {}) == {}
    half_before = {**honest_only, "unwitnessed": {"P.sound": half | {"type_hash": u}}}
    assert witness_codes(half_before, f) == []
    assert A.unwitnessed_kept(half_before, f, {}) == {"P.sound": half | {"unfolded_hash": u}}
    keyed_half = {**honest_only, "unwitnessed": {"P.sound": half | {"unfolded_hash": u}}}
    assert witness_codes(keyed_half, f) == []
    assert witness_codes(keyed_half, g) == ["witnesses.unwitnessed-changed"]
    assert witness_codes({**bare, "unwitnessed": {"P.sound": {"type_hash": h, "unfolded_hash": 3}}}, f) == ["witnesses.form"]


def test_a_premise_witnesses_assumed_lists_is_taken_and_every_other_premise_is_concluded():
    """`witnesses_assumed` ({named assumption: why}) lists the assumptions a witness takes as it stands rather than
    concluding, as it takes one of an `assumed_module`: the guarantee's honest witness needn't conclude a premise headed by
    one, and a witness may take one, as a premise or as a closed hypothesis applied at its instance. Every other premise is
    still concluded, one listed under `assumptions` too. An entry no lock-listed guarantee takes as a premise fails as
    stale, one that is no listed assumption fails, and so does one without its reason; `--moved` renames an entry."""
    hiding = {"binder": "hh", "head": "P.Hiding", "module": "P.Defs"}
    why = "hiding at the pinned key: a computational assumption no theorem concludes"
    both = {"premises": [COMPILED, hiding]}
    pol, f = witnessed(both)
    listed = {**pol, "assumptions": {**pol["assumptions"], "P.Compiled": {"owner": "@lean"}, "P.Hiding": {"owner": "@lean"}}}
    assert A.witness_failures(listed, f, WOWN)[0].startswith(
        "witnesses: P.sound's `honest` witnesses (P.sound_honest) conclude nothing of P.Hiding at the deployed instance")
    taken = {**listed, "witnesses_assumed": {"P.Hiding": why}}
    assert A.honest_premises(f["pins"]["P.sound"], taken, WOWN) == ["P.Compiled"]
    assert witness_codes(taken, f) == []
    other = witnessed(both, honest={"body": wbody("P.Other")})[1]
    assert A.witness_failures(taken, other, WOWN)[0].startswith(
        "witnesses: P.sound's `honest` witnesses (P.sound_honest) conclude nothing of P.Compiled at the deployed instance")

    applied = {"hypotheses": [{"binder": "hh", "type": "P.Hiding P.deploy", "closed": True, "named": None}],
               "premises": [hiding]}
    assert witness_codes(*witnessed(honest=applied)) == ["witnesses.honest-form"] * 2
    assert witness_codes(*witnessed(honest=applied, witnesses_assumed={"P.Hiding": why})) == []
    mixed = {"hypotheses": applied["hypotheses"], "premises": [hiding, {**hiding, "head": "P.Other"}]}
    assert witness_codes(*witnessed(honest=mixed, witnesses_assumed={"P.Hiding": why})) == ["witnesses.honest-form"] * 2

    records = {a: {} for a in listed["assumptions"]}
    assert A.assumed_failures(taken, f, records) == [] and A.assumed_failures(listed, f, records) == []
    assert A.assumed_failures({**taken, "witnesses_assumed": {"P.Gone": why}}, f, records) == [
        "witnesses: `witnesses_assumed` lists P.Gone, which no guarantee the lock lists takes as a premise; remove the entry"]
    assert A.assumed_failures(taken, f, {"P.Hyp": {}}) == [
        "witnesses: `witnesses_assumed` lists P.Hiding, which is no premise listed under `assumptions` or "
        "`open`: a witness takes only a listed one as it stands"]
    for bad in ({"P.Hiding": " "}, ["P.Hiding"]):
        assert [x.code for x in A.assumed_failures({**taken, "witnesses_assumed": bad}, f, records)] == \
            ["witnesses.assumed-form"]
    assert A.renamed_witnesses({"witnesses_assumed": {"P.Hiding": why}}, {"P": "Q"}) == {
        "witnesses_assumed": {"Q.Hiding": why}}


def test_a_domain_hypothesis_says_what_it_restricts_and_the_honest_witness_concludes_it():
    """A premise that is part of what its guarantee says is listed under `domain` with `why`, one line, and no issue; the
    guarantee's record lists it under `premises.domain`. Where the guarantee has honest witnesses, they conclude it at the
    deployed instance, or it fails until it is made open (an update with `open_seed` makes it so), and a witness takes
    none as it stands."""
    seed = {"open_seed": {"P.Defs": {"issue": "LEAN-1", "due": "2026-11-30"}}}
    pol, f = witnessed()
    opened = A.judge(pol | seed, f, FILES, update=True)[1]
    entry = {k: v for k, v in opened["open"]["P.Compiled"].items() if k not in ("issue", "due")}
    scope = {"why": "the compiled circuits the bound is stated over"}
    hand = {k: v for k, v in opened.items() if k != "open"} | {"domain": {"P.Compiled": entry | scope}}
    assert [x.code for x in A.judge(hand, f, FILES)[0]] == ["guarantees.premises-changed"]
    dom = A.judge(hand, f, FILES, update=True)[1]
    assert A.judge(dom, f, FILES)[0] == [] and dom["domain"] == hand["domain"] and "open" not in dom
    assert dom["guarantees"]["P.sound"]["premises"] == {"assumptions": ["P.Hyp"], "domain": ["P.Compiled"], "open": [],
                                                           "reads": []}
    for bad in ({}, {"why": " "}, {"why": "two\nlines"}):
        assert [x.code for x in A.judge(dom | {"domain": {"P.Compiled": entry | bad}}, f, FILES)[0]] == ["domain.form"]

    other = witnessed(honest={"body": wbody("P.Other")})[1]
    uncovered = [x for x in A.judge(dom, other, FILES)[0] if x.code.startswith("domain.")]
    assert uncovered == [
        "domain: P.Compiled is a domain hypothesis of P.sound, whose honest witnesses (P.sound_honest) don't conclude it "
        "at the deployed instance, so it is not shown to hold there: make it `open` with the issue that discharges it and "
        "its due date, or have an honest witness conclude it"]
    assert not A.RECORD_DIFFERENCE.match(uncovered[0]), "open work takes a person: its issue"
    moved = A.judge(dom | seed, other, FILES, update=True)[1]
    assert moved["domain"] == {} and moved["open"]["P.Compiled"] == opened["open"]["P.Compiled"]

    takes = {"hypotheses": [{"binder": "hc", "type": "P.Compiled", "closed": True, "named": "P.Compiled",
                             "module": "P.Defs"}]}
    assert "witnesses.honest-form" not in witness_codes(opened, witnessed(honest=takes)[1])
    assert "witnesses.honest-form" in witness_codes(dom, witnessed(honest=takes)[1]), "it concludes a domain hypothesis"


def test_an_honest_witness_concludes_a_premise_it_states_applied_at_the_arguments_the_guarantee_fixes():
    """An honest witness whose statement is the premise applied at the deployed instance (`P.Compiled P.deploy`) concludes
    it, as `Facts.lean` reads its type, also where that statement is a definition the witness rule reads lifted (`lifted`,
    unfolded once past its head): a domain hypothesis it is stays covered, and the honest witness concludes the premise.
    Where the guarantee fixes an argument (`P.Compiled P.circA`), the witness concludes it only where `Facts.lean` found its
    conjunct at that argument (`agree`, by `isDefEq`): not where it found another, nor where it says nothing."""
    pol, f = witnessed()
    seed = {"open_seed": {"P.Defs": {"issue": "LEAN-1", "due": "2026-11-30"}}}
    opened = A.judge(pol | seed, f, FILES, update=True)[1]
    entry = {k: v for k, v in opened["open"]["P.Compiled"].items() if k not in ("issue", "due")}
    hand = {k: v for k, v in opened.items() if k != "open"} | {
        "domain": {"P.Compiled": entry | {"why": "the compiled circuits the bound is stated over"}}}
    dom = A.judge(hand, f, FILES, update=True)[1]
    assert A.judge(dom, f, FILES)[0] == [] and "P.Compiled" in dom["domain"]
    lifted = wbody("LE.le", conclusion=[wk("LE.le", defs=["P.deploy"])])

    def at(taken, agree=None):
        premise = COMPILED | {"args": taken} | ({} if agree is None else {"agree": agree})
        pol, f = witnessed({"premises": [premise]}, honest={"body": wbody("P.Compiled", lifted=lifted)})
        return [x.code for x in A.judge(dom, f, FILES)[0] if x.code.startswith("domain.")], witness_codes(pol, f)

    hp = witnessed(honest={"body": wbody("P.Compiled", lifted=lifted)})[1]["pins"]["P.sound_honest"]
    assert "P.Compiled" not in A.concluded_constants(A.statement_read(hp)), "read lifted, its head is gone"
    assert at([None]) == ([], []), "an argument over the guarantee's binders fixes nothing: the head decides"
    assert at(["CP.circA \nR0"], True) == ([], [])
    for agree in (False, None):
        assert at(["CP.circA \nR0"], agree) == (["domain.uncovered"], ["witnesses.honest-uncovered"]), agree
    assert A.honest_witnesses(pol) == {"P.sound": ["P.sound_honest"]}


def test_another_listed_guarantee_s_statement_is_a_domain_hypothesis_citing_it():
    """A premise that is the `Guarantees` definition another listed guarantee is lifted into is a domain hypothesis citing
    that guarantee (`cites`), which an update writes; an entry citing anything else fails."""
    now, pol = cited()
    assert pol["domain"] == {"P.Guarantees.U": {"owner": "@lean", "digest": pol["reads"]["P.Guarantees"]["definitions"][
        "P.Guarantees.U"], "cites": "P.u", "why": "P.u's statement, another listed guarantee"}}
    assert pol["guarantees"]["P.t"]["premises"]["domain"] == ["P.Guarantees.U"] and "open" not in pol
    assert A.judge(pol, now, FILES)[0] == []
    wrong = pol | {"domain": {"P.Guarantees.U": pol["domain"]["P.Guarantees.U"] | {"cites": "P.t"}}}
    assert A.judge(wrong, now, FILES)[0] == [
        "domain: P.Guarantees.U's entry cites 'P.t', which is no listed guarantee whose statement is P.Guarantees.U (P.u's "
        "is): cite that guarantee, or drop `cites`"]
    assert not A.RECORD_DIFFERENCE.match(A.judge(wrong, now, FILES)[0][0])


def test_a_conclusion_s_antecedent_the_package_defines_is_named_in_the_record_and_no_witness_is_held_to_it():
    """A premise `premisesOf` reads only as an antecedent of a conclusion, headed by the package's definition (`antecedent`:
    `… ∧ (TapOpens … → bound)`), is named in the guarantee's record (`judge`), and no witness is held to it
    (`stated_premises`): the honest witness needn't conclude it and the nontrivial one may rest on it. The same premise
    stated as the guarantee's hypothesis is concluded, and rested on only through `witnesses_assumed`, as before."""
    tap = {"binder": "ht", "head": "P.TapOpens", "module": "P.Defs"}
    ante = tap | {"antecedent": True}
    pol, f = witnessed({"premises": [COMPILED, ante]})
    assert A.stated_premises(f["pins"]["P.sound"]) == [COMPILED]
    assert A.honest_premises(f["pins"]["P.sound"], pol, WOWN) == ["P.Compiled"]
    assert witness_codes(pol, f) == []
    assert witness_codes(*witnessed({"premises": [COMPILED, ante]}, nontrivial={"premises": [ante]})) == []
    held = witnessed({"premises": [COMPILED, tap]})
    assert A.honest_premises(held[1]["pins"]["P.sound"], held[0], WOWN) == ["P.Compiled", "P.TapOpens"]
    assert A.witness_failures(*held, WOWN)[0].startswith(
        "witnesses: P.sound's `honest` witnesses (P.sound_honest) conclude nothing of P.TapOpens at the deployed instance")
    assert witness_codes(*witnessed(nontrivial={"premises": [tap]})) != []


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
    hard = {"P.t": f["pins"]["P.t"] | {"inline": [premise("P.Hard n", "P.Hard")]}}
    assert A.inline_failures(pol, facts(pins=f["pins"] | hard))[2] == {"P.t": ["P.Hard n"]}
    hard["P.t"]["reads"] = [{"name": "P.Hard", "module": "P.Assumptions"}]
    assert A.inline_failures(pol, facts(pins=f["pins"] | hard))[2] == {}, \
        "an assumptions module's Prop the statement takes inside a definition, which its `reads` record holds"


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
                        "numerals); name it as a Prop and list it in the lock (under `open`, with the issue that "
                        "discharges it and its due date), or state it as a lock-listed guarantee's conclusion; or, where a merge or an older tree's `--update` dropped an "
                        "entry the lock had for it, put that entry back as the earlier lock has it")
    assert A.INLINE_FIX in fails[0]
    assert A.inline_failures(pol, f)[2] == {"P.t": ["P.f n = 3", "x ≤ 2 * y", "P.Guarantees.T", "P.Unlisted n", "1 = 1"]}
    unread = facts(pins=f["pins"] | {"P.t": f["pins"]["P.t"] | {"inline": None}})
    assert inline_codes(pol, unread) == ["guarantees.inline-unread"]
    assert inline_codes(pol, facts(pins={k: {x: y for x, y in v.items() if x != "inline"} for k, v in f["pins"].items()})) \
        == [], "facts of an older audit tool, without `inline`"


def test_update_keeps_a_held_guarantees_inline_entry_while_its_statement_prints_it():
    """A guarantee a witness entry, an unfrozen property or a generic claim holds keeps through `--update` each
    `shape.inline` entry its statement still prints, and drops the rest, so the check without `--update` passes on the
    updated lock, and still does once a merge leaves the guarantee unheld (its witness entry or claim gone, its property
    frozen). The lock `--update` wrote used to lack them, and every records PR put them back by hand."""
    why = "inline when the rule came"
    pol, f = inlined(premise("P.f n = 3", "Eq"), premise("P.k n = 6", "Eq"))
    inline = {"P.t": {"P.f n = 3": why, "P.k n = 6": why, "P.h n = 5": why}}
    witness = {"witnesses": {"P.SecurityProofs.u": {"honest": ["P.t"], "nontrivial": "P.n"}}}
    prop = {"function": "P.f", "outcome": "none"}
    for lock, unheld in ((pol | witness | {"shape": {"inline": inline}}, lambda p: p | {"witnesses": {}}),
                         (pol | {"shape": {"properties": {"P.t": prop}, "inline": inline}},
                          lambda p: p | {"shape": p["shape"] | {"frozen": {"P.t": "frozen: open"}}}),
                         (pol | {"claims": {"generic": {"P.C": "P.t"}}, "shape": {"inline": inline}},
                          lambda p: p | {"claims": {}})):
        fails, out = A.judge(lock, f, FILES, update=True)
        assert [x for x in fails if "inline" in x.code] == []
        assert out["shape"]["inline"] == {"P.t": {"P.f n = 3": why, "P.k n = 6": why}}
        assert inline_codes(out, f) == [] and inline_codes(unheld(out), f) == []
        assert inline_codes(unheld(lock | {"shape": lock["shape"] | {"inline": {}}}), f) == [
            "guarantees.inline-hypothesis"]


def test_a_grandfathered_inline_hypothesis_passes_and_the_list_only_shrinks():
    """`shape.inline` lists the offenders when the rule came, keyed by their text with its whitespace as one space: they
    pass, a new one fails, and `--update` drops each entry whose premise no longer offends (all of a guarantee's, its key
    with them, and the map once empty) and adds none; `--moved`
    carries an entry to its guarantee's new name. A property `shape.properties` holds is the shape rule's; a frozen one
    is judged here. A witness `witnesses` or `claims.witnesses` names is that rule's: it neither fails nor offends, and
    `--update` keeps its entries while their premises still offend. An entry with no reason fails."""
    why = "inline when the rule came"
    pol, f = inlined(premise("P.f n\n    = 3", "Eq"), premise("P.g n = 4", "Eq"))
    allow = {"P.t": {"P.f n = 3": why, "P.g n = 4": why, "P.h n = 5": why}}
    held = pol | {"shape": {"premises": ["P.Assumptions"], "inline": allow}}
    assert inline_codes(held, f) == []
    fails, out = A.judge(held, f, FILES, update=True)
    assert [x for x in fails if "inline" in x.code] == []
    assert out["shape"] == {"premises": ["P.Assumptions"], "inline": {"P.t": {"P.f n = 3": why, "P.g n = 4": why}}}
    grown = inlined(premise("P.f n = 3", "Eq"), premise("P.k n = 6", "Eq"))[1]
    fails = [x for x in A.judge(held, grown, FILES)[0] if "inline" in x.code]
    assert [x.code for x in fails] == ["guarantees.inline-hypothesis"] and fails[0].startswith(
        "guarantees: P.t takes (P.k n = 6): an inline hypothesis")
    assert A.judge(held, grown, FILES, update=True)[1]["shape"]["inline"]["P.t"] == {"P.f n = 3": why}
    fixed = inlined(premise("P.Fits n", "P.Fits"))[1]
    assert A.judge(held, fixed, FILES, update=True)[1]["shape"] == {"premises": ["P.Assumptions"]}
    alone = pol | {"shape": {"inline": {"P.t": {"P.f n = 3": why}}}}
    assert A.judge(alone, fixed, FILES, update=True)[1]["shape"] == {}
    assert A.inline_failures(held, grown)[1:] == ({"P.t": {"P.f n = 3": why}}, {"P.t": ["P.f n = 3", "P.k n = 6"]})
    assert A.renamed_witnesses({"shape": held["shape"]}, {"P.t": "P.SecurityProofs.T"}) == {"shape": {
        "premises": ["P.Assumptions"], "inline": {"P.SecurityProofs.T": allow["P.t"]}}}
    for named in ({"witnesses": {"P.SecurityProofs.u": {"honest": ["P.t"], "nontrivial": "P.n"}}},
                  {"claims": {"modules": ["P.Claims"], "witnesses": {"P.C": {"fails": "P.n", "meets": ["P.t"]}}}}):
        assert A.inline_failures(held | named, grown) == ([], {"P.t": {"P.f n = 3": why}}, {})
    prop = {"function": "P.f", "outcome": "none"}
    assert inline_codes(pol | {"shape": {"properties": {"P.t": prop}}}, grown) == []
    assert inline_codes(pol | {"shape": {"properties": {"P.t": prop}, "frozen": {"P.t": "frozen: open"}}}, grown) == \
        ["guarantees.inline-hypothesis"]
    assert inline_codes(pol | {"shape": {"inline": {"P.t": {"P.f n = 3": " "}}}}, grown) == ["shape.inline-form"]
    assert inline_codes(pol | {"shape": {"inline": ["P.t"]}}, grown) == ["shape.inline-form"]


WHY = "inline when the rule came"


def renamed_inline(was, now, consts, constants, moves=None, fields=None):
    """(policy, facts): `P.t`'s `shape.inline` lists ``was``, and its build takes ``now`` instead, naming ``consts`` and
    showing ``fields`` ([object, field, constant] each); the build holds ``constants`` ({name: its kind, or (its kind, the
    text its SHA-256 is of)}) and was read with ``moves``."""
    pol, f = inlined(premise(now, "Eq") | {"consts": consts, "fields": fields or []})
    table = {n: {"kind": k} if isinstance(k, str) else {"kind": k[0], "sha256": sha(k[1])} for n, k in constants.items()}
    return pol | {"shape": {"inline": {"P.t": {was: WHY}}}}, f | {"constants": table, "moves": moves or {}}


def test_an_inline_entry_of_a_guarantee_the_lock_does_not_list_fails_until_update_drops_it():
    """A `shape.inline` entry whose guarantee the lock doesn't list (one removed or renamed without `--moved`) would
    otherwise stay forever: it fails as a record difference, and `--update` drops it and keeps the rest."""
    why = "inline when the rule came"
    pol, f = inlined(premise("P.f n = 3", "Eq"))
    held = pol | {"shape": {"inline": {"P.t": {"P.f n = 3": why}, "P.gone": {"x = y": why}}}}
    fails = [x for x in A.judge(held, f, FILES)[0] if "inline" in x.code]
    assert [(x.code, x.entry) for x in fails] == [("shape.inline-unlisted", "P.gone")]
    assert fails[0].endswith("run `check.py --update`, which drops the entry") and A.RECORD_DIFFERENCE.match(fails[0])
    fails, out = A.judge(held, f, FILES, update=True)
    assert [x for x in fails if "inline" in x.code] == []
    assert out["shape"] == {"inline": {"P.t": {"P.f n = 3": why}}}
    assert A.regenerated(held, f, FILES, [fails[0] for fails in [[x for x in A.judge(held, f, FILES)[0]
                                                                   if "inline" in x.code]]])["shape"] == out["shape"]


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


#: the move of `Verity.Circuit.Transcript` to `Verity.Circuit.Trace`, as `rename.py --moves` printed it
TRACE_MOVES = {"Claims.Recursion.abortTrace": "Claims.Recursion.abortTranscript", "Claims.Trace": "Claims.Transcript",
               "Verity.Circuit.Transcript": "Verity.Circuit.Trace"}
NCI_CONSTS = ["Eq", "Exists", "Membership.mem", "Nci.admissible", "Nci.values", "SetLike", "SetLike.instMembership"]
NCI_CONSTANTS = {"Nci.admissible": ("def", "admissible"), "Nci.values": ("def", "values"), "Eq": "inductive"}


def test_a_moved_field_on_a_variable_follows_the_move_through_what_the_build_says_it_names():
    """A dotted name that ends in a field on a variable (`D.D.Trace`, `Verity.Circuit.Trace D.D` by field notation) is
    renamed where `--moved` moved the constant the build says that field stands for (`fields`): two premises over a
    trace, as a lock keyed them before `Verity.Circuit.Transcript` moved, take their entries' places with no hand edit,
    and `--update` keys them as printed. Without the move, or with no constant the build reports for the field, each
    fails and `--update` drops it."""
    for d in "Dd":
        was = f"∀ (a : {d}.D.Transcript), SetLike.instMembership.mem (Nci.admissible {d} θv) a → Exists fun s => Eq " \
            "(Nci.values Sst a) (f s)"
        now = was.replace(f"{d}.D.Transcript", f"{d}.D.Trace")
        fields = [[d, "D", "Verity.RegisteredCircuit.D"], [f"{d}.D", "Trace", "Verity.Circuit.Trace"]]
        pol, f = renamed_inline(was, now, NCI_CONSTS, NCI_CONSTANTS, TRACE_MOVES, fields)
        assert inline_codes(pol, f) == [], was
        assert A.inline_failures(pol, f)[1:] == ({"P.t": {now: WHY}}, {"P.t": [now]})
        assert A.judge(pol, f, FILES, update=True)[1]["shape"] == {"inline": {"P.t": {now: WHY}}}
        for moves, shown in [(None, fields), (TRACE_MOVES, fields[:1]), (TRACE_MOVES, [])]:
            pol, f = renamed_inline(was, now, NCI_CONSTS, NCI_CONSTANTS, moves, shown)
            assert inline_codes(pol, f) == ["guarantees.inline-hypothesis"], (was, moves, shown)
            assert A.inline_failures(pol, f)[1] == {}, (was, moves, shown)


def test_a_field_of_another_structure_that_shares_the_moved_word_keeps_its_text():
    """A field that shares the moved field's word but stands for another structure's field (`s.Transcript`,
    `P.Log.Transcript`) stays as printed beside the moved one, and a premise where it changed doesn't take the entry's
    place, since nothing moved its own constant: `P.Log.Trace` for `P.Log.Transcript`, a field that stands for two
    constants, or the moved field's new name already in the old text on another object."""
    fields = [["D", "D", "Verity.RegisteredCircuit.D"], ["D.D", "Trace", "Verity.Circuit.Trace"],
              ["s", "Transcript", "P.Log.Transcript"]]
    was, now = "∀ (a : D.D.Transcript), Eq (P.view a) s.Transcript", "∀ (a : D.D.Trace), Eq (P.view a) s.Transcript"
    pol, f = renamed_inline(was, now, ["Eq", "P.view"], {"P.view": ("def", "view")}, TRACE_MOVES, fields)
    assert inline_codes(pol, f) == []
    assert A.judge(pol, f, FILES, update=True)[1]["shape"] == {"inline": {"P.t": {now: WHY}}}
    traced = [["s", "Trace", "P.Log.Trace"]]
    cases = [("Eq (P.view a) s.Transcript", "Eq (P.view a) s.Trace", traced),
             ("Eq (P.view a) s.Transcript", "Eq (P.view a) s.Trace", traced + [["s", "Trace", "Verity.Circuit.Trace"]]),
             ("∀ (a : D.D.Transcript), Eq a E.D.Trace", "∀ (a : D.D.Trace), Eq a E.D.Trace",
              fields + [["E.D", "Trace", "Verity.Circuit.Trace"]])]
    for was, now, shown in cases:
        pol, f = renamed_inline(was, now, ["Eq", "P.view"], {"P.view": ("def", "view")}, TRACE_MOVES, shown)
        assert inline_codes(pol, f) == ["guarantees.inline-hypothesis"], (was, now, shown)
        assert A.inline_failures(pol, f)[1] == {}, (was, now, shown)


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

structure Circ where
  N : Nat

abbrev Circ.Trace (C : Circ) := Fin C.N → Nat

structure Reg where
  D : Circ

structure Run where
  n : Nat

theorem fielded (D : Reg) (s : Run) (h : ∀ a : D.D.Trace, a = a → s.n = 0) : True := trivial

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
    takes its premise's place; one under another declared definition doesn't. Each premise also names the fields it shows
    by field notation, with their objects and constants, so that an entry under a moved field's old name takes its place."""
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

    traced = policy | {"guarantees": policy["guarantees"] | {"Control.fielded": {"owner": "@lean"}},
                       "shape": {"inline": {"Control.fielded": {"∀ (a : D.D.Transcript), Eq a a → Eq s.n 0": WHY}}}}
    f = A.facts_of(pkg, traced, tmp_path, moved={"Control.Circ.Transcript": "Control.Circ.Trace"})
    part = f["pins"]["Control.fielded"]["inline"][0]
    assert part["type"] == "∀ (a : D.D.Trace), Eq a a → Eq s.n 0"
    assert part["fields"] == [["D", "D", "Control.Reg.D"], ["D.D", "Trace", "Control.Circ.Trace"],
                              ["s", "n", "Control.Run.n"]]
    assert A.judge(traced, f, A.lean_modules(pkg), update=True)[1]["shape"]["inline"] == {
        "Control.fielded": {part["type"]: WHY}}


def test_a_reflexive_statement_needs_no_witnesses_and_its_unwitnessed_entry_drops():
    """A guarantee whose statement, its definition unfolded once, concludes `a = a` with the same `Expr` on both sides
    (`reflexive`: the claims tuples' `claims = claims`, a text record whose change reaches the lock's review) is held to
    no witnesses, behind `∀` and `→` binders too, and `--update` drops its `unwitnessed` entry, which until then is stale.
    An `Eq` that `Facts.lean` doesn't report as a reflexivity (`f x = g x` of definitionally equal `f` and `g`), a
    reflexive `↔`, an `∃` before it or a second conjunct keeps it in scope."""
    refl = {"head": "Eq", "defs": ["P.Holds"], "refl": True, "assumed": False}
    stated = wbody("P.Guarantees.Stated", conclusion=[wk("P.Guarantees.Stated")], lifted=wbody("Eq", conclusion=[refl]))
    pol, f = witnessed({"body": stated})
    pin, pol = f["pins"]["P.sound"], {**pol, "witnesses": {}}
    assert A.reflexive(pin) and A.witness_scope("P.sound", pol, pin) is None
    assert witness_codes(pol, f) == []
    listed = {**pol, "unwitnessed": {"P.sound": {"type_hash": pin["type_sha256"], "unfolded_hash": sha("unfolded")}}}
    assert A.witness_failures(listed, f, WOWN) == [
        "witnesses: `unwitnessed` lists P.sound, which the rule doesn't hold to witnesses; remove the entry"]
    assert A.unwitnessed_kept(listed, f, {}) == {}
    behind = witnessed({"body": wbody("Eq", binders=["∀", "→"], conclusion=[refl])})[1]
    assert A.reflexive(behind["pins"]["P.sound"]) and witness_codes(pol, behind) == []
    for lifted in (wbody("Eq", conclusion=[refl | {"refl": False}]), wbody("Iff", conclusion=[refl | {"head": "Iff"}]),
                   wbody("Eq", binders=["∃"], conclusion=[refl]), wbody("And", conclusion=[refl, wk("P.Holds")])):
        kept = witnessed({"body": stated | {"lifted": lifted}})[1]
        assert not A.reflexive(kept["pins"]["P.sound"]) and witness_codes(pol, kept) == ["witnesses.missing"], lifted


REFLEXIVE = """\
namespace Control.Claims

def Holds (n : Nat) : Prop := n < 5

def Other : Prop := True

def f (n : Nat) : Nat := n

def g (n : Nat) : Nat := n

end Control.Claims

namespace Control.Guarantees

def Stated : Prop :=
  let claims := (Control.Claims.Holds, Control.Claims.Other)
  claims = claims

end Control.Guarantees

namespace Control

theorem SecurityProofs.Stated : Guarantees.Stated := rfl

theorem statedAt : ∀ n : Nat, (Claims.Holds n, Claims.Other) = (Claims.Holds n, Claims.Other) := fun _ => rfl

theorem same : ∀ n : Nat, Claims.Holds (Claims.f n) = Claims.Holds (Claims.g n) := fun _ => rfl

end Control
"""


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_reflexive_statement_is_read_syntactically_on_a_real_build(tmp_path):
    """On a real build, `Facts.lean`'s `refl` is syntactic: the claims tuple `let claims := …; claims = claims` behind a
    `Guarantees` definition, and a tuple equal to itself under a `∀`, are reflexivities and need no witnesses, and
    `--update` drops their `unwitnessed` entries; `Holds (f n) = Holds (g n)`, with `f` and `g` definitionally equal (it
    holds by `rfl`) but different constants, stays in scope and keeps its entry."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    (pkg / "Control" / "Claims.lean").write_text(REFLEXIVE)
    (pkg / "Control.lean").write_text((pkg / "Control.lean").read_text() + "import Control.Claims\n")
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    names = ["Control.SecurityProofs.Stated", "Control.statedAt", "Control.same"]
    policy = unclaimed(A.load(pkg))
    policy["guarantees"] = policy["guarantees"] | {n: {"owner": "@lean"} for n in names}
    f = A.facts_of(pkg, policy, tmp_path)
    pins = {n: f["pins"][n] for n in names}
    assert [A.reflexive(p) for p in pins.values()] == [True, True, False]
    assert [A.witness_scope(n, policy, p) for n, p in pins.items()] == [
        None, None, ("acceptance", "it reads the claim Control.Claims.Holds")]
    own = {"Control", "Control.Assumptions", "Control.Claims", "Control.Defs", "Control.Thms"}
    assert [(x.code, x.entry) for x in A.witness_failures(policy, f, own)] == [("witnesses.missing", "Control.same")]
    stated = pins["Control.SecurityProofs.Stated"]
    allow = {"Control.SecurityProofs.Stated": {"type_hash": stated["type_sha256"],
                                               "unfolded_hash": stated["lift"]["unfolded_sha256"]},
             "Control.statedAt": pins["Control.statedAt"]["type_sha256"], "Control.same": pins["Control.same"]["type_sha256"]}
    listed = policy | {"unwitnessed": allow}
    assert [(x.code, x.entry) for x in A.witness_failures(listed, f, own)] == [
        ("witnesses.unwitnessed-stale", "Control.SecurityProofs.Stated"), ("witnesses.unwitnessed-stale", "Control.statedAt")]
    assert A.unwitnessed_kept(listed, f, {}) == {"Control.same": allow["Control.same"]}


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
        "assumption the lock lists: a guarantee carries an honest and a nontrivial witness")
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
        "`nontrivial` lemma's object (as `HonestH200`'s honest period, built with `accepts := fun _ _ => True`, was)"]
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


def test_a_premise_free_exclusion_theorem_is_witnessed_by_its_bound_failing_at_an_explicit_circuit():
    """An exclusion theorem that takes no premise (`premise_free`: training's `ServingBlocks.UncertifiedElementary`, a
    cap at every unit of every refinement) has none to drop: its nontrivial witness is the bare `¬ bound` at an explicit
    circuit, where a tighter bound fails, and a missing witness is asked for in that form. It still negates the bound,
    constructs its circuit and takes only listed assumptions. A hypothesis no constant heads is a premise all the same,
    and a guarantee with a premise still drops one."""
    cover = {"name": "P.Covering.load", "module": "P.Covering", "kind": "def", "hash": "6" * 32}
    g = {"reads": [cover], "premises": [], "hypotheses": [],
         "body": wbody("LE.le", binders=["∀", "∀"], conclusion=[wk("LE.le", ["P.Covering.load"])])}
    bare = {"body": wbody("LE.le", negated=True, conclusion=[{"head": "Not", "defs": [], "refl": False,
                                                              "assumed": False, "negated": True, "inner": "LE.le"}])}
    pol, f = witnessed(g, nontrivial=bare)
    assert A.premise_free(f["pins"]["P.sound"]) and A.witness_scope("P.sound", pol, f["pins"]["P.sound"])[0] == "exclusion"
    assert A.witness_failures(pol, f, WOWN) == []
    missing = A.witness_failures({**pol, "witnesses": {}}, f, WOWN)[0]
    assert "`P.sound_nontrivial : ¬ LE.le c₀ …` (it takes no premise to drop, so a tighter bound fails at an explicit " \
           "circuit c₀)" in missing and '"kind": "exclusion"' in missing, missing
    assert witness_codes(*witnessed(g, nontrivial={"body": wbody("P.Compiled")})) == ["witnesses.nontrivial-form"]
    assert witness_codes(*witnessed(g, nontrivial={"body": bare["body"] | {"binders": ["∀"]}})) == \
        ["witnesses.instance-binders"]
    other = [{"binder": "h", "type": "P.Other", "closed": True, "named": "P.Other", "module": "P.Defs"}]
    assert witness_codes(*witnessed(g, nontrivial={"body": bare["body"] | {"binders": ["→"]}, "hypotheses": other})) == \
        ["witnesses.nontrivial-assumes"]
    unheaded = g | {"hypotheses": [{"binder": "h", "type": "p n", "closed": False}],
                    "body": g["body"] | {"binders": ["∀", "∀", "→"]}}
    assert not A.premise_free(unheaded)
    assert witness_codes(*witnessed(unheaded, nontrivial=bare)) == ["witnesses.nontrivial-drops"]
    premised = g | {"premises": [COMPILED], "body": g["body"] | {"binders": ["∀", "∀", "→"]}}
    assert witness_codes(*witnessed(premised, nontrivial=bare)) == ["witnesses.nontrivial-drops"]


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


def wnot(inner):
    """A `¬ P …` conjunct of a statement's conclusion, as `Facts.lean`'s `conclusionOf` reports it."""
    return {"head": "Not", "defs": [], "refl": False, "assumed": False, "negated": True, "inner": inner}


def further(*lemmas, guarantee=None, **over):
    """(policy, facts) of `witnessed`'s `P.sound` (its premises `P.Hyp` and `P.Compiled`), each part overridden by
    ``guarantee`` and ``over``, whose `nontrivial` lists `P.sound_nontrivial` and then each of ``lemmas`` ((name, its
    facts)), every one a listed guarantee."""
    pol, f = witnessed(guarantee, **over)
    pins = f["pins"] | {n: fact(hyps=[]) | p for n, p in lemmas}
    return ({**pol, "guarantees": {n: {"owner": "@lean"} for n in pins},
             "witnesses": {"P.sound": {"honest": ["P.sound_honest"],
                                       "nontrivial": ["P.sound_nontrivial", *(n for n, _ in lemmas)]}}},
            facts(pins=pins, own_modules=sorted(WOWN)))


def test_a_guarantee_may_list_further_nontrivial_witnesses_each_dropping_another_premise():
    """`nontrivial` is one theorem or a list of them (a network guarantee's rows and clock clauses, each with an exclusion
    lemma and a `Wf` negation). The first is held to the guarantee's kind as ever; each after it, whatever the kind, is in the
    exclusion form at an explicit instance, (the other premises hold) ∧ ¬ (a named premise) ∧ ¬ (the conclusion), drops
    a premise no earlier one drops, builds its instance and takes only listed assumptions. Every listed name is a witness,
    held to neither `shape` nor `four_names`, and a list counts as the `nontrivial` half an `unwitnessed` entry would
    exempt."""
    no_compiled = ("P.sound_compiled", {"body": wbody("And", conclusion=[wnot("P.Compiled"), wk("P.Hyp"),
                                                                          wnot("P.Holds")])})
    no_hyp = ("P.sound_hyp", {"body": wbody("And", conclusion=[wk("P.Compiled"), wnot("P.Hyp"), wnot("P.Holds")])})
    pol, f = further(no_compiled, no_hyp)
    assert A.witness_failures(pol, f, WOWN) == []
    assert A.claim_witnesses(pol) == {"P.sound_honest", "P.sound_nontrivial", "P.sound_compiled", "P.sound_hyp"}
    assert [x.entry for x in A.judge({**pol, "four_names": ["P"]}, f, FILES)[0] if x.code == "four-names"] == ["P.sound"]
    assert witness_codes(*witnessed()) == []
    assert witness_codes({**witnessed()[0], "witnesses": {"P.sound": {"honest": ["P.sound_honest"],
                                                                      "nontrivial": ["P.sound_nontrivial"]}}},
                         witnessed()[1]) == []
    for bad in ([], ["P.sound_nontrivial", 3], {"P.sound_nontrivial": 1}):
        shaped = {**pol, "witnesses": {"P.sound": {**pol["witnesses"]["P.sound"], "nontrivial": bad}}}
        assert A.witness_failures(shaped, f, WOWN) == [
            "witnesses: witnesses[P.sound] is {honest: [theorem], nontrivial: theorem or [theorem, …], kind: acceptance "
            "or exclusion (optional)}"]

    accept_all = ("P.sound_again", {"body": wbody("P.Holds", negated=True)})
    assert A.witness_failures(*further(accept_all), WOWN) == [
        "witnesses: P.sound_again does not state that P.sound's bound fails at an explicit circuit: one conjunct of its "
        "conclusion is `¬ C …` with C one of P.Holds, and it negates nothing",
        "witnesses: P.sound_again drops no premise of P.sound: each further nontrivial lemma states, at an explicit "
        "circuit, (each other premise holds) ∧ ¬ (the dropped premise) ∧ ¬ (the bound), one conjunct `¬ P …` with P one "
        "of P.Compiled, P.Hyp; a bare `¬ bound` reads none"]
    holds = ("P.sound_holds", {"body": wbody("And", conclusion=[wnot("P.Compiled"), wk("P.Hyp")])})
    assert witness_codes(*further(holds)) == ["witnesses.nontrivial-form"]
    others = ("P.sound_others", {"body": wbody("And", conclusion=[wnot("P.Compiled"), wnot("P.Holds")])})
    assert witness_codes(*further(others)) == ["witnesses.nontrivial-premises"]

    again = ("P.sound_compiled2", no_compiled[1])
    assert A.witness_failures(*further(no_compiled, again), WOWN) == [
        "witnesses: P.sound_compiled2, P.sound's `nontrivial` witness 3, drops only P.Compiled, already dropped by "
        "P.sound_compiled: each further nontrivial lemma drops a premise of P.sound that no earlier one drops, so each "
        "shows another premise matters; drop one of P.Hyp in P.sound_compiled2 instead, or remove it from "
        "witnesses[P.sound].nontrivial"]
    assert A.witness_failures(*further(no_compiled, no_hyp, again), WOWN)[0].endswith(
        "every premise is dropped already, so remove it from witnesses[P.sound].nontrivial")
    both = ("P.sound_both", {"body": wbody("And", conclusion=[wnot("P.Compiled"), wnot("P.Hyp"), wnot("P.Holds")])})
    assert witness_codes(*further(no_compiled, both)) == []

    ranges = ("P.sound_ranges", no_compiled[1] | {"body": no_compiled[1]["body"] | {"binders": ["∀"]}})
    assert witness_codes(*further(ranges)) == ["witnesses.instance-binders"]
    takes = ("P.sound_takes", no_compiled[1] | {"hypotheses": [{"binder": "hc", "type": "P.Compiled c", "closed": False}]})
    assert witness_codes(*further(takes)) == ["witnesses.nontrivial-assumes"]
    unlisted = further(no_compiled)
    unlisted[0]["guarantees"].pop("P.sound_compiled")
    assert witness_codes(*unlisted) == ["witnesses.unlisted-witness"]

    h = f["pins"]["P.sound"]["type_sha256"]
    half = {"type_hash": h, "missing": "nontrivial", "why": "the clock clause's Wf negation"}
    assert witness_codes({**pol, "unwitnessed": {"P.sound": half}}, f) == []
    assert A.unwitnessed_kept({**pol, "unwitnessed": {"P.sound": half}}, f, {}) == {}
    assert A.renamed_witnesses({"witnesses": {"P.a": {"honest": ["P.a_h"], "nontrivial": ["P.a_n", "P.a_m"]}}},
                               {"P": "Q"}) == {"witnesses": {"Q.a": {"honest": ["Q.a_h"], "nontrivial": ["Q.a_n", "Q.a_m"]}}}


def test_an_exclusion_theorems_further_nontrivial_witness_drops_another_premise_than_the_first():
    """At an exclusion theorem the first `nontrivial` lemma drops a premise too, and a further one that drops only the
    same premise is a repeat."""
    cover = {"name": "P.Covering.load", "module": "P.Covering", "kind": "def", "hash": "6" * 32}
    covers = {"binder": "hc", "type": "P.Covers", "closed": True, "named": "P.Covers", "module": "P.Assumptions"}
    g = {"reads": [cover], "premises": [COMPILED, {"binder": "hc", "head": "P.Covers", "module": "P.Assumptions"}],
         "hypotheses": [covers],
         "body": wbody("LE.le", binders=["∀", "→", "→"], conclusion=[wk("LE.le", ["P.Covering.load"])])}
    beats = {"body": wbody("And", conclusion=[wk("P.Compiled"), wnot("P.Covers"), wnot("LE.le")])}
    uncompiled = {"body": wbody("And", conclusion=[wnot("P.Compiled"), wnot("LE.le")]), "hypotheses": [covers]}
    pol, f = further(("P.sound_uncompiled", uncompiled), guarantee=g, nontrivial=beats)
    pol["assumptions"] = {**pol["assumptions"], "P.Covers": {"owner": "@network-accounting"}}
    assert A.witness_failures(pol, f, WOWN) == []
    pol, f = further(("P.sound_uncovered", beats), guarantee=g, nontrivial=beats)
    pol["assumptions"] = {**pol["assumptions"], "P.Covers": {"owner": "@network-accounting"}}
    assert witness_codes(pol, f) == ["witnesses.nontrivial-repeat"]
    assert "already dropped by P.sound_nontrivial" in A.witness_failures(pol, f, WOWN)[0]


def test_a_premise_free_guarantee_lists_no_further_nontrivial_witness():
    """A premise-free exclusion theorem's one `nontrivial` lemma is the bare `¬ bound`, and a further one drops a premise
    like any other, so with none to drop it fails."""
    cover = {"name": "P.Covering.load", "module": "P.Covering", "kind": "def", "hash": "6" * 32}
    g = {"reads": [cover], "premises": [], "hypotheses": [],
         "body": wbody("LE.le", binders=["∀", "∀"], conclusion=[wk("LE.le", ["P.Covering.load"])])}
    bare = {"body": wbody("LE.le", negated=True, conclusion=[wnot("LE.le")])}
    assert witness_codes(*witnessed(g, nontrivial=bare)) == []
    assert witness_codes(*further(("P.sound_tighter", bare), guarantee=g, nontrivial=bare)) == \
        ["witnesses.nontrivial-drops"]


def test_a_named_assumption_may_sit_in_a_module_under_assumptions_module():
    """`assumptions_module` names a module and the modules under it (`Definitions.Assumptions.Pouw`), so an area can keep its
    named assumptions in a submodule of its own; a module that only shares the name's prefix (`P.AssumptionsX`) is not one."""
    def codes(module):
        f = facts(pins={"P.t": fact(reads=[{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "1" * 32},
                                           {"name": "P.Hyp", "module": module, "kind": "def", "hash": "3" * 32}])})
        return [x.code for x in A.judge(POLICY, f, FILES, update=True)[0]]
    named = {"assumptions.outside-module", "open.unlisted"}
    assert not named & set(codes("P.Assumptions.Pouw"))
    assert named & set(codes("P.AssumptionsX"))


def test_assumptions_module_may_list_several_homes():
    """`assumptions_module` may list modules (`security/Properties` lists `Properties.Assumptions` beside the spec's, for an
    assumption stated over the code, which the spec can't import): one declared in any of them is a named assumption, one
    declared outside them all fails naming each, and every listed home is a module the build has. The string form is
    one home, as the one-element list."""
    def sound(module):
        return facts(pins={"P.t": fact(
            signature="P.t (h : P.Hyp) (s : P.Sound) (n : Nat) : P.f n = n",
            hyps=fact()["hypotheses"] + [{"binder": "s", "type": "P.Sound", "closed": True, "named": "P.Sound",
                                          "module": module}],
            reads=fact()["reads"] + [{"name": "P.Sound", "module": module, "kind": "def", "hash": "5" * 32}])})
    two = {**POLICY, "assumptions_module": ["P.Assumptions", "P.Defs"],
           "assumptions": POLICY["assumptions"] | {"P.Sound": {"owner": "@lean"}}}
    fails, pol = A.judge(two, sound("P.Defs"), FILES, update=True)
    assert fails == [] and set(pol["assumptions"]) == {"P.Hyp", "P.Sound"}
    assert pol["guarantees"]["P.t"]["premises"]["assumptions"] == ["P.Hyp", "P.Sound"]
    assert A.judge(pol, sound("P.Defs"), FILES)[0] == []
    outside = [x for x in A.judge(pol | {"assumptions_module": ["P.Assumptions", "P.Thms"]}, sound("P.Defs"), FILES)[0]
               if x.code == "assumptions.outside-module"]
    assert outside == ["assumptions: P.Sound is declared in P.Defs, not in P.Assumptions or P.Thms, where a named "
                       "assumption is declared: move its definition there if it is a real one, or move its entry to "
                       "`open` with the issue that discharges it and its due date, or to `domain` with a one-line `why` "
                       "if it is part of what the guarantee says"]
    one = A.judge(POLICY, facts(), FILES, update=True)
    assert A.judge({**POLICY, "assumptions_module": ["P.Assumptions"]}, facts(), FILES, update=True)[0] == one[0] == []
    assert A.judge({**POLICY, "assumptions_module": ["P.Assumptions"]}, facts(), FILES, update=True)[1] == \
        one[1] | {"assumptions_module": ["P.Assumptions"]}
    missing = A.judge({**POLICY, "assumptions_module": ["P.Assumptions", "P.Nowhere"]}, facts(), FILES, update=True)[0]
    assert [x for x in missing if x.code == "assumptions.module"] == [
        "assumptions: `assumptions_module` names 'P.Nowhere', which neither the build nor the packages it requires has a "
        "module of; name the modules the named assumptions are declared in"]
    for bad in ([], {"P.Assumptions": 1}, ["P.Assumptions", 2]):
        assert "assumptions.module-form" in [x.code for x in A.judge({**POLICY, "assumptions_module": bad}, facts(),
                                                                       FILES, update=True)[0]]


def test_an_assumption_is_listed_by_name_with_its_record():
    """Each assumption has an owner and the digest of its definition as `reads` holds it, so a changed assumption is its own
    record; an entry no guarantee reads fails; an update converts a list of assumption modules into the definitions read
    there, owned by the owner of the guarantees that read them. A named assumption is declared in `assumptions_module`,
    a module the build or a package it requires has (`security/Properties` names `security`'s, which none of its modules
    may import): one declared elsewhere fails until a person moves it or lists it as open, and one premise listed under
    both kinds fails."""
    assert [x.code for x in A.judge({**POLICY, "assumptions_module": "P.Nowhere"}, facts(), FILES, update=True)[0]] == [
        "assumptions.module", "open.unlisted"]
    assert "assumptions.module" not in [x.code for x in A.judge({**POLICY, "assumptions_module": "P.Nowhere"},
                                                                facts() | {"source_modules": ["P.Nowhere"]}, FILES,
                                                                update=True)[0]]
    pol = recorded()
    assert pol["assumptions"] == {"P.Hyp": {"owner": "@lean", "digest": pol["reads"]["P.Assumptions"]["definitions"]["P.Hyp"]}}
    changed = facts(pins={"P.t": fact(reads=[{"name": "P.f", "module": "P.Defs", "kind": "def", "hash": "1" * 32},
                                             {"name": "P.Hyp", "module": "P.Assumptions", "kind": "def", "hash": "4" * 32}])})
    assert "assumptions: P.Hyp's definition in P.Assumptions changed; run `check.py --update` and commit the record " \
           "(the landing message shows the change)" in A.judge(pol, changed, FILES)[0]
    unread = {**pol, "assumptions": pol["assumptions"] | {"P.Other": {"owner": "@lean", "digest": "0" * 64}}}
    assert A.judge(unread, facts(), FILES)[0] == ["assumptions: P.Other is listed, but no guarantee reads it; remove the entry"]
    assert A.judge(unread, facts(), FILES, update=True)[1]["assumptions"] == pol["assumptions"]
    unowned = {**pol, "assumptions": {"P.Hyp": {"digest": pol["assumptions"]["P.Hyp"]["digest"]}}}
    assert A.judge(unowned, facts(), FILES)[0] == ['assumptions: P.Hyp has no owner: write "owner": "@lane" in its entry, or '
                                                   'run `check.py --update --owner @lane`']
    unlisted = {**pol, "assumptions": {}}
    assert A.judge(unlisted, facts(), FILES)[0][0] == ("guarantees: P.t assumes (h : P.Hyp), a closed Prop that is not a "
                                                       "named premise: a Prop constant listed under `assumptions`, "
                                                       "`domain` or `open`")
    elsewhere = {**pol, "assumptions": pol["assumptions"] | {"P.Sound": {"owner": "@lean"}}}
    sound = fact(signature="P.t (h : P.Hyp) (s : P.Sound) (n : Nat) : P.f n = n",
                 hyps=fact()["hypotheses"] + [{"binder": "s", "type": "P.Sound", "closed": True, "named": "P.Sound",
                                               "module": "P.Defs"}],
                 reads=fact()["reads"] + [{"name": "P.Sound", "module": "P.Defs", "kind": "def", "hash": "5" * 32}])
    outside = [x for x in A.judge(elsewhere, facts(pins={"P.t": sound}), FILES)[0] if x.code == "assumptions.outside-module"]
    assert outside == ["assumptions: P.Sound is declared in P.Defs, not in P.Assumptions, where a named assumption is "
                       "declared: move its definition there if it is a real one, or move its entry to `open` with the issue "
                       "that discharges it and its due date, or to `domain` with a one-line `why` if it is part of what "
                       "the guarantee says"]
    assert not A.RECORD_DIFFERENCE.match(outside[0]), "which it is takes a person: its issue, or a move"
    twice = {**pol, "open": {"P.Hyp": pol["assumptions"]["P.Hyp"] | {"issue": "LEAN-1", "due": "2026-11-30"}}}
    assert [x.code for x in A.judge(twice, facts(), FILES)[0]] == ["open.listed-twice"]
    twice = {**pol, "domain": {"P.Hyp": pol["assumptions"]["P.Hyp"] | {"why": "x"}}}
    assert A.judge(twice, facts(), FILES)[0] == ["domain: P.Hyp is listed under `assumptions` and `domain`; keep the one "
                                                 "entry its kind makes it"]
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
    """`heartbeats`, listed and never refused: an override with its reason ending its line, one with the
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
    (tmp_path / "P" / "C.lean").write_text("module\n\npublic initialize pure ()\nmeta initialize pure ()\n")
    hits = A.scans(tmp_path, A.lean_modules(tmp_path), {})
    assert [h.split(" runs code")[0] for h in hits] == ["compile-time: P/A.lean:4", "compile-time: P/A.lean:6",
                                                        "compile-time: P/B.lean:1", "compile-time: P/C.lean:3",
                                                        "compile-time: P/C.lean:4"]
    assert A.scans(tmp_path, A.lean_modules(tmp_path), {"compile_time": {"P.A": "why", "P.B": "why", "P.C": "why"}}) == []
    assert len(A.scans(tmp_path, A.lean_modules(tmp_path), {"compile_time": {"P": "a prefix lists no module"}})) == 5
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
    assert A.as_guarantees(json.loads(json.dumps(old))) == pol
    assert A.judge(A.as_guarantees(old), facts(), FILES) == ([], pol)
    with pytest.raises(ValueError, match=r"the record holds both `pins` and `guarantees`"):
        A.as_guarantees(pol | {"pins": {"P.u": {}}})
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
    def digests():
        return A.dependency_digests(tmp_path / "p", "none/none:none", loaded)
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
    """A nested package with `srcDir = ".."` (`security/Proofs` under `security`) names its modules from the directory
    above it, as Lake does; a library's own `srcDir` is under the package's; a file outside every source directory keeps
    its name from the package."""
    proofs = tmp_path / "security" / "Proofs"
    for rel, text in {"lakefile.toml": 'name = "proofs"\nsrcDir = ".."\n[[lean_lib]]\nname = "Proofs"\n',
                      "Proofs.lean": "", "Pouw/End.lean": ""}.items():
        (proofs / rel).parent.mkdir(parents=True, exist_ok=True)
        (proofs / rel).write_text(text)
    (tmp_path / "security" / "Spec.lean").write_text("")
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


def test_the_lean_checks_audit_passes_record_differences_and_names_the_records_to_commit(tmp_path, monkeypatch):
    """`--regenerate` (the lean-audit step of the Lean check of `next`): a package whose every failure is a record difference,
    so the audit wrote the record `--update` would, passes, and `audit.json` names that record for drive to commit; one with any other
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


def test_a_build_s_log_says_in_one_line_how_many_modules_it_restored_and_why_not_the_others():
    """One line a build, so that any run's log says what the cache missed: the misses by reason, most frequent first,
    each with the first module it names."""
    assert A.staged_line({"modules": 3, "keyed": 3, "missed": {}, "first": {}}) == \
        "3 of 3 keyed modules restored from the artifact cache"
    assert A.staged_line({"modules": 1, "keyed": 6, "missed": {"reads a module not restored": 3, "its source": 2},
                          "first": {"reads a module not restored": "P", "its source": "P.A"}}) == \
        ("1 of 6 keyed modules restored from the artifact cache; missed: 3 reads a module not restored (first P), "
         "2 its source (first P.A)")
    assert A.staged_line({"error": "no keys"}) == "nothing restored from the artifact cache: no keys"
    assert A.staged_line({"modules": 4, "keyed": 9, "missed": {}, "first": {}, "left": 2,
                          "took": {"not in the cache": 3, "the cache holds another version": 1}}) == \
        ("4 of 9 keyed modules restored from the artifact cache; traces taken from the build: 3 not in the cache, "
         "1 the cache holds another version; 2 left to Lake, which the cache could not index")


def test_with_the_node_s_artifact_cache_each_build_step_reads_a_cache_staged_for_it(tmp_path, monkeypatch):
    """Lake finds a module in a cache by its 64-bit input hash alone, so no build reads the node's artifact cache: each
    step reads a directory staged inside it for this build (`LAKE_CACHE_DIR`, with `LAKE_ARTIFACT_CACHE` unset: read,
    never written), holding the modules the cache indexes under their keys, and gone once the build ends. Without the
    sandbox nothing is staged: the sampled recompile, which checks what a build restored, needs it."""
    mk = A.sibling("module_keys")
    pkg, store = tmp_path / "p", tmp_path / "store"
    (pkg / "P").mkdir(parents=True)
    (pkg / A.POLICY).write_text(json.dumps({"roots": ["P"]}))
    (pkg / "lean-toolchain").write_text("leanprover/lean4:v4.34.0\n")
    (pkg / "lakefile.toml").write_text('name = "p"\ndefaultTargets = ["P"]\n\n[[lean_lib]]\nname = "P"\n')
    (pkg / "lake-manifest.json").write_text(json.dumps({"name": "p", "packages": []}))
    (pkg / "P.lean").write_text("theorem p : True := trivial\n")
    (store / "artifacts").mkdir(parents=True)
    (store / "artifacts" / ("a" * 16 + ".olean")).write_bytes(b"P")
    (store / mk.INDEX).mkdir()
    mk.entry_path(store, mk.key("P", mk.keys(pkg), {})).write_text(json.dumps({
        "schema": mk.SCHEMA, "module": "P", "scope": "p", "hash": "1" * 16, "outputs": {"o": ["a" * 16 + ".olean"]},
        "digests": {"a" * 16 + ".olean": mk.C.digest(b"P")}}))
    cache = store / mk.STAGE / f"{os.getpid()}-p"
    seen = []

    def fake_run(argv, cwd, log=None, timeout=0):
        seen.append((argv, (cache / "outputs" / "p" / f"{'1' * 16}.json").is_file()))
        return (1, "fake build") if len(seen) == 2 else (0, "")

    monkeypatch.setattr(A, "sandboxed", lambda: False)
    monkeypatch.setattr(A, "setup", lambda pkg, toolchain, log: (0, ""))
    monkeypatch.setattr(A, "run", fake_run)
    monkeypatch.setattr(A, "ARTIFACTS", store)
    rep = A.audit(pkg, tmp_path / "out", build=True, runs=False)
    assert seen == [(["lake", "build"], False), (["lake", "build", "+P"], False)]
    assert rep["staged"].startswith("not used: this node builds without the sandbox") and "stage" not in rep["seconds"]
    seen.clear()
    monkeypatch.setattr(A, "sandboxed", lambda: True)
    monkeypatch.setattr(A, "in_sandbox", lambda pkg, argv: argv)
    rep = A.audit(pkg, tmp_path / "out", build=True, runs=False)
    assert seen == [(["env", "-u", "LAKE_ARTIFACT_CACHE", f"LAKE_CACHE_DIR={cache}", "lake", "build"], True),
                    (["env", "-u", "LAKE_ARTIFACT_CACHE", f"LAKE_CACHE_DIR={cache}", "lake", "build", "+P"], True)]
    assert rep["staged"] == {"modules": 1, "keyed": 1, "missed": {}, "first": {}, "took": {}, "left": 0}
    assert "stage" in rep["seconds"]
    assert not cache.exists()
    assert rep["failures"] == ["build-roots: exit 1\nfake build"]
    lib = pkg / ".lake" / "build" / "lib" / "lean"
    lib.mkdir(parents=True)
    (pkg / "P" / "Q.lean").write_text("")
    (lib / "P").mkdir()
    (lib / "P" / "Q.trace").write_text(json.dumps({"depHash": "1" * 16}))   # Lake took P's entry for it
    seen.clear()
    monkeypatch.setattr(A, "run", lambda argv, cwd, log=None, timeout=0: (0, ""))
    rep = A.audit(pkg, tmp_path / "out", build=True, runs=False)
    assert [f.code for f in rep["failures"]] == ["build"] and "P.Q (P's)" in rep["failures"][0]
    assert not (lib / "P" / "Q.trace").exists() and not cache.exists()


def test_an_audit_has_its_build_settle_only_the_path_dependencies_its_run_does_not_audit(tmp_path, monkeypatch):
    """A path dependency this run audits is sealed and settled by its own build; the build of one it doesn't audit (in
    an audit's tree, a copy of a kept build) is settled by the builds that require it, or the modules the cache lacks
    there are never indexed (`stage_build`'s ``unaudited``)."""
    d, pkg, given = tmp_path / "d", tmp_path / "p", []
    for at, name, requires in ((d, "D", []), (pkg, "P", [{"type": "path", "name": "d", "dir": "../d"}])):
        at.mkdir()
        (at / A.POLICY).write_text(json.dumps({"roots": [name]}))
        (at / "lean-toolchain").write_text("leanprover/lean4:v4.34.0\n")
        (at / "lakefile.toml").write_text(f'name = "{name.lower()}"\ndefaultTargets = ["{name}"]\n\n[[lean_lib]]\nname = "{name}"\n')
        (at / "lake-manifest.json").write_text(json.dumps({"name": name.lower(), "packages": requires}))
        (at / f"{name}.lean").write_text("")

    def stage_build(mk, pkg, roots, dest, unaudited=()):
        given.append(list(unaudited))
        return {}, {"modules": 0, "keyed": 0, "missed": {}, "first": {}, "took": {}, "left": 0}

    monkeypatch.setattr(A, "sandboxed", lambda: True)
    monkeypatch.setattr(A, "in_sandbox", lambda pkg, argv: argv)
    monkeypatch.setattr(A, "setup", lambda pkg, toolchain, log: (0, ""))
    monkeypatch.setattr(A, "run", lambda argv, cwd, log=None, timeout=0: (1, "fake build"))
    monkeypatch.setattr(A, "ARTIFACTS", tmp_path / "store")
    monkeypatch.setattr(A, "stage_build", stage_build)
    A.audit(pkg, tmp_path / "out", build=True, runs=False)
    A.audit(pkg, tmp_path / "out", build=True, runs=False, ready=lambda q: True)
    A.audit(pkg, tmp_path / "out", build=True, runs=False, ready=lambda q: False)
    assert given == [[d.resolve()], [], []]


def test_before_its_first_build_step_an_audit_leaves_whether_its_builds_run_in_the_sandbox(tmp_path, monkeypatch):
    """What a build compiled goes into the node's artifact cache only on check.py's word that it ran in the sandbox. A
    check.py that is killed writes no report, so that word is in the package's work directory from before the first
    build step, and an audit that never builds, or builds elsewhere than the one before it, leaves no earlier one."""
    pkg, out = tmp_path / "p", tmp_path / "out"
    pkg.mkdir()
    (pkg / A.POLICY).write_text(json.dumps({"roots": ["P"]}))
    (pkg / "lean-toolchain").write_text("leanprover/lean4:v4.34.0\n")
    (pkg / "P.lean").write_text("theorem p : True := trivial\n")
    began = out / "audit-p" / A.BUILDING
    seen = []
    monkeypatch.setattr(A, "setup", lambda pkg, toolchain, log: (0, ""))
    monkeypatch.setattr(A, "run", lambda argv, cwd, log=None, timeout=0: seen.append(json.loads(began.read_text())) or (1, ""))
    monkeypatch.setattr(A, "in_sandbox", lambda pkg, argv: argv)
    for sandbox in (True, False):
        monkeypatch.setattr(A, "sandboxed", lambda: sandbox)
        assert A.audit(pkg, out, build=True, runs=False)["sandbox"] is sandbox
        assert seen[-1] == {"package": str(pkg), "sandbox": sandbox}
    (pkg / A.POLICY).write_text("{")
    assert A.audit(pkg, out, build=True, runs=False)["failures"] and began.exists(), "a package it never opened keeps its own"
    (pkg / A.POLICY).write_text(json.dumps({"roots": ["P"]}))
    monkeypatch.setattr(A, "run", lambda argv, cwd, log=None, timeout=0: (1, ""))
    A.audit(pkg, out, build=False, runs=False, replay=False)
    assert not began.exists(), "this audit built nothing: what the one before it said is gone"


def test_the_sampled_recompile_follows_the_build_and_its_failure_fails_the_audit_before_the_build_is_announced(
        tmp_path, monkeypatch):
    """It takes the modules the build steps' logs say Lake compiled, and the build's env prefix without a cache; a failure
    there is the audit's, and nothing reads the build: neither the seal nor the announcement (`built`) comes."""
    pkg = tmp_path / "p"
    (pkg / "P").mkdir(parents=True)
    (pkg / A.POLICY).write_text(json.dumps({"roots": ["P"]}))
    (pkg / "lean-toolchain").write_text("leanprover/lean4:v4.34.0\n")
    (pkg / "lakefile.toml").write_text('name = "p"\ndefaultTargets = ["P"]\n\n[[lean_lib]]\nname = "P"\n')

    def fake_run(argv, cwd, log=None, timeout=0):
        Path(log).write_text(f"✔ [1/2] Built {'P.A' if argv[-1] == 'build' else 'P'} (1.2s)\n")
        return 0, ""

    calls = []

    def sampled(pkg_, roots, work, built, jobs, log=print):
        calls.append((roots, built, jobs))
        return [A.Failure("sampled", "P.B", "sampled: P.B: planted")], {"population": 1, "modules": ["P.B"], "seconds": 0.0}

    monkeypatch.setattr(A, "sandboxed", lambda: False)
    monkeypatch.setattr(A, "setup", lambda pkg, toolchain, log: (0, ""))
    monkeypatch.setattr(A, "run", fake_run)
    monkeypatch.setattr(A, "sampled_recompile", sampled)
    announced = []
    rep = A.audit(pkg, tmp_path / "out", build=True, runs=False, built=lambda: announced.append(True))
    n = A.build_threads(pkg)
    assert calls == [(["P"], {"P.A", "P"}, ["env", "-u", "LAKE_CACHE_DIR", *([f"LEAN_NUM_THREADS={n}"] if n else [])])]
    assert rep["failure_codes"] == ["sampled"] and rep["sampled"]["modules"] == ["P.B"] and announced == []


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_planted_olean_among_the_restored_modules_fails_once_it_is_sampled(tmp_path, monkeypatch):
    """An index entry of the node's artifact cache that names another version's `.olean`, digest and all, passes every
    check `stage` makes, and Lake restores it with the rest of the build, compiling nothing. Once the sampled recompile
    takes it, it fails naming the module and the file, and keeps what its source compiles to in the overlay's upper layer;
    a sample that misses it passes. Either way the build is never written. `P.M5` logs a warning, which keeps its source's
    path, so restored from another tree it is equal only as an object graph."""
    import random
    if not A.sandboxed():
        pytest.skip("no sandbox here, so the recompile has no overlay to run in")
    spec = importlib.util.spec_from_file_location("lean_audit_for_sampled", HERE.parent / "lean_audit.py")
    LA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LA)
    mk = A.sibling("module_keys")
    lean = {f"P/M{i}.lean": f"module\n\n@[expose] public section\n\ndef P.v{i} : Nat := {i}\n" for i in range(10)}
    lean["P/M5.lean"] += "\ndef P.unused (m : Nat) (k : Nat) : Nat := k\n"
    lean["P.lean"] = "module\n\n" + "".join(f"public import P.M{i}\n" for i in range(10))

    def tree(at: Path, **edits: str) -> Path:
        for rel, text in (lean | edits).items():
            (at / rel).parent.mkdir(parents=True, exist_ok=True)
            (at / rel).write_text(text)
        (at / "lakefile.toml").write_text('name = "p"\ndefaultTargets = ["P"]\n\n[[lean_lib]]\nname = "P"\n')
        shutil.copy(HERE / "fixtures" / "control" / "lean-toolchain", at)
        return at

    def lake(at: Path, *pre: str) -> str:
        r = subprocess.run(["env", *pre, "lake", "build"], cwd=at, capture_output=True, text=True, env=A.lake_env())
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    store = tmp_path / "artifacts"
    arts = LA.LakeArtifacts(store)
    a, v = tree(tmp_path / "a"), tree(tmp_path / "v", **{"P/M2.lean": lean["P/M2.lean"].replace(":= 2", ":= 7")})
    for t in (a, v):
        lake(t)
        assert arts.put({t: {"sandbox": True}}, log=lambda s: None) > 0
    entries = [(f, json.loads(f.read_text())) for f in (store / mk.INDEX).glob("*.json")]
    hash_a = json.loads((a / ".lake/build/lib/lean/P/M2.trace").read_text())["depHash"]
    [(f, e)] = [(f, e) for f, e in entries if e["module"] == "P.M2" and e["hash"] == hash_a]
    [other] = [e for _, e in entries if e["module"] == "P.M2" and e["hash"] != hash_a]
    mine, theirs = (next(n for n in x["outputs"]["o"] if n.endswith(".olean")) for x in (e, other))
    e["outputs"]["o"] = [theirs if n == mine else n for n in e["outputs"]["o"]]
    e["digests"] = {theirs if n == mine else n: other["digests"][theirs] if n == mine else d for n, d in e["digests"].items()}
    f.write_text(json.dumps(e))

    b = tree(tmp_path / "elsewhere" / "b")
    shutil.copy(a / "lake-manifest.json", b)
    monkeypatch.setattr(A, "ARTIFACTS", store)
    cache = store / mk.STAGE / "test"
    staged, said = A.stage_build(mk, b, ["P"], cache)
    assert said == {"modules": 11, "keyed": 11, "missed": {}, "first": {}, "took": {}, "left": 0}
    assert "Built" not in lake(b, "-u", "LAKE_ARTIFACT_CACHE", f"LAKE_CACHE_DIR={cache}")
    lib = b / ".lake" / "build" / "lib" / "lean"
    assert (lib / "P/M2.olean").read_bytes() == (v / ".lake/build/lib/lean/P/M2.olean").read_bytes(), "planted"
    build = b / ".lake" / "build"
    before = files_of(build)
    mods = sorted(["P", *(f"P.M{i}" for i in range(10))])
    samples = {s: random.Random(s).sample(mods, A.SAMPLE_MIN) for s in range(100)}
    miss = next(s for s, d in samples.items() if "P.M2" not in d and "P.M5" in d)
    hit = next(s for s, d in samples.items() if "P.M2" in d)

    monkeypatch.setattr(A, "SAMPLE_RNG", random.Random(miss))
    fails, rep = A.sampled_recompile(b, ["P"], tmp_path / "w1", set(), [], log=lambda s: None)
    assert fails == [] and rep["population"] == 11 and rep["modules"] == sorted(samples[miss]), rep
    assert rep["by_graph"] == ["P.M5"]
    assert files_of(build) == before, "the build is never written"
    assert not (tmp_path / "w1" / "sampled").exists(), "a pass removes the overlay's layers"

    monkeypatch.setattr(A, "SAMPLE_RNG", random.Random(hit))
    fails, rep = A.sampled_recompile(b, ["P"], tmp_path / "w2", set(), [], log=lambda s: None)
    assert rep["modules"] == sorted(samples[hit]) and [(x.code, x.entry) for x in fails] == [("sampled", "P.M2")], fails
    assert ".lake/build/lib/lean/P/M2.olean differs from what its source compiles to (olean differ" in fails[0]
    assert rep["mismatched"] == {"P.M2": str(b.resolve())}, "the report names the module and its package's directory"
    assert files_of(build) == before, "the build is never written"
    upper = tmp_path / "w2" / "sampled" / "0" / "upper" / "build"
    assert (upper / "lib/lean/P/M2.olean").read_bytes() == (a / ".lake/build/lib/lean/P/M2.olean").read_bytes()
    assert not (tmp_path / "w2" / "sampled" / "0" / "work").exists()
    assert all(f.is_file() or f.is_dir() for f in (tmp_path / "w2" / "sampled").rglob("*")), "no whiteout is kept"

    fails, rep = A.sampled_recompile(b, ["P"], tmp_path / "w3", set(mods), [], log=lambda s: None)
    assert fails == [] and rep == {"population": 0, "modules": [], "seconds": rep["seconds"]}, "a module it compiled is never sampled"


def files_of(build: Path) -> dict[Path, tuple[int, int, bytes]]:
    """Each file under ``build``, by its inode, its modification time and its bytes: equal only if nothing wrote it."""
    return {p: (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) for p in build.rglob("*") if p.is_file()}


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_another_package_reads_a_sampled_dependency_s_build_unchanged_while_the_recompile_runs(tmp_path, monkeypatch):
    """A dependent's sample takes modules of its path dependency, whose `.lake/build` other packages' audits read at the same
    time: their tests open its `.olean` files, and `main` re-hashes every package's sealed `.olean` files after each audit
    (`tampered`). Both look inside the recompile's window, as it starts and as Lake returns, and find every file of both
    builds at its path with its bytes; afterwards no file was written. When the sample moved its outputs out and back,
    this failed in the window: `D.olean` was missing."""
    if not A.sandboxed():
        pytest.skip("no sandbox here, so the recompile has no overlay to run in")
    d, p = tmp_path / "ws" / "d", tmp_path / "ws" / "p"
    for at, name, text, req in ((d, "D", "def D.v : Nat := 1\n", ""),
                                (p, "P", "import D\n\ndef P.v : Nat := D.v + 1\n", '\n[[require]]\nname = "d"\npath = "../d"\n')):
        at.mkdir(parents=True)
        (at / "lakefile.toml").write_text(f'name = "{name.lower()}"\ndefaultTargets = ["{name}"]\n{req}\n'
                                          f'[[lean_lib]]\nname = "{name}"\n')
        shutil.copy(HERE / "fixtures" / "control" / "lean-toolchain", at)
        (at / f"{name}.lean").write_text(text)
    r = subprocess.run(["lake", "build"], cwd=p, capture_output=True, text=True, env=A.lake_env())
    assert r.returncode == 0, r.stdout + r.stderr
    builds = [d / ".lake" / "build", p / ".lake" / "build"]
    seals = {str(f): A.sha256(f) for b in builds for f in A.olean_files(b / "lib" / "lean")}
    dep = d / ".lake" / "build" / "lib" / "lean" / "D.olean"
    before, olean = [files_of(b) for b in builds], dep.read_bytes()
    window = []

    def look():
        window.append((A.tampered(dict(seals), []), dep.read_bytes() if dep.is_file() else None))

    real = A.run

    def run(argv, cwd, log=None, **kw):
        if "lake" not in argv or not any(a.startswith("+") for a in argv):
            return real(argv, cwd, log, **kw)
        look()
        got = real(argv, cwd, log, **kw)
        look()
        return got

    monkeypatch.setattr(A, "run", run)
    fails, rep = A.sampled_recompile(p, ["P"], tmp_path / "w", set(), [], log=lambda s: None)
    assert fails == [] and rep["modules"] == ["D", "P"], (fails, rep)
    assert window == [([], olean), ([], olean)], "a reader in the window finds the dependency's build as it was"
    assert [files_of(b) for b in builds] == before, "nothing wrote either build"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_dependent_staged_on_a_cold_cache_rewrites_no_file_its_dependency_s_audit_sealed(tmp_path, monkeypatch):
    """A node's artifact cache holds nothing under these keys. `d` is staged and built, and its `.olean` files are sealed
    as `main` seals them once its build ends; only then is `p`, which requires `d` by path, staged and built, as `main`
    orders them. The cache still lacks `D`: an audit stores what its builds compiled when all of them have ended. When
    staging `p` took `D`'s trace for that, `p`'s Lake compiled `D` again inside `d`'s build, and every audit beside it
    that hashed a sealed `.olean` or ran a test on one in that window found it changed or gone. Now staging `p` writes
    nothing in `d`'s build, Lake finds `D` up to date and compiles `P` alone, and each sealed file is the file the seal
    hashed: its inode, its time and its bytes. What the two builds compiled is then stored, each module once, and a
    second tree compiles neither."""
    spec = importlib.util.spec_from_file_location("lean_audit_for_sealed", HERE.parent / "lean_audit.py")
    LA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LA)
    mk = A.sibling("module_keys")
    store = tmp_path / "artifacts"
    monkeypatch.setattr(A, "ARTIFACTS", store)
    stages = iter(range(99))

    def tree(ws: Path) -> tuple[Path, Path]:
        for name, text, requires in (("D", "def D.v : Nat := 1\n", []), ("P", "import D\n\ndef P.v : Nat := D.v + 1\n", [
                {"type": "path", "scope": "", "name": "d", "manifestFile": "lake-manifest.json", "inherited": False,
                 "dir": "../d", "configFile": "lakefile.toml"}])):
            at = ws / name.lower()
            at.mkdir(parents=True)
            (at / "lakefile.toml").write_text(
                f'name = "{name.lower()}"\ndefaultTargets = ["{name}"]\n'
                + "".join(f'\n[[require]]\nname = "{r["name"]}"\npath = "{r["dir"]}"\n' for r in requires)
                + f'\n[[lean_lib]]\nname = "{name}"\n')
            (at / "lake-manifest.json").write_text(json.dumps({
                "version": "1.2.0", "packagesDir": ".lake/packages", "packages": requires, "name": name.lower(),
                "lakeDir": ".lake", "fixedToolchain": False}))
            shutil.copy(HERE / "fixtures" / "control" / "lean-toolchain", at)
            (at / f"{name}.lean").write_text(text)
        return ws / "d", ws / "p"

    def build(pkg: Path) -> tuple[set[str], dict]:
        """(the modules Lake compiled, what staging took from the tree) of ``pkg``'s build as an audit runs it: staged from
        the node's cache, then `lake build` reading what was staged."""
        cache = tmp_path / "staged" / str(next(stages))
        _, said = A.stage_build(mk, pkg, [pkg.name.upper()], cache)
        r = subprocess.run(["env", "-u", "LAKE_ARTIFACT_CACHE", f"LAKE_CACHE_DIR={cache}", "lake", "build"], cwd=pkg,
                           capture_output=True, text=True, env=A.lake_env())
        assert r.returncode == 0, r.stdout + r.stderr
        shutil.rmtree(cache)
        return {m for m, *_ in A.BUILT.findall(r.stdout + r.stderr)}, said["took"]

    d, p = tree(tmp_path / "ws")
    assert build(d) == ({"D"}, {})
    lib = d / ".lake" / "build" / "lib" / "lean"
    seals = {str(f): A.sha256(f) for f in A.olean_files(lib)}
    sealed = {f: at for f, at in files_of(lib).items() if str(f) in seals}
    trace = (lib / "D.trace").read_bytes()
    assert len(sealed) == len(seals) > 0 and not list((store / mk.INDEX).glob("*")), "d is sealed, and the cache still cold"

    compiled, took = build(p)
    assert {f: at for f, at in files_of(lib).items() if str(f) in seals} == sealed, "p's build rewrote a file d's audit sealed"
    assert A.tampered(dict(seals), []) == [] and (lib / "D.trace").read_bytes() == trace
    assert (compiled, took) == ({"P"}, {}), "p's staging takes no trace of d's, so its Lake compiles P alone"

    assert LA.LakeArtifacts(store).put({d: {"sandbox": True}, p: {"sandbox": True}}, log=lambda s: None) == 2
    d2, p2 = tree(tmp_path / "elsewhere" / "ws")
    assert (build(d2), build(p2)) == ((set(), {}), (set(), {})), "a second tree restores D and P and compiles neither"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_dependent_that_fetches_more_than_its_dependency_stores_both_on_a_cold_cache(tmp_path, monkeypatch):
    """`d` fetches a package `x`; `p` requires `d` by path and fetches `z` too, as a proofs package requires the
    definitions' and fetches more than they do. A node's artifact cache holds nothing. `d` is built, by its own Lake over
    its own copy of `x`, then `p`, whose Lake finds `D` up to date, and what the two builds compiled is stored. `D`'s key
    names `x`'s pin and no `z`, which `d` can't import: it is one key under `d` and under `p`, so the compile `d`'s build
    left is `D`'s entry under `p` as well and `P`, which loads it, is stored above it. When `D`'s key under `p` named
    `z` too, `d`'s own compile of it was no compile of that key, nothing stood under it, and every tree compiled `P`
    again. A second tree compiles neither package's modules."""
    spec = importlib.util.spec_from_file_location("lean_audit_for_fetched", HERE.parent / "lean_audit.py")
    LA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LA)
    mk = A.sibling("module_keys")
    store = tmp_path / "artifacts"
    monkeypatch.setattr(A, "ARTIFACTS", store)
    stages = iter(range(99))
    toolchain = HERE / "fixtures" / "control" / "lean-toolchain"
    git = {**{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},   # the caller's, less what steers git
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
           **{f"GIT_{who}_{what}": v for who in ("AUTHOR", "COMMITTER") for what, v in (("NAME", "a"), ("EMAIL", "a@example.org"))}}

    def manifest(name: str, packages: list[dict]) -> str:
        return json.dumps({"version": "1.2.0", "packagesDir": ".lake/packages", "packages": packages, "name": name,
                           "lakeDir": ".lake", "fixedToolchain": False})

    def fetched(name: str) -> dict:
        """A package `name` with one module, as a Git repository under `upstream`: its entry in a manifest that pins it."""
        repo = tmp_path / "upstream" / name
        repo.mkdir(parents=True)
        (repo / "lakefile.toml").write_text(f'name = "{name}"\n\n[[lean_lib]]\nname = "{name.upper()}"\n')
        (repo / "lake-manifest.json").write_text(manifest(name, []))
        (repo / f"{name.upper()}.lean").write_text(f"def {name.upper()}.v : Nat := 1\n")
        (repo / ".gitignore").write_text(".lake\n")
        shutil.copy(toolchain, repo)
        for argv in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", name]):
            subprocess.run(["git", *argv], cwd=repo, check=True, capture_output=True, env=git)
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True,
                             env=git).stdout.strip()
        return {"url": str(repo), "type": "git", "subDir": None, "scope": "", "rev": rev, "name": name,
                "manifestFile": "lake-manifest.json", "inputRev": rev, "inherited": False, "configFile": "lakefile.toml"}

    x_pin, z_pin = fetched("x"), fetched("z")
    by_path = {"type": "path", "scope": "", "name": "d", "manifestFile": "lake-manifest.json", "inherited": False,
               "dir": "../d", "configFile": "lakefile.toml"}

    def tree(ws: Path) -> tuple[Path, Path]:
        for name, text, requires in (("D", "import X\n\ndef D.v : Nat := X.v\n", [x_pin]),
                                     ("P", "import D\nimport Z\n\ndef P.v : Nat := D.v + Z.v\n",
                                      [by_path, z_pin, x_pin | {"inherited": True}])):
            at = ws / name.lower()
            at.mkdir(parents=True)
            (at / "lakefile.toml").write_text(
                f'name = "{name.lower()}"\ndefaultTargets = ["{name}"]\n'
                + "".join(f'\n[[require]]\nname = "{r["name"]}"\n'
                          + (f'path = "{r["dir"]}"\n' if r["type"] == "path" else f'git = "{r["url"]}"\nrev = "{r["rev"]}"\n')
                          for r in requires if not r["inherited"])
                + f'\n[[lean_lib]]\nname = "{name}"\n')
            (at / "lake-manifest.json").write_text(manifest(name.lower(), requires))
            shutil.copy(toolchain, at)
            (at / f"{name}.lean").write_text(text)
            for r in requires:
                if r["type"] == "git":
                    shutil.copytree(tmp_path / "upstream" / r["name"], at / ".lake" / "packages" / r["name"], symlinks=True)
        return ws / "d", ws / "p"

    def build(pkg: Path) -> set[str]:
        """The modules of `d` and `p` that Lake compiled in ``pkg``'s build as an audit runs it: staged from the node's
        cache, then `lake build` reading what was staged. Each tree compiles its own copies of `x` and `z`, which an
        audit's fetched packages come built."""
        cache = tmp_path / "staged" / str(next(stages))
        _, said = A.stage_build(mk, pkg, [pkg.name.upper()], cache)
        r = subprocess.run(["env", "-u", "LAKE_ARTIFACT_CACHE", f"LAKE_CACHE_DIR={cache}", "lake", "build"], cwd=pkg,
                           capture_output=True, text=True, env=A.lake_env())
        assert r.returncode == 0 and "error" not in said and not said["took"], (said, r.stdout + r.stderr)
        shutil.rmtree(cache)
        return {m for m, *_ in A.BUILT.findall(r.stdout + r.stderr)} & {"D", "P"}

    d, p = tree(tmp_path / "ws")
    assert (build(d), build(p)) == ({"D"}, {"P"}) and not list((store / mk.INDEX).glob("*")), "the cache is cold"
    under = {r.name: mk.keys(r)["D"] for r in (d, p)}
    one_key = (under["d"].env, under["d"].task) == (under["p"].env, under["p"].task)
    stored = LA.LakeArtifacts(store).put({d: {"sandbox": True}, p: {"sandbox": True}}, log=lambda s: None)
    d2, p2 = tree(tmp_path / "elsewhere" / "ws")
    again = (build(d2), build(p2))
    assert (one_key, stored, again) == (True, 2, (set(), set())), "D has one key under d and p, and a second tree compiles neither"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_required_package_moved_is_compiled_by_no_build(tmp_path, monkeypatch):
    """`p` requires `d` by path. Both are built and stored. A second tree holds `d` somewhere else, with the `path` of
    `p`'s `require` and the `dir` of its manifest's entry rewritten and nothing else changed. A key reads neither, and
    neither does Lake's own input hash of a module, under which the staged outputs are restored: the second tree's
    builds compile no module."""
    spec = importlib.util.spec_from_file_location("lean_audit_for_moved", HERE.parent / "lean_audit.py")
    LA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LA)
    mk = A.sibling("module_keys")
    store = tmp_path / "artifacts"
    monkeypatch.setattr(A, "ARTIFACTS", store)
    stages = iter(range(99))

    def tree(ws: Path, at: str) -> tuple[Path, Path]:
        """Under ``ws``: `p`, and `d` at ``at`` from `p`'s directory."""
        p = ws / "p"
        d = Path(os.path.normpath(p / at))
        for pkg, name, text, requires in ((d, "D", "def D.v : Nat := 1\n", []), (p, "P", "import D\n\ndef P.v : Nat := D.v + 1\n", [
                {"type": "path", "scope": "", "name": "d", "manifestFile": "lake-manifest.json", "inherited": False,
                 "dir": at, "configFile": "lakefile.toml"}])):
            pkg.mkdir(parents=True)
            (pkg / "lakefile.toml").write_text(
                f'name = "{name.lower()}"\ndefaultTargets = ["{name}"]\n'
                + "".join(f'\n[[require]]\nname = "{r["name"]}"\npath = "{r["dir"]}"\n' for r in requires)
                + f'\n[[lean_lib]]\nname = "{name}"\n')
            (pkg / "lake-manifest.json").write_text(json.dumps({
                "version": "1.2.0", "packagesDir": ".lake/packages", "packages": requires, "name": name.lower(),
                "lakeDir": ".lake", "fixedToolchain": False}))
            shutil.copy(HERE / "fixtures" / "control" / "lean-toolchain", pkg)
            (pkg / f"{name}.lean").write_text(text)
        return d, p

    def build(pkg: Path) -> set[str]:
        """The modules Lake compiled in ``pkg``'s build as an audit runs it: staged from the node's cache, then built."""
        cache = tmp_path / "staged" / str(next(stages))
        A.stage_build(mk, pkg, [pkg.name.upper()], cache)
        r = subprocess.run(["env", "-u", "LAKE_ARTIFACT_CACHE", f"LAKE_CACHE_DIR={cache}", "lake", "build"], cwd=pkg,
                           capture_output=True, text=True, env=A.lake_env())
        assert r.returncode == 0, r.stdout + r.stderr
        shutil.rmtree(cache)
        return {m for m, *_ in A.BUILT.findall(r.stdout + r.stderr)}

    d, p = tree(tmp_path / "ws", "../d")
    assert (build(d), build(p)) == ({"D"}, {"P"})
    assert LA.LakeArtifacts(store).put({d: {"sandbox": True}, p: {"sandbox": True}}, log=lambda s: None) == 2
    d2, p2 = tree(tmp_path / "elsewhere" / "ws", "../moved/to/d")
    assert (p / "lakefile.toml").read_text() != (p2 / "lakefile.toml").read_text() and d2.parent.name == "to"
    assert (build(d2), build(p2)) == (set(), set()), "the moved package and the one requiring it are restored, not compiled"


def test_without_the_sandbox_a_module_no_build_of_the_run_compiled_fails_the_audit(tmp_path, monkeypatch):
    """Where check.py's own sandbox probe fails, nothing is recompiled, so a module its build didn't compile passes only if
    a build of this run compiled it, as a dependent's path dependency's audit compiled it. One restored or kept fails,
    naming the cause and the fix: `lean_audit.py`'s probe can pass where check.py's fails, and then it restores the kept
    build. An empty population passes, and a sandboxed audit recompiles its sample as before."""
    d, p = tmp_path / "ws" / "d", tmp_path / "ws" / "p"
    for at, name, text, req in ((d, "D", "def D.v : Nat := 1\n", ""),
                                (p, "P", "import D\n\ndef P.v : Nat := D.v + 1\n", '\n[[require]]\nname = "d"\npath = "../d"\n')):
        (at / ".lake" / "build" / "lib" / "lean").mkdir(parents=True)
        (at / "lakefile.toml").write_text(f'name = "{name.lower()}"\ndefaultTargets = ["{name}"]\n{req}\n'
                                          f'[[lean_lib]]\nname = "{name}"\n')
        (at / "lean-toolchain").write_text("leanprover/lean4:v4.34.0\n")
        (at / f"{name}.lean").write_text(text)
        (at / ".lake" / "build" / "lib" / "lean" / f"{name}.trace").write_text("{}")
    (d / "lake-manifest.json").write_text(json.dumps({"packages": []}))
    (p / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "path", "name": "d", "dir": "../d"}]}))
    monkeypatch.setattr(A, "sandboxed", lambda: False)
    monkeypatch.setattr(A, "RUN_COMPILED", set())
    monkeypatch.setattr(A, "run", lambda argv, cwd, log=None, **kw: (1, "recompiled"))

    fails, rep = A.sampled_recompile(p, ["P"], tmp_path / "w1", set(), [], log=lambda s: None)
    assert [(f.code, f.entry) for f in fails] == [("sampled", None)] and rep["population"] == 2 and rep["modules"] == []
    assert "the sandbox probe failed here" in fails[0] and "2 of the 2 modules" in fails[0] and "(D, P)" in fails[0]
    assert fails[0].endswith("rerun on a node whose builds are sandboxed, or after fixing the probe")
    assert not (tmp_path / "w1").exists(), "nothing is recompiled"

    A.RUN_COMPILED.add(((d / ".lake" / "build").resolve(), "D"))
    fails, rep = A.sampled_recompile(p, ["P"], tmp_path / "w3", {"P"}, [], log=lambda s: None)
    assert fails == [] and rep["population"] == 1 and rep["modules"] == [] and rep["unsampled"].startswith("no sandbox")
    fails, rep = A.sampled_recompile(p, ["P"], tmp_path / "w4", {"D", "P"}, [], log=lambda s: None)
    assert fails == [] and rep == {"population": 0, "modules": [], "seconds": rep["seconds"]}

    monkeypatch.setattr(A, "sandboxed", lambda: True)
    monkeypatch.setattr(A, "RUN_COMPILED", set())
    fails, rep = A.sampled_recompile(p, ["P"], tmp_path / "w5", set(), [], log=lambda s: None)
    assert rep["modules"] == ["D", "P"] and "unsampled" not in rep
    assert fails == ["sampled: the recompile of D, P exited 1\nrecompiled"]


def _two_dependents(tmp_path, monkeypatch, b_build_exit=0):
    """`b` and `c` require `d` by path, which no audit of the run builds (its pass was reused), so neither waits for it:
    `b`'s `lake build` compiles `D` and `B` and exits ``b_build_exit``, and `c`'s, after it, finds `D` up to date and
    compiles `C`. Each fake build writes the traces of what it logs; no sandbox. Returns (an audit of a package, the
    sampled recompiles that ran: (package, failures, report), the ones `b`'s audit put off: their arguments, and the real
    `sampled_recompile`)."""
    d, b, c = (tmp_path / "ws" / n for n in "dbc")
    for at in (d, b, c):
        name = at.name.upper()
        req = "" if at == d else '\n[[require]]\nname = "d"\npath = "../d"\n'
        (at / ".lake" / "build" / "lib" / "lean").mkdir(parents=True)
        (at / "lakefile.toml").write_text(f'name = "{at.name}"\ndefaultTargets = ["{name}"]\n{req}\n'
                                          f'[[lean_lib]]\nname = "{name}"\n')
        (at / "lean-toolchain").write_text("leanprover/lean4:v4.34.0\n")
        (at / f"{name}.lean").write_text(f"{'' if at == d else 'import D' + chr(10)}\ndef {name}.v : Nat := 1\n")
        (at / "lake-manifest.json").write_text(json.dumps({"packages": [] if at == d else [
            {"type": "path", "name": "d", "dir": "../d"}]}))
        if at != d:
            (at / A.POLICY).write_text(json.dumps({"roots": [name]}))
    compiles = {("b", "build"): ["D", "B"], ("c", "build"): ["C"]}

    def fake_run(argv, cwd, log=None, timeout=0, on_line=None):
        step = Path(log).stem
        mods = compiles.get((cwd.name, step), [])
        for m in mods:
            ((d if m == "D" else cwd) / ".lake" / "build" / "lib" / "lean" / f"{m}.trace").write_text("{}")
        Path(log).write_text("".join(f"✔ [{i}/{len(mods)}] Built {m} (1.0s)\n" for i, m in enumerate(mods, 1)))
        return (b_build_exit if (cwd.name, step) == ("b", "build") else 0), ""

    real, ran, later = A.sampled_recompile, [], []

    def sampled(pkg, *args, **kw):
        if pkg.name == "b":
            later.append((pkg, *args))
            return [A.Failure("sampled", None, "sampled: b samples later")], {}
        ran.append((pkg.name, *real(pkg, *args, **kw)))
        return [A.Failure("sampled", None, "sampled: the test stops here")], ran[-1][2]

    monkeypatch.setattr(A, "sandboxed", lambda: False)
    monkeypatch.setattr(A, "RUN_COMPILED", set())
    monkeypatch.setattr(A, "setup", lambda pkg, toolchain, log: (0, ""))
    monkeypatch.setattr(A, "run", fake_run)
    monkeypatch.setattr(A, "sampled_recompile", sampled)
    def audit(p):
        return A.audit(tmp_path / "ws" / p, tmp_path / "out", build=True, runs=False, log=lambda s: None)
    return audit, ran, later, real


def test_a_dependency_another_package_s_build_compiled_is_this_run_s_before_that_package_samples(tmp_path, monkeypatch):
    """`b`'s build compiles `D` and releases its hold on `d`'s `.lake`; `c`'s build, after it, finds `D` up to date, so
    `D` isn't among the modules `c` compiled. `c` samples before `b` does, and without the sandbox `D` passes only if `b`'s
    build step counted it as this run's (`note_compiled`), not `b`'s sample, which hasn't run."""
    audit, ran, later, real = _two_dependents(tmp_path, monkeypatch)
    assert audit("b")["failures"] == ["sampled: b samples later"]
    assert audit("c")["failures"] == ["sampled: the test stops here"]
    [(name, fails, rep)] = ran
    assert (name, fails, rep["population"], rep["modules"]) == ("c", [], 1, []) and rep["unsampled"].startswith("no sandbox")
    [(b, *args)] = later
    fails, rep = real(b, *args)
    assert fails == [] and rep["population"] == 0, "b compiled D and B itself"


def test_what_a_failed_build_compiled_before_it_failed_is_this_run_s(tmp_path, monkeypatch):
    """`b`'s build compiles `D` and then fails, so `b`'s audit ends before its sample. `D`'s outputs are still this run's,
    not restored or kept, so `c`, whose build finds `D` up to date, passes without the sandbox rather than failing with a
    `sampled` that says otherwise; the run fails on `b`'s build."""
    audit, ran, later, real = _two_dependents(tmp_path, monkeypatch, b_build_exit=1)
    assert audit("b")["failure_codes"] == ["build"] and later == []
    assert audit("c")["failures"] == ["sampled: the test stops here"]
    [(name, fails, rep)] = ran
    assert (name, fails, rep["population"]) == ("c", [], 1)


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
    with pytest.raises(A.Unpassed, match=r"\.\./p's build did not pass"):
        A.proved(s, json.loads((s / A.POLICY).read_text()), facts(modules=[], pins={}), tmp_path / "work",
                 ready=lambda q: False)


CORE = {"Core/Protocol.lean": "namespace Core\ndef double (n : Nat) : Nat := 2 * n\nend Core\n",
        "Core/Assumptions.lean": "namespace Core\ndef Hyp : Prop := ∀ n : Nat, n + 0 = n\nend Core\n",
        "Core/Guarantees.lean": "import Core.Protocol\nimport Core.Assumptions\nnamespace Core.Guarantees\n"
                                "def Main : Prop := Core.Hyp → ∀ n, Core.double n = n + n\nend Core.Guarantees\n"}
PROOFS = ("import Core.Guarantees\nnamespace Core.SecurityProofs\n"
          "theorem inline (h : Core.Hyp) (n : Nat) : Core.double n = n + n := by unfold Core.double; omega\n"
          "theorem main : Core.Guarantees.Main := fun h n => inline h n\nend Core.SecurityProofs\n")
SPEC = {"assumptions": ["Core.Assumptions"], "assumptions_module": "Core.Assumptions",
        "meaning": ["Core.Protocol", "Core.Assumptions", "Core.Guarantees"],
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
    def build(d):
        return subprocess.run(["lake", "build"], cwd=d, check=True, capture_output=True, env=A.lake_env())
    for d in (one, spec, proofs):
        build(d)
    for d in (one, proofs):
        assert unshaped(A.audit(d, out, update=True, replay=False, runs=False, log=lambda *_: None)) == []
    proofs_policy = (proofs / A.POLICY).read_text()
    assert set(json.loads(proofs_policy)) == {"format", "roots", "dependencies"}
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

    def proved(pkg, policy, facts, work, moved, ready, keep=True, seals=None, keys=False):
        order.append("proved")
        return [], facts | {"pins": {"P.g": {"kind": "theorem", "reads": []}}, "oleans": ["prover.olean"]}, {"../proofs": ["P.g"]}

    def stub(name, result):
        def f(*args):
            given[name] = args
            return result
        f.__name__ = name
        return f

    monkeypatch.setattr(A, "ThreadPoolExecutor", Recording)
    monkeypatch.setattr(A, "facts_of", lambda *a, keep=True, sealed=None: own)
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
    assert order == (["pinned", "final_replay", "dependency_scans", "dependency_digests", "proved"] if spec else ["final_replay"])
    assert given["replay_step"][2] is own and given["dependency_scans"][1] == own["oleans"] == given["dependency_digests"][2]
    assert given["replay_step"][9] is False      # an unsealed audit (no seals here) never writes the shared replay cache
    assert rep["failures"][0] == "unchecked: a hit" and rep["failures"][1].startswith("dependencies: the .olean files of the")
    assert rep["dependency_sources"] == 7 and rep["upstream"] == {"x": 1} and rep["replay"] == {"checked": 1, "reused": 0}
    assert {"audit", "dependency-sources", "upstream", "dependencies", "replay"} <= set(rep["seconds"])


@pytest.mark.parametrize("sandbox", [True, False])
def test_a_node_that_builds_without_the_sandbox_neither_reuses_nor_writes_the_replay_cache(tmp_path, monkeypatch, sandbox):
    """Outside the sandbox a build could have written the shared replay cache, so an audit there gives its replay no
    cache: it reuses no entry, stores none and checks every module, and the report says why. A sandboxed node's replay
    is given the cache."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "lean-toolchain").write_text("leanprover/lean4:v0\n")
    A.dump(pkg, {"roots": ["P"], "dependencies": {}})
    cache = tmp_path / "replay"
    cache.mkdir()
    (cache / "k.json").write_text("{}")
    given = {}

    def replay_step(*args):
        given["cache"] = args[5]
        return [], {"checked": 1, "reused": 0}

    monkeypatch.setattr(A, "sandboxed", lambda: sandbox)
    monkeypatch.setattr(A, "facts_of", lambda *a, keep=True, sealed=None: {
        "lean": "v0", "modules": ["P"], "declarations": 1, "axioms": [], "escapes": [], "oleans": [], "pins": {}})
    monkeypatch.setattr(A, "replay_step", replay_step)
    monkeypatch.setattr(A, "dependency_scans", lambda *a: ([], 0))
    monkeypatch.setattr(A, "dependency_digests", lambda *a: {})
    for name, value in (("judge", ([], {})), ("dropped", []), ("spec_reads", []), ("shape", []), ("unread_spec", {}),
                        ("read_by_dependents", set()), ("reads_home", {}), ("scans", []), ("heartbeats", []),
                        ("anonymous_instances", []), ("run_paths", []), ("listings", ([], {}, [])), ("regenerated", None)):
        monkeypatch.setattr(A, name, lambda *a, value=value, **k: value)
    rep = A.audit(pkg, tmp_path / "out", runs=False, log=lambda *_: None, replay_cache=cache)
    assert given["cache"] == (cache if sandbox else None)
    assert ("replay_cache" in rep) is not sandbox and (sandbox or "sandbox" in rep["replay_cache"])
    assert [p.name for p in cache.iterdir()] == ["k.json"]


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


def test_a_run_whose_lake_a_build_is_still_writing_fails_loudly(tmp_path, monkeypatch):
    """A run reads the package's live `.lake/build` in the sandbox, not a seal, so it is sound only once every build that
    writes that `.lake` has ended. Only a dependent's build writes a package's own `.lake`, and a package with `runs` has
    none, so this never fires in a real run; were a build still holding the package's own `.lake`, the run fails loudly
    rather than read a build mid-write."""
    pkg, ran, tests = _runs(tmp_path, monkeypatch, {"args": []})
    with A._WROTE:
        A._WRITING.add(pkg.resolve() / ".lake")
    try:
        fails, _ = tests()
    finally:
        with A._WROTE:
            A._WRITING.discard(pkg.resolve() / ".lake")
    assert len(fails) == 1 and fails[0].startswith("runs: a build is still writing") and ran == []


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
def test_a_replay_reuses_only_modules_whose_olean_files_and_imports_passed_it(tmp_path, sandboxed_node):
    """On a real build: a second audit reuses every module's replay; a change to `Control.Defs` replays it and what imports
    it; a declaration that skipped the kernel is still rejected beside reused modules, and a failed replay stores the
    modules the kernel accepted (`Control`, which imports the bad one) and never the rejected one, which fails again."""
    pkg, cache = tmp_path / "control", tmp_path / "replay-cache"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    def build():
        return subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    def audit():                                             # store=True: the cache is this test's own private dir
        return A.audit(pkg, tmp_path / "out", update=True, runs=False, log=lambda *_: None, replay_cache=cache, store=True)
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


#: the parts of a module a seal copies and a swap must overwrite to change what the kernel replays
_BAD_PARTS = ("Bad.olean", "Bad.olean.server", "Bad.olean.private", "Bad.ir.sig", "Bad.ir")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_seal_replays_the_build_at_the_seal_not_a_swap_a_dependent_makes(tmp_path, monkeypatch, sandboxed_node):
    """The gap this closes: once a package is sealed, a dependent's build may write its `.lake` (`writable`), so a
    reader of the live files could be shown bytes the build swaps out and back.  The audit copies the build at the seal and
    reads the copies.  Here a package is built with a `Control.Bad` whose declaration skips the kernel (the sealed, true
    state), and a dependent package's compile-time `#eval` overwrites that `.olean` with a good one during the window; the
    replay must reject the sealed declaration all the same and store nothing under the good one's key, and it must keep
    rejecting on a warm cache.  The mutation that reads the live files instead (``Sealed.moved`` mapping nothing) passes the
    swapped-in good bytes, which proves the copies are what protect the verdict."""
    S = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", S, ignore=shutil.ignore_patterns(".lake"))
    kernel = json.loads((HERE / "fixtures" / "violations.json").read_text())["kernel"]
    for rel, (old, new) in kernel["replace"].items():                   # the canonical way to pull Control.Bad into Control
        (S / rel).write_text((S / rel).read_text().replace(old, new))
    A.dump(S, A.load(S) | kernel["policy"])
    lib = S / ".lake" / "build" / "lib" / "lean" / "Control"
    def build(d=S):
        return subprocess.run(["lake", "build"], cwd=d, check=True, capture_output=True, env=A.lake_env())

    def stash(dst: Path) -> Path:
        dst.mkdir()
        for n in _BAD_PARTS:
            if (lib / n).is_file():
                shutil.copyfile(lib / n, dst / n)
        return dst

    def restore(src: Path) -> None:
        for n in _BAD_PARTS:
            (lib / n).unlink(missing_ok=True)
            if (src / n).is_file():
                shutil.copyfile(src / n, lib / n)

    (S / "Control" / "Bad.lean").write_text("theorem Control.bogus : True := trivial\n")
    build()
    good = stash(tmp_path / "good")
    (S / "Control" / "Bad.lean").write_text(kernel["write"]["Control/Bad.lean"])
    build()
    bad = stash(tmp_path / "bad")                        # the sealed, true state: the kernel rejects it

    p = tmp_path / "p"                                   # a dependent whose compile-time code swaps the good bytes in
    (p / "P").mkdir(parents=True)
    (p / "lean-toolchain").write_text((S / "lean-toolchain").read_text())
    (p / "lakefile.toml").write_text('name = "p"\ndefaultTargets = ["P"]\n\n[[require]]\nname = "control"\npath = "../control"'
                                     '\n\n[[lean_lib]]\nname = "P"\n')
    (p / "lake-manifest.json").write_text(json.dumps({"version": "1.2.0", "packagesDir": ".lake/packages", "packages": [
        {"type": "path", "scope": "", "name": "control", "manifestFile": "lake-manifest.json", "inherited": False,
         "dir": "../control", "configFile": "lakefile.toml"}], "name": "p", "lakeDir": ".lake"}))
    (p / "P.lean").write_text("import P.Swap\n")
    names = ", ".join(f'"{n}"' for n in _BAD_PARTS)
    (p / "P" / "Swap.lean").write_text(
        '#eval show IO Unit from do\n'
        f'  for n in [{names}] do\n'
        f'    let g := System.FilePath.mk ("{good}/" ++ n)\n'
        '    if ← g.pathExists then\n'
        f'      IO.FS.writeBinFile (System.FilePath.mk ("{lib}/" ++ n)) (← IO.FS.readBinFile g)\n')
    def swap() -> None:                                  # the dependent's build; its `#eval` swaps the good bytes in
        shutil.rmtree(p / ".lake", ignore_errors=True)   # so Lake recompiles `P.Swap` and runs its `#eval` every window
        build(p)

    def run(out: str, cache: Path) -> dict:
        restore(bad)
        seals = A.Seals(tmp_path / f"sealed-{out}")
        try:
            return A.audit(S, tmp_path / out, update=True, seals=seals, built=swap, replay_cache=cache, runs=False,
                           log=lambda *_: None)
        finally:
            seals.close()

    def rejected(rep: dict) -> bool:
        return any(f.startswith("replay:") and "Control.Bad" in f for f in rep["failures"])

    cache = tmp_path / "cache"
    assert rejected(run("out1", cache)), "the replay read the sealed declaration the kernel rejects, not the swapped-in good one"
    stored = [json.loads(c.read_text()).get("module") for c in cache.glob("*.json")]
    assert "Control.Bad" not in stored, "the rejected module is stored under no key, the swapped-in good bytes' included"
    assert rejected(run("out2", cache)), "a warm cache keeps rejecting: the swap stored no pass that lets the module be skipped"
    monkeypatch.setattr(A.Sealed, "moved", lambda self, path: None)
    assert not rejected(run("out3", tmp_path / "cache-live")), \
        "reading the live files instead passes the swapped-in good bytes: the private copies are what protect the verdict"


def test_a_seal_takes_a_fresh_directory_so_a_reused_pid_reads_no_stale_copy(tmp_path):
    """The hole a seal under ``base/<pid>`` would still have: a killed run leaves its copies there, and a new run under the
    same pid (container pids are low, a slot persists) would read them, since `Sealed.copy` skips a file already present.
    `Seals` takes a fresh `mkdtemp` directory per run instead, with the pid only a prefix so `sweep` still drops a dead
    run's directory.  Here a stale `base/<pid>` directory holds a wrong `.olean` where that layout put it; the seal the run
    takes reads this build's bytes, not the stale copy, and `sweep` removes a gone process's directory in either naming."""
    base = tmp_path / "sealed"
    pkg = tmp_path / "pkg"
    lib = pkg / ".lake" / "build" / "lib" / "lean"
    lib.mkdir(parents=True)
    olean = lib / "Foo.olean"
    olean.write_bytes(b"this build's bytes")
    stale = base / f"{os.getpid()}"                              # a killed run's leftover, the old `base/<pid>` layout
    stale_copy = A.Sealed(pkg, "x", stale / "pkg-0").moved(olean.resolve())
    stale_copy.parent.mkdir(parents=True)
    stale_copy.write_bytes(b"a stale run's wrong bytes")
    seals = A.Seals(base)
    try:
        assert seals.dir != stale and seals.dir.name.startswith(f"{os.getpid()}-"), \
            "the run takes a fresh directory whose pid is only a prefix, not the reused-pid leftover"
        read = A.Sealed(pkg, "x", seals.dir / "pkg-0")
        read.copy(olean.resolve())
        assert read.moved(olean.resolve()).read_bytes() == b"this build's bytes", \
            "the seal reads this build's `.olean`, not the stale copy a reused pid left"
    finally:
        seals.close()
    gone = tmp_path / "noproc"
    gone.mkdir()
    (base / "999999-abcd").mkdir()                               # a dead run's directory, the new `<pid>-<suffix>` layout
    A.Seals(base).sweep(proc=gone)
    assert not stale.exists() and not (base / "999999-abcd").exists(), \
        "sweep removes a directory whose pid prefix names a gone process, old `<pid>` and new `<pid>-<suffix>` alike"


def test_two_packages_of_one_name_sealed_side_by_side_get_a_directory_each(tmp_path, monkeypatch):
    """`pous/check` and `pouw/lean/check` took their seals at once and both copied into `check-25`, numbered by the
    seals finished, so one's facts looked for an `.olean` under the other's numbering of the roots (a missing object
    file). Here the second seal is taken while the first is copying."""
    a, b = tmp_path / "pous" / "check", tmp_path / "pouw" / "lean" / "check"
    seals = A.Seals(tmp_path / "sealed")
    dirs = []

    def take(self, roots, exempt, work, log=print):
        dirs.append(self.dir)
        if self.pkg == a:
            seals.take(b, "x", [], [], work)
        return self

    monkeypatch.setattr(A.Sealed, "take", take)
    try:
        seals.take(a, "x", [], [], tmp_path)
        assert len(dirs) == 2 and dirs[0] != dirs[1], dirs
        assert seals.take(a, "x", [], [], tmp_path).dir == dirs[0] and len(dirs) == 2, "a package seals once"
    finally:
        seals.close()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_seal_skips_an_exempt_module_with_no_olean(tmp_path):
    """A regression the seal had: it seeded every module under the lock's roots, an `exempt` one under them
    included, but an exempt module (a scratch file, a `runs` test) is not built, so `Graph.lean` was sent after an
    `.olean` that is not there and the seal, and every audit of the package, failed.  Here a package has an exempt module
    `Control.Stub` under its root that nothing imports or builds; the sealed audit passes, and the seal seeds and copies
    no `Control/Stub.olean`.  A non-exempt module with no `.olean` still fails, by name, with the fix in the message."""
    S = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", S, ignore=shutil.ignore_patterns(".lake"))
    (S / "Control" / "Stub.lean").write_text("namespace Control\n\ndef stub : Nat := 0\n")
    pol = A.load(S)
    A.dump(S, pol | {"exempt": pol["exempt"] | {"Control.Stub": "a stub under the root that nobody builds"}})
    subprocess.run(["lake", "build"], cwd=S, check=True, capture_output=True, env=A.lake_env())
    stub_olean = S / ".lake" / "build" / "lib" / "lean" / "Control" / "Stub.olean"
    assert not stub_olean.is_file(), "the stub is unbuilt: nothing imports it, so the test exercises an exempt module with no `.olean`"
    seals = A.Seals(tmp_path / "sealed")
    try:
        rep = A.audit(S, tmp_path / "out", update=True, seals=seals, replay=False, runs=False, log=lambda *_: None)
        assert rep["failures"] == [], rep["failures"]
        assert list(seals.dir.rglob("Control/Stub.olean")) == [], "the seal copied the exempt module it should not seed"
        # a module under the roots that is *not* exempt and has no `.olean` still fails, naming the module and the fix
        A.dump(S, pol)
        seals2 = A.Seals(tmp_path / "sealed2")
        try:
            bad = A.audit(S, tmp_path / "out2", seals=seals2, replay=False, runs=False, log=lambda *_: None)
            assert any("Control.Stub" in f and "under `exempt`" in f for f in bad["failures"]), bad["failures"]
        finally:
            seals2.close()
    finally:
        seals.close()


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
def test_the_closure_replay_checks_each_imported_module_once_and_then_reuses_it(tmp_path, sandboxed_node):
    """On a real build: `--closure` sends every module the control imports (the toolchain's `Init`) through the kernel, a
    second audit checks none of them, and a change to an own module checks only it and what imports it."""
    pkg, cache = tmp_path / "control", tmp_path / "replay-cache"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    def build():
        return subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    def audit():                                             # store=True: the cache is this test's own private dir
        return A.audit(pkg, tmp_path / "out", update=True, runs=False, log=lambda *_: None, replay_cache=cache,
                       closure=True, store=True)
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
    ArkLib's entries, stored once and reused by every replay, would be the first evicted, so Security's and Properties'
    replays would check thousands of closure modules again. A replay marks
    each entry it reuses, its own modules' and the closure's, and stores those it checked; nothing else is touched."""
    graph = {"Init": [], "Std": ["Init"], "Old": ["Init"], "A": ["Std"], "B": ["A"]}
    facts = {"graph": {m: {"imports": i} for m, i in graph.items()}, "modules": ["A", "B"], "axioms": []}
    cache = tmp_path / "replay"
    cache.mkdir()
    for m in ("Init", "Old", "A", "Unrelated"):
        (cache / f"{m}.json").write_text("{}")
        os.utime(cache / f"{m}.json", (1000, 1000))
    monkeypatch.setattr(A, "replay_keys", lambda facts, toolchain, digest=None: {m: m for m in facts["graph"]})
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


def test_an_unsealed_replay_reuses_the_cache_but_writes_nothing(tmp_path, monkeypatch):
    """An unsealed audit (a control, a stubbed test) hashes its replay keys from the live `.lake`, so
    a stored verdict a dependent's build could have tampered with would poison a later sealed run. With ``store`` false it
    may reuse the shared cache but writes nothing: neither a new entry for a module it checked, nor a touch of a reused
    entry's mtime. With ``store`` true (a sealed audit) it stores and touches, so the test is not vacuous."""
    facts = {"graph": {"A": {"imports": []}, "B": {"imports": ["A"]}}, "modules": ["A", "B"], "axioms": []}
    monkeypatch.setattr(A, "replay_keys", lambda facts, toolchain, digest=None: {m: m for m in facts["graph"]})
    monkeypatch.setattr(A, "replay_threads", lambda pkg: None)

    def run(cmd, cwd, log, timeout=None):
        (tmp_path / "replay.json").write_text(json.dumps({"axioms": []}))
        return 0, ""

    monkeypatch.setattr(A, "run", run)

    def warm(name):                              # a cache with A already passed, which the replay may reuse
        c = tmp_path / name
        c.mkdir()
        (c / "A.json").write_text("{}")
        os.utime(c / "A.json", (1000, 1000))
        return c

    unsealed = warm("unsealed")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime) for p in unsealed.iterdir()}
    fails, stats = A.replay_step(tmp_path, {"roots": ["A"]}, facts, "t", tmp_path, unsealed, store=False)
    assert fails == [] and (stats["checked"], stats["reused"]) == (1, 1)      # B checked, A reused, from the live keys
    assert {p.name: (p.read_bytes(), p.stat().st_mtime) for p in unsealed.iterdir()} == before

    sealed = warm("sealed")
    A.replay_step(tmp_path, {"roots": ["A"]}, facts, "t", tmp_path, sealed, store=True)
    assert (sealed / "B.json").is_file() and (sealed / "A.json").stat().st_mtime > 1000


def test_an_unsealed_audit_leaves_the_shared_replay_cache_unchanged(tmp_path, monkeypatch, sandboxed_node):
    """lean's rule: an audit without seals keys its replay on the live `.lake`, which a dependent's build can tamper with,
    so it may reuse the shared cache but must write nothing to it, or a tampered verdict could poison a sealed run. An
    unsealed `audit` runs a real `replay_step` against a warm cache and leaves every entry, byte and mtime, as it was; the
    same audit with store=True (its cache a throwaway private dir) does store what it checked."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "lean-toolchain").write_text("v0\n")
    (pkg / "P.lean").write_text("")
    A.dump(pkg, {"roots": ["P"], "dependencies": {}})
    own = {"lean": "v0", "modules": ["P"], "declarations": 1, "axioms": [], "escapes": [], "oleans": [],
           "graph": {"P": {"imports": []}}, "files": {"P": "P.lean"}, "pins": {}}
    monkeypatch.setattr(A, "facts_of", lambda *a, keep=True, sealed=None: dict(own))
    monkeypatch.setattr(A, "replay_keys", lambda facts, toolchain, digest=None: {"P": "pkey"})
    monkeypatch.setattr(A, "replay_threads", lambda pkg: None)

    def run(cmd, cwd, log, timeout=None):
        A.Path(cmd[cmd.index("--run") + 2]).write_text(json.dumps({"axioms": []}))
        return 0, ""

    monkeypatch.setattr(A, "run", run)
    monkeypatch.setattr(A, "dependency_scans", lambda *a: ([], 0))
    monkeypatch.setattr(A, "dependency_digests", lambda *a: {})
    for name, value in (("judge", ([], {})), ("dropped", []), ("spec_reads", []), ("shape", []), ("unread_spec", {}),
                        ("read_by_dependents", set()), ("reads_home", {}), ("scans", []), ("heartbeats", []),
                        ("anonymous_instances", []), ("run_paths", []), ("tree_roles", {}), ("tree_failures", []),
                        ("inline_failures", ([], {}, []))):
        monkeypatch.setattr(A, name, lambda *a, value=value, **k: value)

    def warm(name):
        c = tmp_path / name
        c.mkdir()
        (c / "w.json").write_text("{}")
        os.utime(c / "w.json", (1000, 1000))
        return c

    unsealed = warm("unsealed")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime) for p in unsealed.iterdir()}
    assert A.audit(pkg, tmp_path / "out", replay=True, runs=False, replay_cache=unsealed, log=lambda *_: None)["failures"] == []
    assert {p.name: (p.read_bytes(), p.stat().st_mtime) for p in unsealed.iterdir()} == before

    private = warm("private")
    A.audit(pkg, tmp_path / "out2", replay=True, runs=False, replay_cache=private, store=True, log=lambda *_: None)
    assert (private / "pkey.json").is_file()


def test_two_packages_storing_one_replay_key_at_once_both_succeed(tmp_path, monkeypatch):
    """The gate audits its packages in threads of one process, and two whose closures share a module store its replay key
    at once; with the temporary file named by the pid alone, the second store would rename the first's away and the first's
    rename would fail. The store between the first's write and its rename runs on another thread."""
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


def test_free_memory_is_the_least_of_the_machines_available_and_what_each_cgroup_has_left(tmp_path):
    """`free_gib`: `MemAvailable`, bounded by each limited cgroup at or above this one, less what it uses but its
    inactive page cache."""
    proc, cg = tmp_path / "proc", tmp_path / "cg"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text("MemTotal: 1000000000 kB\nMemAvailable: 500000000 kB\n")
    (proc / "self" / "cgroup").write_text("0::/a/b\n")
    (cg / "a" / "b").mkdir(parents=True)
    (cg / "a" / "memory.max").write_text(str(300 * 2**30))
    (cg / "a" / "memory.current").write_text(str(200 * 2**30))
    (cg / "a" / "memory.stat").write_text(f"anon 1\ninactive_file {50 * 2**30}\n")
    (cg / "a" / "b" / "memory.max").write_text("max")
    assert A.free_gib(proc, cg) == 150
    (cg / "a" / "memory.max").write_text("max")
    assert round(A.free_gib(proc, cg)) == 477


def test_run_hands_over_each_line_of_standard_output_as_it_comes_and_returns_the_whole(tmp_path):
    lines = []
    rc, text = A.run([sys.executable, "-c", "import sys; print('a'); print('b'); print('e', file=sys.stderr)"], tmp_path,
                     tmp_path / "log", on_line=lines.append)
    assert rc == 0 and lines == ["a\n", "b\n"] and text == "a\nb\ne\n" == (tmp_path / "log").read_text()

    def fails(line):
        lines.append(line)
        raise ValueError(line)
    lines.clear()
    rc, text = A.run([sys.executable, "-c", "print('a\\n' * 100000, end='')"], tmp_path, tmp_path / "log", on_line=fails)
    assert rc == 0 and lines == ["a\n"] and text == "a\n" * 100000, "one that raises is dropped, and the pipe still drained"


def pipeline(tmp_path, monkeypatch, files, loop=False, **kw):
    """A `Pipeline` of a package holding ``files`` under the root `P`; without ``loop``, no batch runs on its own."""
    pkg = tmp_path / "pkg"
    for rel in files:
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text("")
    if not loop:
        monkeypatch.setattr(A.Pipeline, "_loop", lambda self: None)
    return A.Pipeline(pkg, {"roots": ["P"]}, "t", tmp_path / "work", tmp_path / "cache", False, **kw)


def test_a_pipeline_batches_the_modules_the_build_reports_once_enough_wait_and_memory_and_the_heavy_slot_are_free(
        tmp_path, monkeypatch):
    """A batch takes the package's modules under its roots that Lake reports built, not a facet, another package's module
    or one outside the roots, once `batch` of them wait, `PIPELINE_FREE_GIB` is free and `_HEAVY_REPLAY` is free, which it
    never waits for; after `end`, none starts."""
    free = [10**6]
    monkeypatch.setattr(A, "free_gib", lambda: free[0])
    p = pipeline(tmp_path, monkeypatch, ["P/A.lean", "P/B.lean", "Other.lean"], batch=2)
    for line in ("✔ [1/9] Built P.A (1.2s)\n", "✔ [2/9] Built P.A:c.o (900ms)\n", "✔ [3/9] Built Mathlib.X (2s)\n",
                 "✔ [4/9] Built Other (2s)\n", "✔ [5/9] Replayed P.B\n"):
        p.line(line)
    assert p.pending == ["P.A"] and not p._go()
    p.line("✔ [6/9] Built P.B (1m 2.5s)\n")
    free[0] = A.PIPELINE_FREE_GIB - 1
    assert not p._go()
    free[0] = A.PIPELINE_FREE_GIB
    assert A._HEAVY_REPLAY.acquire(blocking=False)
    try:
        assert not p._go() and not p.held
    finally:
        A._HEAVY_REPLAY.release()
    assert p._go() and p.held
    A._HEAVY_REPLAY.release()
    p.held = False
    p.end()
    assert p._go() and not p.held


def test_a_pipeline_runs_one_batch_at_a_time_in_the_heavy_slot_and_none_after_the_build(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "free_gib", lambda: 10**6)
    ran, started = [], threading.Event()

    def replay(self, mods):
        if free := A._HEAVY_REPLAY.acquire(blocking=False):
            A._HEAVY_REPLAY.release()
        ran.append((list(mods), not free))
        started.set()

    monkeypatch.setattr(A.Pipeline, "_replay", replay)
    p = pipeline(tmp_path, monkeypatch, ["P/A.lean", "P/B.lean", "P/C.lean"], loop=True, batch=2)
    p.line("Built P.A (1s)\n")
    p.line("Built P.B (1s)\n")
    started.wait()
    p.line("Built P.C (1s)\n")
    p.end()
    p.thread.join()
    assert ran == [(["P.A", "P.B"], True)] and p.pending == ["P.C"]
    assert A._HEAVY_REPLAY.acquire(blocking=False)
    A._HEAVY_REPLAY.release()


def test_the_final_replay_waits_for_the_batch_running_when_the_build_ended(tmp_path, monkeypatch):
    """Under `lean_audit.py`'s `LEAN_NUM_THREADS` the final replay takes no `_HEAVY_REPLAY`, so `wait` is what keeps it from
    checking, beside the last batch, the modules that batch is about to store."""
    monkeypatch.setattr(A, "free_gib", lambda: 10**6)
    order, started, release = [], threading.Event(), threading.Event()

    def replay(self, mods):
        started.set()
        release.wait()
        order.append("batch stored")

    monkeypatch.setattr(A.Pipeline, "_replay", replay)
    p = pipeline(tmp_path, monkeypatch, ["P/A.lean"], loop=True, batch=1)
    p.line("Built P.A (1s)\n")
    started.wait()
    p.end()
    final = threading.Thread(target=lambda: (p.wait(), order.append("final replay")))
    final.start()
    assert final.is_alive() and order == []
    release.set()
    final.join()
    assert order == ["batch stored", "final replay"]


def test_a_batch_whose_modules_and_their_closure_all_passed_runs_no_replay(tmp_path, monkeypatch):
    """A cold build reports every module built, most of them with the bytes an earlier replay passed: a batch whose modules,
    and with `--closure` every module they import, are in the cache under its copies' keys starts no `Replay.lean`, which
    would only load them. One left to check, and it replays."""
    p = pipeline(tmp_path, monkeypatch, ["P/A.lean", "P/B.lean"])
    p.closure = True
    imports = {"D": [], "P.A": ["D"], "P.B": ["P.A"]}
    for m in imports:
        (tmp_path / f"{m}.olean").write_text(m)
    facts = {"graph": {m: {"olean": str(tmp_path / f"{m}.olean"), "imports": i} for m, i in imports.items()},
             "modules": ["P.A", "P.B"], "axioms": []}
    monkeypatch.setattr(A.Pipeline, "copied", lambda self, mods, work: (facts, []))
    calls = []
    monkeypatch.setattr(A, "replay_step", lambda *a, **k: calls.append(a[2]["modules"]) or
                        ([], {"checked": 2, "phases": {}, "peak_gib": 0}))
    keys = A.replay_keys(facts, "t")
    p.replay_cache.mkdir()
    for m in ("P.A", "P.B"):
        A.store_replay(p.replay_cache, keys[m], m)
    p._replay(["P.A", "P.B"])
    assert calls == [["P.A", "P.B"]], "D is not stored"
    A.store_replay(p.replay_cache, keys["D"], "D")
    p._replay(["P.A", "P.B"])
    assert calls == [["P.A", "P.B"]] and p.batches[-1]["checked"] == 0 and "error" not in p.batches[-1]


def test_a_batch_leaves_a_listed_module_and_what_imports_it_to_the_final_replay(tmp_path, monkeypatch):
    """A batch runs at the batch's threads, and with `--closure` checks what its modules import: so a module the replay
    is to check in a step of its own (`replay_first`) is in no batch, and neither is a module that imports it, until that
    step has stored it. The batch's facts are then those of the modules left, closed under their imports."""
    p = pipeline(tmp_path, monkeypatch, ["P/A.lean", "P/Heavy.lean", "P/Uses.lean", "P/Top.lean"])
    p.closure = True
    imports = {"D": [], "P.A": ["D"], "P.Heavy": ["D"], "P.Uses": ["P.Heavy"], "P.Top": ["P.Uses", "P.A"]}
    for m in imports:
        (tmp_path / f"{m}.olean").write_text(m)
    facts = {"graph": {m: {"olean": str(tmp_path / f"{m}.olean"), "imports": i} for m, i in imports.items()},
             "modules": ["P.A", "P.Heavy", "P.Uses", "P.Top"], "axioms": []}
    monkeypatch.setattr(A.Pipeline, "copied", lambda self, mods, work: (facts, []))
    monkeypatch.setattr(A, "replay_first", lambda pkg: [{"module": "P.Heavy", "threads": 1, "why": "w"},
                                                        {"module": "P.Elsewhere", "threads": 1, "why": "w"}])
    calls = []
    monkeypatch.setattr(A, "replay_step", lambda *a, **k: calls.append((a[2]["modules"], sorted(a[2]["graph"]))) or
                        ([], {"checked": len(a[2]["modules"]), "phases": {}, "peak_gib": 0}))
    p.replay_cache.mkdir()
    p._replay(list(facts["modules"]))
    assert calls == [(["P.A"], ["D", "P.A"])] and p.batches[-1]["left"] == 3 and p.batches[-1]["modules"] == 4
    p._replay(["P.Heavy", "P.Uses"])
    assert len(calls) == 1 and p.batches[-1] | {"seconds": 0} == {"modules": 2, "left": 2, "checked": 0, "seconds": 0}
    A.store_replay(p.replay_cache, A.replay_keys(facts, "t")["P.Heavy"], "P.Heavy")
    p._replay(list(facts["modules"]))
    assert calls[1:] == [(facts["modules"], sorted(imports))] and "left" not in p.batches[-1]

    def unreadable(pkg):
        raise RuntimeError("the list does not read")
    monkeypatch.setattr(A, "replay_first", unreadable)
    p._replay(["P.A"])
    assert len(calls) == 2 and "the list does not read" in p.batches[-1]["error"], "the final replay reports it"


def test_a_batch_reads_copies_of_the_files_the_build_can_write_keyed_as_those_bytes_in_place(tmp_path, monkeypatch):
    """A batch's `.olean` files under a `.lake` the build writes (the package's, a fetched package's) are copied once into
    the snapshot, every part, and named there, the search path too; the toolchain's stay where they are. Their keys are
    those of the same bytes in place, so the final replay finds what a batch stored; a file the build rewrites after its
    copy leaves the copy, and the keys of every later batch, as they were, and the final replay finds none of them."""
    p = pipeline(tmp_path, monkeypatch, ["P/A.lean", "P/B.lean"])
    lib = p.pkg / ".lake" / "build" / "lib" / "lean"
    dep = p.pkg / ".lake" / "packages" / "dep" / ".lake" / "build" / "lib" / "lean"
    tc = (tmp_path / "toolchain" / "lib" / "lean").resolve()
    files = {"P.A": lib / "P" / "A.olean", "P.B": lib / "P" / "B.olean", "D": dep / "D.olean", "Init": tc / "Init.olean"}
    imports = {"P.A": ["D", "Init"], "P.B": ["P.A"], "D": ["Init"], "Init": []}
    for m, f in files.items():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(m)
    (lib / "P" / "A.olean.private").write_text("A's private part")
    (lib / "P" / "A.ir").write_text("A's code")

    def run(argv, cwd, log=None, timeout=None, on_line=None):
        Path(argv[argv.index(str(A.TOOL / "Graph.lean")) + 1]).write_text(json.dumps({
            "search": [str(lib), str(dep), str(tc)],
            "graph": {m: {"olean": str(f), "imports": imports[m]} for m, f in files.items()}}))
        return 0, ""

    monkeypatch.setattr(A, "run", run)
    live = {"graph": {m: {"olean": str(f), "imports": imports[m]} for m, f in files.items()}}
    for b in ("b1", "b2"):
        (tmp_path / b).mkdir()
    facts, search = p.copied(["P.A", "P.B"], tmp_path / "b1")
    g = facts["graph"]
    assert facts["modules"] == ["P.A", "P.B"] and g["Init"]["olean"] == str(files["Init"]) and search[2] == str(tc)
    for m in ("P.A", "P.B", "D"):
        assert Path(g[m]["olean"]).is_relative_to(p.snapshot) and Path(g[m]["olean"]).read_text() == m
    assert Path(g["P.A"]["olean"] + ".private").read_text() == "A's private part"
    assert Path(g["P.A"]["olean"]).with_suffix(".ir").read_text() == "A's code"
    assert Path(search[0]) / "P" / "A.olean" == Path(g["P.A"]["olean"]) and Path(search[1]) / "D.olean" == Path(g["D"]["olean"])
    keys = A.replay_keys(facts, "t")
    assert keys == A.replay_keys(live, "t")
    files["D"].write_text("rewritten by the build")
    later, _ = p.copied(["P.B"], tmp_path / "b2")
    assert Path(later["graph"]["D"]["olean"]).read_text() == "D" and A.replay_keys(later, "t") == keys
    assert A.replay_keys(live, "t")["P.B"] != keys["P.B"]


def graph_reports(monkeypatch, search: list[str], graph: dict) -> None:
    """Make `Graph.lean` report ``search`` and ``graph`` (paths as strings), so `copied` sees a build's layout without one."""
    def run(argv, cwd, log=None, timeout=None, on_line=None):
        Path(argv[argv.index(str(A.TOOL / "Graph.lean")) + 1]).write_text(json.dumps({"search": search, "graph": graph}))
        return 0, ""
    monkeypatch.setattr(A, "run", run)


def test_a_batch_abandons_a_symlinked_olean_whose_target_escapes_its_search_directory(tmp_path, monkeypatch):
    """top's A/M swap: a build writes `A.olean` in a writable directory as a symlink to a file outside it. `lean` finds the
    module by name there and loads the target, but a copy's key must be of bytes under the directory `lean` searched, so
    `copied` abandons the batch, which stores nothing; the final replay then checks the module."""
    p = pipeline(tmp_path, monkeypatch, ["P/A.lean"])
    d0 = p.pkg / ".lake" / "build" / "lib" / "lean"
    d1 = p.pkg / ".lake" / "alt" / "lib" / "lean"
    (d1 / "P").mkdir(parents=True)
    (d1 / "P" / "M.olean").write_text("M, outside A's directory")
    (d0 / "P").mkdir(parents=True)
    (d0 / "P" / "A.olean").symlink_to(d1 / "P" / "M.olean")
    graph_reports(monkeypatch, [str(d0), str(d1)], {"P.A": {"olean": str(d0 / "P" / "A.olean"), "imports": []}})
    p._replay(["P.A"])
    assert "resolves outside" in p.batches[-1]["error"] and "P.A" in p.batches[-1]["error"]
    assert not list(p.replay_cache.glob("*.json"))


def test_a_batch_abandons_a_read_in_place_module_that_resolves_into_a_writable_root(tmp_path, monkeypatch):
    """ci's moved module: a module found in a directory no build writes is read in place, uncopied, so its key is of the
    bytes there. If its olean is a symlink into a writable root the prover controls, those bytes are not fixed, so `copied`
    abandons the batch, which stores nothing; the final replay checks it."""
    p = pipeline(tmp_path, monkeypatch, ["P/A.lean"])
    ro = tmp_path / "ro" / "lib" / "lean"            # a directory no build writes: read in place, uncopied
    w = p.pkg / ".lake" / "build" / "lib" / "lean"   # a writable root
    (w / "Lib").mkdir(parents=True)
    (w / "Lib" / "M.olean").write_text("the prover's bytes")
    (ro / "Lib").mkdir(parents=True)
    (ro / "Lib" / "M.olean").symlink_to(w / "Lib" / "M.olean")
    graph_reports(monkeypatch, [str(ro)], {"Lib.M": {"olean": str(ro / "Lib" / "M.olean"), "imports": []}})
    p._replay(["Lib.M"])
    assert "writable root" in p.batches[-1]["error"] and "Lib.M" in p.batches[-1]["error"]
    assert not list(p.replay_cache.glob("*.json"))


def test_a_batch_abandons_a_module_that_shadows_a_toolchain_module(tmp_path, monkeypatch):
    """A package module with a toolchain module's name is found in both the writable package directory and the toolchain's,
    so which one `lean` loads turns on search order, not on a single key's bytes. `copied` abandons the batch, which stores
    nothing; the final replay checks it."""
    p = pipeline(tmp_path, monkeypatch, ["P/A.lean"])
    lib = p.pkg / ".lake" / "build" / "lib" / "lean"
    tc = tmp_path / "toolchain" / "lib" / "lean"
    for d in (lib, tc):
        d.mkdir(parents=True)
        (d / "Init.olean").write_text(f"init at {d}")
    graph_reports(monkeypatch, [str(lib), str(tc)], {"Init": {"olean": str(lib / "Init.olean"), "imports": []}})
    p._replay(["Init"])
    assert "shadowed" in p.batches[-1]["error"] and "Init" in p.batches[-1]["error"]
    assert not list(p.replay_cache.glob("*.json"))


def kernel_control(pkg: Path) -> dict:
    """``pkg``, a copy of the control, given `violations.json`'s `kernel` case: a declaration that skipped the kernel."""
    kernel = json.loads((HERE / "fixtures" / "violations.json").read_text())["kernel"]
    for rel, text in kernel["write"].items():
        (pkg / rel).write_text(text)
    for rel, (old, new) in kernel["replace"].items():
        (pkg / rel).write_text((pkg / rel).read_text().replace(old, new))
    A.dump(pkg, A.load(pkg) | kernel["policy"])
    return kernel


def piped_build(pkg: Path, tmp_path: Path, monkeypatch, closure: bool) -> A.Pipeline:
    """``pkg`` built, its build's lines given to a `Pipeline` that stores into `tmp_path/piped`, and one batch run on
    every module the build reported."""
    monkeypatch.setattr(A.Pipeline, "_loop", lambda self: None)
    pipe = A.Pipeline(pkg, A.load(pkg), (pkg / "lean-toolchain").read_text().strip(), tmp_path / "work",
                      tmp_path / "piped", closure)
    rc, text = A.run(["lake", "build"], pkg, tmp_path / "build.log", on_line=pipe.line)
    assert rc == 0, text
    pipe._replay(list(pipe.pending))
    return pipe


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_pipelined_replay_checks_and_rejects_what_the_replay_without_it_does(tmp_path, monkeypatch, sandboxed_node):
    """On a real build with a declaration that skipped the kernel: a batch on every module the build reported, run on
    copies of their files, stores the four the kernel accepted, `Control` (which imports the bad one) included. The replay
    after the build then checks `Control.Bad` alone, reuses the four, and fails as the replay without a batch fails on a
    cold cache, which checks all five: the same modules, and the same rejection. Only a real build shows Lake's lines,
    `Graph.lean`'s graph and the keys of the copies meeting those of the files in place."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    kernel = kernel_control(pkg)
    pipe = piped_build(pkg, tmp_path, monkeypatch, closure=False)
    own = sorted(m for m in A.lean_modules(pkg) if A.under(["Control"], m))
    (batch,) = pipe.batches
    assert "error" not in batch and batch["modules"] == batch["checked"] == len(own) == 5, batch
    assert any(f.startswith(kernel["fails"]) for f in batch["failures"]), batch

    def audit(cache):
        return A.audit(pkg, tmp_path / cache / "out", update=True, runs=False, log=lambda *_: None,
                       replay_cache=tmp_path / cache)

    def rejections(rep):
        return [f.split("\n")[0] for f in rep["failures"] if f.startswith("replay")]
    serial, piped = audit("serial"), audit("piped")
    assert counts(serial) == (5, 0) and counts(piped) == (1, 4), "Control.Bad alone"
    assert rejections(piped) == rejections(serial) and rejections(serial)[0].startswith(kernel["fails"])


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_pipelined_closure_replay_of_module_system_files_leaves_the_final_one_nothing_to_check(tmp_path, monkeypatch,
                                                                                                  sandboxed_node):
    """With `--closure`, on `module` files, whose private parts the batch copies: the batch checks the own modules and the
    toolchain modules they import, and the replay after the build reuses every one of them. Only the closure's cold
    replay (the toolchain's `Init`) shows a batch storing closure modules the final replay then finds."""
    pkg = tmp_path / "modules"
    shutil.copytree(A.FIXTURES / "modules", pkg, ignore=shutil.ignore_patterns(".lake"))
    pipe = piped_build(pkg, tmp_path, monkeypatch, closure=True)
    (batch,) = pipe.batches
    assert "error" not in batch and batch["failures"] == [] and batch["closure_checked"] > 100, batch
    rep = A.audit(pkg, tmp_path / "out", update=True, runs=False, log=lambda *_: None, replay_cache=tmp_path / "piped",
                  closure=True)
    assert not [f for f in rep["failures"] if f.startswith("replay")], rep["failures"]
    assert counts(rep) == (0, batch["checked"]) and rep["replay"]["closure"]["checked"] == 0, rep["replay"]


def test_a_replay_plan_checks_each_listed_module_first_alone_at_its_threads_and_then_the_rest():
    """`replay_plan`: a step for each listed module the package audits that has not passed, in the list's order, alone,
    at its stated threads and never more than the replay's own; then the rest at the replay's own. An entry that names
    no audited module, or one that passed, plans no step, and a package with no entry has the one step."""
    first = [{"module": "P.Heavy", "threads": 2, "why": "w"}, {"module": "P.Wide", "threads": 64, "why": "w"},
             {"module": "P.Gone", "threads": 1, "why": "w"}, {"module": "P.Done", "threads": 1, "why": "w"}]
    modules = ["P.A", "P.Wide", "P.Heavy", "P.Done", "P.B", "P.Old"]
    passed = {"P.Done", "P.Old"}.__contains__
    assert A.replay_plan(modules, first, passed, 16) == [(["P.Heavy"], 2), (["P.Wide"], 16), (["P.A", "P.B"], 16)]
    assert A.replay_plan(modules, first, passed, 1) == [(["P.Heavy"], 1), (["P.Wide"], 1), (["P.A", "P.B"], 1)]
    assert A.replay_plan(modules, [], passed, 16) == [(["P.A", "P.Wide", "P.Heavy", "P.B"], 16)]
    assert A.replay_plan(modules, first, lambda m: True, 16) == [([], 16)]


def test_the_modules_a_replay_checks_first_are_read_from_the_file_beside_the_tool(tmp_path, monkeypatch):
    """`replay_first`: a package's entries of `replay-first.json`, by its path from the repository's root, in the file's
    order; none for another package, for one outside the repository, or where the file is absent. A file of another
    schema, or with an entry that lacks its reason, states no count of threads or names a module twice, is refused with
    what the file holds, and `replay_in_steps` then fails the package rather than check it in one step unseen. The
    tool's own file reads."""
    monkeypatch.setattr(A, "ROOT", tmp_path)
    pkg, path = tmp_path / "a" / "pkg", tmp_path / A.REPLAY_FIRST
    heavy = {"module": "P.Heavy", "threads": 2, "why": "its declarations each hold much memory in the kernel"}
    wide = {"module": "P.Wide", "threads": 1, "why": "the same"}
    def write(packages, schema=A.REPLAY_FIRST_SCHEMA):
        path.write_text(json.dumps({"schema": schema, "packages": packages}))
    write({"a/pkg": [heavy, wide], "b": [wide]})
    assert A.replay_first(pkg, path) == [heavy, wide] and A.replay_first(tmp_path / "b", path) == [wide]
    assert A.replay_first(tmp_path / "c", path) == [] and A.replay_first(tmp_path.parent / "outside", path) == []
    assert A.replay_first(pkg, tmp_path / "absent.json") == []
    for bad in ({"a/pkg": [heavy | {"threads": 0}]}, {"a/pkg": [heavy | {"threads": "2"}]}, {"a/pkg": [heavy | {"threads": True}]},
                {"a/pkg": [heavy | {"why": " "}]}, {"a/pkg": [{k: v for k, v in heavy.items() if k != "why"}]},
                {"a/pkg": [heavy | {"note": "more"}]}, {"a/pkg": [heavy, heavy | {"threads": 1}]}, {"a/pkg": ["P.Heavy"]},
                [heavy]):
        write(bad)
        with pytest.raises(RuntimeError, match="replay-first.json does not read as"):
            A.replay_first(pkg, path)
    write({"a/pkg": [heavy]}, schema="lean-replay-first/v0")
    with pytest.raises(RuntimeError, match="does not read as"):
        A.replay_first(pkg, path)
    path.write_text("{")
    with pytest.raises(RuntimeError, match="JSONDecodeError"):
        A.replay_first(pkg, path)
    monkeypatch.setattr(A, "TOOL", tmp_path)
    fails, stats = A.replay_in_steps(pkg, {"roots": ["P"]}, {}, "t", tmp_path, None)
    assert [f.code for f in fails] == ["lean"] and "replay-first.json does not read as" in fails[0] and stats["checked"] == 0
    monkeypatch.undo()
    own = json.loads((A.HERE / A.REPLAY_FIRST).read_text())
    assert own["schema"] == A.REPLAY_FIRST_SCHEMA
    assert [A.replay_first(A.ROOT / p) for p in own["packages"]] == list(own["packages"].values())


def test_a_replay_checks_each_listed_module_in_a_step_of_its_own_and_the_last_step_reuses_it(tmp_path, monkeypatch):
    """`replay_in_steps` on a stubbed `Replay.lean`, with two of the package's modules listed and one it does not have.
    Each listed module goes to a `Replay.lean` that names it alone, at its stated threads and never more than the
    replay's own, stated to `lean` as its `-j` (the pool of its tasks is the machine's without it), and is stored under
    the key the whole replay computes; the last step names them `--reused` and checks the rest as a replay without a
    list does, with no `-j`. The stats count each stored module as checked, so the modules checked, the
    failures and the cache's entries are the one step's. A listed step the kernel rejects stores nothing, so the last
    step checks its module with the rest and fails as the one step does. A second replay plans no step. Without a cache
    it may write, the replay is the one step and says so."""
    graph = {"Init": [], "P.A": ["Init"], "P.Heavy": ["P.A"], "P.Wide": ["Init"], "P.B": ["P.Heavy", "P.Wide"]}
    facts = {"graph": {m: {"imports": i} for m, i in graph.items()}, "modules": ["P.A", "P.Heavy", "P.Wide", "P.B"],
             "axioms": []}
    first = [{"module": "P.Wide", "threads": 64, "why": "w"}, {"module": "P.Heavy", "threads": 2, "why": "w"},
             {"module": "P.Gone", "threads": 1, "why": "w"}]
    monkeypatch.setattr(A, "replay_keys", lambda facts, toolchain, digest=None: {m: m for m in facts["graph"]})
    monkeypatch.setattr(A, "replay_threads", lambda pkg: None)
    monkeypatch.setenv("LEAN_NUM_THREADS", "8")
    calls, rejected = [], set()

    def run(cmd, cwd, log, timeout=None):
        rest = cmd[cmd.index("--run") + 3:]
        cut = rest.index("--reused") if "--reused" in rest else len(rest)
        flag = cmd[cmd.index("-j") + 1] if "-j" in cmd[:cmd.index("--run")] else None
        assert flag == next((a.split("=")[1] for a in cmd if a.startswith("LEAN_NUM_THREADS=")), None), cmd
        calls.append((flag, rest[:cut], rest[cut + 1:]))
        bad = sorted(rejected & set(rest[:cut]))
        A.Path(cmd[cmd.index("--run") + 2]).write_text(json.dumps({"rejected": bad} if bad else {"axioms": []}))
        return (1 if bad else 0), ""

    monkeypatch.setattr(A, "run", run)

    def replay(cache, listed=first, **kw):
        monkeypatch.setattr(A, "replay_first", lambda pkg: listed)
        calls.clear()
        fails, stats = A.replay_in_steps(tmp_path, {"roots": ["P"]}, facts, "t", tmp_path, cache and tmp_path / cache, **kw)
        entries = sorted(p.stem for p in (tmp_path / cache).glob("*.json")) if cache else None
        return [f.split("\n")[0] for f in fails], stats, entries, list(calls)

    fails, stats, entries, ran = replay("listed")
    assert ran == [("8", ["P.Wide"], []), ("2", ["P.Heavy"], []), (None, ["P.A", "P.B"], ["P.Heavy", "P.Wide"])]
    assert [(s["module"], s["threads"], s["stored"], s["failures"]) for s in stats["first"]] == \
        [("P.Wide", 8, True, []), ("P.Heavy", 2, True, [])]
    plain = replay("plain", [])
    assert plain[3] == [(None, ["P.A", "P.Heavy", "P.Wide", "P.B"], [])] and "first" not in plain[1]
    assert (fails, stats["checked"], stats["reused"], entries) == (plain[0], plain[1]["checked"], plain[1]["reused"], plain[2])
    assert fails == [] and (stats["checked"], stats["reused"]) == (4, 0) and entries == ["P.A", "P.B", "P.Heavy", "P.Wide"]
    fails, stats, entries, ran = replay("listed")
    assert ran == [(None, [], ["P.A", "P.Heavy", "P.Wide", "P.B"])] and stats["first"] == []
    assert fails == [] and (stats["checked"], stats["reused"]) == (0, 4)

    rejected.add("P.Heavy")
    fails, stats, entries, ran = replay("listed-bad")
    assert ran == [("8", ["P.Wide"], []), ("2", ["P.Heavy"], []), (None, ["P.A", "P.Heavy", "P.B"], ["P.Wide"])]
    assert [(s["module"], s["stored"]) for s in stats["first"]] == [("P.Wide", True), ("P.Heavy", False)]
    assert stats["first"][1]["failures"] == ["replay: the kernel rejected a declaration under P, in P.Heavy (exit 1):"]
    plain = replay("plain-bad", [])
    assert (fails, stats["checked"], stats["reused"], entries) == (plain[0], plain[1]["checked"], plain[1]["reused"], plain[2])
    assert fails == stats["first"][1]["failures"] and stats["checked"] == 4 and entries == ["P.A", "P.B", "P.Wide"]
    rejected.clear()

    for cache, kw in ((None, {}), ("unwritten", {"store": False})):
        fails, stats, entries, ran = replay(cache, **kw)
        assert ran == [(None, ["P.A", "P.Heavy", "P.Wide", "P.B"], [])] and stats["first"].startswith("not used: ")
        assert fails == [] and stats["checked"] == 4 and not entries


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_replay_in_listed_steps_checks_accepts_and_rejects_what_the_one_step_does(tmp_path, monkeypatch, sandboxed_node):
    """On real builds, each audited twice on cold caches, as one step and with modules listed (`replay_first`): the same
    modules checked, the same failures, the same axioms and the same keys stored. The module controls, with `Modular.Uses`
    listed, whose proof uses theorems of modules it imports privately: its own step loads them as an import does and the
    kernel accepts it. The control with a declaration that skipped the kernel, with `Control.Thms`, `Control.Bad` and a
    module it does not have listed: the first is stored by its step, the second is rejected in its step and again in the
    last, as in the one step. Only a real replay shows `Replay.lean` taking a set of one module on its imports, and the
    keys of a step meeting those of the whole replay."""
    def audit(pkg, cache, first):
        monkeypatch.setattr(A, "replay_first", lambda pkg: first)
        return A.audit(pkg, tmp_path / cache / "out", update=True, runs=False, log=lambda *_: None,
                       replay_cache=tmp_path / cache, store=True)

    def same(pkg, tag, first):
        one, listed = audit(pkg, f"{tag}-one", []), audit(pkg, f"{tag}-listed", first)
        assert counts(listed) == counts(one) and listed["axioms"] == one["axioms"]
        assert [f.split("\n")[0] for f in listed["failures"]] == [f.split("\n")[0] for f in one["failures"]]
        assert sorted(p.name for p in (tmp_path / f"{tag}-listed").glob("*.json")) == \
            sorted(p.name for p in (tmp_path / f"{tag}-one").glob("*.json"))
        assert "first" not in one["replay"]
        return one, listed

    modules = tmp_path / "modules"
    shutil.copytree(A.FIXTURES / "modules", modules, ignore=shutil.ignore_patterns(".lake"))
    subprocess.run(["lake", "build"], cwd=modules, check=True, capture_output=True, env=A.lake_env())
    one, listed = same(modules, "modules", [{"module": "Modular.Uses", "threads": 1, "why": "w"}])
    assert one["failures"] == [] and counts(one) == (5, 0), one["failures"]
    (step,) = listed["replay"]["first"]
    assert (step["module"], step["threads"], step["stored"], step["failures"]) == ("Modular.Uses", 1, True, [])

    control = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", control, ignore=shutil.ignore_patterns(".lake"))
    kernel = kernel_control(control)
    subprocess.run(["lake", "build"], cwd=control, check=True, capture_output=True, env=A.lake_env())
    one, listed = same(control, "control", [{"module": "Control.Thms", "threads": 2, "why": "w"},
                                             {"module": "Control.Bad", "threads": 1, "why": "w"},
                                             {"module": "Control.Gone", "threads": 1, "why": "w"}])
    assert counts(one) == (5, 0) and [f for f in one["failures"] if f.startswith(kernel["fails"])]
    thms, bad = listed["replay"]["first"]
    assert (thms["module"], thms["stored"], thms["failures"]) == ("Control.Thms", True, [])
    assert (bad["module"], bad["stored"]) == ("Control.Bad", False) and bad["failures"][0].startswith(kernel["fails"])


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_an_equation_lemma_two_modules_realize_counts_as_declared_by_the_imported_one(tmp_path):
    """`f.eq_1` is realized in B and in C, which don't import each other; D imports C and uses it. With B replayed and C
    imported (it passed), D's use of the lemma is declared by C, which D imports (as ZkQueryRound's `match_7.eq_2` and the
    `else_eq` that SimZk and ZkEq both realize). With B and C both imported (E, replayed, imports B), the imported
    environment names one module for the lemma, B, and D's import of C still declares it (as `Exists.choose.congr_simp`, realized in Init.Control.Lawful.Lemmas and in the UpwardEnumerable that
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


#: a package under the module system. `k.eq_1` is realized in `T`'s proof, which leaves it a theorem in `T`'s public
#: part, and in the bodies of `E`'s and `F`'s exposed definitions, which leave it an axiom in theirs (as
#: `Circuit.computed.eq_1` in `Definitions.Nci.Dark` and `Proofs.Core.Circuit`). `W1` and `W2` each declare `w`, on a
#: `sorry` and on an axiom hidden in their private parts. `P` and `Q` declare `t` with different statements, `G1` and
#: `G2` the definition `g : Nat` with different bodies, `O1` and `O2` the opaque `o : Nat`, `X` the axiom `u` and `Y`
#: the theorem `u`.
REALIZED = {
    "A": "@[expose] public def k (n : Nat) : Nat := n + 1\n",
    "T": "public import R.A\npublic theorem t : k 0 = 1 := by simp [k]\n",
    "E": "public import R.A\n@[expose] public def e : {n : Nat // k n = n + 1} := ⟨0, by simp [k]⟩\n",
    "F": "public import R.A\n@[expose] public def f : {n : Nat // k n = n + 1} := ⟨0, by simp [k]⟩\n",
    "D": "public import R.T\npublic import R.E\npublic import R.F\npublic theorem d : k 0 = 1 := by simp [k]\n",
    "D2": "public import R.E\npublic import R.F\npublic theorem d2 : k 0 = 1 := by simp [k]\n",
    "W1": "public import R.A\npublic theorem w : k 0 = 1 := by sorry\n",
    "W2": "public import R.A\naxiom hid : k 0 = 1\npublic theorem w : k 0 = 1 := hid\n",
    "V": "public import R.W1\npublic import R.W2\npublic theorem v : k 0 = 1 := w\n",
    "P": "public theorem t : 1 = 1 := rfl\n",
    "Q": "public theorem t : 2 = 2 := rfl\n",
    "G1": "public def g : Nat := 1\n",
    "G2": "public def g : Nat := 2\n",
    "O1": "public opaque o : Nat\n",
    "O2": "public opaque o : Nat\n",
    "X": "public axiom u : 1 = 1\n",
    "Y": "public theorem u : 1 = 1 := rfl\n",
    "H": "public theorem hk : 3 = 3 := rfl\n",
    "I": "public import R.A\nimport R.H\ntheorem i_priv : 3 = 3 := hk\npublic theorem i : k 0 = 1 := by simp [k]\n",
    "J": "public import R.I\npublic theorem j : k 0 = 1 := i\n",
    "K": "@[expose] public def kb : Nat := 3\n",
    "L": "public import R.K\n@[expose] public def la : Nat := kb\n",
    "M": "public import R.L\npublic theorem m : la = 3 := rfl\n",
}


@pytest.fixture(scope="module")
def realized(tmp_path_factory) -> Path:
    """`REALIZED` built; no module imports both of a pair that clashes."""
    pkg = tmp_path_factory.mktemp("realized") / "pkg"
    (pkg / "R").mkdir(parents=True)
    shutil.copy(HERE / "fixtures" / "control" / "lean-toolchain", pkg)
    (pkg / "lakefile.toml").write_text('name = "realized"\n\n[[lean_lib]]\nname = "R"\n')
    for m, text in REALIZED.items():
        (pkg / "R" / f"{m}.lean").write_text("module\n" + text)
    tops = ("D", "D2", "V", "P", "Q", "G1", "G2", "O1", "O2", "X", "Y", "J", "M")
    subprocess.run(["lake", "build", *(f"R.{m}" for m in tops)], cwd=pkg, check=True, capture_output=True,
                   env=A.lake_env())
    return pkg


def records_replay(built: Path, tmp_path: Path, *args: str, public: tuple[str, ...] = ()) -> tuple[int, str, dict]:
    """`Replay.lean --records` on `built`, copied once to ``tmp_path``, with the ``public`` modules cut to their public
    parts, as the graph's lean-module runner has its imports: its exit code, its output and what it wrote."""
    pkg = tmp_path / "pkg"
    if not pkg.exists():
        shutil.copytree(built, pkg, symlinks=True)
    for m in public:
        for part in ("olean.private", "olean.server"):
            (pkg / ".lake" / "build" / "lib" / "lean" / "R" / f"{m}.{part}").unlink(missing_ok=True)
    out = tmp_path / "replay.json"
    out.unlink(missing_ok=True)
    (tmp_path / "records").mkdir(exist_ok=True)
    r = subprocess.run(["lake", "env", "lean", "--run", str(A.TOOL / "Replay.lean"), str(out), "--records",
                        str(tmp_path / "records"), *args], cwd=pkg, capture_output=True, text=True, env=A.lake_env())
    return r.returncode, r.stderr + r.stdout, json.loads(out.read_text()) if out.is_file() else {}


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_axioms_reused_public_parts_make_of_one_proposition_are_one_constant(realized, tmp_path):
    """The lean-module runner reads its imports (`--closure-reused`, each passed its own task) at their public parts,
    where `E`'s and `F`'s `k.eq_1` are axioms and `T`'s a theorem: `D`'s and `D2`'s replays pass, as `lean` imported
    them (they refused `Circuit.computed.eq_1`, `Claims.OC.eval.eq_def` and `finalCheck.congr_simp` on the module
    system)."""
    rc, log, _ = records_replay(realized, tmp_path, "R.D2", "--closure-reused", "R.E", "R.F", public=("E", "F"))
    assert rc == 0, log
    rc, log, _ = records_replay(realized, tmp_path, "R.D", "--closure-reused", "R.T", "R.E", "R.F",
                                public=("T", "E", "F"))
    assert rc == 0, log


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_copy_from_a_module_checked_here_must_be_a_theorem_it_replays(realized, tmp_path):
    """`T`, checked here at its private part, holds `k.eq_1` as a theorem, which the kernel checks beside the reused
    axioms; `X`, checked here, holds `u` as an axiom it wrote, which no reused theorem `Y` stands for."""
    rc, log, got = records_replay(realized, tmp_path, "R.D", "--closure", "R.T", "--closure-reused", "R.E", "R.F",
                                  public=("E", "F"))
    assert rc == 0 and got["replayed"] == 3, log  # `T`'s `t` and `k.eq_1`, and `d`
    rc, log, _ = records_replay(realized, tmp_path, "R.X", "--closure-reused", "R.Y", public=("Y",))
    refusal = "u is declared by R.Y and by R.X as an axiom in R.X, which `--closure-reused` doesn't name"
    assert rc != 0 and refusal in log, log


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_axioms_of_a_type_that_is_not_a_proposition_are_refused(realized, tmp_path):
    """The axioms `g : Nat` two reused public parts hold hide definitions with different bodies, so they aren't one
    constant whatever `lean` proved from each."""
    rc, log, _ = records_replay(realized, tmp_path, "R.A", "--closure-reused", "R.G1", "R.G2", public=("G1", "G2"))
    assert rc != 0 and "g is declared by R.G1, R.G2, axioms each, of a type that isn't a proposition" in log, log


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_name_declared_with_two_statements_is_refused(realized, tmp_path):
    """`P`'s theorem `t : 1 = 1` and the axiom `Q`'s public part makes of its `t : 2 = 2` are two constants."""
    rc, log, _ = records_replay(realized, tmp_path, "R.P", "--closure-reused", "R.Q", public=("Q",))
    assert rc != 0 and "t is declared by R.Q and by R.P with different statements" in log, log


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_definitions_or_opaques_two_modules_declare_are_refused(realized, tmp_path):
    """Read at their private parts, reused or checked here, `G1`'s and `G2`'s `g` are definitions and `O1`'s and `O2`'s
    `o` opaques: neither a theorem nor the axiom a public part makes of one."""
    rc, log, _ = records_replay(realized, tmp_path, "R.A", "--closure-reused", "R.G1", "R.G2")
    assert rc != 0 and "g is declared by R.G1 and by R.G2 and not as theorems or axioms" in log, log
    rc, log, _ = records_replay(realized, tmp_path, "R.A", "--closure", "R.O1", "R.O2")
    assert rc != 0 and "o is declared by R.O1 and by R.O2 and not as theorems or axioms" in log, log


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_sorry_or_axiom_hidden_behind_either_copy_of_a_name_reaches_its_importer(realized, tmp_path):
    """`W1` and `W2` each pass their own replay, which records `w`'s axioms; read at their public parts, where each `w`
    is an axiom, they leave `V`'s `v` with the axioms of whichever copy stands for `w`, the `sorry` or the hidden axiom,
    never none."""
    hidden = {}
    for m in ("W1", "W2"):
        rc, log, got = records_replay(realized, tmp_path, f"R.{m}")
        assert rc == 0, log
        hidden[m] = got["declarations"]["w"]
        (tmp_path / "records" / "R").mkdir(parents=True, exist_ok=True)
        (tmp_path / "records" / "R" / f"{m}.json").write_text(json.dumps({"module": f"R.{m}",
                                                                          "axioms": got["declarations"]}))
    assert hidden["W1"] == ["sorryAx"] and hidden["W2"][0].endswith(".hid"), hidden
    for order in (["R.W1", "R.W2"], ["R.W2", "R.W1"]):
        rc, log, got = records_replay(realized, tmp_path, "R.V", "--closure-reused", *order, public=("W1", "W2"))
        assert rc == 0 and got["declarations"]["v"] in (hidden["W1"], hidden["W2"]), (order, log, got)


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_public_part_brings_in_none_of_its_modules_private_imports(realized, tmp_path):
    """`I` imports `H` privately, so `lean` compiles `J` (`public import R.I`) on `I`'s public part without `H`, and the
    lean-module runner reads no part of `H`: `J`'s replay passes without it, as `CallsDecision`'s didn't on `CallsTop`'s
    private import of `CallsFast100`. With `I`'s private part, as an `import all` of it has it, `H` is still needed."""
    pkg = tmp_path / "pkg"
    shutil.copytree(realized, pkg, symlinks=True)
    for f in (pkg / ".lake" / "build" / "lib" / "lean" / "R").glob("H.*"):
        f.unlink()
    rc, log, _ = records_replay(realized, tmp_path, "R.J", "--closure-reused", "R.I", "R.A", public=("A",))
    assert rc != 0 and "R.H does not exist" in log, log
    rc, log, got = records_replay(realized, tmp_path, "R.J", "--closure-reused", "R.I", "R.A", public=("I", "A"))
    assert rc == 0 and got["replayed"] == 1, log


#: `Forge FILE MODULE` rewrites the `.olean` FILE with its imports made private
FORGE = """import Lean
open Lean

unsafe def main : List String → IO UInt32
  | [file, mod] => do
    let (data, _) ← readModuleData file
    saveModuleData (file ++ ".forged") mod.toName { data with imports := data.imports.map ({ · with isExported := false }) }
    IO.FS.rename (file ++ ".forged") file
    return 0
  | _ => return 2
"""


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_constant_only_a_private_import_declares_is_unknown_to_the_kernel(realized, tmp_path):
    """The replay fails closed on a public part that mentions what a private import declares, which no `lean` build
    writes: `L`'s import of `K` made private, its `la` still unfolds to `K`'s `kb`, which `M`'s `m : la = 3 := rfl`
    needs. The replay leaves `K` out, and the kernel rejects `m`."""
    pkg = tmp_path / "pkg"
    shutil.copytree(realized, pkg, symlinks=True)
    lib = pkg / ".lake" / "build" / "lib" / "lean" / "R"
    for f in lib.glob("K.*"):
        f.unlink()
    (tmp_path / "Forge.lean").write_text(FORGE)
    subprocess.run(["lake", "env", "lean", "--run", str(tmp_path / "Forge.lean"), str(lib / "L.olean"), "R.L"], cwd=pkg,
                   check=True, capture_output=True, env=A.lake_env())
    rc, log, got = records_replay(realized, tmp_path, "R.M", "--closure-reused", "R.L", public=("L",))
    assert rc == 1 and got["rejected"] == ["R.M"] and "unknown constant 'kb'" in log, log


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_checked_declaration_waits_for_what_an_unchecked_one_it_unfolds_needs(tmp_path):
    """D's `a_t : f = 6 := rfl` makes the kernel unfold `f`, from M, which joins the replay unchecked (it passed, and it
    imports G, which hasn't), and `f`'s body mentions G's `zz_g`, checked in the same replay. Ordered only by what it
    mentions itself, `a_t` would go to the kernel before `zz_g` was in it: "unknown constant" (as
    `Flock.sizeOf_lt_of_mem_toArray` and `Std.DTreeMap.Raw.Const.toList` did)."""
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
def test_the_replay_reads_a_modules_private_part_and_a_plain_files_olean_alone(tmp_path):
    """A `module`'s proofs are only in its `.olean.private`: without it the replay refuses the module, where the `.olean`
    its importers read would replay each theorem as an axiom. A file outside the module system is its `.olean` alone, as
    Lean imports it: the parts of its build as a `module`, put back beside it (Lake deletes them when it rebuilds the
    file; a copied or restored build tree need not), are that build's, and the replay reads none of them."""
    pkg = tmp_path / "pkg"
    shutil.copytree(A.FIXTURES / "modules", pkg)
    lib = pkg / ".lake" / "build" / "lib" / "lean"

    def replay() -> subprocess.CompletedProcess:
        return subprocess.run(["lake", "env", "lean", "--run", str(A.TOOL / "Replay.lean"), str(tmp_path / "replay.json"),
                               *sorted(A.lean_modules(pkg))], cwd=pkg, capture_output=True, text=True, env=A.lake_env())

    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    r = replay()
    assert r.returncode == 0, r.stderr + r.stdout
    axioms = json.loads((tmp_path / "replay.json").read_text())["axioms"]
    parts = {p: p.read_bytes() for p in lib.rglob("*.olean.*") if p.suffix in (".server", ".private")}
    (lib / "Modular" / "Hidden.olean.private").unlink()
    r = replay()
    assert r.returncode != 0 and "Modular.Hidden is a `module` and " in r.stderr and "Hidden.olean.private is missing" \
        in r.stderr, r.stderr + r.stdout
    for rel, text in json.loads((A.FIXTURES / "module-violations.json").read_text())["plain"]["write"].items():
        (pkg / rel).write_text(text)
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=A.lake_env())
    for p, b in parts.items():
        p.write_bytes(b)
    r = replay()
    assert r.returncode == 0, r.stderr + r.stdout
    assert json.loads((tmp_path / "replay.json").read_text())["axioms"] == axioms


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_controls(tmp_path):
    toolchain = (A.FIXTURES / "control" / "lean-toolchain").read_text().strip()
    assert A.controls(toolchain, tmp_path, log=lambda *_: None) == []


def test_a_step_records_the_peak_memory_of_every_process_it_waited_for(tmp_path):
    """`run` reaps with `os.wait4`, so a step's `peak_gib` covers the grandchildren a `lake` runs, not only `lake` itself;
    the output and exit code are what `subprocess.run` gave.  It runs in an interpreter of its own: a child's `ru_maxrss`
    counts the resident set it was forked with, so in a test worker grown past 0.3 GiB even `true` would peak there."""
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
    """A statement such as `OnePair`, binding `w k` before its two acceptances, or `LiveCoinBinding`, taking
    `c.root = c'.root`, is equivalent to a form that fits. Binders reorder with the acceptances, the premises and what they mention first, so a
    hypothesis on a binder the acceptance doesn't read is the conclusion's, when the conclusion reads that binder too; an
    equation of one field of two inputs the acceptance reads is the field bound once. Nothing else moves: a hypothesis on
    what the acceptance reads, one whose proof a kept binder mentions, a structure's fields, a probability statement's
    hypotheses, and any statement whose facts don't say what each binder mentions are judged as written."""
    entry = {"function": "P.check", "outcome": "accept"}
    def acc(name, at, uses):
        return hyp(name, at, ["P.check"]) | {"uses": uses}
    def q(name, at, uses, **kw):
        return hyp(name, at, ["P.Q"]) | {"uses": uses} | kw
    def inputs(*ns):
        return [{"name": n, "at": a, "head": None, "uses": []} for n, a in ns]
    def names(got):
        return [w.split(" : ")[0] for w in A.refusals(entry, [got], [])]
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
    def c(applies, guarded=False):
        return {"applies": applies, "guarded": guarded}
    def event(*cs, head="G.prob"):
        return {"head": head, "conjuncts": list(cs)}
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
    def pol(frozen):
        return {"guarantees": {"P.a": {}, "P.b": {}}, "shape": {"premises": [], "properties": entries, "frozen": frozen}}
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
    out = A.shape(pol | {"shape": {"frozen": {"P.b": "unstated: x", "P.A": "frozen: y"}}}, facts)
    assert sorted(f.split()[1] for f in out) == ["shape.frozen[P.A]", "shape.frozen[P.b]"], out
    assert A.category("falsified (conjecture): rests on P.C") == "falsified" and A.category("lemmas") is None


def test_a_lemma_or_a_completeness_theorem_is_refused_with_its_fix():
    """The lock lists what is assumed and what is proved: a `shape.frozen` reason starting `lemma` or `completeness` is
    no category, and its failure (`UNLISTED`) says to take the theorem off `guarantees`, or to name a completeness theorem
    under its guarantee's `witnesses` entry when it is that guarantee's honest witness."""
    facts = {"pins": {g: {"shape": [part(inputs=[("x", 0)])]} for g in ("P.b", "P.c", "P.d")}}
    pol = {"guarantees": {"P.b": {}, "P.c": {}, "P.d": {}},
           "shape": {"frozen": {"P.b": "lemma: a step of P.a's proof", "P.c": "completeness (honest): P.a's device",
                                "P.d": "frozen: not swept yet"}}}
    out = A.shape(pol, facts)
    assert [(f.code, f.entry) for f in out] == [("shape.frozen-unlisted", "P.b"), ("shape.frozen-unlisted", "P.c")], out
    assert "starts with `lemma`: a lemma is not listed: remove it from `guarantees`" in out[0]
    assert "starts with `completeness`: a completeness theorem is not listed: name it under its guarantee's " \
           "`witnesses` entry" in out[1]
    assert A.category("lemma: x") is None and A.category("completeness: x") is None
    assert not set(A.UNLISTED) & set(A.CATEGORIES)


def test_a_property_on_a_conjecture_lists_it_and_one_on_a_falsified_conjecture_comes_off():
    """`shape.conjectures`: a hypothesis whose head is a named conjecture is a premise, and a property resting on one
    (a hypothesis reaching it, even inside an existential) lists it under `rests_on`, so it is listed as conditional; a
    conjecture marked falsified refuses every property resting on it, which stays frozen only as `falsified`; an open
    conjecture names its falsification test, and every entry is a named assumption with a reason."""
    entry = {"function": "P.check", "outcome": "accept"}
    rests = part([ACCEPTS, hyp("c", 2, ["P.Conj"]) | {"head": "P.Conj"}], inputs=[("x", 0)])
    def pol(status, frozen=None, rests_on=("P.Conj",)):
        return {
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
    assert ["neither a premise the lock lists" in out[0], "gives no reason" in out[1],
            "names no falsification test" in out[2]] == [True] * 3, out
    refuted = {"pins": facts["pins"] | {"P.refuted": {"reads": [{"name": "P.Conj", "module": "P.Assumptions"}]}}}
    assert [x.code for x in A.shape(bad, refuted)][:1] == ["shape.conjecture-reason"], \
        "a conjecture a guarantee only reads (one that refutes it) is named by its `reads` record"
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
    """Restoring only security/Proofs' own build with `--dev-cache` would rebuild most of its modules: Security and Flock,
    which it requires by path, would be built again, not byte-identically, and everything above them would compile. With `--dev-cache`, setup restores each path dependency's build too, lending one with no fetched dependencies of its own the root's for the restore; without it, or
    after a failed setup.sh, it restores nothing more."""
    calls, seen = [], []
    deps = [tmp_path / "security", tmp_path / "flock"]
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
    assert seen == [{"security": True, "flock": False}] and not (deps[0] / ".lake" / "packages").exists()
    calls.clear()
    assert A.dev_restore(tmp_path / "Proofs", 2, "x", tmp_path / "setup.log") == (2, "x") and calls == []
    monkeypatch.setattr(A, "DEV_CACHE", False)
    assert A.dev_restore(tmp_path / "Proofs", 0, "", tmp_path / "setup.log") == (0, "") and calls == []
