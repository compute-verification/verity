"""`tools/verity/lean/merge.py`: two branches' `lean-audit.json` merged as `check.py --update` would write the merged tree's record,
and, where a record can't say what that is, merged by rule with each entry taken from one side named; as a function, and as
git's merge driver, which leaves a conflict only where neither side is a record or a guard refuses (a lock format newer
than the driver's, the worktree's copy merging in newer merge rules, a merge that loses most of a section)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_merge", HERE.parent / "merge.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
A = M.A

STD = ["Classical.choice", "Quot.sound", "propext"]
FILES = {"P": "P.lean", "P.Defs": "P/Defs.lean", "P.More": "P/More.lean", "P.Thms": "P/Thms.lean"}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def read(name: str, module: str, h: str) -> dict:
    return {"name": name, "module": module, "kind": "def", "hash": h, "sha256": sha(h)}


F, G, H, K = read("P.f", "P.Defs", "1" * 16), read("P.g", "P.Defs", "2" * 16), read("P.h", "P.Defs", "3" * 16), \
    read("P.k", "P.More", "4" * 16)


def thm(reads: list[dict], statement: str = "P.f n = n", h: str = "00000000000000aa") -> dict:
    return {"kind": "theorem", "module": "P.Thms", "signature": f"(n : Nat) : {statement}", "axioms": STD, "type_hash": h,
            "type_sha256": sha(h), "hypotheses": [], "reads": reads}


def built(**guarantees: dict) -> dict:
    """The facts of a build whose guarantees are ``guarantees`` (name suffix: its facts)."""
    return {"lean": "4.34.0", "modules": list(FILES), "imports": {}, "declarations": {}, "axioms": STD, "offenders": [],
            "declared_axioms": [], "escapes": [], "pins": {f"P.{n}": f for n, f in guarantees.items()}}


def record(deps: dict | None = None, **guarantees: dict) -> dict:
    """The record `check.py --update` writes for a package whose guarantees are ``guarantees`` (name suffix: its facts)."""
    fails, pol = A.judge({"roots": ["P"], "meaning": ["P"], "guarantees": {f"P.{n}": {} for n in guarantees}}, built(**guarantees),
                         FILES, update=True, owner="@lean")
    assert fails == []
    return pol | {"dependencies": deps or {"mathlib": "a" * 64, "toolchain": "c" * 64}}


def text_conflicts(base: dict, ours: dict, theirs: dict, tmp: Path) -> bool:
    for n, d in (("base", base), ("ours", ours), ("theirs", theirs)):
        (tmp / n).write_text(A.text(d))
    return subprocess.run(["git", "merge-file", "-p", *(str(tmp / n) for n in ("ours", "base", "theirs"))],
                          capture_output=True).returncode != 0


def test_two_branches_that_each_add_guarantees_merge_to_the_record_of_the_merged_tree(tmp_path):
    base = record(t=thm([F]))
    ours = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"), deps={"mathlib": "b" * 64, "toolchain": "c" * 64})
    theirs = record(t=thm([F]), v=thm([H, K], "P.h n = P.k n", "00000000000000cc"))
    merged, why = M.merge(base, ours, theirs)
    assert why == []
    tree = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"), v=thm([H, K], "P.h n = P.k n", "00000000000000cc"),
                  deps={"mathlib": "b" * 64, "toolchain": "c" * 64})
    assert A.text(merged) == A.text(tree)
    assert list(merged["reads"]["P.Defs"]["definitions"]) == ["P.f", "P.g", "P.h"]
    assert all(set(g) == {"digest", "definitions"} for g in merged["reads"].values())
    assert text_conflicts(base, ours, theirs, tmp_path)
    assert M.merge(base, ours, ours) == (ours, [])


def test_a_hand_written_section_one_side_extends_is_kept():
    base = record(t=thm([F]))
    ours = base | {"meaning": ["P", "P.More"]}
    theirs = record(t=thm([F]), v=thm([H], "P.h n = n", "00000000000000cc"))
    merged, why = M.merge(base, ours, theirs)
    assert why == [] and merged["meaning"] == ["P", "P.More"] and "P.v" in merged["guarantees"]
    both, why = M.merge(base, ours, base | {"meaning": ["P", "P.Other"]})
    assert why == [] and set(both["meaning"]) == {"P", "P.More", "P.Other"}, "two sides that each extend it merge"


def test_a_derived_record_both_sides_changed_merges_as_theirs_and_the_merged_build_regenerates_it():
    """A guarantee restated on both sides, a definition's hash and a new guarantee's record each written two ways: what the
    build determines, so theirs' value is the candidate, and the merged tree's build (a third statement) regenerates it."""
    base = record(t=thm([F]))
    u, v = thm([F, G], "P.g n = n", "00000000000000bb"), thm([H], "P.h n = n", "00000000000000cc")
    ours = record(t=thm([F], "P.f n ≤ n", "00000000000000dd"), u=u)
    theirs = record(t=thm([F], "P.f n < n + 1", "00000000000000ee"), v=v)
    merged, why = M.merge(base, ours, theirs, candidate := [])
    assert why == [] and candidate == ["guarantees[P.t] (theirs)"]
    assert merged["guarantees"]["P.t"] == theirs["guarantees"]["P.t"]
    tree = built(t=thm([F], "P.f n ≤ n + 1", "00000000000000ab"), u=u, v=v)
    fails = A.judge(merged, tree, FILES)[0]
    assert {f.code for f in fails} == {"guarantees.signature"} and all(map(A.RECORD_DIFFERENCE.match, fails)), fails
    regen = A.regenerated(merged, tree, FILES, fails)
    assert A.judge(regen, tree, FILES)[0] == [] and regen["guarantees"]["P.t"]["signature"] == ["(n : Nat) : P.f n ≤ n + 1"]

    f9, f8 = read("P.f", "P.Defs", "9" * 16), read("P.f", "P.Defs", "8" * 16)
    ours, theirs = record(t=thm([f9]), u=u), record(t=thm([f8]), v=v)
    merged, why = M.merge(base, ours, theirs, candidate := [])
    assert why == [] and candidate == ["reads[P.Defs].definitions[P.f] (theirs)"]
    assert merged["reads"]["P.Defs"]["definitions"]["P.f"] == theirs["reads"]["P.Defs"]["definitions"]["P.f"]
    tree = built(t=thm([read("P.f", "P.Defs", "7" * 16)]), u=u, v=v)
    fails = A.judge(merged, tree, FILES)[0]
    assert fails and all(map(A.RECORD_DIFFERENCE.match, fails)), fails
    assert A.judge(A.regenerated(merged, tree, FILES, fails), tree, FILES)[0] == []

    w = thm([F], "P.f n = 0 + n", "00000000000000ff")
    merged, why = M.merge(base, record(t=thm([F]), u=u), record(t=thm([F]), u=w), candidate := [])
    assert why == [] and candidate == ["guarantees[P.u] (theirs)"] and merged["guarantees"]["P.u"]["type_hash"] == sha(
        "00000000000000ff")


def test_what_only_a_person_writes_is_theirs_where_both_sides_changed_it_beside_derived_candidates():
    """A dependency bumped two ways, an owner set two ways and a `frozen` entry written two ways merge as theirs', each
    named, whatever derived record merges as a candidate beside them; a field only one side changed is that side's."""
    base = record(t=thm([F]))
    def restated(s, h):
        return record(t=thm([F], s, h))
    ours, theirs = restated("P.f n ≤ n", "00000000000000dd"), restated("P.f n < n + 1", "00000000000000ee")
    def deps(d):
        return {"dependencies": {"mathlib": d * 64, "toolchain": "c" * 64}}
    merged, why = M.merge(base, ours | deps("b"), theirs | deps("d"))
    assert why == ["dependencies[mathlib]: changed on both sides, differently: took theirs"]
    assert merged["dependencies"]["mathlib"] == "d" * 64
    def owner(rec, who):
        return rec | {"guarantees": {"P.t": rec["guarantees"]["P.t"] | {"owner": who}}}
    merged, why = M.merge(base, owner(ours, "@proofs"), owner(theirs, "@circuits"))
    assert why == ["guarantees[P.t].owner: changed on both sides, differently: took theirs"]
    assert merged["guarantees"]["P.t"]["owner"] == "@circuits"
    def frozen(rec, why):
        return rec | {"shape": {"frozen": {"P.t": why}}}
    merged, why = M.merge(base, frozen(ours, "lemma"), frozen(theirs, "restatable"))
    assert why == ["shape[frozen][P.t]: added on both sides, differently: took theirs"]
    assert merged["shape"]["frozen"] == {"P.t": "restatable"}
    def hyp(rec, who):
        return rec | {"assumptions": {"P.Hyp": {"owner": who, "digest": "a" * 64}}}
    merged, why = M.merge(hyp(base, "@lean"), hyp(ours, "@proofs"), hyp(theirs, "@circuits"))
    assert why == ["assumptions[P.Hyp].owner: changed on both sides, differently: took theirs"]
    assert merged["assumptions"]["P.Hyp"]["owner"] == "@circuits"
    def opened(rec, issue, due="2026-11-30"):
        return rec | {"open": {"P.Sound": {"owner": "@lean", "digest": "a" * 64, "issue": issue, "due": due}}}
    merged, why = M.merge(opened(base, "LEAN-1"), opened(ours, "LEAN-2"), opened(theirs, "LEAN-3"))
    assert why == ["open[P.Sound].issue: changed on both sides, differently: took theirs"]
    assert merged["open"]["P.Sound"]["issue"] == "LEAN-3"
    merged, why = M.merge(opened(base, "LEAN-1"), opened(ours, "LEAN-2"), opened(theirs, "LEAN-1", "2026-12-31"))
    assert why == [] and merged["open"]["P.Sound"] == {"owner": "@lean", "digest": "a" * 64, "issue": "LEAN-2",
                                                       "due": "2026-12-31"}, "one side's issue, the other's due date"
    def scoped(rec, why, cites=None):
        return rec | {"domain": {"P.Scope": {"owner": "@lean", "digest": "a" * 64, "why": why}
                                 | ({"cites": cites} if cites else {})}}
    merged, why = M.merge(scoped(base, "x"), scoped(ours, "y"), scoped(theirs, "z"))
    assert why == ["domain[P.Scope].why: changed on both sides, differently: took theirs"]
    assert merged["domain"]["P.Scope"]["why"] == "z"
    merged, why = M.merge(scoped(base, "x"), scoped(ours, "y"), scoped(theirs, "x", "P.u"))
    assert why == [] and merged["domain"]["P.Scope"] == {"owner": "@lean", "digest": "a" * 64, "why": "y", "cites": "P.u"}


def hand_written(n: str | None = None, reason: str | None = None) -> dict:
    """A record's hand-written sections with one entry each, plus side ``n``'s own entries; with ``reason``, the base's
    entries hold that reason instead, and with "" they are removed."""
    def entry(**f):
        return {"owner": "@lean", "digest": "a" * 64} | f
    rec = {"roots": ["P"], "exempt": {"P.E": "why"}, "reads_exempt": {"P.R": "why"}, "four_names_except": {"P.W": "why"},
           "layers": {"D": {"imports": ["D"], "except": [], "why": "why"}}, "unwitnessed": {"Claims.U": "why"},
           "open": {"P.O": entry(issue="LEAN-1", due="2026-11-30")}, "domain": {"P.D": entry(why="why")},
           "assumptions": {"P.H": entry()}, "shape": {"frozen": {"P.F": "why"}, "inline": {"P.t": {"a": "why"}}}}
    tables = ([rec[k] for k in ("exempt", "reads_exempt", "four_names_except", "unwitnessed")]
              + [rec["shape"]["frozen"], rec["shape"]["inline"]["P.t"]])
    if n is not None:
        rec["roots"].append(f"P.{n}")
        rec["layers"]["D"]["except"].append(f"P.X{n}")
        rec["open"][f"P.O{n}"] = entry(issue=f"LEAN-{n}", due="2026-12-31")
        rec["domain"][f"P.D{n}"] = entry(why=f"why {n}")
        rec["assumptions"][f"P.H{n}"] = entry(owner=f"@{n}")
        for t in tables:
            t[f"new.{n}"] = f"why {n}"
    if reason is not None:
        for t, f in ((rec["open"]["P.O"], "due"), (rec["domain"]["P.D"], "why"), (rec["layers"]["D"], "why")):
            t[f] = reason
        for t in tables:
            t[next(iter(t))] = reason
        if not reason:
            for t in (rec["open"], rec["domain"], rec["layers"], *tables):
                del t[next(iter(t))]
    return rec


def test_hand_written_entries_two_sides_add_merge_as_their_union():
    """Roots, the exemptions, a layer's `except`, open premises with their issue and due date, domain hypotheses with their
    why, assumptions' owners and the reasons in `unwitnessed`, `frozen` and `shape.inline`: two sides that each add entries
    of their own keep both sides', naming none, and so do two that add the same entry the same way."""
    one, two = hand_written("1"), hand_written("2")
    merged, why = M.merge(hand_written(), one, two, candidate := [])
    assert why == [] and candidate == []
    for k in ("exempt", "reads_exempt", "four_names_except", "unwitnessed", "open", "domain", "assumptions"):
        assert merged[k] == one[k] | two[k], k
    assert sorted(merged["roots"]) == ["P", "P.1", "P.2"] and sorted(merged["layers"]["D"]["except"]) == ["P.X1", "P.X2"]
    assert merged["shape"] == {"frozen": one["shape"]["frozen"] | two["shape"]["frozen"],
                               "inline": {"P.t": one["shape"]["inline"]["P.t"] | two["shape"]["inline"]["P.t"]}}
    assert M.merge(hand_written(), one, one) == (one, [])


def test_a_hand_written_entry_one_side_removed_and_the_other_changed_is_the_changed_one():
    """A changed side beats a removed one, whichever side removed it: an open premise's new due date, a domain
    hypothesis's new why, a frozen or inline reason, a layer's rule and an exemption's or a claim's reason each stay with
    the changed side's value, named (in ``why``, or as a candidate where the lock run judges the entry)."""
    removed, changed = hand_written(reason=""), hand_written(reason="changed")
    for ours, theirs, side in ((removed, changed, "theirs"), (changed, removed, "ours")):
        merged, why = M.merge(hand_written(), ours, theirs, candidate := [])
        assert merged == changed, side
        assert why == [f"{p}: removed on one side and changed on the other: kept {side}' change" for p in (
            "exempt[P.E]", "reads_exempt[P.R]", "four_names_except[P.W]", "layers[D]", "shape[frozen][P.F]",
            "shape[inline][P.t][a]")]
        assert candidate == [f"unwitnessed[Claims.U] ({side})", "open[P.O]", "domain[P.D]"]


def test_an_owner_merges_apart_from_the_statement():
    base = record(t=thm([F]))
    unowned = {**base, "guarantees": {"P.t": A.unowned(base["guarantees"]["P.t"])}}
    restated = record(t=thm([F], "P.f n ≤ n", "00000000000000dd"))
    merged, why = M.merge(unowned, base, {**restated, "guarantees": {"P.t": A.unowned(restated["guarantees"]["P.t"])}})
    assert why == [] and merged["guarantees"] == restated["guarantees"], "one side's owner, the other's restatement"
    other = {**base, "guarantees": {"P.t": base["guarantees"]["P.t"] | {"owner": "@proofs"}}}
    merged, why = M.merge(unowned, base, other)
    assert why == ["guarantees[P.t].owner: added on both sides, differently: took theirs"]
    assert merged["guarantees"]["P.t"]["owner"] == "@proofs"


def test_a_definition_one_side_stops_reading_leaves_the_merge_unless_the_other_side_changed_its_hash():
    """A reads-only clash (one side moves definitions out of a module that the other side's new guarantees read).  No
    guarantee of the side that stopped reading the definition reads it, so in the merged tree only the other side's new
    ones may: the removal applies, named as a candidate, unless that side also changed the definition's hash, where the
    union stays with its hash and the Lean check of `next` regenerates."""
    base = record(t=thm([F, G]))
    ours = record(t=thm([F], "P.f n + 0 = n", "00000000000000dd"))
    theirs = record(t=thm([F, G]), v=thm([H], "P.h n = n", "00000000000000cc"))
    merged, why = M.merge(base, ours, theirs, candidate := [])
    assert why == [] and candidate == ["reads[P.Defs].definitions[P.g] (removed)"]
    assert A.text(merged) == A.text(record(t=thm([F], "P.f n + 0 = n", "00000000000000dd"),
                                           v=thm([H], "P.h n = n", "00000000000000cc")))
    other = record(t=thm([F, read("P.g", "P.Defs", "7" * 16)]), v=thm([H], "P.h n = n", "00000000000000cc"))
    merged, why = M.merge(base, ours, other, candidate := [])
    assert why == [] and candidate == ["reads[P.Defs].definitions[P.g]"]
    group = merged["reads"]["P.Defs"]
    assert group["definitions"] == other["reads"]["P.Defs"]["definitions"], "with the other side's hash when it changed it"
    assert list(group["definitions"]) == ["P.f", "P.g", "P.h"] and group["digest"] == A.module_digest(group["definitions"])
    assert M.merge(base, ours, record(t=thm([F, G]), v=thm([K], "P.k n = n", "00000000000000cc")))[1] == []


def test_a_module_one_side_stops_reading_keeps_only_what_the_other_side_s_new_guarantees_may_read():
    """A module one side reads nowhere keeps, from the other side's group, what that side added and what it may read
    through its new or restated guarantees; a group left without definitions is dropped."""
    base = record(t=thm([F, K]))
    ours = record(t=thm([F], "P.f n + 0 = n", "00000000000000dd"))
    m = read("P.m", "P.More", "6" * 16)
    theirs = record(t=thm([F, K]), w=thm([m], "P.m n = n", "00000000000000cc"))
    merged, why = M.merge(base, ours, theirs, candidate := [])
    assert why == [] and candidate == ["reads[P.More].definitions[P.k] (removed)", "reads[P.More]"]
    assert A.text(merged) == A.text(record(t=thm([F], "P.f n + 0 = n", "00000000000000dd"), w=thm([m], "P.m n = n",
                                                                                                  "00000000000000cc")))
    theirs = record(t=thm([F, read("P.k", "P.More", "5" * 16)]))
    merged, why = M.merge(base, ours, theirs, candidate := [])
    assert why == [] and candidate == ["reads[P.More]"] and "P.More" not in merged["reads"]
    assert A.text(merged) == A.text(ours)
    assert A.judge(merged, built(t=thm([F], "P.f n + 0 = n", "00000000000000dd")), FILES)[0] == []


def test_a_definition_one_side_moved_to_another_module_leaves_the_module_the_other_side_changed():
    """A side that moves definitions to a new module (`Definitions.Flock.Circuit`'s to `Definitions.Core.Boolean` in queue
    merge 0aaf13f4f6) while the other changes the old module's readers: the union kept them under both modules, which no
    build writes."""
    base = record(t=thm([F, G]), s=thm([G], "P.g n = n", "00000000000000bb"))
    ours = record(t=thm([F, G]))
    moved_f = read("P.f", "P.More", "1" * 16)
    theirs = record(t=thm([moved_f, G]), s=thm([G], "P.g n = n", "00000000000000bb"))
    merged, why = M.merge(base, ours, theirs, candidate := [])
    assert why == [] and candidate == []
    assert A.text(merged) == A.text(record(t=thm([moved_f, G])))
    assert list(merged["reads"]["P.Defs"]["definitions"]) == ["P.g"] and list(merged["reads"]["P.More"]["definitions"]) == ["P.f"]
    assert merged["reads"]["P.Defs"]["digest"] == theirs["reads"]["P.Defs"]["digest"]

    u = thm([G], "P.g n = 0 + n", "00000000000000cc")
    merged, why = M.merge(base, record(t=thm([F, G]), s=thm([G], "P.g n = n", "00000000000000bb"), u=u), theirs, candidate := [])
    assert why == [] and candidate == [], "ours' new reader leaves P.Defs as the base had it, so theirs' groups stand"
    assert A.text(merged) == A.text(record(t=thm([moved_f, G]), s=thm([G], "P.g n = n", "00000000000000bb"), u=u))


def test_an_assumption_s_digest_changed_two_ways_or_removed_against_changed_merges_and_regenerates():
    """The lock records a digest for every premise, so two PRs that each change a premise's definition clash on it:
    theirs' digest is the candidate, and an entry one side dropped while the other changed it stays until the merged build
    says no guarantee reads it (`assumptions.unread`)."""
    base = record(t=thm([F, G]))
    drop_g = record(t=thm([F]), u=thm([F], "P.f n = 0 + n", "00000000000000bb"))
    keep_g = record(t=thm([F, G]), v=thm([H], "P.h n = n", "00000000000000cc"))
    def hyp(rec, d):
        return rec | {"assumptions": {"P.Hyp": {"owner": "@lean", "digest": d * 64}}}
    merged, why = M.merge(hyp(base, "a"), hyp(drop_g, "b"), hyp(keep_g, "c"), candidate := [])
    assert why == [] and "assumptions[P.Hyp].digest (theirs)" in candidate
    assert merged["assumptions"] == {"P.Hyp": {"owner": "@lean", "digest": "c" * 64}}
    merged, why = M.merge(hyp(base, "a"), drop_g, hyp(keep_g, "c"), candidate := [])
    assert why == [] and "assumptions[P.Hyp]" in candidate and merged["assumptions"]["P.Hyp"]["digest"] == "c" * 64
    tree = built(t=thm([F]), u=thm([F], "P.f n = 0 + n", "00000000000000bb"), v=thm([H], "P.h n = n", "00000000000000cc"))
    fails = A.judge(merged, tree, FILES)[0]
    assert "assumptions.unread" in {f.code for f in fails} and all(map(A.RECORD_DIFFERENCE.match, fails)), fails
    regen = A.regenerated(merged, tree, FILES, fails)
    assert A.judge(regen, tree, FILES)[0] == [] and "P.Hyp" not in regen.get("assumptions", {})


def test_a_guarantee_one_side_removed_stays_removed_unless_the_other_side_changed_its_owner_which_keeps_it():
    """A restatement is the build's, so the removal applies over it; an owner is a person's, so the side that changed it
    keeps the guarantee, named (a changed side beats a removed one)."""
    base = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"))
    ours = record(t=thm([F]))
    theirs = record(t=thm([F]), u=thm([F, G], "P.g n = n + 0", "00000000000000dd"))
    merged, why = M.merge(base, ours, theirs, candidate := [])
    assert why == [] and candidate == ["guarantees[P.u] (removed)"] and merged["guarantees"] == ours["guarantees"]
    owner = theirs | {"guarantees": theirs["guarantees"] | {"P.u": theirs["guarantees"]["P.u"] | {"owner": "@proofs"}}}
    merged, why = M.merge(base, ours, owner)
    assert why == ["guarantees[P.u]: removed on one side and changed on the other: kept theirs' change"]
    assert merged["guarantees"]["P.u"] == owner["guarantees"]["P.u"]
    merged, why = M.merge(base, owner, ours)
    assert why == ["guarantees[P.u]: removed on one side and changed on the other: kept ours' change"]
    assert merged["guarantees"]["P.u"] == owner["guarantees"]["P.u"]


def claims(**sections: dict) -> dict:
    return {"claims": {"modules": ["Definitions.Claims"], "witnesses": {}, "not": {}, "generic": {}} | sections}


def test_proofs_zkoutleak_main_moved_to_claims_generic_leaves_claims_not_where_the_branch_still_has_it():
    """Merging `main` into a branch: `main` moved `Claims.ZKOutLeak` from `claims.not` to `claims.generic`, and the
    branch still had it under `claims.not`, with the base's reason or an older one. The merged record lists it under
    `claims.generic` alone; the driver had kept both, which the Lean check refuses (`claims.generic-both`)."""
    leak, excuse = "Claims.ZKOutLeak", {"Claims.Bits": "a helper"}
    base = claims(**{"not": excuse | {leak: "a corollary of `ZKOut`, proved of every `ZKOut`"}})
    main = claims(**{"not": excuse, "generic": {leak: "Claims.zkOutLeak"}})
    for reason in ("a corollary of `ZKOut`, proved of every `ZKOut`", "proved of every protocol zero knowledge"):
        branch = claims(**{"not": excuse | {leak: reason}})
        merged, why = M.merge(base, branch, main)
        assert why == [] and merged["claims"]["not"] == excuse and merged["claims"]["generic"] == {leak: "Claims.zkOutLeak"}


def test_a_premise_one_side_moved_to_another_section_leaves_the_one_the_other_side_changed_it_in():
    """Ours made `P.Hyp` a domain hypothesis while theirs gave the named assumption another owner: it is a domain
    hypothesis alone, as ours put it, and the other sections' entries merge as before."""
    hyp, other = {"owner": "@lean", "digest": "a" * 64}, {"P.Other": {"owner": "@lean", "digest": "b" * 64}}
    domain = {"owner": "@lean", "digest": "a" * 64, "why": "a deployment's choice"}
    base = {"assumptions": {"P.Hyp": hyp} | other}
    ours = {"assumptions": other, "domain": {"P.Hyp": domain}}
    theirs = {"assumptions": {"P.Hyp": hyp | {"owner": "@proofs"}} | other}
    merged, why = M.merge(base, ours, theirs)
    assert why == [] and merged["assumptions"] == other and merged["domain"] == {"P.Hyp": domain}


def test_an_entry_both_sides_placed_in_different_sections_is_where_theirs_put_it_naming_both():
    """A claim ours witnessed and theirs made generic, and a premise each side brought into a different section: neither
    side's placement is the base's, so the entry sits where theirs put it, with theirs' value, in that section only, and
    `check.py --update` on the merged tree classifies it."""
    base = claims(**{"not": {"Claims.C": "no claim yet"}})
    ours = claims(witnesses={"Claims.C": {"fails": "Claims.C_fails", "meets": ["Claims.C_meets"]}}) \
        | {"open": {"P.Hyp": {"owner": "@lean", "digest": "a" * 64, "issue": "LEAN-1", "due": "2026-11-06"}}}
    theirs = claims(generic={"Claims.C": "Claims.c"}) \
        | {"domain": {"P.Hyp": {"owner": "@lean", "digest": "a" * 64, "why": "a deployment's choice"}}}
    merged, why = M.merge(base, ours, theirs)
    assert why == [
        "Claims.C: ours put it under witnesses and theirs under generic, from not in the base (claims.witnesses[Claims.C] and "
        "claims.generic[Claims.C]): took theirs, which `check.py --update` classifies",
        "P.Hyp: ours put it under open and theirs under domain, from neither in the base (open[P.Hyp] and domain[P.Hyp]): "
        "took theirs, which `check.py --update` classifies"]
    assert merged["claims"]["generic"] == {"Claims.C": "Claims.c"}
    assert "Claims.C" not in merged["claims"]["witnesses"] and "Claims.C" not in merged["claims"]["not"]
    assert merged["domain"] == theirs["domain"] and "open" not in merged


def test_an_entry_both_sides_moved_out_of_the_base_s_section_is_where_theirs_moved_it_with_theirs_fields():
    """An open premise ours made a domain hypothesis while theirs, which also changed its owner, made it a named
    assumption: theirs' section and fields, the base's section emptied, and one warning line naming the entry."""
    def entry(**fields):
        return {"owner": "@lean", "digest": "a" * 64} | fields
    base = record(t=thm([F])) | {"open": {"P.Hyp": entry(issue="LEAN-1", due="2026-11-06")}}
    ours = record(t=thm([F])) | {"domain": {"P.Hyp": entry(why="a deployment's choice")}}
    theirs = record(t=thm([F])) | {"assumptions": {"P.Hyp": entry(owner="@proofs")}}
    merged, why = M.merge(base, ours, theirs)
    assert why == ["P.Hyp: ours put it under domain and theirs under assumptions, from open in the base (domain[P.Hyp] and "
                   "assumptions[P.Hyp]): took theirs, which `check.py --update` classifies"]
    assert merged["assumptions"] == theirs["assumptions"] and "open" not in merged and "domain" not in merged
    assert M.merge(base, ours, theirs) == (merged, why), "deterministic"
    swapped, why = M.merge(base, theirs, ours)
    assert swapped["domain"] == ours["domain"] and "assumptions" not in swapped and "open" not in swapped, \
        "theirs, not a pick"


FIXTURES = HERE / "fixtures" / "merge"


def test_a_definition_taken_out_of_a_module_next_adds_readers_to_merges_as_the_side_that_took_it_out(tmp_path, capsys):
    """A branch merged onto `next` moves `Hash.Digest` out of `Definitions.Hash.Basic` while `next` adds readers there.
    The fixture is that group on each side with the records of its readers (a sample of the moving side's new ones), as
    locks once listed them.  Read without them, `next`'s group is the base's, so the merge is the moving side's group,
    cleanly."""
    sides = json.loads((FIXTURES / "hash-basic-definition-taken-out-onto-next.json").read_text())
    for n in ("base", "ours", "theirs"):
        (tmp_path / n).write_text(A.text(sides[n]))
    assert M.main([*(str(tmp_path / n) for n in ("base", "ours", "theirs")), "Security/lean-audit.json"]) == 0
    assert capsys.readouterr().err == ""
    group = json.loads((tmp_path / "ours").read_text())["reads"]["Definitions.Hash.Basic"]
    hash_basic = {n: sides[n]["reads"]["Definitions.Hash.Basic"] for n in sides}
    assert hash_basic["ours"]["guarantees"] != hash_basic["base"]["guarantees"]
    assert hash_basic["ours"]["definitions"] == hash_basic["base"]["definitions"]
    assert group == {"digest": A.module_digest(hash_basic["theirs"]["definitions"]),
                     "definitions": dict(sorted(hash_basic["theirs"]["definitions"].items()))}


def test_a_branch_onto_next_leaves_no_read_of_the_name_it_renamed(tmp_path, capsys):
    """A branch that renames `J10R.Layout` to `NoiseLayout`, merged onto `next` as drive merged it: `next` added readers of
    `Definitions.Accounting.TS.J10R`, and the union kept `Layout` and its four fields beside the new names, which the
    after-landing Lean check, with no regenerate, fails on.  The fixture is that group on each side with the records of its
    readers, as locks once listed them.  Read without them, `next`'s group is the base's, so the merge is the renaming
    branch's group, cleanly."""
    sides = json.loads((FIXTURES / "j10r-layout-renamed-onto-next.json").read_text())
    m = "Definitions.Accounting.TS.J10R"
    old = [d for d in sides["base"]["reads"][m]["definitions"] if ".Layout" in d]
    assert len(old) == 5 and all(d in sides["ours"]["reads"][m]["definitions"] for d in old)
    assert sides["ours"]["reads"][m]["guarantees"] != sides["base"]["reads"][m]["guarantees"]
    for n in ("base", "ours", "theirs"):
        (tmp_path / n).write_text(A.text(sides[n]))
    assert M.main([*(str(tmp_path / n) for n in ("base", "ours", "theirs")), "Security/Properties/lean-audit.json"]) == 0
    assert capsys.readouterr().err == ""
    group = json.loads((tmp_path / "ours").read_text())["reads"][m]
    assert group["definitions"] == sides["theirs"]["reads"][m]["definitions"]
    assert not set(old) & set(group["definitions"]) and f"{m.removeprefix('Definitions.')}.NoiseLayout" in group["definitions"]
    assert "guarantees" not in group


def test_a_definition_one_side_deleted_with_its_guarantee_stays_deleted():
    """A side that deletes a guarantee and the definition only it read (next's `NetTiming.Guarantees.EncardAccSeqsLeBlocks`),
    while the other adds a reader of the module: the union brought the definition back.  Where the other side added or
    restated no guarantee, only changed the module, nothing of the merged tree can read it, and no candidate is named."""
    base = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"))
    ours = record(t=thm([F]))
    theirs = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"), v=thm([H], "P.h n = n", "00000000000000cc"))
    merged, why = M.merge(base, ours, theirs, candidate := [])
    assert why == [] and candidate == ["reads[P.Defs].definitions[P.g] (removed)"]
    assert A.text(merged) == A.text(record(t=thm([F]), v=thm([H], "P.h n = n", "00000000000000cc")))
    t, u = thm([F, H], "P.h n = n", "00000000000000aa"), thm([F, G], "P.g n = n", "00000000000000bb")
    f9 = read("P.f", "P.Defs", "9" * 16)
    theirs = record(t=thm([f9, H], "P.h n = n", "00000000000000aa"), u=thm([f9, G], "P.g n = n", "00000000000000bb"))
    merged, why = M.merge(record(t=t, u=u), record(t=t), theirs, candidate := [])
    assert why == [] and candidate == []
    assert A.text(merged) == A.text(record(t=thm([f9, H], "P.h n = n", "00000000000000aa")))


def test_a_32_bit_group_merged_with_a_rehashed_one_is_a_candidate_the_train_regenerates():
    base = record(t=thm([F]))
    old = base | {"reads": A.read_groups(A.guarantee_reads({"pins": {"P.t": thm([F])}}), old=True)}
    ours = old | {"reads": A.read_groups(A.guarantee_reads({"pins": {"P.t": thm([F]), "P.u": thm([F, G])}}), old=True),
                  "guarantees": record(t=thm([F]), u=thm([F, G]))["guarantees"]}
    merged, why = M.merge(old, ours, base, candidate := [])
    assert why == [] and candidate == ["reads[P.Defs] (32-bit and SHA-256 hashes mixed)"]
    tree = built(t=thm([F]), u=thm([F, G]))
    fails = A.judge(merged, tree, FILES)[0]
    assert fails and all(map(A.RECORD_DIFFERENCE.match, fails)), fails
    assert A.judge(A.regenerated(merged, tree, FILES, fails), tree, FILES)[0] == []


def before_the_rename(rec: dict) -> dict:
    """``rec`` as a branch from before guarantees were called that writes it: under `pins`."""
    return {("pins" if k == "guarantees" else k): v for k, v in rec.items()}


def listing_readers(rec: dict, key: str = "guarantees", **guarantees: dict) -> dict:
    """``rec`` as a lock that still lists readers writes it, each `reads` group listing under ``key`` the guarantees that read it
    (``guarantees``, as for `record`)."""
    by = A.readers(A.guarantee_reads(built(**guarantees)))
    return rec | {"reads": {m: g | {key: by[m]} for m, g in rec["reads"].items()}}


def test_a_side_whose_reads_list_their_readers_merges_to_today_s_record_and_normalizes(tmp_path):
    """A branch whose locks still list readers merged with one whose don't, either way round, or two such branches (one from before
    the rename too) are read without their readers, so the merge is the merged tree's record; `--normalize` drops them from
    a lock alone."""
    t, u, v = thm([F]), thm([F, G], "P.g n = n", "00000000000000bb"), thm([H, K], "P.h n = P.k n", "00000000000000cc")
    base, ours, theirs, tree = record(t=t), record(t=t, u=u), record(t=t, v=v), record(t=t, u=u, v=v)
    old = {"base": listing_readers(base, t=t), "ours": before_the_rename(listing_readers(ours, "pins", t=t, u=u)),
           "theirs": listing_readers(theirs, t=t, v=v)}
    assert all(A.with_readers(A.as_guarantees(d)) for d in old.values())
    for sides in ({"base": old["base"], "ours": ours, "theirs": old["theirs"]},
                  {"base": base, "ours": old["ours"], "theirs": theirs}, old):
        for n, d in sides.items():
            (tmp_path / n).write_text(A.text(d))
        assert M.main([str(tmp_path / n) for n in ("base", "ours", "theirs")]) == 0
        assert (tmp_path / "ours").read_text() == A.text(tree)
    path = tmp_path / "lean-audit.json"
    path.write_text(A.text(listing_readers(tree, t=t, u=u, v=v)))
    assert M.main(["--normalize", str(path)]) == 0 and path.read_text() == A.text(tree)
    assert M.normalize(path) == 0


def test_a_side_from_before_the_rename_merges_under_today_s_names(tmp_path, capsys):
    """A side under both the old and today's names is no record, so the other side's stands, with a warning."""
    base = record(t=thm([F]))
    ours = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"))
    theirs = record(t=thm([F]), v=thm([H, K], "P.h n = P.k n", "00000000000000cc"))
    tree = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"), v=thm([H, K], "P.h n = P.k n", "00000000000000cc"))
    for n, d in (("base", before_the_rename(base)), ("ours", ours), ("theirs", before_the_rename(theirs))):
        (tmp_path / n).write_text(A.text(d))
    assert M.main([str(tmp_path / n) for n in ("base", "ours", "theirs")]) == 0
    assert (tmp_path / "ours").read_text() == A.text(tree)
    capsys.readouterr()
    (tmp_path / "theirs").write_text(A.text(theirs | {"pins": {"P.w": {}}}))
    assert M.main([str(tmp_path / n) for n in ("base", "ours", "theirs")]) == 0
    assert (tmp_path / "ours").read_text() == A.text(tree)
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and err[0].startswith(f"merge.py: {tmp_path / 'ours'}: theirs is not a record (")
    assert err[0].endswith("): took ours")


def test_a_side_that_is_not_json_is_the_other_side_and_only_two_leave_git_s_conflict(tmp_path, capsys, monkeypatch):
    """Bad JSON on one side takes the other side's record, a bad base merges as an empty one, a merge that raises takes
    theirs, and only two bad sides leave git's own text merge, as a conflict (exit 1)."""
    base = record(t=thm([F]))
    ours = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"))
    theirs = record(t=thm([F]), v=thm([H, K], "P.h n = P.k n", "00000000000000cc"))
    files = [str(tmp_path / n) for n in ("base", "ours", "theirs")]

    def run(b, o, t) -> tuple[int, str, list[str]]:
        for n, d in (("base", b), ("ours", o), ("theirs", t)):
            (tmp_path / n).write_text(d if isinstance(d, str) else A.text(d))
        rc = M.main(files + ["Security/lean-audit.json"])
        return rc, (tmp_path / "ours").read_text(), capsys.readouterr().err.splitlines()
    rc, got, err = run(base, "{\"guarantees\": ", theirs)
    assert (rc, got) == (0, A.text(theirs)) and len(err) == 1
    assert err[0].startswith("merge.py: Security/lean-audit.json: ours is not a record (")
    assert err[0].endswith(": took theirs")
    rc, got, err = run(base, ours, "[1, 2]")
    assert (rc, got) == (0, A.text(ours)) and err == [
        "merge.py: Security/lean-audit.json: theirs is not a record (not a JSON object): took ours"]
    rc, got, err = run("<<<<<<<", ours, theirs)
    assert rc == 0 and json.loads(got)["guarantees"].keys() == {"P.t", "P.u", "P.v"}
    assert err[0].startswith("merge.py: Security/lean-audit.json: base is not a record (")
    assert err[0].endswith("): merged as if the base were empty")
    monkeypatch.setattr(M, "merge", lambda *_: 1 / 0)
    rc, got, err = run(base, ours, theirs)
    assert (rc, got) == (0, A.text(theirs)) and err == [
        "merge.py: Security/lean-audit.json: the merge failed (ZeroDivisionError: division by zero): took theirs"]
    rc, got, err = run(base, "{\"a\": 1,\n", "{\"a\": 2,\n")
    assert rc == 1 and "<<<<<<< ours" in got and ">>>>>>> theirs" in got and len(err) == 1
    assert err[0].startswith("merge.py: Security/lean-audit.json: neither side is a record (ours is not a record (")
    assert err[0].endswith("): left to git's merge, as a conflict")


def test_normalize_recomputes_each_digest_from_its_definitions(tmp_path):
    rec = record(t=thm([F]), v=thm([H, K], "P.h n = P.k n", "00000000000000cc"))
    stale = json.loads(json.dumps(rec))
    for g in stale["reads"].values():
        g["digest"] = "0" * 32
    path = tmp_path / "lean-audit.json"
    path.write_text(A.text(stale))
    assert M.main(["--normalize", str(path)]) == 0
    assert path.read_text() == A.text(rec)
    assert M.normalize(path) == 0


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert not check or r.returncode == 0, r.stderr
    return r


def test_git_merges_two_branches_that_add_guarantees_and_takes_theirs_where_both_changed_one_entry(tmp_path, monkeypatch):
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    (repo / "tools" / "verity" / "lean").mkdir(parents=True)
    for f in ("merge.py", "check.py", "modules.py"):
        shutil.copy(HERE.parent / f, repo / "tools" / "verity" / "lean" / f)
    (repo / ".gitattributes").write_text("lean-audit.json merge=lean-audit\n")
    rec = repo / "pkg" / "lean-audit.json"
    rec.parent.mkdir()
    git(repo, "init", "-q", "-b", "main")
    u, v = thm([F, G], "P.g n = n", "00000000000000bb"), thm([H, K], "P.h n = P.k n", "00000000000000cc")
    branches = {"main": record(t=thm([F])), "a": record(t=thm([F]), u=u), "b": record(t=thm([F]), v=v),
                "c": record(t=thm([F], "P.f n ≤ n", "00000000000000dd"), deps={"mathlib": "b" * 64, "toolchain": "c" * 64}),
                "d": record(t=thm([F], "P.f n < n + 1", "00000000000000ee"), deps={"mathlib": "d" * 64, "toolchain": "c" * 64})}
    for name, doc in branches.items():
        if name != "main":
            git(repo, "checkout", "-q", "-b", name, "main")
        rec.write_text(A.text(doc))
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", name)
    git(repo, "checkout", "-q", "a")

    plain = git(repo, "merge", "--no-edit", "b", check=False)
    assert plain.returncode and "CONFLICT (content): Merge conflict in pkg/lean-audit.json" in plain.stdout
    git(repo, "merge", "--abort")

    subprocess.run([sys.executable, "tools/verity/lean/merge.py", "--install"], cwd=repo, check=True, capture_output=True)
    assert git(repo, "config", "merge.lean-audit.driver").stdout.strip() == M.driver()
    git(repo, "merge", "--no-edit", "b")
    assert rec.read_text() == A.text(record(t=thm([F]), u=u, v=v))

    git(repo, "checkout", "-q", "c")
    merged = git(repo, "merge", "--no-edit", "d")
    assert [s for s in merged.stderr.splitlines() if s.startswith("merge.py:")] == [
        "merge.py: pkg/lean-audit.json: dependencies[mathlib]: changed on both sides, differently: took theirs",
        "merge.py: pkg/lean-audit.json: guarantees[P.t] (theirs): taken without regenerating it; the lock run after "
        "landing regenerates it"]
    assert git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split() == []
    assert rec.read_text() == A.text(branches["d"])


def test_shape_entries_and_frozen_names_merge_one_by_one():
    """Two branches that each freeze or enter a different guarantee merge; one entry set two ways is theirs', named."""
    base = {"shape": {"premises": [], "properties": {}, "frozen": {"P.a": "RESTATE"}}}
    ours = {"shape": {"premises": [], "properties": {"P.b": {"function": "P.f", "outcome": "accept"}}, "frozen": {"P.a": "RESTATE"}}}
    theirs = {"shape": {"premises": [], "properties": {}, "frozen": {"P.c": "lemma"}}}
    merged, why = M.merge(base, ours, theirs)
    assert why == [] and merged["shape"] == {"premises": [], "properties": {"P.b": {"function": "P.f", "outcome": "accept"}},
                                             "frozen": {"P.c": "lemma"}}
    merged, why = M.merge(base, {"shape": {"frozen": {"P.a": "x"}}}, {"shape": {"frozen": {"P.a": "y"}}})
    assert why == ["shape[frozen][P.a]: changed on both sides, differently: took theirs"]
    assert merged["shape"]["frozen"] == {"P.a": "y"}


def test_two_sides_restating_one_guarantee_s_inline_premises_merge_premise_by_premise():
    """`Security/Properties/lean-audit.json`'s `shape.inline` entry for `scheduledRequirement` at c92bc0a53f (ours),
    5e3e9ea235 (theirs) and their merge base fc4958a86b, where all three list the guarantee with one record: both sides add
    the two premises restated over `sessionLayout`, ours keeping the two texts they restate and theirs dropping them.
    Compared whole, the guarantee's map would be one side's; premise by premise the added ones stay, and the two texts ours
    has as the base had them go, since ours states the guarantee as theirs does (`held` keeps none).  The sides gave each
    added premise a different reason: each keeps theirs', named; with ours' reasons as theirs', the guarantee merges to
    theirs' map, naming none."""
    sides = json.loads((FIXTURES / "inline-scheduled-requirement.json").read_text())
    g = "Properties.RegisteredCircuit.SecurityProofs.scheduledRequirement"
    assert sides["base"]["guarantees"] == sides["ours"]["guarantees"] == sides["theirs"]["guarantees"] and g in sides["base"]["guarantees"]
    base, ours, theirs = (sides[n]["shape"]["inline"][g] for n in ("base", "ours", "theirs"))
    added, dropped = [p for p in theirs if p not in base], [p for p in base if p not in theirs]
    assert len(added) == len(dropped) == 2 and all(p in ours for p in added) and all(ours[p] == base[p] for p in dropped)
    assert all(ours[p] != theirs[p] for p in added)
    merged, why = M.merge(sides["base"], sides["ours"], sides["theirs"], candidate := [])
    assert merged["shape"]["inline"][g] == theirs
    assert why == [f"shape[inline][{g}][{p}]: added on both sides, differently: took theirs" for p in ours if p in added]
    assert candidate == []
    agreed = sides["ours"] | {"shape": {"inline": {g: ours | {p: theirs[p] for p in added}}}}
    merged, why = M.merge(sides["base"], agreed, sides["theirs"], candidate := [])
    assert why == [] and merged["shape"]["inline"][g] == theirs and candidate == []


def test_an_inline_premise_both_sides_give_different_reasons_keeps_theirs_and_the_other_shape_sections_merge_whole():
    """`shape.inline` merges premise by premise (`DEEPER`), and a premise both sides give different reasons, added or
    changed, keeps theirs' reason, named (the check asks only that one be given), while a reason one side changed and the
    other dropped is the changed one; `frozen`, `properties` and `conjectures` entries still merge whole, so two sides that
    change one entry differently take theirs' whole entry even where they changed different fields of it."""
    prop = {"function": "P.f", "outcome": "accept"}
    base = {"shape": {"inline": {"P.t": {"a": "why a", "b": "why b"}}, "properties": {"P.p": prop}, "frozen": {"P.z": "lemma"}}}

    def inline(**ps):
        return {"shape": base["shape"] | {"inline": {"P.t": ps}}}

    def frozen(why):
        return {"shape": base["shape"] | {"frozen": {"P.z": why}}}

    def prop_(**f):
        return {"shape": base["shape"] | {"properties": {"P.p": prop | f}}}

    merged, why = M.merge(base, inline(a="why a", b="why b", c="why c"), inline(b="why b", d="why d"))
    assert why == [] and merged["shape"]["inline"] == {"P.t": {"b": "why b", "c": "why c", "d": "why d"}}
    assert M.merge(base, inline(a="why a", b="why b", c="one"), inline(a="why a", b="why b", c="one"))[1] == []
    merged, why = M.merge(base, inline(a="why a", b="why b", c="one"), inline(a="why a", b="why b", c="another"),
                          candidate := [])
    assert merged["shape"]["inline"] == {"P.t": {"a": "why a", "b": "why b", "c": "another"}} and candidate == []
    assert why == ["shape[inline][P.t][c]: added on both sides, differently: took theirs"]
    merged, why = M.merge(base, inline(a="ours", b="why b"), inline(a="theirs", b="why b"), candidate := [])
    assert merged["shape"]["inline"]["P.t"]["a"] == "theirs" and candidate == []
    assert why == ["shape[inline][P.t][a]: changed on both sides, differently: took theirs"]
    merged, why = M.merge(base, inline(b="why b"), inline(a="changed", b="why b"))
    assert why == ["shape[inline][P.t][a]: removed on one side and changed on the other: kept theirs' change"]
    assert merged["shape"]["inline"] == {"P.t": {"a": "changed", "b": "why b"}}
    merged, why = M.merge(base, inline(b="why b"), inline(a="why a"))
    assert why == [] and "inline" in merged["shape"] and merged["shape"]["inline"] == {}, \
        "a guarantee the record doesn't list, left with no premise, goes"
    merged, why = M.merge(base, frozen("restatable"), frozen("open"))
    assert why == ["shape[frozen][P.z]: changed on both sides, differently: took theirs"]
    assert merged["shape"]["frozen"] == {"P.z": "open"}
    merged, why = M.merge(base, prop_(function="P.g"), prop_(outcome="refuse"))
    assert why == ["shape[properties][P.p]: changed on both sides, differently: took theirs"]
    assert merged["shape"]["properties"] == {"P.p": prop | {"outcome": "refuse"}}, "theirs' whole entry, not field by field"


def test_a_branch_regenerated_on_an_older_tree_keeps_the_inline_texts_next_still_prints(tmp_path, capsys):
    """`Security/lean-audit.json` at 3be133850d (theirs), a branch whose `--update` ran on a tree based on 538c00f54e (the base),
    merged onto `next` at ab4de4a2d2 (ours).  The branch's tree prints two of `LiveCoinsDrawn`'s five premises, so its
    `--update` dropped the other three, renamed `Coins` texts that `next`'s tree still prints; premise by premise the merge
    applied the drop, and the merged lock failed `next`'s Lean check on all three.  `next` in turn re-keyed one of
    `ComputeOfPouw`'s texts under `Accounting.OpClass.oracle`, dropping the `Accounting.Op.oracle` one the branch has as the
    base had it.  Every side lists both guarantees with one record, and `reads` holds the two `Guarantees` definitions their
    signatures name: `next` restated both (each definition has another hash), and the branch neither.  So the three
    `LiveCoinsDrawn` premises stay, each named on stderr, and the old `Op.oracle` text goes: the branch states
    `ComputeOfPouw` as the base does, so the merged tree prints it as `next`'s, where the text no longer offends."""
    sides = json.loads((FIXTURES / "inline-live-coins-onto-next.json").read_text())
    live, pouw = "Accounting.SecurityProofs.Remote.LiveCoinsDrawn", "Accounting.SecurityProofs.ComputeOfPouw"
    assert sides["base"]["guarantees"] == sides["ours"]["guarantees"] == sides["theirs"]["guarantees"]
    defs = {n: {d: h for g in sides[n]["reads"].values() for d, h in g["definitions"].items()} for n in sides}
    assert len(defs["base"]) == 2 and defs["theirs"] == defs["base"]
    assert all(defs["ours"][d] != h for d, h in defs["base"].items())
    inline = {n: sides[n]["shape"]["inline"] for n in sides}
    assert inline["ours"][live] == inline["base"][live] and inline["theirs"][pouw] == inline["base"][pouw]
    gone = {live: [p for p in inline["base"][live] if p not in inline["theirs"][live]],
            pouw: [p for p in inline["base"][pouw] if p not in inline["ours"][pouw]]}
    assert len(gone[live]) == 3 and all("Draw.Coins" in p or "Draw.DrawCoins" in p for p in gone[live])
    assert [p for p in inline["ours"][pouw] if p not in inline["base"][pouw]] == [
        p.replace("Accounting.Op.oracle", "Accounting.OpClass.oracle") for p in gone[pouw]]
    for n in ("base", "ours", "theirs"):
        (tmp_path / n).write_text(A.text(sides[n]))
    assert M.main([*(str(tmp_path / n) for n in ("base", "ours", "theirs")), "Security/lean-audit.json"]) == 0
    err = capsys.readouterr().err.strip().splitlines()
    assert err == [f"merge.py: Security/lean-audit.json: shape[inline][{live}][{p}] (kept): a shape.inline premise one "
                   f"side dropped and the other, restating its guarantee, has as the base had it, kept; `check.py "
                   f"--update` drops it where the merged tree's premise no longer offends" for p in gone[live]]
    merged = json.loads((tmp_path / "ours").read_text())
    assert merged["shape"]["inline"] == {live: inline["base"][live], pouw: inline["ours"][pouw]}
    assert merged["guarantees"] == sides["base"]["guarantees"] and merged["reads"] == sides["ours"]["reads"]


def test_an_inline_premise_one_side_dropped_stays_where_the_other_restated_its_guarantee():
    """A premise one side dropped and the other has with the base's reason stays (`held`), alone or as a guarantee's whole
    entry, whichever side dropped it, while the side that has it states the guarantee otherwise than the base and the
    dropping side: another record, or another hash of a definition its signature names; with the other side adding a premise
    of its own, both are there.  Where that side states it as the base does, or as the dropping side does, the premise goes,
    as a three-way merge drops it.  A guarantee one side removed from the record loses its entry, a premise both sides
    dropped goes, and a reason one side changed while the other dropped the premise is the changed one, named."""
    rec = {"owner": "@lean", "signature": ["P.t : P.T"], "assumptions": [], "type_hash": "a" * 64}
    new, other = rec | {"type_hash": "b" * 64}, rec | {"type_hash": "c" * 64}
    t, u, kept_a = {"a": "why a", "b": "why b"}, {"c": "why c"}, ["shape[inline][P.t][a] (kept)"]
    base = {"guarantees": {"P.t": rec, "P.u": rec}, "shape": {"inline": {"P.t": t, "P.u": u}}}

    def inline(said: dict | None = None, **ps: dict) -> dict:
        """``base`` with `shape.inline` holding ``ps``, P.t's premises as ``t`` and P.u's as ``u``, and the records ``said``."""
        return base | {"guarantees": base["guarantees"] | (said or {}),
                       "shape": {"inline": {f"P.{g}": v for g, v in ps.items()}}}
    no_a, restated = inline(t={"b": "why b"}, u=u), inline({"P.t": new, "P.u": new}, t=t, u=u)
    for ours, theirs in ((no_a, restated), (restated, no_a)):
        merged, why = M.merge(base, ours, theirs, candidate := [])
        assert why == [] and merged["shape"]["inline"] == {"P.t": t, "P.u": u} and candidate == kept_a
        assert merged["guarantees"]["P.t"] == new
    for ours, theirs in ((no_a, base), (base, no_a)):
        merged, why = M.merge(base, ours, theirs, candidate := [])
        assert why == [] and merged["shape"]["inline"] == {"P.t": {"b": "why b"}, "P.u": u} and candidate == []
    alike = inline({"P.t": new}, t={"b": "why b"}, u=u)
    merged, why = M.merge(base, alike, inline({"P.t": new}, t=t, u=u), candidate := [])
    assert why == [] and merged["shape"]["inline"] == {"P.t": {"b": "why b"}, "P.u": u} and candidate == []
    merged, why = M.merge(base, alike, inline({"P.t": other}, t=t, u=u), candidate := [])
    assert merged["shape"]["inline"] == {"P.t": t, "P.u": u} and kept_a[0] in candidate
    was, reread = ({"reads": {"M": {"definitions": {"P.T": h * 64}}}} for h in "de")
    merged, why = M.merge(base | was, no_a | was, restated | {"guarantees": base["guarantees"]} | reread, candidate := [])
    assert why == [] and merged["shape"]["inline"] == {"P.t": t, "P.u": u} and candidate == kept_a
    merged, why = M.merge(base, inline(t=t), restated, candidate := [])
    assert why == [] and merged["shape"]["inline"] == {"P.t": t, "P.u": u} and candidate == ["shape[inline][P.u][c] (kept)"]
    merged, why = M.merge(base, inline(t=t), base, candidate := [])
    assert why == [] and merged["shape"]["inline"] == {"P.t": t} and candidate == []
    merged, why = M.merge(base, base | {"shape": {}}, restated, candidate := [])
    assert why == [] and merged["shape"]["inline"] == {"P.t": t, "P.u": u} and len(candidate) == 3
    merged, why = M.merge(base, no_a, inline({"P.t": new}, t=t | {"d": "why d"}, u=u), candidate := [])
    assert why == [] and merged["shape"]["inline"] == {"P.t": t | {"d": "why d"}, "P.u": u} and candidate == kept_a
    merged, why = M.merge(base, no_a, inline(t=t | {"d": "why d"}, u=u), candidate := [])
    assert why == [] and merged["shape"]["inline"] == {"P.t": {"b": "why b", "d": "why d"}, "P.u": u} and candidate == []

    unlisted = {"guarantees": {"P.u": rec}, "shape": {"inline": {"P.u": u}}}
    merged, why = M.merge(base, unlisted, base, candidate := [])
    assert why == [] and list(merged["guarantees"]) == ["P.u"] and merged["shape"]["inline"] == {"P.u": u} and candidate == []
    merged, why = M.merge(base, no_a, inline(t={"b": "why b"}, u=u | {"d": "why d"}), candidate := [])
    assert why == [] and merged["shape"]["inline"] == {"P.t": {"b": "why b"}, "P.u": u | {"d": "why d"}} and candidate == []
    merged, why = M.merge(base, no_a, inline(t={"a": "changed", "b": "why b"}, u=u))
    assert why == ["shape[inline][P.t][a]: removed on one side and changed on the other: kept theirs' change"]
    assert merged["shape"]["inline"] == {"P.t": {"a": "changed", "b": "why b"}, "P.u": u}


def test_a_renamed_premise_merges_to_one_entry_and_a_text_a_restated_statement_still_prints_stays():
    """`TileUnpredictableOfNCPU`'s premise as a rename of its binder `γ₀` to `g₀` left it, which changed the printed text and no
    record.  Theirs renames it, restating the guarantee (another hash of the `Guarantees` definition its signature names),
    while ours has the old text as the base had it: the merge has the new text alone, not the old one beside it (`next`
    held 19 such old texts over its `INLINE` cap).  A base that lists both texts, merged with a side whose `--update` dropped
    the old one (the lock run) and one that has both as the base had them, keeps the new one alone too.  Where the side
    that has both restated the guarantee and the other didn't (a branch whose `--update` ran on a tree before the rename,
    which prints the old text only), the new text stays: the merged statement prints it."""
    g, d = "Accounting.SecurityProofs.TileUnpredictableOfNCPU", "Accounting.Guarantees.TileUnpredictableOfNCPU"
    rec = {"owner": "@memory-accounting", "signature": [f"{g} : {d}"], "assumptions": [], "type_hash": "a" * 64}
    old, new = (f"∀ (u : Nat), instLTNat.lt u U.layout.N → Real.instLE.le P.d.cast (instHMul.hMul {b} 3)" for b in "γg")
    why = "a bound that is no plain range, inline when the rule came"

    def lock(h: str, *texts: str) -> dict:
        """A lock listing ``g`` with the premises ``texts``, reading ``d`` at hash ``h``."""
        return {"guarantees": {g: rec}, "reads": {"Definitions.Accounting.Guarantees.Pouw": {"definitions": {d: h * 64}}},
                "shape": {"inline": {g: dict.fromkeys(texts, why)}}}
    merged, _ = M.merge(lock("1", old), lock("1", old), lock("2", new), candidate := [])
    assert merged["shape"]["inline"] == {g: {new: why}} and candidate == []
    for ours, theirs in ((lock("1", new), lock("1", old, new)), (lock("1", old, new), lock("1", new))):
        merged, _ = M.merge(lock("1", old, new), ours, theirs, candidate := [])
        assert merged["shape"]["inline"] == {g: {new: why}} and candidate == []
    merged, _ = M.merge(lock("1", old, new), lock("2", old, new), lock("1", old), candidate := [])
    assert merged["shape"]["inline"] == {g: {old: why, new: why}} and candidate == [f"shape[inline][{g}][{new}] (kept)"]


def shaped(n: int, *more: int) -> dict:
    """A lock shaped like main's, with ``n`` `reads` groups and those ``more`` names, one definition each."""
    def group(i: int) -> dict:
        defs = {f"M{i:04}.d": sha(f"d{i}")}
        return {"digest": A.module_digest(defs), "definitions": defs}
    return {"format": A.FORMAT, "reads": {f"M{i:04}": group(i) for i in sorted({*range(n), *more})}}


def test_check_py_writes_the_lock_format_merge_py_knows():
    """`check.py` writes its `FORMAT` into every lock (`dump`) and `merge.py` merges up to its own: the two move together, so
    a change to what a lock holds bumps both, and an older copy leaves a lock written after it to git's conflict."""
    assert A.FORMAT == M.FORMAT and A.ORDER[0] == "format"


def test_every_lock_check_py_writes_carries_its_format(tmp_path):
    A.dump(tmp_path, {"guarantees": {}, "roots": ["P"]})
    assert list(json.loads((tmp_path / A.POLICY).read_text()).items()) == [("format", A.FORMAT), ("roots", ["P"]),
                                                                          ("guarantees", {})]


def test_a_lock_in_a_format_newer_than_the_driver_s_is_left_to_git_s_conflict(tmp_path, capsys):
    """A lock without `format` is one from before it (0), and one in the driver's own format merges; one in a newer format,
    or one whose format isn't a number, holds what this copy may not know how to merge, so the driver leaves git's text
    merge, as a conflict, saying to merge main's `tools/verity/lean` first."""
    base = record(t=thm([F]))
    files = [str(tmp_path / n) for n in ("base", "ours", "theirs")]

    def run(theirs: dict) -> int:
        for n, d in (("base", base), ("ours", base | {"meaning": ["P", "P.More"]}), ("theirs", theirs)):
            (tmp_path / n).write_text(A.text(d))
        return M.main(files)
    assert run(base | {"format": M.FORMAT}) == 0 and json.loads((tmp_path / "ours").read_text())["format"] == M.FORMAT
    assert json.loads((tmp_path / "ours").read_text())["meaning"] == ["P", "P.More"]
    capsys.readouterr()
    for f in (M.FORMAT + 1, "1", True):
        assert run(base | {"format": f}) == 1
        assert (f"theirs is lock format {f!r}, which this merge.py (format {M.FORMAT}) doesn't know: merge main's "
                f"tools/verity/lean first") in capsys.readouterr().err


def test_a_merge_that_loses_most_of_a_side_s_reads_groups_is_left_to_git_s_conflict(tmp_path, monkeypatch, capsys):
    """An older `merge.py` that merged main into an old branch collapsed Security's lock to one `reads` group (main has 1,316)
    and Properties' to one (416), and only the quick suites caught it, indirectly.  A merged record that drops more than half
    of a side's entries in a section, beyond what the base accounts for, is left to git's conflict (`losses`): a conflict is
    seen, where the loss would land."""
    base, ours, theirs = shaped(40), shaped(40, 40), shaped(40, 41)
    merged, why = M.merge(base, ours, theirs)
    assert why == [] and len(merged["reads"]) == 42 and M.losses(base, ours, theirs, merged) == []
    collapsed = merged | {"reads": dict(list(merged["reads"].items())[:1])}
    assert [s.split(", losing")[0] for s in M.losses(base, ours, theirs, collapsed)] == [
        "reads: the merge keeps 1 of ours' 41", "reads: the merge keeps 1 of theirs' 41"]

    files = [str(tmp_path / n) for n in ("base", "ours", "theirs")]
    for n, d in (("base", base), ("ours", ours), ("theirs", theirs)):
        (tmp_path / n).write_text(A.text(d))
    monkeypatch.setattr(M, "merge", lambda *a, **k: (collapsed, []))
    assert M.main([*files, "Security/lean-audit.json"]) == 1
    assert "<<<<<<< ours\n" in (tmp_path / "ours").read_text()
    err = capsys.readouterr().err
    assert err.startswith("merge.py: Security/lean-audit.json: the merged record loses most of a section") and \
        "reads: the merge keeps 1 of ours' 41, losing 40 the base doesn't account for (M0001, M0002, M0003, ...)" in err, err


def test_what_the_base_accounts_for_is_no_loss():
    """A group the other side removed, as the base had it, is that side's removal; an entry the merge moved to another
    section of its `EXCLUSIVE` group (an open premise now a domain hypothesis) is kept; a section of one entry that the
    merge drops is a loss."""
    base, ours = shaped(40), shaped(40, 40)
    theirs = {"format": A.FORMAT, "reads": {m: g for m, g in base["reads"].items() if m >= "M0030"}}
    merged, _ = M.merge(base, ours, theirs)
    assert sorted(merged["reads"]) == [f"M{i:04}" for i in range(30, 41)] and M.losses(base, ours, theirs, merged) == []

    premise = {"digest": "d" * 64, "issue": "LEAN-1"}
    base = ours = {"open": {f"P.{i}": premise for i in range(4)}, "roots": ["P"]}
    theirs = {"open": {"P.3": premise}, "domain": {f"P.{i}": {"digest": "d" * 64, "why": "w"} for i in range(3)},
              "roots": ["P"]}
    assert M.losses(base, ours, theirs, theirs) == []
    assert M.losses(base, ours, theirs, theirs | {"domain": {}}) == [
        "open: the merge keeps 1 of ours' 4, losing 3 the base doesn't account for (P.0, P.1, P.2)",
        "domain: the merge keeps 0 of theirs' 3, losing 3 the base doesn't account for (P.0, P.1, P.2)"]
    assert M.losses(base, ours, theirs, theirs | {"roots": []}) == [
        "roots: the merge keeps 0 of ours' 1, losing 1 the base doesn't account for (\"P\")",
        "roots: the merge keeps 0 of theirs' 1, losing 1 the base doesn't account for (\"P\")"]


def test_an_old_branch_merging_main_merges_by_origin_main_s_driver_and_its_own_copy_refuses_to(tmp_path, monkeypatch):
    """Git runs the driver from the worktree of the side merged into, so an old branch that merged main merged main's lock
    by the branch's older `merge.py`: here one that collapses `reads` to one group, as one did.  The registered command
    (`driver`, `research.clone.RECORD_DRIVER`) runs `origin/main`'s copy where the repository has that ref, which merges
    every group.  Without it the worktree's copy runs, which sees that the side `git merge` merges in has `merge.py`
    changes it lacks and leaves git's conflict, naming both copies (`stale`)."""
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    repo, cut = tmp_path / "repo", "    held(base, ours, theirs, out, candidate)\n"
    (lean := repo / "tools" / "verity" / "lean").mkdir(parents=True)
    for f in ("check.py", "settings.py"):
        shutil.copy(HERE.parent / f, lean / f)
    driver = (HERE.parent / "merge.py").read_text()
    assert (old := driver.replace(cut, cut + '    out["reads"] = dict(list(out["reads"].items())[:1])\n')) != driver
    (lean / "merge.py").write_text(old)
    (repo / ".gitattributes").write_text("lean-audit.json merge=lean-audit\n")
    path = repo / "lean-audit.json"

    def commit(msg: str, doc: dict) -> None:
        path.write_text(A.text(doc))
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", msg)
    git(repo, "init", "-q", "-b", "main")
    commit("base", shaped(40))
    subprocess.run([sys.executable, "tools/verity/lean/merge.py", "--install"], cwd=repo, check=True, capture_output=True)
    git(repo, "checkout", "-q", "-b", "old")
    commit("old: a group", shaped(40, 40))
    git(repo, "checkout", "-q", "main")
    (lean / "merge.py").write_text(driver)
    commit("main: a group, and merge.py without the collapse", shaped(40, 41))
    git(repo, "checkout", "-q", "old")

    mine, theirs = (git(repo, "rev-parse", f"{b}:tools/verity/lean/merge.py").stdout for b in ("old", "main"))
    refused = git(repo, "merge", "--no-edit", "main", check=False)
    assert refused.returncode == 1 and "CONFLICT (content): Merge conflict in lean-audit.json" in refused.stdout
    assert (f"the side merged in, main ({git(repo, 'rev-parse', 'main').stdout[:10]}), has tools/verity/lean/merge.py "
            f"{theirs[:10]}, with changes this checkout's {mine[:10]} lacks") in refused.stderr, refused.stderr
    git(repo, "merge", "--abort")

    git(repo, "update-ref", "refs/remotes/origin/main", "main")
    git(repo, "merge", "--no-edit", "main")
    assert json.loads(path.read_text()) == shaped(40, 40, 41)
    assert (lean / "merge.py").read_text() == driver


def test_two_branches_that_each_widen_a_layer_merge_its_imports_and_reasons():
    """Each side added imports to `Definitions.Accounting`'s layer and a clause to its `why`."""
    base = {"layers": {"D": {"imports": ["Mathlib", "D"], "except": [], "why": "trusted text; they import no proof"},
                       "E": {"imports": ["E"], "except": [], "why": "e"}}}
    ours = {"layers": {"D": {"imports": ["Mathlib", "P.Audit", "D"], "except": [], "why": "trusted text; M reads P; they import no proof"},
                       "E": {"imports": ["E"], "except": [], "why": "e"}}}
    theirs = {"layers": {"D": {"imports": ["Mathlib", "Core", "D"], "except": [], "why": "trusted text; C reads Core; they import no proof"},
                         "E": {"imports": ["E", "F"], "except": [], "why": "e"}}}
    merged, why = M.merge(base, ours, theirs)
    assert why == []
    assert merged["layers"] == {"D": {"imports": ["Mathlib", "Core", "P.Audit", "D"], "except": [],
                                      "why": "trusted text; C reads Core; M reads P; they import no proof"},
                                "E": {"imports": ["E", "F"], "except": [], "why": "e"}}
    gone = {"layers": {"D": {"imports": ["D"], "except": [], "why": "trusted text"}, "E": base["layers"]["E"]}}
    merged, why = M.merge(base, gone, theirs)
    assert why == [] and merged["layers"]["D"] == {"imports": ["Core", "D"], "except": [], "why": "trusted text; C reads Core"}, \
        "what one side removed stays removed"
    merged, why = M.merge(base, {"layers": {"D": {**ours["layers"]["D"], "mode": "x"}}},
                          {"layers": {"D": {**ours["layers"]["D"], "mode": "y"}}})
    assert why == ["layers[D].mode: added on both sides, differently: took theirs"] and merged["layers"]["D"]["mode"] == "y"


def test_two_branches_that_each_add_a_root_merge_the_roots():
    """Each side adds a test module to Flock's roots."""
    base = {"roots": ["Flock", "Flock.Test.RecOpening", "VStar"]}
    ours = {"roots": ["Flock", "Flock.Test.RecOpen", "Flock.Test.RecOpening", "VStar"]}
    theirs = {"roots": ["Flock", "Flock.Test.RecOpening", "Flock.Test.ReadWhole", "VStar"]}
    merged, why = M.merge(base, ours, theirs)
    assert why == []
    assert merged["roots"] == ["Flock", "Flock.Test.RecOpen", "Flock.Test.RecOpening", "Flock.Test.ReadWhole", "VStar"]
    merged, why = M.merge(base, {"roots": ["Flock", "VStar"]}, theirs)
    assert why == [] and merged["roots"] == ["Flock", "Flock.Test.ReadWhole", "VStar"], "what one side removed stays removed"


def _lean(tmp: Path, base: str, ours: str, theirs: str) -> tuple[str, int]:
    for n, t in (("base", base), ("ours", ours), ("theirs", theirs)):
        (tmp / n).write_text(t)
    return M.lean(tmp / "base", tmp / "ours", tmp / "theirs")


def test_a_lean_file_whose_sides_only_add_imports_merges_and_any_other_conflict_stays_git_s(tmp_path):
    """Each side adds a module to Guarantees.lean's import list at the same place."""
    base = "import G.Bandwidth\nimport G.Joint\n\ntheorem t : True := trivial\n"
    ours = "import G.Memory\nimport G.Bandwidth\nimport G.Joint\n\ntheorem t : True := trivial\n"
    theirs = "import G.Actual\nimport G.Bandwidth\nimport G.Joint\n\ntheorem t : True := by trivial\n"
    assert _lean(tmp_path, base, ours, theirs) == (
        "import G.Actual\nimport G.Memory\nimport G.Bandwidth\nimport G.Joint\n\ntheorem t : True := by trivial\n", 0)
    text, left = _lean(tmp_path, base, ours.replace(":= trivial", ":= rfl"), theirs)
    assert left == 1 and "<<<<<<< ours" in text and "import G.Actual\nimport G.Memory\n" in text


def test_an_import_one_side_removed_stays_removed_where_the_other_side_moved_it(tmp_path):
    """Ours removes `G.B` and adds `G.F`; theirs moves `G.B` to the end.  Git sees the removal on both sides where `G.B` was
    and two additions at the end, so a merge of that hunk alone brings `G.B` back."""
    imports = lambda *ms: "".join(f"import G.{m}\n" for m in ms) + "\ntheorem t : True := trivial\n"
    text, left = _lean(tmp_path, imports("A", "B", "C", "D", "E"), imports("A", "C", "D", "E", "F"), imports("A", "C", "D", "E", "B"))
    assert left == 0 and text == imports("A", "C", "D", "E", "F")


def test_a_module_system_file_s_imports_merge_in_every_form(tmp_path):
    """On Lean's module system an import list is `public import` lines, among them `meta` and `all` ones (`headers.py`):
    two sides that each add one merge, and one a side removed stays removed."""
    def imports(*ls: str) -> str:
        return "module\n\n" + "".join(f"{s}\n" for s in ls) + "\n@[expose] public section\n\ntheorem t : True := trivial\n"

    text, left = _lean(tmp_path, imports("public import G.A", "public meta import G.B", "import all G.C"),
                       imports("public import G.A", "public import G.X -- ours", "import all G.C"),
                       imports("public import G.A", "public meta import G.B", "meta import G.Y", "import all G.C"))
    assert left == 0 and text == imports("public import G.A", "meta import G.Y", "public import G.X -- ours", "import all G.C")
    assert M._import_names(text) == {"public import G.A", "meta import G.Y", "public import G.X -- ours", "import all G.C"}
    assert not M._import("public section") and not M._import("module") and not M._import("import A import B")


def test_imports_both_sides_add_merge_and_one_both_sides_remove_stays_removed(tmp_path):
    def imports(*ms):
        return "".join(f"import G.{m}\n" for m in ms) + "\ntheorem t : True := trivial\n"
    text, left = _lean(tmp_path, imports("A", "B", "C"), imports("A", "X", "C"), imports("A", "Y", "C"))
    assert left == 0 and text == imports("A", "Y", "X", "C"), "both added: the union; both removed B: gone"
    text, left = _lean(tmp_path, imports("A", "B", "C"), imports("A", "X", "B", "C"), imports("A", "Y", "B", "C"))
    assert left == 0 and text == imports("A", "Y", "X", "B", "C")


def test_a_module_s_import_lists_merge_in_each_form(tmp_path):
    """A module's imports (`public import`, `meta import`, `public meta import`, `import all`) merge as a plain file's do."""
    def imports(*ms):
        return "module\n\n" + "".join(f"{m}\n" for m in ms) + "\n@[expose] public section\n\ntheorem t : True := trivial\n"
    a, c = "public import G.A", "import all G.C"
    text, left = _lean(tmp_path, imports(a, "public import G.B", c), imports(a, "meta import G.X", c),
                       imports(a, "public meta import G.Y", "public import G.B", c))
    assert left == 0 and text == imports(a, "public meta import G.Y", "meta import G.X", c), text


def test_two_ports_of_a_plain_file_that_each_change_its_imports_merge(tmp_path):
    """Both sides put the file on the module system (`modules.py`) and changed its imports: what merges is each side's change
    beyond the port, so theirs' removal of an import ours kept applies, though the base spells it `import`."""
    def ported(*ms):
        return "module\n\n" + "".join(f"public import {m}\n" for m in ms) + "\n@[expose] public section\n\ndef x := 1\n"
    base = "import G.A\nimport G.B\n\ndef x := 1\n"
    text, left = _lean(tmp_path, base, ported("G.A", "G.B", "G.C"), ported("G.A", "G.D"))
    assert left == 0 and text == ported("G.A", "G.D", "G.C"), text


def test_a_branch_from_before_the_port_merges_as_its_codemod_rerun_would(tmp_path):
    """Ours is the port with a hand fix (`import all` of a core module); theirs, a branch made before the port, adds an import
    and changes a proof. Theirs is ported first, so its import comes in public, its proof applies, and nothing conflicts."""
    def ported(*ms, proof="trivial"):
        return ("module\n\n" + "".join(f"{m}\n" for m in ms) + "\n@[expose] public section\n\n"
                f"theorem x : True := {proof}\n")
    base = "import G.A\n\ntheorem x : True := trivial\n"
    ours = ported("public import G.A", "import all Init.Data.Array.Basic")
    theirs = "import G.A\nimport G.B\n\ntheorem x : True := by trivial\n"
    text, left = _lean(tmp_path, base, ours, theirs)
    assert left == 0 and text == ported("public import G.A", "public import G.B", "import all Init.Data.Array.Basic",
                                        proof="by trivial"), text
    text, left = _lean(tmp_path, base, theirs, ours)
    assert left == 0 and text == ported("public import G.A", "import all Init.Data.Array.Basic", "public import G.B",
                                        proof="by trivial"), text


def test_4690_onto_next_drops_the_import_4690_retired(tmp_path):
    """`Security/Definitions/Accounting.lean` as git gave it to the driver in a merge where one side retired a module and the
    other moved its import: the merge drops the retired import, which would otherwise stop Security building."""
    sides = json.loads((FIXTURES / "accounting-retired-import-onto-next.json").read_text())
    text, left = _lean(tmp_path, sides["base"], sides["ours"], sides["theirs"])
    gone = "import Definitions.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod\n"
    assert left == 0 and gone in sides["base"] and gone in sides["ours"] and gone not in sides["theirs"]
    assert gone not in text
    assert M._import_names(text) == M._import_names(sides["theirs"]) | (M._import_names(sides["ours"]) - M._import_names(
        sides["base"]))


def test_a_lakefile_root_or_require_one_side_removed_stays_removed_where_the_other_side_moved_it():
    """As for imports: ours removes a root (a require) and appends another, theirs moves it to the end."""
    import tempfile
    import tomllib
    tmp = Path(tempfile.mkdtemp())
    def roots(*ms):
        return 'name = "p"\n\n[[lean_lib]]\nname = "P"\nroots = [' + ",\n  ".join(f'"{m}"' for m in ms) + "]\n"
    text, left = _lakefile(tmp, roots("A", "B", "C", "D", "E"), roots("A", "C", "D", "E", "F"), roots("A", "C", "D", "E", "B"))
    assert left == 0 and tomllib.loads(text)["lean_lib"][0]["roots"] == ["A", "C", "D", "E", "F"], text
    def req(*ns):
        return 'name = "p"\n' + "".join(f'\n[[require]]\nname = "{n}"\npath = "../{n}"\n' for n in ns)
    text, left = _lakefile(tmp, req("a", "b", "c"), req("a", "c", "d"), req("a", "c", "b"))
    assert left == 0 and [r["name"] for r in tomllib.loads(text)["require"]] == ["a", "c", "d"], text
    text, left = _lakefile(tmp, req("a", "c"), req("a", "c", "x"), req("a", "c", "y"))
    assert left == 0 and [r["name"] for r in tomllib.loads(text)["require"]] == ["a", "c", "x", "y"], "both added: both"


def test_a_root_or_require_one_side_removed_stays_removed_where_git_merges_the_other_side_s_move_cleanly(tmp_path):
    """Roots A..H, ours removes B, theirs moves B after D.  Git merges that without a
    conflict, to [A, C, D, B, E, F, G, H]; the whole merged file is held to `kept`, so B stays removed.  Requires alike."""
    import tomllib
    def roots(*ms):
        return 'name = "p"\n\n[[lean_lib]]\nname = "P"\nroots = [' + ",\n  ".join(f'"{m}"' for m in ms) + "]\n"
    def req(*ns):
        return 'name = "p"\n' + "".join(f'\n[[require]]\nname = "{n}"\npath = "../{n}"\n' for n in ns) + \
            '\n[[lean_lib]]\nname = "P"\n'
    for side, (o, a, b) in {"roots": (roots(*"ABCDEFGH"), roots(*"ACDEFGH"), roots(*"ACDBEFGH")),
                            "require": (req(*"abcdefgh"), req(*"acdefgh"), req(*"acdbefgh"))}.items():
        for n, t in (("base", o), ("ours", a), ("theirs", b)):
            (tmp_path / n).write_text(t)
        clean = subprocess.run(["git", "merge-file", "-p", *(str(tmp_path / n) for n in ("ours", "base", "theirs"))],
                               capture_output=True, text=True)
        assert clean.returncode == 0 and tomllib.loads(clean.stdout) != tomllib.loads(a), "git alone brings it back"
        text, left = _lakefile(tmp_path, o, a, b)
        assert left == 0 and text == a, text
    text, left = _lakefile(tmp_path, req(*"abcd"), req(*"acd"), req(*"acdbx"))
    assert left == 0 and [r["name"] for r in tomllib.loads(text)["require"]] == ["a", "c", "d", "x"], text


def test_git_merges_two_branches_that_each_import_a_new_module_with_the_installed_driver(tmp_path, monkeypatch):
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    (repo / "tools" / "verity" / "lean").mkdir(parents=True)
    for f in ("merge.py", "check.py", "headers.py", "modules.py"):
        shutil.copy(HERE.parent / f, repo / "tools" / "verity" / "lean" / f)
    (repo / ".gitattributes").write_text(f"*.lean merge={M.LEAN}\n")
    g = repo / "G.lean"
    git(repo, "init", "-q", "-b", "main")
    for name, first in (("main", ""), ("a", "import G.Memory\n"), ("b", "import G.Actual\n")):
        if name != "main":
            git(repo, "checkout", "-q", "-b", name, "main")
        g.write_text(first + "import G.Joint\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", name)
    git(repo, "checkout", "-q", "a")
    plain = git(repo, "merge", "--no-edit", "b", check=False)
    assert plain.returncode and "CONFLICT (content): Merge conflict in G.lean" in plain.stdout
    git(repo, "merge", "--abort")
    subprocess.run([sys.executable, "tools/verity/lean/merge.py", "--install"], cwd=repo, check=True, capture_output=True)
    git(repo, "merge", "--no-edit", "b")
    assert g.read_text() == "import G.Actual\nimport G.Memory\nimport G.Joint\n"


LAKEFILE = '''# Properties: the code's properties.
name = "security_properties"

[[require]]
name = "security"
path = ".."

[[require]]
name = "pouw_window"
path = "../../pouw/window"

[[lean_lib]]
name = "Properties"
roots = ["Properties.Pouw", "Proofs.Properties.Pouw",
  "Properties.Nci", "Proofs.Properties.Nci"]
'''


def _lakefile(tmp: Path, base: str, ours: str, theirs: str) -> tuple[str, int]:
    for n, t in (("base", base), ("ours", ours), ("theirs", theirs)):
        (tmp / n).write_text(t)
    return M.lakefile(tmp / "base", tmp / "ours", tmp / "theirs")


def test_a_lakefile_whose_sides_each_require_a_package_and_add_roots_merges():
    """Each side requires a package after the same block and appends roots to the same line, as in the Properties
    lakefile; the merge keeps both, wrapped as ours wraps, and stays TOML."""
    import tempfile
    import tomllib
    tmp = Path(tempfile.mkdtemp())
    def req(n):
        return f'\n[[require]]\nname = "{n}"\npath = "../../{n}"\n'
    anchor = 'path = "../../pouw/window"\n'
    tail = '"Proofs.Properties.Nci"]'
    ours = LAKEFILE.replace(anchor, anchor + req("pous_check")).replace(tail, '"Proofs.Properties.Nci", "Properties.Pous"]')
    theirs = LAKEFILE.replace(anchor, anchor + req("network_check")).replace(
        tail, '"Proofs.Properties.Nci", "Properties.Network"]')
    text, left = _lakefile(tmp, LAKEFILE, ours, theirs)
    assert left == 0, text
    doc = tomllib.loads(text)
    assert [r["name"] for r in doc["require"]] == ["security", "pouw_window", "pous_check", "network_check"]
    assert doc["lean_lib"][0]["roots"] == ["Properties.Pouw", "Proofs.Properties.Pouw", "Properties.Nci",
                                           "Proofs.Properties.Nci", "Properties.Network", "Properties.Pous"]
    assert text.startswith("# Properties: the code's properties.\n") and all(len(s) <= M.WIDTH for s in text.splitlines())
    gone = LAKEFILE.replace(', "Proofs.Properties.Pouw"', "")
    text, left = _lakefile(tmp, LAKEFILE, gone.replace(tail, '"Proofs.Properties.Nci", "Properties.Pous"]'), theirs)
    assert left == 0 and tomllib.loads(text)["lean_lib"][0]["roots"] == [
        "Properties.Pouw", "Properties.Nci", "Proofs.Properties.Nci", "Properties.Network", "Properties.Pous"], text
    def comment(s):
        return LAKEFILE.replace("the code's properties.", s)
    text, left = _lakefile(tmp, LAKEFILE, comment("ours."), comment("theirs."))
    assert left == 1 and "<<<<<<< ours" in text, "a comment both sides changed stays git's"
    text, left = _lakefile(tmp, LAKEFILE, LAKEFILE.replace('"security_properties"', '"a"'),
                           LAKEFILE.replace('"security_properties"', '"b"'))
    assert left == 1, "so does a value"


PYPROJECT = '''[project]
name = "p"
version = "0.1.0"
dependencies = ["numpy>=1.26",
  "torch"]

[tool.verity.tests]
paths = ["tests"]
inputs = ["verity/__init__.py", "verity/settings.py",
          "verity/ml/circuits",
          # the H1 units read both.
          "verity/core/protocols/pouw/core_check.py", "verity/core/protocols/pouw/leaf_lanes.py",
          "verity/ml/protocols/pous/circuit.py"]
untraced = ["fixtures"]
'''


def _pyproject(tmp: Path, base: str, ours: str, theirs: str) -> tuple[str, int]:
    for n, t in (("base", base), ("ours", ours), ("theirs", theirs)):
        (tmp / n).write_text(t)
    return M.pyproject(tmp / "base", tmp / "ours", tmp / "theirs")


def test_a_pyproject_whose_sides_each_add_a_suite_input_merges_and_any_other_conflict_stays_gits(tmp_path):
    """Two PRs that each add a file a suite reads, at the same place in `inputs`, conflict line by line; they
    merge as a list.  A value both sides changed, and a list outside `[tool.verity.tests]` both sides added to, stay git's."""
    import tomllib
    tail = '"verity/ml/protocols/pous/circuit.py"]'
    def add(*xs):
        return PYPROJECT.replace(tail, tail[:-1] + "".join(f',\n          "{x}"' for x in xs) + "]")
    text, left = _pyproject(tmp_path, PYPROJECT, add("verity/a.py"), add("verity/b.py"))
    assert left == 0, text
    doc = tomllib.loads(text)
    assert doc["tool"]["verity"]["tests"]["inputs"][-3:] == ["verity/ml/protocols/pous/circuit.py", "verity/b.py",
                                                             "verity/a.py"]
    assert doc["project"] == tomllib.loads(PYPROJECT)["project"] and "# the H1 units read both." in text
    gone = PYPROJECT.replace(tail, '"verity/a.py"]')
    text, left = _pyproject(tmp_path, PYPROJECT, gone, add("verity/b.py"))
    assert left == 0, text
    assert set(tomllib.loads(text)["tool"]["verity"]["tests"]["inputs"][-3:]) == {
        "verity/core/protocols/pouw/leaf_lanes.py", "verity/a.py", "verity/b.py"}, "what ours removed stays removed"
    def version(v):
        return add(f"verity/{v}.py").replace('version = "0.1.0"', f'version = "{v}"')
    text, left = _pyproject(tmp_path, PYPROJECT, version("1"), version("2"))
    assert left and "<<<<<<< ours" in text, "a version both sides changed stays git's"
    def dep(d):
        return PYPROJECT.replace('"torch"]', f'"torch", "{d}"]')
    text, left = _pyproject(tmp_path, PYPROJECT, dep("scipy"), dep("jax"))
    assert left and "<<<<<<< ours" in text, "a dependency both sides added stays git's"


def test_git_merges_two_branches_that_each_add_a_suite_input_with_the_installed_driver(tmp_path, monkeypatch):
    import tomllib
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    (repo / "tools" / "verity" / "lean").mkdir(parents=True)
    for f in ("merge.py", "check.py", "modules.py"):
        shutil.copy(HERE.parent / f, repo / "tools" / "verity" / "lean" / f)
    (repo / ".gitattributes").write_text(f"pyproject.toml merge={M.PYPROJECT}\n")
    pp = repo / "pyproject.toml"
    tail = '"verity/ml/protocols/pous/circuit.py"]'
    git(repo, "init", "-q", "-b", "main")
    for name in ("main", "a", "b"):
        if name != "main":
            git(repo, "checkout", "-q", "-b", name, "main")
        pp.write_text(PYPROJECT if name == "main" else PYPROJECT.replace(tail, tail[:-1] + f', "verity/{name}.py"]'))
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", name)
    git(repo, "checkout", "-q", "a")
    subprocess.run([sys.executable, "tools/verity/lean/merge.py", "--install"], cwd=repo, check=True, capture_output=True)
    git(repo, "merge", "--no-edit", "b")
    assert {"verity/a.py", "verity/b.py"} <= set(tomllib.loads(pp.read_text())["tool"]["verity"]["tests"]["inputs"])


def test_git_merges_two_branches_that_each_require_a_package_with_the_installed_driver(tmp_path, monkeypatch):
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    (repo / "tools" / "verity" / "lean").mkdir(parents=True)
    for f in ("merge.py", "check.py", "modules.py"):
        shutil.copy(HERE.parent / f, repo / "tools" / "verity" / "lean" / f)
    (repo / ".gitattributes").write_text(f"lakefile.toml merge={M.LAKEFILE}\n")
    lf = repo / "lakefile.toml"
    anchor = 'path = "../../pouw/window"\n'
    git(repo, "init", "-q", "-b", "main")
    for name, extra in (("main", ""), ("a", '\n[[require]]\nname = "x"\npath = "x"\n'),
                        ("b", '\n[[require]]\nname = "y"\npath = "y"\n')):
        if name != "main":
            git(repo, "checkout", "-q", "-b", name, "main")
        lf.write_text(LAKEFILE.replace(anchor, anchor + extra))
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", name)
    git(repo, "checkout", "-q", "a")
    subprocess.run([sys.executable, "tools/verity/lean/merge.py", "--install"], cwd=repo, check=True, capture_output=True)
    git(repo, "merge", "--no-edit", "b")
    assert 'name = "x"\npath = "x"\n\n[[require]]\nname = "y"\npath = "y"\n' in lf.read_text()


def test_lists_of_names_merge_where_both_sides_append_one():
    """Each side appends a module to `meaning`, which as one value would merge as theirs' and drop ours' module; every list
    of names (`meaning`, `claims.modules`, `shape.premises`...) merges as a list, as `roots` did."""
    base = {"meaning": ["A"], "claims": {"modules": ["C"], "witnesses": {}}, "shape": {"premises": ["P"]}}
    ours = {"meaning": ["A", "B"], "claims": {"modules": ["C", "D"], "witnesses": {}}, "shape": {"premises": ["P", "Q"]}}
    theirs = {"meaning": ["A", "C"], "claims": {"modules": ["C", "E"], "witnesses": {}}, "shape": {"premises": ["P", "R"]}}
    merged, why = M.merge(base, ours, theirs)
    assert why == []
    assert merged["meaning"] == ["A", "C", "B"] or merged["meaning"] == ["A", "B", "C"]
    assert set(merged["meaning"]) == {"A", "B", "C"} and set(merged["claims"]["modules"]) == {"C", "D", "E"}
    assert set(merged["shape"]["premises"]) == {"P", "Q", "R"}
    merged, why = M.merge(base, {**ours, "meaning": []}, theirs)
    assert why == [] and merged["meaning"] == ["C"], "what one side removed stays removed"
    merged, why = M.merge({"roots": ["X"], "x": 1}, {"roots": ["X"], "x": 2}, {"roots": ["X"], "x": 3})
    assert why == ["x: changed on both sides, differently: took theirs"] and merged["x"] == 3, \
        "a value both sides changed is theirs', named"


def test_a_witness_entry_both_sides_changed_is_ours_unless_theirs_alone_is_at_its_merged_record():
    """An `unwitnessed` hash both sides changed, which no merged record holds, and a `witnesses` entry both sides changed,
    which no record keys, merge as ours', named as candidates; a `witnesses` entry one side removed while the other changed
    it is the changed one, and an entry only one side changed is that side's, as before.  With the guarantee's records,
    theirs' hash, the merged record's, is theirs'."""
    base = {"unwitnessed": {"G": "h0", "H": "k0"},
            "witnesses": {"V": {"honest": ["V_h"], "nontrivial": "V_n"}, "W": {"honest": ["W_h"], "nontrivial": "W_n"}}}
    ours = {"unwitnessed": {"G": "h1", "H": "k1"}, "witnesses": {"V": {"honest": ["V_h1"], "nontrivial": "V_n"}}}
    theirs = {"unwitnessed": {"G": "h2", "H": "k0"},
              "witnesses": {"V": {"honest": ["V_h2"], "nontrivial": "V_n"},
                            "W": {"honest": ["W_h"], "nontrivial": "W_n", "kind": "exclusion"}}}
    candidate: list[str] = []
    merged, why = M.merge(base, ours, theirs, candidate)
    assert why == []
    assert merged["unwitnessed"] == {"G": "h1", "H": "k1"}
    assert merged["witnesses"] == {"V": ours["witnesses"]["V"], "W": theirs["witnesses"]["W"]}
    assert candidate == ["unwitnessed[G] (ours, neither side at its merged record)", "witnesses[V] (ours)",
                         "witnesses[W] (theirs)"]
    rec = {s: d | {"guarantees": {"G": {"owner": "@lean", "type_hash": h}}}
           for s, d, h in (("base", base, "h0"), ("ours", ours, "h1"), ("theirs", theirs, "h2"))}
    merged, _ = M.merge(rec["base"], rec["ours"], rec["theirs"], candidate := [])
    assert merged["unwitnessed"]["G"] == "h2" and "unwitnessed[G] (theirs, at its merged record)" in candidate


V5 = "Flock.SecurityProofs.SampledKSLiveV5"


def test_an_unwitnessed_entry_both_sides_changed_is_the_side_at_its_merged_record(tmp_path, capsys):
    """Drive's merge of a lock change onto `next`: each side re-keyed SampledKSLiveV5's `unwitnessed` entry with its
    restated record, and the merged record, theirs', holds theirs' entry's hashes, so the entry is theirs'.  Where
    theirs left the record as the base had it, the merged record is ours', and theirs' entry is a stale value that
    would re-key ours': it is ours'."""
    sides = json.loads((FIXTURES / "unwitnessed-at-the-merged-record-onto-next.json").read_text())
    for n in ("base", "ours", "theirs"):
        (tmp_path / n).write_text(A.text(sides[n]))
    assert M.main([*(str(tmp_path / n) for n in ("base", "ours", "theirs")), "Security/lean-audit.json"]) == 0
    merged = json.loads((tmp_path / "ours").read_text())
    assert merged["unwitnessed"][V5] == sides["theirs"]["unwitnessed"][V5] != sides["ours"]["unwitnessed"][V5]
    assert merged["unwitnessed"][V5]["unfolded_hash"] == merged["guarantees"][V5]["unfolded_hash"]
    assert f"unwitnessed[{V5}] (theirs, at its merged record): kept as named" in capsys.readouterr().err
    stale = {**sides["theirs"], "guarantees": sides["base"]["guarantees"]}
    merged, _ = M.merge(sides["base"], sides["ours"], stale, candidate := [])
    assert merged["guarantees"][V5] == sides["ours"]["guarantees"][V5]
    assert merged["unwitnessed"][V5] == sides["ours"]["unwitnessed"][V5]
    assert f"unwitnessed[{V5}] (ours, at its merged record)" in candidate


def test_an_unwitnessed_entry_neither_side_has_at_its_merged_record_is_ours_named(tmp_path, capsys):
    """A merge whose incoming side's SampledKSLiveV5 entry is a value theirs' own record no longer holds. Neither
    side's entry holds the merged record's hashes, so the entry is ours', named on stderr: where the merged tree
    doesn't state it, the Lean check fails it as a witness failure."""
    sides = json.loads((FIXTURES / "unwitnessed-stale-on-the-incoming-side.json").read_text())
    assert sides["theirs"]["unwitnessed"][V5]["unfolded_hash"] != sides["theirs"]["guarantees"][V5]["unfolded_hash"]
    for n in ("base", "ours", "theirs"):
        (tmp_path / n).write_text(A.text(sides[n]))
    assert M.main([*(str(tmp_path / n) for n in ("base", "ours", "theirs")), "Security/lean-audit.json"]) == 0
    assert json.loads((tmp_path / "ours").read_text())["unwitnessed"][V5] == sides["ours"]["unwitnessed"][V5]
    assert (f"merge.py: Security/lean-audit.json: unwitnessed[{V5}] (ours, neither side at its merged record): kept as "
            "named; the Lean check of the merged tree judges it, and a stale entry fails there as a witness failure\n"
            in capsys.readouterr().err)


SEC, PROPS, FLOCK, POUS = "Security", "Security/Properties", "verity/core/service/flock", "verity/core/protocols/pous"


def packages(unwitnessed: dict[str, int], inline: dict[str, int], doc: str = "never grows",
             above: str = "the size of each lock's `unwitnessed`") -> str:
    """`tests/test_lean_packages.py` as far as its ratchet caps go: each cap's line as the file writes it, between the
    comment above it and its test."""
    return (f"import json\n\n\n#: {above}\nUNWITNESSED = {json.dumps(unwitnessed)}\n\n\n"
            f'def test_the_unwitnessed_list_only_shrinks():\n    """A lock\'s `unwitnessed` {doc}."""\n\n\n'
            f"#: the size of each lock's `shape.inline`\nINLINE = {json.dumps(inline)}\n\n\n"
            f'def test_the_inline_list_only_shrinks():\n    """A lock\'s `shape.inline` never grows."""\n')


def _caps(tmp: Path, base: str, ours: str, theirs: str) -> tuple[str, int]:
    for n, t in (("base", base), ("ours", ours), ("theirs", theirs)):
        (tmp / n).write_text(t)
    return M.caps(tmp / "base", tmp / "ours", tmp / "theirs")


def test_two_branches_that_each_move_the_ratchet_caps_merge_them_as_the_larger(tmp_path, capsys):
    """Every PR that witnesses a guarantee, and drive's lock run, edit a lock's count on one line, so any two conflict as
    text. Each key merges as the larger of the two sides' caps, on each cap line, with theirs' other edits kept."""
    base = packages({SEC: 354, PROPS: 184, FLOCK: 8}, {SEC: 1422, PROPS: 129, FLOCK: 6})
    ours = packages({SEC: 355, PROPS: 183, FLOCK: 7}, {SEC: 1423, PROPS: 129, FLOCK: 6})
    theirs = packages({SEC: 354, PROPS: 185, FLOCK: 8}, {SEC: 1422, PROPS: 128, FLOCK: 6}, doc="only shrinks")
    want = packages({SEC: 355, PROPS: 185, FLOCK: 8}, {SEC: 1423, PROPS: 129, FLOCK: 6}, doc="only shrinks")
    for n, t in (("base", base), ("ours", ours), ("theirs", theirs)):
        (tmp_path / n).write_text(t)
    plain = subprocess.run(["git", "merge-file", "-p", *(str(tmp_path / n) for n in ("ours", "base", "theirs"))],
                           capture_output=True, text=True)
    assert plain.returncode == 2, "git alone conflicts on both cap lines"
    assert _caps(tmp_path, base, ours, theirs) == (want, 0)
    assert M.main(["--caps", *(str(tmp_path / n) for n in ("base", "ours", "theirs")), "tests/test_lean_packages.py"]) == 0
    assert (tmp_path / "ours").read_text() == want and capsys.readouterr().err == ""


def test_a_cap_key_one_side_adds_or_lacks_counts_from_the_base(tmp_path):
    """A key only one side has counts as the base's count on the other side: a package one side gives a cap keeps that
    count, placed after the key the side has before it; one a side dropped is the larger of the base's and the other
    side's; and one both sides dropped is gone."""
    inline = {SEC: 1422}
    text, left = _caps(tmp_path, packages({SEC: 354, PROPS: 184}, inline), packages({SEC: 355, PROPS: 184, FLOCK: 7}, inline),
                       packages({SEC: 354, PROPS: 185, POUS: 2}, inline))
    assert (text, left) == (packages({SEC: 355, PROPS: 185, POUS: 2, FLOCK: 7}, inline), 0)
    text, left = _caps(tmp_path, packages({SEC: 354, FLOCK: 8}, inline), packages({SEC: 355}, inline),
                       packages({SEC: 354, FLOCK: 7}, inline))
    assert (text, left) == (packages({SEC: 355, FLOCK: 8}, inline), 0)
    text, left = _caps(tmp_path, packages({SEC: 354, FLOCK: 8}, inline), packages({SEC: 355}, inline),
                       packages({SEC: 353}, inline))
    assert (text, left) == (packages({SEC: 355}, inline), 0)


def test_a_cap_whose_comment_both_sides_reword_merges_as_the_larger_under_theirs_comment(tmp_path):
    """A PR that moves a cap rewords the comment above it, so the count and the comment sit in one hunk: the count merges
    as the larger all the same, a comment both sides reworded differently is theirs', and one both reworded alike is that."""
    inline = {SEC: 1422}
    base = packages({SEC: 354, PROPS: 184}, inline)
    text, left = _caps(tmp_path, base, packages({SEC: 355, PROPS: 184}, inline, above="ours"),
                       packages({SEC: 354, PROPS: 185}, inline, above="theirs"))
    assert (text, left) == (packages({SEC: 355, PROPS: 185}, inline, above="theirs"), 0)
    text, left = _caps(tmp_path, base, packages({SEC: 381, PROPS: 188}, inline, above="both"),
                       packages({SEC: 384, PROPS: 186}, inline, above="both"))
    assert (text, left) == (packages({SEC: 384, PROPS: 188}, inline, above="both"), 0)
    alike = packages({SEC: 381, PROPS: 188}, inline, above="both")
    assert _caps(tmp_path, base, alike, alike) == (alike, 0), "a line both sides changed alike is that line, not twice"


def test_a_cap_one_side_brings_is_that_side_s_beside_the_caps_that_merge_as_the_larger(tmp_path):
    """A branch that brings a new cap (a new check's count) and a branch that moves an old cap and rewords its comment
    merge as when both hold the same caps: the new cap is its side's line, whichever side brings it, and the old one
    merges as the larger under theirs' comment."""
    inline = {SEC: 1422}
    base = packages({SEC: 354, PROPS: 184}, inline)
    new = '\n\n#: the spec\'s theorems\nSPEC_THEOREMS = {"Security": 154}\n'
    ours, theirs = packages({SEC: 355, PROPS: 184}, inline, above="ours"), packages({SEC: 380, PROPS: 185}, inline, above="theirs")
    assert _caps(tmp_path, base, ours, theirs + new) == (packages({SEC: 380, PROPS: 185}, inline, above="theirs") + new, 0)
    assert _caps(tmp_path, base, ours + new, theirs) == (packages({SEC: 380, PROPS: 185}, inline, above="theirs") + new, 0)
    text, left = _caps(tmp_path, base + new, ours, theirs + new)
    assert left == 1 and "<<<<<<< ours\n" in text, "a cap the base holds and a side dropped stays git's"


def test_caps_both_sides_brought_over_a_base_without_them_merge_as_the_larger_count(tmp_path):
    """A merge base from before the caps came (`main` while the PR bringing them is still on `next`) holds no cap line,
    so both sides added each one: each key merges as the larger count, not the sum, under theirs' comment."""
    base = "import json\n\n\ndef test_one_mathlib():\n    pass\n"
    ours = base + "\n\n" + packages({SEC: 535, PROPS: 205}, {SEC: 1688}, above="ours")
    theirs = base + "\n\n" + packages({SEC: 509, PROPS: 207}, {SEC: 1700}, above="theirs")
    text, left = _caps(tmp_path, base, ours, theirs)
    assert (text, left) == (base + "\n\n" + packages({SEC: 535, PROPS: 207}, {SEC: 1700}, above="theirs"), 0)


def test_a_cap_two_sides_each_lowered_from_one_base_merges_as_the_larger_not_below_its_count(tmp_path, capsys):
    """`checked` fa8d2365: drive's lock run lowered `UNWITNESSED["Security"]` from 331 to 311, its count then, and a PR,
    from an older base at 331, lowered it by its own 24 to 307. As counters (311 - 24) the merged cap was 287 against a
    merged count of 305, since what the two lowerings dropped overlapped, and every PR on it failed `grew`. As the larger,
    311 holds the count, and the lock run lowers it to the count and its room."""
    inline = {SEC: 917, PROPS: 132}
    base, checked, pr = (packages({SEC: n, PROPS: 191}, inline) for n in (331, 311, 307))
    want = packages({SEC: 311, PROPS: 191}, inline)
    assert _caps(tmp_path, base, checked, pr) == _caps(tmp_path, base, pr, checked) == (want, 0)
    assert M.main(["--caps", *(str(tmp_path / n) for n in ("base", "ours", "theirs")), "tests/test_lean_packages.py"]) == 0
    assert (tmp_path / "ours").read_text() == want and capsys.readouterr().err == ""


def test_a_conflict_beyond_the_cap_lines_and_comments_stays_git_s(tmp_path, capsys):
    """Only the caps and the comments merge: a conflict in a docstring stays git's beside a cap that merges, and the caps
    stay git's where the sides hold different caps. Both sides lowering one cap below what counters could hold (both
    dropped entries, maybe the same ones) merges as the larger."""
    inline = {SEC: 1422}
    base = packages({SEC: 354, PROPS: 184}, inline)
    ours, theirs = packages({SEC: 355, PROPS: 184}, inline, doc="shrinks"), packages({SEC: 354, PROPS: 185}, inline, doc="drops")
    text, left = _caps(tmp_path, base, ours, theirs)
    assert left == 1 and f"UNWITNESSED = {json.dumps({SEC: 355, PROPS: 185})}\n" in text
    assert "<<<<<<< ours\n    \"\"\"A lock's `unwitnessed` shrinks.\"\"\"\n" in text and ">>>>>>> theirs\n" in text
    assert M.main(["--caps", *(str(tmp_path / n) for n in ("base", "ours", "theirs")), "tests/test_lean_packages.py"]) == 1
    assert "merge.py: tests/test_lean_packages.py: 1 conflict(s) beyond the ratchet caps" in capsys.readouterr().err
    text, left = _caps(tmp_path, base, packages({SEC: 355, PROPS: 184}, inline).replace("UNWITNESSED = ", "INLINE = "),
                       packages({SEC: 354, PROPS: 185}, inline))
    assert left == 1 and "<<<<<<< ours\n" in text and "||||||| base\n" in text, text
    assert _caps(tmp_path, packages({SEC: 2, PROPS: 5}, inline), packages({SEC: 0, PROPS: 5}, inline),
                 packages({SEC: 1, PROPS: 6}, inline)) == (packages({SEC: 1, PROPS: 6}, inline), 0)


def test_git_merges_two_branches_that_each_move_a_cap_with_the_installed_driver(tmp_path, monkeypatch):
    from research.clone import RECORD_DRIVER

    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    (repo / "tools" / "verity" / "lean").mkdir(parents=True)
    (repo / "tests").mkdir()
    for f in ("merge.py", "check.py", "modules.py"):
        shutil.copy(HERE.parent / f, repo / "tools" / "verity" / "lean" / f)
    (repo / ".gitattributes").write_text(f"tests/test_lean_packages.py merge={M.CAPS}\n")
    caps = repo / "tests" / "test_lean_packages.py"
    git(repo, "init", "-q", "-b", "main")
    inline = {SEC: 1422, PROPS: 129, FLOCK: 6}
    for name, counts in (("main", {SEC: 354, PROPS: 184, FLOCK: 8}), ("a", {SEC: 355, PROPS: 183, FLOCK: 7}),
                         ("b", {SEC: 354, PROPS: 185, FLOCK: 8})):
        if name != "main":
            git(repo, "checkout", "-q", "-b", name, "main")
        caps.write_text(packages(counts, inline))
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", name)
    git(repo, "checkout", "-q", "a")
    plain = git(repo, "merge", "--no-edit", "b", check=False)
    assert plain.returncode and "CONFLICT (content): Merge conflict in tests/test_lean_packages.py" in plain.stdout
    git(repo, "merge", "--abort")
    subprocess.run([sys.executable, "tools/verity/lean/merge.py", "--install"], cwd=repo, check=True, capture_output=True)
    driver = git(repo, "config", f"merge.{M.CAPS}.driver").stdout.strip()
    assert driver == RECORD_DRIVER[f"merge.{M.CAPS}.driver"] == M.driver("--caps ")
    git(repo, "merge", "--no-edit", "b")
    assert caps.read_text() == packages({SEC: 355, PROPS: 185, FLOCK: 8}, inline)


def test_git_merges_caps_and_comments_over_two_merge_bases_with_the_installed_driver(tmp_path, monkeypatch):
    """Two branches that each merged the other's cap change, and then moved on, have two merge bases; git merges those with
    this driver into the base it merges the branches over, so every cap and comment merges, base and all."""
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com", "PYTHONDONTWRITEBYTECODE": "1"}.items():
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    (repo / "tools" / "verity" / "lean").mkdir(parents=True)
    (repo / "tests").mkdir()
    for f in ("merge.py", "check.py", "modules.py"):
        shutil.copy(HERE.parent / f, repo / "tools" / "verity" / "lean" / f)
    (repo / ".gitattributes").write_text(f"tests/test_lean_packages.py merge={M.CAPS}\n")
    caps = repo / "tests" / "test_lean_packages.py"
    inline = {SEC: 1422}

    def commit(branch: str, start: str | None, count: int, above: str):
        if start:
            git(repo, "checkout", "-q", "-b", branch, start)
        caps.write_text(packages({SEC: count}, inline, above=above))
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", branch)

    git(repo, "init", "-q", "-b", "main")
    commit("main", None, 354, "main")
    subprocess.run([sys.executable, "tools/verity/lean/merge.py", "--install"], cwd=repo, check=True, capture_output=True)
    commit("x", "main", 355, "x")
    commit("y", "main", 356, "y")
    for branch, start, other in (("a", "x", "y"), ("b", "y", "x")):
        git(repo, "checkout", "-q", "-b", branch, start)
        git(repo, "merge", "--no-edit", other)
        assert caps.read_text() == packages({SEC: 356}, inline, above=other)
    commit("a2", "a", 358, "a2")
    commit("b2", "b", 353, "b2")
    assert len(git(repo, "merge-base", "--all", "a2", "b2").stdout.split()) == 2
    git(repo, "checkout", "-q", "a2")
    git(repo, "merge", "--no-edit", "b2")
    assert caps.read_text() == packages({SEC: 358}, inline, above="b2")
