"""`tools/verity/lean/merge.py`: two branches' `lean-audit.json` merged as `check.py --update` would write the merged tree's record,
and refused where a record can't say what that is; as a function, and as git's merge driver."""
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
    assert merged["reads"]["P.Defs"]["guarantees"] == ["P.t", "P.u", "P.v"] and list(merged["reads"]["P.Defs"]["definitions"]) == [
        "P.f", "P.g", "P.h"]
    assert text_conflicts(base, ours, theirs, tmp_path)
    assert M.merge(base, ours, ours) == (ours, [])


def test_a_hand_written_section_one_side_extends_is_kept():
    base = record(t=thm([F]))
    ours = base | {"meaning": ["P", "P.More"]}
    theirs = record(t=thm([F]), v=thm([H], "P.h n = n", "00000000000000cc"))
    merged, why = M.merge(base, ours, theirs)
    assert why == [] and merged["meaning"] == ["P", "P.More"] and "P.v" in merged["guarantees"]
    both, why = M.merge(base, ours, base | {"meaning": ["P", "P.Other"]})
    assert why == [] and set(both["meaning"]) == {"P", "P.More", "P.Other"}, "two sides that each extend it merge (#2600)"


def test_a_derived_record_both_sides_changed_merges_as_ours_and_the_merged_build_regenerates_it():
    """A guarantee restated on both sides, a definition's hash and a new guarantee's record each written two ways: what the
    build determines, so ours' value is the candidate, and the merged tree's build (a third statement) regenerates it."""
    base = record(t=thm([F]))
    u, v = thm([F, G], "P.g n = n", "00000000000000bb"), thm([H], "P.h n = n", "00000000000000cc")
    ours = record(t=thm([F], "P.f n ≤ n", "00000000000000dd"), u=u)
    theirs = record(t=thm([F], "P.f n < n + 1", "00000000000000ee"), v=v)
    merged, why = M.merge(base, ours, theirs, candidate := [])
    assert why == [] and candidate == ["guarantees[P.t] (ours)"]
    assert merged["guarantees"]["P.t"] == ours["guarantees"]["P.t"]
    tree = built(t=thm([F], "P.f n ≤ n + 1", "00000000000000ab"), u=u, v=v)
    fails = A.judge(merged, tree, FILES)[0]
    assert {f.code for f in fails} == {"guarantees.signature"} and all(map(A.RECORD_DIFFERENCE.match, fails)), fails
    regen = A.regenerated(merged, tree, FILES, fails)
    assert A.judge(regen, tree, FILES)[0] == [] and regen["guarantees"]["P.t"]["signature"] == ["(n : Nat) : P.f n ≤ n + 1"]

    f9, f8 = read("P.f", "P.Defs", "9" * 16), read("P.f", "P.Defs", "8" * 16)
    ours, theirs = record(t=thm([f9]), u=u), record(t=thm([f8]), v=v)
    merged, why = M.merge(base, ours, theirs, candidate := [])
    assert why == [] and candidate == ["reads[P.Defs].definitions[P.f] (ours)"]
    assert merged["reads"]["P.Defs"]["definitions"]["P.f"] == ours["reads"]["P.Defs"]["definitions"]["P.f"]
    tree = built(t=thm([read("P.f", "P.Defs", "7" * 16)]), u=u, v=v)
    fails = A.judge(merged, tree, FILES)[0]
    assert fails and all(map(A.RECORD_DIFFERENCE.match, fails)), fails
    assert A.judge(A.regenerated(merged, tree, FILES, fails), tree, FILES)[0] == []

    w = thm([F], "P.f n = 0 + n", "00000000000000ff")
    merged, why = M.merge(base, record(t=thm([F]), u=u), record(t=thm([F]), u=w), candidate := [])
    assert why == [] and candidate == ["guarantees[P.u] (ours)"] and merged["guarantees"]["P.u"]["type_hash"] == sha(
        "00000000000000bb")


def test_what_only_a_person_writes_still_refuses_beside_derived_candidates():
    """A dependency bumped two ways, an owner set two ways and a `frozen` entry written two ways refuse, whatever derived
    record merges as a candidate beside them."""
    base = record(t=thm([F]))
    restated = lambda s, h: record(t=thm([F], s, h))
    ours, theirs = restated("P.f n ≤ n", "00000000000000dd"), restated("P.f n < n + 1", "00000000000000ee")
    deps = lambda d: {"dependencies": {"mathlib": d * 64, "toolchain": "c" * 64}}
    assert M.merge(base, ours | deps("b"), theirs | deps("d"))[1] == [
        "dependencies[mathlib]: changed on both sides, differently"]
    owner = lambda rec, who: rec | {"guarantees": {"P.t": rec["guarantees"]["P.t"] | {"owner": who}}}
    assert M.merge(base, owner(ours, "@proofs"), owner(theirs, "@circuits"))[1] == [
        "guarantees[P.t].owner: changed on both sides, differently"]
    frozen = lambda rec, why: rec | {"shape": {"frozen": {"P.t": why}}}
    assert M.merge(base, frozen(ours, "lemma"), frozen(theirs, "restatable"))[1] == [
        "shape[frozen][P.t]: added on both sides, differently"]
    hyp = lambda rec, who: rec | {"assumptions": {"P.Hyp": {"owner": who, "digest": "a" * 64}}}
    assert M.merge(hyp(base, "@lean"), hyp(ours, "@proofs"), hyp(theirs, "@circuits"))[1] == [
        "assumptions[P.Hyp].owner: changed on both sides, differently"]


def test_an_owner_merges_apart_from_the_statement():
    base = record(t=thm([F]))
    unowned = {**base, "guarantees": {"P.t": A.unowned(base["guarantees"]["P.t"])}}
    restated = record(t=thm([F], "P.f n ≤ n", "00000000000000dd"))
    merged, why = M.merge(unowned, base, {**restated, "guarantees": {"P.t": A.unowned(restated["guarantees"]["P.t"])}})
    assert why == [] and merged["guarantees"] == restated["guarantees"], "one side's owner, the other's restatement"
    other = {**base, "guarantees": {"P.t": base["guarantees"]["P.t"] | {"owner": "@proofs"}}}
    assert M.merge(unowned, base, other)[1] == ["guarantees[P.t].owner: added on both sides, differently"]


def test_a_definition_one_side_stops_reading_merges_as_the_union_when_the_other_side_changed_its_module():
    """A reads-only clash (lean's One-vocabulary PRs move definitions out of a module that queued PRs' new guarantees read):
    the record doesn't say which guarantees read the definition, so it stays, and the train's Lean check regenerates."""
    base = record(t=thm([F, G]))
    ours = record(t=thm([F], "P.f n + 0 = n", "00000000000000dd"))
    theirs = record(t=thm([F, G]), v=thm([H], "P.h n = n", "00000000000000cc"))
    candidate: list[str] = []
    merged, why = M.merge(base, ours, theirs, candidate)
    assert why == [] and candidate == ["reads[P.Defs].definitions[P.g]"]
    assert merged["guarantees"] == record(t=thm([F], "P.f n + 0 = n", "00000000000000dd"),
                                          v=thm([H], "P.h n = n", "00000000000000cc"))["guarantees"]
    group = merged["reads"]["P.Defs"]
    assert group["definitions"] == theirs["reads"]["P.Defs"]["definitions"] and list(group["definitions"]) == ["P.f", "P.g", "P.h"]
    assert group["guarantees"] == ["P.t", "P.v"] and group["digest"] == A.module_digest(group["definitions"])
    other = record(t=thm([F, read("P.g", "P.Defs", "7" * 16)]), v=thm([H], "P.h n = n", "00000000000000cc"))
    merged, why = M.merge(base, ours, other, candidate := [])
    assert why == [] and candidate == ["reads[P.Defs].definitions[P.g]"]
    assert merged["reads"]["P.Defs"]["definitions"] == other["reads"]["P.Defs"]["definitions"], \
        "with the other side's hash when it changed it"
    assert M.merge(base, ours, record(t=thm([F, G]), v=thm([K], "P.k n = n", "00000000000000cc")))[1] == []


def test_a_module_one_side_stops_reading_merges_as_the_union_and_one_left_unread_is_dropped():
    base = record(t=thm([F, K]))
    ours = record(t=thm([F], "P.f n + 0 = n", "00000000000000dd"))
    m = read("P.m", "P.More", "6" * 16)
    theirs = record(t=thm([F, K]), w=thm([m], "P.m n = n", "00000000000000cc"))
    candidate: list[str] = []
    merged, why = M.merge(base, ours, theirs, candidate)
    assert why == [] and candidate == ["reads[P.More]"]
    assert merged["reads"]["P.More"]["definitions"] == theirs["reads"]["P.More"]["definitions"]
    assert list(merged["reads"]["P.More"]["definitions"]) == ["P.k", "P.m"]
    assert merged["reads"]["P.More"]["guarantees"] == ["P.w"], "t stopped reading it on ours"
    rehashed = read("P.k", "P.More", "5" * 16)
    merged, why = M.merge(base, ours, record(t=thm([F, rehashed])), candidate := [])
    assert why == [] and "P.More" not in merged["reads"] and candidate == ["reads[P.More]"] * 2, \
        "no guarantee of the merged record reads it: dropped"


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
    assert why == [] and candidate == ["reads[P.More] (definitions moved from P.Defs)"], \
        "ours' new reader of P.Defs may read P.f where it moved"
    assert A.text(merged) == A.text(record(t=thm([moved_f, G]), s=thm([G], "P.g n = n", "00000000000000bb"), u=u))


def test_an_assumption_s_digest_changed_two_ways_or_removed_against_changed_merges_and_regenerates():
    """LEAN-60 (#4656) records a digest for every premise, so two PRs that each change a premise's definition clash on it:
    ours' digest is the candidate, and an entry one side dropped while the other changed it stays until the merged build
    says no guarantee reads it (`assumptions.unread`)."""
    base = record(t=thm([F, G]))
    drop_g = record(t=thm([F]), u=thm([F], "P.f n = 0 + n", "00000000000000bb"))
    keep_g = record(t=thm([F, G]), v=thm([H], "P.h n = n", "00000000000000cc"))
    hyp = lambda rec, d: rec | {"assumptions": {"P.Hyp": {"owner": "@lean", "digest": d * 64}}}
    merged, why = M.merge(hyp(base, "a"), hyp(drop_g, "b"), hyp(keep_g, "c"), candidate := [])
    assert why == [] and "assumptions[P.Hyp].digest (ours)" in candidate
    assert merged["assumptions"] == {"P.Hyp": {"owner": "@lean", "digest": "b" * 64}}
    merged, why = M.merge(hyp(base, "a"), drop_g, hyp(keep_g, "c"), candidate := [])
    assert why == [] and "assumptions[P.Hyp]" in candidate and merged["assumptions"]["P.Hyp"]["digest"] == "c" * 64
    tree = built(t=thm([F]), u=thm([F], "P.f n = 0 + n", "00000000000000bb"), v=thm([H], "P.h n = n", "00000000000000cc"))
    fails = A.judge(merged, tree, FILES)[0]
    assert "assumptions.unread" in {f.code for f in fails} and all(map(A.RECORD_DIFFERENCE.match, fails)), fails
    regen = A.regenerated(merged, tree, FILES, fails)
    assert A.judge(regen, tree, FILES)[0] == [] and "P.Hyp" not in regen.get("assumptions", {})


def test_a_guarantee_one_side_removed_stays_removed_unless_the_other_side_changed_its_owner():
    base = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"))
    ours = record(t=thm([F]))
    theirs = record(t=thm([F]), u=thm([F, G], "P.g n = n + 0", "00000000000000dd"))
    merged, why = M.merge(base, ours, theirs, candidate := [])
    assert why == [] and candidate == ["guarantees[P.u] (removed)"] and merged["guarantees"] == ours["guarantees"]
    owner = theirs | {"guarantees": theirs["guarantees"] | {"P.u": theirs["guarantees"]["P.u"] | {"owner": "@proofs"}}}
    assert M.merge(base, ours, owner)[1] == ["guarantees[P.u]: removed on one side and changed on the other"]


FIXTURES = HERE / "fixtures" / "merge"


def test_4406_onto_next_merges_its_reads_as_a_candidate(tmp_path, capsys):
    """#4406 (cursor/lean-vocab-hash-fc8d at 172198e129) onto `next` at aa5d70a9f1, base main's b6b9338dcb (26 Oct): #4406
    takes `Hash.Digest` out of `Definitions.Hash.Basic` while `next` adds readers there, and the driver refused the train.
    The fixture is that group on each side with the records of its readers (a sample of #4406's new ones).  #4406's readers
    of the module are the base's, unrestated, and its record reads `Hash.Digest` nowhere, so the removal applies: `main`'s
    record since reads it nowhere either."""
    sides = json.loads((FIXTURES / "hash-basic-4406-onto-next.json").read_text())
    for n in ("base", "ours", "theirs"):
        (tmp_path / n).write_text(A.text(sides[n]))
    assert M.main([*(str(tmp_path / n) for n in ("base", "ours", "theirs")), "Security/lean-audit.json"]) == 0
    err = capsys.readouterr().err.strip().splitlines()
    assert err == ["merge.py: Security/lean-audit.json: merged derived records as a candidate (the union, or ours' value where "
                   "marked), which the train's Lean check regenerates: reads[Definitions.Hash.Basic].definitions[Hash.Digest] "
                   "(removed); reads[Definitions.Hash.Basic].definitions[Hash.Digest.bytes] (removed)"]
    group = json.loads((tmp_path / "ours").read_text())["reads"]["Definitions.Hash.Basic"]
    theirs = sides["theirs"]["reads"]["Definitions.Hash.Basic"]["definitions"]
    assert "Hash.Digest" not in theirs and group["definitions"] == theirs
    assert set(group["guarantees"]) == set(sides["ours"]["reads"]["Definitions.Hash.Basic"]["guarantees"]) | set(
        sides["theirs"]["reads"]["Definitions.Hash.Basic"]["guarantees"])


def test_5977_onto_next_leaves_no_read_of_the_name_it_renamed(tmp_path, capsys):
    """#5977 (fcfc2901b9, `J10R.Layout` renamed `NoiseLayout`) onto `next` at e50948ba0c, base 5f41e99513, as drive merged it
    (9a58b81858, 28 Oct): `next` added readers of `Definitions.Accounting.TS.J10R`, and the union kept `Layout` and its four
    fields beside the new names, so `next`'s after-landing Lean check failed on the record (CI-270; #6044's
    `PouwCheck.Window`, #5926's `RemoteAudit.Schedule.Block` and #5933's `Compose.PeriodRun` alike).  The fixture is that group
    on each side with the records of its readers."""
    sides = json.loads((FIXTURES / "j10r-5977-onto-next.json").read_text())
    m = "Definitions.Accounting.TS.J10R"
    old = [d for d in sides["base"]["reads"][m]["definitions"] if ".Layout" in d]
    assert len(old) == 5 and all(d in sides["ours"]["reads"][m]["definitions"] for d in old)
    for n in ("base", "ours", "theirs"):
        (tmp_path / n).write_text(A.text(sides[n]))
    assert M.main([*(str(tmp_path / n) for n in ("base", "ours", "theirs")), "Security/Properties/lean-audit.json"]) == 0
    assert capsys.readouterr().err.count("(removed)") == 5
    group = json.loads((tmp_path / "ours").read_text())["reads"][m]
    assert group["definitions"] == sides["theirs"]["reads"][m]["definitions"]
    assert not set(old) & set(group["definitions"]) and f"{m.removeprefix('Definitions.')}.NoiseLayout" in group["definitions"]
    assert group["guarantees"] == sides["ours"]["reads"][m]["guarantees"]


def test_a_definition_one_side_deleted_with_its_guarantee_stays_deleted_and_one_it_stopped_reading_stays():
    """A side that deletes a guarantee and the definition only it read (next's `NetTiming.Guarantees.EncardAccSeqsLeBlocks`),
    while the other adds a reader of the module: the union brought the definition back.  A side that restated a reader may
    only have stopped reading a definition that still exists, so there the union stays."""
    base = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"))
    ours = record(t=thm([F]))
    theirs = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"), v=thm([H], "P.h n = n", "00000000000000cc"))
    merged, why = M.merge(base, ours, theirs, candidate := [])
    assert why == [] and candidate == ["reads[P.Defs].definitions[P.g] (removed)"]
    assert A.text(merged) == A.text(record(t=thm([F]), v=thm([H], "P.h n = n", "00000000000000cc")))
    merged, why = M.merge(base, record(t=thm([F], "P.f n + 0 = n", "00000000000000dd")), theirs, candidate := [])
    assert why == [] and candidate == ["reads[P.Defs].definitions[P.g]"] and "P.g" in merged["reads"]["P.Defs"]["definitions"]


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
    """``rec`` as a branch from before guarantees were called that writes it: `pins`, and each module's readers as `pins`."""
    out = {("pins" if k == "guarantees" else k): v for k, v in rec.items()}
    out["reads"] = {m: {("pins" if k == "guarantees" else k): v for k, v in g.items()} for m, g in rec["reads"].items()}
    return out


def test_a_side_from_before_the_rename_merges_under_today_s_names(tmp_path):
    base = record(t=thm([F]))
    ours = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"))
    theirs = record(t=thm([F]), v=thm([H, K], "P.h n = P.k n", "00000000000000cc"))
    tree = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"), v=thm([H, K], "P.h n = P.k n", "00000000000000cc"))
    for n, d in (("base", before_the_rename(base)), ("ours", ours), ("theirs", before_the_rename(theirs))):
        (tmp_path / n).write_text(A.text(d))
    assert M.main([str(tmp_path / n) for n in ("base", "ours", "theirs")]) == 0
    assert (tmp_path / "ours").read_text() == A.text(tree)
    (tmp_path / "theirs").write_text(A.text(theirs | {"pins": {"P.w": {}}}))
    assert M.main([str(tmp_path / n) for n in ("base", "ours", "theirs")]) == 1
    assert (tmp_path / "ours").read_text() == A.text(tree)


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


def test_git_merges_two_branches_that_add_guarantees_and_refuses_a_real_conflict(tmp_path, monkeypatch):
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    (repo / "tools" / "verity" / "lean").mkdir(parents=True)
    for f in ("merge.py", "check.py"):
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
    assert git(repo, "config", "merge.lean-audit.driver").stdout.strip() == "python3 tools/verity/lean/merge.py %O %A %B %P"
    git(repo, "merge", "--no-edit", "b")
    assert rec.read_text() == A.text(record(t=thm([F]), u=u, v=v))

    git(repo, "checkout", "-q", "c")
    refused = git(repo, "merge", "--no-edit", "d", check=False)
    assert refused.returncode and "merge.py: pkg/lean-audit.json: refused" in refused.stderr
    assert "dependencies[mathlib]: changed on both sides, differently" in refused.stderr
    assert "guarantees[P.t]" not in refused.stderr, "the restatement alone would merge as a candidate"
    assert git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split() == ["pkg/lean-audit.json"]
    assert rec.read_text() == A.text(branches["c"])


def test_shape_entries_and_frozen_names_merge_one_by_one():
    """Two branches that each freeze or enter a different guarantee merge; one entry set two ways is refused."""
    base = {"shape": {"premises": [], "properties": {}, "frozen": {"P.a": "RESTATE"}}}
    ours = {"shape": {"premises": [], "properties": {"P.b": {"function": "P.f", "outcome": "accept"}}, "frozen": {"P.a": "RESTATE"}}}
    theirs = {"shape": {"premises": [], "properties": {}, "frozen": {"P.c": "lemma"}}}
    merged, why = M.merge(base, ours, theirs)
    assert why == [] and merged["shape"] == {"premises": [], "properties": {"P.b": {"function": "P.f", "outcome": "accept"}},
                                             "frozen": {"P.c": "lemma"}}
    _, why = M.merge(base, {"shape": {"frozen": {"P.a": "x"}}}, {"shape": {"frozen": {"P.a": "y"}}})
    assert why == ["shape[frozen][P.a]: changed on both sides, differently"]


def test_two_branches_that_each_widen_a_layer_merge_its_imports_and_reasons():
    """#1860 against #1842 (12 Oct): each side added imports to `Definitions.Accounting` and a clause to its `why`."""
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
    _, why = M.merge(base, {"layers": {"D": {**ours["layers"]["D"], "mode": "x"}}}, {"layers": {"D": {**ours["layers"]["D"], "mode": "y"}}})
    assert why == ["layers[D].mode: added on both sides, differently"]


def test_two_branches_that_each_add_a_root_merge_the_roots():
    """c3-spec-2 against c3-n11-readcol (13 Oct): each side added a test module to Flock's roots."""
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
    """#1860 against #1850 (12 Oct): each side added a module to Guarantees.lean's import list at the same place."""
    base = "import G.Bandwidth\nimport G.Joint\n\ntheorem t : True := trivial\n"
    ours = "import G.Memory\nimport G.Bandwidth\nimport G.Joint\n\ntheorem t : True := trivial\n"
    theirs = "import G.Actual\nimport G.Bandwidth\nimport G.Joint\n\ntheorem t : True := by trivial\n"
    assert _lean(tmp_path, base, ours, theirs) == (
        "import G.Actual\nimport G.Memory\nimport G.Bandwidth\nimport G.Joint\n\ntheorem t : True := by trivial\n", 0)
    text, left = _lean(tmp_path, base, ours.replace(":= trivial", ":= rfl"), theirs)
    assert left == 1 and "<<<<<<< ours" in text and "import G.Actual\nimport G.Memory\n" in text


def test_an_import_one_side_removed_stays_removed_where_the_other_side_moved_it(tmp_path):
    """Ours removes `G.B` and adds `G.F`; theirs moves `G.B` to the end.  Git sees the removal on both sides where `G.B` was
    and two additions at the end, so a merge of that hunk alone brings `G.B` back (#4690 onto `next`)."""
    imports = lambda *ms: "".join(f"import G.{m}\n" for m in ms) + "\ntheorem t : True := trivial\n"
    text, left = _lean(tmp_path, imports("A", "B", "C", "D", "E"), imports("A", "C", "D", "E", "F"), imports("A", "C", "D", "E", "B"))
    assert left == 0 and text == imports("A", "C", "D", "E", "F")


def test_imports_both_sides_add_merge_and_one_both_sides_remove_stays_removed(tmp_path):
    imports = lambda *ms: "".join(f"import G.{m}\n" for m in ms) + "\ntheorem t : True := trivial\n"
    text, left = _lean(tmp_path, imports("A", "B", "C"), imports("A", "X", "C"), imports("A", "Y", "C"))
    assert left == 0 and text == imports("A", "Y", "X", "C"), "both added: the union; both removed B: gone"
    text, left = _lean(tmp_path, imports("A", "B", "C"), imports("A", "X", "B", "C"), imports("A", "Y", "B", "C"))
    assert left == 0 and text == imports("A", "Y", "X", "B", "C")


def test_4690_onto_next_drops_the_import_4690_retired(tmp_path):
    """`Security/Definitions/Accounting.lean` as git gave it to the driver when drive merged #4690 onto `next` (5534d5ccd4,
    26 Oct): the merge kept the import of the module #4690 retired, and Security stopped building (ci's check 83067ace)."""
    sides = json.loads((FIXTURES / "accounting-4690-onto-next.json").read_text())
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
    roots = lambda *ms: 'name = "p"\n\n[[lean_lib]]\nname = "P"\nroots = [' + ",\n  ".join(f'"{m}"' for m in ms) + "]\n"
    text, left = _lakefile(tmp, roots("A", "B", "C", "D", "E"), roots("A", "C", "D", "E", "F"), roots("A", "C", "D", "E", "B"))
    assert left == 0 and tomllib.loads(text)["lean_lib"][0]["roots"] == ["A", "C", "D", "E", "F"], text
    req = lambda *ns: 'name = "p"\n' + "".join(f'\n[[require]]\nname = "{n}"\npath = "../{n}"\n' for n in ns)
    text, left = _lakefile(tmp, req("a", "b", "c"), req("a", "c", "d"), req("a", "c", "b"))
    assert left == 0 and [r["name"] for r in tomllib.loads(text)["require"]] == ["a", "c", "d"], text
    text, left = _lakefile(tmp, req("a", "c"), req("a", "c", "x"), req("a", "c", "y"))
    assert left == 0 and [r["name"] for r in tomllib.loads(text)["require"]] == ["a", "c", "x", "y"], "both added: both"


def test_a_root_or_require_one_side_removed_stays_removed_where_git_merges_the_other_side_s_move_cleanly(tmp_path):
    """ci's case against ef47f3bd (#4814): roots A..H, ours removes B, theirs moves B after D.  Git merges that without a
    conflict, to [A, C, D, B, E, F, G, H]; the whole merged file is held to `kept`, so B stays removed.  Requires alike."""
    import tomllib
    roots = lambda *ms: 'name = "p"\n\n[[lean_lib]]\nname = "P"\nroots = [' + ",\n  ".join(f'"{m}"' for m in ms) + "]\n"
    req = lambda *ns: 'name = "p"\n' + "".join(f'\n[[require]]\nname = "{n}"\npath = "../{n}"\n' for n in ns) + \
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
    for f in ("merge.py", "check.py"):
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
    """#1541 against #1573 and the Properties lakefile's other conflicts (Oct): each side required a package after the same
    block and appended roots to the same line; the merge keeps both, wrapped as ours wraps, and stays TOML."""
    import tempfile
    import tomllib
    tmp = Path(tempfile.mkdtemp())
    req = lambda n: f'\n[[require]]\nname = "{n}"\npath = "../../{n}"\n'
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
    comment = lambda s: LAKEFILE.replace("the code's properties.", s)
    text, left = _lakefile(tmp, LAKEFILE, comment("ours."), comment("theirs."))
    assert left == 1 and "<<<<<<< ours" in text, "a comment both sides changed stays git's"
    text, left = _lakefile(tmp, LAKEFILE, LAKEFILE.replace('"security_properties"', '"a"'),
                           LAKEFILE.replace('"security_properties"', '"b"'))
    assert left == 1, "so does a value"


def test_git_merges_two_branches_that_each_require_a_package_with_the_installed_driver(tmp_path, monkeypatch):
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    (repo / "tools" / "verity" / "lean").mkdir(parents=True)
    for f in ("merge.py", "check.py"):
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
    """#2600 against next (18 Oct, proofs): each side appended a module to `meaning`, which merged as one value and refused
    the merge; every list of names (`meaning`, `claims.modules`, `shape.premises`...) merges as a list, as `roots` did."""
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
    _, why = M.merge({"roots": ["X"], "x": 1}, {"roots": ["X"], "x": 2}, {"roots": ["X"], "x": 3})
    assert why and "x" in why[0], "a value both sides changed still refuses"
