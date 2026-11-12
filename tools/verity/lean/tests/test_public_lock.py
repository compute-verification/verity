"""`tools/verity/lean/public_lock.py`: the lock the export ships states only what the copied files hold. An entry named
after what they lack is gone, an entry goes with a witness they lack, a kept entry citing what they lack refuses the lock,
and no `owner`, `issue` or `due` is left. The lock here is synthetic, and so is what the copied files hold."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("lean_public_lock_under_test",
                                              Path(__file__).resolve().parent.parent / "public_lock.py")
PL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PL)

#: what the copied files hold, by kind: declarations and namespaces under `Lib`, modules under `Mod`. What the export
#: leaves out is under `Lib.Gone`, the modules under `Gone`, and the package `../Gone`. `Mod` and `Mod.Proofs` are
#: prefixes with copied modules under them and no copied file of their own.
HELD = {
    PL.DECLARATION: {"Lib.G", "Lib.G_honest", "Lib.G_nontrivial", "Lib.H", "Lib.A", "Lib.D", "Lib.O", "Lib.Claim",
                     "Lib.Claim_fails", "Lib.Claim_meets", "Lib.Entry", "Lib.Stop", "Lib.Event"},
    PL.CONSTANT: {"Lib.escape", "Lib.escape.go", "Lib.Shape.mk"},
    PL.NAMESPACE: {"Lib", "Lib.Sha"},
    PL.MODULE: {"Mod.Defs", "Mod.Run"},
    PL.PREFIX: {"Mod", "Mod.Defs", "Mod.Proofs", "Mod.Run"},
    PL.IMPORT: {"Mod", "Mod.Defs", "Mathlib"},
    PL.PACKAGE: {"../Proofs"},
    PL.ARGUMENT: {"python3", "scripts/vectors.py", "{repo}", "--out", "{out}"},
}


def has(kind: str, name: str) -> bool:
    return name in HELD[kind]


def record(**more) -> dict:
    return {"owner": "@lane", "signature": ["(x : Nat)"], "assumptions": [],
            "premises": {"assumptions": [], "domain": [], "open": [], "reads": []}, "type_hash": "00"} | more


def lock() -> dict:
    """A lock with an entry the copied files hold and one they lack under every rule's path."""
    return {
        "format": 3,
        "roots": ["Mod", "Gone"],
        "proved_in": ["../Proofs", "../Gone"],
        "exempt": {"Mod.Run": "a runner", "Gone.Run": "a runner"},
        "runs": {"Mod.Run": {"generate": ["python3", "scripts/vectors.py", "{repo}", "--out", "{out}"], "why": "vectors"},
                 "Gone.Run": {"why": "vectors"}, "Mod.Proofs": {"why": "vectors"}},
        "axioms": ["propext"],
        "assumptions_module": "Mod.Defs",
        "assumptions": {"Lib.A": {"owner": "@lane", "digest": "01"}, "Lib.Gone.A": {"owner": "@lane", "digest": "02"}},
        "domain": {"Lib.D": {"owner": "@lane", "digest": "03", "why": "a range", "cites": "Lib.H"},
                   "Lib.Gone.D": {"owner": "@lane", "digest": "04", "why": "a range"}},
        "open": {"Lib.O": {"owner": "@lane", "digest": "05", "issue": "LANE-1", "due": "2026-01-01"},
                 "Lib.Gone.O": {"owner": "@lane", "digest": "06", "issue": "LANE-2", "due": "2026-01-02"}},
        "open_seed": {"Lib.O": "LANE-1"},
        "open_default_issue": "LANE-0",
        "default_owner": "@lane",
        "trust": {"base": {"Lib.A": "the host is honest", "Lib.Gone.A": "the host is honest"},
                  "extra": {"Lib.G": {"Lib.O": "a rule"}, "Lib.Gone.G": {"Lib.Gone.O": "a rule"}}},
        "deployed": ["Lib.G", "Lib.Gone.G"],
        "trust_gap": {"Lib.G": ["Lib.O"], "Lib.Gone.G": ["Lib.Gone.O"]},
        "meaning": {"Mod.Defs": "what it means", "Gone.Defs": "what it means"},
        "escapes": {"Lib.escape": "a partial def", "Lib.escape.go": "its loop", "Lib.Gone.escape": "a partial def"},
        "verifier": {"entries": ["Lib.Entry"], "stop": {"Lib.Stop": "a hash"}, "hash": "Lib.Sha", "allow": ["Nat.add"]},
        "claims": {"modules": ["Mod.Defs", "Gone.Defs"],
                   "witnesses": {"Lib.Claim": {"fails": "Lib.Claim_fails", "meets": ["Lib.Claim_meets"]},
                                 "Lib.Gone.Claim": {"fails": "Lib.Gone.Claim_fails", "meets": []}},
                   "not": {"Lib.A": "a premise", "Lib.Gone.A": "a premise"},
                   "generic": {"Lib.D": "Lib.G", "Lib.Gone.D": "Lib.Gone.G"}},
        "guarantee_lemmas": {"witnesses": {"Lib.G": {"fails": "Lib.Claim_fails", "meets": ["Lib.Claim_meets"], "by": "Lib.H"},
                                           "Lib.Gone.G": {"fails": "Lib.Gone.G_fails", "meets": [], "by": "Lib.Gone.H"}},
                             "not": {"Lib.H": "a digest", "Lib.Gone.H": "a digest"}},
        "witnesses": {"Lib.G": {"honest": ["Lib.G_honest"], "nontrivial": "Lib.G_nontrivial", "kind": "exclusion"},
                      "Lib.Gone.G": {"honest": ["Lib.Gone.G_honest"], "nontrivial": ["Lib.Gone.G_nontrivial"]}},
        "unwitnessed": {"Lib.H": "07", "Lib.Gone.H": {"type_hash": "08", "why": "no witness yet"}},
        "witnesses_assumed": ["Lib.G_honest", "Lib.Gone.G_honest"],
        "compile_time": ["Mod.Run", "Gone.Run", "Mod.Proofs"],
        "dependency_unchecked": ["Mathlib.Tactic"],
        "layers": {"Mod.Defs": {"imports": ["Mod.Defs", "Mathlib", "Gone.Defs"],
                                "except": ["Mod.Run", "Gone.Run", "Mod.Proofs"], "why": "the spec"},
                   "Gone.Defs": {"imports": ["Mathlib"], "why": "the spec"}},
        "reads_exempt": ["Mod.Run", "Gone.Run"],
        "trees": {"definitions": ["Mod.Defs", "Gone.Defs"], "properties": ["Mod.Defs"], "proofs": ["Mod.Proofs", "Gone.Proofs"],
                  "except": {"Mod.Run": "a runner", "Gone.Run": "a runner"}},
        "upstream": {"imports": ["Mathlib"], "watch": {"a lemma": {"about": "what Lib.Gone.G would use"}}},
        "four_names": ["Lib", "Gone"],
        "four_names_except": ["Lib.G", "Lib.Gone.G"],
        "guarantees": {"Lib.G": record(assumptions=["Lib.A"], premises={"assumptions": ["Lib.A"], "domain": ["Lib.D"],
                                                                      "open": ["Lib.O"], "reads": ["Lib.Shape.mk"]}),
                       "Lib.H": record(unfolded_hash="09"),
                       "Lib.Gone.G": record(premises={"assumptions": ["Lib.Gone.A"], "domain": ["Lib.Gone.D"],
                                                      "open": ["Lib.Gone.O"], "reads": ["Lib.Gone.Shape.mk"]}),
                       "Lib.Gone.H": record()},
        "shape": {"premises": ["Mod.Defs", "Gone.Defs"], "statements": ["Mod.Defs", "Gone.Defs"],
                  "properties": {"Lib.G": {"function": "Lib.check", "outcome": "accept", "rests_on": ["Lib.H"]},
                                 "Lib.Gone.G": {"function": "Lib.Gone.check", "outcome": "accept"}},
                  "frozen": {"Lib.H": "restatable: a bound", "Lib.Gone.H": "restatable: a bound"},
                  "conjectures": {"Lib.O": {"status": "open", "test": "t", "why": "w"},
                                  "Lib.Gone.O": {"status": "open", "test": "t", "why": "w"}},
                  "events": ["Lib.Event", "Lib.Gone.Event"],
                  "inline": {"Lib.G": {"x < 2": "a range"}, "Lib.Gone.G": {"x < 2": "a range"}}},
        "reads": {"Mod.Defs": {"digest": "0a", "definitions": {"Lib.Shape": "0b", "Lib.Shape.casesOn": "0c"}},
                  "Gone.Defs": {"digest": "0d", "definitions": {"Lib.Gone.Shape": "0e"}},
                  "Mod.Proofs": {"digest": "10", "definitions": {}}},
        "dependencies": {"Mathlib.Tactic": "0f"},
    }


def gone(node) -> bool:
    """Whether ``node`` spells, in a key or a value at any depth, a name of what the export leaves out."""
    if isinstance(node, dict):
        return any(gone(k) or gone(v) for k, v in node.items())
    if isinstance(node, list):
        return any(map(gone, node))
    return isinstance(node, str) and ("Gone" in node.split(".") or node == "../Gone")


def test_an_entry_stays_only_if_the_copied_files_hold_its_name():
    out, dropped = PL.public(lock(), has)
    assert dropped == {
        "roots": ["Gone"], "proved_in": ["../Gone"], "exempt": ["Gone.Run"], "runs": ["Gone.Run", "Mod.Proofs"],
        "assumptions": ["Lib.Gone.A"], "domain": ["Lib.Gone.D"], "open": ["Lib.Gone.O"],
        "trust.base": ["Lib.Gone.A"], "trust.extra": ["Lib.Gone.G"], "deployed": ["Lib.Gone.G"], "trust_gap": ["Lib.Gone.G"],
        "meaning": ["Gone.Defs"], "escapes": ["Lib.Gone.escape"],
        "claims.modules": ["Gone.Defs"], "claims.witnesses": ["Lib.Gone.Claim"], "claims.not": ["Lib.Gone.A"],
        "claims.generic": ["Lib.Gone.D"],
        "guarantee_lemmas.witnesses": ["Lib.Gone.G"], "guarantee_lemmas.not": ["Lib.Gone.H"],
        "witnesses": ["Lib.Gone.G"], "unwitnessed": ["Lib.Gone.H"], "witnesses_assumed": ["Lib.Gone.G_honest"],
        "compile_time": ["Gone.Run", "Mod.Proofs"], "layers": ["Gone.Defs"], "layers.*.imports": ["Gone.Defs"],
        "layers.*.except": ["Gone.Run", "Mod.Proofs"], "reads_exempt": ["Gone.Run"],
        "trees.definitions": ["Gone.Defs"], "trees.proofs": ["Gone.Proofs"], "trees.except": ["Gone.Run"],
        "four_names": ["Gone"], "four_names_except": ["Lib.Gone.G"], "guarantees": ["Lib.Gone.G", "Lib.Gone.H"],
        "shape.premises": ["Gone.Defs"], "shape.statements": ["Gone.Defs"], "shape.properties": ["Lib.Gone.G"],
        "shape.frozen": ["Lib.Gone.H"], "shape.conjectures": ["Lib.Gone.O"], "shape.events": ["Lib.Gone.Event"],
        "shape.inline": ["Lib.Gone.G"], "reads": ["Gone.Defs", "Mod.Proofs"],
    }
    kept = {k: v for k, v in out.items() if k != "upstream"}
    assert not gone(kept)
    assert PL.missing(out, has) == []
    assert PL.counts(out) == {path: 1 + (path in ("escapes", "guarantees", "layers.*.imports"))
                              for path, _ in PL.ENTRIES}


def test_every_rule_has_an_entry_in_the_synthetic_lock():
    source = lock()
    rules = (*PL.ENTRIES, *PL.WITNESSES, *PL.GENERATORS, *PL.CITES)
    stated = {path for path, *_ in PL._stated(source, rules)}
    assert stated == {path for path, _ in rules}
    assert PL.unruled(source) == []
    assert {s.partition(".")[0] for s in (*PL.OPAQUE, *PL.LANE_SECTIONS)} <= set(source)


def test_a_kept_entry_keeps_its_record_and_the_lock_its_other_sections():
    source = lock()
    out, _ = PL.public(source, has)
    assert out["guarantees"]["Lib.H"] == {k: v for k, v in source["guarantees"]["Lib.H"].items() if k != "owner"}
    assert out["reads"] == {"Mod.Defs": source["reads"]["Mod.Defs"]}
    assert out["runs"] == {"Mod.Run": source["runs"]["Mod.Run"]}
    assert out["layers"] == {"Mod.Defs": {"imports": ["Mod.Defs", "Mathlib"], "except": ["Mod.Run"], "why": "the spec"}}
    assert out["verifier"] == source["verifier"]
    for section in ("format", "axioms", "dependencies", "dependency_unchecked", "upstream"):
        assert out[section] == source[section]
    assert list(out) == [k for k in source if k not in PL.LANE_SECTIONS]


def test_the_lock_it_is_given_is_left_as_it_was():
    source = lock()
    before = copy.deepcopy(source)
    PL.public(source, has)
    assert source == before


def test_no_entry_carries_an_owner_an_issue_or_a_due_date():
    source = lock()
    owners = len(source["guarantees"]) + len(source["assumptions"]) + len(source["domain"]) + len(source["open"])
    assert len(list(PL.lane_fields(source))) == owners + 2 * len(source["open"]) + len(PL.LANE_SECTIONS)
    out, _ = PL.public(source, has)
    assert list(PL.lane_fields(out)) == []
    assert not set(PL.LANE_SECTIONS) & set(out)
    assert out["open"] == {"Lib.O": {"digest": "05"}}
    assert out["assumptions"] == {"Lib.A": {"digest": "01"}}
    assert out["domain"] == {"Lib.D": {"digest": "03", "why": "a range", "cites": "Lib.H"}}
    assert all("owner" not in r for r in out["guarantees"].values())


def test_a_lane_field_left_at_any_depth_is_named():
    out, _ = PL.public(lock(), has)
    out["guarantees"]["Lib.G"]["premises"]["owner"] = "@lane"
    out["shape"]["conjectures"]["Lib.O"]["due"] = "2026-01-01"
    out["default_owner"] = "@lane"
    assert PL.missing(out, has) == ["guarantees.Lib.G.premises.owner: a lane field",
                                    "shape.conjectures.Lib.O.due: a lane field", "default_owner: a lane field"]


def test_a_runner_goes_with_a_file_of_its_generator_the_copied_files_lack():
    """The data of a run is what its `generate` writes: where the export leaves out a file that command names, the public
    tree can't make the data, so the entry goes. Its module stays exempt, as the test it still is."""
    source = lock()
    source["runs"]["Mod.Run"]["generate"] = ["python3", "{repo}/gone/vectors.py", "{repo}", "--out", "{out}"]
    out, dropped = PL.public(source, has)
    assert dropped["runs.*.generate"] == ["Mod.Run"] and out["runs"] == {}
    assert out["exempt"] == {"Mod.Run": "a runner"}
    assert PL.missing(out, has) == []
    kept, dropped = PL.public(lock(), has)
    assert "runs.*.generate" not in dropped and kept["runs"] == {"Mod.Run": lock()["runs"]["Mod.Run"]}
    kept["runs"]["Mod.Run"]["generate"][1] = "tests/vectors.py"
    assert PL.missing(kept, has) == ["runs[Mod.Run].generate: tests/vectors.py"]


def test_an_entry_goes_with_a_witness_the_copied_files_lack():
    for path, where in (("witnesses.*.honest", ("witnesses", "Lib.G", "honest")),
                        ("witnesses.*.nontrivial", ("witnesses", "Lib.G", "nontrivial")),
                        ("claims.witnesses.*.fails", ("claims", "witnesses", "Lib.Claim", "fails")),
                        ("claims.witnesses.*.meets", ("claims", "witnesses", "Lib.Claim", "meets")),
                        ("guarantee_lemmas.witnesses.*.fails", ("guarantee_lemmas", "witnesses", "Lib.G", "fails")),
                        ("guarantee_lemmas.witnesses.*.meets", ("guarantee_lemmas", "witnesses", "Lib.G", "meets")),
                        ("guarantee_lemmas.witnesses.*.by", ("guarantee_lemmas", "witnesses", "Lib.G", "by"))):
        source = lock()
        *above, entry, field = where
        holder = source
        for k in above:
            holder = holder[k]
        holder[entry][field] = ["Lib.Gone.W"] if isinstance(holder[entry][field], list) else "Lib.Gone.W"
        out, dropped = PL.public(source, has)
        assert dropped[path] == [entry]
        section = out
        for k in above:
            section = section[k]
        assert entry not in section
        assert entry in out["guarantees"] or entry == "Lib.Claim"
        assert PL.missing(out, has) == []


@pytest.mark.parametrize("where, value, named", [
    (("domain", "Lib.D", "cites"), "Lib.Gone.H", "domain[Lib.D].cites: Lib.Gone.H"),
    (("trust", "extra", "Lib.G"), {"Lib.Gone.O": "a rule"}, "trust.extra[Lib.G]: Lib.Gone.O"),
    (("trust_gap", "Lib.G"), ["Lib.Gone.O"], "trust_gap[Lib.G]: Lib.Gone.O"),
    (("claims", "generic", "Lib.D"), "Lib.Gone.G", "claims.generic[Lib.D]: Lib.Gone.G"),
    (("guarantees", "Lib.G", "assumptions"), ["Lib.Gone.A"], "guarantees[Lib.G].assumptions: Lib.Gone.A"),
    (("guarantees", "Lib.G", "premises", "assumptions"), ["Lib.Gone.A"], "guarantees[Lib.G].premises.assumptions: Lib.Gone.A"),
    (("guarantees", "Lib.G", "premises", "domain"), ["Lib.Gone.D"], "guarantees[Lib.G].premises.domain: Lib.Gone.D"),
    (("guarantees", "Lib.G", "premises", "open"), ["Lib.Gone.O"], "guarantees[Lib.G].premises.open: Lib.Gone.O"),
    (("guarantees", "Lib.G", "premises", "reads"), ["Lib.Gone.Shape.mk"], "guarantees[Lib.G].premises.reads: Lib.Gone.Shape.mk"),
    (("shape", "properties", "Lib.G", "rests_on"), ["Lib.Gone.H"], "shape.properties[Lib.G].rests_on: Lib.Gone.H"),
    (("verifier", "entries"), ["Lib.Gone.Entry"], "verifier.entries: Lib.Gone.Entry"),
    (("verifier", "stop"), {"Lib.Gone.Stop": "a hash"}, "verifier.stop: Lib.Gone.Stop"),
    (("verifier", "hash"), "Gone.Sha", "verifier.hash: Gone.Sha"),
])
def test_a_kept_entry_that_cites_what_the_copied_files_lack_refuses_the_lock(where, value, named):
    source = lock()
    holder = source
    for k in where[:-1]:
        holder = holder[k]
    holder[where[-1]] = value
    with pytest.raises(ValueError, match="cite what it doesn't copy") as e:
        PL.public(source, has)
    assert named in str(e.value)


def test_a_lock_as_it_stands_names_each_thing_the_copied_files_lack():
    left = PL.missing(lock(), has)
    for name in ("roots: Gone", "proved_in: ../Gone", "guarantees: Lib.Gone.G", "reads: Gone.Defs",
                 "layers[Mod.Defs].imports: Gone.Defs", "witnesses[Lib.Gone.G].honest: Lib.Gone.G_honest",
                 "guarantees[Lib.Gone.G].premises.reads: Lib.Gone.Shape.mk", "guarantees.Lib.G.owner: a lane field",
                 "open.Lib.O.issue: a lane field", "open.Lib.O.due: a lane field", "open_seed: a lane field"):
        assert name in left


def test_a_refusal_names_at_most_so_many_and_counts_the_rest():
    source = lock()
    source["verifier"]["entries"] = [f"Lib.Gone.E{i}" for i in range(PL.SHOWN + 3)]
    with pytest.raises(ValueError, match="verifier.entries: Lib.Gone.E0; .*; and 3 more"):
        PL.public(source, has)


@pytest.mark.parametrize("section, value, named", [
    ("lemmas_of", {"Lib.G": ["Lib.lemma"]}, "lemmas_of"),
    ("shape", {"premises": [], "theorems": ["Lib.G"]}, "shape.theorems"),
    ("claims", {"modules": [], "also": {}}, "claims.also"),
    ("verifier", {"entries": [], "reach": []}, "verifier.reach"),
    ("trees", ["Mod.Defs"], "trees"),
])
def test_a_section_no_rule_covers_refuses_the_lock(section, value, named):
    source = lock() | {section: value}
    assert PL.unruled(source) == [named]
    with pytest.raises(ValueError, match=f"no rule says what {named} names"):
        PL.public(source, has)


def test_a_lock_with_few_sections_is_read_as_one():
    source = {"format": 3, "roots": ["Mod"], "exempt": {"Mod.Run": "a runner"}, "default_owner": "@lane"}
    assert PL.public(source, has) == ({"format": 3, "roots": ["Mod"], "exempt": {"Mod.Run": "a runner"}}, {})
    assert PL.public({"format": 3, "roots": ["Mod"]}, has) == ({"format": 3, "roots": ["Mod"]}, {})


def test_a_section_that_is_one_name_goes_with_it():
    assert PL.public({"format": 3, "assumptions_module": "Gone.Defs"}, has) == ({"format": 3},
                                                                              {"assumptions_module": ["Gone.Defs"]})


def test_a_locks_text_is_read_entry_by_entry() -> None:
    """Each key of a dict section is an entry with its record, a section of subsections is read by subsection, and a
    section that is no dict is one entry."""
    got = {"format": 3, "roots": ["Lib"], "exempt": {"Lib.Run": "a runner"},
           "shape": {"frozen": {"Lib.G": {"why": "as it was"}}, "events": ["Lib.E"]}}
    assert list(PL.entries(got)) == [
        ("format", "", 3), ("roots", "", ["Lib"]), ("exempt", "Lib.Run", {"Lib.Run": "a runner"}),
        ("shape.frozen", "Lib.G", {"Lib.G": {"why": "as it was"}}), ("shape.events", "", ["Lib.E"])]


def test_what_a_locks_strings_still_hold_is_named_by_section_and_entry() -> None:
    """`leftovers` reads every string, keys too, and names each find by the entry that holds it; what counts is the
    caller's to say."""
    got = {"exempt": {"Lib.Run": "a runner, by Ada", "Lib.Ada": {"why": ["plain", "for Ada and Ada"]}},
           "shape": {"frozen": {"Lib.G": "plain"}, "events": ["Ada.E"]}, "format": 3}

    def find(text: str) -> list[tuple[str, str]]:
        return [("name", "Ada")] * text.count("Ada")

    assert PL.leftovers(got, find) == [
        ("exempt", "Lib.Run", "name", "Ada"), ("exempt", "Lib.Ada", "name", "Ada"), ("exempt", "Lib.Ada", "name", "Ada"),
        ("exempt", "Lib.Ada", "name", "Ada"), ("shape.events", "", "name", "Ada")]
    assert PL.leftovers(got, lambda text: []) == []


def test_the_names_and_paths_a_reason_spells() -> None:
    text = "As Lib.Gone.walk' and Lib.Kept (see ./lean/Lib/Gone.lean, tests/data/ and lean/Lib/Gone.lean.), e.g. Lib.Kept."
    assert PL.names(text) == ["Lib.Gone.walk'", "Lib.Kept", "Gone.lean", "e.g"]
    assert PL.paths(text) == ["lean/Lib/Gone.lean", "tests/data"]
    assert PL.names("a plain reason") == [] and PL.paths("a plain reason, and/or none") == ["and/or"]


def test_the_refusal_names_each_place_by_file_section_and_entry_and_says_the_fix() -> None:
    left = [("pkg/lean-audit.json", "layers", "Mod", "pr", "#1234"),
            ("pkg/lean-audit.json", "shape.frozen", "Lib.G", "declaration", "Lib.Gone.G"),
            ("other/lean-audit.json", "compile_time", "", "path", "pkg/Gone.lean")]
    said = PL.refusal(left)
    assert said.startswith("public_lock: 3 place(s)")
    for named in ("pkg/lean-audit.json: layers[Mod] holds pr '#1234'",
                  "pkg/lean-audit.json: shape.frozen[Lib.G] holds declaration 'Lib.Gone.G'",
                  "other/lean-audit.json: compile_time holds path 'pkg/Gone.lean'"):
        assert named in said
    for fix in ("never rewords a lock", "reword each in the lock itself", "keep the sentence", "public_boundary.toml",
                "[[residue]]", "`until`"):
        assert fix in said
    many = [("pkg/lean-audit.json", "exempt", f"Mod.M{i}", "pr", f"#{i}") for i in range(PL.SHOWN + 3)]
    said = PL.refusal(many)
    assert f"{PL.SHOWN + 3} place(s)" in said and "; and 3 more" in said
    assert f"'#{PL.SHOWN - 1}'" in said and f"'#{PL.SHOWN}'" not in said


def test_a_lock_reworded_at_source_has_nothing_to_refuse() -> None:
    """The fix the refusal names, end to end on one lock: the reason loses its citation and the name of what the export
    leaves out, keeps its sentence, and `leftovers` finds nothing."""
    def find(text: str):
        return [("pr", t) for t in text.split() if t.startswith("#")] + \
            [("declaration", n) for n in PL.names(text) if n.startswith("Lib.Gone")]

    lock = {"format": 3, "layers": {"Mod": {"imports": ["Mathlib"], "why": "trusted text, as Lib.Gone.G reads it (see #1234 )"}}}
    assert [(kind, text) for *_, kind, text in PL.leftovers(lock, find)] == [("pr", "#1234"), ("declaration", "Lib.Gone.G")]
    lock["layers"]["Mod"]["why"] = "trusted text, as the guarantee over it reads it"
    assert PL.leftovers(lock, find) == []
