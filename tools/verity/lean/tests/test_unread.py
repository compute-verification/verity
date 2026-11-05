"""`tools/verity/lean/unread.py`: the closure over the facts `Unread.lean` writes (no Lean needed: synthetic facts, and the
fixture tree's facts kept beside it), and, when `lake` is installed, that those kept facts are what `Unread.lean` writes
on the fixture's build."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("unread", HERE.parent / "unread.py")
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)

FIXTURE = HERE / "fixtures" / "unread"
FACTS = HERE / "fixtures" / "unread.facts.jsonl"
REGENERATE = (f"python tools/verity/lean/unread.py --root {FIXTURE.relative_to(U.ROOT)} --build --out DIR, then copy "
              f"DIR/unread-facts.jsonl over {FACTS.relative_to(U.ROOT)}")


def facts_of(tmp_path: Path, modules: list[str], constants: list[dict]) -> U.Facts:
    path = tmp_path / "facts.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in [*({"module": m, "found": True, "imports": []} for m in modules),
                                                             *constants]))
    return U.Facts(path)


def test_each_constant_belongs_to_its_declaration_and_a_shared_matcher_reads_no_owner(tmp_path):
    names = ["A.S", "A.S.mk", "A.S.x", "A.S.casesOn", "A.f", "A.f.match_1", "A.g", "A.f.eq_1", "A.S.ext"]
    idx = {n: i for i, n in enumerate(names)}
    c = [{"n": "A.S", "m": 0, "k": "inductive", "r": [1, 3], "f": ["structure"], "u": [idx["A.S.mk"]]},
         {"n": "A.S.mk", "m": 0, "k": "ctor", "p": "A.S", "f": [], "u": [idx["A.S"]]},
         {"n": "A.S.x", "m": 0, "k": "def", "r": [2, 2], "p": "A.S", "f": [], "u": [idx["A.S"]]},
         {"n": "A.S.casesOn", "m": 0, "k": "def", "f": [], "u": [idx["A.S"]]},
         {"n": "A.f", "m": 0, "k": "def", "r": [5, 7], "f": [], "u": [idx["A.f.match_1"]]},
         {"n": "A.f.match_1", "m": 0, "k": "def", "f": [], "u": []},
         {"n": "A.g", "m": 0, "k": "def", "r": [9, 11], "f": [], "u": [idx["A.f.match_1"], idx["A.S.x"]]},
         {"n": "A.f.eq_1", "m": 0, "k": "theorem", "f": [], "u": [idx["A.f"]]},
         {"n": "A.S.ext", "m": 0, "k": "theorem", "r": [1, 3], "f": [], "u": [idx["A.S"]]}]
    facts = facts_of(tmp_path, ["A"], c)
    owner = U.owners(facts)
    assert [names[owner[i]] for i in range(len(names))] == ["A.S", "A.S", "A.S", "A.S", "A.f", "A.f", "A.g", "A.f", "A.S"]
    reached = U.closure(facts, owner, {idx["A.g"]})
    assert {names[i] for i in range(len(names)) if reached[i]} == {"A.g", "A.f.match_1", "A.S.x", "A.S", "A.S.mk"}
    assert U.kind_of(facts, idx["A.S"]) == "structure"


def test_the_longest_chain_and_the_imports_once_modules_go():
    imports = {"T": ["M", "S"], "S": ["M"], "M": ["C"], "C": [], "D": ["C"], "G": ["D"]}
    nodes = sorted(imports)
    assert U.longest(nodes, imports, {m: 1 for m in nodes}) == (4, ["C", "M", "S", "T"])
    assert U.longest(nodes, imports, {"T": 1, "S": 1, "M": 1, "C": 1, "D": 10, "G": 10}) == (21, ["C", "D", "G"])
    after = U.contract(nodes, imports, {"S", "D"})
    assert after == {"T": ["M"], "M": ["C"], "C": [], "G": ["C"]}
    assert U.longest(sorted(after), after, {m: 1 for m in after}) == (3, ["C", "M", "T"])
    # a deleted module reached twice, once before what it imports: A imports B and C, B imports C, all three go
    deep = {"K": ["A"], "A": ["B", "C"], "B": ["C"], "C": ["X"], "X": []}
    assert U.contract(sorted(deep), deep, {"A", "B", "C"}) == {"K": ["X"], "X": []}


#: what the fixture's closure finds: each module's status, and each read module's unread declarations
STATUS = {"Code": "read", "Definitions": "read", "Definitions.Attr": "attributes", "Definitions.Checks": "read",
          "Definitions.Core": "read",
          "Definitions.Dead": "unread", "Definitions.Gone": "unread aggregator", "Definitions.Notation": "read",
          "Definitions.Orphan": "unbuilt", "Properties.Code": "read", "Proofs.Flock": "read", "Proofs.Flock.Main": "read",
          "Proofs.Flock.Spare": "unread", "Proofs.Properties.Code": "read"}
UNREAD = {"Definitions.Core": {"Core.pick": "def", "Core.unusedHelper": "def", "Core.Width": "abbrev",
                               "Core.Pair": "structure", "Core.instInhabitedBox": "instance", "Core.unusedLemma": "theorem"},
          "Definitions.Dead": {"Dead.lonely": "def", "Dead.lonely_pos": "theorem"},
          "Proofs.Flock.Spare": {"Spare.quadruple": "theorem"}}


def check_report(rep: dict) -> None:
    mods = rep["modules"]
    assert {m: r["status"] for m, r in mods.items()} == STATUS
    got = {m: {n: k for n, k, _ in r["unread_declarations"]} for m, r in mods.items() if r["unread_declarations"]}
    assert got == UNREAD
    core = mods["Definitions.Core"]
    assert core["kept_only_by"] == {"defeq": 1, "coercion": 1, "lock module": 0, "cited": 1}
    assert rep["seeds"]["coercion"] == 1
    assert mods["Proofs.Flock.Main"]["kept_only_by"]["defeq"] == 1
    assert core["unread_cited_in_docs"] == ["Core.unusedHelper"]
    assert mods["Definitions.Checks"]["kept_only_by"]["cited"] == 1
    assert rep["cited_by_code"]["Definitions.Checks"] == ["tools/reader.toml"]
    assert mods["Definitions.Attr"]["foreign_attributes"] == ["Core.unusedLemma"]
    dead = rep["areas"]["Definitions/Dead"]
    assert (dead["wholly_unread_modules"], dead["wholly_unread_lines"]) == (1, mods["Definitions.Dead"]["lines"])
    assert rep["areas"]["Definitions/Core"]["owners"] == {"@proofs": 1, "@lean": 1}
    flock = rep["chains"]["Proofs.Flock"]
    assert [m for m, _, _ in flock["before_by_modules"]["chain"]] == [
        "Definitions.Core", "Proofs.Flock.Main", "Proofs.Flock.Spare", "Proofs.Flock"]
    assert [m for m, _, _ in flock["after_by_modules"]["chain"]] == ["Definitions.Core", "Proofs.Flock.Main", "Proofs.Flock"]
    t = rep["totals"]
    assert t["wholly_unread_modules"] == 3 and t["unbuilt_modules"] == 1
    assert t["unread"] == sum(len(v) for v in UNREAD.values())


def test_the_fixture_s_closure_keeps_what_is_read_and_finds_the_rest():
    """The fixture's kept facts: the guarantee's proof and what it reads, the properties, the code, a notation, a
    name a tool reads, a module of checks a tool names by path, the `@[defeq]` theorems and a coercion instance, which no
    term names, are kept; a matcher another declaration shares, a name in `reads` or in a README, and an instance
    nothing uses keep nothing; a module that gives another's theorem `simp` stays."""
    tree = U.Tree(FIXTURE, U.tracked(FIXTURE))
    facts = U.Facts(FACTS)
    coe = facts.index["Core.instCoeWrapNat"]
    assert facts.instance_of[coe] == "Coe" and not any(coe in u for u in facts.used)
    assert facts.index["Core.unwrap"] in facts.used[facts.index["Properties.Code.Unwraps"]]
    rep = U.analyse(tree, facts)
    check_report(rep)
    assert "Proofs.Flock" in U.markdown(rep, None)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_kept_facts_are_what_unread_lean_writes_on_the_fixture(tmp_path):
    """The only test that runs `Unread.lean` (it builds the fixture's four packages and compiles the program): the
    synthetic and kept facts above can't catch a change in what Lean puts in an `.olean`."""
    root = tmp_path / "unread"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns(".lake"))
    tree = U.Tree(root, U.tracked(root))
    U.build(tree, "Security/Proofs", setup=False)
    out = tmp_path / "out"
    out.mkdir()
    fresh = U.write_facts(tree, "Security/Proofs", out)
    check_report(U.analyse(tree, U.Facts(fresh)))
    assert fresh.read_text() == FACTS.read_text(), f"the kept facts are stale: {REGENERATE}"
