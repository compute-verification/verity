"""`tools/verity/lean/unread.py`: the closure over the facts `Unread.lean` writes (no Lean needed: synthetic facts, and the
fixture tree's facts kept beside it), and, when `lake` is installed, that those kept facts are what `Unread.lean` writes
on the fixture's build."""
from __future__ import annotations

import gzip
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


def test_a_cited_path_names_the_modules_it_ends_written_out_or_built_from_literals():
    files = {"Proofs.Core.Prims.Vectors": "security/Proofs/Core/Prims/Vectors.lean",
             "Proofs.Core.TC.Vectors": "security/Proofs/Core/TC/Vectors.lean", "Root": "Root.lean"}
    modules = {k: m for m, f in files.items() for k in (m, f, m.replace(".", "/") + ".lean")}
    named = U.path_namer(modules)
    text = ('FILES = (LEAN / "Prims" / "Vectors.lean", Path(ROOT, "security", "Proofs/Core/TC", "Vectors.lean"))\n'
            'ANY = here / "Vectors.lean"\nTOP = "../Root.lean"\n')
    paths = U.lean_paths(text)
    assert {"Prims/Vectors.lean", "security/Proofs/Core/TC/Vectors.lean"} <= set(paths)
    assert set().union(*map(named, paths)) == {"Proofs.Core.Prims.Vectors", "Proofs.Core.TC.Vectors", "Root"}
    assert named("Vectors.lean") == set() and named("/abs/x/Security/Proofs/Core/TC/Vectors.lean") == {"Proofs.Core.TC.Vectors"}
    assert all((U.ROOT / f).is_file() for f in U.NOT_READERS), "NOT_READERS names this tool's files where they are"


def test_a_wholly_unread_module_is_deletable_alone_only_when_nothing_outside_those_mentions_it(tmp_path):
    mods = ["K", "A", "B", "C", "D", "E", "F", "G"]
    names = ["K.k", "A.a", "B.b", "C.c", "D.d", "E.e", "E.e.match_1", "F.Ns.x", "G.y", "G.z"]
    idx = {n: i for i, n in enumerate(names)}
    uses = {"K.k": ["A.a", "E.e.match_1"], "A.a": ["B.b"], "C.c": ["D.d"]}
    homes = {"G.z": "K"}
    c = [{"n": n, "m": mods.index(homes.get(n, n.split(".")[0])), "k": "def", "f": [], "u": [idx[u] for u in uses.get(n, ())]}
         for n in names]
    facts = facts_of(tmp_path, mods, c)
    mod_of = [facts.modules[facts.module[i]]["module"] for i in range(len(names))]
    shared = [bool(U.SHARED.search(x)) for x in names]
    opens = {"K": U.opened("open scoped Ns Std in\nopen G hiding y\n  open Q.R -- open F\nopen H (s)\n")}
    assert opens["K"] == ["Ns", "Std", "G", "Q.R", "H"]
    alone, mentioned_by = U.deletable(facts, mod_of, shared, set(mods) - {"K"}, opens)
    # K mentions A, A mentions B; C mentions D but nothing outside mentions C; a matcher of E's is shared; K opens a
    # namespace only F holds, and one that K holds too
    assert alone == {"C", "D", "E", "G"}
    assert dict(mentioned_by) == {"A": {"K"}, "F": {"K"}}


#: what the fixture's closure finds: each module's status, and each read module's unread declarations
STATUS = {"Code": "read", "Definitions": "read", "Definitions.Attr": "attributes", "Definitions.Checks": "read",
          "Definitions.Core": "read",
          "Definitions.Dead": "unread", "Definitions.Gone": "unread aggregator", "Definitions.Notation": "read",
          "Definitions.Orphan": "unbuilt", "Properties.Code": "read", "Proofs.Flock": "read", "Proofs.Flock.Main": "read",
          "Proofs.Flock.Spare": "unread", "Proofs.Properties.Code": "read"}
UNREAD = {"Definitions.Core": {"Core.pick": "def", "Core.unusedHelper": "def", "Core.Width": "abbrev",
                               "Core.Pair": "structure", "Core.instInhabitedBox": "instance", "Core.unusedLemma": "theorem"},
          "Definitions.Dead": {"Dead.lonely": "def", "Dead.lonely_pos": "theorem"},
          "Proofs.Flock.Main": {"Core.usesDead": "theorem"},
          "Proofs.Flock.Spare": {"Spare.quadruple": "theorem"}}
#: each wholly unread module: whether it is deletable alone, and the modules that stay and mention it
GONE = {"Definitions.Dead": (False, ["Proofs.Flock.Main"]), "Definitions.Gone": (True, []),
        "Proofs.Flock.Spare": (False, ["Proofs.Flock"])}


def check_report(rep: dict) -> None:
    mods = rep["modules"]
    assert {m: r["status"] for m, r in mods.items()} == STATUS
    got = {m: {n: k for n, k, _ in r["unread_declarations"]} for m, r in mods.items() if r["unread_declarations"]}
    assert got == UNREAD
    core = mods["Definitions.Core"]
    assert core["kept_only_by"] == {"defeq": 1, "coercion": 1, "lock module": 0, "lock text": 0, "lock doc": 0, "cited": 1,
                                    "SecurityProofs": 0}
    assert rep["seeds"]["coercion"] == 1
    assert mods["Proofs.Flock.Main"]["kept_only_by"]["defeq"] == 1
    assert mods["Proofs.Flock.Main"]["kept_only_by"]["SecurityProofs"] == 3, "the proof no lock names, and its two lemmas"
    assert rep["areas"]["Proofs/Flock"]["kept_only_by"]["SecurityProofs"] == 3 == rep["totals"]["kept_only_by"]["SecurityProofs"]
    assert core["unread_cited_in_docs"] == ["Core.unusedHelper"]
    assert mods["Definitions.Checks"]["kept_only_by"]["cited"] == 1
    assert rep["cited_by_code"]["Definitions.Checks"] == ["tools/checks.py"]
    assert "Definitions.Dead" not in rep["cited_by_code"], "a module a record names (tools/reader.toml) keeps nothing"
    assert mods["Definitions.Attr"]["foreign_attributes"] == ["Core.unusedLemma"]
    dead = rep["areas"]["Definitions/Dead"]
    assert (dead["wholly_unread_modules"], dead["wholly_unread_lines"]) == (1, mods["Definitions.Dead"]["lines"])
    assert rep["areas"]["Definitions/Core"]["owners"] == {"@proofs": 1, "@lean": 1}
    flock = rep["chains"]["Proofs.Flock"]
    assert [m for m, _, _ in flock["before_by_modules"]["chain"]] == [
        "Definitions.Core", "Proofs.Flock.Main", "Proofs.Flock.Spare", "Proofs.Flock"]
    assert [m for m, _, _ in flock["after_by_modules"]["chain"]] == ["Definitions.Core", "Proofs.Flock.Main", "Proofs.Flock"]
    assert [m for m, _, _ in flock["after_alone_by_modules"]["chain"]] == [
        "Definitions.Core", "Proofs.Flock.Main", "Proofs.Flock.Spare", "Proofs.Flock"]
    assert {m: (r["deletable_alone"], r["mentioned_by"]) for m, r in mods.items() if "deletable_alone" in r} == GONE
    t = rep["totals"]
    assert t["wholly_unread_modules"] == 3 and t["unbuilt_modules"] == 1
    assert (t["alone_modules"], t["alone_lines"]) == (1, mods["Definitions.Gone"]["lines"])
    assert t["unread"] == sum(len(v) for v in UNREAD.values())


def test_the_fixture_s_closure_keeps_what_is_read_and_finds_the_rest():
    """The fixture's kept facts: the guarantee's proof and what it reads, the properties, the code, a notation, a
    name a tool reads, a module of checks a tool names by path, the `@[defeq]` theorems and a coercion instance, which no
    term names, are kept; a matcher another declaration shares, a name in `reads` or in a README, and an instance
    nothing uses keep nothing; a module that gives another's theorem `simp` stays. The guarantee's proof, which no lock
    names, and the lemmas only it reads are kept by the `SecurityProofs` seed alone, which the report counts per module,
    per area and in all; once a lock names the proof, the seed alone keeps nothing and no other count moves."""
    tree = U.Tree(FIXTURE, U.tracked(FIXTURE))
    facts = U.Facts(FACTS)
    coe = facts.index["Core.instCoeWrapNat"]
    assert facts.instance_of[coe] == "Coe" and not any(coe in u for u in facts.used)
    assert facts.index["Core.unwrap"] in facts.used[facts.index["Properties.Code.Unwraps"]]
    rep = U.analyse(tree, facts)
    check_report(rep)
    assert "Proofs.Flock" in U.markdown(rep, None)
    assert "names in files outside Lean 2, constants with a `SecurityProofs` component 3." in U.markdown(rep, None)
    tree.locks["security/Proofs"]["guarantees"] = {"Core.SecurityProofs.Spec": {"owner": "@proofs"}}
    named = U.analyse(tree, facts)
    assert named["totals"]["kept_only_by"] == rep["totals"]["kept_only_by"] | {"SecurityProofs": 0}
    assert {k: v for k, v in named["totals"].items() if k != "kept_only_by"} == \
        {k: v for k, v in rep["totals"].items() if k != "kept_only_by"}
    assert U.counts(named) == U.counts(rep)


def test_facts_files_join_by_name_gzipped_or_not(tmp_path):
    """Each audit writes its own facts: a module one file only names is the one another found, its foreign attributes are
    the union, and a constant both hold mentions what either says."""
    one = [{"module": "A", "found": True, "imports": []}, {"module": "B", "found": False},
           {"n": "A.a", "m": 0, "k": "def", "r": [1, 1], "f": [], "u": [1]}, {"n": "A.b", "m": 0, "k": "def", "f": [], "u": []}]
    two = [{"module": "B", "found": True, "imports": ["A"], "foreign": ["A.b"]},
           {"module": "A", "found": True, "imports": [], "foreign": ["X.x"]},
           {"n": "B.b", "m": 0, "k": "theorem", "r": [1, 1], "f": [], "u": [1]},
           {"n": "A.a", "m": 1, "k": "def", "r": [1, 1], "f": [], "u": [0]}]
    p1, p2 = tmp_path / "one.jsonl", tmp_path / "two.jsonl.gz"
    p1.write_text("".join(json.dumps(x) + "\n" for x in one))
    p2.write_bytes(gzip.compress("".join(json.dumps(x) + "\n" for x in two).encode()))
    f = U.Facts(p1, p2)
    assert f.name == ["A.a", "A.b", "B.b"] and [list(u) for u in f.used] == [[1, 2], [], [0]]
    assert [f.modules[f.module[i]]["module"] for i in range(3)] == ["A", "A", "B"]
    assert f.modules[f.module_index["B"]] == two[0] and f.modules[f.module_index["A"]]["foreign"] == ["X.x"]
    assert U.names("its witness is 'Dead.lonely', see x.y") == {"Dead.lonely'", "Dead.lonely", "x.y"}


def test_a_lock_s_texts_and_the_document_they_name_seed_and_a_module_of_checks_of_a_read_definition_stays(tmp_path):
    """A lock's text that names a declaration (quoted as a string literal closes it) or a module seeds it, and so does what
    a document it names holds; with no tool naming it, the module of checks stays, since it tests `Core.double`, which is
    read."""
    root = tmp_path / "unread"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns(".lake"))
    (root / "security" / "PROTOCOL.md").write_text("The helper is `Core.unusedHelper`.\n")
    files = [f for f in U.tracked(FIXTURE) if f != "tools/checks.py"] + ["security/PROTOCOL.md"]
    tree = U.Tree(root, files)
    tree.locks["security"]["guarantees"]["Core.Guarantees.Spec"]["why"] = (
        "its witness is 'Dead.lonely', Proofs.Flock.Spare states the rest, and PROTOCOL.md says why")
    rep = U.analyse(tree, U.Facts(FACTS), full=False)
    mods = rep["modules"]
    assert rep["lock_docs"] == {"security/PROTOCOL.md": ["security/lean-audit.json"]}
    assert {m: mods[m]["status"] for m in ("Definitions.Dead", "Proofs.Flock.Spare", "Definitions.Checks")} == {
        "Definitions.Dead": "read", "Proofs.Flock.Spare": "read", "Definitions.Checks": "checks"}
    assert [n for n, _, _ in mods["Definitions.Dead"]["unread_declarations"]] == ["Dead.lonely_pos"]
    assert "Core.unusedHelper" not in {n for n, _, _ in mods["Definitions.Core"]["unread_declarations"]}
    assert rep["totals"]["wholly_unread_modules"] == 0 and rep["chains"] == {} and not mods["Definitions.Core"]["kept_only_by"]
    assert "Read only because" not in U.markdown(rep, None)


def test_the_ratchet_fails_when_an_area_s_wholly_unread_modules_rise_and_its_caps_only_come_down(tmp_path):
    """No caps is a cap of nothing; caps at the counts pass; caps above them pass and leave the lowered caps; a cap below
    its count fails naming the area's modules and the fix. `--update-caps` writes the counts when there are no caps, lowers
    caps above them, and never raises one."""
    files, caps, out = U.tracked(FIXTURE), tmp_path / "caps.json", tmp_path / "out"
    now = U.counts(U.analyse(U.Tree(FIXTURE, files), U.Facts(FACTS), full=False))
    assert now == {"Definitions/Dead": {"modules": 1, "declarations": 2}, "Definitions/Gone": {"modules": 1, "declarations": 0},
                   "Proofs/Flock": {"modules": 1, "declarations": 1}}
    r = U.check(FIXTURE, files, [FACTS], out, caps)
    assert len(r["failures"]) == 3 and not r["lower"] and (out / "unread.json").is_file()
    caps.write_text(U.caps_text(now))
    r = U.check(FIXTURE, files, [FACTS], out, caps)
    assert r["failures"] == [] and not r["lower"] and not (out / "unread-caps.json").exists()
    higher = {a: {"modules": v["modules"] + 1, "declarations": v["declarations"] + 5} for a, v in now.items()} | {
        "Proofs/Gone": {"modules": 2, "declarations": 9}}
    caps.write_text(U.caps_text(higher))
    r = U.check(FIXTURE, files, [FACTS], out, caps)
    assert r["failures"] == [] and r["lower"] and U.load_caps(out / "unread-caps.json") == now
    tight = now | {"Definitions/Dead": {"modules": 1, "declarations": 1}}
    caps.write_text(U.caps_text(tight))
    (fail,) = U.check(FIXTURE, files, [FACTS], out, caps)["failures"]
    assert fail.startswith(f"unread: Definitions/Dead has 1 wholly unread modules (2 declarations), above its cap of 1 "
                           f"(1 declarations) in {caps}: Definitions.Dead (2 declarations). ") and (
        "delete it, or seed it: a lock/code reference" in fail)
    cli = ["--root", str(FIXTURE), "--facts", str(FACTS), "--out", str(tmp_path / "cli"), "--caps", str(tmp_path / "new.json")]
    assert U.main([*cli, "--update-caps"]) == 0 and U.load_caps(tmp_path / "new.json") == now
    (tmp_path / "new.json").write_text(U.caps_text(higher))
    assert U.main(cli) == 0 and U.main([*cli, "--update-caps"]) == 0 and U.load_caps(tmp_path / "new.json") == now
    (tmp_path / "new.json").write_text(U.caps_text(tight))
    assert U.main([*cli, "--update-caps"]) == 1 and U.load_caps(tmp_path / "new.json") == tight
    with pytest.raises(ValueError, match="not a lean-unread-caps/v1 file"):
        U.load_caps(FACTS.parent / "unread" / "security" / "lean-audit.json")


def test_the_kept_caps_are_a_caps_file_as_update_caps_writes_it():
    caps = U.load_caps(U.CAPS)
    assert caps and all(set(v) == set(U.CAP_KEYS) and v["modules"] > 0 for v in caps.values())
    assert U.CAPS.read_text() == U.caps_text(caps), "write it with `unread.py --caps ... --update-caps`"


def test_build_builds_what_the_lean_check_builds_of_each_analysis_package(monkeypatch):
    """`--build` builds, in ENV, each analysis package's default targets and then the modules its lock lists under
    `roots`, as the Lean check builds a package: a root no default target imports would otherwise be `unbuilt`, and what
    it alone reads would count as unread on the tool's own build and as read on a check's facts."""
    tree = U.Tree(FIXTURE, U.tracked(FIXTURE))
    ran = []
    monkeypatch.setattr(U.subprocess, "run", lambda argv, cwd, check: ran.append((argv, cwd, check)))
    U.build(tree, "security/Proofs", setup=False)
    assert all(cwd == FIXTURE / "security" / "Proofs" and check for _, cwd, check in ran)
    assert [argv for argv, _, _ in ran] == [
        ["lake", "build"], ["lake", "build", "+Proofs.Flock", "+Proofs.Properties.Code"],
        ["lake", "build", "@security"], ["lake", "build", "@security/+Definitions"],
        ["lake", "build", "@security_properties"], ["lake", "build", "@security_properties/+Properties.Code"]]
    del tree.locks["security"]
    ran.clear()
    U.build(tree, "security/Proofs", setup=False)
    assert [argv[2:] for argv, _, _ in ran] == [[], ["+Proofs.Flock", "+Proofs.Properties.Code"], ["@security"],
                                                ["@security_properties"], ["@security_properties/+Properties.Code"]]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_kept_facts_are_what_unread_lean_writes_on_the_fixture(tmp_path):
    """The only test that runs `Unread.lean` (it builds the fixture's four packages and compiles the program): the
    synthetic and kept facts above can't catch a change in what Lean puts in an `.olean`. And the only one that runs
    `build`'s commands: Lake takes them, and a module that no default target imports is built once the lock lists it
    under `roots`, not before."""
    root = tmp_path / "unread"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns(".lake"))
    tree = U.Tree(root, U.tracked(root))
    U.build(tree, "security/Proofs", setup=False)
    out = tmp_path / "out"
    out.mkdir()
    fresh = U.write_facts(tree, "security/Proofs", out)
    check_report(U.analyse(tree, U.Facts(fresh)))
    assert fresh.read_text() == FACTS.read_text(), f"the kept facts are stale: {REGENERATE}"
    (root / "security" / "Proofs" / "Flock" / "Lone.lean").write_text("theorem Proofs.Flock.lone : True := trivial\n")
    U.build(U.Tree(root, U.tracked(root)), "security/Proofs", setup=False)
    assert not list(root.rglob("Lone.olean")), "no default target imports it"
    lock = root / "security" / "Proofs" / "lean-audit.json"
    rec = json.loads(lock.read_text())
    lock.write_text(json.dumps(rec | {"roots": [*rec["roots"], "Proofs.Flock.Lone"]}))
    U.build(U.Tree(root, U.tracked(root)), "security/Proofs", setup=False)
    assert list(root.rglob("Lone.olean")), "a root of the lock is built though no default target imports it"
