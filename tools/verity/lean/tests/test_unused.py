"""`tools/verity/lean/unused.py`: its keep rules and its deletion on a stand-in package and `Used.lean` document (no Lean
needed), and `Used.lean` on the controls when `lake` is installed."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_unused", HERE.parent / "unused.py")
U = importlib.util.module_from_spec(spec)
sys.modules.setdefault("lean_unused", U)
spec.loader.exec_module(U)

#: the stand-in package's modules: {module: its source}
SOURCES = {
    "Prune": "import Prune.A.Main\nimport Prune.B.Group\n",
    "Prune.A.Main": "import Prune.A.Base\nimport Prune.B.Notation\nimport Prune.B.Attr\nimport Prune.B.Refd\n\n"
                    "theorem main : ⟪base⟫ = 1 := rfl\ntheorem other : refd = 2 := rfl\n",
    "Prune.A.Base": "def base : Nat := 1\n",
    "Prune.A.Side": "import Prune.A.Base\n\n-- names `Prune.B.Named`, in a comment\ntheorem side : base = 1 := rfl\n",
    "Prune.B.Group": "/-! An aggregator of what nothing uses. -/\nimport Prune.B.Old\n",
    "Prune.B.Old": "import Prune.A.Side\n\ntheorem old : True := trivial\n",
    "Prune.B.Notation": "notation \"⟪\" x \"⟫\" => x\nlocal notation \"unseen\" => 0\n",
    "Prune.B.Attr": "theorem attr_lemma : 0 + 0 = 0 := rfl\nattribute [simp] attr_lemma\n",
    "Prune.B.Named": "import Prune.B.Dep\n\ntheorem named : True := trivial\n",
    "Prune.B.Dep": "theorem dep : True := trivial\n",
    "Prune.B.Refd": "def refd : Nat := 2\n",
}
#: {module: (constants, used, refs)}, as `Used.lean` would report them
FACTS = {"Prune": (0, 0, []), "Prune.A.Main": (2, 1, ["Prune.A.Base", "Prune.B.Refd"]), "Prune.A.Base": (1, 1, []),
         "Prune.A.Side": (1, 0, ["Prune.A.Base"]), "Prune.B.Group": (0, 0, []), "Prune.B.Old": (1, 0, []),
         "Prune.B.Notation": (2, 0, []), "Prune.B.Attr": (1, 0, []), "Prune.B.Named": (1, 0, []),
         "Prune.B.Dep": (1, 0, []), "Prune.B.Refd": (1, 0, [])}


def path_of(module: str) -> str:
    return module.replace(".", "/") + ".lean"


@pytest.fixture
def stand_in(tmp_path):
    """(the package, its lock, the `Used.lean` document, a test file naming `Prune.B.Named`, as a path from the root)."""
    pkg = tmp_path / "prune"
    for m, src in SOURCES.items():
        (pkg / path_of(m)).parent.mkdir(parents=True, exist_ok=True)
        (pkg / path_of(m)).write_text(src)
    (pkg / "lakefile.toml").write_text('name = "prune"\ndefaultTargets = ["Prune"]\n\n[[lean_lib]]\nname = "Prune"\n')
    policy = {"roots": ["Prune"], "guarantees": {"Prune.A.main": {"owner": "@lean"}}}
    (pkg / U.C.POLICY).write_text(json.dumps(policy))
    used = {"pins": {"Prune.A.main": "Prune.A.Main"},
            "modules": {m: {"olean": f"/x/.lake/build/lib/lean/{m}.olean", "constants": c, "used": u,
                            "by": "Prune.A.main" if u else None, "refs": r} for m, (c, u, r) in FACTS.items()},
            "imports": {}}
    test = tmp_path / "tests" / "test_named.py"
    test.parent.mkdir()
    test.write_text('MODULE = "Prune.B.Named"\nSOURCE = "import Prune.B.Named\\n#check named"\n')
    return pkg, policy, used, os.path.relpath(test, U.ROOT)


def test_decide_keeps_by_each_rule(stand_in):
    pkg, policy, used, test = stand_in
    got = U.decide(pkg, policy, used, [test], ["Prune.B.Old"])
    why = {m: r.split(":", 1)[0] for m, r in got["kept"].items()}
    assert why == {"Prune": "root", "Prune.A.Main": "used", "Prune.A.Base": "used", "Prune.B.Refd": "refs",
                   "Prune.B.Notation": "syntax", "Prune.B.Attr": "attribute", "Prune.B.Named": "named",
                   "Prune.B.Dep": "named"}
    assert got["unused"] == ["Prune.A.Side", "Prune.B.Group", "Prune.B.Old"]
    assert got["frozen"] == ["Prune.B.Old"]


def test_what_a_queued_pr_touches_or_imports_is_kept(stand_in):
    pkg, policy, used, test = stand_in
    edits = {"4999": {"files": [f"{U.rel(pkg)}/Prune/A/Side.lean", "README.md"], "imports": ["Mathlib.Logic.Basic"]}}
    got = U.decide(pkg, policy, used, [test], [], edits)
    assert got["kept"]["Prune.A.Side"] == "queued: #4999 touch or import it"
    assert got["unused"] == ["Prune.B.Group", "Prune.B.Old"]
    imports = {"5000": {"files": [], "imports": ["Prune.B.Group"]}}
    got = U.decide(pkg, policy, used, [test], [], imports)
    assert got["unused"] == [] and got["kept"]["Prune.A.Side"].startswith("queued: imported by Prune.B.Group")
    named = {"5001": {"files": [], "imports": [], "names": ["Prune.B.Old", "README.md"]}}
    got = U.decide(pkg, policy, used, [test], [], named)
    assert got["kept"]["Prune.B.Old"] == "queued: #5001 touch or import it"
    assert got["kept"]["Prune.B.Group"].startswith("aggregator: ") and got["unused"] == []
    umbrella = {"5002": {"files": [f"{U.rel(pkg)}/Prune/B/Group.lean"], "imports": []}}
    got = U.decide(pkg, policy, used, [test], [], umbrella)
    assert got["kept"]["Prune.B.Group"].startswith("queued: ") and got["unused"] == ["Prune.A.Side", "Prune.B.Old"]
    doc = U.listing(pkg, got, {U.rel(pkg): policy}, used, [test], {"Prune.B.Group": got["kept"]["Prune.B.Group"]})
    assert [r["module"] for r in doc["held"]] == ["Prune.B.Group"] and doc["total"]["held"] == 1
    assert "## Held: unused but for a queued PR or as a check (1 modules" in U.table(doc)


def test_a_conformance_check_is_kept_without_its_imports(stand_in):
    pkg, policy, used, test = stand_in
    (pkg / "Prune/B/Old.lean").write_text("import Prune.A.Side\n\nexample : 1 + 1 = 2 := rfl\ntheorem old : True := trivial\n")
    got = U.decide(pkg, policy, used, [test], [])
    assert got["kept"]["Prune.B.Old"].startswith("checks: ") and got["kept"]["Prune.B.Group"].startswith("aggregator: ")
    assert got["unused"] == ["Prune.A.Side"]
    assert U.decide(pkg, policy, used, [test], [], checks=False)["unused"] == ["Prune.A.Side", "Prune.B.Group", "Prune.B.Old"]
    names = ["Prune.X.ChainVectors", "Prune.X.PeelChecks.Gram0", "Prune.X.CheckSum", "Prune.Vectorsx"]
    assert [bool(U.CHECK_NAME.search(m)) for m in names] == [True, True, False, False]


def test_syntax_nobody_writes_and_a_local_one_keep_nothing(stand_in):
    pkg, policy, used, test = stand_in
    (pkg / path_of("Prune.A.Main")).write_text(SOURCES["Prune.A.Main"].replace("⟪base⟫", "base"))
    got = U.decide(pkg, policy, used, [test], [])
    assert "Prune.B.Notation" in got["unused"]
    (pkg / path_of("Prune.A.Side")).write_text(SOURCES["Prune.A.Side"] + "macro_rules | `(tactic| rfl) => `(tactic| rfl)\n")
    assert U.decide(pkg, policy, used, [test], [])["kept"]["Prune.A.Side"].startswith("syntax: ")


def test_an_unbuilt_module_is_kept(stand_in):
    pkg, policy, used, test = stand_in
    del used["modules"]["Prune.B.Old"]
    assert U.decide(pkg, policy, used, [test], [])["kept"]["Prune.B.Old"].startswith("unbuilt: ")


def test_listing_carries_bytes_area_owner_and_totals(stand_in):
    pkg, policy, used, test = stand_in
    decided = U.decide(pkg, policy, used, [test], ["Prune.B.Old"])
    doc = U.listing(pkg, decided, {U.rel(pkg): policy}, used, [test])
    rows = {r["module"]: r for r in doc["unused"]}
    assert rows["Prune.A.Side"]["owner"] == "@lean" and rows["Prune.A.Side"]["owner_from"] == "lock"
    assert rows["Prune.A.Side"]["bytes"] == len(SOURCES["Prune.A.Side"].encode())
    assert doc["areas"]["A"]["modules"] == 1 and doc["areas"]["B"] == {
        "modules": 1, "bytes": len(SOURCES["Prune.B.Group"].encode()), "frozen": 1,
        "frozen_bytes": len(SOURCES["Prune.B.Old"].encode()), "owners": {"-": 2}}
    assert doc["total"]["unused"] == 3 and doc["total"]["frozen"] == 1 and doc["total"]["kept_by"]["named"] == 2
    assert "| Prune.A.Side |" in U.table(doc)


def test_delete_rewrites_the_imports_through_what_it_deleted(stand_in):
    pkg, policy, used, test = stand_in
    doc = U.listing(pkg, U.decide(pkg, policy, used, [test], ["Prune.B.Old"]), {U.rel(pkg): policy}, used, [test])
    mods = U.C.lean_modules(pkg)
    (pkg / "README.md").write_text(f"`{U.rel(pkg)}/Prune/B/Group.lean` groups them\n")
    (pkg / "x" / U.C.POLICY).parent.mkdir()
    (pkg / "x" / U.C.POLICY).write_text('{"reads": {"Prune.B.Group.Sub": {}, "Prune.B.Group": {}}}\n')
    (pkg / "y" / U.C.POLICY).parent.mkdir()
    (pkg / "y" / U.C.POLICY).write_text('{"reads": {"Prune.B.Group.Sub": {}}}\n')
    assert U.delete(pkg, "B", doc, used, set()) == ["Prune.B.Group"]
    files = [f"{U.rel(pkg)}/{f}" for f in ("README.md", f"x/{U.C.POLICY}", f"y/{U.C.POLICY}")]
    assert U.stale_mentions(pkg, ["Prune.B.Group"], files, mods) == [
        f"{U.rel(pkg)}/README.md: {U.rel(pkg)}/Prune/B/Group.lean", f"{U.rel(pkg)}/x/{U.C.POLICY}: Prune.B.Group"]
    assert not (pkg / path_of("Prune.B.Group")).exists()
    assert (pkg / "Prune.lean").read_text() == "import Prune.A.Main\nimport Prune.B.Old\n"
    assert U.delete(pkg, "A", doc, used, set()) == ["Prune.A.Side"]
    assert (pkg / path_of("Prune.B.Old")).read_text() == "import Prune.A.Base\n\ntheorem old : True := trivial\n"
    assert U.delete(pkg, "A", doc, used, {"Prune.A.Side"}) == []


def test_rewrite_keeps_each_line_s_form_and_drops_what_is_reached():
    reach = {"A": {"B"}, "B": set(), "C": set()}.get
    text = "module\n\npublic import A\npublic import D -- gone\nimport E\n\n/-- doc -/\ndef x := 1\n"
    got = U.rewrite_imports(text, {"D", "E"}, {"D": ["B", "C"], "E": ["C"]}, lambda m: reach(m) or set())
    assert got == "module\n\npublic import A\npublic import C\n\n/-- doc -/\ndef x := 1\n"


def test_references_name_paths_modules_and_constants(tmp_path):
    mods = {"Prune.A.Main": "Prune/A/Main.lean", "Prune.A": "Prune/A.lean", "Prune.B": "Prune/B.lean"}
    pkg = os.path.relpath(tmp_path / "prune", U.ROOT)
    files = {"tests/test_a.py": f'"{pkg}/Prune/A/Main.lean"\n', "b.sh": "lake env lean -e Prune.B.thm\n",
             "c.toml": f'paths = ["{pkg}/Prune/A"]\n', "d.md": f"see `{pkg}/Prune/B.lean`\n",
             "e.py": "Prune.Bx Prune.A.X Prune.A.Main.Y\n",
             "f.lean": "import Prune.B\n\n/-! `Prune.A.Main` proves it. -/\n", "tests/test_g.py": f'"{pkg}/Prune/A/" "{pkg}/Prune/B"\n',
             "tests/test_h.py": 'SOURCE = "import Prune.A\\n#check 1"\n',
             "tests/test_i.py": 'FILES = ["Prune/B.lean"]\nP = LEAN / "Prune" / "A" / "Main.lean"\nQ = "Prune/B"\n'}
    (tmp_path / "prune" / "Prune" / "B").mkdir(parents=True)
    for f, text in files.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text)
    got = U.references([os.path.relpath(tmp_path / f, U.ROOT) for f in files], pkg, mods)
    named = {k: sorted(Path(f).name for f, _ in v) for k, v in got.items()}
    assert named == {"Prune.A.Main": ["test_a.py", "test_i.py"], "Prune.B": ["b.sh", "f.lean", "test_i.py"],
                     "Prune.A": ["c.toml", "test_h.py"],
                     ("docs", "Prune.B"): ["d.md"], ("docs", "Prune.A.Main"): ["f.lean"]}
    reads = {(k, Path(f).name): r for k, v in got.items() for f, r in v}
    assert not reads["Prune.A.Main", "test_a.py"] and reads["Prune.B", "b.sh"] and reads["Prune.B", "f.lean"]
    assert not reads["Prune.A", "c.toml"] and reads["Prune.A", "test_h.py"]


#: the controls with an unused module (`Extra`), an aggregator of it (`Group`), a notation the proofs write (`Syn`) and a
#: definition only an unlisted theorem of a used module reads (`Base`); `Scratch` (exempt, so its imports are kept)
#: imports only `Defs`
CONTROL_EDITS = {
    "Control.lean": "import Control.Assumptions\nimport Control.Defs\nimport Control.Thms\nimport Control.Group\n",
    "Control/Base.lean": "namespace Control\n\ndef base : Nat := 1\n\nend Control\n",
    "Control/Syn.lean": "import Control.Defs\n\nnamespace Control\n\nnotation \"dbl \" n:max => double n\n\nend Control\n",
    "Control/Extra.lean": "import Control.Defs\n\nnamespace Control\n\ntheorem extra_unused : double 2 = 4 := rfl\n\nend Control\n",
    "Control/Group.lean": "import Control.Extra\n",
    "Scratch.lean": "import Control.Defs\n\n#check @Control.double\n",
}


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_used_lean_on_the_controls(tmp_path):
    """`Used.lean` compiled and run on a real build: the stand-in document above is what it writes (each guarantee's
    module, the closure's counts, `refs`), which only a build shows."""
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    for f, text in CONTROL_EDITS.items():
        (pkg / f).write_text(text)
    thms = pkg / "Control" / "Thms.lean"
    thms.write_text(thms.read_text().replace("import Control.Defs\n", "import Control.Defs\nimport Control.Base\n"
                                             "import Control.Syn\n")
                    .replace("\nend Control", "\ntheorem helper2 : base + 1 = dbl 1 := rfl\n\nend Control"))
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True, env=U.C.lake_env())
    policy = U.C.load(pkg)
    pins, extra = U.names_of(policy, (pkg / U.C.POLICY).read_text())
    used = U.run_used(pkg, policy["roots"], pins, sorted(set(extra) - set(pins)), tmp_path / "used.json")
    assert used["pins"] == {"Control.main_thm": "Control.Thms", "Control.Small_fails": "Control.Thms",
                            "Control.Small_meets": "Control.Thms"}
    facts = used["modules"]
    assert set(facts) == {"Control", "Control.Assumptions", "Control.Defs", "Control.Thms", "Control.Base", "Control.Syn",
                          "Control.Extra", "Control.Group"}
    assert all(facts[m]["used"] > 0 for m in ("Control.Assumptions", "Control.Defs", "Control.Thms"))
    assert facts["Control.Thms"]["used"] < facts["Control.Thms"]["constants"]
    assert all(facts[m]["used"] == 0 for m in ("Control", "Control.Base", "Control.Syn", "Control.Extra", "Control.Group"))
    assert facts["Control"]["constants"] == facts["Control.Group"]["constants"] == 0
    assert sorted(facts["Control.Thms"]["refs"]) == ["Control.Assumptions", "Control.Base", "Control.Defs"]
    assert facts["Control.Extra"]["refs"] == ["Control.Defs"]
    assert "Init.Prelude" in used["imports"] and set(used["imports"]["Control.Group"]) - {"Init"} == {"Control.Extra"}
    got = U.decide(pkg, policy, used, [], [])
    why = {m: r.split(":", 1)[0] for m, r in got["kept"].items()}
    assert why == {"Control": "root", "Control.Assumptions": "used", "Control.Defs": "used", "Control.Thms": "used",
                   "Control.Base": "refs", "Control.Syn": "syntax", "Scratch": "exempt", "RunOk": "runs"}
    assert got["unused"] == ["Control.Extra", "Control.Group"]
