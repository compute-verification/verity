"""The Lean runners on real Lean, in a toy checkout the Lean families list and the reference executor runs: the auditor's
control package (`tools/verity/lean/tests/fixtures/control`), a package under the module system, one with a `sorry`, an
axiom and `native_decide` in private proofs and a private theorem that skipped the kernel, and one that doesn't compile.
Each module compiles from its reads alone to Lake's bytes and gets its replay and its declarations' axiom sets, an
importer's taken through its import's record; an `import all` without its import's private part is an error; a lock
passes, is stale with the command that regenerates it, or names a root no task builds. Skipped where the fixture's
toolchain isn't installed. Slow (`lean` for each module, `Facts.lean` compiled for each `check.py`): only real Lean
shows that the setup file, the import artifacts and a build made of the reads are what the toolchain and the audit take."""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import git, toy_tree

from verity_graph import ROOT, graph, local, runners
from verity_graph.families import lean_lock, lean_module
from verity_graph.runners import lean_lock as lock_runner
from verity_graph.runners import lean_module as module_runner

la = lean_module.la
CONTROL = "tools/verity/lean/tests/fixtures/control"
TOOLCHAIN = (ROOT / CONTROL / "lean-toolchain").read_text().strip()
BIN = module_runner.toolchain_dir(TOOLCHAIN) / "bin"
LISTERS = {"lean-module": lean_module.tasks, "lean-lock": lean_lock.tasks}
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not (BIN / "lean").is_file(), reason=f"no Lean toolchain {TOOLCHAIN}")]


def manifest(name: str) -> str:
    return json.dumps({"version": "1.1.0", "packagesDir": ".lake/packages", "packages": [], "name": name, "lakeDir": ".lake"})


#: a package under the module system: public, private and `import all` imports, and a file outside it importing its modules
MS = {
    "A.lean": "module\npublic def a : Nat := 1\ntheorem a_eq : a = 1 := rfl\ndef privA : Nat := 2\n",
    "B.lean": "module\npublic import A\npublic def b : Nat := a + 1\n",
    "C.lean": "module\nimport A\npublic def c : Nat := 3\ntheorem c_eq : c = 3 := rfl\n",
    "D.lean": "module\nimport B\ntheorem d : True := trivial\n",
    "F.lean": "import B\ntheorem f : b = b := rfl\n",
    "G.lean": "module\nimport all A\ntheorem g : privA = 2 := rfl\n",
    "Ms.lean": "import C\nimport D\nimport F\nimport G\n",
    "lakefile.toml": 'name = "ms"\ndefaultTargets = ["Ms"]\n\n[[lean_lib]]\nname = "Ms"\n'
                     'roots = ["A", "B", "C", "D", "F", "G", "Ms"]\n',
    "lake-manifest.json": manifest("ms"),
}
#: a `sorry`, an axiom and `native_decide` in proofs only `A`'s private part holds; `B`, whose public theorem rests on them;
#: and `K`, whose private theorem skipped the kernel (`debug.skipKernelTC`, spelled so that no source scan finds it)
AX = {
    "A.lean": "module\npublic def a : Nat := 1\ntheorem p : a = 1 := by sorry\naxiom hidden : a = 1\n"
              "public theorem h2 : a = 1 := hidden\npublic theorem h3 : a = 1 := by sorry\n"
              "public theorem h4 : 2 + 2 = 4 := by native_decide\n",
    "B.lean": "module\npublic import A\npublic theorem b : a = 1 ∧ a = 1 ∧ 2 + 2 = 4 := ⟨h2, h3, h4⟩\n",
    "K.lean": "module\nmeta import Lean\nopen Lean Elab Command in\n#eval show CommandElabM Unit from liftCoreM do\n"
              "  let opt := Name.mkStr (Name.mkStr .anonymous \"debug\") (\"skipKernel\" ++ \"TC\")\n"
              "  let n := Name.mkNum (`_private ++ `K) 0 ++ `bogus\n  withOptions (fun o => o.setBool opt true) <| addDecl\n"
              "    (.thmDecl { name := n, levelParams := [], type := mkConst ``False, value := mkConst ``True.intro })\n",
    "lakefile.toml": 'name = "ax"\ndefaultTargets = ["Ax"]\n\n[[lean_lib]]\nname = "Ax"\nroots = ["A", "B", "K"]\n',
    "lake-manifest.json": manifest("ax"),
}
BAD = {"lakefile.toml": 'name = "bad"\ndefaultTargets = ["Bad"]\n\n[[lean_lib]]\nname = "Bad"\n',
       "Bad.lean": "theorem bad : 1 = 2 := rfl\n", "lake-manifest.json": manifest("bad")}


@pytest.fixture(scope="module")
def lean_env(tmp_path_factory) -> Path:
    """The check's cache, holding the worker's `lean` env, shared by this file's tests, so `Replay.lean` compiles once."""
    return tmp_path_factory.mktemp("check-cache")


@pytest.fixture
def tree(tmp_path: Path, lean_env: Path, monkeypatch) -> Path:
    """A checkout of the control package (`ctl/`), `MS` (`ms/`), `AX` (`ax/`), `BAD` (`bad/`), the auditor and the Lean
    runners, which the runners and the auditor read as the repository."""
    tracked = graph.tracked(ROOT)
    files = {f: (ROOT / f).read_text() for f in (*la.tool_files(tracked), *graph.runner_sources("lean-module"),
                                                 *graph.runner_sources("lean-lock"))}
    files |= {"ctl/" + f[len(CONTROL) + 1:]: (ROOT / f).read_text() for f in tracked if f.startswith(CONTROL + "/")}
    for pkg, fs in (("ms", MS), ("ax", AX), ("bad", BAD)):
        files |= {f"{pkg}/{f}": text for f, text in fs.items()} | {f"{pkg}/lean-toolchain": TOOLCHAIN + "\n"}
    root = toy_tree(tmp_path / "tree", (ROOT / graph.SPEC).read_text(), files)
    for m in (module_runner, lock_runner, la):
        monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setenv("CHECK_CACHE", str(lean_env))
    return root


def run(root: Path, doc: dict, work: Path) -> dict:
    """``doc``'s tasks by the reference executor, each through its runner in this process."""
    out = local.check(root, None, work, lambda t, w: runners.execute(t, doc, w), 0, None, io.StringIO(), doc)
    assert not out["errors"] and not out["cancelled"], {k: v["detail"] for k, v in out["errors"].items()}
    return out["results"]


def only(doc: dict, prefix: str) -> dict:
    return doc | {"tasks": [t for t in doc["tasks"] if t["id"].startswith(prefix)]}


def artifact(work: Path, res: dict, name: str) -> bytes:
    return (work / "artifacts" / res["outputs"][name].partition(":")[2]).read_bytes()


def lake(root: Path, pkg: str, tmp_path: Path) -> Path:
    """A copy of ``pkg`` that `lake build` built."""
    d = tmp_path / "lake" / pkg
    shutil.copytree(root / pkg, d)
    p = subprocess.run([str(BIN / "lake"), "build"], cwd=d, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return d


def lib(d: Path, path: str) -> Path:
    """Where Lake's build of the package at ``d`` holds the output a task writes at ``path``."""
    return d / ".lake" / "build" / "lib" / "lean" / path.removeprefix("out/lean/")


def test_each_module_compiles_from_its_reads_alone_to_lake_s_bytes(tree, tmp_path):
    doc = only(graph.listing(tree, {"lean-module"}, listers=LISTERS), "lean-module:ctl/")
    got = run(tree, doc, tmp_path / "x")
    assert sorted(got) == [f"lean-module:ctl/{f}.lean" for f in ("Control", "Control/Assumptions", "Control/Defs",
                                                                 "Control/Thms")]
    built = lake(tree, "ctl", tmp_path)
    differ = []
    for t in doc["tasks"]:
        res, src = got[t["id"]], t["id"].partition(":")[2]
        assert res["status"] == "pass" and res["detail"]["imports"] == sum(r["output"] == "olean" for r in t["reads"]), res
        for o in t["outputs"]:
            if o["name"] != "axioms" and (mine := artifact(tmp_path / "x", res, o["name"])) != (
                    theirs := lib(built, o["path"]).read_bytes()):
                assert str(built / src.removeprefix("ctl/")).encode() in theirs and src.encode() in mine \
                    and str(tree).encode() not in mine, (src, o["name"])
                differ.append(f"{src} {o['name']}")
    assert differ == ["ctl/Control/Thms.lean olean"], \
        "a module's .olean keeps its warnings' source path as `lean` got it, the tree's in the runner and Lake's absolute one"
    axioms = {t["vars"]["module"]: json.loads(artifact(tmp_path / "x", got[t["id"]], "axioms")) for t in doc["tasks"]}
    assert axioms["Control.Defs"] == {"module": "Control.Defs", "axioms": {}}
    assert axioms["Control.Thms"]["axioms"]["Control.main_thm"] == ["Quot.sound", "propext"]
    assert "Control.helper" not in axioms["Control.Thms"]["axioms"]
    assert got["lean-module:ctl/Control/Thms.lean"]["detail"]["replayed"] > 0


def test_a_module_system_package_compiles_part_by_part_to_lake_s_bytes(tree, tmp_path):
    doc = only(graph.listing(tree, {"lean-module"}, listers=LISTERS), "lean-module:ms/")
    got = run(tree, doc, tmp_path / "x")
    built = lake(tree, "ms", tmp_path)
    a = next(t for t in doc["tasks"] if t["id"] == "lean-module:ms/A.lean")
    assert {o["name"] for o in a["outputs"]} >= set(lean_module.PARTS), "A is a `module` file"
    for t in doc["tasks"]:
        assert got[t["id"]]["status"] == "pass", got[t["id"]]
        for o in t["outputs"]:
            if o["name"] != "axioms":
                assert artifact(tmp_path / "x", got[t["id"]], o["name"]) == lib(built, o["path"]).read_bytes(), \
                    (t["id"], o["name"])


def test_an_import_all_whose_task_did_not_read_the_private_part_is_an_error_naming_the_import(tree, tmp_path):
    doc = only(graph.listing(tree, {"lean-module"}, listers=LISTERS), "lean-module:ms/")
    a, g = (next(t for t in doc["tasks"] if t["id"] == f"lean-module:ms/{f}.lean") for f in "AG")
    assert runners.execute(a, doc, tmp_path / "a")["status"] == "pass"
    olean = next(o["path"] for o in a["outputs"] if o["name"] == "olean")
    (tmp_path / "g" / olean).parent.mkdir(parents=True)
    shutil.copyfile(tmp_path / "a" / olean, tmp_path / "g" / olean)
    res = runners.execute(g | {"reads": [{"task": a["id"], "output": "olean"}]}, doc, tmp_path / "g")
    assert res["status"] == "error" and res["detail"]["missing"] == ["A"], res["detail"]


def test_axioms_in_private_proofs_fail_their_module_and_reach_an_importer_through_its_record(tree, tmp_path):
    doc = only(graph.listing(tree, {"lean-module"}, listers=LISTERS), "lean-module:ax/")
    a, b = (next(t for t in doc["tasks"] if t["id"] == f"lean-module:ax/{f}.lean") for f in "AB")
    native = "_private.A.0.h4._native.native_decide.ax_1_1"    # the axiom `native_decide` adds for each use
    res = runners.execute(a, doc, tmp_path / "a")
    assert res["status"] == "fail" and res["detail"]["axioms"] == {
        "_private.A.0.hidden": ["_private.A.0.hidden", "h2"], native: [native, "h4"], "sorryAx": ["_private.A.0.p", "h3"]}, \
        res["detail"]

    def importer(reads: tuple[str, ...], name: str) -> dict:
        """B's task reading ``reads`` of A's outputs alone (A's check skipped), and its record."""
        work = tmp_path / name
        for o in a["outputs"]:
            if o["name"] in reads:
                (work / o["path"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tmp_path / "a" / o["path"], work / o["path"])
        res = runners.execute(b | {"reads": [{"task": a["id"], "output": o} for o in reads]}, doc, work)
        out = next(o["path"] for o in b["outputs"] if o["name"] == "axioms")
        assert res["status"] == "fail" and res["detail"]["imports"] == 1, res
        return json.loads((work / out).read_text())["axioms"]

    assert importer(("olean", "axioms"), "public") == {"b": [native, "_private.A.0.hidden", "sorryAx"]}, \
        "A's public part and its record alone: B's theorem carries the axioms A's private proofs use"
    assert importer(("olean",), "unrecorded") == {"b": ["a", "h2", "h3", "h4"]}, \
        "without A's record, what B uses of A's public part counts as axioms: the task fails rather than miss one"


def test_a_private_declaration_that_skipped_the_kernel_fails_its_own_module(tree, tmp_path):
    doc = graph.task_graph(tree, "lean-module:ax/K.lean", listers=LISTERS)
    res = runners.execute(next(t for t in doc["tasks"] if t["id"] == "lean-module:ax/K.lean"), doc, tmp_path / "k")
    assert res["status"] == "fail" and res["detail"].get("rejected") == ["K"], res["detail"]


def test_a_module_that_does_not_compile_fails_with_lean_s_errors(tree, tmp_path):
    doc = graph.task_graph(tree, "lean-module:bad/Bad.lean", listers=LISTERS)
    out = local.check(tree, None, tmp_path / "x", lambda t, w: runners.execute(t, doc, w), 0, None, io.StringIO(), doc)
    res = out["results"]["lean-module:bad/Bad.lean"]
    assert res["status"] == "fail" and res["detail"]["error"] == "lean exited 1", res
    assert res["detail"]["messages"][0].startswith("bad/Bad.lean:1: "), res["detail"]


def lock(tree: Path, tmp_path: Path, policy: dict, name: str) -> dict:
    """The lean-lock task of the control package with ``policy`` committed as its lock, run with what it reads."""
    (tree / "ctl" / "lean-audit.json").write_text(json.dumps(policy, indent=1) + "\n")
    git(tree, "commit", "-q", "-am", name)
    doc = graph.task_graph(tree, "lean-lock:ctl", listers=LISTERS)
    out = local.check(tree, None, tmp_path / name, lambda t, w: runners.execute(t, doc, w), 0, None, io.StringIO(), doc)
    assert not out["errors"], {k: v["detail"] for k, v in out["errors"].items()}
    return out["results"]["lean-lock:ctl"]


def test_a_lock_passes_on_its_records_and_is_stale_with_the_command_that_regenerates_them(tree, tmp_path):
    built = lake(tree, "ctl", tmp_path)
    p = subprocess.run([sys.executable, str(tree / "tools/verity/lean/check.py"), "--update", "--no-replay", "--no-runs",
                        "--out", str(tmp_path / "update"), str(built)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    recorded = json.loads((built / "lean-audit.json").read_text())
    res = lock(tree, tmp_path, recorded, "recorded")
    assert res["status"] == "pass", res["detail"]
    assert json.loads(artifact(tmp_path / "recorded", res, "records")) == recorded

    stale = recorded | {"guarantees": recorded["guarantees"] | {"Control.main_thm": {"owner": "@lean"}}}
    res = lock(tree, tmp_path, stale, "stale")
    assert res["status"] == "fail" and res["detail"]["regenerate"] == "python3 tools/verity/lean/check.py --build --update ctl"
    assert "is stale" in res["detail"]["error"] and res["detail"]["regenerate"] in res["detail"]["error"]
    assert json.loads(artifact(tmp_path / "stale", res, "records")) == recorded, "its records are the lock it regenerates"


def test_a_lock_naming_a_root_no_task_builds_fails(tree, tmp_path):
    policy = json.loads((tree / "ctl" / "lean-audit.json").read_text())
    res = lock(tree, tmp_path, policy | {"roots": [*policy["roots"], "Control.Missing"]}, "missing")
    assert res["status"] == "fail" and "Control.Missing" in res["detail"]["error"], res["detail"]
