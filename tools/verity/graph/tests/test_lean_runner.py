"""The Lean runners on real Lean, in a toy checkout the Lean families list and the reference executor runs: the auditor's
control package (`tools/verity/lean/tests/fixtures/control`), a package under the module system and one that doesn't
compile. Each module compiles from its reads alone to Lake's bytes and gets its replay and its axioms; a lock passes, is
stale with the command that regenerates it, or names a root no task builds. Skipped where the fixture's toolchain isn't
installed. Slow (`lean` for each module, `Facts.lean` compiled for each `check.py`): only real Lean shows that the setup
file, the import artifacts and a build made of the reads are what the toolchain and the audit take."""
from __future__ import annotations

import io
import json
import os
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
    "H.lean": "module\npublic import A\ntheorem h : a = 1 := by sorry\naxiom hidden : a = 1\npublic theorem h2 : a = 1 := hidden\n",
    "Ms.lean": "import C\nimport D\nimport F\nimport G\nimport H\n",
    "lakefile.toml": 'name = "ms"\ndefaultTargets = ["Ms"]\n\n[[lean_lib]]\nname = "Ms"\n'
                     'roots = ["A", "B", "C", "D", "F", "G", "H", "Ms"]\n',
    "lake-manifest.json": manifest("ms"),
}
BAD = {"lakefile.toml": 'name = "bad"\ndefaultTargets = ["Bad"]\n\n[[lean_lib]]\nname = "Bad"\n',
       "Bad.lean": "theorem bad : 1 = 2 := rfl\n", "lake-manifest.json": manifest("bad")}


@pytest.fixture(scope="module")
def lean_env(tmp_path_factory) -> Path:
    """The worker's `lean` env, shared by this file's tests, so `Replay.lean` and `Facts.lean` compile once."""
    return tmp_path_factory.mktemp("lean-env")


@pytest.fixture
def tree(tmp_path: Path, lean_env: Path, monkeypatch) -> Path:
    """A checkout of the control package (`ctl/`), `MS` (`ms/`), `BAD` (`bad/`), the auditor and the Lean runners, which
    the runners and the auditor read as the repository."""
    tracked = graph.tracked(ROOT)
    files = {f: (ROOT / f).read_text() for f in (*la.tool_files(tracked), *graph.runner_sources("lean-module"),
                                                 *graph.runner_sources("lean-lock"))}
    files |= {"ctl/" + f[len(CONTROL) + 1:]: (ROOT / f).read_text() for f in tracked if f.startswith(CONTROL + "/")}
    for pkg, fs in (("ms", MS), ("bad", BAD)):
        files |= {f"{pkg}/{f}": text for f, text in fs.items()} | {f"{pkg}/lean-toolchain": TOOLCHAIN + "\n"}
    root = toy_tree(tmp_path / "tree", (ROOT / graph.SPEC).read_text(), files)
    for m in (module_runner, lock_runner, la):
        monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setenv("VERITY_LEAN_ENV", str(lean_env))
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
    p = subprocess.run([str(BIN / "lake"), "build"], cwd=d, capture_output=True, text=True,
                       env={**os.environ, "PATH": f"{BIN}{os.pathsep}{os.environ['PATH']}"})
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
        assert res["status"] == "pass" and res["detail"]["imports"] == len(t["reads"]), res
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
    h = json.loads(artifact(tmp_path / "x", got["lean-module:ms/H.lean"], "axioms"))["axioms"]
    assert h["_private.H.0.h"] == ["sorryAx"], "a private theorem's sorry is in its axioms"
    assert h["h2"] == ["_private.H.0.hidden"], "so is an axiom only a public theorem's unexposed proof uses"


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
