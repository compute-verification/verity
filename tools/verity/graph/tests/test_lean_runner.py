"""The Lean runners on real Lean, in a toy checkout the Lean families list and the reference executor runs: the auditor's
control package (`tools/verity/lean/tests/fixtures/control`), a package under the module system, one with a `sorry`, an
axiom and `native_decide` in private proofs and a private theorem that skipped the kernel, one that doesn't compile, and
one whose env fetches a package with a `sorry` in a private proof. The toolchain's env passes, every module of it replayed
and recorded, the executables that can't be imported together in rounds; the fetched package's `sorry` fails its env, and
its missing private part is an error; that env, layered on the toolchain's, replays its own module alone and records the
toolchain's as their record has them, and a base record of other bytes than the worker's is an error. Each module compiles from its reads alone to Lake's bytes and gets its replay and
its declarations' axiom sets, an importer's taken through its import's record and the env's through the env's; a record
of another env or of other bytes than the worker's, and an `import all` without its import's private part, are errors; a
lock passes, is stale with the command that regenerates it, or names a root no task builds, and a spec whose `proved_in`
package requires it by path and a package with a fetched package's premise pass on the records a Lake build gives them,
as `check.py` audits them. Skipped where the fixture's
toolchain isn't installed. Slow (`lean` for each module, the toolchain's every module through the kernel once for the
file, `Facts.lean` compiled for each `check.py`): only real Lean shows that the setup file, the import artifacts and a
build made of the reads are what the toolchain and the audit take, and that the env's replay covers what its modules
import."""
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
from verity_graph.families import lean_env as env_family
from verity_graph.families import lean_lock, lean_module
from verity_graph.runners import lean_env as env_runner
from verity_graph.runners import lean_lock as lock_runner
from verity_graph.runners import lean_module as module_runner

la = lean_module.la
CONTROL = "tools/verity/lean/tests/fixtures/control"
TOOLCHAIN = (ROOT / CONTROL / "lean-toolchain").read_text().strip()
BIN = module_runner.toolchain_dir(TOOLCHAIN) / "bin"
LISTERS = {"lean-module": lean_module.tasks, "lean-lock": lean_lock.tasks, "lean-env": env_family.tasks}
#: the env every toy package but `dep` compiles in, the toolchain alone, named after the first of them
ENV = "lean-env:ax"
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not (BIN / "lean").is_file(), reason=f"no Lean toolchain {TOOLCHAIN}")]


def manifest(name: str, *fetched: str) -> str:
    return json.dumps({"version": "1.1.0", "packagesDir": ".lake/packages", "name": name, "lakeDir": ".lake",
                       "packages": [{"type": "git", "scope": "", "name": n, "rev": "0" * 40, "inputRev": "main",
                                     "url": f"https://example.invalid/{n}", "manifestFile": "lake-manifest.json",
                                     "inherited": False, "configFile": "lakefile.toml"} for n in fetched]})


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
#: a package whose env fetches `evil` (`fetched_env` builds it as Lake would): a module whose public theorem has a `sorry`
#: for its proof, which its private part alone holds
EVIL = "module\n\nprelude\npublic import Init.Prelude\n\npublic theorem q : True := sorry\n"
DEP = {"lakefile.toml": 'name = "dep"\ndefaultTargets = ["Dep"]\n\n[[require]]\nname = "evil"\n'
                        'git = "https://example.invalid/evil"\n\n[[lean_lib]]\nname = "Dep"\n',
       "Dep.lean": "prelude\nimport Evil\n", "lake-manifest.json": manifest("dep", "evil")}
#: `evil` with a definition whose public part hides its body, which takes 100000 steps of `Nat.rec` to evaluate, and a
#: theorem the kernel checks at once where `big` is an axiom, and only by evaluating `big` where it has its body
DEEP = {"Deep": "module\n\nprelude\npublic import Init.Prelude\n\npublic noncomputable def big : Nat :=\n"
                "  Nat.rec (motive := fun _ => Nat → Nat) (fun acc => acc) (fun _ r acc => r (Nat.succ acc)) 100000 0\n",
        "Evil": "module\n\nprelude\npublic import Deep\n\n"
                "public theorem t (h : Eq big 100000) : Eq (id big) 100000 := h\n"}


@pytest.fixture(scope="module")
def lean_env(tmp_path_factory) -> Path:
    """The check's cache, holding the worker's `lean` env, shared by this file's tests, so `Replay.lean` compiles once."""
    return tmp_path_factory.mktemp("check-cache")


class Envs:
    """The lean-env tasks this file's tests ran, each once, by key, with its result and its record, which each test's
    executor takes as a cache would: the toolchain's env replays every module of the toolchain."""

    def __init__(self, store: Path) -> None:
        self.store, self.got = store, {}

    def run(self, task: dict, doc: dict) -> tuple[dict, Path]:
        if task["key"] not in self.got:
            work = self.store / f"{len(self.got)}"
            self.got[task["key"]] = (runners.execute(task, doc, work), work / env_family.record(task["vars"]["package"]))
        return self.got[task["key"]]

    def launch(self, doc: dict):
        def go(t: dict, w: Path) -> dict:
            if t["kind"] != "lean-env":
                return runners.execute(t, doc, w)
            res, record = self.run(t, doc)
            if record.is_file():
                stage(record, w, env_family.record(t["vars"]["package"]))
            return res
        return go

    def stage(self, doc: dict, work: Path) -> Path:
        """``work`` holding the record of the env ``doc``'s module tasks read."""
        res, record = self.run(next(t for t in doc["tasks"] if t["id"] == ENV), doc)
        assert res["status"] == "pass", res["detail"]
        return stage(record, work, env_family.record("ax"))


def stage(record: Path, work: Path, at: str) -> Path:
    """``work`` holding ``record`` at ``at``."""
    (work / at).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record, work / at)
    return work / at


@pytest.fixture(scope="module")
def envs(tmp_path_factory) -> Envs:
    return Envs(tmp_path_factory.mktemp("envs"))


@pytest.fixture
def tree(tmp_path: Path, lean_env: Path, monkeypatch) -> Path:
    """A checkout of the control package (`ctl/`), `MS` (`ms/`), `AX` (`ax/`), `BAD` (`bad/`), `DEP` (`dep/`), the auditor
    and the Lean runners, which the runners and the auditor read as the repository."""
    tracked = graph.tracked(ROOT)
    files = {f: (ROOT / f).read_text() for f in (*la.tool_files(tracked), *env_runner.TOOLS,
                                                 *(s for k in ("lean-module", "lean-lock", "lean-env")
                                                   for s in graph.runner_sources(k)))}
    files |= {"ctl/" + f[len(CONTROL) + 1:]: (ROOT / f).read_text() for f in tracked if f.startswith(CONTROL + "/")}
    for pkg, fs in (("ms", MS), ("ax", AX), ("bad", BAD), ("dep", DEP)):
        files |= {f"{pkg}/{f}": text for f, text in fs.items()} | {f"{pkg}/lean-toolchain": TOOLCHAIN + "\n"}
    root = toy_tree(tmp_path / "tree", (ROOT / graph.SPEC).read_text(), files)
    for m in (module_runner, lock_runner, env_runner, la):
        monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setenv("CHECK_CACHE", str(lean_env))
    return root


def run(root: Path, doc: dict, work: Path, envs: Envs) -> dict:
    """``doc``'s tasks by the reference executor, each through its runner in this process."""
    out = local.check(root, None, work, envs.launch(doc), 0, None, io.StringIO(), doc)
    assert not out["errors"] and not out["cancelled"], {k: v["detail"] for k, v in out["errors"].items()}
    return out["results"]


def only(doc: dict, prefix: str) -> dict:
    """``doc`` with the tasks whose ids start with ``prefix`` and the envs they read."""
    keep = [t for t in doc["tasks"] if t["id"].startswith(prefix)]
    envs = {r["task"] for t in keep for r in t["reads"] if r["task"].startswith("lean-env:")}
    return doc | {"tasks": [t for t in doc["tasks"] if t in keep or t["id"] in envs]}


def modules(doc: dict) -> list[dict]:
    return [t for t in doc["tasks"] if t["kind"] == "lean-module"]


def artifact(work: Path, res: dict, name: str) -> bytes:
    return (work / "artifacts" / res["outputs"][name].partition(":")[2]).read_bytes()


def lake(root: Path, pkg: str, tmp_path: Path, *nested: str, fetched: dict[str, Path] | None = None) -> Path:
    """A copy of ``pkg`` that `lake build` built, and each of its ``nested`` packages, with ``fetched`` ({name: checkout})
    copied into its `.lake/packages`."""
    d = tmp_path / "lake" / pkg
    shutil.copytree(root / pkg, d)
    for n, src in (fetched or {}).items():
        shutil.copytree(src, d / ".lake" / "packages" / n, symlinks=True)
    for at in (d, *(d / n for n in nested)):
        p = subprocess.run([str(BIN / "lake"), "build"], cwd=at, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return d


def lib(d: Path, path: str) -> Path:
    """Where Lake's build of the package at ``d`` holds the output a task writes at ``path``."""
    return d / ".lake" / "build" / "lib" / "lean" / path.removeprefix("out/lean/")


def test_each_module_compiles_from_its_reads_alone_to_lake_s_bytes(tree, tmp_path, envs):
    doc = only(graph.listing(tree, {"lean-module"}, listers=LISTERS), "lean-module:ctl/")
    got = run(tree, doc, tmp_path / "x", envs)
    assert sorted(got) == [ENV, *(f"lean-module:ctl/{f}.lean" for f in ("Control", "Control/Assumptions", "Control/Defs",
                                                                       "Control/Thms"))]
    built = lake(tree, "ctl", tmp_path)
    differ = []
    for t in modules(doc):
        res, src = got[t["id"]], t["id"].partition(":")[2]
        assert res["status"] == "pass" and res["detail"]["imports"] == sum(r["output"] == "olean" for r in t["reads"]), res
        assert {"task": ENV, "output": "axioms"} in t["reads"] and res["detail"]["env_modules"] > 0, res
        for o in t["outputs"]:
            if o["name"] != "axioms" and (mine := artifact(tmp_path / "x", res, o["name"])) != (
                    theirs := lib(built, o["path"]).read_bytes()):
                assert str(built / src.removeprefix("ctl/")).encode() in theirs and src.encode() in mine \
                    and str(tree).encode() not in mine, (src, o["name"])
                differ.append(f"{src} {o['name']}")
    assert differ == ["ctl/Control/Thms.lean olean"], \
        "a module's .olean keeps its warnings' source path as `lean` got it, the tree's in the runner and Lake's absolute one"
    axioms = {t["vars"]["module"]: json.loads(artifact(tmp_path / "x", got[t["id"]], "axioms")) for t in modules(doc)}
    assert axioms["Control.Defs"] == {"module": "Control.Defs", "axioms": {}}
    assert axioms["Control.Thms"]["axioms"]["Control.main_thm"] == ["Quot.sound", "propext"]
    assert "Control.helper" not in axioms["Control.Thms"]["axioms"]
    assert got["lean-module:ctl/Control/Thms.lean"]["detail"]["replayed"] > 0


def test_a_module_system_package_compiles_part_by_part_to_lake_s_bytes(tree, tmp_path, envs):
    doc = only(graph.listing(tree, {"lean-module"}, listers=LISTERS), "lean-module:ms/")
    got = run(tree, doc, tmp_path / "x", envs)
    built = lake(tree, "ms", tmp_path)
    a = next(t for t in doc["tasks"] if t["id"] == "lean-module:ms/A.lean")
    assert {o["name"] for o in a["outputs"]} >= set(lean_module.PARTS), "A is a `module` file"
    for t in modules(doc):
        assert got[t["id"]]["status"] == "pass", got[t["id"]]
        for o in t["outputs"]:
            if o["name"] != "axioms":
                assert artifact(tmp_path / "x", got[t["id"]], o["name"]) == lib(built, o["path"]).read_bytes(), \
                    (t["id"], o["name"])


def test_an_import_all_whose_task_did_not_read_the_private_part_is_an_error_naming_the_import(tree, tmp_path, envs):
    doc = only(graph.listing(tree, {"lean-module"}, listers=LISTERS), "lean-module:ms/")
    a, g = (next(t for t in doc["tasks"] if t["id"] == f"lean-module:ms/{f}.lean") for f in "AG")
    envs.stage(doc, tmp_path / "a")
    assert runners.execute(a, doc, tmp_path / "a")["status"] == "pass"
    olean = next(o["path"] for o in a["outputs"] if o["name"] == "olean")
    (tmp_path / "g" / olean).parent.mkdir(parents=True)
    shutil.copyfile(tmp_path / "a" / olean, tmp_path / "g" / olean)
    envs.stage(doc, tmp_path / "g")
    res = runners.execute(g | {"reads": [{"task": a["id"], "output": "olean"}, {"task": ENV, "output": "axioms"}]}, doc,
                          tmp_path / "g")
    assert res["status"] == "error" and res["detail"]["missing"] == ["A"], res["detail"]


def test_axioms_in_private_proofs_fail_their_module_and_reach_an_importer_through_its_record(tree, tmp_path, envs):
    doc = only(graph.listing(tree, {"lean-module"}, listers=LISTERS), "lean-module:ax/")
    a, b = (next(t for t in doc["tasks"] if t["id"] == f"lean-module:ax/{f}.lean") for f in "AB")
    native = "_private.A.0.h4._native.native_decide.ax_1_1"    # the axiom `native_decide` adds for each use
    envs.stage(doc, tmp_path / "a")
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
        envs.stage(doc, work)
        res = runners.execute(b | {"reads": [*({"task": a["id"], "output": o} for o in reads),
                                             {"task": ENV, "output": "axioms"}]}, doc, work)
        out = next(o["path"] for o in b["outputs"] if o["name"] == "axioms")
        assert res["status"] == "fail" and res["detail"]["imports"] == 1, res
        return json.loads((work / out).read_text())["axioms"]

    assert importer(("olean", "axioms"), "public") == {"b": [native, "_private.A.0.hidden", "sorryAx"]}, \
        "A's public part and its record alone: B's theorem carries the axioms A's private proofs use"
    assert importer(("olean",), "unrecorded") == {"b": ["a", "h2", "h3", "h4"]}, \
        "without A's record, what B uses of A's public part counts as axioms: the task fails rather than miss one"


def test_a_private_declaration_that_skipped_the_kernel_fails_its_own_module(tree, tmp_path, envs):
    doc = graph.task_graph(tree, "lean-module:ax/K.lean", listers=LISTERS)
    envs.stage(doc, tmp_path / "k")
    res = runners.execute(next(t for t in doc["tasks"] if t["id"] == "lean-module:ax/K.lean"), doc, tmp_path / "k")
    assert res["status"] == "fail" and res["detail"].get("rejected") == ["K"], res["detail"]


def test_a_module_that_does_not_compile_fails_with_lean_s_errors(tree, tmp_path, envs):
    doc = graph.task_graph(tree, "lean-module:bad/Bad.lean", listers=LISTERS)
    out = local.check(tree, None, tmp_path / "x", envs.launch(doc), 0, None, io.StringIO(), doc)
    res = out["results"]["lean-module:bad/Bad.lean"]
    assert res["status"] == "fail" and res["detail"]["error"] == "lean exited 1", res
    assert res["detail"]["messages"][0].startswith("bad/Bad.lean:1: "), res["detail"]


def lock(tree: Path, tmp_path: Path, policy: dict, name: str, envs: Envs, pkg: str = "ctl") -> dict:
    """The lean-lock task of ``pkg`` (the control package) with ``policy`` committed as its lock, run with what it reads."""
    (tree / pkg / "lean-audit.json").write_text(json.dumps(policy, indent=1) + "\n")
    git(tree, "add", "-A")
    git(tree, "commit", "-q", "-m", name)
    doc = graph.task_graph(tree, f"lean-lock:{pkg}", listers=LISTERS)
    out = local.check(tree, None, tmp_path / name, envs.launch(doc), 0, None, io.StringIO(), doc)
    assert not out["errors"], {k: v["detail"] for k, v in out["errors"].items()}
    return out["results"][f"lean-lock:{pkg}"]


def updated(tree: Path, built: Path, tmp_path: Path, *nested: str) -> dict:
    """The lock `check.py --update` records for the Lake build at ``built``, after those of its ``nested`` packages (the
    ones its `proved_in` names, whose audit its own runs), which go back into ``tree``'s copy."""
    for at in (*(built / n for n in nested), built):
        p = subprocess.run([sys.executable, str(tree / "tools/verity/lean/check.py"), "--update", "--no-replay",
                            "--no-runs", "--out", str(tmp_path / "update" / at.name), str(at)], capture_output=True, text=True)
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    for n in nested:
        (tree / built.name / n / "lean-audit.json").write_text((built / n / "lean-audit.json").read_text())
    return json.loads((built / "lean-audit.json").read_text())


def test_a_lock_passes_on_its_records_and_is_stale_with_the_command_that_regenerates_them(tree, tmp_path, envs):
    recorded = updated(tree, lake(tree, "ctl", tmp_path), tmp_path)
    res = lock(tree, tmp_path, recorded, "recorded", envs)
    assert res["status"] == "pass", res["detail"]
    assert json.loads(artifact(tmp_path / "recorded", res, "records")) == recorded

    stale = recorded | {"guarantees": recorded["guarantees"] | {"Control.main_thm": {"owner": "@lean"}}}
    res = lock(tree, tmp_path, stale, "stale", envs)
    assert res["status"] == "fail" and res["detail"]["regenerate"] == "python3 tools/verity/lean/check.py --build --update ctl"
    assert "is stale" in res["detail"]["error"] and res["detail"]["regenerate"] in res["detail"]["error"]
    assert json.loads(artifact(tmp_path / "stale", res, "records")) == recorded, "its records are the lock it regenerates"


def test_a_lock_naming_a_root_no_task_builds_fails(tree, tmp_path, envs):
    policy = json.loads((tree / "ctl" / "lean-audit.json").read_text())
    res = lock(tree, tmp_path, policy | {"roots": [*policy["roots"], "Control.Missing"]}, "missing", envs)
    assert res["status"] == "fail" and "Control.Missing" in res["detail"]["error"], res["detail"]


def write(root: Path, files: dict[str, str]) -> None:
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)


#: what each toy package below holds beside its sources
TOY = {"lean-toolchain": TOOLCHAIN + "\n"}
#: a spec, `sp`, and the package its `proved_in` names, `sp/pf`, which requires it by path, as Security and Security/Proofs
SP = {**TOY, "Sp.lean": "import Sp.Defs\n", "Sp/Defs.lean": "def Sp.Claim (n : Nat) : Prop := n = n\n",
      "lakefile.toml": 'name = "sp"\ndefaultTargets = ["Sp"]\n\n[[lean_lib]]\nname = "Sp"\n',
      "lake-manifest.json": manifest("sp"), **{f"pf/{f}": t for f, t in TOY.items()},
      "pf/Pf.lean": "import Sp.Defs\n\ntheorem Pf.claim (n : Nat) : Sp.Claim n := rfl\n",
      "pf/lakefile.toml": 'name = "pf"\ndefaultTargets = ["Pf"]\n\n[[require]]\nname = "sp"\npath = ".."\n\n'
                          '[[lean_lib]]\nname = "Pf"\n',
      "pf/lake-manifest.json": json.dumps({"version": "1.1.0", "packagesDir": ".lake/packages", "name": "pf",
                                           "lakeDir": ".lake",
                                           "packages": [{"type": "path", "scope": "", "name": "sp", "inherited": False,
                                                         "manifestFile": "lake-manifest.json", "dir": "..",
                                                         "configFile": "lakefile.toml"}]}),
      "pf/lean-audit.json": json.dumps({"roots": ["Pf"]}),
      "lean-audit.json": json.dumps({"roots": ["Sp"], "proved_in": ["pf"], "guarantees": {"Pf.claim": {"owner": "@lean"}},
                                     "layers": {"Sp.Defs": {"imports": [], "why": "the spec imports nothing"}},
                                     "shape": {"frozen": {"Pf.claim": "frozen: a toy guarantee"}}})}
#: `ext`, a fetched package, and `fx`, which fetches it: its guarantee's premise is headed by `ext`'s `Q`, which its lock
#: lists as inline, as Security's lists premises Mathlib heads
EXT = {**TOY, "Ext.lean": "prelude\nimport Init.Prelude\n\ndef Ext.Q (n : Nat) : Prop := Eq n n\n",
       "lakefile.toml": 'name = "ext"\ndefaultTargets = ["Ext"]\n\n[[lean_lib]]\nname = "Ext"\n',
       "lake-manifest.json": manifest("ext")}
FX = {**TOY, "Fx.lean": "prelude\nimport Ext\n\ntheorem Fx.g (n : Nat) (h : Ext.Q n) : Ext.Q n := h\n",
      "lakefile.toml": 'name = "fx"\ndefaultTargets = ["Fx"]\n\n[[require]]\nname = "ext"\n'
                       'git = "https://example.invalid/ext"\nrev = "main"\n\n[[lean_lib]]\nname = "Fx"\n',
      "lean-audit.json": json.dumps({"roots": ["Fx"], "guarantees": {"Fx.g": {"owner": "@lean"}},
                                     "shape": {"frozen": {"Fx.g": "frozen: a toy guarantee"},
                                               "inline": {"Fx.g": {"Ext.Q n": "a toy premise a fetched package heads."}}}})}


def test_a_spec_whose_proved_in_package_requires_it_by_path_passes_on_a_lake_build_s_records(tree, tmp_path, envs):
    """Each package's build holds its own modules alone, as Lake's does: `pf`'s audit counts what it loads from its build
    as its own modules (`proved_in.unaudited`), and `sp`'s, which `pf` imports, aren't among them."""
    write(tree, {f"sp/{f}": t for f, t in SP.items()})
    recorded = updated(tree, lake(tree, "sp", tmp_path, "pf"), tmp_path, "pf")
    res = lock(tree, tmp_path, recorded, "proved-in", envs, "sp")
    assert res["status"] == "pass", res["detail"]


def test_a_premise_a_fetched_package_heads_passes_on_a_lake_build_s_records(tree, tmp_path, envs, monkeypatch):
    """The worker's env is a `.lake/packages`, as a checkout's: Lake resolves a fetched package's directory through the
    task's symlink to it, and the audit takes an `.olean` outside `.lake/packages` for one this repository builds, whose
    constant heading a premise (as Mathlib's `Nat.Prime` does Security's) would be an assumption the lock must list."""
    monkeypatch.setattr(env_family, "SCOPE", "imports")
    ext = tmp_path / "ext"
    write(ext, EXT)
    git(ext, "init", "-q", "-b", "main")
    git(ext, "add", "-A")
    git(ext, "commit", "-q", "-m", "ext")
    p = subprocess.run([str(BIN / "lake"), "build"], cwd=ext, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    m = json.loads(manifest("fx", "ext"))
    m["packages"][0]["rev"] = git(ext, "rev-parse", "HEAD")
    write(tree, {f"fx/{f}": t for f, t in FX.items()} | {"fx/lake-manifest.json": json.dumps(m)})
    home = module_runner.fetched_dir(la.deps_key(tree / "fx"))
    shutil.rmtree(home, ignore_errors=True)
    shutil.copytree(ext, home / "ext", symlinks=True)
    (home / module_runner.READY).write_text(la.deps_key(tree / "fx") + "\n")
    recorded = updated(tree, lake(tree, "fx", tmp_path, fetched={"ext": ext}), tmp_path)
    res = lock(tree, tmp_path, recorded, "fetched", envs, "fx")
    assert res["status"] == "pass", res["detail"]


def test_the_toolchain_s_env_passes_with_every_module_recorded_and_its_executables_in_rounds(tree, envs):
    doc = graph.task_graph(tree, ENV, listers=LISTERS)
    task = doc["tasks"][0]
    assert task["vars"] == {"package": "ax", "scope": "whole"} and task["reads"] == [], task
    res, record = envs.run(task, doc)
    assert res["status"] == "pass", res["detail"]
    head, body = module_runner.env_header(record), json.loads(record.read_text())
    tc = module_runner.toolchain_dir(TOOLCHAIN) / "lib" / "lean"
    want = {".".join(o.relative_to(tc).with_suffix("").parts) for o in tc.rglob("*.olean")}
    assert head["env"] == head["toolchain"] == TOOLCHAIN and head["schema"] == env_runner.SCHEMA and head["scope"] == "whole"
    assert head["base"] is None and "base" not in res["detail"]
    assert set(head["files"]) == set(body["axioms"]) == want and res["detail"]["modules"] == len(want)
    assert head["files"]["Init.Prelude"] == module_runner.module_digest(tc / "Init" / "Prelude.olean")
    assert {"LeanChecker", "LeanIR", "Leanc"} <= set(res["detail"]["deferred"]) and len(res["detail"]["rounds"]) > 1, \
        "each executable's root declares `main`: all but one wait for a later round"
    sets = body["sets"]
    assert sets[body["axioms"]["Init.Core"]["propext"]] == ["propext"]
    assert sets[body["axioms"]["Init.Core"]["Lean.reduceBool"]] == ["Lean.trustCompiler"], \
        "the toolchain's own uses of its axioms are recorded, not judged"
    assert "Lean.trustCompiler" in res["detail"]["toolchain_outside_standard"]


def fetched_env(tree: Path, private: bool = True, sources: dict[str, str] | None = None) -> Path:
    """`dep`'s fetched packages on this worker, as `lean_module.packages` finds them made: `evil` built by `lean` from
    ``sources`` (`Evil` from `EVIL` by default), a module at a time, with their private parts or without them."""
    home = module_runner.fetched_dir(la.deps_key(tree / "dep"))
    lib = home / "evil" / ".lake" / "build" / "lib" / "lean"
    shutil.rmtree(home, ignore_errors=True)
    lib.mkdir(parents=True)
    for m, text in (sources or {"Evil": EVIL}).items():
        (home / "evil" / f"{m}.lean").write_text(text)
        p = subprocess.run([str(BIN / "lean"), f"{m}.lean", "-o", str(lib / f"{m}.olean")], cwd=home / "evil",
                           capture_output=True, text=True, env={**os.environ, "LEAN_PATH": str(lib)})
        assert p.returncode == 0 and (lib / f"{m}.olean.private").is_file(), p.stdout + p.stderr
    if not private:
        for f in lib.glob("*.olean.private"):
            f.unlink()
    (home / module_runner.READY).write_text(la.deps_key(tree / "dep") + "\n")
    return home


def test_an_env_module_whose_private_proof_holds_a_sorry_fails_its_env(tree, tmp_path, monkeypatch):
    monkeypatch.setattr(env_family, "SCOPE", "imports")
    doc = graph.task_graph(tree, "lean-env:dep", listers=LISTERS)
    task = doc["tasks"][0]
    assert task["vars"] == {"package": "dep", "scope": "imports", "modules": '["Evil"]'}, task
    fetched_env(tree)
    res = runners.execute(task, doc, tmp_path / "env")
    assert res["status"] == "fail" and res["detail"]["axioms"] == {"sorryAx": ["q"]}, res["detail"]
    body = json.loads((tmp_path / "env" / env_family.record("dep")).read_text())
    assert set(body["axioms"]) == {"Evil", "Init.Prelude"}
    assert body["sets"][body["axioms"]["Evil"]["q"]] == ["sorryAx"], "an importer of `q` takes its `sorry` from the record"

    monkeypatch.setattr(env_runner, "UNPROVED", {"Evil": ("q",)})
    res = runners.execute(task, doc, tmp_path / "unproved")
    assert res["status"] == "pass" and res["detail"]["unproved"] == ["q"], res["detail"]
    body = json.loads((tmp_path / "unproved" / env_family.record("dep")).read_text())
    assert body["sets"][body["axioms"]["Evil"]["q"]] == ["sorryAx"], "listed, `q` still carries its `sorry` to importers"
    monkeypatch.setattr(env_runner, "UNPROVED", {"Evil": ("q", "p")})
    res = runners.execute(task, doc, tmp_path / "stale")
    assert res["status"] == "fail" and res["detail"]["stale"] == ["p"], res["detail"]

    fetched_env(tree, private=False)
    res = runners.execute(task, doc, tmp_path / "public")
    assert res["status"] == "error" and any("private part" in line for line in res["detail"]["log"]), res["detail"]


def test_a_declaration_the_kernel_stops_on_passes_on_the_imports_lean_checked_it_on_and_only_there(tree, tmp_path,
                                                                                                    monkeypatch):
    """On a small stack (`LEAN_STACK_SIZE_KB`) the kernel stops on `t` where `big` has its body, as it stops on CompPoly's
    irreducibility certificates on its own; `lean` checked `t` on `Deep`'s public part, where it passes, and so does the
    replay's second check. Where `lean` checked it with `big`'s body (`import all`), the second check stops too, and the
    env fails."""
    monkeypatch.setattr(env_family, "SCOPE", "imports")
    doc = graph.task_graph(tree, "lean-env:dep", listers=LISTERS)
    module_runner.program("Replay", module_runner.toolchain_dir(TOOLCHAIN), tmp_path)

    def replay(sources: dict[str, str], work: Path) -> dict:
        fetched_env(tree, sources=sources)
        with monkeypatch.context() as m:
            m.setenv("LEAN_STACK_SIZE_KB", "2048")
            return runners.execute(doc["tasks"][0], doc, work)

    res = replay(DEEP, tmp_path / "public")
    assert res["status"] == "pass" and res["detail"]["rounds"][0]["on_public_imports"] == ["t"], res["detail"]
    res = replay(DEEP | {"Evil": "module\n\nprelude\nimport all Deep\n\n"
                                 "set_option maxRecDepth 100000000 in\nset_option maxHeartbeats 0 in\n"
                                 "theorem t (h : Eq big 100000) : Eq (id big) 100000 := h\n"}, tmp_path / "all")
    assert res["status"] == "fail" and res["detail"]["rejected"] == ["Evil"], res["detail"]


def test_an_env_with_no_build_of_its_fetched_packages_beyond_mathlib_s_cache_is_an_error(tree, tmp_path, envs):
    doc = graph.task_graph(tree, "lean-env:dep", listers=LISTERS)
    shutil.rmtree(fetched_env(tree) / "evil" / ".lake" / "build")
    envs.stage(doc, tmp_path / "env")
    res = runners.execute(next(t for t in doc["tasks"] if t["id"] == "lean-env:dep"), doc, tmp_path / "env")
    assert res["status"] == "error" and "no build of evil" in res["detail"]["error"], \
        "a worker without warm dependencies or the bundle would record an env without them"


def test_an_env_layered_on_the_toolchain_s_replays_its_own_modules_alone_and_records_the_toolchain_s_as_its_base_did(
        tree, tmp_path, envs, monkeypatch):
    doc = graph.task_graph(tree, "lean-env:dep", listers=LISTERS)
    task = next(t for t in doc["tasks"] if t["id"] == "lean-env:dep")
    assert task["reads"] == [{"task": ENV, "output": "axioms"}], "dep fetches evil, on the toolchain ax's env is alone"
    evil = fetched_env(tree) / "evil" / ".lake" / "build" / "lib" / "lean" / "Evil.olean"
    monkeypatch.setattr(env_runner, "UNPROVED", {"Evil": ("q",)})
    base = envs.stage(doc, tmp_path / "base")
    given = json.loads(base.read_text())

    def layered(name: str, edit=lambda line: line) -> tuple[dict, Path]:
        record = stage(base, tmp_path / name, env_family.record("ax"))
        record.write_text("".join(edit(line) for line in record.read_text().splitlines(keepends=True)))
        return runners.execute(task, doc, tmp_path / name), tmp_path / name / env_family.record("dep")

    res, record = layered("layered")
    assert res["status"] == "pass", res["detail"]
    assert res["detail"]["base"] == ENV and res["detail"]["base_modules"] == len(given["files"])
    assert [r["modules"] for r in res["detail"]["rounds"]] == [1], \
        "Evil alone goes to the kernel, importing the toolchain's modules with their sets from ax's record"
    head, body = module_runner.env_header(record), json.loads(record.read_text())
    assert head["base"] == ENV and head["env"] == env_family.env_key(tree / "dep") != TOOLCHAIN
    assert set(body["axioms"]) == {*given["axioms"], "Evil"} and head["files"] == given["files"] | {
        "Evil": module_runner.module_digest(evil)}
    for m, cs in given["axioms"].items():
        assert {c: given["sets"][i] for c, i in cs.items()} == {c: body["sets"][i] for c, i in body["axioms"][m].items()}, m
    assert body["sets"][body["axioms"]["Evil"]["q"]] == ["sorryAx"]

    mod = graph.task_graph(tree, "lean-module:dep/Dep.lean", listers=LISTERS)
    dep = next(t for t in mod["tasks"] if t["id"] == "lean-module:dep/Dep.lean")
    assert dep["reads"] == [{"task": "lean-env:dep", "output": "axioms"}]
    stage(record, tmp_path / "dep", env_family.record("dep"))
    res = runners.execute(dep, mod, tmp_path / "dep")
    assert res["status"] == "pass" and res["detail"]["env_modules"] == 2, \
        ("a module of the layered env takes Evil's sets and Init.Prelude's from its record", res["detail"])

    def files(f):
        return lambda line: f'"files":{json.dumps(f(json.loads(line[8:-2])))},\n' if line.startswith('"files":') else line

    res, _ = layered("bytes", files(lambda d: d | {"Init.Prelude": "sha512:" + "0" * 128}))
    assert res["status"] == "error" and res["detail"]["differ"] == ["Init.Prelude"], res["detail"]
    res, _ = layered("unhashed", files(lambda d: {k: v for k, v in d.items() if k != "Init.Core"}))
    assert res["status"] == "error" and "a line and a digest for each" in res["detail"]["error"], \
        ("a base module without its digest would go unchecked", res["detail"])
    res, _ = layered("absent", lambda line: files(lambda d: d | {"Nowhere": "sha512:" + "0" * 128})(line)
                     + ('"Nowhere":{},\n' if line == '"axioms":{\n' else ""))
    assert res["status"] == "error" and res["detail"]["absent"] == ["Nowhere"], res["detail"]


def test_a_fetched_declaration_may_use_lake_s_family_axioms_and_no_axiom_a_fetched_package_or_init_declares():
    records = {
        "Lake.Build.Data": {"Lake.DataType.bool": ["Lake.DataType.bool"], "Lake.get": ["Lake.DataType.bool", "propext"]},
        "Init.Prelude": {"sorryAx": ["sorryAx"]},
        "Init.Core": {"Lean.reduceBool": ["Lean.trustCompiler"]},
        "Lake.Fake": {"Lake.DataType.fake": ["Lake.DataType.fake"]},
        "runLinter": {"lint": ["Lake.DataType.bool", "propext"], "main": ["sorryAx"]},
        "Evil": {"q": ["sorryAx"], "r": ["Lake.DataType.bool", "Lean.ofReduceBool"], "s": ["Lake.DataType.fake"]},
    }
    outside, unjudged, stale = env_runner.judge(records, {"runLinter": {"main"}},
                                                {"Lake.Build.Data", "Init.Prelude", "Init.Core"}, {})
    assert outside == {"Lake.DataType.fake": ["s", "Lake.DataType.fake"], "Lean.ofReduceBool": ["r"], "sorryAx": ["q"]}, \
        "Lake's own family axioms pass; one a fetched module declares (under Lake's names too) and Init's fail"
    assert unjudged == {"Lake.DataType.bool": 2, "Lean.trustCompiler": 1, "sorryAx": 1}
    assert stale == []


def test_a_fetched_declaration_unproved_lists_may_use_sorry_and_no_other_axiom_and_an_entry_without_one_is_stale():
    records = {"Pkg.A": {"a": ["sorryAx"], "b": ["Lean.ofReduceBool", "sorryAx"], "c": ["sorryAx"], "d": ["propext"]},
               "Pkg.B": {"a": ["sorryAx"]}}
    outside, _, stale = env_runner.judge(records, {}, set(), {"Pkg.A": ("a", "b", "d", "e"), "Pkg.Elsewhere": ("f",)})
    assert outside == {"Lean.ofReduceBool": ["b"], "sorryAx": ["c", "a"]}, \
        "a listed declaration may use `sorryAx` alone, and only in the module listed"
    assert stale == ["d", "e"], "a listed declaration of a recorded module that uses no `sorryAx` (or is gone) is stale"


def test_a_module_task_refuses_an_env_record_of_another_env_or_of_other_bytes_than_its_worker_s(tree, tmp_path, envs):
    doc = graph.task_graph(tree, "lean-module:ctl/Control/Defs.lean", listers=LISTERS)
    task = next(t for t in doc["tasks"] if t["kind"] == "lean-module")

    def edited(name: str, edit) -> dict:
        record = envs.stage(doc, tmp_path / name)
        lines = record.read_text().splitlines(keepends=True)
        record.write_text("".join(edit(line) for line in lines))
        return runners.execute(task, doc, tmp_path / name)

    assert edited("same", lambda line: line)["status"] == "pass"
    res = edited("other", lambda line: '"env":"leanprover/lean4:v0.0.0",\n' if line.startswith('"env":') else line)
    assert res["status"] == "error" and "is of env 'leanprover/lean4:v0.0.0'" in res["detail"]["error"], res["detail"]

    def files(f):
        return lambda line: f'"files":{json.dumps(f(json.loads(line[8:-2])))},\n' if line.startswith('"files":') else line

    res = edited("bytes", files(lambda d: d | {"Init.Prelude": "sha512:" + "0" * 128}))
    assert res["status"] == "error" and res["detail"]["differ"] == ["Init.Prelude"], res["detail"]
    res = edited("unrecorded", files(lambda d: {k: v for k, v in d.items() if k != "Init.Core"}))
    assert res["status"] == "error" and res["detail"]["unrecorded"] == ["Init.Core"], res["detail"]
