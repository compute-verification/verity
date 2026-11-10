"""The lean-package family: on this tree a task per audited Lake package, its inputs what an audit of it can read
(`lean_audit.inputs`) and its slot tasks/v1's (`tasks.lean_slot`); on a toy checkout, the packages a package requires by path
and those its lock names under `proved_in`, its nested packages left out, and a manifest that isn't JSON refused; and its runner
runs `lean_audit.py --package` as a shard's audit, telling a fail, the package's, from an error, the machine's, on a stand-in."""
from __future__ import annotations

import importlib.util
import os

import pytest
from conftest import SPEC, toy_tree

from verity_graph import ROOT, contract, graph, runners
from verity_graph.families import lean_package as family
from verity_graph.runners import lean_package as runner

spec = importlib.util.spec_from_file_location("lean_package_tasks", ROOT / "tools" / "verity" / "check" / "tasks.py")
TASKS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TASKS)


def _table() -> str:
    """`[family.lean-package]` as this tree's tasks.toml has it."""
    text = (ROOT / graph.SPEC).read_text()
    start = text.index("[family.lean-package]")
    return text[start:text.index("\n[", start + 1)] + "\n"


def test_on_this_tree_a_task_is_a_shard_s_audit_of_one_package():
    """One task per package the audit audits, its inputs every file an audit of it can read (`lean_audit.inputs`, which reads
    the checkout `lean_audit.py` is in) and its runner, its cores and memory the slot tasks/v1 gives the package's audit."""
    doc = graph.listing(ROOT, {"lean-package"})
    files = graph.tracked(ROOT)
    fam = graph.spec(ROOT)["family"]["lean-package"]
    pkgs = family.la.packages(files)
    assert [t["id"] for t in doc["tasks"]] == sorted(f"lean-package:{p}" for p in pkgs) and "Security" in pkgs
    for t in doc["tasks"]:
        pkg = t["vars"]["package"]
        assert t["inputs"] == sorted({*family.la.inputs(pkg, files), *graph.runner_sources("lean-package")}), pkg
        assert t["resources"] == {**TASKS.lean_slot(pkg), "disk_gb": fam["resources"]["disk_gb"]}, pkg
        assert t["key"] and not t["reads"] and t["outputs"] == [{"name": "report", "path": family.REPORT}]
    assert family.SECURITY == {k: TASKS.LEAN_SLOT[k] for k in ("cores", "mem_gb")}
    assert fam["resources"] == TASKS.LEAN_PKG_SLOT and TASKS.REGENERATE is False


MANIFEST = '{{"packages": [{{"type": "path", "dir": "{dir}", "name": "x"}}, {{"type": "git", "name": "mathlib"}}]}}'
TOY = {"a/lean-audit.json": '{"proved_in": ["../c"]}', "a/lakefile.toml": "", "a/A.lean": "",
       "a/lake-manifest.json": MANIFEST.format(dir="../b"), "a/sub/lakefile.toml": "", "a/sub/X.lean": "",
       "b/lakefile.toml": "", "b/B.lean": "",
       "c/lean-audit.json": "{}", "c/lakefile.toml": "", "c/C.lean": "", "c/lake-manifest.json": MANIFEST.format(dir="../d"),
       "d/lakefile.toml": "", "d/D.lean": "",
       "Security/lean-audit.json": "{}", "Security/lakefile.toml": "", "Security/S.lean": "",
       "e/lean-audit.json": "{}", "e/lakefile.toml": "", "e/lake-manifest.json": MANIFEST.format(dir="../Security"),
       "tools/verity/lean/check.py": "", "tools/verity/lean/tests/test_check.py": "", "tools/verity/lean/README.md": "",
       "tools/verity/lean/tests/fixture/lean-audit.json": "{}", f"{graph.RUNNERS}/lean_package.py": ""}


def test_a_package_s_inputs_are_its_own_its_path_requirements_and_its_provers(tmp_path):
    """`a` requires `b` by path and names `c` its prover, which requires `d`: its audit reads all four packages' files but those
    of `a/sub`, a package nested in it, and the auditor's but its tests and docs. `b` and `d` hold no lock, and the auditor's
    fixtures are no package: none of them has a task. A package under `Security`, or requiring one there, asks `SECURITY`."""
    root = toy_tree(tmp_path / "t", SPEC + "\n" + _table(), TOY)
    got = {t["id"]: t for t in graph.listing(root, {"lean-package"}, listers={"lean-package": family.tasks})["tasks"]}
    assert sorted(got) == ["lean-package:Security", "lean-package:a", "lean-package:c", "lean-package:e"]
    tool = {"tools/verity/lean/check.py", *graph.runner_sources("lean-package")}
    own = {p: {f for f in TOY if f.startswith(p + "/")} for p in ("b", "c", "d", "Security", "e")}
    assert set(got["lean-package:a"]["inputs"]) == {"a/lean-audit.json", "a/lakefile.toml", "a/A.lean", "a/lake-manifest.json",
                                                    *own["b"], *own["c"], *own["d"], *tool}
    assert set(got["lean-package:c"]["inputs"]) == {*own["c"], *own["d"], *tool}
    assert set(got["lean-package:e"]["inputs"]) == {*own["e"], *own["Security"], *tool}
    fam = graph.spec(root)["family"]["lean-package"]
    for t in got.values():
        security = t["vars"]["package"] in ("Security", "e")
        assert t["resources"] == ({**fam["resources"], **family.SECURITY} if security else fam["resources"]), t["id"]
    (root / "c/lake-manifest.json").write_text("{not json")
    with pytest.raises(graph.GraphError, match="c/lake-manifest.json isn't JSON"):
        graph.listing(root, {"lean-package"}, listers={"lean-package": family.tasks})


STANDIN = """import json, os, signal, sys
from pathlib import Path


def transient(rep):
    return all("could not fetch" in f for f in rep.get("failures", []))


if __name__ == "__main__":
    a = sys.argv[1:]
    out = Path(a[a.index("--out") + 1])
    assert a[a.index("--package") + 1] == "pkg" and not {"--regenerate", "--fresh", "--bookkeeping"} & set(a)
    how = Path(os.environ["HOW"]).read_text()
    if how == "kill":
        os.kill(os.getpid(), signal.SIGKILL)
    if how in ("usage", "no report"):
        sys.exit(2 if how == "usage" else 1)
    out.mkdir(parents=True, exist_ok=True)
    fails = {"lock": ["record: Pkg.claim: its statement differs from the lock's"], "fetch": ["setup: could not fetch mathlib"]}
    rep = {"package": "pkg", "failures": fails.get(how, [])}
    failed = ["pkg"] if rep["failures"] else ["unread"] if how == "unread" else []
    (out / "audit.json").write_text(json.dumps({"passed": not failed, "failed": failed, "packages": [rep],
                                                "cached": {"pkg": "k"} if how == "reused" else {}}))
    sys.exit(1 if failed else 0)
"""


@pytest.mark.parametrize(("how", "status"), [("pass", "pass"), ("reused", "pass"), ("lock", "fail"), ("unread", "fail"),
                                             ("fetch", "error"), ("no report", "error"), ("usage", "error"), ("kill", "error")])
def test_the_runner_tells_the_package_s_fail_from_the_machine_s_error(tmp_path, monkeypatch, how, status):
    """`lean_audit.py --out WORK/lean-audit --package PKG`, lake on the PATH, no record regenerated: a package that failed its
    audit (its lock, the unread ratchet) is a fail naming its failures; failures that are all a fetch the audit couldn't make, a
    missing report, a usage error and a kill are an error, no result."""
    root = tmp_path / "tree"
    for p, text in {runner.AUDIT: STANDIN, "bin/lake": "#!/bin/sh\n"}.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(text)
        (root / p).chmod(0o755)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(root / "bin"), prepend=os.pathsep)
    (tmp_path / "how").write_text(how)
    monkeypatch.setenv("HOW", str(tmp_path / "how"))
    task = {"id": "lean-package:pkg", "kind": "lean-package", "about": "x", "env": "lean",
            "argv": ["verity", "graph", "run", "lean-package:pkg"], "vars": {"package": "pkg"}, "inputs": [], "reads": [],
            "outputs": [{"name": "report", "path": family.REPORT}], "resources": {"cores": 16, "mem_gb": 64, "disk_gb": 60},
            "timeout_s": 60, "key": None}
    res = runners.execute(task, {"envs": {"lean": {"digest": contract.digest(b"lean")}}, "files": {}, "tasks": [task]},
                          tmp_path / "work", runner.run)
    assert contract.result_errors(res, task) == [], res
    assert res["status"] == status, res
    if how == "lock":
        assert res["detail"]["failed"] == ["pkg"]
        assert res["detail"]["failures"] == ["pkg: record: Pkg.claim: its statement differs from the lock's"]
    if how == "unread":
        assert res["detail"]["failed"] == ["unread"] and res["detail"]["failures"] == []
    assert res["detail"].get("reused") is (how == "reused") or status == "error"
    if how == "fetch":
        assert "couldn't be fetched" in res["detail"]["error"]
